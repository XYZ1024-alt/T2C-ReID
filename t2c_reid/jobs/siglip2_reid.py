"""SigLIP 2-backed T2C-ReID two-stage training job builder.

Stage-1 aligns frozen image features with identity-aware prompt text through
SigLIP's supervised all-pairs sigmoid objective. Stage-2 trains the ReID head,
TFC centers, prompt bank, and (by default) the SigLIP 2 vision tower, while
aligning each image against every training-identity text anchor.

Validation and inference use only global + camera prompts. Identity prompts
never touch query/gallery retrieval.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, replace
from itertools import batched
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from scripts.train import StageMetadata, TrainingJob, TwoStageTrainingJob
from t2c_reid.anchors import IdentityAnchorProvider
from t2c_reid.clothes import (
    ClothesAdversarialSchedule,
    ClothesClassifier,
    build_clothes_table,
)
from t2c_reid.configuration import HOLDOUT_DATASETS, TrainingConfig
from t2c_reid.data import (
    PRCC_CLOTHES_GROUPS,
    PRCC_HOLDOUT_SEED,
    ReIDSample,
    load_market_split,
    load_msmt17_manifest,
    load_prcc_split,
    split_prcc_holdout,
)
from t2c_reid.datasets import (
    DEFAULT_INSTANCES_PER_IDENTITY,
    PYTHON_DATA_BACKEND,
    IdentityBalancedBatchSampler,
    ReIDImageBatch,
    ReIDImageDataset,
    ReIDImageDatasetConfig,
    ReIDMetadataDataset,
    ReIDMetadataDatasetConfig,
    RustReIDBatchCollator,
    build_camera_id_map,
    build_person_id_map,
    collate_reid_batches,
    require_data_backend,
)
from t2c_reid.ema import ModelEma
from t2c_reid.evaluation import (
    ReIDMetrics,
    evaluate_reid,
    evaluate_reid_with_rerank,
    require_evaluation_backend,
)
from t2c_reid.features import l2_normalize
from t2c_reid.model import T2CReIDModel
from t2c_reid.precision import PrecisionController, PrecisionPolicy, resolve_precision
from t2c_reid.prompts import PromptBank, PromptConfig
from t2c_reid.retrieval import FUSED_RETRIEVAL, require_retrieval_mode
from t2c_reid.siglip2_backbone import (
    SIGLIP2_MODEL_ID,
    TransformersSiglip2ImageEncoder,
    TransformersSiglip2TextEncoder,
    siglip2_feature_dim,
    siglip2_max_num_patches,
    siglip2_patch_size,
    siglip2_text_hidden_dim,
    siglip2_uses_patchified_inputs,
    validate_siglip2_image_size,
)
from t2c_reid.tfc import CameraAwareTFCBank
from t2c_reid.training import (
    Stage1LossBreakdown,
    Stage1LossConfig,
    Stage2LossBreakdown,
    Stage2LossConfig,
    Stage2LossInputs,
    TrainingBatch,
    stage1_alignment_loss,
    stage1_alignment_loss_from_visual,
    stage2_loss_breakdown,
)
from t2c_reid.transforms import (
    TRAIN_COLOR_JITTER,
    TRAIN_GRAYSCALE_PROB,
    ImageTransformConfig,
    Siglip2ImageTransform,
    Siglip2TrainImageTransform,
)

DEFAULT_RANKS = (1, 5, 10)
# Extra PRCC query set evaluated against the same gallery; the primary
# query/mAP is cross-clothes (camera C).
SAME_CLOTHES_QUERY = "same_clothes"
MODEL_EMA_STATE_KEY = "model_ema"
PROMPT_TEMPLATE_PREFIX = "a photo of a"
PROMPT_TEMPLATE_SUFFIX = "person ."
STAGE1_TRAIN_LOSS_METRIC_NAMES = ("loss", "alignment_loss")
STAGE2_TRAIN_LOSS_METRIC_NAMES = (
    "loss",
    "alignment_loss",
    "reid_loss",
    "triplet_loss",
    "triplet_active_fraction",
    "tfc_loss",
    "tfc_local_loss",
    "tfc_global_loss",
    "tfc_cross_modal_loss",
    "tfc_cross_camera_loss",
    "tfc_transfer_reg_loss",
    "tfc_cross_camera_coverage",
    "clothes_loss",
    "clothes_adversarial_loss",
    "clothes_accuracy",
)
STAGE1 = "stage1"
STAGE2 = "stage2"

Siglip2Loader = Callable[[str], "Siglip2LoadResult"]


@dataclass(frozen=True)
class Siglip2LoadResult:
    model: torch.nn.Module
    image_processor: Any
    tokenizer: Any


@dataclass(frozen=True)
class Siglip2ModelSpec:
    feature_dim: int
    text_hidden_dim: int
    patch_size: int
    max_num_patches: int
    patch_count: int
    vision_input_format: str
    text_padding_side: str
    include_bos_token: bool
    mask_text_padding: bool
    bos_token_id: int
    eos_token_id: int
    pad_token_id: int


@dataclass(frozen=True)
class JobDataConfig:
    dataset: str
    root: Path
    validation_holdout_ids: int = 0
    validation_holdout_seed: int = PRCC_HOLDOUT_SEED


@dataclass(frozen=True)
class Siglip2ReIDJobConfig:
    dataset: str
    data_root: Path
    siglip2_model_name: str
    siglip2_checkpoint: Path | None
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    data_backend: str
    prefetch_factor: int
    pin_memory: bool
    persistent_workers: bool
    rust_data_threads: int
    evaluation_backend: str
    evaluation_chunk_size: int
    lr: float
    image_encoder_lr: float
    device: torch.device
    precision: PrecisionPolicy
    gradient_checkpointing: bool
    image_size: tuple[int, int]
    beta: float
    context_length: int
    tfc_momentum: float
    tfc_tail_momentum: float
    tfc_class_balance_beta: float
    tfc_local_weight: float
    tfc_global_weight: float
    tfc_cross_modal_weight: float
    tfc_cross_camera_weight: float
    tfc_contrast_temperature: float
    tfc_transfer_reg_weight: float
    triplet_margin: float
    triplet_metric: str
    tfc_weight: float
    alignment_weight: float
    id_logit_scale: float
    label_smoothing: float
    stage1_epochs: int
    stage2_epochs: int
    validation_interval: int
    freeze_image_encoder_stage1: bool
    freeze_image_encoder_stage2: bool
    freeze_text_encoder: bool
    stage2_first_epoch: int = 1
    freeze_prompt_bank_stage2: bool = False
    reid_head: str = "linear"
    retrieval_mode: str = "fused"
    beta_warmup_epochs: int = 0
    report_rerank: bool = False
    stage1_lr_scheduler: str = "none"
    stage1_warmup_epochs: int = 0
    stage2_lr_scheduler: str = "none"
    stage2_warmup_epochs: int = 0
    num_instances: int = DEFAULT_INSTANCES_PER_IDENTITY
    camera_balanced_sampling: bool = False
    color_jitter: tuple[float, float, float, float] = TRAIN_COLOR_JITTER
    grayscale_prob: float = TRAIN_GRAYSCALE_PROB
    sie_coe: float = 0.0
    stage1_feature_cache: bool = True
    grad_clip_norm: float = 0.0
    flip_tta: bool = False
    validation_holdout_ids: int = 0
    validation_holdout_seed: int = PRCC_HOLDOUT_SEED
    image_encoder_frozen_layers: int = 0
    image_encoder_layer_decay: float = 1.0
    model_ema_decay: float = 0.0
    clothes_adversarial_weight: float = 0.0
    clothes_adversarial_start_epoch: int = 2
    clothes_classifier_lr: float = 3.5e-4


@dataclass(frozen=True)
class DatasetBundle:
    train: ReIDImageDataset | ReIDMetadataDataset
    # Train split re-viewed through the eval transform: the Stage-1 feature
    # cache extracts frozen image features without augmentation noise.
    train_eval: ReIDImageDataset | ReIDMetadataDataset
    query: ReIDImageDataset | ReIDMetadataDataset
    gallery: ReIDImageDataset | ReIDMetadataDataset
    num_train_ids: int
    num_cameras: int
    identity_counts: torch.Tensor
    identity_camera_counts: torch.Tensor
    # Named secondary query sets scored against ``gallery`` and reported as
    # metric extras; the primary ``query`` alone drives mAP and model selection.
    extra_queries: tuple[tuple[str, ReIDImageDataset | ReIDMetadataDataset], ...] = ()
    # Raw dataset camera id of each dense camera index.
    camera_raw_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class SplitSamples:
    train: Sequence[ReIDSample]
    query: Sequence[ReIDSample]
    gallery: Sequence[ReIDSample]
    extra_queries: tuple[tuple[str, Sequence[ReIDSample]], ...] = ()


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    query: DataLoader
    gallery: DataLoader
    extra_queries: tuple[tuple[str, DataLoader], ...] = ()


@dataclass(frozen=True)
class TransformBundle:
    train: Any
    eval: Any


@dataclass(frozen=True)
class Stage1CachedBatch:
    """One identity-balanced Stage-1 training batch served from the feature cache."""

    visual_raw: torch.Tensor
    person_ids: torch.Tensor
    camera_ids: torch.Tensor


class Stage1FeatureCache:
    """Frozen Stage-1 train-split image features, extracted lazily once per run.

    Only valid while the Stage-1 image tower is frozen (enforced at job build
    time). Extraction runs the eval-transform view of the train split under
    ``torch.no_grad()`` at the start of the first Stage-1 training epoch, so
    the cached ``visual_raw`` features are exactly what the frozen tower
    (including any SIE camera injection) would produce for every later epoch.
    """

    def __init__(
        self,
        dataset: ReIDImageDataset | ReIDMetadataDataset,
        config: Siglip2ReIDJobConfig,
    ):
        self._dataset = dataset
        self._config = config
        self._visual_raw: torch.Tensor | None = None
        self._person_ids: torch.Tensor | None = None
        self._camera_ids: torch.Tensor | None = None

    def ensure_extracted(
        self,
        model: Siglip2ReIDTrainingModel,
        precision: PrecisionController,
    ) -> None:
        if self._visual_raw is not None:
            return
        loader = _loader(self._dataset, self._config, shuffle=False)
        was_training = model.training
        model.eval()
        visual_parts: list[torch.Tensor] = []
        person_parts: list[torch.Tensor] = []
        camera_parts: list[torch.Tensor] = []
        with torch.no_grad(), precision.autocast():
            for batch in loader:
                images = batch.images.to(self._config.device, non_blocking=True)
                cameras = batch.camera_ids.to(self._config.device, non_blocking=True)
                visual_parts.append(
                    model.retrieval_model.encode_visual_raw(images, cameras)
                )
                person_parts.append(
                    batch.person_ids.to(self._config.device, non_blocking=True)
                )
                camera_parts.append(cameras)
        if was_training:
            model.train()
        if not visual_parts:
            raise ValueError("stage1 feature cache extraction produced no batches")
        self._visual_raw = torch.cat(visual_parts)
        self._person_ids = torch.cat(person_parts)
        self._camera_ids = torch.cat(camera_parts)

    def batches(self) -> Iterator[Stage1CachedBatch]:
        """Identity-balanced batches over the cached tensors (same PK sampling as the train loader)."""
        if (
            self._visual_raw is None
            or self._person_ids is None
            or self._camera_ids is None
        ):
            raise ValueError("stage1 feature cache must be extracted before batching")
        sampler = IdentityBalancedBatchSampler(
            self._person_ids.tolist(),
            batch_size=self._config.batch_size,
            instances_per_identity=self._config.num_instances,
            camera_labels=(
                self._camera_ids.tolist()
                if self._config.camera_balanced_sampling
                else None
            ),
        )
        for batch_indices in sampler:
            indices = torch.tensor(
                batch_indices, dtype=torch.long, device=self._visual_raw.device
            )
            yield Stage1CachedBatch(
                visual_raw=self._visual_raw[indices],
                person_ids=self._person_ids[indices],
                camera_ids=self._camera_ids[indices],
            )


@dataclass(frozen=True)
class StageTrainingRuntime:
    model: Siglip2ReIDTrainingModel
    loaders: LoaderBundle
    optimizer: torch.optim.Optimizer
    stage: str
    loss_config: Any
    device: torch.device
    beta_schedule: BetaSchedule | None = None
    freeze_config: Siglip2ReIDJobConfig | None = None
    lr_scheduler: StageLRScheduler | None = None
    anchor_provider: IdentityAnchorProvider | None = None
    feature_cache: Stage1FeatureCache | None = None
    precision: PrecisionController | None = None
    gradient_accumulation_steps: int = 1
    grad_clip_norm: float = 0.0
    model_ema: ModelEma | None = None
    clothes_schedule: ClothesAdversarialSchedule | None = None


@dataclass(frozen=True)
class ValidationRuntime:
    model: Siglip2ReIDTrainingModel
    loaders: LoaderBundle
    device: torch.device
    retrieval_mode: str
    model_config: Siglip2ReIDJobConfig
    beta_schedule: BetaSchedule | None = None
    report_rerank: bool = False
    precision: PrecisionController | None = None
    model_ema: ModelEma | None = None


@dataclass(frozen=True)
class Stage2AuxiliaryState:
    """Stage-2 checkpoint side state: precision/scaler plus the optional weight EMA.

    ``model_state`` always holds the live training weights; the EMA shadow is
    stored under ``model_ema`` so a resumed run continues the same average.
    """

    precision: PrecisionController
    model_ema: ModelEma | None = None

    def state_dict(self) -> dict[str, Any]:
        state = self.precision.state_dict()
        if self.model_ema is not None:
            state[MODEL_EMA_STATE_KEY] = self.model_ema.state_dict()
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        ema_state = state.get(MODEL_EMA_STATE_KEY)
        if (ema_state is None) != (self.model_ema is None):
            raise ValueError(
                "checkpoint model EMA state does not match this run's "
                "model_ema_decay setting"
            )
        self.precision.load_state_dict(dict(state))
        if self.model_ema is not None:
            if not isinstance(ema_state, Mapping):
                raise TypeError("checkpoint model EMA state must be a mapping")
            self.model_ema.load_state_dict(ema_state)


@dataclass(frozen=True)
class BetaSchedule:
    """Linear ramp of the fused retrieval beta from 0 to ``beta`` over ``warmup_epochs`` Stage-2 epochs.

    The fused text feature ``f_t_eval`` carries only global+camera signal — no identity — so
    blended into ``f_eval`` early on, before the image backbone has learned anything
    discriminative, it actively pushes samples toward their camera cluster and *lowers*
    mAP below the image-only floor. The warmup lets the image encoder learn discriminative
    features first, then blends the camera-conditioned text in once the image stream is
    stable. At ``epoch == 1`` the effective beta is ``0`` (pure image feature); at
    ``epoch == warmup_epochs + 1`` (and every epoch after) the effective beta is ``beta``.
    """

    beta: float
    warmup_epochs: int
    first_epoch: int = 1

    def effective_beta(self, stage_epoch: int) -> float:
        if stage_epoch < 1:
            raise ValueError("stage_epoch must be positive")
        if self.warmup_epochs <= 0:
            return self.beta
        if stage_epoch <= 1:
            return 0.0
        return self.beta * min(1.0, (stage_epoch - 1) / self.warmup_epochs)

    def apply(self, model: Siglip2ReIDTrainingModel, epoch: int) -> None:
        stage_epoch = epoch - self.first_epoch + 1
        model.retrieval_model.beta = self.effective_beta(stage_epoch)


@dataclass(frozen=True)
class StageLRScheduler:
    """Warmup + cosine-decay of Stage-2 learning rates over Stage-2-local epochs.

    Linear warmup from ``base_lr / warmup_epochs`` at stage epoch 1 up to the full
    ``base_lr`` at ``warmup_epochs``, then a cosine decay toward ~0 by ``total_epochs``.
    Every param group is scaled by the same factor so the grouped backbone/new
    learning rates keep their ratio; groups marked ``fixed_lr`` (the CAL clothes
    discriminator) keep their constant rate. ``warmup_epochs == 0`` disables warmup and
    applies pure cosine decay from stage epoch 1.
    """

    base_lrs: tuple[float, ...]
    total_epochs: int
    warmup_epochs: int
    first_epoch: int = 1

    def scale(self, stage_epoch: int) -> float:
        if stage_epoch < 1:
            raise ValueError("stage_epoch must be positive")
        if self.warmup_epochs > 0 and stage_epoch <= self.warmup_epochs:
            return stage_epoch / self.warmup_epochs
        decay_start = self.warmup_epochs + 1
        decay_epochs = max(1, self.total_epochs - self.warmup_epochs)
        progress = min(1.0, max(0.0, (stage_epoch - decay_start) / decay_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    def apply(self, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        stage_epoch = epoch - self.first_epoch + 1
        factor = self.scale(stage_epoch)
        for group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            if not group.get("fixed_lr", False):
                group["lr"] = base_lr * factor


class Siglip2ReIDTrainingModel(torch.nn.Module):
    def __init__(
        self,
        retrieval_model: T2CReIDModel,
        classifier: torch.nn.Module,
        tfc_bank: CameraAwareTFCBank,
        clothes_classifier: ClothesClassifier | None = None,
    ):
        super().__init__()
        self.retrieval_model = retrieval_model
        self.classifier = classifier
        self.tfc_bank = tfc_bank
        self.clothes_classifier = clothes_classifier

    def encode_retrieval(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
        retrieval_mode: str = FUSED_RETRIEVAL,
    ) -> torch.Tensor:
        """Validation / inference retrieval feature.

        The feature head (e.g. BNNeck) lives inside the retrieval model, so
        training and eval share a single retrieval path; this wrapper only
        delegates.
        """
        return self.retrieval_model.encode_retrieval(
            images, camera_ids, retrieval_mode=retrieval_mode
        )

    def encode_retrieval_views(
        self,
        images: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.retrieval_model.encode_retrieval_views(images, camera_ids)


class BNNeck(torch.nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(feature_dim)
        self.freeze_bias()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.bn(features)

    def freeze_bias(self) -> None:
        self.bn.bias.requires_grad_(False)


def build_training_job(
    training_config: TrainingConfig,
    siglip2_loader: Siglip2Loader = lambda model_name: load_transformers_siglip2(
        model_name
    ),
) -> TwoStageTrainingJob | TrainingJob:
    config = _job_config_from_training_config(training_config)
    loaded_siglip2 = siglip2_loader(config.siglip2_model_name)
    _load_siglip2_checkpoint_if_requested(
        loaded_siglip2.model, config.siglip2_checkpoint, config.device
    )
    spec = _validate_loaded_siglip2(loaded_siglip2, config.image_size)
    _configure_gradient_checkpointing(
        loaded_siglip2.model, config.gradient_checkpointing
    )
    transforms = TransformBundle(
        train=Siglip2TrainImageTransform(
            loaded_siglip2.image_processor,
            image_size=config.image_size,
            color_jitter=config.color_jitter,
            grayscale_prob=config.grayscale_prob,
        ),
        eval=Siglip2ImageTransform(
            loaded_siglip2.image_processor, image_size=config.image_size
        ),
    )
    data = load_dataset_bundle(
        JobDataConfig(
            config.dataset,
            config.data_root,
            config.validation_holdout_ids,
            config.validation_holdout_seed,
        ),
        transforms,
        backend=config.data_backend,
    )
    shared_model = _build_training_model(
        config,
        loaded_siglip2.model,
        data,
        spec=spec,
        prefix_token_ids=_encode_template_token_ids(
            loaded_siglip2.tokenizer, PROMPT_TEMPLATE_PREFIX
        ),
        suffix_token_ids=_encode_template_token_ids(
            loaded_siglip2.tokenizer, PROMPT_TEMPLATE_SUFFIX
        ),
    ).to(config.device)
    precision = PrecisionController(config.precision)
    # Averages the Stage-2 retrieval path only; the classifier and TFC bank
    # never take part in evaluation.
    model_ema = (
        ModelEma(shared_model.retrieval_model, config.model_ema_decay)
        if config.model_ema_decay > 0.0
        else None
    )
    loaders = _build_loaders(data, config)
    (
        stage1_runtime,
        stage2_runtime,
        optimizer_stage1,
        optimizer_stage2,
        stage2_beta_schedule,
    ) = _build_runtimes(
        config,
        shared_model,
        loaders,
        data.num_train_ids,
        precision=precision,
        stage1_feature_cache=_build_stage1_feature_cache(config, data),
        model_ema=model_ema,
    )
    metadata = _stage_metadata(config, spec, data)
    stage2_job = TrainingJob(
        model=shared_model,
        optimizer=optimizer_stage2,
        train_one_epoch=_train_one_epoch(stage2_runtime),
        validate=_validate(
            ValidationRuntime(
                shared_model,
                loaders,
                config.device,
                config.retrieval_mode,
                config,
                beta_schedule=stage2_beta_schedule,
                report_rerank=config.report_rerank,
                precision=precision,
                model_ema=model_ema,
            )
        ),
        stage_metadata=metadata,
        checkpoint_metadata=_checkpoint_metadata(config, spec, data),
        auxiliary_state=Stage2AuxiliaryState(precision, model_ema),
    )
    if config.stage1_epochs <= 0:
        return stage2_job
    stage1_job = TrainingJob(
        model=shared_model,
        optimizer=optimizer_stage1,
        train_one_epoch=_train_one_epoch(stage1_runtime),
        validate=_noop_validate(),
        stage_metadata=metadata,
        checkpoint_metadata=_checkpoint_metadata(config, spec, data),
        auxiliary_state=precision,
    )
    return TwoStageTrainingJob(
        stage1=stage1_job,
        stage2=stage2_job,
        stage_metadata=metadata,
    )


def load_transformers_siglip2(model_name: str) -> Siglip2LoadResult:
    if model_name != SIGLIP2_MODEL_ID:
        raise ValueError(
            f"this training job only supports {SIGLIP2_MODEL_ID!r}, got {model_name!r}"
        )
    try:
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
    except ImportError as exc:
        raise ImportError("transformers with SigLIP 2 support is required") from exc
    model = AutoModel.from_pretrained(model_name)
    if getattr(getattr(model, "config", None), "model_type", None) != "siglip":
        raise ValueError(
            f"{SIGLIP2_MODEL_ID!r} must load as the fixed Transformers "
            "model_type 'siglip'"
        )
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        raise ValueError("SigLIP 2 processor must expose image_processor")
    return Siglip2LoadResult(model, image_processor, tokenizer)


def _validate_loaded_siglip2(
    loaded: Siglip2LoadResult,
    image_size: tuple[int, int],
) -> Siglip2ModelSpec:
    model = loaded.model
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    if model_type != "siglip":
        raise ValueError(
            "the fixed SigLIP 2 checkpoint must load as Transformers "
            f"model_type 'siglip', got {model_type!r}"
        )
    patch_size = siglip2_patch_size(model)
    max_num_patches = siglip2_max_num_patches(model)
    feature_dim = siglip2_feature_dim(model)
    text_hidden_dim = siglip2_text_hidden_dim(model)
    patch_count = validate_siglip2_image_size(image_size, model)

    uses_patchified_inputs = siglip2_uses_patchified_inputs(model)
    if uses_patchified_inputs:
        processor_patch_size = _processor_integer(loaded.image_processor, "patch_size")
        processor_max_patches = _processor_integer(
            loaded.image_processor, "max_num_patches"
        )
        if processor_patch_size != patch_size:
            raise ValueError(
                "SigLIP 2 processor/model patch size mismatch "
                f"({processor_patch_size} != {patch_size})"
            )
        if processor_max_patches != max_num_patches:
            raise ValueError(
                "SigLIP 2 processor/model patch budget mismatch "
                f"({processor_max_patches} != {max_num_patches}); load the "
                "processor from the same checkpoint instead of relying on class "
                "defaults"
            )
    else:
        _validate_fixed_image_processor(
            loaded.image_processor,
            model,
            patch_size,
            max_num_patches,
        )

    embeddings = getattr(getattr(model, "vision_model", None), "embeddings", None)
    embedding_patch_size = getattr(embeddings, "patch_size", None)
    embedding_num_patches = getattr(embeddings, "num_patches", None)
    position_embedding = getattr(embeddings, "position_embedding", None)
    position_count = getattr(
        getattr(position_embedding, "weight", None), "shape", (None,)
    )[0]
    if embedding_patch_size != patch_size:
        raise ValueError(
            "SigLIP 2 vision embeddings patch size disagrees with model config"
        )
    if embedding_num_patches != max_num_patches or position_count != max_num_patches:
        raise ValueError(
            "SigLIP 2 vision positional embedding budget disagrees with model config"
        )

    bos, eos, pad = _resolve_siglip2_token_ids(
        model,
        loaded.tokenizer,
        strict_config_match=uses_patchified_inputs,
    )
    text_padding_side = (
        "left"
        if uses_patchified_inputs
        else str(getattr(loaded.tokenizer, "padding_side", "right"))
    )
    if text_padding_side not in {"left", "right"}:
        raise ValueError(
            f"unsupported SigLIP 2 tokenizer padding side: {text_padding_side!r}"
        )
    if not uses_patchified_inputs and text_padding_side != "right":
        raise ValueError(
            "the fixed SigLIP 2 tokenizer must use right padding to preserve "
            "its final-position pooling semantics"
        )
    include_bos_token = uses_patchified_inputs
    mask_text_padding = uses_patchified_inputs or (
        "attention_mask" in tuple(getattr(loaded.tokenizer, "model_input_names", ()))
    )
    if not uses_patchified_inputs and mask_text_padding:
        raise ValueError(
            "the fixed SigLIP 2 tokenizer must expose the checkpoint-native "
            "input_ids-only text contract"
        )
    return Siglip2ModelSpec(
        feature_dim=feature_dim,
        text_hidden_dim=text_hidden_dim,
        patch_size=patch_size,
        max_num_patches=max_num_patches,
        patch_count=patch_count,
        vision_input_format=("patchified" if uses_patchified_inputs else "fixed_bchw"),
        text_padding_side=text_padding_side,
        include_bos_token=include_bos_token,
        mask_text_padding=mask_text_padding,
        bos_token_id=bos,
        eos_token_id=eos,
        pad_token_id=pad,
    )


def _validate_fixed_image_processor(
    image_processor: Any,
    model: Any,
    patch_size: int,
    max_num_patches: int,
) -> None:
    size = getattr(image_processor, "size", None)
    if isinstance(size, dict):
        height = size.get("height")
        width = size.get("width")
    else:
        height = getattr(size, "height", None)
        width = getattr(size, "width", None)
    if size is None:
        raise ValueError(
            "fixed SigLIP 2 image processor must expose a height/width size"
        )
    model_size = getattr(
        getattr(getattr(model, "config", None), "vision_config", None),
        "image_size",
        None,
    )
    if not all(isinstance(value, int) and value > 0 for value in (height, width)):
        raise ValueError(
            "fixed SigLIP 2 image processor size must contain positive height/width"
        )
    if not isinstance(model_size, int) or height != model_size or width != model_size:
        raise ValueError(
            "fixed SigLIP 2 processor/model pretraining size mismatch "
            f"({height}x{width} != {model_size}x{model_size})"
        )
    processor_budget = (height // patch_size) * (width // patch_size)
    if processor_budget != max_num_patches:
        raise ValueError(
            "fixed SigLIP 2 processor/model patch budget mismatch "
            f"({processor_budget} != {max_num_patches})"
        )


def _processor_integer(image_processor: Any, name: str) -> int:
    value = getattr(image_processor, name, None)
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or int(value[0]) != int(value[1]):
            raise ValueError(f"SigLIP 2 image processor {name} must be square")
        value = int(value[0])
    if not isinstance(value, int) or value < 1:
        raise ValueError(
            f"SigLIP 2 image processor must expose positive integer {name}"
        )
    return value


def _configure_gradient_checkpointing(model: torch.nn.Module, enabled: bool) -> None:
    method_name = (
        "gradient_checkpointing_enable" if enabled else "gradient_checkpointing_disable"
    )
    method = getattr(model, method_name, None)
    if not callable(method):
        raise ValueError(  # noqa: TRY004 - incompatible model contract
            f"SigLIP 2 model does not support {method_name}"
        )
    method()


def load_dataset_bundle(
    config: JobDataConfig,
    transforms,
    *,
    backend: str = PYTHON_DATA_BACKEND,
) -> DatasetBundle:
    backend = require_data_backend(backend)
    splits = _load_split_samples(config)
    _require_non_empty_splits(splits)
    extra_query_samples = [
        sample for _, samples in splits.extra_queries for sample in samples
    ]
    camera_map = build_camera_id_map(
        [*splits.train, *splits.query, *splits.gallery, *extra_query_samples]
    )
    train_person_map = build_person_id_map(splits.train)
    eval_person_map = build_person_id_map(
        [*splits.query, *splits.gallery, *extra_query_samples]
    )
    bundle = _transform_bundle(transforms)
    identity_counts, identity_camera_counts = _training_statistics(
        splits.train,
        train_person_map,
        camera_map,
    )
    return DatasetBundle(
        train=_build_image_dataset(
            splits.train, train_person_map, camera_map, bundle.train, backend
        ),
        train_eval=_build_image_dataset(
            splits.train, train_person_map, camera_map, bundle.eval, backend
        ),
        query=_build_image_dataset(
            splits.query, eval_person_map, camera_map, bundle.eval, backend
        ),
        gallery=_build_image_dataset(
            splits.gallery, eval_person_map, camera_map, bundle.eval, backend
        ),
        num_train_ids=len(train_person_map),
        num_cameras=len(camera_map),
        identity_counts=identity_counts,
        identity_camera_counts=identity_camera_counts,
        extra_queries=tuple(
            (
                name,
                _build_image_dataset(
                    samples, eval_person_map, camera_map, bundle.eval, backend
                ),
            )
            for name, samples in splits.extra_queries
        ),
        camera_raw_ids=tuple(sorted(camera_map, key=camera_map.__getitem__)),
    )


def _training_statistics(
    samples: Sequence[ReIDSample],
    person_id_map: dict[int, int],
    camera_id_map: dict[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    identity_camera_counts = torch.zeros(
        (len(person_id_map), len(camera_id_map)), dtype=torch.long
    )
    for sample in samples:
        person_id = person_id_map[sample.person_id]
        camera_id = camera_id_map[sample.camera_id]
        identity_camera_counts[person_id, camera_id] += 1
    return identity_camera_counts.sum(dim=1), identity_camera_counts


def _build_image_dataset(
    samples: Sequence[ReIDSample],
    person_id_map: dict[int, int],
    camera_id_map: dict[int, int],
    transform: Any,
    backend: str,
) -> ReIDImageDataset | ReIDMetadataDataset:
    if backend == PYTHON_DATA_BACKEND:
        return ReIDImageDataset(
            ReIDImageDatasetConfig(samples, person_id_map, camera_id_map, transform)
        )
    native_config = getattr(transform, "native_config", None)
    if not isinstance(native_config, ImageTransformConfig):
        raise TypeError(
            "Rust data backend requires a transform with ImageTransformConfig"
        )
    return ReIDMetadataDataset(
        ReIDMetadataDatasetConfig(samples, person_id_map, camera_id_map, native_config)
    )


def _transform_bundle(transforms) -> TransformBundle:
    if isinstance(transforms, TransformBundle):
        return transforms
    return TransformBundle(train=transforms, eval=transforms)


def _job_config_from_training_config(config: TrainingConfig) -> Siglip2ReIDJobConfig:
    if config.dataset is None:
        raise ValueError("dataset is required for t2c_reid.jobs.siglip2_reid")
    if config.data_root is None:
        raise ValueError("data_root is required for t2c_reid.jobs.siglip2_reid")
    if config.stage1_feature_cache and not config.freeze_image_encoder_stage1:
        raise ValueError(
            "stage1_feature_cache requires freeze_image_encoder_stage1=true: a "
            "trainable Stage-1 image encoder makes cached image features stale"
        )
    device = torch.device(config.device)
    image_size = (config.image_height, config.image_width)
    data_backend = require_data_backend(config.data_backend)
    evaluation_backend = require_evaluation_backend(config.evaluation_backend)
    pin_memory_arg = config.pin_memory
    pin_memory = (
        device.type == "cuda" if pin_memory_arg is None else bool(pin_memory_arg)
    )
    persistent_workers_arg = config.persistent_workers
    persistent_workers = (
        config.num_workers > 0
        if persistent_workers_arg is None
        else bool(persistent_workers_arg)
    )
    for value, name in (
        (config.batch_size, "batch_size"),
        (config.eval_batch_size, "eval_batch_size"),
        (config.gradient_accumulation_steps, "gradient_accumulation_steps"),
        (config.prefetch_factor, "prefetch_factor"),
        (config.rust_data_threads, "rust_data_threads"),
        (config.evaluation_chunk_size, "evaluation_chunk_size"),
        (config.image_height, "image_height"),
        (config.image_width, "image_width"),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if config.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if config.num_workers == 0 and persistent_workers_arg is True:
        raise ValueError("persistent_workers=true requires num_workers to be positive")
    precision = resolve_precision(config.precision, device)
    color_jitter = _validated_color_augmentation(
        config.color_jitter, config.grayscale_prob
    )
    if config.siglip2_model_name != SIGLIP2_MODEL_ID:
        raise ValueError(
            f"this training job only supports {SIGLIP2_MODEL_ID!r}, got "
            f"{config.siglip2_model_name!r}"
        )
    if config.image_encoder_frozen_layers < 0:
        raise ValueError("image_encoder_frozen_layers must be non-negative")
    if (
        not math.isfinite(config.image_encoder_layer_decay)
        or not 0.0 < config.image_encoder_layer_decay <= 1.0
    ):
        raise ValueError("image_encoder_layer_decay must satisfy 0 < decay <= 1")
    if (
        not math.isfinite(config.model_ema_decay)
        or not 0.0 <= config.model_ema_decay < 1.0
    ):
        raise ValueError("model_ema_decay must satisfy 0 <= decay < 1 (0 disables)")
    _validate_clothes_adversarial_config(config)
    _validate_tfc_config(
        tfc_weight=config.tfc_weight,
        head_momentum=config.tfc_momentum,
        tail_momentum=config.tfc_tail_momentum,
        class_balance_beta=config.tfc_class_balance_beta,
        local_weight=config.tfc_local_weight,
        global_weight=config.tfc_global_weight,
        cross_modal_weight=config.tfc_cross_modal_weight,
        cross_camera_weight=config.tfc_cross_camera_weight,
        contrast_temperature=config.tfc_contrast_temperature,
        transfer_reg_weight=config.tfc_transfer_reg_weight,
    )
    return Siglip2ReIDJobConfig(
        dataset=config.dataset,
        data_root=config.data_root,
        siglip2_model_name=config.siglip2_model_name,
        siglip2_checkpoint=config.siglip2_checkpoint,
        batch_size=config.batch_size,
        eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_workers=config.num_workers,
        data_backend=data_backend,
        prefetch_factor=config.prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        rust_data_threads=config.rust_data_threads,
        evaluation_backend=evaluation_backend,
        evaluation_chunk_size=config.evaluation_chunk_size,
        lr=config.lr,
        image_encoder_lr=config.image_encoder_lr,
        device=device,
        precision=precision,
        gradient_checkpointing=config.gradient_checkpointing,
        image_size=image_size,
        beta=config.beta,
        context_length=config.context_length,
        tfc_momentum=config.tfc_momentum,
        tfc_tail_momentum=config.tfc_tail_momentum,
        tfc_class_balance_beta=config.tfc_class_balance_beta,
        tfc_local_weight=config.tfc_local_weight,
        tfc_global_weight=config.tfc_global_weight,
        tfc_cross_modal_weight=config.tfc_cross_modal_weight,
        tfc_cross_camera_weight=config.tfc_cross_camera_weight,
        tfc_contrast_temperature=config.tfc_contrast_temperature,
        tfc_transfer_reg_weight=config.tfc_transfer_reg_weight,
        triplet_margin=config.triplet_margin,
        triplet_metric=config.triplet_metric,
        tfc_weight=config.tfc_weight,
        alignment_weight=config.alignment_weight,
        id_logit_scale=config.id_logit_scale,
        label_smoothing=config.label_smoothing,
        stage1_epochs=config.stage1_epochs,
        stage2_epochs=config.epochs,
        validation_interval=config.validation_interval,
        freeze_image_encoder_stage1=config.freeze_image_encoder_stage1,
        freeze_image_encoder_stage2=config.freeze_image_encoder_stage2,
        freeze_text_encoder=config.freeze_text_encoder,
        stage2_first_epoch=config.stage1_epochs + 1,
        freeze_prompt_bank_stage2=config.freeze_prompt_bank_stage2,
        reid_head=config.reid_head,
        retrieval_mode=require_retrieval_mode(config.retrieval_mode),
        beta_warmup_epochs=config.beta_warmup_epochs,
        report_rerank=config.report_rerank,
        stage1_lr_scheduler=config.stage1_lr_scheduler,
        stage1_warmup_epochs=config.stage1_warmup_epochs,
        stage2_lr_scheduler=config.stage2_lr_scheduler,
        stage2_warmup_epochs=config.stage2_warmup_epochs,
        num_instances=config.num_instances,
        camera_balanced_sampling=config.camera_balanced_sampling,
        color_jitter=color_jitter,
        grayscale_prob=config.grayscale_prob,
        sie_coe=config.sie_coe,
        stage1_feature_cache=config.stage1_feature_cache,
        grad_clip_norm=_validated_grad_clip_norm(config.grad_clip_norm),
        flip_tta=config.flip_tta,
        validation_holdout_ids=config.validation_holdout_ids,
        validation_holdout_seed=config.validation_holdout_seed,
        image_encoder_frozen_layers=config.image_encoder_frozen_layers,
        image_encoder_layer_decay=config.image_encoder_layer_decay,
        model_ema_decay=config.model_ema_decay,
        clothes_adversarial_weight=config.clothes_adversarial_weight,
        clothes_adversarial_start_epoch=config.clothes_adversarial_start_epoch,
        clothes_classifier_lr=config.clothes_classifier_lr,
    )


def _validate_clothes_adversarial_config(config: TrainingConfig) -> None:
    weight = config.clothes_adversarial_weight
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("clothes_adversarial_weight must be finite and non-negative")
    if config.clothes_adversarial_start_epoch < 1:
        raise ValueError("clothes_adversarial_start_epoch must be positive")
    lr = config.clothes_classifier_lr
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("clothes_classifier_lr must be finite and positive")
    if weight > 0.0 and config.dataset != "prcc":
        raise ValueError(
            "clothes_adversarial_weight > 0 requires dataset=prcc: clothes "
            "labels are derived from the PRCC camera outfit groups"
        )


def _validated_color_augmentation(
    color_jitter: Sequence[float], grayscale_prob: float
) -> tuple[float, float, float, float]:
    # Fail before the multi-gigabyte model load; the transform config owns the
    # actual range rules.
    values = tuple(float(value) for value in color_jitter)
    ImageTransformConfig(
        image_size=(1, 1),
        mean=(0.0, 0.0, 0.0),
        std=(1.0, 1.0, 1.0),
        training=True,
        color_jitter=values,
        grayscale_prob=grayscale_prob,
    )
    return values


def _validated_grad_clip_norm(value: float) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(
            "grad_clip_norm must be finite and non-negative (0 disables clipping)"
        )
    return value


def _validate_tfc_config(
    *,
    tfc_weight: float,
    head_momentum: float,
    tail_momentum: float,
    class_balance_beta: float,
    local_weight: float,
    global_weight: float,
    cross_modal_weight: float,
    cross_camera_weight: float,
    contrast_temperature: float,
    transfer_reg_weight: float,
) -> None:
    if not math.isfinite(tfc_weight) or tfc_weight < 0.0:
        raise ValueError("tfc_weight must be finite and non-negative")
    if not (
        math.isfinite(head_momentum)
        and math.isfinite(tail_momentum)
        and 0.0 <= head_momentum <= tail_momentum < 1.0
    ):
        raise ValueError("TFC momentum must satisfy 0 <= head <= tail < 1")
    if not math.isfinite(class_balance_beta) or not 0.0 <= class_balance_beta < 1.0:
        raise ValueError("tfc_class_balance_beta must satisfy 0 <= beta < 1")
    weights = (local_weight, global_weight, cross_modal_weight, cross_camera_weight)
    if any(
        not math.isfinite(weight) or weight < 0.0
        for weight in (*weights, transfer_reg_weight)
    ):
        raise ValueError("TFC loss weights must be finite and non-negative")
    if tfc_weight > 0.0 and sum(weights) == 0.0:
        raise ValueError(
            "positive tfc_weight requires at least one positive TFC main loss weight"
        )
    if not math.isfinite(contrast_temperature) or contrast_temperature <= 0.0:
        raise ValueError("tfc_contrast_temperature must be finite and positive")


def _build_runtimes(
    config: Siglip2ReIDJobConfig,
    model: Siglip2ReIDTrainingModel,
    loaders: LoaderBundle,
    num_train_ids: int,
    precision: PrecisionController,
    stage1_feature_cache: Stage1FeatureCache | None,
    model_ema: ModelEma | None = None,
) -> tuple[
    StageTrainingRuntime,
    StageTrainingRuntime,
    torch.optim.Optimizer,
    torch.optim.Optimizer,
    BetaSchedule | None,
]:
    _apply_freezing(model, config, stage=STAGE1)
    optimizer_stage1 = _build_optimizer(model, config)
    siglip2_model = _siglip2_model_for(model.retrieval_model)
    stage1_runtime = StageTrainingRuntime(
        model=model,
        loaders=loaders,
        optimizer=optimizer_stage1,
        stage=STAGE1,
        loss_config=_stage1_loss_config(siglip2_model),
        device=config.device,
        freeze_config=config,
        lr_scheduler=_build_stage1_lr_scheduler(optimizer_stage1, config),
        feature_cache=stage1_feature_cache,
        precision=precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        grad_clip_norm=config.grad_clip_norm,
    )
    _apply_freezing(model, config, stage=STAGE2)
    optimizer_stage2 = _build_optimizer(model, config)
    stage2_loss_config = _stage2_loss_config(config, siglip2_model)
    stage2_beta_schedule = BetaSchedule(
        beta=config.beta,
        warmup_epochs=config.beta_warmup_epochs,
        first_epoch=config.stage2_first_epoch,
    )
    stage2_lr_scheduler = _build_stage2_lr_scheduler(optimizer_stage2, config)
    stage2_anchor_provider = IdentityAnchorProvider(
        model.retrieval_model,
        num_train_ids=num_train_ids,
        # Fixed anchors require both the prompt bank and text tower frozen.
        frozen=config.freeze_prompt_bank_stage2 and config.freeze_text_encoder,
    )
    stage2_runtime = StageTrainingRuntime(
        model=model,
        loaders=loaders,
        optimizer=optimizer_stage2,
        stage=STAGE2,
        loss_config=stage2_loss_config,
        device=config.device,
        beta_schedule=stage2_beta_schedule,
        freeze_config=config,
        lr_scheduler=stage2_lr_scheduler,
        anchor_provider=stage2_anchor_provider,
        precision=precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        grad_clip_norm=config.grad_clip_norm,
        model_ema=model_ema,
        clothes_schedule=(
            ClothesAdversarialSchedule(
                weight=config.clothes_adversarial_weight,
                start_epoch=config.clothes_adversarial_start_epoch,
                first_epoch=config.stage2_first_epoch,
            )
            if model.clothes_classifier is not None
            else None
        ),
    )
    return (
        stage1_runtime,
        stage2_runtime,
        optimizer_stage1,
        optimizer_stage2,
        stage2_beta_schedule,
    )


def _build_stage1_feature_cache(
    config: Siglip2ReIDJobConfig,
    data: DatasetBundle,
) -> Stage1FeatureCache | None:
    if not config.stage1_feature_cache:
        return None
    return Stage1FeatureCache(data.train_eval, config)


def _apply_freezing(
    model: Siglip2ReIDTrainingModel, config: Siglip2ReIDJobConfig, stage: str
) -> None:
    retrieval = model.retrieval_model
    siglip2_model = _siglip2_model_for(retrieval)
    image_trainable = _image_encoder_trainable(config, stage)
    text_trainable = not config.freeze_text_encoder
    _set_module_requires_grad(siglip2_model.vision_model, image_trainable)
    if image_trainable and config.image_encoder_frozen_layers > 0:
        _freeze_lower_vision_layers(
            siglip2_model.vision_model, config.image_encoder_frozen_layers
        )
    # The SIE camera embedding feeds the vision tower, so it follows the
    # image-encoder freeze state of the current stage.
    if retrieval.image_encoder.sie_embedding is not None:
        _set_module_requires_grad(
            retrieval.image_encoder.sie_embedding, image_trainable
        )
    _set_module_requires_grad(siglip2_model.text_model, text_trainable)
    # SigLIP's calibrated temperature and bias remain fixed for both stages.
    siglip2_model.logit_scale.requires_grad_(False)
    siglip2_model.logit_bias.requires_grad_(False)
    prompt_trainable = stage == STAGE1 or not config.freeze_prompt_bank_stage2
    retrieval.prompt_bank.requires_grad_(prompt_trainable)
    if stage == STAGE1:
        # Stage-1 trains the prompt bank, so any precomputed camera text is stale.
        retrieval.set_inference_text_cache(None)
    model.classifier.requires_grad_(stage == STAGE2)
    model.tfc_bank.requires_grad_(stage == STAGE2)
    if model.clothes_classifier is not None:
        model.clothes_classifier.requires_grad_(stage == STAGE2)
    retrieval.feature_head.requires_grad_(stage == STAGE2)
    if isinstance(retrieval.feature_head, BNNeck):
        retrieval.feature_head.freeze_bias()


def _freeze_lower_vision_layers(vision_model: torch.nn.Module, count: int) -> None:
    # The patch/position embeddings sit below block 0, so freezing any block
    # freezes them too.
    layers = _vision_encoder_layers(vision_model)
    if count > len(layers):
        raise ValueError(
            f"image_encoder_frozen_layers={count} exceeds the {len(layers)} "
            "vision encoder layers"
        )
    _set_module_requires_grad(vision_model.embeddings, False)
    for layer in layers[:count]:
        _set_module_requires_grad(layer, False)


def _vision_encoder_layers(vision_model: torch.nn.Module) -> torch.nn.ModuleList:
    layers = getattr(getattr(vision_model, "encoder", None), "layers", None)
    if not isinstance(layers, torch.nn.ModuleList):
        raise TypeError(
            "partial vision freezing and layer-wise LR decay require "
            "vision_model.encoder.layers"
        )
    return layers


def _image_encoder_trainable(config: Siglip2ReIDJobConfig, stage: str) -> bool:
    if stage == STAGE1:
        return not config.freeze_image_encoder_stage1
    if stage == STAGE2:
        return not config.freeze_image_encoder_stage2
    raise ValueError(f"unknown training stage: {stage!r}")


def _siglip2_model_for(retrieval_model: T2CReIDModel) -> torch.nn.Module:
    if not isinstance(retrieval_model.image_encoder, TransformersSiglip2ImageEncoder):
        raise TypeError(
            "SigLIP 2 freezing requires a TransformersSiglip2ImageEncoder-backed "
            "retrieval model"
        )
    return retrieval_model.image_encoder.siglip2_model


def _set_module_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(value)


# Parameters owned by the pretrained SigLIP 2 vision tower use the smaller
# backbone learning rate.
BACKBONE_PARAMETER_PREFIXES = (
    "retrieval_model.image_encoder.siglip2_model.vision_model.",
)
# AdamW weight decay is applied to matrix weights only. Norms, biases,
# prompts, and the SIE embedding train without decay.
WEIGHT_DECAY = 1e-4
NO_DECAY_PARAMETER_PREFIXES = (
    "retrieval_model.prompt_bank.",
    "retrieval_model.image_encoder.sie_embedding.",
    "tfc_bank.camera_transfer_logits",
)
# The CAL clothes discriminator follows the official Simple-CCReID optimizer:
# its own constant learning rate, outside the Stage-2 warmup/cosine schedule.
# Under the shared schedule it would stay near chance through the warmup.
CLOTHES_PARAMETER_PREFIX = "clothes_classifier."
OPTIMIZER_FAMILY_ORDER = {"backbone": 0, "new": 1, "clothes": 2}


def _build_optimizer(
    model: Siglip2ReIDTrainingModel, config: Siglip2ReIDJobConfig
) -> torch.optim.Optimizer:
    # With layer-wise decay each vision layer id gets its own groups; layer
    # None keeps the single backbone family used when the decay is 1.
    layer_decay = config.image_encoder_layer_decay
    num_layers = (
        len(
            _vision_encoder_layers(
                _siglip2_model_for(model.retrieval_model).vision_model
            )
        )
        if layer_decay < 1.0
        else 0
    )
    grouped: dict[tuple[str, int | None, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith(BACKBONE_PARAMETER_PREFIXES)
        if backbone:
            family = "backbone"
        elif name.startswith(CLOTHES_PARAMETER_PREFIX):
            family = "clothes"
        else:
            family = "new"
        layer_id = (
            _vision_layer_id(name, num_layers) if backbone and layer_decay < 1.0 else None
        )
        no_decay = parameter.ndim <= 1 or name.startswith(NO_DECAY_PARAMETER_PREFIXES)
        grouped.setdefault((family, layer_id, no_decay), []).append(parameter)
    if not grouped:
        raise ValueError(
            "no trainable parameters were found for the requested stage; "
            "enable at least one of the prompt_bank/classifier/text_encoder"
        )
    family_lrs = {
        "backbone": config.image_encoder_lr,
        "new": config.lr,
        "clothes": config.clothes_classifier_lr,
    }
    # Deepest vision layer first, so param_groups[0] (the reported lr) keeps
    # the undecayed image_encoder_lr.
    ordered = sorted(
        grouped,
        key=lambda key: (
            OPTIMIZER_FAMILY_ORDER[key[0]],
            -(key[1] if key[1] is not None else 0),
            key[2],
        ),
    )
    param_groups: list[dict[str, Any]] = []
    for family, layer_id, no_decay in ordered:
        lr = family_lrs[family]
        name = family
        if layer_id is not None:
            lr *= layer_decay ** (num_layers + 1 - layer_id)
            name = f"{family}_layer{layer_id}"
        param_groups.append(
            {
                "params": grouped[(family, layer_id, no_decay)],
                "lr": lr,
                "weight_decay": 0.0 if no_decay else WEIGHT_DECAY,
                "name": f"{name}_no_decay" if no_decay else name,
                "fixed_lr": family == "clothes",
            }
        )
    return torch.optim.AdamW(param_groups)


def _vision_layer_id(name: str, num_layers: int) -> int:
    """0 for the embeddings, i + 1 for encoder block i, L + 1 for the head."""

    local = name.removeprefix(BACKBONE_PARAMETER_PREFIXES[0])
    if local.startswith("embeddings."):
        return 0
    if local.startswith("encoder.layers."):
        return int(local.split(".")[2]) + 1
    return num_layers + 1


def _pretrained_siglip_calibration(
    siglip2_model: torch.nn.Module,
) -> tuple[float, float]:
    logit_scale = getattr(siglip2_model, "logit_scale", None)
    logit_bias = getattr(siglip2_model, "logit_bias", None)
    if not isinstance(logit_scale, torch.Tensor):
        raise ValueError(  # noqa: TRY004 - incompatible model contract
            "SigLIP 2 model must expose a logit_scale tensor"
        )
    if not isinstance(logit_bias, torch.Tensor):
        raise ValueError(  # noqa: TRY004 - incompatible model contract
            "SigLIP 2 model must expose a logit_bias tensor"
        )
    return float(logit_scale.detach().exp()), float(logit_bias.detach())


def _stage1_loss_config(siglip2_model: torch.nn.Module) -> Stage1LossConfig:
    logit_scale, logit_bias = _pretrained_siglip_calibration(siglip2_model)
    return Stage1LossConfig(logit_scale=logit_scale, logit_bias=logit_bias)


def _stage2_loss_config(
    config: Siglip2ReIDJobConfig,
    siglip2_model: torch.nn.Module,
) -> Stage2LossConfig:
    logit_scale, logit_bias = _pretrained_siglip_calibration(siglip2_model)
    return Stage2LossConfig(
        logit_scale=logit_scale,
        logit_bias=logit_bias,
        triplet_margin=config.triplet_margin,
        triplet_metric=config.triplet_metric,
        tfc_weight=config.tfc_weight,
        alignment_weight=config.alignment_weight,
        id_logit_scale=config.id_logit_scale,
        label_smoothing=config.label_smoothing,
        tfc_tail_momentum=config.tfc_tail_momentum,
        tfc_class_balance_beta=config.tfc_class_balance_beta,
        tfc_local_weight=config.tfc_local_weight,
        tfc_global_weight=config.tfc_global_weight,
        tfc_cross_modal_weight=config.tfc_cross_modal_weight,
        tfc_cross_camera_weight=config.tfc_cross_camera_weight,
        tfc_contrast_temperature=config.tfc_contrast_temperature,
        tfc_transfer_reg_weight=config.tfc_transfer_reg_weight,
    )


def _build_stage1_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Siglip2ReIDJobConfig,
) -> StageLRScheduler | None:
    """Stage-1 shares :class:`StageLRScheduler`; Stage-1 epochs are already 1-based."""

    return _build_stage_lr_scheduler(
        optimizer,
        name=config.stage1_lr_scheduler,
        option="stage1_lr_scheduler",
        total_epochs=config.stage1_epochs,
        warmup_epochs=config.stage1_warmup_epochs,
        first_epoch=1,
    )


def _build_stage2_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Siglip2ReIDJobConfig,
) -> StageLRScheduler | None:
    return _build_stage_lr_scheduler(
        optimizer,
        name=config.stage2_lr_scheduler,
        option="stage2_lr_scheduler",
        total_epochs=config.stage2_epochs,
        warmup_epochs=config.stage2_warmup_epochs,
        first_epoch=config.stage2_first_epoch,
    )


def _build_stage_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    option: str,
    total_epochs: int,
    warmup_epochs: int,
    first_epoch: int,
) -> StageLRScheduler | None:
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"unsupported {option}: {name!r}")
    return StageLRScheduler(
        base_lrs=tuple(float(group["lr"]) for group in optimizer.param_groups),
        total_epochs=total_epochs,
        warmup_epochs=warmup_epochs,
        first_epoch=first_epoch,
    )


def _stage_metadata(
    config: Siglip2ReIDJobConfig,
    spec: Siglip2ModelSpec,
    data: DatasetBundle,
) -> StageMetadata:
    """Bundle two-stage config into the canonical ``StageMetadata`` container.

    Returning a ``StageMetadata`` (rather than a raw dict) keeps
    ``TwoStageTrainingJob.stage_metadata`` typing consistent and stops
    ``train.py`` from mistaking ``dict.values`` (a bound method) for a
    mapping when it records stage config in the tracking backend.
    """
    return StageMetadata(
        values={
            "checkpoint_schema_version": 3,
            "backbone_family": "siglip2",
            "dataset": config.dataset,
            "siglip2_model_name": config.siglip2_model_name,
            "siglip2_checkpoint": str(config.siglip2_checkpoint)
            if config.siglip2_checkpoint is not None
            else None,
            "stage1_epochs": config.stage1_epochs,
            "stage2_epochs": config.stage2_epochs,
            "stage2_first_epoch": config.stage2_first_epoch,
            "validation_interval": config.validation_interval,
            "validation_holdout_ids": config.validation_holdout_ids,
            "validation_holdout_seed": config.validation_holdout_seed,
            "batch_size": config.batch_size,
            "effective_batch_size": (
                config.batch_size * config.gradient_accumulation_steps
            ),
            "eval_batch_size": config.eval_batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "precision_requested": config.precision.requested,
            "precision_resolved": config.precision.resolved,
            "gradient_checkpointing": config.gradient_checkpointing,
            "num_workers": config.num_workers,
            "data_backend": config.data_backend,
            "prefetch_factor": config.prefetch_factor,
            "pin_memory": config.pin_memory,
            "persistent_workers": config.persistent_workers,
            "rust_data_threads": config.rust_data_threads,
            "evaluation_backend": config.evaluation_backend,
            "evaluation_chunk_size": config.evaluation_chunk_size,
            "lr": config.lr,
            "image_encoder_lr": config.image_encoder_lr,
            "image_encoder_frozen_layers": config.image_encoder_frozen_layers,
            "image_encoder_layer_decay": config.image_encoder_layer_decay,
            "model_ema_decay": config.model_ema_decay,
            "clothes_adversarial_weight": config.clothes_adversarial_weight,
            "clothes_adversarial_start_epoch": config.clothes_adversarial_start_epoch,
            "clothes_classifier_lr": config.clothes_classifier_lr,
            "beta": config.beta,
            "beta_warmup_epochs": config.beta_warmup_epochs,
            "alignment_weight": config.alignment_weight,
            "id_logit_scale": config.id_logit_scale,
            "label_smoothing": config.label_smoothing,
            "tfc_weight": config.tfc_weight,
            "triplet_margin": config.triplet_margin,
            "triplet_metric": config.triplet_metric,
            "tfc_momentum": config.tfc_momentum,
            "tfc_tail_momentum": config.tfc_tail_momentum,
            "tfc_class_balance_beta": config.tfc_class_balance_beta,
            "tfc_local_weight": config.tfc_local_weight,
            "tfc_global_weight": config.tfc_global_weight,
            "tfc_cross_modal_weight": config.tfc_cross_modal_weight,
            "tfc_cross_camera_weight": config.tfc_cross_camera_weight,
            "tfc_contrast_temperature": config.tfc_contrast_temperature,
            "tfc_transfer_reg_weight": config.tfc_transfer_reg_weight,
            "tfc_version": "camera_aware_cross_modal_v1",
            "num_train_ids": data.num_train_ids,
            "num_cameras": data.num_cameras,
            "pid_camera_count_fingerprint": _pid_camera_count_fingerprint(
                data.identity_camera_counts
            ),
            "context_length": config.context_length,
            "freeze_image_encoder_stage1": config.freeze_image_encoder_stage1,
            "freeze_image_encoder_stage2": config.freeze_image_encoder_stage2,
            "freeze_text_encoder": config.freeze_text_encoder,
            "freeze_prompt_bank_stage2": config.freeze_prompt_bank_stage2,
            "reid_head": config.reid_head,
            "retrieval_mode": config.retrieval_mode,
            "report_rerank": config.report_rerank,
            "stage2_lr_scheduler": config.stage2_lr_scheduler,
            "stage2_warmup_epochs": config.stage2_warmup_epochs,
            "num_instances": config.num_instances,
            "camera_balanced_sampling": config.camera_balanced_sampling,
            "color_jitter": list(config.color_jitter),
            "grayscale_prob": config.grayscale_prob,
            "sie_coe": config.sie_coe,
            "stage1_feature_cache": config.stage1_feature_cache,
            "feature_dim": spec.feature_dim,
            "image_size": "x".join(str(side) for side in config.image_size),
            "patch_size": spec.patch_size,
            "patch_count": spec.patch_count,
            "max_num_patches": spec.max_num_patches,
            "vision_input_format": spec.vision_input_format,
            "text_padding_side": spec.text_padding_side,
            "include_bos_token": spec.include_bos_token,
            "mask_text_padding": spec.mask_text_padding,
            "prompt_template_prefix": PROMPT_TEMPLATE_PREFIX,
            "prompt_template_suffix": PROMPT_TEMPLATE_SUFFIX,
        }
    )


def _checkpoint_metadata(
    config: Siglip2ReIDJobConfig,
    spec: Siglip2ModelSpec,
    data: DatasetBundle,
) -> dict[str, Any]:
    metadata = _base_checkpoint_metadata(config, spec, data)
    # Recorded only for holdout-capable datasets, so earlier schema-3
    # Market-1501/MSMT17 checkpoints still resume.
    if config.dataset in HOLDOUT_DATASETS:
        metadata["validation_holdout_ids"] = config.validation_holdout_ids
        if config.validation_holdout_ids:
            metadata["validation_holdout_seed"] = config.validation_holdout_seed
    # Recorded only when enabled, so checkpoints from before these options
    # still resume. They change the optimizer groups and the auxiliary state.
    if config.image_encoder_frozen_layers:
        metadata["image_encoder_frozen_layers"] = config.image_encoder_frozen_layers
    if config.image_encoder_layer_decay != 1.0:
        metadata["image_encoder_layer_decay"] = config.image_encoder_layer_decay
    if config.model_ema_decay:
        metadata["model_ema_decay"] = config.model_ema_decay
    if config.clothes_adversarial_weight:
        metadata["clothes_adversarial_weight"] = config.clothes_adversarial_weight
        metadata["clothes_adversarial_start_epoch"] = (
            config.clothes_adversarial_start_epoch
        )
    return metadata


def _base_checkpoint_metadata(
    config: Siglip2ReIDJobConfig,
    spec: Siglip2ModelSpec,
    data: DatasetBundle,
) -> dict[str, Any]:
    return {
        "schema_version": 3,
        "backbone_family": "siglip2",
        "dataset": config.dataset,
        "model_id": config.siglip2_model_name,
        "feature_dim": spec.feature_dim,
        "num_train_ids": data.num_train_ids,
        "num_cameras": data.num_cameras,
        "pid_camera_count_fingerprint": _pid_camera_count_fingerprint(
            data.identity_camera_counts
        ),
        "tfc_version": "camera_aware_cross_modal_v1",
        "tfc_weight": config.tfc_weight,
        "beta": config.beta,
        "beta_warmup_epochs": config.beta_warmup_epochs,
        "stage1_epochs": config.stage1_epochs,
        "stage2_first_epoch": config.stage2_first_epoch,
        "tfc_momentum": config.tfc_momentum,
        "tfc_tail_momentum": config.tfc_tail_momentum,
        "tfc_class_balance_beta": config.tfc_class_balance_beta,
        "tfc_local_weight": config.tfc_local_weight,
        "tfc_global_weight": config.tfc_global_weight,
        "tfc_cross_modal_weight": config.tfc_cross_modal_weight,
        "tfc_cross_camera_weight": config.tfc_cross_camera_weight,
        "tfc_contrast_temperature": config.tfc_contrast_temperature,
        "tfc_transfer_reg_weight": config.tfc_transfer_reg_weight,
        "image_size": tuple(config.image_size),
        "patch_size": spec.patch_size,
        "patch_count": spec.patch_count,
        "max_num_patches": spec.max_num_patches,
        "vision_input_format": spec.vision_input_format,
        "text_padding_side": spec.text_padding_side,
        "include_bos_token": spec.include_bos_token,
        "mask_text_padding": spec.mask_text_padding,
        "precision": config.precision.resolved,
    }


def _pid_camera_count_fingerprint(counts: torch.Tensor) -> str:
    if counts.dtype != torch.long or counts.ndim != 2:
        raise ValueError("pid-camera counts must be a rank-2 torch.long tensor")
    digest = hashlib.sha256()
    digest.update(int(counts.shape[0]).to_bytes(8, "little", signed=False))
    digest.update(int(counts.shape[1]).to_bytes(8, "little", signed=False))
    for value in counts.detach().cpu().contiguous().view(-1).tolist():
        digest.update(int(value).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _load_split_samples(config: JobDataConfig) -> SplitSamples:
    if not config.root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {config.root}")
    if config.validation_holdout_ids and config.dataset not in HOLDOUT_DATASETS:
        raise ValueError(
            f"validation_holdout_ids is supported only for {HOLDOUT_DATASETS}, "
            f"got {config.dataset!r}"
        )
    if config.dataset == "market1501":
        return SplitSamples(
            train=load_market_split(config.root, "train"),
            query=load_market_split(config.root, "query"),
            gallery=load_market_split(config.root, "gallery"),
        )
    if config.dataset == "msmt17":
        return SplitSamples(
            # Standard MSMT17 protocol trains on list_train.txt + list_val.txt.
            train=[
                *load_msmt17_manifest(config.root, "train"),
                *load_msmt17_manifest(config.root, "val"),
            ],
            query=load_msmt17_manifest(config.root, "query"),
            gallery=load_msmt17_manifest(config.root, "gallery"),
        )
    if config.dataset == "prcc":
        return _load_prcc_split_samples(config)
    raise ValueError(f"Unsupported dataset: {config.dataset}")


def _load_prcc_split_samples(config: JobDataConfig) -> SplitSamples:
    """PRCC CAL protocol: all camera-A gallery, C cross-clothes primary query.

    With ``validation_holdout_ids > 0`` the same protocol runs on identities
    held out of the training split and the test split is not loaded at all.
    """
    train = load_prcc_split(config.root, "train")
    if config.validation_holdout_ids:
        holdout = split_prcc_holdout(
            train, config.validation_holdout_ids, seed=config.validation_holdout_seed
        )
        return SplitSamples(
            train=holdout.train,
            query=holdout.query_cross,
            gallery=holdout.gallery,
            extra_queries=((SAME_CLOTHES_QUERY, holdout.query_same),),
        )
    return SplitSamples(
        train=train,
        query=load_prcc_split(config.root, "query_cross"),
        gallery=load_prcc_split(config.root, "gallery"),
        extra_queries=(
            (SAME_CLOTHES_QUERY, load_prcc_split(config.root, "query_same")),
        ),
    )


def _require_non_empty_splits(splits: SplitSamples) -> None:
    if not splits.train:
        raise ValueError("training split is empty")
    if not splits.query:
        raise ValueError("query split is empty")
    if not splits.gallery:
        raise ValueError("gallery split is empty")
    for name, samples in splits.extra_queries:
        if not samples:
            raise ValueError(f"{name} query split is empty")


def _build_training_model(
    config: Siglip2ReIDJobConfig,
    siglip2_model: torch.nn.Module,
    data: DatasetBundle,
    spec: Siglip2ModelSpec,
    prefix_token_ids: tuple[int, ...],
    suffix_token_ids: tuple[int, ...],
) -> Siglip2ReIDTrainingModel:
    prompt_bank = PromptBank(
        PromptConfig(
            num_cameras=data.num_cameras,
            num_train_ids=data.num_train_ids,
            context_length=config.context_length,
            embedding_dim=spec.text_hidden_dim,
        )
    )
    image_encoder = TransformersSiglip2ImageEncoder(
        siglip2_model, num_cameras=data.num_cameras, sie_coe=config.sie_coe
    )
    text_encoder = TransformersSiglip2TextEncoder(
        siglip2_model,
        context_length=config.context_length,
        bos_token_id=spec.bos_token_id,
        eos_token_id=spec.eos_token_id,
        pad_token_id=spec.pad_token_id,
        prefix_token_ids=prefix_token_ids,
        suffix_token_ids=suffix_token_ids,
        left_padding=spec.text_padding_side == "left",
        include_bos_token=spec.include_bos_token,
        mask_padding=spec.mask_text_padding,
    )
    retrieval = T2CReIDModel(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        prompt_bank=prompt_bank,
        beta=config.beta,
        feature_head=_build_feature_head(config.reid_head, spec.feature_dim),
    )
    classifier = torch.nn.Linear(spec.feature_dim, data.num_train_ids, bias=False)
    tfc_bank = CameraAwareTFCBank(
        identity_counts=data.identity_counts,
        identity_camera_counts=data.identity_camera_counts,
        feature_dim=spec.feature_dim,
        head_momentum=config.tfc_momentum,
        tail_momentum=config.tfc_tail_momentum,
        class_balance_beta=config.tfc_class_balance_beta,
    )
    return Siglip2ReIDTrainingModel(
        retrieval,
        classifier,
        tfc_bank,
        clothes_classifier=_build_clothes_classifier(config, data, spec.feature_dim),
    )


def _build_clothes_classifier(
    config: Siglip2ReIDJobConfig,
    data: DatasetBundle,
    feature_dim: int,
) -> ClothesClassifier | None:
    if config.clothes_adversarial_weight <= 0.0:
        return None
    unknown = sorted(set(data.camera_raw_ids) - set(PRCC_CLOTHES_GROUPS))
    if unknown or len(data.camera_raw_ids) != data.num_cameras:
        raise ValueError(
            f"clothes adversarial training requires PRCC cameras, got {unknown}"
        )
    table = build_clothes_table(
        data.identity_camera_counts,
        [PRCC_CLOTHES_GROUPS[camera] for camera in data.camera_raw_ids],
    )
    return ClothesClassifier(feature_dim, table)


def _load_siglip2_checkpoint_if_requested(
    model: torch.nn.Module,
    checkpoint: Path | None,
    device: torch.device,
) -> None:
    if checkpoint is None:
        return
    if not checkpoint.exists():
        raise FileNotFoundError(f"SigLIP 2 checkpoint does not exist: {checkpoint}")
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    state_dict = (
        payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    )
    if not isinstance(state_dict, dict):
        raise TypeError(
            "SigLIP 2 checkpoint must be a state_dict or contain a state_dict key"
        )
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"unexpected SigLIP 2 checkpoint keys: {unexpected}")
    if missing:
        raise ValueError(f"missing SigLIP 2 checkpoint keys: {missing}")


def _build_feature_head(reid_head: str, projection_dim: int) -> torch.nn.Module:
    if reid_head == "linear":
        return torch.nn.Identity()
    if reid_head == "bnneck":
        return BNNeck(projection_dim)
    raise ValueError(f"unsupported reid_head: {reid_head!r}")


def _resolve_siglip2_token_ids(
    siglip2_model: torch.nn.Module,
    tokenizer: Any,
    *,
    strict_config_match: bool,
) -> tuple[int, int, int]:
    if tokenizer is None:
        raise ValueError("a SigLIP 2 tokenizer is required")
    text_config = getattr(getattr(siglip2_model, "config", None), "text_config", None)
    vocab_size = getattr(text_config, "vocab_size", None)
    if not isinstance(vocab_size, int) or vocab_size < 1:
        raise ValueError("SigLIP 2 text config must expose a positive vocab_size")
    resolved: list[int] = []
    for name in ("bos_token_id", "eos_token_id", "pad_token_id"):
        tokenizer_id = getattr(tokenizer, name, None)
        config_id = getattr(text_config, name, None)
        if not isinstance(tokenizer_id, int):
            raise ValueError(  # noqa: TRY004 - incompatible tokenizer contract
                f"SigLIP 2 tokenizer must expose integer {name}"
            )
        if tokenizer_id < 0 or tokenizer_id >= vocab_size:
            raise ValueError(
                f"SigLIP 2 tokenizer {name}={tokenizer_id} is outside the model "
                f"vocabulary of {vocab_size} tokens"
            )
        if strict_config_match:
            if not isinstance(config_id, int):
                raise ValueError(f"SigLIP 2 text config must expose integer {name}")
            if tokenizer_id != config_id:
                raise ValueError(
                    f"SigLIP 2 tokenizer/model {name} mismatch "
                    f"({tokenizer_id} != {config_id})"
                )
        resolved.append(tokenizer_id)
    return resolved[0], resolved[1], resolved[2]


def _encode_template_token_ids(tokenizer: Any, text: str) -> tuple[int, ...]:
    """Encode a constant template fragment without adding special tokens."""
    if tokenizer is None:
        raise ValueError(
            "a SigLIP 2 tokenizer is required to encode the prompt template"
        )
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"]
    token_ids = tuple(int(token_id) for token_id in input_ids)
    if not token_ids:
        raise ValueError(f"prompt template fragment {text!r} encoded to no token ids")
    return token_ids


def _build_loaders(data: DatasetBundle, config: Siglip2ReIDJobConfig) -> LoaderBundle:
    return LoaderBundle(
        train=_train_loader(data.train, config),
        query=_loader(data.query, config, shuffle=False),
        gallery=_loader(data.gallery, config, shuffle=False),
        extra_queries=tuple(
            (name, _loader(dataset, config, shuffle=False))
            for name, dataset in data.extra_queries
        ),
    )


def _train_loader(
    dataset: ReIDImageDataset | ReIDMetadataDataset,
    config: Siglip2ReIDJobConfig,
) -> DataLoader:
    sampler = IdentityBalancedBatchSampler(
        dataset.person_ids,
        batch_size=config.batch_size,
        instances_per_identity=config.num_instances,
        camera_labels=dataset.camera_ids if config.camera_balanced_sampling else None,
    )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        **_data_loader_kwargs(dataset, config),
    )


def _loader(
    dataset: ReIDImageDataset | ReIDMetadataDataset,
    config: Siglip2ReIDJobConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        shuffle=shuffle,
        **_data_loader_kwargs(dataset, config),
    )


def _data_loader_kwargs(
    dataset: ReIDImageDataset | ReIDMetadataDataset,
    config: Siglip2ReIDJobConfig,
) -> dict[str, Any]:
    if isinstance(dataset, ReIDMetadataDataset):
        collate_fn = RustReIDBatchCollator(
            dataset.transform_config,
            threads=config.rust_data_threads,
        )
    else:
        collate_fn = collate_reid_batches
    options: dict[str, Any] = {
        "num_workers": config.num_workers,
        "collate_fn": collate_fn,
        "pin_memory": config.pin_memory,
        "persistent_workers": config.persistent_workers,
    }
    if config.num_workers > 0:
        options["prefetch_factor"] = config.prefetch_factor
    return options


def _train_one_epoch(runtime: StageTrainingRuntime):
    def train(epoch: int, reporter) -> dict[str, float]:
        if runtime.freeze_config is not None:
            _apply_freezing(runtime.model, runtime.freeze_config, runtime.stage)
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        if runtime.lr_scheduler is not None:
            runtime.lr_scheduler.apply(runtime.optimizer, epoch)
        if runtime.clothes_schedule is not None:
            runtime.clothes_schedule.apply(runtime.model.clothes_classifier, epoch)
        if runtime.anchor_provider is not None:
            runtime.anchor_provider.start_epoch()
        if runtime.stage == STAGE2 and runtime.freeze_config is not None:
            if runtime.precision is None:
                raise ValueError("stage2 training requires a precision controller")
            _ensure_camera_text_cache(
                runtime.model, runtime.freeze_config, runtime.precision
            )
        if runtime.model_ema is not None and not runtime.model_ema.initialized:
            # Bound after Stage-2 freezing, so it tracks exactly the Stage-2
            # trainable retrieval weights.
            runtime.model_ema.initialize()
        runtime.model.train()
        if runtime.precision is None:
            raise ValueError("training requires a precision controller")
        metric_names = _train_metric_names(runtime.stage)
        totals = {name: 0.0 for name in metric_names}
        micro_batch_count = 0
        source = reporter.batches(_train_batches(runtime))
        for window in batched(source, runtime.gradient_accumulation_steps):
            runtime.optimizer.zero_grad(set_to_none=True)
            window_totals = {name: 0.0 for name in metric_names}
            window_size = len(window)
            tfc_transaction = (
                runtime.stage == STAGE2
                and float(getattr(runtime.loss_config, "tfc_weight", 0.0)) > 0.0
            )
            if tfc_transaction:
                runtime.model.tfc_bank.begin_update_window()
            try:
                for batch in window:
                    with runtime.precision.autocast():
                        breakdown = _micro_batch_breakdown(
                            runtime, batch, runtime.stage
                        )
                    values = _breakdown_metric_values(breakdown, runtime.stage)
                    runtime.precision.backward(breakdown.total / window_size)
                    for name in metric_names:
                        totals[name] += values[name]
                        window_totals[name] += values[name]
                    micro_batch_count += 1
                runtime.precision.clip_grad_norm(
                    runtime.optimizer,
                    _clipped_parameters(runtime.model),
                    runtime.grad_clip_norm,
                )
                update_succeeded = runtime.precision.step(runtime.optimizer)
            except Exception:
                if tfc_transaction:
                    runtime.model.tfc_bank.rollback_update_window()
                raise
            if tfc_transaction:
                if update_succeeded:
                    runtime.model.tfc_bank.commit_update_window()
                else:
                    runtime.model.tfc_bank.rollback_update_window()
            if update_succeeded:
                if runtime.model_ema is not None:
                    runtime.model_ema.update()
                reported = {
                    name: window_totals[name] / window_size for name in metric_names
                }
                reported["lr"] = _optimizer_lr(runtime.optimizer)
                reporter.report_batch(reported)
        return _average_train_metrics(
            totals,
            micro_batch_count,
            runtime.optimizer,
            runtime.stage,
        )

    return train


def _clipped_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    # The clothes discriminator trains on detached features; keeping it out of
    # the global norm stops its gradients from rescaling the backbone update.
    clothes_classifier = getattr(model, "clothes_classifier", None)
    excluded = (
        set()
        if clothes_classifier is None
        else {id(p) for p in clothes_classifier.parameters()}
    )
    return [p for p in model.parameters() if id(p) not in excluded]


def _train_batches(runtime: StageTrainingRuntime):
    """The epoch's batch source: the PK train loader, or Stage-1 cached features.

    With the Stage-1 feature cache enabled, the (frozen) image tower runs once
    — lazily, at the start of the first Stage-1 training epoch — and every
    training step of every Stage-1 epoch is served from the cached tensors.
    """
    if runtime.stage == STAGE1 and runtime.feature_cache is not None:
        if runtime.precision is None:
            raise ValueError("stage1 feature cache requires a precision controller")
        runtime.feature_cache.ensure_extracted(runtime.model, runtime.precision)
        return runtime.feature_cache.batches()
    return runtime.loaders.train


def _noop_validate():
    def validate(epoch: int) -> ReIDMetrics:
        # Stage-1 has no ReID classifier trained yet; reporting random mAP would be misleading.
        return ReIDMetrics(
            map=float("nan"), cmc={rank: float("nan") for rank in DEFAULT_RANKS}
        )

    return validate


def _validate(runtime: ValidationRuntime):
    def validate(epoch: int) -> ReIDMetrics:
        _apply_freezing(runtime.model, runtime.model_config, STAGE2)
        if runtime.precision is None:
            raise ValueError("validation requires a precision controller")
        _ensure_camera_text_cache(
            runtime.model, runtime.model_config, runtime.precision
        )
        if runtime.beta_schedule is not None:
            runtime.beta_schedule.apply(runtime.model, epoch)
        runtime.model.eval()
        extras: dict[str, float] = {}
        if runtime.model_ema is not None:
            # Live-weight score of the same epoch, so one run shows whether
            # the EMA helps; the EMA score stays the selection metric.
            raw = _score_feature_sets(
                runtime,
                _validation_features(runtime, runtime.loaders.query),
                _validation_features(runtime, runtime.loaders.gallery),
                evaluate_reid,
            )
            extras["raw_mAP"] = raw.map
            extras["raw_rank_1"] = raw.cmc[1]
        with _model_ema_scope(runtime.model_ema):
            query = _validation_features(runtime, runtime.loaders.query)
            gallery = _validation_features(runtime, runtime.loaders.gallery)
            config = runtime.model_config
            # Secondary queries never drive model selection, so they are scored
            # only at the final epoch instead of on every validation pass.
            final_epoch = config.stage2_first_epoch + config.stage2_epochs - 1
            extra_queries = tuple(
                (name, _validation_features(runtime, loader))
                for name, loader in (
                    runtime.loaders.extra_queries if epoch >= final_epoch else ()
                )
            )
        metrics = _score_feature_sets(runtime, query, gallery, evaluate_reid)
        if query.image_only is not None and gallery.image_only is not None:
            # Fused-vs-image-only ablation from the same forward pass; the
            # primary mAP stays the configured retrieval_mode.
            image_only = _score_feature_sets(
                runtime, query.image_only, gallery.image_only, evaluate_reid
            )
            extras["image_only_mAP"] = image_only.map
            extras["image_only_rank_1"] = image_only.cmc[1]
        for name, features in extra_queries:
            extra = _score_feature_sets(runtime, features, gallery, evaluate_reid)
            extras[f"{name}_mAP"] = extra.map
            extras[f"{name}_rank_1"] = extra.cmc[1]
        if runtime.report_rerank:
            rerank = _score_feature_sets(
                runtime, query, gallery, evaluate_reid_with_rerank
            )
            extras["rerank_mAP"] = rerank.map
            extras["rerank_rank_1"] = rerank.cmc[1]
        return ReIDMetrics(map=metrics.map, cmc=metrics.cmc, extras=extras)

    return validate


def _model_ema_scope(model_ema: ModelEma | None) -> AbstractContextManager[None]:
    return nullcontext() if model_ema is None else model_ema.applied()


def _validation_features(runtime: ValidationRuntime, loader: DataLoader) -> FeatureSet:
    return _extract_features(
        runtime.model,
        loader,
        runtime.device,
        runtime.retrieval_mode,
        runtime.precision,
        flip_tta=runtime.model_config.flip_tta,
        with_image_only=runtime.retrieval_mode == FUSED_RETRIEVAL,
    )


def _score_feature_sets(
    runtime: ValidationRuntime,
    query: FeatureSet,
    gallery: FeatureSet,
    evaluate: Callable[..., ReIDMetrics],
) -> ReIDMetrics:
    return evaluate(
        query.features,
        gallery.features,
        query_ids=query.person_ids,
        gallery_ids=gallery.person_ids,
        query_cams=query.camera_ids,
        gallery_cams=gallery.camera_ids,
        ranks=DEFAULT_RANKS,
        backend=runtime.model_config.evaluation_backend,
        query_chunk_size=runtime.model_config.evaluation_chunk_size,
    )


@dataclass(frozen=True)
class FeatureSet:
    features: torch.Tensor
    person_ids: tuple[int, ...]
    camera_ids: tuple[int, ...]
    # Image-only view extracted alongside fused features, when requested.
    image_only: FeatureSet | None = None


def _extract_features(
    model: Siglip2ReIDTrainingModel,
    loader: DataLoader,
    device: torch.device,
    retrieval_mode: str,
    precision: PrecisionController | None = None,
    flip_tta: bool = False,
    with_image_only: bool = False,
) -> FeatureSet:
    """Extract retrieval features for one eval loader.

    ``with_image_only`` (fused mode only) also returns the image-only view from
    the same image forward, so both retrieval modes are scored per validation.
    """
    if precision is None:
        precision = PrecisionController(PrecisionPolicy("fp32", "fp32", device.type))
    if with_image_only and retrieval_mode != FUSED_RETRIEVAL:
        raise ValueError("with_image_only requires fused retrieval_mode")

    def encode(images: torch.Tensor, cameras: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if with_image_only:
            image_only, fused = model.encode_retrieval_views(images, cameras)
            return fused, image_only
        return (model.encode_retrieval(images, cameras, retrieval_mode=retrieval_mode),)

    view_parts: list[list[torch.Tensor]] = [[] for _ in range(2 if with_image_only else 1)]
    person_ids: list[int] = []
    camera_ids: list[int] = []
    with torch.no_grad(), precision.autocast():
        for batch in loader:
            images = batch.images.to(device, non_blocking=True)
            cameras = batch.camera_ids.to(device, non_blocking=True)
            views = encode(images, cameras)
            if flip_tta:
                # Both views already come back L2-normalized, so summing them
                # and renormalizing averages the two directions. Doubles eval
                # cost; training is untouched.
                flipped = encode(torch.flip(images, dims=(3,)), cameras)
                views = tuple(
                    l2_normalize(view.float() + flipped_view.float())
                    for view, flipped_view in zip(views, flipped, strict=True)
                )
            for parts, view in zip(view_parts, views, strict=True):
                parts.append(view.float().cpu())
            person_ids.extend(batch.original_person_ids)
            camera_ids.extend(batch.original_camera_ids)
    if not person_ids:
        raise ValueError(
            "eval loader produced no samples; cannot extract query/gallery features"
        )
    feature_sets = [
        FeatureSet(torch.cat(parts), tuple(person_ids), tuple(camera_ids))
        for parts in view_parts
    ]
    if with_image_only:
        return replace(feature_sets[0], image_only=feature_sets[1])
    return feature_sets[0]


def _training_batch(batch: ReIDImageBatch, device: torch.device) -> TrainingBatch:
    return TrainingBatch(
        images=batch.images.to(device, non_blocking=True),
        camera_ids=batch.camera_ids.to(device, non_blocking=True),
        person_ids=batch.person_ids.to(device, non_blocking=True),
    )


def _micro_batch_breakdown(
    runtime: StageTrainingRuntime,
    batch: ReIDImageBatch | Stage1CachedBatch,
    stage: str,
) -> Stage1LossBreakdown | Stage2LossBreakdown:
    if stage == STAGE1:
        return _stage1_step(runtime, batch)
    return _stage2_step(runtime, _training_batch(batch, runtime.device))


def _breakdown_metric_values(
    breakdown: Stage1LossBreakdown | Stage2LossBreakdown,
    stage: str,
) -> dict[str, float]:
    if stage == STAGE1:
        if not isinstance(breakdown, Stage1LossBreakdown):
            raise TypeError("stage1 must produce Stage1LossBreakdown")
        return _stage1_metric_values(breakdown)
    if not isinstance(breakdown, Stage2LossBreakdown):
        raise TypeError("stage2 must produce Stage2LossBreakdown")
    return _stage2_metric_values(breakdown)


def _stage1_step(
    runtime: StageTrainingRuntime,
    batch: ReIDImageBatch | Stage1CachedBatch,
) -> Stage1LossBreakdown:
    model = runtime.model.retrieval_model
    if isinstance(batch, Stage1CachedBatch):
        return stage1_alignment_loss_from_visual(
            model,
            l2_normalize(batch.visual_raw),
            camera_ids=batch.camera_ids,
            person_ids=batch.person_ids,
            config=runtime.loss_config,
        )
    return stage1_alignment_loss(
        model, _training_batch(batch, runtime.device), runtime.loss_config
    )


def _stage2_step(
    runtime: StageTrainingRuntime, batch: TrainingBatch
) -> Stage2LossBreakdown:
    if runtime.anchor_provider is None:
        raise ValueError("stage2 training requires an identity anchor provider")
    inputs = Stage2LossInputs(
        classifier=runtime.model.classifier,
        tfc_bank=runtime.model.tfc_bank,
        anchors=runtime.anchor_provider.anchors(),
        config=runtime.loss_config,
        clothes_classifier=getattr(runtime.model, "clothes_classifier", None),
    )
    return stage2_loss_breakdown(runtime.model.retrieval_model, batch, inputs)


def _ensure_camera_text_cache(
    model: Siglip2ReIDTrainingModel,
    config: Siglip2ReIDJobConfig,
    precision: PrecisionController,
) -> None:
    """Precompute the per-camera retrieval text once when it is provably constant.

    With the prompt bank frozen in Stage-2 AND the text encoder frozen, the
    ``encode_inference_text`` output is a constant per camera, so it is
    encoded once and indexed in every Stage-2 forward and validation pass.
    """
    if not (config.freeze_prompt_bank_stage2 and config.freeze_text_encoder):
        return
    retrieval = model.retrieval_model
    if retrieval.inference_text_cache is not None:
        return
    camera_prompts = retrieval.prompt_bank.camera_prompts
    camera_ids = torch.arange(camera_prompts.shape[0], device=camera_prompts.device)
    with torch.no_grad(), precision.autocast():
        cache = retrieval.encode_inference_text(camera_ids)
    retrieval.set_inference_text_cache(cache.detach())


def _train_metric_names(stage: str) -> tuple[str, ...]:
    if stage == STAGE1:
        return STAGE1_TRAIN_LOSS_METRIC_NAMES
    return STAGE2_TRAIN_LOSS_METRIC_NAMES


def _stage1_metric_values(breakdown: Stage1LossBreakdown) -> dict[str, float]:
    return {
        "loss": _tensor_metric_value(breakdown.total),
        "alignment_loss": _tensor_metric_value(breakdown.alignment),
    }


def _stage2_metric_values(breakdown: Stage2LossBreakdown) -> dict[str, float]:
    return {
        "loss": _tensor_metric_value(breakdown.total),
        "reid_loss": _tensor_metric_value(breakdown.identity),
        "triplet_loss": _tensor_metric_value(breakdown.triplet),
        "triplet_active_fraction": _tensor_metric_value(
            breakdown.triplet_active_fraction
        ),
        "alignment_loss": _tensor_metric_value(breakdown.alignment),
        "tfc_loss": _tensor_metric_value(breakdown.tfc),
        "tfc_local_loss": _tensor_metric_value(breakdown.tfc_local),
        "tfc_global_loss": _tensor_metric_value(breakdown.tfc_global),
        "tfc_cross_modal_loss": _tensor_metric_value(breakdown.tfc_cross_modal),
        "tfc_cross_camera_loss": _tensor_metric_value(breakdown.tfc_cross_camera),
        "tfc_transfer_reg_loss": _tensor_metric_value(
            breakdown.tfc_transfer_regularization
        ),
        "tfc_cross_camera_coverage": _tensor_metric_value(
            breakdown.tfc_cross_camera_coverage
        ),
        "clothes_loss": _tensor_metric_value(breakdown.clothes),
        "clothes_adversarial_loss": _tensor_metric_value(
            breakdown.clothes_adversarial
        ),
        "clothes_accuracy": _tensor_metric_value(breakdown.clothes_accuracy),
    }


def _average_train_metrics(
    totals: dict[str, float],
    batch_count: int,
    optimizer: torch.optim.Optimizer,
    stage: str,
) -> dict[str, float]:
    if batch_count < 1:
        raise ValueError(f"{stage} training loader produced no batches")
    averaged = {name: totals[name] / batch_count for name in totals}
    averaged["lr"] = _optimizer_lr(optimizer)
    return averaged


def _tensor_metric_value(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def _optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        raise ValueError("optimizer has no parameter groups")
    return float(optimizer.param_groups[0]["lr"])
