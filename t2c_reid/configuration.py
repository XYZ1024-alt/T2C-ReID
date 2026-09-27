"""Hydra configuration contract for T2C-ReID training."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from hydra import compose, initialize_config_module
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf

from t2c_reid.datasets import SUPPORTED_DATA_BACKENDS
from t2c_reid.evaluation import SUPPORTED_EVALUATION_BACKENDS
from t2c_reid.precision import SUPPORTED_PRECISIONS
from t2c_reid.retrieval import SUPPORTED_RETRIEVAL_MODES
from t2c_reid.siglip2_backbone import SIGLIP2_MODEL_ID
from t2c_reid.wandb import WANDB_MODES

TRAIN_CONFIG_MODULE = "t2c_reid.configs.training"
EVALUATE_CONFIG_MODULE = "t2c_reid.configs.evaluation"
BENCHMARK_CONFIG_MODULE = "t2c_reid.configs.benchmark"
TRAIN_CONFIG_NAME = "train"
EVALUATE_CONFIG_NAME = "evaluate"
BENCHMARK_CONFIG_NAME = "benchmark"
SUPPORTED_DATASETS = ("market1501", "msmt17", "prcc")
HOLDOUT_DATASETS = ("prcc",)
SUPPORTED_BENCHMARK_MODES = ("all", "data", "evaluation", "rerank")
SUPPORTED_LR_SCHEDULERS = ("none", "cosine")
SUPPORTED_REID_HEADS = ("linear", "bnneck")
SUPPORTED_RSS_WORKER_BACKENDS = ("python", "rust")
SUPPORTED_TRIPLET_METRICS = ("euclidean", "cosine")


@dataclass(kw_only=True)
class TrainingConfig:
    """Typed application configuration composed by Hydra."""

    job_builder: str
    epochs: int
    validation_interval: int
    checkpoint_dir: Path
    resume: Path | None
    seed: int | None
    enable_wandb: bool
    wandb_project: str
    wandb_entity: str | None
    wandb_mode: str
    wandb_dir: Path | None
    run_name: str
    dataset: str
    data_root: Path
    validation_holdout_ids: int
    validation_holdout_seed: int
    siglip2_model_name: str
    siglip2_checkpoint: Path | None
    batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    precision: str
    image_height: int
    image_width: int
    num_instances: int
    camera_balanced_sampling: bool
    color_jitter: list[float]
    grayscale_prob: float
    num_workers: int
    data_backend: str
    prefetch_factor: int
    pin_memory: bool | None
    persistent_workers: bool | None
    rust_data_threads: int
    evaluation_backend: str
    evaluation_chunk_size: int
    lr: float
    image_encoder_lr: float
    image_encoder_frozen_layers: int
    image_encoder_layer_decay: float
    model_ema_decay: float
    clothes_adversarial_weight: float
    clothes_adversarial_start_epoch: int
    sie_coe: float
    device: str
    beta: float
    beta_warmup_epochs: int
    stage1_lr_scheduler: str
    stage1_warmup_epochs: int
    stage2_lr_scheduler: str
    stage2_warmup_epochs: int
    grad_clip_norm: float
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
    stage1_epochs: int
    stage1_feature_cache: bool
    id_logit_scale: float
    label_smoothing: float
    reid_head: str
    retrieval_mode: str
    report_rerank: bool
    flip_tta: bool
    gradient_checkpointing: bool
    freeze_prompt_bank_stage2: bool
    freeze_image_encoder_stage1: bool
    freeze_image_encoder_stage2: bool
    freeze_text_encoder: bool
    sanity_gate_epochs: int
    sanity_gate_factor: float


@dataclass(kw_only=True)
class EvaluationConfig:
    """Typed application configuration for NPZ feature evaluation."""

    features: Path
    output: Path | None
    ranks: list[int]
    evaluation_backend: str
    evaluation_chunk_size: int
    report_rerank: bool
    rerank_k1: int
    rerank_k2: int
    rerank_lambda: float


@dataclass(kw_only=True)
class BenchmarkConfig:
    """Typed application configuration for native backend benchmarks."""

    mode: str
    dataset: str
    data_root: Path | None
    data_samples: int
    batch_size: int
    num_workers: int
    rust_data_threads: int
    image_height: int
    image_width: int
    query_count: int
    gallery_count: int
    feature_dim: int
    rerank_query_count: int
    rerank_gallery_count: int
    chunk_size: int
    runs: int
    warmup_runs: int
    seed: int
    output: Path | None
    rss_worker_backend: str | None


def register_configs() -> None:
    store = ConfigStore.instance()
    store.store(name="train_schema", node=TrainingConfig)
    store.store(name="evaluate_schema", node=EvaluationConfig)
    store.store(name="benchmark_schema", node=BenchmarkConfig)


def compose_training_config(overrides: Sequence[str] = ()) -> TrainingConfig:
    """Compose the packaged training config for programmatic callers and tests."""

    config = _compose_application_config(
        TRAIN_CONFIG_MODULE, TRAIN_CONFIG_NAME, overrides
    )
    return training_config_from_dict_config(config)


def compose_evaluation_config(overrides: Sequence[str]) -> EvaluationConfig:
    config = _compose_application_config(
        EVALUATE_CONFIG_MODULE,
        EVALUATE_CONFIG_NAME,
        overrides,
    )
    resolved = _config_from_dict_config(config, EvaluationConfig)
    validate_evaluation_config(resolved)
    return resolved


def compose_benchmark_config(overrides: Sequence[str] = ()) -> BenchmarkConfig:
    config = _compose_application_config(
        BENCHMARK_CONFIG_MODULE,
        BENCHMARK_CONFIG_NAME,
        overrides,
    )
    resolved = _config_from_dict_config(config, BenchmarkConfig)
    validate_benchmark_config(resolved)
    return resolved


def training_config_from_dict_config(config: DictConfig) -> TrainingConfig:
    resolved = _config_from_dict_config(config, TrainingConfig)
    validate_training_config(resolved)
    return resolved


def evaluation_config_from_dict_config(config: DictConfig) -> EvaluationConfig:
    resolved = _config_from_dict_config(config, EvaluationConfig)
    validate_evaluation_config(resolved)
    return resolved


def benchmark_config_from_dict_config(config: DictConfig) -> BenchmarkConfig:
    resolved = _config_from_dict_config(config, BenchmarkConfig)
    validate_benchmark_config(resolved)
    return resolved


def _compose_application_config(
    config_module: str,
    config_name: str,
    overrides: Sequence[str],
) -> DictConfig:
    with initialize_config_module(config_module=config_module, job_name=config_name):
        return compose(config_name=config_name, overrides=list(overrides))


def _config_from_dict_config[ConfigT](
    config: DictConfig,
    expected_type: type[ConfigT],
) -> ConfigT:
    resolved = OmegaConf.to_object(config)
    if not isinstance(resolved, expected_type):
        raise TypeError(
            f"Hydra config did not resolve to {expected_type.__name__}; "
            f"got {type(resolved).__name__}"
        )
    return resolved


def validate_training_config(config: TrainingConfig) -> None:
    choices = (
        ("dataset", config.dataset, SUPPORTED_DATASETS),
        ("precision", config.precision, SUPPORTED_PRECISIONS),
        ("data_backend", config.data_backend, SUPPORTED_DATA_BACKENDS),
        (
            "evaluation_backend",
            config.evaluation_backend,
            SUPPORTED_EVALUATION_BACKENDS,
        ),
        ("wandb_mode", config.wandb_mode, WANDB_MODES),
        ("stage1_lr_scheduler", config.stage1_lr_scheduler, SUPPORTED_LR_SCHEDULERS),
        ("stage2_lr_scheduler", config.stage2_lr_scheduler, SUPPORTED_LR_SCHEDULERS),
        ("triplet_metric", config.triplet_metric, SUPPORTED_TRIPLET_METRICS),
        ("reid_head", config.reid_head, SUPPORTED_REID_HEADS),
        ("retrieval_mode", config.retrieval_mode, SUPPORTED_RETRIEVAL_MODES),
    )
    for name, value, supported in choices:
        if value not in supported:
            raise ValueError(f"{name} must be one of {supported}, got {value!r}")
    if config.siglip2_model_name != SIGLIP2_MODEL_ID:
        raise ValueError(
            f"siglip2_model_name must be {SIGLIP2_MODEL_ID!r}, "
            f"got {config.siglip2_model_name!r}"
        )
    if config.validation_holdout_ids < 0:
        raise ValueError("validation_holdout_ids must be non-negative")
    if config.validation_holdout_ids > 0 and config.dataset not in HOLDOUT_DATASETS:
        raise ValueError(
            f"validation_holdout_ids is supported only for {HOLDOUT_DATASETS}, "
            f"got dataset {config.dataset!r}"
        )
    if (
        config.dataset in HOLDOUT_DATASETS
        and config.validation_holdout_ids == 0
        and config.validation_interval < config.epochs
    ):
        # Without a holdout the validation set is the test split; validating
        # before the final epoch would select best.pth on test labels.
        raise ValueError(
            f"{config.dataset} with validation_holdout_ids=0 evaluates the test "
            "split, so validation_interval must be >= epochs (final epoch only); "
            f"got validation_interval={config.validation_interval}, epochs={config.epochs}"
        )


def validate_evaluation_config(config: EvaluationConfig) -> None:
    if config.evaluation_backend not in SUPPORTED_EVALUATION_BACKENDS:
        raise ValueError(
            "evaluation_backend must be one of "
            f"{SUPPORTED_EVALUATION_BACKENDS}, got {config.evaluation_backend!r}"
        )
    if not config.ranks or any(rank < 1 for rank in config.ranks):
        raise ValueError("ranks must contain positive integers")
    if config.evaluation_chunk_size < 1:
        raise ValueError("evaluation_chunk_size must be positive")


def validate_benchmark_config(config: BenchmarkConfig) -> None:
    if config.mode not in SUPPORTED_BENCHMARK_MODES:
        raise ValueError(
            f"mode must be one of {SUPPORTED_BENCHMARK_MODES}, got {config.mode!r}"
        )
    if config.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got {config.dataset!r}"
        )
    if (
        config.rss_worker_backend is not None
        and config.rss_worker_backend not in SUPPORTED_RSS_WORKER_BACKENDS
    ):
        raise ValueError(
            "rss_worker_backend must be null or one of "
            f"{SUPPORTED_RSS_WORKER_BACKENDS}, got {config.rss_worker_backend!r}"
        )


register_configs()
