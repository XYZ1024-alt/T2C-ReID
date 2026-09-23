"""Stage-1 and Stage-2 loss wiring for T2C-ReID.

Stage-1 aligns image features with identity-aware prompt features. Stage-2
combines ReID, all-identity SigLIP alignment, and camera-aware cross-modal TFC::

    L_total = L_id + L_triplet + alignment_weight * L_alignment + tfc_weight * L_TFC
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from t2c_reid.features import l2_normalize
from t2c_reid.losses import (
    batch_hard_triplet,
    siglip_identity_anchor_loss,
    supervised_siglip_loss,
)
from t2c_reid.model import T2CReIDModel
from t2c_reid.tfc import (
    CameraAwareTFCBank,
    CameraAwareTFCLossConfig,
    DEFAULT_CLASS_BALANCE_BETA,
    DEFAULT_CONTRAST_TEMPERATURE,
    DEFAULT_CROSS_CAMERA_WEIGHT,
    DEFAULT_CROSS_MODAL_WEIGHT,
    DEFAULT_GLOBAL_WEIGHT,
    DEFAULT_LOCAL_WEIGHT,
    DEFAULT_TAIL_MOMENTUM,
    DEFAULT_TRANSFER_REG_WEIGHT,
)

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_LOGIT_BIAS = 0.0
DEFAULT_TRIPLET_MARGIN = 0.3
DEFAULT_TRIPLET_METRIC = "euclidean"
DEFAULT_TFC_WEIGHT = 1.0
DEFAULT_ALIGNMENT_WEIGHT = 1.0
DEFAULT_LABEL_SMOOTHING = 0.0
DEFAULT_ID_LOGIT_SCALE = 1.0


@dataclass(frozen=True)
class TrainingBatch:
    images: torch.Tensor
    camera_ids: torch.Tensor
    person_ids: torch.Tensor


@dataclass(frozen=True)
class Stage1LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    logit_bias: float = DEFAULT_LOGIT_BIAS


@dataclass(frozen=True)
class Stage2LossConfig:
    logit_scale: float = DEFAULT_LOGIT_SCALE
    logit_bias: float = DEFAULT_LOGIT_BIAS
    triplet_margin: float = DEFAULT_TRIPLET_MARGIN
    triplet_metric: str = DEFAULT_TRIPLET_METRIC
    tfc_weight: float = DEFAULT_TFC_WEIGHT
    alignment_weight: float = DEFAULT_ALIGNMENT_WEIGHT
    label_smoothing: float = DEFAULT_LABEL_SMOOTHING
    id_logit_scale: float = DEFAULT_ID_LOGIT_SCALE
    tfc_tail_momentum: float = DEFAULT_TAIL_MOMENTUM
    tfc_class_balance_beta: float = DEFAULT_CLASS_BALANCE_BETA
    tfc_local_weight: float = DEFAULT_LOCAL_WEIGHT
    tfc_global_weight: float = DEFAULT_GLOBAL_WEIGHT
    tfc_cross_modal_weight: float = DEFAULT_CROSS_MODAL_WEIGHT
    tfc_cross_camera_weight: float = DEFAULT_CROSS_CAMERA_WEIGHT
    tfc_contrast_temperature: float = DEFAULT_CONTRAST_TEMPERATURE
    tfc_transfer_reg_weight: float = DEFAULT_TRANSFER_REG_WEIGHT

    def tfc_loss_config(self) -> CameraAwareTFCLossConfig:
        return CameraAwareTFCLossConfig(
            local_weight=self.tfc_local_weight,
            global_weight=self.tfc_global_weight,
            cross_modal_weight=self.tfc_cross_modal_weight,
            cross_camera_weight=self.tfc_cross_camera_weight,
            contrast_temperature=self.tfc_contrast_temperature,
            transfer_reg_weight=self.tfc_transfer_reg_weight,
        )


@dataclass(frozen=True)
class Stage2LossInputs:
    classifier: torch.nn.Module
    tfc_bank: CameraAwareTFCBank
    anchors: torch.Tensor
    config: Stage2LossConfig = field(default_factory=Stage2LossConfig)


@dataclass(frozen=True)
class Stage1LossBreakdown:
    alignment: torch.Tensor

    @property
    def total(self) -> torch.Tensor:
        return self.alignment


@dataclass(frozen=True)
class Stage2LossBreakdown:
    alignment: torch.Tensor
    identity: torch.Tensor
    triplet: torch.Tensor
    triplet_active_fraction: torch.Tensor
    tfc: torch.Tensor
    tfc_local: torch.Tensor
    tfc_global: torch.Tensor
    tfc_cross_modal: torch.Tensor
    tfc_cross_camera: torch.Tensor
    tfc_transfer_regularization: torch.Tensor
    tfc_cross_camera_coverage: torch.Tensor
    tfc_weight: float
    alignment_weight: float

    @property
    def total(self) -> torch.Tensor:
        return (
            self.identity
            + self.triplet
            + self.alignment_weight * self.alignment
            + self.tfc_weight * self.tfc
        )


def stage1_alignment_loss(
    model: T2CReIDModel,
    batch: TrainingBatch,
    config: Stage1LossConfig,
) -> Stage1LossBreakdown:
    visual = model.encode_visual(batch.images, batch.camera_ids)
    return stage1_alignment_loss_from_visual(
        model,
        visual,
        camera_ids=batch.camera_ids,
        person_ids=batch.person_ids,
        config=config,
    )


def stage1_alignment_loss_from_visual(
    model: T2CReIDModel,
    visual: torch.Tensor,
    camera_ids: torch.Tensor,
    person_ids: torch.Tensor,
    config: Stage1LossConfig,
) -> Stage1LossBreakdown:
    """Stage-1 alignment from a precomputed normalized image feature."""

    text = model.encode_training_text(camera_ids, person_ids)
    alignment = supervised_siglip_loss(
        visual,
        text,
        person_ids,
        logit_scale=config.logit_scale,
        logit_bias=config.logit_bias,
    )
    return Stage1LossBreakdown(alignment=alignment)


def stage2_loss_breakdown(
    model: T2CReIDModel,
    batch: TrainingBatch,
    inputs: Stage2LossInputs,
) -> Stage2LossBreakdown:
    """Compute Stage-2 losses from one image forward."""

    outputs = model.forward_stage2(batch.images, batch.camera_ids, batch.person_ids)
    logits = inputs.classifier(outputs["bn"]) * inputs.config.id_logit_scale
    alignment = siglip_identity_anchor_loss(
        outputs["visual"],
        inputs.anchors,
        batch.person_ids,
        logit_scale=inputs.config.logit_scale,
        logit_bias=inputs.config.logit_bias,
    )
    identity = F.cross_entropy(
        logits,
        batch.person_ids,
        label_smoothing=inputs.config.label_smoothing,
    )
    triplet = batch_hard_triplet(
        outputs["visual_raw"],
        batch.person_ids,
        inputs.config.triplet_margin,
        metric=inputs.config.triplet_metric,
    )

    if inputs.config.tfc_weight == 0.0:
        zero = outputs["retrieval"].float().sum() * 0.0
        return Stage2LossBreakdown(
            alignment=alignment,
            identity=identity,
            triplet=triplet.loss,
            triplet_active_fraction=triplet.active_fraction,
            tfc=zero,
            tfc_local=zero,
            tfc_global=zero,
            tfc_cross_modal=zero,
            tfc_cross_camera=zero,
            tfc_transfer_regularization=zero,
            tfc_cross_camera_coverage=zero.detach(),
            tfc_weight=inputs.config.tfc_weight,
            alignment_weight=inputs.config.alignment_weight,
        )

    visual = l2_normalize(outputs["bn"])
    text = _encode_tfc_text_teacher(model, batch.camera_ids, batch.person_ids)
    inputs.tfc_bank.update(visual, text, batch.person_ids, batch.camera_ids)
    tfc = inputs.tfc_bank.loss(
        outputs["retrieval"],
        visual,
        batch.person_ids,
        batch.camera_ids,
        beta=model.beta,
        config=inputs.config.tfc_loss_config(),
    )
    return Stage2LossBreakdown(
        alignment=alignment,
        identity=identity,
        triplet=triplet.loss,
        triplet_active_fraction=triplet.active_fraction,
        tfc=tfc.total,
        tfc_local=tfc.local,
        tfc_global=tfc.global_center,
        tfc_cross_modal=tfc.cross_modal,
        tfc_cross_camera=tfc.cross_camera,
        tfc_transfer_regularization=tfc.transfer_regularization,
        tfc_cross_camera_coverage=tfc.cross_camera_coverage,
        tfc_weight=inputs.config.tfc_weight,
        alignment_weight=inputs.config.alignment_weight,
    )


def _encode_tfc_text_teacher(
    model: T2CReIDModel,
    camera_ids: torch.Tensor,
    person_ids: torch.Tensor,
) -> torch.Tensor:
    """Encode exact training prompts without gradients or text-tower dropout."""

    siglip2_model = getattr(model.text_encoder, "siglip2_model", None)
    text_tower = getattr(siglip2_model, "text_model", None)
    was_training = bool(getattr(text_tower, "training", False))
    if isinstance(text_tower, torch.nn.Module):
        text_tower.eval()
    try:
        with torch.no_grad():
            return model.encode_training_text(camera_ids, person_ids).detach()
    finally:
        if isinstance(text_tower, torch.nn.Module):
            text_tower.train(was_training)
