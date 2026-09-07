"""Loss functions used by the T2C-ReID training design."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from t2c_reid.features import l2_normalize

DEFAULT_LOGIT_SCALE = 1.0
DEFAULT_LOGIT_BIAS = 0.0
DEFAULT_MARGIN = 0.3
DEFAULT_TRIPLET_METRIC = "euclidean"


def supervised_siglip_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    person_ids: torch.Tensor,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    logit_bias: float = DEFAULT_LOGIT_BIAS,
) -> torch.Tensor:
    """Native SigLIP pairwise loss with every same-identity pair positive.

    PK batches repeat identities. The positive matrix therefore uses identity
    equality instead of the diagonal-only target used during generic SigLIP
    pretraining.
    """

    if image_features.shape != text_features.shape:
        raise ValueError("image_features and text_features must have identical shapes")
    _validate_person_ids(person_ids, image_features.shape[0])
    positives = person_ids.unsqueeze(0) == person_ids.unsqueeze(1)
    return _siglip_pairwise_loss(
        image_features,
        text_features,
        positives,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
    )


def siglip_identity_anchor_loss(
    image_features: torch.Tensor,
    text_anchors: torch.Tensor,
    person_ids: torch.Tensor,
    logit_scale: float = DEFAULT_LOGIT_SCALE,
    logit_bias: float = DEFAULT_LOGIT_BIAS,
) -> torch.Tensor:
    """Native SigLIP loss against the text anchors of every train identity."""

    if image_features.ndim != 2:
        raise ValueError("image_features must be a rank-2 tensor")
    if text_anchors.ndim != 2:
        raise ValueError("text_anchors must be a rank-2 tensor")
    if image_features.shape[1] != text_anchors.shape[1]:
        raise ValueError(
            "image_features and text_anchors must share the feature dimension"
        )
    _validate_person_ids(person_ids, image_features.shape[0])
    num_anchors = text_anchors.shape[0]
    if torch.any(person_ids < 0) or torch.any(person_ids >= num_anchors):
        raise ValueError(f"person_ids contains indices outside [0, {num_anchors})")
    anchor_ids = torch.arange(num_anchors, device=person_ids.device)
    positives = person_ids.unsqueeze(1) == anchor_ids.unsqueeze(0)
    return _siglip_pairwise_loss(
        image_features,
        text_anchors,
        positives,
        logit_scale=logit_scale,
        logit_bias=logit_bias,
    )


def _siglip_pairwise_loss(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    positives: torch.Tensor,
    *,
    logit_scale: float,
    logit_bias: float,
) -> torch.Tensor:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise ValueError("SigLIP features must both be rank-2 tensors")
    expected_shape = (image_features.shape[0], text_features.shape[0])
    if tuple(positives.shape) != expected_shape:
        raise ValueError(f"positive mask must have shape {expected_shape}")
    if positives.dtype != torch.bool:
        raise ValueError("positive mask must be boolean")

    # Normalize and score in FP32 even under autocast. This is the numerically
    # sensitive part of SigLIP's all-pairs logsigmoid objective.
    with torch.autocast(device_type=image_features.device.type, enabled=False):
        image = l2_normalize(image_features.float())
        text = l2_normalize(text_features.float())
        logits = float(logit_scale) * (image @ text.T) + float(logit_bias)
        signs = torch.where(
            positives,
            torch.ones((), dtype=logits.dtype, device=logits.device),
            -torch.ones((), dtype=logits.dtype, device=logits.device),
        )
        return -F.logsigmoid(signs * logits).sum(dim=1).mean()


def batch_hard_triplet_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    margin: float = DEFAULT_MARGIN,
    metric: str = DEFAULT_TRIPLET_METRIC,
) -> torch.Tensor:
    _validate_triplet_inputs(features, labels)
    # Autocast matmul rounding can cancel nearby squared distances and erase
    # their gradients after clamping. Mining must use the same FP32 distances.
    with torch.autocast(device_type=features.device.type, enabled=False):
        distances = _pairwise_distances(features.float(), metric)
        losses, valid = _anchor_losses(distances, labels, margin)
    # One host sync per call to honour the documented error. The per-anchor
    # loop this replaced synced once per sample, which dominated step time at
    # ReID-sized PK batches.
    if not bool(torch.any(valid)):
        raise ValueError(
            "batch_hard_triplet_loss requires at least one valid positive and negative"
        )
    return losses[valid].mean()


def _pairwise_distances(features: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "euclidean":
        squared_norms = features.pow(2).sum(dim=1)
        squared = (
            squared_norms.unsqueeze(1)
            + squared_norms.unsqueeze(0)
            - 2.0 * features @ features.T
        )
        return squared.clamp_min(1e-12).sqrt()
    if metric == "cosine":
        return 1.0 - l2_normalize(features) @ l2_normalize(features).T
    raise ValueError(f"unsupported triplet metric: {metric!r}")


def _anchor_losses(
    distances: torch.Tensor,
    labels: torch.Tensor,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batch-hard loss per anchor, plus the mask of anchors that have both a positive and a negative.

    Rows without a valid positive or negative still produce a finite entry, so
    the caller must drop them with the returned mask rather than trusting the
    value. Sentinels are finite (not +/-inf) so an all-masked row cannot turn
    into a NaN in the backward pass.
    """

    same_identity = labels.unsqueeze(0) == labels.unsqueeze(1)
    self_mask = torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    positive_mask = same_identity & ~self_mask
    negative_mask = ~same_identity
    valid = torch.any(positive_mask, dim=1) & torch.any(negative_mask, dim=1)

    # Quarter of the dtype maximum: far beyond any real distance, yet small
    # enough that ``hardest_positive - hardest_negative`` cannot overflow to
    # -inf on an invalid row even in FP16.
    unreachable = torch.finfo(distances.dtype).max / 4.0
    hardest_positive = (
        distances.masked_fill(~positive_mask, -unreachable).max(dim=1).values
    )
    hardest_negative = (
        distances.masked_fill(~negative_mask, unreachable).min(dim=1).values
    )
    return F.relu(hardest_positive - hardest_negative + margin), valid


def _validate_person_ids(person_ids: torch.Tensor, batch_size: int) -> None:
    if person_ids.dtype != torch.long or person_ids.ndim != 1:
        raise ValueError("person_ids must be a rank-1 torch.long tensor")
    if person_ids.shape[0] != batch_size:
        raise ValueError("person_ids must match the feature batch size")


def _validate_triplet_inputs(features: torch.Tensor, labels: torch.Tensor) -> None:
    if features.ndim != 2:
        raise ValueError("features must be a rank-2 tensor")
    if labels.dtype != torch.long or labels.ndim != 1:
        raise ValueError("labels must be a rank-1 torch.long tensor")
    if features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have the same batch size")
