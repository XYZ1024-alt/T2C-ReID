"""Clothes-based adversarial loss (CAL) for clothes-changing ReID.

Reference: Gu et al., "Clothes-Changing Person Re-identification with RGB
Modality Only", CVPR 2022 (official Simple-CCReID implementation).

A cosine clothes classifier is trained with cross-entropy on *detached*
Stage-2 features, so it only learns to recognise clothes. The backbone is
trained against the *detached* classifier weights with a multi-positive
cross-entropy whose positives are every clothes class of the sample's
identity: its denominator excludes the other clothes of the same identity, so
the feature is never rewarded for separating one person's outfits, and the
``epsilon`` share of the target pulls it toward all of them::

    log p_c = s * cos(f, w_c) - log(exp(s * cos(f, w_c)) + sum_{j in S-} exp(s * cos(f, w_j)))
    q_c     = (1 - epsilon) * [c == y_c] + epsilon / |S+| * [c in S+]
    L_adv   = -sum_c q_c * log p_c

Both gradient paths are disjoint, so one backward over
``L_clothes + weight * L_adv`` reproduces the official two-optimizer update.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from t2c_reid.features import l2_normalize

CLOTHES_LOGIT_SCALE = 16.0
CLOTHES_ADVERSARIAL_EPSILON = 0.1


def build_clothes_table(
    identity_camera_counts: torch.Tensor,
    camera_groups: Sequence[int],
) -> torch.Tensor:
    """Map every observed ``(pid, camera)`` pair to a dense clothes class.

    ``camera_groups[c]`` is the outfit group of camera index ``c``; cameras in
    the same group show the same clothes. Classes are numbered pid-major in
    group order. Unobserved pairs map to ``-1``.
    """

    if identity_camera_counts.ndim != 2:
        raise ValueError("identity_camera_counts must be a (num_ids, num_cameras) tensor")
    num_ids, num_cameras = identity_camera_counts.shape
    if len(camera_groups) != num_cameras:
        raise ValueError(
            f"camera_groups has {len(camera_groups)} entries for {num_cameras} cameras"
        )
    if num_ids < 2:
        raise ValueError("the clothes adversarial loss requires at least two identities")
    groups = sorted(set(camera_groups))
    table = torch.full((num_ids, num_cameras), -1, dtype=torch.long)
    next_class = 0
    for person in range(num_ids):
        for group in groups:
            cameras = [
                camera
                for camera in range(num_cameras)
                if camera_groups[camera] == group
                and int(identity_camera_counts[person, camera]) > 0
            ]
            if cameras:
                table[person, cameras] = next_class
                next_class += 1
    return table


def clothes_adversarial_loss(
    logits: torch.Tensor,
    clothes_ids: torch.Tensor,
    positive_mask: torch.Tensor,
    epsilon: float = CLOTHES_ADVERSARIAL_EPSILON,
) -> torch.Tensor:
    """CAL multi-positive cross-entropy over already-scaled clothes logits."""

    if logits.ndim != 2 or positive_mask.shape != logits.shape:
        raise ValueError("logits and positive_mask must share a (batch, num_clothes) shape")
    if clothes_ids.shape != (logits.shape[0],):
        raise ValueError("clothes_ids must have one entry per logit row")
    positive = positive_mask.bool()
    if not bool(positive.gather(1, clothes_ids.unsqueeze(1)).all()):
        raise ValueError("every target clothes class must be marked positive")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = logits.float()
        negative_lse = torch.logsumexp(
            logits.masked_fill(positive, -math.inf), dim=1, keepdim=True
        )
        log_prob = logits - torch.logaddexp(logits, negative_lse)
        positive_float = positive.float()
        target = epsilon * positive_float / positive_float.sum(dim=1, keepdim=True)
        target = target + (1.0 - epsilon) * F.one_hot(
            clothes_ids, logits.shape[1]
        ).float()
        return -(target * log_prob).sum(dim=1).mean()


@dataclass(frozen=True)
class ClothesLossBreakdown:
    classifier: torch.Tensor
    adversarial: torch.Tensor
    accuracy: torch.Tensor


class ClothesClassifier(torch.nn.Module):
    """Cosine clothes discriminator plus the derived clothes-label lookups.

    ``adversarial_weight`` is the effective per-epoch weight of the
    adversarial term; :class:`ClothesAdversarialSchedule` sets it.
    """

    def __init__(self, feature_dim: int, clothes_table: torch.Tensor):
        super().__init__()
        if clothes_table.ndim != 2 or clothes_table.dtype != torch.long:
            raise ValueError("clothes_table must be a rank-2 int64 tensor")
        num_clothes = int(clothes_table.max()) + 1
        if num_clothes < 2:
            raise ValueError("the clothes adversarial loss requires at least two clothes classes")
        weight = torch.randn(num_clothes, feature_dim)
        self.weight = torch.nn.Parameter(l2_normalize(weight))
        positive_mask = torch.zeros(clothes_table.shape[0], num_clothes, dtype=torch.bool)
        for person, row in enumerate(clothes_table):
            positive_mask[person, row[row >= 0]] = True
        # Derived from the training split, which the checkpoint fingerprint
        # already pins, so neither lookup is saved.
        self.register_buffer("clothes_table", clothes_table.clone(), persistent=False)
        self.register_buffer("positive_mask", positive_mask, persistent=False)
        self.adversarial_weight = 0.0

    @property
    def num_clothes(self) -> int:
        return self.weight.shape[0]

    def clothes_ids(self, person_ids: torch.Tensor, camera_ids: torch.Tensor) -> torch.Tensor:
        clothes_ids = self.clothes_table[person_ids, camera_ids]
        if bool((clothes_ids < 0).any()):
            raise ValueError("batch contains a (pid, camera) pair absent from training")
        return clothes_ids

    def logits(self, features: torch.Tensor, *, detach_weight: bool) -> torch.Tensor:
        weight = self.weight.detach() if detach_weight else self.weight
        with torch.autocast(device_type=features.device.type, enabled=False):
            cosine = l2_normalize(features.float()) @ l2_normalize(weight.float()).t()
        return CLOTHES_LOGIT_SCALE * cosine

    def loss(
        self,
        features: torch.Tensor,
        person_ids: torch.Tensor,
        camera_ids: torch.Tensor,
    ) -> ClothesLossBreakdown:
        clothes_ids = self.clothes_ids(person_ids, camera_ids)
        classifier_logits = self.logits(features.detach(), detach_weight=False)
        adversarial = clothes_adversarial_loss(
            self.logits(features, detach_weight=True),
            clothes_ids,
            self.positive_mask[person_ids],
        )
        accuracy = (classifier_logits.argmax(dim=1) == clothes_ids).float().mean()
        return ClothesLossBreakdown(
            classifier=F.cross_entropy(classifier_logits, clothes_ids),
            adversarial=adversarial,
            accuracy=accuracy.detach(),
        )


@dataclass(frozen=True)
class ClothesAdversarialSchedule:
    """Enable the adversarial term from Stage-2-local ``start_epoch`` onward.

    The discriminator trains from the first Stage-2 epoch, so it is already
    fitted when the backbone starts to fight it.
    """

    weight: float
    start_epoch: int
    first_epoch: int = 1

    def apply(self, classifier: ClothesClassifier, epoch: int) -> None:
        stage_epoch = epoch - self.first_epoch + 1
        classifier.adversarial_weight = (
            self.weight if stage_epoch >= self.start_epoch else 0.0
        )
