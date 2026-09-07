"""Feature normalization and fusion helpers."""

from __future__ import annotations

import torch
from torch.nn import functional as F

DEFAULT_EPS = 1e-12


def l2_normalize(features: torch.Tensor, eps: float = DEFAULT_EPS) -> torch.Tensor:
    if features.ndim < 2:
        raise ValueError("features must have at least two dimensions")
    return F.normalize(features, p=2.0, dim=-1, eps=eps)


def fuse_features(
    visual: torch.Tensor, text: torch.Tensor, beta: float
) -> torch.Tensor:
    """Fuse unit directions so beta is independent of either feature norm."""
    if visual.shape != text.shape:
        raise ValueError(
            f"visual and text shapes must match: {visual.shape} != {text.shape}"
        )
    if beta == 0.0:
        # Return the pure image-side feature bitwise: double normalization is
        # not exactly idempotent in floating point.
        return l2_normalize(visual)
    return l2_normalize(l2_normalize(visual) + beta * l2_normalize(text))
