"""Exponential moving average of trainable weights for evaluation.

``ModelEma`` keeps a shadow copy of the tensors that were trainable when it
was initialized, plus every floating-point buffer (BNNeck running statistics),
and blends them after each successful optimizer update:

```text
decay_t = min(decay, t / (t + 1))      # t = updates applied so far
shadow <- decay_t * shadow + (1 - decay_t) * live
```

The warmup makes the shadow the plain mean of the first ``1 / (1 - decay)``
updated weights, so the initialization snapshot (and default BNNeck statistics)
is dropped at the first update instead of lingering with a ``decay^t`` share.

Training always runs on the live weights. ``applied`` swaps the shadow copy in
for the duration of a ``with`` block (validation) and restores the live
weights afterwards, so optimizer state, checkpoints of ``model_state``, and
the next training step never see the averaged weights.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch


class ModelEma:
    def __init__(self, module: torch.nn.Module, decay: float) -> None:
        if not math.isfinite(decay) or not 0.0 < decay < 1.0:
            raise ValueError("model EMA decay must satisfy 0 < decay < 1")
        self.module = module
        self.decay = decay
        self.num_updates = 0
        self._names: tuple[str, ...] = ()
        self._live: list[torch.Tensor] = []
        self._shadow: list[torch.Tensor] = []

    @property
    def initialized(self) -> bool:
        return bool(self._names)

    @property
    def tracked_names(self) -> tuple[str, ...]:
        return self._names

    def initialize(self) -> None:
        """Snapshot the currently trainable parameters and floating buffers."""

        names = [
            name
            for name, parameter in self.module.named_parameters()
            if parameter.requires_grad
        ]
        names += [
            name
            for name, buffer in self.module.named_buffers()
            if buffer.is_floating_point()
        ]
        if not names:
            raise ValueError(
                "model EMA found no trainable parameters or floating-point "
                "buffers to track"
            )
        self._bind(names)
        with torch.no_grad():
            self._shadow = [tensor.detach().clone() for tensor in self._live]
        self.num_updates = 0

    @torch.no_grad()
    def update(self) -> None:
        self._require_initialized()
        live = [tensor.detach() for tensor in self._live]
        decay = min(self.decay, self.num_updates / (self.num_updates + 1))
        torch._foreach_lerp_(self._shadow, live, 1.0 - decay)
        self.num_updates += 1

    @contextmanager
    def applied(self) -> Iterator[None]:
        """Evaluate with the averaged weights, restoring the live ones on exit."""

        self._require_initialized()
        with torch.no_grad():
            backup = [tensor.detach().clone() for tensor in self._live]
            _copy_into(self._live, self._shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                _copy_into(self._live, backup)

    def state_dict(self) -> dict[str, Any]:
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "shadow": dict(zip(self._names, self._shadow, strict=True)),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        decay = state.get("decay")
        if decay != self.decay:
            raise ValueError(
                f"checkpoint model EMA decay does not match this run "
                f"({decay!r} != {self.decay!r})"
            )
        shadow = state.get("shadow")
        if not isinstance(shadow, Mapping):
            raise TypeError("checkpoint model EMA shadow must be a mapping")
        if not shadow:
            self._names, self._live, self._shadow = (), [], []
            self.num_updates = 0
            return
        self._bind(list(shadow))
        restored: list[torch.Tensor] = []
        for name, live in zip(self._names, self._live, strict=True):
            value = shadow[name]
            if not isinstance(value, torch.Tensor) or value.shape != live.shape:
                raise ValueError(
                    f"checkpoint model EMA tensor {name!r} does not match the "
                    f"model shape {tuple(live.shape)}"
                )
            restored.append(
                value.detach().to(device=live.device, dtype=live.dtype, copy=True)
            )
        self._shadow = restored
        self.num_updates = int(state.get("num_updates", 0))

    def _bind(self, names: Sequence[str]) -> None:
        tensors = dict(self.module.named_parameters())
        tensors.update(self.module.named_buffers())
        missing = [name for name in names if name not in tensors]
        if missing:
            raise ValueError(
                f"model EMA tracks tensors missing from the model: {missing[:5]}"
            )
        self._names = tuple(names)
        self._live = [tensors[name] for name in names]

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise ValueError("model EMA is used before it was initialized")


def _copy_into(targets: list[torch.Tensor], sources: list[torch.Tensor]) -> None:
    for target, source in zip(targets, sources, strict=True):
        target.copy_(source)
