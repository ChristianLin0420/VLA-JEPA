"""Runtime state types for VLA-JEPA memory.

The objects in this module deliberately contain activations only.  Learned
initial states and update parameters belong to :class:`RecurrentMemory`; an
episode's state must never be registered as a model parameter or buffer.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch


@dataclass(frozen=True)
class MemoryState:
    """Explicit, batch-local memory carried between policy decisions.

    Attributes:
        working: Recurrent working slots with shape ``[B, S, D]`` in FP32.
        episodic: Optional associative state ``[B, Dv, Dk]`` in FP32.  It is
            unused by the Phase-1 implementation but is part of the stable
            public state contract.
        steps: Number of completed writes for each row, shape ``[B]`` and
            dtype ``torch.int64``.
        valid: Whether each row represents an active, non-padding episode,
            shape ``[B]`` and dtype ``torch.bool``.
        keys: Optional per-slot content keys ``[B, S, Dk]`` in FP32 (schema 2).
            ``None`` under schema 1.
    """

    working: torch.Tensor
    episodic: Optional[torch.Tensor]
    steps: torch.Tensor
    valid: torch.Tensor
    keys: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if not isinstance(self.working, torch.Tensor) or self.working.ndim != 3:
            raise ValueError("working must be a tensor with shape [B, S, D]")
        if self.working.dtype != torch.float32:
            raise TypeError(f"working must be float32, got {self.working.dtype}")

        batch_size = self.working.shape[0]
        if not isinstance(self.steps, torch.Tensor) or self.steps.shape != (batch_size,):
            raise ValueError(f"steps must have shape [{batch_size}]")
        if self.steps.dtype != torch.int64:
            raise TypeError(f"steps must be int64, got {self.steps.dtype}")

        if not isinstance(self.valid, torch.Tensor) or self.valid.shape != (batch_size,):
            raise ValueError(f"valid must have shape [{batch_size}]")
        if self.valid.dtype != torch.bool:
            raise TypeError(f"valid must be bool, got {self.valid.dtype}")

        if self.episodic is not None:
            if not isinstance(self.episodic, torch.Tensor) or self.episodic.ndim != 3:
                raise ValueError("episodic must be None or a tensor with shape [B, Dv, Dk]")
            if self.episodic.shape[0] != batch_size:
                raise ValueError("episodic and working must have the same batch size")
            if self.episodic.dtype != torch.float32:
                raise TypeError(f"episodic must be float32, got {self.episodic.dtype}")

        if self.keys is not None:
            if not isinstance(self.keys, torch.Tensor) or self.keys.ndim != 3:
                raise ValueError("keys must be None or a tensor with shape [B, S, Dk]")
            if self.keys.shape[0] != batch_size:
                raise ValueError("keys and working must have the same batch size")
            if self.keys.dtype != torch.float32:
                raise TypeError(f"keys must be float32, got {self.keys.dtype}")

        tensors = [self.steps, self.valid]
        if self.episodic is not None:
            tensors.append(self.episodic)
        if self.keys is not None:
            tensors.append(self.keys)
        if any(t.device != self.working.device for t in tensors):
            raise ValueError("all MemoryState tensors must be on the same device")

    @property
    def batch_size(self) -> int:
        return int(self.working.shape[0])

    def detach(self) -> "MemoryState":
        """Detach all floating activation state from its current graph."""

        return MemoryState(
            working=self.working.detach(),
            episodic=self.episodic.detach() if self.episodic is not None else None,
            steps=self.steps.detach(),
            valid=self.valid.detach(),
            keys=self.keys.detach() if self.keys is not None else None,
        )

    def to(self, *, device: torch.device) -> "MemoryState":
        """Move state to ``device`` without changing any tensor dtype."""

        return MemoryState(
            working=self.working.to(device=device),
            episodic=self.episodic.to(device=device) if self.episodic is not None else None,
            steps=self.steps.to(device=device),
            valid=self.valid.to(device=device),
            keys=self.keys.to(device=device) if self.keys is not None else None,
        )


@dataclass(frozen=True)
class MemoryRead:
    """Read bank and detached-or-live diagnostic tensors for one decision."""

    tokens: torch.Tensor
    diagnostics: Dict[str, torch.Tensor]

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor) or self.tokens.ndim != 3:
            raise ValueError("tokens must be a tensor with shape [B, S, D]")
        if self.tokens.dtype != torch.float32:
            raise TypeError(f"memory read tokens must be float32, got {self.tokens.dtype}")
        if not isinstance(self.diagnostics, dict):
            raise TypeError("diagnostics must be a dictionary")
