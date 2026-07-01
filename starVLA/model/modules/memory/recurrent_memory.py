"""Bounded FP32 recurrent working memory for Phase-1 VLA-JEPA training."""

import math
from typing import Optional

import torch
from torch import nn

from .state import MemoryRead, MemoryState


class RecurrentMemory(nn.Module):
    """Eight-slot-style gated recurrent memory over safe Qwen tokens.

    The class is stateless with respect to episodes: every method receives and
    returns a :class:`MemoryState`.  Only learned initialization and update
    weights are stored on the module.
    """

    def __init__(
        self,
        source_dim: int = 2048,
        memory_dim: int = 512,
        num_slots: int = 8,
        num_heads: int = 8,
        update_gate_init: float = 0.1,
        dropout: float = 0.0,
        init_std: float = 0.02,
    ) -> None:
        super().__init__()
        if source_dim <= 0 or memory_dim <= 0 or num_slots <= 0:
            raise ValueError("source_dim, memory_dim, and num_slots must be positive")
        if num_heads <= 0 or memory_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide memory_dim")
        if not 0.0 < update_gate_init < 1.0:
            raise ValueError("update_gate_init must lie strictly between zero and one")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.source_dim = int(source_dim)
        self.memory_dim = int(memory_dim)
        self.num_slots = int(num_slots)
        self.num_heads = int(num_heads)

        self.initial_slots = nn.Parameter(torch.empty(num_slots, memory_dim, dtype=torch.float32))
        self.slot_ids = nn.Parameter(torch.empty(num_slots, memory_dim, dtype=torch.float32))
        self.source_norm = nn.LayerNorm(source_dim)
        self.source_projection = nn.Linear(source_dim, memory_dim)
        self.slot_norm = nn.LayerNorm(memory_dim)
        self.update_attention = nn.MultiheadAttention(
            embed_dim=memory_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.update_gate = nn.Linear(2 * memory_dim, memory_dim)
        self.candidate_projection = nn.Linear(memory_dim, memory_dim)

        self._reset_parameters(update_gate_init=update_gate_init, init_std=init_std)

    def _reset_parameters(self, *, update_gate_init: float, init_std: float) -> None:
        nn.init.normal_(self.initial_slots, mean=0.0, std=init_std)
        nn.init.normal_(self.slot_ids, mean=0.0, std=init_std)
        nn.init.xavier_uniform_(self.source_projection.weight)
        nn.init.zeros_(self.source_projection.bias)
        nn.init.xavier_uniform_(self.candidate_projection.weight)
        nn.init.zeros_(self.candidate_projection.bias)

        # A zero gate weight makes the documented initial update probability
        # exact while still allowing ordinary gradients to train the gate.
        nn.init.zeros_(self.update_gate.weight)
        gate_logit = math.log(update_gate_init / (1.0 - update_gate_init))
        nn.init.constant_(self.update_gate.bias, gate_logit)

    @property
    def device(self) -> torch.device:
        return self.initial_slots.device

    def _validate_mask(
        self,
        mask: torch.Tensor,
        *,
        name: str,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(mask, torch.Tensor) or mask.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [{batch_size}]")
        if mask.dtype != torch.bool:
            raise TypeError(f"{name} must be bool, got {mask.dtype}")
        if mask.device != device:
            raise ValueError(f"{name} must be on device {device}, got {mask.device}")
        return mask

    def _validate_state(self, state: MemoryState) -> None:
        if not isinstance(state, MemoryState):
            raise TypeError("state must be a MemoryState")
        expected = (state.batch_size, self.num_slots, self.memory_dim)
        if state.working.shape != expected:
            raise ValueError(f"working must have shape {expected}, got {tuple(state.working.shape)}")
        if state.working.device != self.device:
            raise ValueError(
                f"memory state is on {state.working.device}, but module parameters are on {self.device}"
            )

    def _validate_source(self, source_tokens: torch.Tensor, state: MemoryState) -> None:
        if not isinstance(source_tokens, torch.Tensor) or source_tokens.ndim != 3:
            raise ValueError("source_tokens must have shape [B, N, source_dim]")
        expected_prefix = (state.batch_size,)
        if source_tokens.shape[:1] != expected_prefix or source_tokens.shape[-1] != self.source_dim:
            raise ValueError(
                "source_tokens must have shape "
                f"[{state.batch_size}, N, {self.source_dim}], got {tuple(source_tokens.shape)}"
            )
        if source_tokens.shape[1] == 0:
            raise ValueError("source_tokens must contain at least one token")
        if source_tokens.device != state.working.device:
            raise ValueError("source_tokens and state must be on the same device")
        if not source_tokens.is_floating_point():
            raise TypeError("source_tokens must use a floating-point dtype")

    def init_state(
        self,
        batch_size: int,
        device: torch.device,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """Create a graph-bearing learned initial state for a new batch."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        device = torch.device(device)
        if device != self.device:
            raise ValueError(f"requested device {device} does not match module device {self.device}")

        if valid_mask is None:
            valid = torch.ones(batch_size, device=device, dtype=torch.bool)
        else:
            valid = self._validate_mask(
                valid_mask, name="valid_mask", batch_size=batch_size, device=device
            )

        learned = self.initial_slots.to(dtype=torch.float32).unsqueeze(0).expand(batch_size, -1, -1)
        working = learned.clone()
        working = torch.where(valid[:, None, None], working, torch.zeros_like(working))
        return MemoryState(
            working=working,
            episodic=None,
            steps=torch.zeros(batch_size, device=device, dtype=torch.int64),
            valid=valid.clone(),
        )

    def reset_state(
        self,
        state: MemoryState,
        reset_mask: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """Reset selected rows out of place.

        Without ``valid_mask``, reset rows become active and unselected rows
        retain their prior validity.  When supplied, ``valid_mask`` defines the
        activity of every returned row, which is useful for padded batches.
        """

        self._validate_state(state)
        reset = self._validate_mask(
            reset_mask,
            name="reset_mask",
            batch_size=state.batch_size,
            device=state.working.device,
        )
        if valid_mask is None:
            valid = torch.where(reset, torch.ones_like(state.valid), state.valid)
        else:
            valid = self._validate_mask(
                valid_mask,
                name="valid_mask",
                batch_size=state.batch_size,
                device=state.working.device,
            )

        initial = self.initial_slots.to(dtype=torch.float32).unsqueeze(0).expand(state.batch_size, -1, -1)
        working = torch.where(reset[:, None, None], initial, state.working)
        working = torch.where(valid[:, None, None], working, torch.zeros_like(working))
        steps = torch.where(reset, torch.zeros_like(state.steps), state.steps)
        steps = torch.where(valid, steps, torch.zeros_like(steps))

        episodic = state.episodic
        if episodic is not None:
            episodic = torch.where(reset[:, None, None], torch.zeros_like(episodic), episodic)
            episodic = torch.where(valid[:, None, None], episodic, torch.zeros_like(episodic))

        return MemoryState(
            working=working.clone(),
            episodic=episodic.clone() if episodic is not None else None,
            steps=steps.clone(),
            valid=valid.clone(),
        )

    def read(
        self,
        source_tokens: torch.Tensor,
        state: MemoryState,
        read_mask: Optional[torch.Tensor] = None,
    ) -> MemoryRead:
        """Return the previous working slots without modifying ``state``."""

        self._validate_state(state)
        self._validate_source(source_tokens, state)
        if read_mask is None:
            active = state.valid
        else:
            active = self._validate_mask(
                read_mask,
                name="read_mask",
                batch_size=state.batch_size,
                device=state.working.device,
            ) & state.valid

        tokens = torch.where(
            active[:, None, None], state.working, torch.zeros_like(state.working)
        )
        diagnostics = {
            "working_norm": state.working.norm(dim=-1).mean(dim=-1),
            "steps": state.steps.to(dtype=torch.float32),
            "active": active.to(dtype=torch.float32),
        }
        return MemoryRead(tokens=tokens, diagnostics=diagnostics)

    def write(
        self,
        source_tokens: torch.Tensor,
        state: MemoryState,
        update_mask: Optional[torch.Tensor] = None,
    ) -> MemoryState:
        """Write current safe source tokens into a new FP32 working state."""

        self._validate_state(state)
        self._validate_source(source_tokens, state)
        if update_mask is None:
            active = state.valid
        else:
            active = self._validate_mask(
                update_mask,
                name="update_mask",
                batch_size=state.batch_size,
                device=state.working.device,
            ) & state.valid

        # Disable an enclosing mixed-precision context for recurrent math.
        with torch.autocast(device_type=source_tokens.device.type, enabled=False):
            source = source_tokens.to(dtype=torch.float32)
            source = self.source_projection(self.source_norm(source))

            previous = state.working
            slot_identity = self.slot_ids.to(dtype=torch.float32).unsqueeze(0)
            query = self.slot_norm(previous + slot_identity)
            context, _ = self.update_attention(
                query=query,
                key=source,
                value=source,
                need_weights=False,
            )

            gate_input = torch.cat((self.slot_norm(previous), context), dim=-1)
            gate = torch.sigmoid(self.update_gate(gate_input))
            candidate = torch.tanh(self.candidate_projection(context))
            proposed = (1.0 - gate) * previous + gate * candidate

            working = torch.where(active[:, None, None], proposed, previous)
            steps = state.steps + active.to(dtype=torch.int64)

        return MemoryState(
            working=working,
            episodic=state.episodic,
            steps=steps,
            valid=state.valid,
        )
