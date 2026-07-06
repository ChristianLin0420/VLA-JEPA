"""Zero- or small-gated residual cross-attention for policy conditioning."""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .state import MemoryState


class ResidualMemoryFusion(nn.Module):
    """Inject a bounded memory read into consumer tokens through a scalar gate."""

    def __init__(
        self,
        consumer_dim: int = 2048,
        memory_dim: int = 512,
        bottleneck_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if consumer_dim <= 0 or memory_dim <= 0 or bottleneck_dim <= 0:
            raise ValueError("consumer_dim, memory_dim, and bottleneck_dim must be positive")
        if num_heads <= 0 or bottleneck_dim % num_heads != 0:
            raise ValueError("num_heads must be positive and divide bottleneck_dim")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1)")

        self.consumer_dim = int(consumer_dim)
        self.memory_dim = int(memory_dim)
        self.bottleneck_dim = int(bottleneck_dim)

        self.consumer_norm = nn.LayerNorm(consumer_dim)
        self.memory_norm = nn.LayerNorm(memory_dim)
        self.query_projection = nn.Linear(consumer_dim, bottleneck_dim)
        self.memory_projection = nn.Linear(memory_dim, bottleneck_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=bottleneck_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(bottleneck_dim, consumer_dim)
        self.gate = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

        # Runtime knobs and diagnostics (plain attributes, never checkpointed):
        # `residual_scale` rescales the gated residual (λ dose-response) and
        # `capture_diagnostics` gates ALL diagnostic work — injection_ratio
        # (a GPU->host sync) and the read-attention map.  When it is False the
        # default training path performs no diagnostic arithmetic and
        # `last_fusion_diagnostics` is None.  Caveat: need_weights=True takes
        # the MHA off the SDPA fastpath, so capture-on vs capture-off outputs
        # can differ at ulp level; hold the capture context fixed across any
        # paired comparison.
        self.residual_scale = 1.0
        self.capture_diagnostics = False
        self.last_fusion_diagnostics = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for projection in (
            self.query_projection,
            self.memory_projection,
            self.output_projection,
        ):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        consumer_tokens: torch.Tensor,
        memory_tokens: torch.Tensor,
        *,
        bypass: bool = False,
    ) -> torch.Tensor:
        if not isinstance(consumer_tokens, torch.Tensor) or consumer_tokens.ndim != 3:
            raise ValueError("consumer_tokens must have shape [B, N, consumer_dim]")
        if not isinstance(memory_tokens, torch.Tensor) or memory_tokens.ndim != 3:
            raise ValueError("memory_tokens must have shape [B, S, memory_dim]")
        if consumer_tokens.shape[0] != memory_tokens.shape[0]:
            raise ValueError("consumer_tokens and memory_tokens must have the same batch size")
        if consumer_tokens.shape[-1] != self.consumer_dim:
            raise ValueError(
                f"expected consumer dimension {self.consumer_dim}, got {consumer_tokens.shape[-1]}"
            )
        if memory_tokens.shape[-1] != self.memory_dim:
            raise ValueError(
                f"expected memory dimension {self.memory_dim}, got {memory_tokens.shape[-1]}"
            )
        if memory_tokens.shape[1] == 0:
            raise ValueError("memory_tokens must contain at least one slot")
        if consumer_tokens.device != memory_tokens.device:
            raise ValueError("consumer_tokens and memory_tokens must be on the same device")
        if not consumer_tokens.is_floating_point() or not memory_tokens.is_floating_point():
            raise TypeError("consumer_tokens and memory_tokens must be floating point")
        need_weights = bool(self.capture_diagnostics)
        if bypass:
            self.last_fusion_diagnostics = {"injection_ratio": 0.0} if need_weights else None
            return consumer_tokens

        original_dtype = consumer_tokens.dtype
        with torch.autocast(device_type=consumer_tokens.device.type, enabled=False):
            consumer = consumer_tokens.to(dtype=torch.float32)
            memory = memory_tokens.to(dtype=torch.float32)
            query = self.query_projection(self.consumer_norm(consumer))
            key_value = self.memory_projection(self.memory_norm(memory))
            attended, read_attention = self.attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=need_weights,
            )
            residual = self.output_projection(attended)
            gated_residual = torch.tanh(self.gate) * self.residual_scale * residual

        if need_weights:
            self.last_fusion_diagnostics = {
                "injection_ratio": float(gated_residual.norm() / consumer.norm()),
                "read_attention": read_attention.detach(),
            }
        else:
            self.last_fusion_diagnostics = None
        return consumer_tokens + gated_residual.to(dtype=original_dtype)


class SparseKeyMemoryFusion(nn.Module):
    """Content-addressed top-2 memory read with a whitened residual and one time-tap token.

    Consumes a schema-2 :class:`MemoryState` (content ``keys`` present) and
    returns ``[B, N+1, C]``: the consumer tokens with a gated, whitened content
    residual added, plus one explicit time token derived from ``state.steps``.
    """

    def __init__(
        self,
        consumer_dim: int = 2048,
        memory_dim: int = 512,
        key_dim: int = 128,
        num_slots: int = 8,
        content_gate_init: float = 0.0,
        content_gate_fixed: bool = False,
    ) -> None:
        super().__init__()
        if consumer_dim <= 0 or memory_dim <= 0 or key_dim <= 0:
            raise ValueError("consumer_dim, memory_dim, and key_dim must be positive")
        if num_slots < 2:
            raise ValueError("num_slots must be at least 2 for a top-2 read")

        self.consumer_dim = int(consumer_dim)
        self.memory_dim = int(memory_dim)
        self.key_dim = int(key_dim)
        self.num_slots = int(num_slots)

        self.consumer_norm = nn.LayerNorm(consumer_dim)
        self.qk_proj = nn.Linear(consumer_dim, key_dim)
        self.key_norm = nn.LayerNorm(key_dim)
        self.temperature = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.value_proj = nn.Linear(memory_dim, memory_dim)
        self.out_proj = nn.Linear(memory_dim, consumer_dim)
        self.gate_input_proj = nn.Linear(consumer_dim, 64)
        self.gate_mlp = nn.Sequential(nn.Linear(64 + 3, 64), nn.GELU(), nn.Linear(64, 1))
        self.time_mlp = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, consumer_dim))
        # gamma_c is initialized to content_gate_init directly: tanh(x) ~= x for
        # the small openings used here, so the effective gate starts at ~ the value.
        # content_gate_fixed makes the valve unclosable: g_c == 1 and gamma_c is a
        # non-trainable buffer, so no gradient path can suppress the injection.
        self.content_gate_fixed = bool(content_gate_fixed)
        if self.content_gate_fixed:
            self.register_buffer(
                "gamma_c", torch.tensor(float(content_gate_init), dtype=torch.float32)
            )
        else:
            self.gamma_c = nn.Parameter(
                torch.tensor(float(content_gate_init), dtype=torch.float32)
            )

        inv_freq = torch.exp(-math.log(10_000.0) * torch.arange(32, dtype=torch.float32) / 32.0)
        self.register_buffer("time_inv_freq", inv_freq, persistent=False)

        # Runtime attributes, same conventions as ResidualMemoryFusion.
        # `last_residual` holds the live pre-gate content residual r for the
        # NCE anchor; it is set on every non-bypass forward and None otherwise.
        self.residual_scale = 1.0
        self.capture_diagnostics = False
        self.last_fusion_diagnostics = None
        self.last_residual = None

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        projections = [self.qk_proj, self.value_proj, self.out_proj, self.gate_input_proj]
        projections += [m for block in (self.gate_mlp, self.time_mlp) for m in block if isinstance(m, nn.Linear)]
        for projection in projections:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def _sparse_attention(self, consumer: torch.Tensor, keys: torch.Tensor):
        """Top-2 softmax attention over normalized slot keys: ([B,N,S], [B,N,2])."""

        query = self.qk_proj(self.consumer_norm(consumer))
        scores = torch.matmul(query, self.key_norm(keys).transpose(1, 2)) / math.sqrt(self.key_dim)
        top_scores, top_indices = scores.topk(2, dim=-1)
        weights = torch.softmax(top_scores / self.temperature.clamp_min(1.0e-4), dim=-1)
        attention = torch.zeros_like(scores).scatter(-1, top_indices, weights)
        return attention, top_scores

    def _time_tap(self, steps: torch.Tensor) -> torch.Tensor:
        angles = torch.log1p(steps.to(dtype=torch.float32))[:, None] * self.time_inv_freq[None, :]
        return self.time_mlp(torch.cat((angles.sin(), angles.cos()), dim=-1)).unsqueeze(1)

    def _validate(self, consumer_tokens: torch.Tensor, state: MemoryState) -> None:
        if not isinstance(consumer_tokens, torch.Tensor) or consumer_tokens.ndim != 3:
            raise ValueError("consumer_tokens must have shape [B, N, consumer_dim]")
        if consumer_tokens.shape[-1] != self.consumer_dim:
            raise ValueError(
                f"expected consumer dimension {self.consumer_dim}, got {consumer_tokens.shape[-1]}"
            )
        if consumer_tokens.shape[1] == 0:
            raise ValueError("consumer_tokens must contain at least one token")
        if not consumer_tokens.is_floating_point():
            raise TypeError("consumer_tokens must be floating point")
        if not isinstance(state, MemoryState):
            raise TypeError("state must be a MemoryState")
        batch_size = consumer_tokens.shape[0]
        if state.working.shape != (batch_size, self.num_slots, self.memory_dim):
            raise ValueError(
                f"working must have shape [{batch_size}, {self.num_slots}, {self.memory_dim}], "
                f"got {tuple(state.working.shape)}"
            )
        if state.keys is None or state.keys.shape != (batch_size, self.num_slots, self.key_dim):
            raise ValueError(
                f"state.keys must have shape [{batch_size}, {self.num_slots}, {self.key_dim}]"
            )
        if consumer_tokens.device != state.working.device:
            raise ValueError("consumer_tokens and state must be on the same device")

    def forward(
        self,
        consumer_tokens: torch.Tensor,
        state: MemoryState,
        *,
        bypass: bool = False,
    ) -> torch.Tensor:
        self._validate(consumer_tokens, state)
        need_weights = bool(self.capture_diagnostics)
        if bypass:
            self.last_residual = None
            self.last_fusion_diagnostics = (
                {"injection_ratio": 0.0, "match_margin": 0.0, "tap_norm": 0.0}
                if need_weights
                else None
            )
            zero_tap = consumer_tokens.new_zeros(consumer_tokens.shape[0], 1, self.consumer_dim)
            return torch.cat((consumer_tokens, zero_tap), dim=1)

        original_dtype = consumer_tokens.dtype
        with torch.autocast(device_type=consumer_tokens.device.type, enabled=False):
            consumer = consumer_tokens.to(dtype=torch.float32)
            attention, top_scores = self._sparse_attention(consumer, state.keys)
            residual = self.out_proj(torch.matmul(attention, self.value_proj(state.working)))
            whitened = F.layer_norm(residual, (self.consumer_dim,))

            max_score = top_scores[..., 0].amax(dim=1, keepdim=True)
            margin = (top_scores[..., 0] - top_scores[..., 1]).mean(dim=1, keepdim=True)
            entropy = -(attention * attention.clamp_min(1.0e-12).log()).sum(dim=-1).mean(dim=1, keepdim=True)
            gate_features = torch.cat(
                (self.gate_input_proj(whitened.mean(dim=1)), max_score, margin, entropy), dim=-1
            )
            if self.content_gate_fixed:
                content_gate = whitened.new_ones(whitened.shape[0], 1, 1)
            else:
                content_gate = torch.sigmoid(self.gate_mlp(gate_features)).unsqueeze(-1)
            content = torch.tanh(self.gamma_c) * self.residual_scale * (content_gate * whitened)
            tap = self._time_tap(state.steps)

        self.last_residual = residual
        if need_weights:
            self.last_fusion_diagnostics = {
                "injection_ratio": float(content.norm() / consumer.norm()),
                "match_margin": float(margin.mean()),
                "tap_norm": float(tap.norm(dim=-1).mean()),
            }
        else:
            self.last_fusion_diagnostics = None
        return torch.cat(
            (consumer_tokens + content.to(dtype=original_dtype), tap.to(dtype=original_dtype)),
            dim=1,
        )
