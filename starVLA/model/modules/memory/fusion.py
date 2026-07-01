"""Zero- or small-gated residual cross-attention for policy conditioning."""

import torch
from torch import nn


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
        if bypass:
            return consumer_tokens

        original_dtype = consumer_tokens.dtype
        with torch.autocast(device_type=consumer_tokens.device.type, enabled=False):
            consumer = consumer_tokens.to(dtype=torch.float32)
            memory = memory_tokens.to(dtype=torch.float32)
            query = self.query_projection(self.consumer_norm(consumer))
            key_value = self.memory_projection(self.memory_norm(memory))
            attended, _ = self.attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=False,
            )
            residual = self.output_projection(attended)
            gated_residual = torch.tanh(self.gate) * residual

        return consumer_tokens + gated_residual.to(dtype=original_dtype)
