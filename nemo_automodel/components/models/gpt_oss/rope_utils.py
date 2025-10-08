import functools
import math

import torch


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1, x2 = torch.chunk(x, 2, dim=-1)
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.dtype = dtype
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    @functools.cache
    @torch.no_grad()
    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device) / self.head_dim)
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.head_dim / 2
            # NTK by parts
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_tokens = query.shape[1]
        cos, sin = self._compute_cos_sin(num_tokens)

        query = apply_rotary_emb(query, cos, sin)

        key = apply_rotary_emb(key, cos, sin)
        return query, key


@torch.no_grad()
def position_ids_to_freqs_cis(
    rotary_emb: RotaryEmbedding, position_ids: torch.Tensor, qkv_format: str = "bshd"
) -> torch.Tensor:
    if qkv_format == "thd":
        position_ids = position_ids.unsqueeze(0)

    concentration, inv_freq = rotary_emb._compute_concentration_and_inv_freq()
    inv_freq = inv_freq.to(device=position_ids.device, dtype=torch.float32)
    # angles: (B, T, D/2)
    angles = torch.einsum("bt,d->btd", position_ids.to(dtype=torch.float32), inv_freq)
    cos = torch.cos(angles) * concentration
    sin = torch.sin(angles) * concentration
    freqs_cis = torch.cat([cos, sin], dim=-1)
    if qkv_format == "thd":
        freqs_cis = freqs_cis.squeeze(0)
    return freqs_cis
