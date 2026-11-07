# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test-owned FP32 Flint mathematics on PyTorch's SDPA math backend, and full-vocabulary distribution metrics."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel


@dataclass(frozen=True)
class ReferenceConfig:
    """A literal prefix of the released schedule, independent of native config."""

    source_num_hidden_layers: int
    num_hidden_layers: int
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    shared_expert_intermediate_size: int
    layer_types: tuple[str, ...]
    no_rope_layer_indices: tuple[int, ...]
    mlp_only_layers: tuple[int, ...]

    @classmethod
    def from_checkpoint(cls, raw: Mapping[str, object], *, num_hidden_layers: int = 4) -> ReferenceConfig:
        """Read explicit source schedules; never regenerate a shortened schedule."""
        source_layers = int(raw["num_hidden_layers"])
        if not 1 <= num_hidden_layers <= source_layers:
            raise ValueError("reference prefix must lie within the source layer extent")
        layer_types = tuple(raw["layer_types"])
        no_rope = tuple(int(x) for x in raw["no_rope_layer_indices"])
        dense = tuple(int(x) for x in raw["mlp_only_layers"])
        if len(layer_types) != source_layers or any(
            x not in ("full_attention", "sliding_attention") for x in layer_types
        ):
            raise ValueError("source must specify every layer's attention type")
        full = tuple(i for i, kind in enumerate(layer_types) if kind == "full_attention")
        if tuple(sorted(no_rope)) != full:
            raise ValueError("source no-RoPE indices must equal its full-attention layers")
        if len(set(dense)) != len(dense) or any(i < 0 or i >= source_layers for i in dense):
            raise ValueError("source dense-layer indices are invalid")
        required = {
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "softmax_type": "learnable",
            "decoder_sparse_step": 1,
            "router_score_function": "sigmoid",
            "norm_topk_prob": True,
            "tie_word_embeddings": False,
            "use_sliding_window": True,
            "output_router_logits": False,
        }
        if any(raw.get(k) != value for k, value in required.items()):
            raise ValueError("source architecture is outside the independent Flint reference contract")
        rope_parameters = raw.get("rope_parameters")
        if isinstance(rope_parameters, dict) and rope_parameters.get("rope_type", "default") != "default":
            raise ValueError("reference supports only the source default NeoX RoPE")
        theta = rope_parameters["rope_theta"] if isinstance(rope_parameters, dict) else raw["rope_theta"]
        config = cls(
            source_num_hidden_layers=source_layers,
            num_hidden_layers=num_hidden_layers,
            **{
                name: int(raw[name])
                for name in (
                    "vocab_size",
                    "hidden_size",
                    "intermediate_size",
                    "num_attention_heads",
                    "num_key_value_heads",
                    "head_dim",
                    "sliding_window",
                    "num_experts",
                    "num_experts_per_tok",
                    "moe_intermediate_size",
                    "shared_expert_intermediate_size",
                )
            },
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(theta),
            layer_types=layer_types[:num_hidden_layers],
            no_rope_layer_indices=tuple(i for i in no_rope if i < num_hidden_layers),
            mlp_only_layers=tuple(i for i in dense if i < num_hidden_layers),
        )
        if (
            min(
                config.vocab_size,
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.num_key_value_heads,
                config.head_dim,
                config.sliding_window,
                config.num_experts,
                config.moe_intermediate_size,
                config.shared_expert_intermediate_size,
            )
            <= 0
            or config.num_attention_heads % config.num_key_value_heads
            or config.head_dim % 2
            or not 1 <= config.num_experts_per_tok <= config.num_experts
            or not math.isfinite(config.rms_norm_eps)
            or config.rms_norm_eps <= 0
            or not math.isfinite(config.rope_theta)
            or config.rope_theta <= 0
        ):
            raise ValueError("invalid reference dimensions or normalization/RoPE settings")
        return config

    def expected_shapes(self) -> dict[str, tuple[int, ...]]:
        """Enumerate the exact selected HF/source [out,in] matrices and vectors."""
        h, d = self.hidden_size, self.head_dim
        shapes = {
            "model.embed_tokens.weight": (self.vocab_size, h),
            "model.norm.weight": (h,),
            "lm_head.weight": (self.vocab_size, h),
        }
        for i in range(self.num_hidden_layers):
            prefix = f"model.layers.{i}."
            shapes.update(
                {
                    prefix + "input_layernorm.weight": (h,),
                    prefix + "post_attention_layernorm.weight": (h,),
                    prefix + "self_attn.q_proj.weight": (self.num_attention_heads * d, h),
                    prefix + "self_attn.k_proj.weight": (self.num_key_value_heads * d, h),
                    prefix + "self_attn.v_proj.weight": (self.num_key_value_heads * d, h),
                    prefix + "self_attn.o_proj.weight": (h, self.num_attention_heads * d),
                    prefix + "self_attn.q_norm.weight": (d,),
                    prefix + "self_attn.k_norm.weight": (d,),
                    prefix + "self_attn.softmax_offset": (self.num_attention_heads,),
                }
            )
            if i in self.mlp_only_layers:
                families = [("mlp.", self.intermediate_size)]
            else:
                shapes[prefix + "mlp.gate.weight"] = (self.num_experts, h)
                families = [(f"mlp.experts.{e}.", self.moe_intermediate_size) for e in range(self.num_experts)]
                families.append(("mlp.shared_expert.", self.shared_expert_intermediate_size))
            for family, width in families:
                shapes[prefix + family + "gate_proj.weight"] = (width, h)
                shapes[prefix + family + "up_proj.weight"] = (width, h)
                shapes[prefix + family + "down_proj.weight"] = (h, width)
        return shapes


def rms_norm(values: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Normalize FP32 [...,width] with shared FP32 [width] scale."""
    return values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight


def neox_rope(values: Tensor, positions: Tensor, theta: float) -> Tensor:
    """Rotate FP32 [B,S,heads,D] using paired half-channels and [B,S] positions."""
    dimension = values.shape[-1]
    inverse = theta ** (-torch.arange(0, dimension, 2, device=values.device, dtype=torch.float32) / dimension)
    angles = positions.float().unsqueeze(-1) * inverse
    cosine, sine = angles.cos().unsqueeze(2), angles.sin().unsqueeze(2)
    first, second = values.chunk(2, -1)
    return torch.cat((first * cosine - second * sine, second * cosine + first * sine), -1)


def sdpa_sink_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    sinks: Tensor,
    *,
    attention_mask: Tensor,
    sliding_window: int | None,
    query_chunk_size: int = 128,
) -> Tensor:
    """Apply causal GQA with an unscaled zero-value sink through SDPA math.

    Args:
        query: FP32 [B,S,Hq,D] prepared queries.
        key: FP32 [B,S,Hkv,D] prepared keys, Hq divisible by Hkv.
        value: Same-shaped FP32 values.
        sinks: FP32 [Hq] learned extra denominator logits.
        attention_mask: Boolean/binary [B,S] valid tokens, independent of IDs.
        sliding_window: Allowed real-key count including self; None is full.
        query_chunk_size: Positive bounded query-row count per SDPA call.
            Every query still attends all permitted keys in the complete sequence.

    Returns:
        FP32 [B,S,Hq,D] attention values. A dummy key/value of zeros contributes
        score zero before the additive mask supplies the unscaled sink logit.
        The sink stays valid when every real key is masked, producing zero.
    """
    if query.ndim != 4 or key.ndim != 4 or key.shape != value.shape:
        raise ValueError("Q/K/V must have [B,S,heads,D] layouts")
    b, s, heads, d = query.shape
    if key.shape[:2] != (b, s) or key.shape[-1] != d or heads % key.shape[2] or sinks.shape != (heads,):
        raise ValueError("invalid GQA or sink shapes")
    if min(b, s, heads, d, key.shape[2], query_chunk_size) <= 0:
        raise ValueError("attention axes and chunk size must be positive")
    if attention_mask.shape != (b, s) or not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must be binary [B,S]")
    if sliding_window is not None and sliding_window <= 0:
        raise ValueError("sliding_window must include at least self")
    for tensor in (query, key, value, sinks):
        if tensor.dtype != torch.float32 or tensor.device != query.device:
            raise TypeError("the independent SDPA baseline requires FP32 tensors on one device")
    valid = attention_mask.to(device=query.device, dtype=torch.bool)
    k = torch.cat((key.transpose(1, 2), key.new_zeros((b, key.shape[2], 1, d))), 2)
    v = torch.cat((value.transpose(1, 2), value.new_zeros((b, value.shape[2], 1, d))), 2)
    key_positions = torch.arange(s, device=query.device)
    outputs = []
    with torch.autocast(device_type=query.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
        for start in range(0, s, query_chunk_size):
            end = min(start + query_chunk_size, s)
            distances = torch.arange(start, end, device=query.device)[:, None] - key_positions[None, :]
            allowed = distances >= 0
            if sliding_window is not None:
                allowed &= distances < sliding_window
            allowed = allowed[None, None] & valid[:, None, None, :] & valid[:, None, start:end, None]
            bias = torch.full((b, heads, end - start, s + 1), -torch.inf, device=query.device, dtype=torch.float32)
            bias[..., :s].masked_fill_(allowed, 0)
            bias[..., s] = sinks[None, :, None]
            outputs.append(
                F.scaled_dot_product_attention(
                    query[:, start:end].transpose(1, 2),
                    k,
                    v,
                    attn_mask=bias,
                    dropout_p=0.0,
                    is_causal=False,
                    enable_gqa=True,
                ).transpose(1, 2)
            )
    return torch.cat(outputs, 1)


@dataclass(frozen=True)
class ReferenceStates:
    """FP32 [B,S,H] embedding, every decoder output and final normalized output."""

    embedding: Tensor
    layers: tuple[Tensor, ...]
    final_norm: Tensor


class FlintSDPAReference:
    """Independent FP32 functional model backed by literal source tensor names."""

    def __init__(self, config: ReferenceConfig, weights: Mapping[str, Tensor], *, query_chunk_size: int = 128) -> None:
        """Validate an exact FP32 source-prefix weight mapping on one CPU/CUDA device."""
        shapes = config.expected_shapes()
        if set(weights) != set(shapes):
            missing, extra = sorted(set(shapes) - set(weights)), sorted(set(weights) - set(shapes))
            raise ValueError(f"reference weights must match exactly; missing={missing}, extra={extra}")
        devices = {tensor.device for tensor in weights.values()}
        if len(devices) != 1 or query_chunk_size <= 0:
            raise ValueError("reference weights must share a device and chunk size must be positive")
        for name, shape in shapes.items():
            tensor = weights[name]
            if tensor.shape != shape or tensor.dtype != torch.float32:
                raise ValueError(f"reference tensor {name} must have FP32 shape {shape}")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"nonfinite reference weight: {name}")
        self.config = config
        self.weights = dict(weights)
        self.query_chunk_size = query_chunk_size

    def _mlp(self, values: Tensor, prefix: str) -> Tensor:
        """Apply unfused FP32 SwiGLU to [...,H] using source [out,in] matrices."""
        w = self.weights
        return F.linear(
            F.silu(F.linear(values, w[prefix + "gate_proj.weight"])) * F.linear(values, w[prefix + "up_proj.weight"]),
            w[prefix + "down_proj.weight"],
        )

    @torch.inference_mode()
    def forward_hidden(
        self, input_ids: Tensor, *, attention_mask: Tensor, position_ids: Tensor | None = None
    ) -> ReferenceStates:
        """Evaluate all selected layers with explicit FP32 unfused arithmetic.

        Args:
            input_ids: int64 [B,S] source vocabulary IDs on the weights' device.
            attention_mask: Binary [B,S] validity; EOS IDs may remain valid.
            position_ids: Optional nonnegative int64 [B,S] RoPE positions.

        Returns:
            Complete FP32 embedding, each layer's post-residual states and
            final RMS-normalized states; no positions or vocabulary rows sampled.
        """
        c, w = self.config, self.weights
        if input_ids.ndim != 2 or input_ids.dtype != torch.int64:
            raise ValueError("input_ids must be int64 [B,S]")
        if input_ids.device != w["model.embed_tokens.weight"].device:
            raise ValueError("input IDs and reference weights must share a device")
        if torch.any(input_ids < 0) or torch.any(input_ids >= c.vocab_size):
            raise ValueError("input ID outside the source vocabulary")
        b, s = input_ids.shape
        if attention_mask.shape != (b, s):
            raise ValueError("attention_mask must have [B,S] layout")
        if position_ids is None:
            position_ids = torch.arange(s, device=input_ids.device).expand(b, -1)
        if position_ids.shape != (b, s) or position_ids.dtype != torch.int64 or torch.any(position_ids < 0):
            raise ValueError("position_ids must be nonnegative int64 [B,S]")
        outputs = []
        with torch.autocast(device_type=input_ids.device.type, enabled=False):
            hidden = F.embedding(input_ids, w["model.embed_tokens.weight"])
            embedding = hidden
            for i in range(c.num_hidden_layers):
                p = f"model.layers.{i}."
                normalized = rms_norm(hidden, w[p + "input_layernorm.weight"], c.rms_norm_eps)
                q = F.linear(normalized, w[p + "self_attn.q_proj.weight"]).reshape(
                    b, s, c.num_attention_heads, c.head_dim
                )
                k = F.linear(normalized, w[p + "self_attn.k_proj.weight"]).reshape(
                    b, s, c.num_key_value_heads, c.head_dim
                )
                v = F.linear(normalized, w[p + "self_attn.v_proj.weight"]).reshape(
                    b, s, c.num_key_value_heads, c.head_dim
                )
                q = rms_norm(q, w[p + "self_attn.q_norm.weight"], c.rms_norm_eps)
                k = rms_norm(k, w[p + "self_attn.k_norm.weight"], c.rms_norm_eps)
                if i not in c.no_rope_layer_indices:
                    q, k = neox_rope(q, position_ids, c.rope_theta), neox_rope(k, position_ids, c.rope_theta)
                attention = sdpa_sink_attention(
                    q,
                    k,
                    v,
                    w[p + "self_attn.softmax_offset"],
                    attention_mask=attention_mask,
                    sliding_window=c.sliding_window if c.layer_types[i] == "sliding_attention" else None,
                    query_chunk_size=self.query_chunk_size,
                )
                hidden = hidden + F.linear(attention.flatten(-2), w[p + "self_attn.o_proj.weight"])
                normalized = rms_norm(hidden, w[p + "post_attention_layernorm.weight"], c.rms_norm_eps)
                if i in c.mlp_only_layers:
                    mlp = self._mlp(normalized, p + "mlp.")
                else:
                    flat = normalized.reshape(-1, c.hidden_size)
                    scores = F.linear(flat, w[p + "mlp.gate.weight"]).sigmoid()
                    ids = torch.topk(scores, c.num_experts_per_tok, dim=-1, sorted=False).indices
                    route = scores.gather(-1, ids)
                    route = route / (route.sum(-1, keepdim=True) + 1e-20)
                    contributions = flat.new_zeros((flat.shape[0], c.num_experts_per_tok, c.hidden_size))
                    for expert in range(c.num_experts):
                        tokens, slots = torch.where(ids == expert)
                        if tokens.numel():
                            values = self._mlp(flat.index_select(0, tokens), p + f"mlp.experts.{expert}.")
                            contributions[tokens, slots] = values * route[tokens, slots, None]
                    mlp = contributions.sum(1).reshape(b, s, c.hidden_size)
                    mlp = mlp + self._mlp(normalized, p + "mlp.shared_expert.")
                hidden = hidden + mlp
                outputs.append(hidden)
            final_norm = rms_norm(hidden, w["model.norm.weight"], c.rms_norm_eps)
        return ReferenceStates(embedding, tuple(outputs), final_norm)

    def head(self, normalized: Tensor) -> Tensor:
        """Project FP32 [...,H] states to all vocabulary logits [...,V]."""
        if normalized.dtype != torch.float32:
            raise TypeError("reference head input must be FP32")
        with torch.autocast(device_type=normalized.device.type, enabled=False):
            return F.linear(normalized, self.weights["lm_head.weight"])


MEAN_KL_LIMIT = 1e-3


def _cosine(dot: float, reference_square: float, candidate_square: float) -> float:
    """Return global cosine, defining two all-zero vectors as identical."""
    if reference_square == 0 or candidate_square == 0:
        return float(reference_square == candidate_square)
    return dot / math.sqrt(reference_square * candidate_square)


class LogitAccumulator:
    """Accumulate all-vocabulary metrics without retaining vocabulary-sized rows."""

    def __init__(self, vocab_size: int) -> None:
        """Initialize the fixed mean-KL gate and empty per-position diagnostics."""
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least two")
        self.vocab_size = vocab_size
        self._positions: list[tuple[int, int]] = []
        self._kl: list[float] = []
        self._jsd: list[float] = []
        self._absolute_sum = 0.0
        self._max_absolute = 0.0
        self._dot = 0.0
        self._reference_square = 0.0
        self._candidate_square = 0.0
        self._top1_equal = 0

    @torch.no_grad()
    def update(
        self,
        reference: Tensor,
        candidate: Tensor,
        *,
        positions: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Compare a bounded chunk, covering every vocabulary entry.

        Args:
            reference: Reference logits [rows,vocabulary], FP32/BF16 on CPU/CUDA.
            candidate: Same-shaped candidate logits on the same device.
            positions: Original (batch,sequence) coordinate for each row.
                Defaults to consecutive positions of batch zero across updates.

        Raises:
            ValueError: Shapes, vocabulary width, coordinates or devices differ.
            FloatingPointError: Any logit or computed distribution is nonfinite.
        """
        if (
            reference.ndim != 2
            or reference.shape != candidate.shape
            or reference.shape[0] == 0
            or reference.shape[1] != self.vocab_size
            or reference.device != candidate.device
        ):
            raise ValueError("expected matching nonempty [rows,vocabulary] logits on one device")
        if not reference.is_floating_point() or not candidate.is_floating_point():
            raise TypeError("logits must be floating point")
        if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
            raise FloatingPointError("nonfinite logits")
        offset = len(self._positions)
        coordinates = list(positions) if positions is not None else [(0, offset + i) for i in range(reference.shape[0])]
        if len(coordinates) != reference.shape[0] or any(b < 0 or s < 0 for b, s in coordinates):
            raise ValueError("positions must contain one nonnegative coordinate per row")
        if len(set(coordinates)) != len(coordinates) or set(coordinates).intersection(self._positions):
            raise ValueError("a position must not be compared twice")

        # Distribution arithmetic follows PR4375: log-softmax values are FP32.
        # Global scalar sums use FP64 solely to keep chunking from losing digits.
        ref, cand = reference.float(), candidate.float()
        logp, logq = ref.log_softmax(-1), cand.log_softmax(-1)
        p, q = logp.exp(), logq.exp()
        kl = (p * (logp - logq)).sum(-1)
        logm = torch.logaddexp(logp, logq) - math.log(2)
        jsd = 0.5 * ((p * (logp - logm)).sum(-1) + (q * (logq - logm)).sum(-1))
        if not torch.isfinite(kl).all() or not torch.isfinite(jsd).all():
            raise FloatingPointError("nonfinite KL/JSD")
        self._positions.extend(coordinates)
        self._kl.extend(kl.cpu().tolist())
        self._jsd.extend(jsd.cpu().tolist())
        difference = ref.double() - cand.double()
        self._absolute_sum += difference.abs().sum().item()
        self._max_absolute = max(self._max_absolute, difference.abs().max().item())
        self._dot += (ref.double() * cand.double()).sum().item()
        self._reference_square += ref.double().square().sum().item()
        self._candidate_square += cand.double().square().sum().item()
        self._top1_equal += (ref.argmax(-1) == cand.argmax(-1)).sum().item()

    def result(self) -> dict[str, object]:
        """Return full-coverage metrics and the sole fixed mean-KL acceptance gate."""
        if not self._positions:
            raise ValueError("no positions were compared")
        kl = torch.tensor(self._kl, dtype=torch.float64)
        jsd = torch.tensor(self._jsd, dtype=torch.float64)
        count = len(self._positions)
        maximum_index = int(kl.argmax())
        mean_kl = kl.mean().item()
        return {
            "evaluated_positions": count,
            "vocab_size": self.vocab_size,
            "compared_logit_values": count * self.vocab_size,
            "all_vocabulary_entries_compared": True,
            "log_softmax_dtype": "float32",
            "distribution_units": "nats",
            "kl_direction": "reference || candidate",
            "mean_kl": mean_kl,
            "p95_kl": torch.quantile(kl, 0.95).item(),
            "max_kl": kl[maximum_index].item(),
            "max_kl_position": {
                "batch_index": self._positions[maximum_index][0],
                "position": self._positions[maximum_index][1],
            },
            "min_kl_unclamped": kl.min().item(),
            "mean_jsd": jsd.mean().item(),
            "logit_cosine": _cosine(self._dot, self._reference_square, self._candidate_square),
            "mean_abs_logit_difference": self._absolute_sum / (count * self.vocab_size),
            "max_abs_logit_difference": self._max_absolute,
            "top1_agreement": self._top1_equal / count,
            "top1_agreement_percent": 100 * self._top1_equal / count,
            "acceptance": {
                "metric": "mean_kl",
                "operator": "<",
                "threshold": MEAN_KL_LIMIT,
                "passed": mean_kl < MEAN_KL_LIMIT,
            },
            # Tiny negative KL/JSD from FP32 roundoff are retained rather than
            # silently clamped. These arrays expose the actual per-row evidence.
            "per_position": [
                {"batch_index": b, "position": s, "kl": k, "jsd": j}
                for (b, s), k, j in zip(self._positions, self._kl, self._jsd)
            ],
        }


def compare_logits(
    reference: Tensor,
    candidate: Tensor,
    *,
    attention_mask: Tensor | None = None,
    rows_per_chunk: int = 32,
) -> dict[str, object]:
    """Compare [batch,sequence,vocabulary] tensors over all explicitly valid rows.

    Args:
        reference: Floating reference logits [B,S,V] on CPU/CUDA.
        candidate: Same-shaped candidate logits on the same device.
        attention_mask: Optional Boolean/binary [B,S] validity; token IDs never
            determine validity. Omitting it includes every position.
        rows_per_chunk: Positive number of full-vocabulary rows per update.

    Returns:
        Distribution metrics, explicit evaluated coordinates and mean-KL gate.
    """
    if reference.ndim != 3 or reference.shape != candidate.shape or rows_per_chunk <= 0:
        raise ValueError("expected matching [batch,sequence,vocabulary] and positive chunk size")
    batch, sequence, vocabulary = reference.shape
    valid = torch.ones((batch, sequence), dtype=torch.bool, device=reference.device)
    if attention_mask is not None:
        if attention_mask.shape != (batch, sequence) or not torch.all((attention_mask == 0) | (attention_mask == 1)):
            raise ValueError("attention_mask must be binary [batch,sequence]")
        valid = attention_mask.to(device=reference.device, dtype=torch.bool)
    coordinates = valid.nonzero()
    accumulator = LogitAccumulator(vocabulary)
    for start in range(0, coordinates.shape[0], rows_per_chunk):
        indexes = coordinates[start : start + rows_per_chunk]
        rows = indexes[:, 0] * sequence + indexes[:, 1]
        accumulator.update(
            reference.reshape(-1, vocabulary).index_select(0, rows),
            candidate.reshape(-1, vocabulary).index_select(0, rows),
            positions=[tuple(pair) for pair in indexes.cpu().tolist()],
        )
    return accumulator.result()


def compare_hidden(reference: Tensor, candidate: Tensor) -> dict[str, float | int]:
    """Compare complete [batch,sequence,hidden] layer states in FP64 scalar sums."""
    if reference.shape != candidate.shape or reference.numel() == 0:
        raise ValueError("hidden states must have identical nonempty shapes")
    if not torch.isfinite(reference).all() or not torch.isfinite(candidate).all():
        raise FloatingPointError("nonfinite hidden states")
    ref, cand = reference.double(), candidate.double()
    difference = cand - ref
    square = ref.square().sum().item()
    error_square = difference.square().sum().item()
    return {
        "elements": reference.numel(),
        "mean_abs_difference": difference.abs().mean().item(),
        "max_abs_difference": difference.abs().max().item(),
        "relative_rmse": math.sqrt(error_square) / max(math.sqrt(square), 1e-30),
        "cosine": _cosine((ref * cand).sum().item(), square, cand.square().sum().item()),
    }
