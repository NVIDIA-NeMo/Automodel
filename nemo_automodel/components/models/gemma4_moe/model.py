# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Gemma4 MoE NeMo Automodel support.

Replaces the HF-native Gemma4 MoE (dense matmul over all experts) with NeMo's
GroupedExperts backend, enabling Expert Parallelism (EP) via the standard
MoE parallelizer.
"""

from collections.abc import MutableMapping
from typing import Any, Iterator, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.shared.import_utils import UnavailableError, UnavailableMeta


def _make_missing(name: str):
    return UnavailableMeta(name, (), {"_msg": "transformers.models.gemma4 is not available."})


try:
    from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
    from transformers.models.gemma4 import modeling_gemma4 as _g4
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4Config,
        Gemma4TextConfig,
    )

    Gemma4RMSNorm = _g4.Gemma4RMSNorm
    Gemma4TextModel = _g4.Gemma4TextModel
    Gemma4TextScaledWordEmbedding = _g4.Gemma4TextScaledWordEmbedding
    Gemma4CausalLMOutputWithPast = _g4.Gemma4CausalLMOutputWithPast

    # These classes were renamed in transformers 5.5 (Gemma4X → Gemma4TextX)
    # TODO have only transformers 5.5 version of these classes ?
    Gemma4Attention = getattr(_g4, "Gemma4TextAttention", None) or _g4.Gemma4Attention
    Gemma4DecoderLayer = getattr(_g4, "Gemma4TextDecoderLayer", None) or _g4.Gemma4DecoderLayer
    Gemma4MLP = getattr(_g4, "Gemma4TextMLP", None) or _g4.Gemma4MLP
    Gemma4RotaryEmbedding = getattr(_g4, "Gemma4TextRotaryEmbedding", None) or _g4.Gemma4RotaryEmbedding
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4ForConditionalGeneration as HFGemma4ForConditionalGeneration,
    )
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4Model as HFGemma4Model,
    )

    _GEMMA4_HF_AVAILABLE = True
except (ModuleNotFoundError, ImportError, AttributeError):
    _GEMMA4_HF_AVAILABLE = False
    Gemma4Config = _make_missing("Gemma4Config")
    Gemma4TextConfig = _make_missing("Gemma4TextConfig")
    Gemma4Attention = _make_missing("Gemma4Attention")
    Gemma4DecoderLayer = _make_missing("Gemma4DecoderLayer")
    Gemma4MLP = _make_missing("Gemma4MLP")
    Gemma4RMSNorm = _make_missing("Gemma4RMSNorm")
    Gemma4RotaryEmbedding = _make_missing("Gemma4RotaryEmbedding")
    Gemma4TextModel = _make_missing("Gemma4TextModel")
    Gemma4TextScaledWordEmbedding = _make_missing("Gemma4TextScaledWordEmbedding")
    HFGemma4ForConditionalGeneration = _make_missing("Gemma4ForConditionalGeneration")
    HFGemma4Model = _make_missing("Gemma4Model")
    Gemma4CausalLMOutputWithPast = _make_missing("Gemma4CausalLMOutputWithPast")
    BaseModelOutputWithPast = _make_missing("BaseModelOutputWithPast")
    CausalLMOutputWithPast = _make_missing("CausalLMOutputWithPast")

from nemo_automodel._transformers.model_capabilities import ModelCapabilities
from nemo_automodel.components.models.common import BackendConfig, compute_lm_head_logits
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.layers import MoE, MoEConfig
from nemo_automodel.shared.utils import dtype_from_str as get_dtype

from .cp_attention import attach_gemma4_cp_ring_attention, gemma4_vision_group_ids
from .cp_batch import make_contiguous_shard_cp_batch_and_ctx


class _Gemma4KVShareHolder:
    """Cache-free holder that lets HF gemma4 kv-sharing fire under ``use_cache=False``.

    E2B/E4B share K/V across the trailing ``num_kv_shared_layers`` layers: each shared
    layer reads its source layer's K/V from ``past_key_values.shared_layers`` (see HF
    ``Gemma4Attention.forward``). HF gates that read on ``past_key_values is not None``,
    which is ``None`` whenever ``use_cache=False`` -- and ``use_cache`` is forced off by
    activation checkpointing and the model-owned CP path. The shared layers then fall
    back to their (frozen, unused) K/V projections and produce garbage, inflating the
    loss ~4x.

    Passing this lightweight object as ``past_key_values`` satisfies the gate so HF's
    own kv-sharing logic runs: source layers populate ``shared_layers`` and shared
    layers read it. ``update`` is a pass-through (no per-token accumulation, so no cache
    memory growth), and ``get_seq_length`` returns 0 so the causal mask is built with a
    zero cache offset (correct for a training forward).
    """

    def __init__(self) -> None:
        self.shared_layers: dict = {}

    def get_seq_length(self, *args, **kwargs) -> int:
        return 0

    def get_mask_sizes(self, query_length: int, layer_idx=None) -> tuple[int, int]:
        # (kv_length, kv_offset): no cache -> kv spans the current query, zero offset.
        return query_length, 0

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        return key_states, value_states


def _kv_sharing_active(text_config) -> bool:
    """True if the (dense) text config uses gemma4 kv-sharing (E2B/E4B)."""
    return int(getattr(text_config, "num_kv_shared_layers", 0) or 0) > 0


from .state_dict_adapter import Gemma4MoEStateDictAdapter


class _FSDPSafeSharedKVStates(MutableMapping):
    """A dict-like store for Gemma4 key/value sharing that is safe to pass through FSDP2.

    What Gemma4 needs:
        In Gemma4, the later "kv-shared" attention layers do not compute their
        own keys/values -- they reuse the keys/values produced by an earlier
        layer. HuggingFace implements this by passing ONE shared key/value store
        into every decoder layer's ``forward()``: the earlier layer writes its
        keys/values into the store, and the later layers read them back out. This
        only works if every layer is handed the *same* store object.

    Why a plain ``dict`` breaks under FSDP2:
        With FSDP2 each decoder layer is wrapped as its own unit, and the default
        mixed-precision setting (``cast_forward_inputs=True``) makes FSDP2 look at
        every argument passed to a layer and cast its float tensors to bf16. It
        does this with torch's ``_apply_to_tensors``, which, whenever it sees a
        ``dict`` (or ``list``/``tuple``/``set``/...), builds a brand-new copy of
        it. So if the shared store is a plain ``dict``, each layer receives its
        own private copy: the earlier layer's writes land in a copy that is thrown
        away, and the later layers read from an empty copy -- which raises
        ``KeyError: 'sliding_attention'``.

    Why this class fixes it:
        ``_apply_to_tensors`` only copies the specific types it knows about
        (``dict``, ``OrderedDict``, ``list``, ``tuple``, ``set``, namedtuples,
        dataclasses, ``PackedSequence``); any other object it leaves alone and
        passes straight through. This class behaves like a dict (it implements
        ``MutableMapping``) but is deliberately NOT a ``dict`` subclass, so FSDP2
        hands the SAME instance to every layer and the writes are preserved.

        Note: this dict-based sharing is how transformers 5.8.x works. In 5.5 the
        shared keys/values were stored on the ``Cache`` object instead (which
        FSDP2 also passes through untouched), so this problem did not exist there.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __getitem__(self, key: str) -> tuple[torch.Tensor, torch.Tensor]:
        return self._store[key]

    def __setitem__(self, key: str, value: tuple[torch.Tensor, torch.Tensor]) -> None:
        self._store[key] = value

    def __delitem__(self, key: str) -> None:
        del self._store[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Gemma4-specific router that outputs NeMo-compatible (weights, indices)
# ---------------------------------------------------------------------------
class Gemma4Gate(nn.Module):
    """Gemma4 Router reimplemented to output NeMo Gate format.

    HF Gemma4Router applies: RMSNorm(no_scale) → root_size scaling → learnable
    scale → Linear → softmax → top-k → renormalize which is different from the standard Gate class in layer.py.
    This class reproduces that logic but returns (weights, indices, aux_loss) as expected by GroupedExperts.
    """

    def __init__(self, config: Gemma4TextConfig):
        super().__init__()
        hidden_size = config.hidden_size
        num_experts = config.num_experts
        self.topk = config.top_k_experts
        self.num_experts = num_experts

        # Thread dtype explicitly from config.torch_dtype so the router's own
        # params (proj weight + scale) stay aligned with the rest of the model
        # (fp32 under fp32 master weights) even when construction is not wrapped
        # in local_torch_dtype(). Gemma4RMSNorm(with_scale=False) holds no
        # learnable parameter, so it needs no dtype threading.
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        self.norm = Gemma4RMSNorm(hidden_size, eps=config.rms_norm_eps, with_scale=False)
        self.proj = nn.Linear(hidden_size, num_experts, bias=False, dtype=dtype)
        self.scale = nn.Parameter(torch.ones(hidden_size, dtype=dtype))
        scalar_root_size = hidden_size**-0.5
        self.register_buffer("root_size", torch.tensor(scalar_root_size), persistent=False)

    def forward(self, x, token_mask=None, cp_mesh=None):
        # Router math runs in fp32 regardless of parameter/activation storage
        # dtype. With bf16 storage (e.g. bf16 weights + fp32 master weights via
        # TE FusedAdam), bf16 logits can flip top-k expert selection, so keep
        # the linear, scaling, and softmax in fp32 — mirroring the fp32 routing
        # the standard MoE Gate enforces via gate_precision. Weights are cast
        # back to the input dtype for the downstream expert computation.
        input_dtype = x.dtype

        x_norm = self.norm(x).to(torch.float32)
        x_norm = x_norm * self.root_size.to(torch.float32)
        x_norm = x_norm * self.scale.to(torch.float32)

        expert_scores = F.linear(x_norm, self.proj.weight.to(torch.float32))
        router_probs = F.softmax(expert_scores, dim=-1)

        weights, indices = torch.topk(router_probs, k=self.topk, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=1e-20)
        return weights.to(input_dtype), indices, None

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        pass


# ---------------------------------------------------------------------------
# Gemma4MoE – MoE subclass with Gemma4Gate instead of the default Gate
# ---------------------------------------------------------------------------
class Gemma4MoE(MoE):
    """NeMo MoE that uses Gemma4Gate (with pre-norm routing) instead of
    the standard Gate. Subclasses MoE so that ``isinstance(m, MoE)`` is True,
    which the EP parallelizer relies on."""

    def __init__(self, moe_config: MoEConfig, backend: BackendConfig, text_config: Gemma4TextConfig):
        super().__init__(moe_config, backend)
        # Replace the gate created by MoE.__init__ with Gemma4-specific gate
        self.gate = Gemma4Gate(text_config)

    def forward(self, x, padding_mask=None, cp_mesh=None, *, gate_input=None):
        """Forward with optional separate gate input.

        HF Gemma4 passes unnormalized residual to the router and normalized
        input to the experts.  The decoder layer calls this with
        ``gate_input=x`` (raw residual) so the gate receives unnormalized
        input while experts receive ``pre_feedforward_layernorm_2(x)``.
        """
        if cp_mesh is None:
            cp_mesh = self.cp_mesh

        shape = x.size()
        x = x.view(-1, self.dim)
        if padding_mask is not None:
            token_mask = (~padding_mask).flatten()
        else:
            token_mask = torch.ones(x.size(0), dtype=torch.bool, device=x.device)

        # Use separate gate_input for routing when provided (fixes double-norm)
        g = gate_input.view(-1, self.dim) if gate_input is not None else x
        weights, indices, aux_loss = self.gate(g, token_mask, cp_mesh)

        x_latent = self.fc1_latent_proj(x) if self.fc1_latent_proj is not None else x
        y = self.experts(x_latent, token_mask, weights, indices)
        if self.fc2_latent_proj is not None:
            y = self.fc2_latent_proj(y)
        return y.view(shape)


# ---------------------------------------------------------------------------
# Custom decoder layer
# ---------------------------------------------------------------------------
class Gemma4MoEDecoderLayer(nn.Module):
    """Gemma4 decoder layer with NeMo MoE backend.

    Reuses HF attention and dense MLP, replaces HF Router+MoEBlock with
    NeMo Gemma4MoE (Gemma4Gate + GroupedExperts).
    """

    def __init__(
        self,
        config: Gemma4TextConfig,
        layer_idx: int,
        moe_config: MoEConfig,
        backend: BackendConfig,
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        # Reuse HF modules
        self.self_attn = Gemma4Attention(config=config, layer_idx=layer_idx)
        attach_gemma4_cp_ring_attention(self.self_attn)
        self.mlp = Gemma4MLP(config, layer_idx)

        # Norms
        self.input_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_1 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm_2 = Gemma4RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

        # NeMo MoE
        self.moe = Gemma4MoE(moe_config, backend, config)

        # layer_scalar: per-layer output scaling. We register a buffer on every layer so DCP
        # can always load the weight when present.
        # It is present only for sliding window layers. Regular attentionlayers without a
        # checkpoint value for the layer_scalar keep ones (identity scaling).
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        x: torch.Tensor,
        *,
        position_embeddings: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        padding_mask: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        shared_kv_states: dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # --- Attention ---
        residual = x
        x = self.input_layernorm(x)
        # HF's Gemma4TextAttention requires a `shared_kv_states` dict: kv-sharing
        # layers read their keys/values from it and full-length layers write into
        # it. The backend threads one dict through every layer so sharing works;
        # for configs without kv-sharing (e.g. 26B-A4B) it stays empty.
        attn_kwargs = kwargs
        if (
            getattr(self.config, "_attn_implementation", None) == "flex_attention"
            and getattr(self.self_attn, "head_dim", 0) > 256
            and "kernel_options" not in kwargs
        ):
            attn_kwargs = {
                **kwargs,
                "kernel_options": {
                    "BLOCK_M": 32,
                    "BLOCK_N": 32,
                    "BLOCK_M1": 32,
                    "BLOCK_N1": 32,
                    "BLOCK_M2": 32,
                    "BLOCK_N2": 32,
                    "num_stages": 1,
                    "num_warps": 4,
                },
            }
        if padding_mask is not None and getattr(self.self_attn, "_cp_uses_attention_hook", False):
            attn_kwargs = {**attn_kwargs, "padding_mask": padding_mask}
        x, _ = self.self_attn(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            shared_kv_states=shared_kv_states if shared_kv_states is not None else {},
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            mm_token_type_ids=mm_token_type_ids,
            **attn_kwargs,
        )
        if getattr(self.config, "_attn_implementation", None) == "flex_attention" and padding_mask is not None:
            x = x.masked_fill(padding_mask[..., None], 0)
        x = self.post_attention_layernorm(x)
        x = residual + x

        # --- Dense MLP + MoE in parallel ---
        residual = x

        dense_out = self.pre_feedforward_layernorm(x)
        dense_out = self.mlp(dense_out)
        dense_out = self.post_feedforward_layernorm_1(dense_out)

        moe_input = self.pre_feedforward_layernorm_2(x)
        moe_out = self.moe(moe_input, padding_mask, gate_input=x)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        moe_out = self.post_feedforward_layernorm_2(moe_out)

        x = dense_out + moe_out
        x = self.post_feedforward_layernorm(x)
        x = residual + x

        # Apply per-layer output scaling, multiplied by 1 if it is not present in the checkpoint,
        # otherwise uses the scalar value from the checkpoint.
        x = x * self.layer_scalar

        return x


def _convert_bool_4d_mask_to_additive(attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert a 4D bool allowed-mask to HF additive format (0.0 allowed, -inf masked)."""
    if attention_mask.ndim != 4 or attention_mask.dtype != torch.bool:
        return attention_mask
    additive = torch.zeros(attention_mask.shape, dtype=dtype, device=attention_mask.device)
    return additive.masked_fill(~attention_mask, torch.finfo(dtype).min)


def _derive_padding_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Derive 2D padding mask (True = pad) from 1D, 2D, or 4D attention mask."""
    if attention_mask.ndim == 2:
        return attention_mask == 0
    if attention_mask.ndim == 4:
        diagonal = torch.diagonal(attention_mask[:, 0], dim1=-2, dim2=-1)
        if attention_mask.dtype == torch.bool:
            return diagonal.logical_not()
        return diagonal != 0
    return attention_mask.bool().logical_not()


def _build_packed_gemma4_causal_mask_mapping(
    packed_seq_ids: torch.Tensor,
    mm_token_type_ids: torch.Tensor,
    *,
    dtype: torch.dtype,
    sliding_window: int | None,
    as_additive: bool = False,
    as_block_mask: bool = False,
    flex_block_size: int | tuple[int, int] = 128,
) -> dict[str, torch.Tensor]:
    """Build Gemma4 full/sliding masks for packed VLM sequences.

    ``packed_seq_ids`` contains 1-based document ids and 0 for padding.
    Full-attention layers remain plain packed causal attention. Sliding layers
    also include Gemma4's same-image-token bidirectional edges.
    """
    if packed_seq_ids.ndim != 2:
        raise ValueError(f"_packed_seq_ids must be a 2D [B, S] tensor, got shape={tuple(packed_seq_ids.shape)}")
    if mm_token_type_ids.shape != packed_seq_ids.shape:
        raise ValueError(
            "mm_token_type_ids must have the same shape as _packed_seq_ids, "
            f"got {tuple(mm_token_type_ids.shape)} vs {tuple(packed_seq_ids.shape)}"
        )

    if as_additive and as_block_mask:
        raise ValueError("Only one of as_additive and as_block_mask may be set.")

    batch_size, seq_len = packed_seq_ids.shape
    device = packed_seq_ids.device
    positions = torch.arange(seq_len, device=device)
    q_positions = positions.view(1, seq_len, 1)
    kv_positions = positions.view(1, 1, seq_len)

    vision_group_ids = gemma4_vision_group_ids(mm_token_type_ids)

    if as_block_mask:
        from torch.nn.attention.flex_attention import create_block_mask

        def _full_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            q_pack_id = packed_seq_ids[batch_idx, q_idx]
            kv_pack_id = packed_seq_ids[batch_idx, kv_idx]
            allowed = (q_pack_id == kv_pack_id) & (q_pack_id > 0) & (kv_idx <= q_idx)
            return torch.where(q_pack_id <= 0, kv_idx == 0, allowed)

        def _sliding_mask_mod(batch_idx, head_idx, q_idx, kv_idx):
            q_pack_id = packed_seq_ids[batch_idx, q_idx]
            kv_pack_id = packed_seq_ids[batch_idx, kv_idx]
            same_doc = (q_pack_id == kv_pack_id) & (q_pack_id > 0)
            allowed = same_doc & (kv_idx <= q_idx)
            if sliding_window is not None:
                allowed = allowed & ((q_idx - kv_idx) < sliding_window)
            q_group = vision_group_ids[batch_idx, q_idx]
            kv_group = vision_group_ids[batch_idx, kv_idx]
            same_vision_group = (q_group == kv_group) & (q_group >= 0)
            allowed = (allowed | same_vision_group) & same_doc
            return torch.where(q_pack_id <= 0, kv_idx == 0, allowed)

        return {
            "full_attention": create_block_mask(
                _full_mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=flex_block_size,
            ),
            "sliding_attention": create_block_mask(
                _sliding_mask_mod,
                B=batch_size,
                H=None,
                Q_LEN=seq_len,
                KV_LEN=seq_len,
                device=device,
                BLOCK_SIZE=flex_block_size,
            ),
        }

    valid_q = packed_seq_ids[:, :, None] > 0
    valid_kv = packed_seq_ids[:, None, :] > 0
    same_doc = (packed_seq_ids[:, :, None] == packed_seq_ids[:, None, :]) & valid_q & valid_kv
    causal = kv_positions <= q_positions

    full_mask = same_doc & causal
    sliding_mask = full_mask
    if sliding_window is not None:
        sliding_mask = sliding_mask & ((q_positions - kv_positions) < sliding_window)

    same_vision_group = (vision_group_ids[:, :, None] == vision_group_ids[:, None, :]) & (
        vision_group_ids[:, :, None] >= 0
    )
    sliding_mask = (sliding_mask | same_vision_group) & same_doc

    full_mask = full_mask.view(batch_size, 1, seq_len, seq_len)
    sliding_mask = sliding_mask.view(batch_size, 1, seq_len, seq_len)

    if as_additive:
        min_dtype = torch.finfo(dtype).min
        full_mask = torch.where(full_mask, torch.zeros((), dtype=dtype, device=device), min_dtype)
        sliding_mask = torch.where(sliding_mask, torch.zeros((), dtype=dtype, device=device), min_dtype)

    return {
        "full_attention": full_mask,
        "sliding_attention": sliding_mask,
    }


# ---------------------------------------------------------------------------
# Text model backend
# ---------------------------------------------------------------------------
class Gemma4MoETextModelBackend(nn.Module):
    """Gemma4 text decoder rebuilt with NeMo MoE blocks."""

    def __init__(
        self,
        config: Gemma4TextConfig,
        backend: BackendConfig,
        *,
        moe_config: MoEConfig | None = None,
        moe_overrides: dict | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.config = config
        if moe_config is not None and moe_overrides is not None:
            raise ValueError("Cannot pass both moe_config and moe_overrides; use one or the other.")

        self.padding_idx = getattr(config, "pad_token_id", None)
        self.vocab_size = config.vocab_size

        # Resolve model dtype once from config.torch_dtype and thread it into
        # MoEConfig so the MoE expert params stay aligned with the rest of the
        # model (fp32 under fp32 master weights). HF-native submodules
        # (attention, Gemma4MLP, Gemma4RMSNorm, Gemma4TextScaledWordEmbedding)
        # inherit their dtype from torch.get_default_dtype() via the
        # local_torch_dtype() context established by _init_model().
        model_dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        moe_defaults = dict(
            dim=config.hidden_size,
            inter_dim=config.intermediate_size,
            moe_inter_dim=getattr(config, "moe_intermediate_size", None)
            or getattr(config, "expert_intermediate_size", None),
            n_routed_experts=config.num_experts,
            n_shared_experts=0,
            n_activated_experts=config.top_k_experts,
            n_expert_groups=0,
            n_limited_groups=0,
            train_gate=True,
            gate_bias_update_factor=0.0,
            score_func="softmax",
            route_scale=1.0,
            aux_loss_coeff=0.0,
            norm_topk_prob=True,
            expert_activation="geglu",
            softmax_before_topk=False,
            dtype=model_dtype,
        )
        if moe_overrides:
            moe_defaults.update(moe_overrides)
        self.moe_config = moe_config or MoEConfig(**moe_defaults)

        self.embed_tokens = Gemma4TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            embed_scale=config.hidden_size**0.5,
        )

        self.layers = nn.ModuleDict(
            {
                str(layer_id): Gemma4MoEDecoderLayer(config, layer_id, self.moe_config, backend)
                for layer_id in range(config.num_hidden_layers)
            }
        )

        self.norm = Gemma4RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Gemma4RotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        past_key_values=None,
        use_cache: bool | None = None,
        cp_enabled: bool = False,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        if past_key_values is not None or use_cache:
            raise NotImplementedError("KV cache not supported for the Gemma4 MoE backend.")

        packed_seq_ids = kwargs.get("_packed_seq_ids")
        if not cp_enabled:
            kwargs.pop("_packed_seq_ids", None)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            cache_position = torch.arange(0, inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if padding_mask is None and attention_mask is not None:
            padding_mask = _derive_padding_mask(attention_mask)

        if attention_mask is not None:
            attention_mask = _convert_bool_4d_mask_to_additive(attention_mask, inputs_embeds.dtype)

        hidden_states = inputs_embeds

        # Build causal masks. When use_bidirectional_attention == "vision" (e.g.
        # gemma-4-26B-A4B, gemma-4-31B), HF uses create_causal_mask_mapping to
        # build a vision-aware mask where tokens inside the same vision group
        # attend to each other bidirectionally (not just causally). Missing this
        # logic causes gen_kl_error to be ~10x higher on multimodal inputs.
        use_vision_bidirectional_mask = getattr(self.config, "use_bidirectional_attention", None) == "vision"
        if use_vision_bidirectional_mask and mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros(inputs_embeds.shape[:2], dtype=torch.long, device=inputs_embeds.device)

        if cp_enabled:
            # The CP hook replaces HF's SDPA call with Gemma4's Flex ring
            # attention. Force the HF attention dispatcher through SDPA so
            # configs that default to eager attention still enter the hook.
            self.config._attn_implementation = "sdpa"
            # CP uses Gemma4's model-owned attention hook. HF's local 4D masks
            # have the wrong key range for CP, so the hook rebuilds the
            # local-query/global-key Gemma4 mask from model metadata.
            causal_mask_mapping = {"full_attention": None, "sliding_attention": None}
        elif use_vision_bidirectional_mask and packed_seq_ids is not None:
            causal_mask_mapping = _build_packed_gemma4_causal_mask_mapping(
                packed_seq_ids.to(device=inputs_embeds.device),
                mm_token_type_ids.to(device=inputs_embeds.device),
                dtype=inputs_embeds.dtype,
                sliding_window=getattr(self.config, "sliding_window", None),
                as_block_mask=getattr(self.config, "_attn_implementation", None) == "flex_attention",
                flex_block_size=(32, 32) if getattr(self.config, "head_dim", 0) > 256 else 128,
            )
        elif use_vision_bidirectional_mask:
            from transformers.models.gemma4.modeling_gemma4 import create_causal_mask_mapping

            causal_mask_mapping = create_causal_mask_mapping(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
                mm_token_type_ids=mm_token_type_ids,
                pixel_values=pixel_values,
                is_training=self.training,
            )
        else:
            from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
                "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            }

        position_embeddings = {}
        for layer_type in set(self.config.layer_types):
            position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

        # One shared key/value store passed to every layer: the later kv-shared
        # layers reuse the keys/values that earlier layers write here. It must be
        # an _FSDPSafeSharedKVStates and NOT a plain dict -- under FSDP2 each
        # layer is wrapped separately and FSDP2's input casting makes a fresh copy
        # of any plain dict per layer, so the writes would be lost. See
        # _FSDPSafeSharedKVStates for the full explanation.
        shared_kv_states = _FSDPSafeSharedKVStates()
        for decoder_layer in self.layers.values():
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings[decoder_layer.attention_type],
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                padding_mask=padding_mask,
                mm_token_type_ids=mm_token_type_ids if cp_enabled else None,
                shared_kv_states=shared_kv_states,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value


# ---------------------------------------------------------------------------
# Wrapper that exposes language_model properties for parallelizer
# ---------------------------------------------------------------------------
class Gemma4MoEModel(HFGemma4Model):
    """Thin wrapper that exposes ``language_model`` internals as properties
    expected by the NeMo training loop."""

    @property
    def layers(self):
        return self.language_model.layers

    @property
    def embed_tokens(self):
        return self.language_model.embed_tokens

    @property
    def norm(self):
        return self.language_model.norm


# ---------------------------------------------------------------------------
# Top-level conditional-generation model
# ---------------------------------------------------------------------------
class Gemma4ForConditionalGeneration(HFCheckpointingMixin, HFGemma4ForConditionalGeneration, MoEFSDPSyncMixin):
    supports_gradient_checkpointing = True
    # RoPE inv_freq must stay fp32: initialize_weights casts the model to bf16 and
    # nn.Module.to rounds floating buffers; cast_model_to_dtype restores keep-fp32
    # modules afterwards (see llama/rope_utils.py).
    _keep_in_fp32_modules = ["rotary_emb"]
    """Gemma4 VL conditional generation model with NeMo MoE backend.

    When the checkpoint has ``enable_moe_block=True`` in its text config,
    replaces the HF-native language model with ``Gemma4MoETextModelBackend``
    (NeMo GroupedExperts + Gemma4Gate).  Otherwise falls through to vanilla HF.
    """

    @classmethod
    def get_capabilities(cls, config: "Gemma4Config") -> ModelCapabilities:
        """Return the capabilities for a specific config (no model instance needed).

        Dispatches in two layers so the same class can serve every Gemma4
        checkpoint honestly:

        1. If ``config.text_config.enable_moe_block`` is True → MoE variant
           (e.g. ``google/gemma-4-26B-A4B-it``).
        2. Else if ``config.audio_config`` is not ``None`` → dense + audio
           variant (e.g. ``google/gemma-4-E2B-it``, ``google/gemma-4-E4B-it``).
        3. Else → plain dense variant (e.g. ``google/gemma-4-31B-it``).

        Args:
            config: The model's ``Gemma4Config`` (or anything exposing a
                ``text_config`` with ``enable_moe_block`` and an
                ``audio_config`` attribute).

        Returns:
            A populated :class:`ModelCapabilities` for this specific config.
        """
        text_config = getattr(config, "text_config", config)
        if bool(getattr(text_config, "enable_moe_block", False)):
            # MoE variant: gemma-4-26B-A4B-it
            return ModelCapabilities(
                supports_tp=False,
                supports_cp=True,
                supports_pp=False,
                supports_ep=True,
            )
        if getattr(config, "audio_config", None) is not None:
            # Dense + audio variant: gemma-4-E2B-it, gemma-4-E4B-it.
            # CP supported; TP/PP/EP off. Two features beyond plain-dense 31B were
            # exercised:
            #   * per-layer inputs (``hidden_size_per_layer_input``): under CP,
            #     computed on the full sequence in ``prepare_model_inputs_for_cp``
            #     and sharded contiguously on the seq dim alongside
            #     ``inputs_embeds`` (the 4D ``per_layer_inputs`` tensor is a known
            #     key in cp_batch); the HF dense forward applies the token-local
            #     projection per shard (verified bit-identical to non-CP prep).
            #   * KV-sharing: when the KV cache is active each shared layer reads
            #     its source layer's CP-local sharded K/V from
            #     ``DynamicCache.shared_layers`` and rotates it through the same
            #     ring as any other K/V. Under the activation checkpointing that
            #     CP training uses HF disables the cache, so shared layers
            #     recompute their own K/V -- identical between CP and non-CP, so
            #     parity holds either way (the dead shared-layer K/V projections
            #     are frozen by ``freeze_unused_kv_sharing_params``).
            # TP is intentionally OFF: HF's ``Gemma4Model.forward`` builds the
            # per-layer inputs via ``torch.where(multimodal_mask, pad_embedding,
            # inputs_embeds)`` where ``pad_embedding`` is sliced from the (TP-
            # sharded) embedding weight. Under DTensor this raises "mixed
            # torch.Tensor and DTensor" -- an HF-side limitation we cannot fix
            # without patching frozen transformers source. (Plain-dense 31B has
            # no ``hidden_size_per_layer_input`` so it skips this branch and TP
            # works there.)
            return ModelCapabilities(
                supports_tp=False,
                supports_cp=True,
                supports_pp=False,
                supports_ep=False,
            )
        # Plain dense variant: gemma-4-31B-it
        return ModelCapabilities(
            supports_tp=True,
            supports_cp=True,
            supports_pp=True,
            supports_ep=False,
        )

    @classmethod
    def from_config(
        cls,
        config: Gemma4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        **kwargs,
    ):
        return cls(config, moe_config=moe_config, backend=backend, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        if not _GEMMA4_HF_AVAILABLE:
            raise UnavailableError("transformers.models.gemma4 is not available.")
        config = Gemma4Config.from_pretrained(pretrained_model_name_or_path)
        return cls.from_config(config, *model_args, **kwargs)

    def setup_cp_attention(self, cp_mesh) -> None:
        """Install Gemma4's model-owned p2p ring CP attention (dense path).

        Idempotent: flips the ``_cp_enabled`` flag the forward reads and installs
        the ring on every self-attn module (each was given a per-module
        ``setup_cp_attention`` by ``attach_gemma4_cp_ring_attention`` at
        construction). Invoked from Gemma4's own batch-sharding callable
        (``_cp_shard_batch``) the first time the recipe hands it the CP submesh, so
        the install is fully model-owned -- no framework dispatch is required.
        """
        if getattr(self, "_cp_enabled", False):
            return
        self._cp_enabled = True
        for module in self.modules():
            if module is self:
                continue
            module_setup = getattr(module, "setup_cp_attention", None)
            if callable(module_setup):
                module_setup(cp_mesh)

    def _cp_shard_batch(self, cp_mesh, tp_mesh, batch, *, loss_mask=None, padding_token_id=0):
        """Gemma4-owned CP batch sharder that also self-installs the ring.

        Attached to the batch as ``_cp_make_batch_fn`` by
        ``prepare_model_inputs_for_cp``. ``cp_utils.make_cp_batch_and_ctx`` calls it
        with the CP submesh, which is the one place Gemma4 receives ``cp_mesh`` on a
        model-owned path -- so install the ring here (idempotent) before sharding,
        rather than depending on the framework to call ``setup_cp_attention``.
        """
        self.setup_cp_attention(cp_mesh)
        return make_contiguous_shard_cp_batch_and_ctx(
            cp_mesh, tp_mesh, batch, loss_mask=loss_mask, padding_token_id=padding_token_id
        )

    def __init__(
        self,
        config: Gemma4Config,
        moe_config: MoEConfig | None = None,
        backend: BackendConfig | None = None,
        text_config: dict | None = None,
        **kwargs,
    ):
        if not _GEMMA4_HF_AVAILABLE:
            raise UnavailableError("transformers.models.gemma4 is not available.")
        backend = backend or BackendConfig()

        # Merge text_config overrides (e.g. from YAML) into the proper config
        # object before HF __init__ which needs a real PretrainedConfig.
        if text_config is not None and isinstance(text_config, dict):
            cfg_text = config.text_config if hasattr(config, "text_config") else config
            for k, v in text_config.items():
                setattr(cfg_text, k, v)

        # Compat: older checkpoints used expert_intermediate_size, v5.5+ uses moe_intermediate_size.
        cfg_text = config.text_config if hasattr(config, "text_config") else config
        if not getattr(cfg_text, "moe_intermediate_size", None) and getattr(cfg_text, "expert_intermediate_size", None):
            cfg_text.moe_intermediate_size = cfg_text.expert_intermediate_size
        if not getattr(cfg_text, "expert_intermediate_size", None) and getattr(cfg_text, "moe_intermediate_size", None):
            cfg_text.expert_intermediate_size = cfg_text.moe_intermediate_size

        # _init_model() only overrides the top-level hf_config.torch_dtype; for
        # VL configs the nested text_config / vision_config keep their original
        # dtype (typically bf16 from the checkpoint's config.json). Propagate
        # the user-requested dtype to every nested sub-config that exposes a
        # torch_dtype attribute, before constructing the HF parent and our
        # text backend.
        top_dtype = getattr(config, "torch_dtype", None)
        if top_dtype is not None:
            for sub_cfg in vars(config).values():
                if sub_cfg is not config and hasattr(sub_cfg, "torch_dtype"):
                    sub_cfg.torch_dtype = top_dtype

        # Initialize the HF parent (creates self.model, self.lm_head, vision tower, etc.)
        super().__init__(config)

        self.backend = backend

        text_config = config.text_config if hasattr(config, "text_config") else config
        enable_moe = getattr(text_config, "enable_moe_block", False)

        pad_token_id = getattr(text_config, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = getattr(text_config, "eos_token_id", None)
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            pad_token_id = eos_token_id
        self.pad_token_id = pad_token_id if pad_token_id is not None else -1

        if not enable_moe:
            # Dense Gemma4 — keep vanilla HF model. Attach the model-owned p2p ring
            # CP attention to each HF self-attn so setup_cp_attention can install it
            # when CP is enabled. (The MoE path attaches it per Gemma4MoEDecoderLayer.)
            for module in self.modules():
                if isinstance(module, Gemma4Attention):
                    attach_gemma4_cp_ring_attention(module)
            return

        # --- MoE path: replace the text model ---
        moe_overrides = kwargs.pop("moe_overrides", None)
        self.model.__class__ = Gemma4MoEModel
        self.model.language_model = Gemma4MoETextModelBackend(
            text_config,
            backend=self.backend,
            moe_config=moe_config,
            moe_overrides=moe_overrides,
        )

        # Expose moe_config for the MoE parallelizer assertion
        self.model.moe_config = self.model.language_model.moe_config

        # HF's super().__init__() tied lm_head.weight to the *original* text
        # embed_tokens, but the language_model replacement above swapped in a
        # fresh embed_tokens and orphaned that alias. Re-tie through our own
        # tie_weights() override so the public hook (also invoked by AutoModel
        # and checkpoint load via ensure_tied_lm_head) re-points to the active
        # MoE embedding. The shared Parameter survives the in-place cast in
        # initialize_weights().
        self.tie_weights()

        self.vocab_size = text_config.vocab_size
        # State dict adapter for HF ↔ NeMo weight conversion
        if self.backend.enable_hf_state_dict_adapter:
            self.state_dict_adapter = Gemma4MoEStateDictAdapter(
                text_config,
                self.model.language_model.moe_config,
                self.backend,
                dtype=get_dtype(getattr(text_config, "torch_dtype", None), torch.bfloat16),
            )

    def tie_weights(self, *_args: object, **_kwargs: object) -> None:
        """Tie ``lm_head`` to the active text ``embed_tokens`` when requested.

        Overrides HF's generic tying so that any caller after the MoE
        ``language_model`` swap (construction, AutoModel, and checkpoint load
        via ``ensure_tied_lm_head``) re-points ``lm_head`` to the *active*
        embedding rather than whatever HF's ``get_input_embeddings()``
        indirection resolves to. No-op when the config requests untied
        embeddings.

        Accepts and ignores positional/keyword arguments (e.g. HF v5's
        ``recompute_mapping``) so it stays drop-in compatible with the HF
        ``init_weights() -> tie_weights(...)`` call path.

        The controlling flag is the top-level ``Gemma4Config.tie_word_embeddings``
        (verified against HF: the top-level flag decides tying regardless of the
        nested ``text_config`` value), so read it first and only fall back to
        ``text_config`` for configs that don't expose a top-level flag.
        """
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        if getattr(self.config, "tie_word_embeddings", getattr(text_config, "tie_word_embeddings", False)):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        position_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        _pre_embed_only: bool = False,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Any,
    ):
        if _pre_embed_only:
            return self.prepare_model_inputs_for_cp(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                mm_token_type_ids=mm_token_type_ids,
            )

        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(
                self.config.text_config if hasattr(self.config, "text_config") else self.config,
                "output_hidden_states",
                False,
            )
        )

        if cache_position is None:
            if input_ids is not None:
                seq_len = input_ids.shape[-1]
                cache_position = torch.arange(seq_len, device=input_ids.device)
            elif inputs_embeds is not None:
                seq_len = inputs_embeds.shape[1]
                cache_position = torch.arange(seq_len, device=inputs_embeds.device)

        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        cp_enabled = getattr(self, "_cp_enabled", False)
        if not getattr(text_config, "enable_moe_block", False):
            per_layer_inputs = kwargs.pop("per_layer_inputs", None)
            if cp_enabled:
                if pixel_values is not None:
                    raise NotImplementedError(
                        "Context parallelism with Gemma4 pixel_values requires pre-computed inputs_embeds. "
                        "Call prepare_model_inputs_for_cp before CP sharding and pass inputs_embeds instead."
                    )
                if input_ids is not None and inputs_embeds is None:
                    inputs_embeds = self.model.get_input_embeddings()(input_ids)
                if inputs_embeds is None:
                    raise ValueError("Gemma4 CP dense forward requires either input_ids or inputs_embeds.")

                use_cache = kwargs.pop("use_cache", None)
                past_key_values = kwargs.pop("past_key_values", None)
                logits_to_keep = kwargs.pop("logits_to_keep", logits_to_keep)
                kwargs.pop("labels", None)

                # E2B/E4B only: under use_cache=False (forced by CP + activation
                # checkpointing) HF's kv-sharing read is disabled and the trailing
                # shared layers fall back to their dead K/V projections. Inject a
                # cache-free holder so HF's own kv-sharing fires. 31B (no kv-sharing)
                # is unaffected: the holder is not created. See _Gemma4KVShareHolder.
                if past_key_values is None and _kv_sharing_active(text_config):
                    past_key_values = _Gemma4KVShareHolder()

                # Dense Gemma4 rides HF's decoder layers, which don't thread the
                # CP/vision metadata down to self_attn (the MoE backend passes it via
                # kwargs). Stash the CP-sharded metadata on each ring-hooked attention
                # module so the ring builds the vision-bidirectional / packed masks
                # rather than a plain causal mask (which corrupts multimodal attention).
                cp_meta = {
                    "mm_token_type_ids": mm_token_type_ids,
                    "padding_mask": padding_mask,
                    "_packed_seq_ids": kwargs.get("_packed_seq_ids"),
                    "_gemma4_vision_group_ids": kwargs.get("_gemma4_vision_group_ids"),
                }
                # Left set (not cleared) so the activation-checkpoint recompute in
                # backward sees the same metadata; each CP forward overwrites it.
                for _mod in self.modules():
                    if getattr(_mod, "_cp_uses_attention_hook", False):
                        _mod._cp_dense_metadata = cp_meta

                text_outputs = self.model.language_model(
                    input_ids=None,
                    inputs_embeds=inputs_embeds,
                    per_layer_inputs=per_layer_inputs,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    padding_mask=padding_mask,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    **kwargs,
                )
                hidden_states = text_outputs.last_hidden_state
                slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
                logits = self.lm_head(hidden_states[:, slice_indices, :])
                if (final_logit_softcapping := getattr(text_config, "final_logit_softcapping", None)) is not None:
                    logits = logits / final_logit_softcapping
                    logits = torch.tanh(logits)
                    logits = logits * final_logit_softcapping
                return Gemma4CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=text_outputs.past_key_values,
                    hidden_states=text_outputs.hidden_states,
                    attentions=text_outputs.attentions,
                    image_hidden_states=None,
                    audio_hidden_states=None,
                )

            if (
                mm_token_type_ids is None
                and getattr(text_config, "use_bidirectional_attention", None) == "vision"
                and self.training
            ):
                ref = input_ids if input_ids is not None else inputs_embeds
                mm_token_type_ids = torch.zeros(ref.shape[:2], dtype=torch.long, device=ref.device)

            # E2B/E4B only: same cache-free kv-sharing holder as the CP path above,
            # so the trailing shared layers read their source K/V even when the HF
            # cache is off (use_cache=False / activation checkpointing). 31B and other
            # non-kv-sharing variants get None here, preserving the prior behavior.
            kv_share_holder = (
                _Gemma4KVShareHolder()
                if (kwargs.get("past_key_values") is None and _kv_sharing_active(text_config))
                else kwargs.pop("past_key_values", None)
            )
            kwargs.pop("past_key_values", None)

            # Dense path — delegate to HF forward (which already supports
            # logits_to_keep + output_hidden_states and returns a ModelOutput
            # carrying logits and hidden_states).
            #
            # Gemma4 HF transformers shares keys/values between attention layers by passing
            # one shared key/value store into every decoder layer (earlier layers
            # write it, later "kv-shared" layers read it). NOTE: this dict-based
            # design is new in transformers 5.8.x; 5.5 stored the shared
            # keys/values on the Cache object instead.
            #
            # Under FSDP2 each layer is wrapped separately and the input-casting
            # step (cast_forward_inputs) makes a fresh copy of any plain dict
            # before each layer runs, so the earlier layer's writes never reach
            # the later layers -> KeyError. Pass an _FSDPSafeSharedKVStates
            # instead, which FSDP2 leaves as one shared object. setdefault avoids
            # overwriting a store a caller already provided (e.g. the drafter).
            kwargs.setdefault("shared_kv_states", _FSDPSafeSharedKVStates())
            return super().forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                cache_position=cache_position,
                pixel_values=pixel_values,
                image_position_ids=image_position_ids,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                output_hidden_states=output_hidden_states,
                past_key_values=kv_share_holder,
                **kwargs,
            )

        # --- MoE forward path ---
        if input_ids is not None and inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)

        # Handle vision tokens
        if pixel_values is not None:
            if cp_enabled:
                raise NotImplementedError(
                    "Context parallelism with Gemma4 pixel_values requires pre-computed inputs_embeds. "
                    "Call prepare_model_inputs_for_cp before CP sharding and pass inputs_embeds instead."
                )

            image_features = self.model.get_image_features(
                pixel_values, image_position_ids=image_position_ids, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)

            if mm_token_type_ids is not None:
                special_image_mask = mm_token_type_ids == 1
            elif input_ids is not None:
                special_image_mask = input_ids == self.config.image_token_id
            else:
                special_image_mask = torch.zeros(inputs_embeds.shape[:2], dtype=torch.bool, device=inputs_embeds.device)

            image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_features)

        outputs = self.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            padding_mask=padding_mask,
            mm_token_type_ids=mm_token_type_ids,
            pixel_values=pixel_values,
            cp_enabled=cp_enabled,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state

        logits = compute_lm_head_logits(self.lm_head, hidden_states, logits_to_keep).logits

        if (final_logit_softcapping := getattr(text_config, "final_logit_softcapping", None)) is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping

        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states if output_hidden_states else None,
        )

    def _get_special_image_mask(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mm_token_type_ids is not None:
            return mm_token_type_ids == 1
        return input_ids == self.config.image_token_id

    def _prepare_per_layer_inputs_for_cp(
        self,
        input_ids: torch.Tensor,
        special_image_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        language_model = getattr(self.model, "language_model", None)
        if language_model is None or not getattr(language_model, "hidden_size_per_layer_input", None):
            return None

        pad_token_id = self._get_text_pad_token_id()
        llm_input_ids = input_ids.masked_fill(special_image_mask, pad_token_id)
        return language_model.get_per_layer_inputs(llm_input_ids, None)

    def _get_text_pad_token_id(self) -> int:
        pad_token_id = getattr(self, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            cfg = getattr(self, "config", None)
            cfg_text = getattr(cfg, "text_config", cfg)
            pad_token_id = getattr(cfg_text, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            eos_token_id = getattr(getattr(self, "config", None), "eos_token_id", None)
            if eos_token_id is None:
                cfg_text = getattr(getattr(self, "config", None), "text_config", None)
                eos_token_id = getattr(cfg_text, "eos_token_id", None) if cfg_text else None
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            pad_token_id = eos_token_id
        if pad_token_id is None or pad_token_id < 0:
            raise ValueError("Gemma4 per-layer inputs require a valid pad_token_id.")
        return pad_token_id

    def prepare_model_inputs_for_cp(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Prepare Gemma4 embeddings on the full sequence before CP sharding."""
        if input_ids is None:
            raise ValueError("prepare_model_inputs_for_cp requires input_ids.")

        special_image_mask = self._get_special_image_mask(input_ids, mm_token_type_ids)
        llm_input_ids = input_ids.masked_fill(special_image_mask, self._get_text_pad_token_id())
        inputs_embeds = self.model.get_input_embeddings()(llm_input_ids)
        prepared_inputs: dict[str, Any] = {
            "inputs_embeds": inputs_embeds,
            "mm_token_type_ids": mm_token_type_ids
            if mm_token_type_ids is not None
            else special_image_mask.to(torch.long),
            "_cp_make_batch_fn": self._cp_shard_batch,
            "_gemma4_vision_group_ids": gemma4_vision_group_ids(
                mm_token_type_ids if mm_token_type_ids is not None else special_image_mask.to(torch.long)
            ),
            "_cp_metadata_seq_dims": {"_gemma4_vision_group_ids": 1},
            "_cp_metadata_pad_values": {"_gemma4_vision_group_ids": -1},
        }

        per_layer_inputs = self._prepare_per_layer_inputs_for_cp(input_ids, special_image_mask)
        if per_layer_inputs is not None:
            prepared_inputs["per_layer_inputs"] = per_layer_inputs

        if pixel_values is not None:
            image_features = self.model.get_image_features(
                pixel_values, image_position_ids=image_position_ids, return_dict=True
            ).pooler_output
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
            prepared_inputs["inputs_embeds"] = inputs_embeds.masked_scatter(image_mask, image_features)

        return prepared_inputs

    def prepare_inputs_embeds_for_cp(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        image_position_ids: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.prepare_model_inputs_for_cp(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_position_ids=image_position_ids,
            mm_token_type_ids=mm_token_type_ids,
        )["inputs_embeds"]

    @torch.no_grad()
    def initialize_weights(
        self,
        buffer_device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        text_config = self.config.text_config if hasattr(self.config, "text_config") else self.config
        if not getattr(text_config, "enable_moe_block", False):
            for p in self.parameters():
                p.data = p.data.to(dtype)
            return

        # Guard: HF's super().__init__() calls post_init() -> init_weights() ->
        # initialize_weights() *before* __init__ replaces the language model
        # with Gemma4MoETextModelBackend (which uses ModuleDict).  At that
        # point layers is still HF's ModuleList which leads to an AttributeError, just cast and return.
        # Needed only when constructing the model directly, doesn't affect when loading a ckpt via from_pretrained().
        language_model = self.model.language_model
        if not isinstance(language_model, Gemma4MoETextModelBackend):
            cast_model_to_dtype(self, dtype)
            return

        buffer_device = buffer_device or torch.device(f"cuda:{torch.cuda.current_device()}")

        with buffer_device:
            for layer in language_model.layers.values():
                layer.moe.init_weights(buffer_device)

        cast_model_to_dtype(self, dtype)


if _GEMMA4_HF_AVAILABLE:
    ModelClass = Gemma4ForConditionalGeneration
