# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""HY V4 lightning indexer and gated sparse MLA layers.

Contains the HyV4Indexer for top-k sparse attention selection
and HyV4MLA which integrates the indexer with Multi-head Latent Attention.
"""

from typing import Any

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import (
    BackendConfig,
    initialize_linear_module,
    initialize_rms_norm_module,
)
from nemo_automodel.components.models.hy_v4.config import HyV4Config
from nemo_automodel.components.models.hy_v4.cp import hy_v4_cp_all_gather, hy_v4_cp_enabled
from nemo_automodel.components.models.hy_v4.optimized_kernels import (
    cudnn_indexer_topk,
    cudnn_sparse_attention,
    is_cudnn_dsa_available,
)
from nemo_automodel.components.models.hy_v4.rope_utils import apply_rotary_emb, mla_softmax_scale
from nemo_automodel.shared.utils import dtype_from_str as get_dtype


def _full_tensor_if_dtensor(tensor: torch.Tensor) -> torch.Tensor:
    """Materialize a model-owned parameter for a parent kernel call.

    Args:
        tensor: Local tensor or DTensor with arbitrary parameter layout.

    Returns:
        A local clone with the tensor's global shape for DTensor input, or its
        original shape otherwise. The result never aliases ``tensor``.
    """
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.clone()


class HyV4FP32Parameter(nn.Module):
    """Callable FP32 parameter holder that forms an independent FSDP unit."""

    def __init__(self, value: torch.Tensor):
        """Create an independently shardable FP32 parameter.

        Args:
            value: Initial parameter value with arbitrary layout. Storage is
                copied, converted to FP32, and does not alias this tensor.
        """
        super().__init__()
        self.weight = nn.Parameter(value.to(torch.float32).clone())

    def forward(self) -> torch.Tensor:
        """Return a materialized local clone with ``weight``'s global layout."""
        return _full_tensor_if_dtensor(self.weight)


class HyV4Indexer(nn.Module):
    """Indexer for top-k sparse attention selection.

    Ported from vLLM's HY V4 indexer. Computes attention
    scores between queries and keys with per-head weights, applies ReLU activation,
    then selects the top-k positions to attend to.

    Key features:
    - Uses LayerNorm (not RMSNorm) for key normalization
    - Has a weights_proj that learns per-head importance weights
    - ReLU activation on attention scores before weighting
    """

    def __init__(self, config: HyV4Config, backend: BackendConfig):
        super().__init__()

        self.num_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.hidden_size = config.hidden_size
        self.softmax_scale = self.head_dim**-0.5

        self.backend = backend
        linear_impl = backend.linear
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # Project Q from q_lora residual -> num_heads * head_dim
        self.wq_b = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.num_heads * self.head_dim,
            bias=False,
            dtype=dtype,
        )

        # Project K from hidden states -> single head_dim (shared across heads)
        self.wk = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.head_dim,
            bias=False,
            dtype=dtype,
        )

        # vLLM HY V4 uses LayerNorm (not RMSNorm) with eps=1e-6.
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6, dtype=dtype)

        # Per-head weight projection from hidden states
        self.weights_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.hidden_size,
            out_features=self.num_heads,
            bias=False,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        q_resid: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Compute top-k indices for sparse attention.

        Args:
            x: Packed hidden states of shape ``[tokens, hidden]``.
            q_resid: Packed Query-LoRA residual of shape ``[tokens, q_lora_rank]``.
            freqs_cis: Packed complex RoPE tensor of shape
                ``[tokens, rope_head_dim / 2]``.
            attention_mask: Must be ``None``. Document boundaries are carried
                by ``cu_seqlens`` in the vLLM-compatible packed path.
            **attn_kwargs: Attention metadata. Optimized packed backends require an int32
                ``cu_seqlens`` tensor of shape ``[sequences + 1]``. Packed CP may also
                supply ``hy_v4_cp_query_indices`` of shape ``[tokens]`` and
                ``cu_seqlens_padded`` of shape ``[sequences + 1]``. The cuDNN backend
                requires CUDA bfloat16 query/key tensors.

        Returns:
            Contiguous int32 top-k indices with shape
            ``[tokens, 1, index_topk]``.
        """
        if self.backend.attn != "cudnn" or x.dim() != 2:
            raise NotImplementedError(
                "HY V4 follows the vLLM sparse-attention forward and requires "
                "backend.attn='cudnn' with packed THD inputs."
            )
        if attention_mask is not None:
            raise ValueError("Packed HY V4 indexer masks must be represented by cu_seqlens, not attention_mask.")
        num_tokens = x.shape[0]

        # Project Q from q_lora residual
        q = self.wq_b(q_resid).view(num_tokens, self.num_heads, self.head_dim)

        # The HY V4 checkpoint lays out indexer features as [nope, rope].  Both
        # slices use the same interleaved (Megatron/PTM) RoPE convention as MLA.
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # Project K from hidden states
        k = self.k_norm(self.wk(x))

        k_nope, k_pe = torch.split(k, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format="thd")
        k_pe = apply_rotary_emb(k_pe.unsqueeze(-2), freqs_cis, qkv_format="thd").squeeze(-2)

        q = torch.cat([q_nope, q_pe], dim=-1)
        k = torch.cat([k_nope, k_pe], dim=-1)
        cp_group = attn_kwargs.get("_hy_v4_cp_group")
        cp_enabled = hy_v4_cp_enabled(cp_group)
        if cp_enabled:
            k = hy_v4_cp_all_gather(k, dim=0, cp_group=cp_group)

        cu_seqlens = attn_kwargs.get("cu_seqlens")
        if cu_seqlens is None:
            raise ValueError("cuDNN DSA indexer requires 'cu_seqlens' in attn_kwargs (THD packing metadata).")
        if not is_cudnn_dsa_available():
            raise RuntimeError(
                "backend.attn='cudnn' requires the optional cuDNN DSA and FlashMLA kernels, "
                "but they are unavailable in this environment."
            )
        # vLLM folds q's quantization scale, indexer softmax scale, and
        # n_head**-0.5 into these weights. The BF16 cuDNN path has no q
        # quantization scale, so the remaining two factors are applied here.
        head_weights = self.weights_proj(x).float() * (self.num_heads**-0.5) * self.softmax_scale
        return cudnn_indexer_topk(
            q.contiguous(),
            k.contiguous(),
            head_weights.contiguous(),
            cu_seqlens.flatten().to(torch.int32),
            self.index_topk,
            query_indices=attn_kwargs.get("hy_v4_cp_query_indices"),
            cu_seqlens_padded=(
                attn_kwargs["cu_seqlens_padded"].flatten().to(torch.int32)
                if "cu_seqlens_padded" in attn_kwargs
                else None
            ),
            packed_metadata=attn_kwargs.get("_cudnn_dsa_packed_metadata"),
        )

    def init_weights(self, init_std: float = 0.02):
        for module in [self.wq_b, self.wk, self.weights_proj]:
            if hasattr(module, "weight"):
                nn.init.trunc_normal_(module.weight, mean=0.0, std=init_std)
        self.k_norm.reset_parameters()


class HyV4MLA(nn.Module):
    """Multi-head Latent Attention with Indexer for sparse attention.

    This extends the V3 MLA with an Indexer module that performs
    top-k selection for sparse attention. The indexer uses the
    q_lora residual and hidden states to compute which positions
    to attend to.
    """

    def __init__(self, config: HyV4Config, backend: BackendConfig, skip_topk: bool = False):
        """Initialize MLA with an optional sparse-attention indexer.

        Args:
            config: Model config carrying MLA and indexer dimensions.
            backend: Backend selection for attention/linear/norm kernels.
            skip_topk: When ``True``, this layer owns no indexer and instead reuses the
                top-k selection of the previous "full" indexer layer (HY V4 IndexShare).
                ``forward`` then requires ``prev_topk_indices`` to be supplied. Defaults
                to ``False`` (the layer runs its own indexer), preserving the full-indexer
                behavior used by HY V4 configs without IndexShare metadata.
        """
        super().__init__()

        self.skip_topk = skip_topk
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = (
            config.qk_head_dim if hasattr(config, "qk_head_dim") else (self.qk_nope_head_dim + self.qk_rope_head_dim)
        )
        self.v_head_dim = config.v_head_dim
        self.index_topk = config.index_topk
        self.gated_mla = config.gated_mla
        self.learnable_sink = config.learnable_sink

        self.backend = backend
        self._cp_group = None
        linear_impl = backend.linear
        rms_norm_impl = backend.rms_norm

        hidden_size = config.hidden_size
        dtype = get_dtype(getattr(config, "torch_dtype", None), torch.bfloat16)

        # HY V4 always uses q_lora (q_lora_rank is not None)
        self.q_a_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.q_lora_rank,
            bias=False,
            dtype=dtype,
        )
        self.q_a_layernorm = initialize_rms_norm_module(
            rms_norm_impl=rms_norm_impl,
            dim=self.q_lora_rank,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.q_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.q_lora_rank,
            out_features=self.n_heads * self.qk_head_dim,
            bias=False,
            dtype=dtype,
        )

        self.kv_a_proj_with_mqa = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=hidden_size,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
            dtype=dtype,
        )
        self.kv_a_layernorm = initialize_rms_norm_module(
            rms_norm_impl=rms_norm_impl,
            dim=self.kv_lora_rank,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
        self.kv_b_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.kv_lora_rank,
            out_features=self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
            dtype=dtype,
        )
        self.o_proj = initialize_linear_module(
            linear_impl=linear_impl,
            in_features=self.n_heads * self.v_head_dim,
            out_features=hidden_size,
            bias=False,
            dtype=dtype,
        )
        if self.gated_mla:
            self.linear_gate = initialize_linear_module(
                linear_impl=linear_impl,
                in_features=hidden_size,
                out_features=self.n_heads * self.v_head_dim,
                bias=False,
                dtype=dtype,
            )
        else:
            self.linear_gate = None

        if self.learnable_sink:
            self.learnable_sink_param = HyV4FP32Parameter(
                torch.full((self.n_heads,), float(config.learnable_sink_init), dtype=torch.float32)
            )
        else:
            self.learnable_sink_param = None
        # Shared with the HY V4 DSpark draft, which trains on this model's hidden
        # states and must use the identical (YaRN-corrected) attention temperature.
        self.softmax_scale = mla_softmax_scale(self.qk_head_dim)

        # Initialize the Indexer. "shared" layers (HY V4 IndexShare) own no indexer and
        # reuse the previous full layer's top-k indices passed in via `prev_topk_indices`.
        self.indexer = None if skip_topk else HyV4Indexer(config, backend)

    def setup_cp_attention(self, cp_mesh) -> None:
        """Record the model-owned packed-CP group installed by ``apply_cp``."""
        self._cp_group = cp_mesh.get_group()

    def _apply_output_gate(self, hidden_states: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
        """Apply HY4-preview's elementwise sigmoid MLA output gate.

        Args:
            hidden_states: Packed source states shaped ``[tokens, hidden]``.
            attn_out: Per-head attention values shaped
                ``[tokens, heads, value_head_dim]``.

        Returns:
            New gated values with the same shape and dtype as ``attn_out``.
        """
        if self.linear_gate is None:
            return attn_out
        gate = self.linear_gate(hidden_states)
        gate = gate.unflatten(-1, (self.n_heads, self.v_head_dim))
        return attn_out * torch.sigmoid(gate)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        return_topk_indices: bool = False,
        **attn_kwargs: Any,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Run MLA with (optionally shared) DSA sparse attention.

        Args:
            x: Packed hidden states of shape ``[tokens, hidden]``.
            freqs_cis: Packed complex RoPE tensor of shape
                ``[tokens, rope_head_dim / 2]``.
            attention_mask: Must be ``None``; packed boundaries use ``cu_seqlens``.
            prev_topk_indices: Top-k index tensor from the most recent full indexer layer.
                The cuDNN path uses int32 ``[tokens, 1, index_topk]``.
                Required only for a shared layer (``skip_topk=True``).
            return_topk_indices: When ``True``, return ``(attn_out, topk_indices)`` so the
                caller can thread the selection to subsequent shared layers (HY V4 IndexShare).
                When ``False`` (default), return just ``attn_out``.
            **attn_kwargs: Attention metadata. Optimized packed backends require an int32
                ``cu_seqlens`` tensor of shape ``[sequences + 1]``. Packed CP may also
                provide ``_hy_v4_cp_group`` and packed query-position tensors. The cuDNN
                backend requires CUDA bfloat16 inputs.

        Returns:
            Attention output tensor of shape ``[tokens, hidden]``. When
            ``return_topk_indices`` is true,
            returns that tensor with the top-k index tensor described above.
            The attention output is newly computed; an IndexShare layer passes
            ``prev_topk_indices`` through unchanged as the selection output.
        """
        if self.backend.attn != "cudnn" or x.dim() != 2:
            raise NotImplementedError(
                "HY V4 follows the vLLM sparse-attention forward and requires "
                "backend.attn='cudnn' with packed THD inputs."
            )
        if attention_mask is not None:
            raise ValueError("Packed HY V4 attention masks must be represented by cu_seqlens.")
        num_tokens = x.shape[0]
        cp_group = attn_kwargs.get("_hy_v4_cp_group", self._cp_group)
        if cp_group is not None:
            attn_kwargs["_hy_v4_cp_group"] = cp_group

        # Compute q_resid for indexer and main attention path
        q_resid = self.q_a_layernorm(self.q_a_proj(x))

        # Get top-k indices: run our own indexer ("full" layer), or reuse the previous
        # full layer's selection ("shared" layer, HY V4 IndexShare).
        if self.indexer is not None:
            topk_indices = self.indexer(x, q_resid, freqs_cis, None, **attn_kwargs)
        else:
            if prev_topk_indices is None:
                raise ValueError(
                    "Shared DSA layers (skip_topk=True) require top-k indices from a previous "
                    "full indexer layer; got prev_topk_indices=None."
                )
            topk_indices = prev_topk_indices

        # Compute Q from q_resid
        q = self.q_b_proj(q_resid)

        q = q.view(num_tokens, self.n_heads, self.qk_head_dim)

        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv = self.kv_a_proj_with_mqa(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_layernorm(kv)

        # For MLA, k_pe needs an extra head dimension for apply_rotary_emb
        k_pe = k_pe.unsqueeze(1)

        # Apply rotary embeddings to q_pe and k_pe
        q_pe = apply_rotary_emb(q_pe, freqs_cis, qkv_format="thd")
        k_pe = apply_rotary_emb(k_pe, freqs_cis, qkv_format="thd")

        # Remove the head dimension we added to k_pe
        k_pe = k_pe.squeeze(1)
        cp_enabled = hy_v4_cp_enabled(cp_group)
        if cp_enabled:
            kv = hy_v4_cp_all_gather(kv, dim=0, cp_group=cp_group)
            k_pe = hy_v4_cp_all_gather(k_pe, dim=0, cp_group=cp_group)

        # The optimized sparse path runs on the absorbed latent representation. The k_nope
        # up-projection is folded into the query, and w_vc maps the latent output back.
        materialize_effective_weight = getattr(self.kv_b_proj, "materialize_effective_weight", None)
        kv_b_weight = (
            materialize_effective_weight() if materialize_effective_weight is not None else self.kv_b_proj.weight
        )
        w = kv_b_weight.view(self.n_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank)
        w_kc = w[:, : self.qk_nope_head_dim, :]
        w_vc = w[:, self.qk_nope_head_dim :, :]
        q_absorbed = torch.einsum("thd,hdc->thc", q_nope, w_kc.to(q_nope.dtype))
        q_sparse = torch.cat([q_absorbed, q_pe], dim=-1).to(torch.bfloat16)
        kv_latent = torch.cat([kv, k_pe], dim=-1).unsqueeze(1).to(torch.bfloat16)
        if not is_cudnn_dsa_available():
            raise RuntimeError(
                "backend.attn='cudnn' requires the optional cuDNN DSA and FlashMLA kernels, "
                "but they are unavailable in this environment."
            )
        expected_topk_shape = (num_tokens, 1, self.index_topk)
        if tuple(topk_indices.shape) != expected_topk_shape:
            raise ValueError(
                "cuDNN DSA sparse attention requires fixed-width top-k indices of shape "
                f"{expected_topk_shape}; got {tuple(topk_indices.shape)}."
            )
        latent_out = cudnn_sparse_attention(
            q_sparse,
            kv_latent,
            topk_indices,
            self.softmax_scale,
            attn_sink=(
                self.learnable_sink_param().float().contiguous() if self.learnable_sink_param is not None else None
            ),
            topk_length=attn_kwargs.get("_cudnn_dsa_topk_length"),
            all_rows_nonempty=bool(attn_kwargs.get("_cudnn_dsa_all_rows_nonempty", False)),
            valid_row_indices=attn_kwargs.get("_cudnn_dsa_valid_row_indices"),
        )
        attn_out = torch.einsum("thc,hdc->thd", latent_out, w_vc.to(latent_out.dtype))
        attn_out = self._apply_output_gate(x, attn_out)
        x = self.o_proj(attn_out.flatten(-2))
        if return_topk_indices:
            return x, topk_indices
        return x

    def init_weights(self, _buffer_device: torch.device, init_std: float = 0.02):
        linear_list = [
            self.q_a_proj,
            self.q_b_proj,
            self.kv_a_proj_with_mqa,
            self.kv_b_proj,
            self.o_proj,
        ]
        if self.linear_gate is not None:
            linear_list.append(self.linear_gate)

        for linear in linear_list:
            nn.init.trunc_normal_(linear.weight, mean=0.0, std=init_std)

        norms = [self.kv_a_layernorm, self.q_a_layernorm]
        for norm in norms:
            norm.reset_parameters()

        # Initialize indexer weights ("shared" layers own no indexer).
        if self.indexer is not None:
            self.indexer.init_weights(init_std)

        if self.learnable_sink_param is not None:
            nn.init.constant_(self.learnable_sink_param.weight, 0.0)
