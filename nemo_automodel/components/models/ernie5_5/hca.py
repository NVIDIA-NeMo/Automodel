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

"""ERNIE 5.5 heavily compressed attention (HCA).

HCA is DeepSeek V4's ratio-128 attention with two additions: a low-rank
virtual-head mix of the per-head attention output, and a sigmoid gate on the
grouped output projection that is driven by the query latent. DeepSeek V4's
projections, norms, compressor, rotary helper and eager attention are reused
unchanged. Only the order in which they are called is restated here, because
both additions sit between two steps of ``DeepseekV4Attention.forward``.

Provenance: the operation order follows the ERNIE 5.5 HCA of Megatron-Bridge (Apache License 2.0) at revision
``a38833622b88969564fe8bfa053c5e4e4989c6cd``:
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/hca_attention.py

Supported inputs are one document per row starting at position zero, an
explicit causal sliding-window mask, the eager attention backend and the torch
rotary backend. Packed rows, context parallelism, a KV cache and the sparse
attention kernels raise an error.
"""

from typing import Any

import torch
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4TrainCache,
    _apply_partial_rope,
    eager_attention_with_sink,
)
from nemo_automodel.components.models.ernie5_5.layers import DENSE_ATTENTION_BACKENDS, virtual_head_mix

_HCA_COMPRESS_RATIO = 128


def _compressed_key_mask(attention_mask: torch.Tensor, pooled: int, ratio: int) -> torch.Tensor:
    """Build the additive mask over compressed keys for rows that start at position zero.

    Query ``q`` sees compressed key ``p`` when all ``ratio`` tokens pooled into
    that key are the query itself or precede it: ``p < (q + 1) // ratio``.

    Args:
        attention_mask: Additive mask of shape [batch, mask_heads, sequence, sequence]
            over the uncompressed keys, where mask_heads is 1 or the head count. Only
            its shape, dtype and device are used.
        pooled: Number of compressed keys.
        ratio: Number of tokens pooled into one compressed key.

    Returns:
        Tensor of shape [batch, mask_heads, sequence, pooled] with the dtype and
        device of ``attention_mask``: zero for visible keys and the dtype's most
        negative finite value for hidden keys. It is a broadcast view of
        [sequence, pooled] storage.
    """
    batch, mask_heads, sequence, _ = attention_mask.shape
    device, dtype = attention_mask.device, attention_mask.dtype
    complete_windows = (torch.arange(sequence, device=device) + 1) // ratio
    visible = torch.arange(pooled, device=device).view(1, pooled) < complete_windows.view(sequence, 1)
    mask = torch.where(
        visible,
        torch.zeros((), dtype=dtype, device=device),
        torch.full((), torch.finfo(dtype).min, dtype=dtype, device=device),
    )
    return mask.view(1, 1, sequence, pooled).expand(batch, mask_heads, sequence, pooled)


class Ernie5_5HCAVirtualHeads(nn.Module):
    """Keep the FP32 virtual-head parameters in a callable FSDP unit.

    Parameters, both FP32: ``vha_postmix_U`` of shape [heads, rank], allocated without initialization and filled
    by the owning layer, and ``vha_postmix_V`` of shape [heads, rank], initialized to zeros.
    """

    def __init__(self, num_heads: int, rank: int) -> None:
        """Allocate the two mixing factors.

        Args:
            num_heads: Number of attention heads.
            rank: Rank of the virtual-head mixing factors.
        """
        super().__init__()
        self.vha_postmix_U = nn.Parameter(torch.empty(num_heads, rank, dtype=torch.float32))
        self.vha_postmix_V = nn.Parameter(torch.zeros(num_heads, rank, dtype=torch.float32))

    def forward(self, attention: torch.Tensor) -> torch.Tensor:
        """Mix the attention output across heads with ``I + V U^T``.

        Args:
            attention: Tensor of shape [batch, sequence, heads, head_dim].

        Returns:
            Newly allocated tensor with the shape and dtype of ``attention``.
        """
        return virtual_head_mix(attention, self.vha_postmix_U, self.vha_postmix_V)


class Ernie5_5HCAAttention(DeepseekV4Attention):
    """DeepSeek V4 ratio-128 attention with ERNIE virtual-head mixing and output gating.

    Construction, parameter names and the initialization of the DeepSeek V4
    parameters are inherited. ``forward`` restates the eager, single-document
    path of ``DeepseekV4Attention.forward`` as of NVIDIA-NeMo/Automodel commit
    6f44dbf2390a4eb31bcdc67898bde98ce408f915
    (``nemo_automodel/components/models/deepseek_v4/layers.py``), with the
    virtual-head mix inserted after the inverse rotation and the gate after the
    grouped projection. With a zero ``vha_postmix_V`` and a zero gate weight the
    output is exactly half of the parent's output; the unit tests compare the two
    so that a change to the parent's forward which this class does not follow
    fails a test.

    The config must describe a compression-ratio-128 layer. It is not sufficient
    by itself to construct a full ERNIE 5.5 model.
    """

    def __init__(
        self,
        config: DeepseekV4Config,
        layer_idx: int,
        backend: BackendConfig | None = None,
        *,
        vha_rank: int = 16,
    ) -> None:
        """Build the DeepSeek V4 attention modules and the ERNIE additions.

        Args:
            config: DeepSeek V4 config describing the attention dimensions.
            layer_idx: Index into ``config.compress_ratios``; that entry must be 128.
            backend: Backend selection. ``attn`` must be ``"eager"``, ``"torch"`` or
                ``"sdpa"``; all three run DeepSeek V4's eager sink attention in PyTorch,
                and no scaled-dot-product or sparse kernel is called. None selects
                ``"eager"``. The Triton rotary kernel is rejected.
            vha_rank: Rank of the virtual-head mixing factors.

        Raises:
            ValueError: If the layer's compression ratio is not 128 or ``vha_rank``
                is not positive.
            NotImplementedError: If ``backend.attn`` is not one of the three dense
                names, or the backend or config selects the Triton rotary kernel or the
                fused query-norm rotation.
        """
        ratios = config.compress_ratios
        if not ratios or ratios[layer_idx] != _HCA_COMPRESS_RATIO:
            ratio = ratios[layer_idx] if ratios else None
            raise ValueError(
                f"ERNIE HCA requires compression ratio {_HCA_COMPRESS_RATIO}; layer {layer_idx} has {ratio}"
            )
        if vha_rank <= 0:
            raise ValueError(f"vha_rank must be positive; got {vha_rank}")
        backend = backend or BackendConfig(attn="eager")
        if backend.attn not in DENSE_ATTENTION_BACKENDS:
            raise NotImplementedError(
                f"ERNIE HCA supports only the dense attention backends {DENSE_ATTENTION_BACKENDS}, "
                f"which run the eager PyTorch path; got attn={backend.attn!r}"
            )
        if backend.rope == "triton":
            raise NotImplementedError("ERNIE HCA does not support the Triton rotary kernel (rope='triton')")
        if config.fuse_qnorm_rope:
            raise NotImplementedError("ERNIE HCA does not support fuse_qnorm_rope")
        super().__init__(config, layer_idx, backend)
        self.vha = Ernie5_5HCAVirtualHeads(config.num_attention_heads, vha_rank)
        self.gate_proj = nn.Linear(config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False)
        nn.init.normal_(self.vha_postmix_U, std=0.02)

    @property
    def vha_postmix_U(self) -> torch.Tensor:
        """FP32 virtual-head input factor of shape [heads, vha_rank], owned by ``vha``."""
        return self.vha.vha_postmix_U

    @property
    def vha_postmix_V(self) -> torch.Tensor:
        """FP32 virtual-head output factor of shape [heads, vha_rank], owned by ``vha``."""
        return self.vha.vha_postmix_V

    def setup_cp_attention(self, cp_mesh: DeviceMesh) -> None:
        """Reject context parallelism, which the inherited hook would otherwise enable.

        Args:
            cp_mesh: Context-parallel device mesh offered by the parallelizer.

        Raises:
            NotImplementedError: Always.
        """
        del cp_mesh
        raise NotImplementedError("ERNIE HCA does not support context parallelism")

    def _validate_inputs(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None,
        start_pos: int,
        position_ids: torch.Tensor | None,
        extra: dict[str, Any],
    ) -> torch.Tensor:
        """Reject every input outside the single-document eager path.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            attention_mask: Additive mask of shape [batch, 1 or heads, sequence, sequence].
            start_pos: First query position; must be zero.
            position_ids: Must be None; positions are the row offsets.
            extra: Remaining keyword arguments of ``forward``.

        Returns:
            The validated ``attention_mask``, unchanged.

        Raises:
            ValueError: If the mask is missing or misshaped.
            NotImplementedError: For packed rows, a nonzero start position, explicit
                position ids or a context-parallel group.
            TypeError: For any other keyword argument.
        """
        batch, sequence = hidden_states.shape[:2]
        expected = f"[{batch}, 1 or heads, {sequence}, {sequence}]"
        if (
            not isinstance(attention_mask, torch.Tensor)
            or attention_mask.ndim != 4
            or not attention_mask.is_floating_point()
        ):
            found = (
                f"{tuple(attention_mask.shape)} {attention_mask.dtype}"
                if isinstance(attention_mask, torch.Tensor)
                else type(attention_mask).__name__
            )
            raise ValueError(
                f"ERNIE HCA requires an explicit 4D additive attention mask of shape {expected}; got {found}"
            )
        if attention_mask.shape[0] != batch or tuple(attention_mask.shape[2:]) != (sequence, sequence):
            raise ValueError(
                f"ERNIE HCA requires an explicit 4D additive attention mask of shape {expected}; "
                f"got {tuple(attention_mask.shape)}"
            )
        if extra.get("packed_seq_ids") is not None:
            raise NotImplementedError("Packed ERNIE HCA inputs are not supported")
        if start_pos != 0:
            raise NotImplementedError(f"Nonzero ERNIE HCA start positions are not supported; got {start_pos}")
        if attention_mask.shape[1] not in (1, self.num_heads):
            raise ValueError(
                f"ERNIE HCA attention mask must have 1 or {self.num_heads} heads; got {tuple(attention_mask.shape)}"
            )
        if position_ids is not None:
            raise NotImplementedError(
                "ERNIE HCA derives positions from the row offsets; explicit position_ids are not supported"
            )
        if extra.get("_dsv4_cp_group") is not None:
            raise NotImplementedError("ERNIE HCA does not support context parallelism")
        unexpected = sorted(set(extra) - {"packed_seq_ids", "_dsv4_cp_group"})
        if unexpected:
            raise TypeError(f"Unexpected ERNIE HCA forward arguments: {unexpected}")
        return attention_mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_embeddings_compress: tuple[torch.Tensor, torch.Tensor] | None = None,
        rotary_compress: nn.Module | None = None,
        start_pos: int = 0,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run windowed attention over local and compressed keys, then mix heads and gate.

        The caller is responsible for the mask contents: a causal mask limited to
        the sliding window. Only its shape and dtype kind are checked.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden]; each row is one
                document starting at position zero.
            position_embeddings: Main-rope cosine and sine, each of shape
                [batch, sequence, rotary_channels]. Unused: a ratio-128 layer rotates
                with the compress tables. Kept for the parent's signature.
            attention_mask: Required additive mask of shape
                [batch, 1 or heads, sequence, sequence] over the uncompressed keys.
            position_embeddings_compress: Required compress-rope cosine and sine, each
                of shape [batch, sequence, rotary_channels], for positions
                ``0..sequence-1``.
            rotary_compress: Required rotary module for the compressed keys.
            start_pos: First query position; must be zero.
            position_ids: Must be None.
            **kwargs: ``packed_seq_ids`` and ``_dsv4_cp_group`` are accepted only as
                None; any other name raises.

        Returns:
            Output of shape [batch, sequence, hidden], and the attention
            probabilities of shape [batch, heads, sequence, sequence + pooled], where
            pooled is the number of compressed keys. The probabilities have the
            activation dtype.

        Raises:
            ValueError: If ``position_embeddings_compress`` or ``rotary_compress`` is None. Every other rejected
                input raises in ``_validate_inputs``: ``ValueError`` for a missing or misshaped mask,
                ``NotImplementedError`` for packed rows, a nonzero start position, explicit position ids or a
                context-parallel group, and ``TypeError`` for any other keyword argument.
        """
        del position_embeddings
        attention_mask = self._validate_inputs(hidden_states, attention_mask, start_pos, position_ids, kwargs)
        if position_embeddings_compress is None or rotary_compress is None:
            raise ValueError("ERNIE HCA requires position_embeddings_compress and rotary_compress")
        batch, sequence = hidden_states.shape[:2]
        cos, sin = position_embeddings_compress

        q_residual = self.q_norm(self.wq_a(hidden_states))
        query = self._project_query(q_residual, cos, sin)
        keys = self.kv_norm(self.wkv(hidden_states)).view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        keys = _apply_partial_rope(keys, cos, sin, self.rope_head_dim, backend=self.backend.rope)

        pooled_keys, _, _, _ = self.compressor(
            hidden_states,
            q_residual=q_residual,
            rotary=rotary_compress,
            position_embeddings=position_embeddings_compress,
            cache=DeepseekV4TrainCache(),
            layer_idx=self.layer_idx,
            start_pos=0,
            # A rank whose row is shorter than one window still builds the compressor
            # graph when another rank of the parameter sync group has a full window.
            enable_hca_fsdp_graph_alignment=self.training,
        )
        pooled = pooled_keys.shape[2]
        keys = torch.cat([keys, pooled_keys], dim=2)
        if pooled > 0:
            attention_mask = torch.cat(
                [attention_mask, _compressed_key_mask(attention_mask, pooled, self.compress_ratio)], dim=-1
            )

        attended, probabilities = eager_attention_with_sink(
            self,
            query,
            keys,
            keys,
            attention_mask,
            dropout=self.attention_dropout if self.training else 0.0,
            scaling=self.scaling,
        )
        # The values carry the key rotation; undo it on the output.
        attended = _apply_partial_rope(
            attended.transpose(1, 2), cos, -sin, self.rope_head_dim, backend=self.backend.rope
        ).transpose(1, 2)

        attended = self.vha(attended)
        grouped = attended.reshape(batch, sequence, -1).view(batch, sequence, self.config.o_groups, -1)
        projected = self.wo_a(grouped).flatten(2)
        gated = projected * torch.sigmoid(self.gate_proj(q_residual))
        return self.wo_b(gated), probabilities

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize inherited and ERNIE parameters after meta materialization.

        ``vha_postmix_V`` is set to zero, so the virtual-head mix starts as the identity.

        Args:
            buffer_device: Forwarded to ``DeepseekV4Attention.init_weights``.
            init_std: Standard deviation of the weight draws: truncated normal for the inherited projections and
                the gate, normal for ``vha_postmix_U``.
        """
        super().init_weights(buffer_device, init_std)
        nn.init.normal_(self.vha_postmix_U, mean=0.0, std=init_std)
        nn.init.zeros_(self.vha_postmix_V)
        nn.init.trunc_normal_(self.gate_proj.weight, mean=0.0, std=init_std)
