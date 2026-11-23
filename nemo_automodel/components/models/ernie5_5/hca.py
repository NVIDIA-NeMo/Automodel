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

Two kinds of input are supported, both with the torch rotary backend. A row
that holds one document starting at position zero runs the eager attention under
an explicit causal sliding-window mask. A packed row, or one rank's shard of it
under context parallelism, runs sparse attention over the window keys and the
complete compressed keys of each query's document; neighbouring ranks exchange
the window keys at the shard boundary and all ranks share the compressed keys. A
KV cache raises an error.
"""

from typing import Any

import torch
import torch.distributed as dist
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
from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
from nemo_automodel.components.models.deepseek_v41.cp import gather_window
from nemo_automodel.components.models.ernie5_5.layers import DENSE_ATTENTION_BACKENDS, shard_start, virtual_head_mix

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
            backend: Backend selection. ``attn`` selects the sparse attention of packed
                rows: ``"cudnn"`` runs the cuDNN kernels, and ``"eager"``, ``"torch"`` and
                ``"sdpa"`` run the PyTorch reference. A row that holds one document runs
                DeepSeek V4's eager sink attention with every value. None selects
                ``"eager"``. The Triton rotary kernel is rejected.
            vha_rank: Rank of the virtual-head mixing factors.

        Raises:
            ValueError: If the layer's compression ratio is not 128 or ``vha_rank``
                is not positive.
            NotImplementedError: If ``backend.attn`` is neither ``"cudnn"`` nor one of
                the three dense names, or the backend or config selects the Triton rotary
                kernel or the fused query-norm rotation.
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
        if backend.attn != "cudnn" and backend.attn not in DENSE_ATTENTION_BACKENDS:
            raise NotImplementedError(
                f"ERNIE HCA supports attn='cudnn' and the dense attention backends {DENSE_ATTENTION_BACKENDS}; "
                f"got attn={backend.attn!r}"
            )
        if backend.rope == "triton":
            raise NotImplementedError("ERNIE HCA does not support the Triton rotary kernel (rope='triton')")
        if config.fuse_qnorm_rope:
            raise NotImplementedError("ERNIE HCA does not support fuse_qnorm_rope")
        super().__init__(config, layer_idx, backend)
        self.vha = Ernie5_5HCAVirtualHeads(config.num_attention_heads, vha_rank)
        self.gate_proj = nn.Linear(config.q_lora_rank, config.o_groups * config.o_lora_rank, bias=False)
        self.cp_group: dist.ProcessGroup | None = None
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
        """Record the context-parallel group whose ranks hold consecutive shards of a row.

        Args:
            cp_mesh: Context-parallel device mesh offered by the parallelizer.
        """
        self.cp_group = cp_mesh.get_group()

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
            NotImplementedError: For a nonzero start position, explicit position ids
                or a context-parallel group.
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
        if extra.get("_dsv4_cp_group") is not None or self.cp_group is not None:
            raise NotImplementedError("ERNIE HCA supports context parallelism for rows with packed_seq_ids only")
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

        Without ``packed_seq_ids`` every row is one document starting at position
        zero, and the caller is responsible for the mask contents: a causal mask
        limited to the sliding window. Only its shape and dtype kind are checked.
        With ``packed_seq_ids`` the row is packed and takes no mask; see
        ``_attend_packed``.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden]. With context
                parallelism, ``sequence`` is this rank's shard of a packed row.
            position_embeddings: Main-rope cosine and sine, each of shape
                [batch, sequence, rotary_channels]. Unused: a ratio-128 layer rotates
                with the compress tables. Kept for the parent's signature.
            attention_mask: Additive mask of shape
                [batch, 1 or heads, sequence, sequence] over the uncompressed keys;
                required for an unpacked row and None for a packed one.
            position_embeddings_compress: Required compress-rope cosine and sine, each
                of shape [batch, sequence, rotary_channels], for the position of every
                token in its document.
            rotary_compress: Required rotary module for the compressed keys.
            start_pos: First query position; must be zero.
            position_ids: Integer tensor of shape [batch, sequence] with the position
                of every token in its document; required for a packed row and None
                for an unpacked one.
            **kwargs: ``packed_seq_ids``, an integer tensor of shape [batch, sequence]
                with the document of every token, selects the packed path.
                ``_dsv4_cp_group`` is accepted only as None; any other name raises.

        Returns:
            Output of shape [batch, sequence, hidden], and the attention
            probabilities of shape [batch, heads, sequence, sequence + pooled], where
            pooled is the number of compressed keys. The probabilities have the
            activation dtype. A packed row returns None in their place.

        Raises:
            ValueError: If ``position_embeddings_compress`` or ``rotary_compress`` is None, or a packed row comes
                with a mask, without ``position_ids``, with a nonzero start position or with another keyword
                argument. Every other rejected input of an unpacked row raises in ``_validate_inputs``:
                ``ValueError`` for a missing or misshaped mask, ``NotImplementedError`` for a nonzero start
                position, explicit position ids or a context-parallel group, and ``TypeError`` for any other
                keyword argument.
        """
        del position_embeddings
        packed_seq_ids = kwargs.get("packed_seq_ids")
        if packed_seq_ids is None:
            attention_mask = self._validate_inputs(hidden_states, attention_mask, start_pos, position_ids, kwargs)
        elif attention_mask is not None or position_ids is None or start_pos != 0 or len(kwargs) != 1:
            raise ValueError("A packed ERNIE HCA row takes position_ids and packed_seq_ids, and no attention mask")
        if position_embeddings_compress is None or rotary_compress is None:
            raise ValueError("ERNIE HCA requires position_embeddings_compress and rotary_compress")
        batch, sequence = hidden_states.shape[:2]
        cos, sin = position_embeddings_compress

        q_residual = self.q_norm(self.wq_a(hidden_states))
        query = self._project_query(q_residual, cos, sin)
        keys = self.kv_norm(self.wkv(hidden_states)).view(batch, sequence, 1, self.head_dim).transpose(1, 2)
        keys = _apply_partial_rope(keys, cos, sin, self.rope_head_dim, backend=self.backend.rope)
        if packed_seq_ids is not None:
            attended = self._attend_packed(
                hidden_states,
                q_residual,
                query,
                keys.squeeze(1),
                position_embeddings_compress,
                rotary_compress,
                position_ids,
                packed_seq_ids,
            )
            return self._project_output(attended, q_residual, cos, sin), None

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
        return self._project_output(attended, q_residual, cos, sin), probabilities

    def _attend_packed(
        self,
        hidden_states: torch.Tensor,
        q_residual: torch.Tensor,
        query: torch.Tensor,
        keys: torch.Tensor,
        rope: tuple[torch.Tensor, torch.Tensor],
        rotary: nn.Module,
        position_ids: torch.Tensor,
        packed_seq_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Attend every query to the window keys and the complete compressed keys of its own document.

        Every document starts at a row offset that is a multiple of the compression ratio, and so does every
        context-parallel shard, so each compressed key pools the tokens of one document. A rank receives the last
        ``sliding_window - 1`` keys of the preceding rank and the compressed keys of all ranks.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            q_residual: Query latent of shape [batch, sequence, q_lora_rank].
            query: Rotated query of shape [batch, heads, sequence, head_dim].
            keys: Rotated keys of shape [batch, sequence, head_dim].
            rope: Compress-rope cosine and sine, each of shape [batch, sequence, rotary_channels].
            rotary: Rotary module for the compressed keys.
            position_ids: Integer tensor of shape [batch, sequence] with the position of every token in its
                document.
            packed_seq_ids: Integer tensor of shape [batch, sequence] with the document of every token, counted
                from one along the row; zero marks padding.

        Returns:
            Attention output of shape [batch, sequence, heads, head_dim].
        """
        sequence = position_ids.shape[1]
        keys, first_key = gather_window(keys, self.cp_group, window_size=self.sliding_window)
        pooled, _, pooled_documents, pooled_windows = self.compressor(
            hidden_states,
            q_residual=q_residual,
            rotary=rotary,
            position_embeddings=rope,
            cache=DeepseekV4TrainCache(),
            layer_idx=self.layer_idx,
            start_pos=0,
            position_ids=position_ids,
            packed_seq_ids=packed_seq_ids,
            cp_group=self.cp_group,
        )
        offsets = shard_start(self.cp_group, sequence) + torch.arange(sequence, device=keys.device)
        real = packed_seq_ids > 0
        # A padding token attends to itself only, so every query has a key.
        first = torch.where(real, torch.maximum(offsets - self.sliding_window + 1, offsets - position_ids), offsets)
        window = first.unsqueeze(-1) + torch.arange(self.sliding_window, device=keys.device)
        window = torch.where(window <= offsets.view(1, -1, 1), window - first_key, -1)
        visible = (
            real.unsqueeze(-1)
            & (packed_seq_ids.unsqueeze(-1) == pooled_documents.unsqueeze(1))
            & (pooled_windows.unsqueeze(1) < ((position_ids + 1) // self.compress_ratio).unsqueeze(-1))
        )
        compressed = torch.where(visible, torch.arange(pooled.shape[2], device=keys.device) + keys.shape[1], -1)
        return dsv4_sparse_attention(
            query.transpose(1, 2).contiguous(),
            torch.cat((keys, pooled.squeeze(1)), dim=1),
            self.sinks_param(query),
            torch.cat((window, compressed), dim=-1),
            self.scaling,
            backend="cudnn" if self.backend.attn == "cudnn" else "torch",
            all_rows_nonempty=True,
        )

    def _project_output(
        self, attended: torch.Tensor, q_residual: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        """Undo the key rotation, mix the heads and apply the gated grouped output projection.

        Args:
            attended: Attention output of shape [batch, sequence, heads, head_dim].
            q_residual: Query latent of shape [batch, sequence, q_lora_rank].
            cos: Compress-rope cosine of shape [batch, sequence, rotary_channels].
            sin: Compress-rope sine of shape [batch, sequence, rotary_channels].

        Returns:
            Output of shape [batch, sequence, hidden].
        """
        batch, sequence = attended.shape[:2]
        # The values carry the key rotation; undo it on the output.
        attended = _apply_partial_rope(
            attended.transpose(1, 2), cos, -sin, self.rope_head_dim, backend=self.backend.rope
        ).transpose(1, 2)
        attended = self.vha(attended)
        grouped = attended.reshape(batch, sequence, -1).view(batch, sequence, self.config.o_groups, -1)
        projected = self.wo_a(grouped).flatten(2)
        gated = projected * torch.sigmoid(self.gate_proj(q_residual))
        return self.wo_b(gated)

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
