# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""MiniMax M3's MSA attention adapter: packed documents through the SM100 sparse kernels."""

from typing import Any

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.gpt_oss.rope_utils import apply_rotary_emb_qk
from nemo_automodel.components.models.minimax_m3_vl._msa import (
    _MSAFlatAttention,
    _MSAPackedLayout,
    _reject_unsupported_msa_configuration,
    _validate_msa_topology,
)
from nemo_automodel.components.models.minimax_m3_vl.kernels.msa_forward_select import (
    select_blocks,
    warm_selection_variants,
)
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Attention


class MiniMaxM3MSAAttention(MiniMaxM3Attention):
    """A sparse attention layer that runs MSA's SM100 kernels on compact packed documents.

    Same interface as :class:`MiniMaxM3Attention`, and the third adapter at the seam
    ``Block.__init__`` already owns (alongside the plain and CP-aware ones). Everything MSA needs
    that the others do not -- the compact [tokens, ...] layout, non-differentiable block selection
    into ``q2k``, and the fused forward/backward -- lives behind this ``forward``.

    Rejections are layered by when the answer can change: backend and topology at construction,
    context parallelism at ``setup_cp_attention``, and per-step runtime arguments once in
    ``MiniMaxM3TextModel.forward``. Nothing is re-checked per layer.
    """

    def __init__(self, config: Any, backend: BackendConfig, *, is_sparse_attention_layer: bool = True) -> None:
        super().__init__(config, backend, is_sparse_attention_layer=is_sparse_attention_layer)
        _reject_unsupported_msa_configuration(backend)
        _validate_msa_topology(
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            num_index_heads=self.indexer.num_index_heads,
            block_size=self.indexer.block_size,
            topk_blocks=self.indexer.topk_blocks,
            index_head_dim=self.indexer.index_head_dim,
            attention_dropout=float(getattr(config, "attention_dropout", 0.0) or 0.0),
            score_type=self.indexer.score_type,
        )
        self._msa_attn = _MSAFlatAttention(self.head_dim**-0.5)

    def _selection_arguments(self) -> dict[str, int]:
        """Return the indexer settings block selection is keyed on.

        The scorer warm-up and every per-layer selection call must agree on all five: the warm-up
        otherwise compiles variants production never reaches, and a microbatch scored twice with
        different values would silently reuse the first plan.

        Returns:
            The keyword arguments ``select_blocks`` and ``warm_selection_variants`` both take.
        """
        indexer = self.indexer
        return {
            "index_heads": indexer.num_index_heads,
            "block_size": indexer.block_size,
            "topk_blocks": indexer.topk_blocks,
            "init_blocks": indexer.init_blocks,
            "local_blocks": indexer.local_blocks,
        }

    def _initialize_attention(self, softmax_scale: float) -> tuple[None, None]:
        """Own no generic attention backend; MSA runs its own SM100 kernels.

        Building one anyway would import TransformerEngine before any MSA rejection could fire, so a
        misconfigured layer would report a missing optional dependency instead of its real problem.
        A leftover ``DotProductAttention`` would also send ``apply_cp`` down TE's context-parallel
        branch (moe/parallelizer.py:983) instead of the ``setup_cp_attention`` rejection below.

        Args:
            softmax_scale: Unused; this layer passes its own scale to the MSA kernels.

        Returns:
            ``(None, None)``.
        """
        return None, None

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize the projections and, once per process, compile every scorer variant.

        Args:
            buffer_device: The device weights are materialized on. Construction happens on the meta
                device, so this is the first point that knows where the scorer will run.
            init_std: Standard deviation of the truncated-normal projection initialization.
        """
        super().init_weights(buffer_device, init_std)
        warm_selection_variants(buffer_device, index_dim=self.indexer.index_head_dim, **self._selection_arguments())

    def setup_cp_attention(self, cp_mesh: Any) -> None:
        """Reject context parallelism at setup; MSA has no CP-aware selection or kernel.

        Args:
            cp_mesh: The context-parallel submesh ``apply_cp`` would install.

        Raises:
            NotImplementedError: Always. Defining this method is load-bearing: the parallelizer
                dispatches on ``hasattr(self_attn, "setup_cp_attention")`` and otherwise only logs
                a warning, leaving CP silently unapplied.
        """
        raise NotImplementedError(
            "MiniMax M3 backend.sparse_attn='msa' requires cp_size=1; disable context parallelism "
            "or set backend.sparse_attn='generic'."
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        freqs_cis: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        _msa_layout: _MSAPackedLayout,
        **attn_kwargs: Any,
    ) -> torch.Tensor:
        """Map hidden states and rotary tables to an output of the same layout as ``x``.

        Args:
            x: Tensor of shape [batch, sequence, hidden], packed to [tokens, hidden] here and
                unpacked again before returning.
            freqs_cis: Rotary table of shape [batch, sequence, rotary_dim], packed alongside ``x``.
            attention_mask: Ignored. Document isolation travels as ``_msa_layout``; the model sets
                the mask to None for every layer once MSA is on.
            _msa_layout: Packed-microbatch layout owned by ``MiniMaxM3TextModel``; required.
            **attn_kwargs: Backend arguments, none of which this path reads.

        Returns:
            Tensor of shape [batch, sequence, hidden]; padding rows are zero.
        """
        x = _msa_layout.pack(x)
        freqs_cis = _msa_layout.pack(freqs_cis)
        q, k, v = self._project_qkv(x)
        with torch.no_grad():
            index_q, index_k = self.indexer._project_qk(x, freqs_cis=freqs_cis, cp_size=1, cp_rank=0)
            q2k = select_blocks(_msa_layout, index_q, index_k, **self._selection_arguments())
        q, k = apply_rotary_emb_qk(q, k, freqs_cis, format="thd", rope_fusion=self._rope_fusion)
        out = self._msa_attn(q, k, v, q2k, layout=_msa_layout)
        return _msa_layout.unpack(self.o_proj(out.flatten(1)))
