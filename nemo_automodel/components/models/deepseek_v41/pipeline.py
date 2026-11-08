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

"""Model-owned pipeline partition dependencies and CSA2 tensor wire layouts."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .attention import DeepseekV41AttentionState
from .config import DeepseekV41Config


@dataclass(frozen=True)
class DeepseekV41BackboneOutput:
    """Backbone values before or after the stage-local final norm.

    Attributes:
        hidden_states: [batch, local_sequence, streams, hidden] on an intermediate
            stage, or normalized [batch, local_sequence, hidden] on the last stage.
        pre_mix: FP32 coefficients [batch, local_sequence, streams]. They remain
            differentiable across a pipeline boundary.
        attention_state: Immutable CSA2 values described by DeepseekV41AttentionState.
        residual_streams: Optional pre-block states, each
            [batch, local_sequence, streams, hidden], for parity diagnostics.
    """

    hidden_states: torch.Tensor
    pre_mix: torch.Tensor
    residual_streams: tuple[torch.Tensor, ...] | None
    attention_state: DeepseekV41AttentionState


class DeepseekV41PipelineMixin:
    """Preserve model dependencies across the configured contiguous stages."""

    config: DeepseekV41Config
    model: nn.Module
    lm_head: nn.Module | None
    _pp_keep_self_forward = True
    _pp_return_hidden_states_supported = True
    _pp_return_hidden_states = False

    def customize_pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module | None = None,
    ) -> list[list[str]]:
        """Retain the supplied layer split and attach stage-owned dependencies.

        Manual module_fqns_per_model_part uses the framework's existing
        complete-module-list contract. This hook augments framework-generated
        splits; it does not choose or move layer boundaries.
        """
        stages = [list(modules) for modules in module_names_per_stage]
        for modules in stages:
            if self.model.engram_hash is not None:
                name = f"{layers_prefix}engram_hash"
                if name not in modules:
                    modules.append(name)
        if self.model.vision is not None:
            for name in ("vision", "aligner", "image_start", "image_end", "image_newline"):
                fqn = f"{layers_prefix}{name}"
                if fqn not in stages[0]:
                    stages[0].append(fqn)
        return stages

    def _pipeline_layer_bounds(self) -> tuple[int, int]:
        """Validate a nonempty contiguous stage and return its half-open layer range."""
        layers = list(map(int, self.model.layers))
        if not layers or layers != list(range(layers[0], layers[-1] + 1)):
            raise ValueError("DeepSeek V4.1 pipeline stages require nonempty contiguous language layers")
        start, end = layers[0], layers[-1] + 1
        if (self.model.embed_tokens is not None) != (start == 0):
            raise ValueError("Only the stage containing layer 0 must own token embeddings")
        last = end == self.config.text_config.num_hidden_layers
        if (self.lm_head is not None) != last or (self.model.norm is not None) != last:
            raise ValueError("Only the final language stage must own the final norm and LM head")
        if any(layer.engram is not None for layer in self.model.layers.values()) and self.model.engram_hash is None:
            raise ValueError("A stage containing Engram must retain model.engram_hash in its module list")
        return start, end

    def _shared_state_ratio(self, boundary: int) -> int:
        """Return the live KV ratio at a supplied cut, or zero if its state is dead.

        This determines the wire contract, not the partition. Window-only layers
        preserve a lineage; a new Full layer replaces it before any consumer.
        """
        text = self.config.text_config
        for layer in range(boundary, text.num_hidden_layers):
            if text.compress_ratios[layer]:
                return 0 if layer in text.kv_source_layer_ids else text.compress_ratios[layer]
        return 0

    def _shared_state_metas(
        self, boundary: int, batch: int, sequence: int, dtype: torch.dtype
    ) -> tuple[torch.Tensor, ...]:
        """Describe live state tensors using the layouts of _pack_pipeline_state."""
        ratio = self._shared_state_ratio(boundary)
        if not ratio:
            return ()
        text = self.config.text_config
        cp_size = 1 if self.cp_mesh is None else self.cp_mesh.size()
        width = sequence * cp_size // ratio
        byte_width = text.index_head_dim * torch.empty((), dtype=dtype).element_size()
        metas = (
            torch.empty(batch, width, text.head_dim, dtype=dtype, device="meta"),
            torch.empty(batch, width, byte_width, dtype=torch.uint8, device="meta"),
            torch.empty(batch, sequence, min(text.index_topk, width), dtype=torch.long, device="meta"),
            torch.empty(batch, width, dtype=torch.bool, device="meta"),
            torch.empty(batch, width, dtype=torch.long, device="meta"),
        )
        source = max(layer for layer in text.kv_source_layer_ids if layer < boundary)
        if source <= text.candidate_source_layer_id < boundary:
            metas += (torch.empty(batch, sequence, width, dtype=torch.bool, device="meta"),)
        return metas

    def _pack_pipeline_state(self, state: DeepseekV41AttentionState, boundary: int) -> tuple[torch.Tensor, ...]:
        """Expose immutable live CSA2 tensors to native PP autograd.

        Args:
            state: Values with layouts in DeepseekV41AttentionState. Compressed
                KV keeps its gradient history; frozen index keys use a byte view
                so PP does not allocate or expect meaningless floating gradients.
            boundary: Index of the next language layer.

        Returns:
            Empty tuple at a dead-state boundary. Otherwise: differentiable KV
            [batch, global_compressed, head_dim], byte-view index keys
            [batch, global_compressed, index_head_dim * element_size], int64
            top-k [batch, local_sequence, topk], bool validity and int64 document
            IDs [batch, global_compressed], then an optional bool candidate mask
            [batch, local_sequence, global_compressed]. Document IDs are -1 in
            unpacked mode. Tensors may alias state; consumers must not mutate them.
            The derived backend sparse plan is rebuilt on the receiving stage.
        """
        ratio = self._shared_state_ratio(boundary)
        if not ratio:
            return ()
        if (
            state.compression_ratio != ratio
            or state.compressed_kv is None
            or state.index_keys is None
            or state.topk_indices is None
            or state.compressed_valid is None
        ):
            raise ValueError("Pipeline boundary is missing its live CSA2 source state")
        documents = state.compressed_seq_ids
        if documents is None:
            documents = torch.full_like(state.compressed_valid, -1, dtype=torch.long)
        tensors = (
            state.compressed_kv,
            state.index_keys.detach().contiguous().view(torch.uint8),
            state.topk_indices,
            state.compressed_valid,
            documents.to(torch.long),
        )
        if state.candidates is not None:
            tensors += (state.candidates,)
        return tuple(t.contiguous() for t in tensors)

    def _unpack_pipeline_state(
        self,
        tensors: tuple[torch.Tensor, ...],
        *,
        boundary: int,
        hidden_states: torch.Tensor,
        packed: bool,
    ) -> DeepseekV41AttentionState:
        """Restore stage-local immutable CSA2 state without detaching shared KV.

        Args:
            tensors: Wire tensors documented by _pack_pipeline_state.
            boundary: Index of this stage's first language layer.
            hidden_states: Tensor [batch, local_sequence, streams, hidden] whose
                device and compute dtype also define the incoming KV contract.
            packed: Whether token metadata contains packed document IDs.

        Returns:
            CSA2 state with layouts in DeepseekV41AttentionState. Its tensors
            alias the received payload and retain its KV gradient path.
        """
        expected = self._shared_state_metas(boundary, *hidden_states.shape[:2], hidden_states.dtype)
        if len(tensors) != len(expected) or any(
            t.shape != meta.shape or t.dtype != meta.dtype or t.device != hidden_states.device
            for t, meta in zip(tensors, expected)
        ):
            raise ValueError("Pipeline CSA2 payload does not match this stage's source, shape or dtype")
        if not tensors:
            return DeepseekV41AttentionState()
        kv, keys, indices, valid, documents, *candidates = tensors
        return DeepseekV41AttentionState(
            compressed_kv=kv,
            index_keys=keys.view(hidden_states.dtype),
            topk_indices=indices,
            compressed_valid=valid,
            compression_ratio=self._shared_state_ratio(boundary),
            compressed_seq_ids=documents if packed else None,
            candidates=candidates[0] if candidates else None,
        )

    def get_pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
        """Describe local CP-shard tensors sent between pipeline stages.

        Returns:
            Input/output meta tensors. The intermediate payload consists of
            hidden streams [microbatch, local_sequence, streams, hidden] in the
            compute dtype, FP32 pre-mix [microbatch, local_sequence, streams],
            and int64 token metadata [microbatch, local_sequence, 5]. Metadata
            columns are token ID, RoPE position, validity, document ID, and
            vision-token type. A document-ID column of -1 preserves unpacked mode.
            Live CSA2 tensors from _pack_pipeline_state follow this prefix when
            the next stage needs the current source. Final output is hidden [microbatch, local_sequence,
            hidden] for a linear loss, or FP32 logits [microbatch, local_sequence, vocab].
            seq_len is already local; do not divide it by CP again.
        """
        start, end = self._pipeline_layer_bounds()
        text = self.config.text_config
        shape = (microbatch_size, seq_len)
        payload = (
            torch.empty(*shape, text.hc_mult, text.hidden_size, device="meta", dtype=dtype),
            torch.empty(*shape, text.hc_mult, device="meta", dtype=torch.float32),
            torch.empty(*shape, 5, device="meta", dtype=torch.long),
        )
        inputs = (
            (torch.empty(*shape, device="meta", dtype=torch.long),)
            if is_first
            else payload + self._shared_state_metas(start, microbatch_size, seq_len, dtype)
        )
        if self.lm_head is None:
            outputs = payload + self._shared_state_metas(end, microbatch_size, seq_len, dtype)
        else:
            width = text.hidden_size if self._pp_return_hidden_states else text.vocab_size
            output_dtype = dtype if self._pp_return_hidden_states else torch.float32
            outputs = (torch.empty(*shape, width, device="meta", dtype=output_dtype),)
        return inputs, outputs
