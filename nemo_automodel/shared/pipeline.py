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

"""Typed model-owned contracts for pipeline parallelism."""

from enum import Enum

import torch
import torch.nn as nn

StageMetadata = tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]


class PipelineForwardStyle(Enum):
    """How a model part obtains its pipeline-aware forward implementation."""

    PATCH_HF = "patch_hf"
    MODEL = "model"


class PipelineModelMixin:
    """Explicit pipeline behavior contract for model-owned specialization.

    Models inherit this mixin only when they need to preserve their own forward,
    customize stage ownership, or declare nonstandard stage-boundary tensors.
    The generic HuggingFace pipeline path does not require the mixin.
    """

    pipeline_forward_style = PipelineForwardStyle.PATCH_HF

    def pipeline_stage_modules(
        self,
        module_names_per_stage: list[list[str]],
        *,
        layers_prefix: str,
        text_model: nn.Module,
    ) -> list[list[str]]:
        """Return the module FQNs owned by each global pipeline stage.

        Args:
            module_names_per_stage: Default module FQNs for every global stage.
            layers_prefix: Fully qualified prefix containing transformer layers.
            text_model: Text-model module whose layers are being partitioned.

        Returns:
            Module FQNs for every global pipeline stage.
        """
        del layers_prefix, text_model
        return module_names_per_stage

    def pipeline_stage_metas(
        self,
        *,
        is_first: bool,
        microbatch_size: int,
        seq_len: int,
        dtype: torch.dtype,
    ) -> StageMetadata | None:
        """Return model-specific stage-boundary tensor metadata when required.

        Args:
            is_first: Whether this model part owns the first global stage.
            microbatch_size: Local samples per pipeline microbatch.
            seq_len: Full input sequence length before any context sharding.
            dtype: Activation dtype used at the pipeline boundary.

        Returns:
            ``None`` for the standard causal-LM contract, otherwise input and
            output tuples of meta tensors. Tensor ranks are model-defined; their
            leading axes represent local microbatch and sequence or token axes.
        """
        del is_first, microbatch_size, seq_len, dtype
        return None


def causal_lm_stage_metas(
    *,
    is_first: bool,
    has_lm_head: bool,
    emits_hidden_states: bool,
    microbatch_size: int,
    input_seq_len: int,
    output_seq_len: int,
    hidden_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    logits_dtype: torch.dtype | None = None,
) -> StageMetadata:
    """Construct the common causal-LM pipeline boundary metadata.

    Args:
        is_first: Whether the stage consumes token IDs.
        has_lm_head: Whether the stage owns the vocabulary projection.
        emits_hidden_states: Whether a final stage intentionally bypasses its
            vocabulary projection for a fused loss.
        microbatch_size: Local samples per pipeline microbatch.
        input_seq_len: Sequence length consumed by the first stage.
        output_seq_len: Per-rank sequence length transferred between stages.
        hidden_size: Hidden-state width.
        vocab_size: Unpadded vocabulary width emitted by the LM head.
        dtype: Hidden-state and logits dtype.
        logits_dtype: Optional logits dtype when the LM head intentionally uses
            different precision from hidden activations.

    Returns:
        Input and output meta tensors. The first-stage input has shape
        [microbatch, input_sequence]; other inputs and hidden outputs have shape
        [microbatch, output_sequence, hidden], and logits have shape
        [microbatch, output_sequence, vocab].
    """
    hidden_shape = (microbatch_size, output_seq_len, hidden_size)
    if is_first:
        inputs = (torch.empty(microbatch_size, input_seq_len, device="meta", dtype=torch.long),)
    else:
        inputs = (torch.empty(*hidden_shape, device="meta", dtype=dtype),)
    output_width = hidden_size if not has_lm_head or emits_hidden_states else vocab_size
    output_dtype = logits_dtype if has_lm_head and not emits_hidden_states and logits_dtype is not None else dtype
    outputs = (torch.empty(microbatch_size, output_seq_len, output_width, device="meta", dtype=output_dtype),)
    return inputs, outputs


def context_parallel_seq_len(seq_len: int, cp_size: int) -> int:
    """Return the padded per-rank sequence length for round-robin CP sharding."""
    return (seq_len + (-seq_len) % (2 * cp_size)) // cp_size if cp_size > 1 else seq_len
