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

"""Typed model-owned contracts for pipeline parallelism.

Stage-boundary tensor metadata is intentionally absent from this contract.
``torch.distributed.pipelining.PipelineStage`` infers it at runtime from the
real inputs handed to ``schedule.step()``, propagating each stage's observed
output metadata to the next stage. Runtime inference observes the actual dtype,
context-parallel sharding, and tuple arity of a stage boundary, so models no
longer declare it.
"""

from enum import Enum

import torch
import torch.nn as nn

PP_MEDIA_INDEX_KEY = "pp_media_index"


class PipelineForwardStyle(Enum):
    """How a model part obtains its pipeline-aware forward implementation."""

    PATCH_HF = "patch_hf"
    MODEL = "model"


class PipelineModelMixin:
    """Explicit pipeline behavior contract for model-owned specialization.

    Models inherit this mixin only when they need to preserve their own forward
    or customize stage ownership. The generic HuggingFace pipeline path does not
    require the mixin.
    """

    pipeline_forward_style = PipelineForwardStyle.PATCH_HF

    #: Whether the final model part can skip its vocabulary projection and emit
    #: hidden states for a fused loss such as ``FusedLinearCrossEntropy``.
    #: Models keeping their own forward must honor ``_pp_return_hidden_states``
    #: to opt in.
    pipeline_supports_hidden_state_output: bool = False

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


def pp_media_chunk(module: nn.Module, name: str, pp_media_index: torch.Tensor | None):
    """Return the media tensor staged for the microbatch currently being executed.

    Pipeline schedules may execute microbatches in a schedule-defined order and
    additionally run one probe forward per stage for runtime shape inference, so
    the executing microbatch cannot be tracked with a cursor. Instead the batch
    carries ``pp_media_index``: an int64 tensor of shape [batch] whose entry for
    every sample is that sample's microbatch index. ``torch.distributed.pipelining``
    splits it along the batch axis in lockstep with the model input, so each
    forward receives a [microbatch] slice identifying its own chunk.

    Args:
        module: Stage-0 module holding ``_pp_media_chunks``, a mapping of media
            name to one tensor per microbatch.
        name: Media key to look up, e.g. ``"pixel_values"``.
        pp_media_index: Tensor of shape [microbatch] holding this microbatch's
            index, or None when the batch carries no media.

    Returns:
        The staged tensor for this microbatch, or None when this microbatch
        carries no media for that key. A microbatch whose samples are all
        text-only is staged as an empty tensor so the per-microbatch chunk list
        stays positionally aligned; it is reported here as None so callers skip
        their media path instead of invoking an encoder on zero inputs.
    """
    chunks = getattr(module, "_pp_media_chunks", None)
    if not chunks or pp_media_index is None:
        return None
    staged = chunks.get(name)
    if staged is None:
        return None
    index = int(pp_media_index.reshape(-1)[0])
    if index < 0 or index >= len(staged):
        raise IndexError(
            f"pp_media_index {index} is out of range for {len(staged)} staged '{name}' chunks; "
            "the media staging and the schedule disagree about the microbatch count."
        )
    chunk = staged[index]
    if chunk is None or (isinstance(chunk, torch.Tensor) and chunk.numel() == 0):
        return None
    return chunk
