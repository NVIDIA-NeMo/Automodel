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

"""Gemma3 tensor-parallel plan."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ParallelStyle, SequenceParallel

from nemo_automodel.components.models.common.tp_plan import DecoderTPPaths, RotaryEmbedParallel, decoder_tp_plan

_TEXT_TP_PATHS = DecoderTPPaths(
    embedding="model.embed_tokens",
    layer="model.layers.*",
    norm="model.norm",
    output_head="lm_head",
)
_VLM_TP_PATHS = DecoderTPPaths(
    embedding="model.language_model.embed_tokens",
    layer="model.language_model.layers.*",
    norm="model.language_model.norm",
    output_head="lm_head",
)


def get_tp_plan(model, *, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Return the Gemma3 text-decoder TP plan for text and multimodal variants."""
    inner_model = getattr(model, "model", None)
    paths = _VLM_TP_PATHS if getattr(inner_model, "language_model", None) is not None else _TEXT_TP_PATHS
    plan = decoder_tp_plan(paths=paths, sequence_parallel=sequence_parallel)
    if not sequence_parallel:
        return plan

    layer = paths.layer
    plan.update(
        {
            f"{paths.norm.rsplit('.', 1)[0]}.rotary_emb": RotaryEmbedParallel(use_local_output=True),
            f"{paths.norm.rsplit('.', 1)[0]}.rotary_emb_local": RotaryEmbedParallel(use_local_output=True),
            f"{layer}.pre_feedforward_layernorm": SequenceParallel(),
            f"{layer}.post_feedforward_layernorm": SequenceParallel(),
        }
    )
    return plan
