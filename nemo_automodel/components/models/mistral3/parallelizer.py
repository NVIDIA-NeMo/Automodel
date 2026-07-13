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

"""Ministral3 and Mistral3 VLM tensor-parallel plans."""

from __future__ import annotations

from torch.distributed.tensor.parallel import ParallelStyle

from nemo_automodel.components.models.common.tp_plan import DecoderTPPaths, decoder_tp_plan

_TP_PATHS = DecoderTPPaths(
    embedding="model.embed_tokens",
    layer="model.layers.*",
    norm="model.norm",
    output_head="lm_head",
)


def get_tp_plan(model, *, sequence_parallel: bool = False) -> dict[str, ParallelStyle]:
    """Return a plan for a Ministral3 decoder or its Mistral3 VLM wrapper."""
    del model
    return decoder_tp_plan(paths=_TP_PATHS, sequence_parallel=sequence_parallel)
