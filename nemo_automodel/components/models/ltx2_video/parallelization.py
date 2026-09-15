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

"""FSDP2 strategy for the diffusers ``LTX2VideoTransformer3DModel``."""

from __future__ import annotations

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models.hunyuan_video15.parallelization import HunyuanParallelizationStrategy


class LTX2ParallelizationStrategy(HunyuanParallelizationStrategy):
    """Parallelization strategy for the LTX-2 video+audio transformer.

    ``LTX2VideoTransformer3DModel`` exposes its layers as ``transformer_blocks``
    but names the attention/FFN submodules ``attn1``/``attn2``/``ff``, which the
    Default strategy's submodule-level activation checkpointing does not
    recognize — leaving attention and MLP activations un-checkpointed and OOM-ing
    on the long combined video+audio token sequence. Wrapping each whole block
    (as the HunyuanVideo strategy does) restores the expected memory profile.
    """


class LTX2VideoTransformer3DModel:
    """Contract for the diffusers ``LTX2VideoTransformer3DModel``; bound by the diffusion pipeline before sharding."""

    parallel_spec: ParallelSpec = ParallelSpec(strategy=LTX2ParallelizationStrategy())
