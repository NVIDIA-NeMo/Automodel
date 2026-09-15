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

"""Parallelization contract for the transformers ``GPT2LMHeadModel``.

NeMo AutoModel does not re-implement GPT-2 for fine-tuning; the nanoGPT-style pretraining model in
``nanogpt.py`` is unrelated and keeps its blocks at ``h``, which the generic layer heuristic already
finds. The transformers model keeps them under ``transformer.h``.
"""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec

GPT2_PARALLEL_SPEC = ParallelSpec(layer_groups={"language": ("transformer.h",)})


class GPT2LMHeadModel:
    """Contract for the transformers ``GPT2LMHeadModel``; bound by the loader onto its wrapper class."""

    parallel_spec: ParallelSpec = GPT2_PARALLEL_SPEC
