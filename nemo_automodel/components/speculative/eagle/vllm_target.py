# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""vLLM adapter for the EAGLE-3 target backend.

A thin engine adapter on top of the engine-agnostic contract in
:mod:`nemo_automodel.components.speculative.eagle.target_runner`: it builds a
vLLM runner and wraps it in :class:`RunnerEagle3TargetModel`. It mirrors
:mod:`nemo_automodel.components.speculative.eagle.sglang_target` exactly, only
swapping the engine, so the supervision contract (shift / aux-concatenation
semantics, aux-layer defaulting, embedding access) is shared and a vLLM run is
numerically equivalent to the co-located HuggingFace one.

vLLM is imported lazily (only inside :meth:`VLLMEagle3TargetModel.from_pretrained`),
and the vLLM-internal forward is isolated further in
:mod:`nemo_automodel.components.speculative.eagle.vllm_runner`, so importing this
module never pulls in vLLM.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from nemo_automodel.components.speculative.eagle.target_runner import RunnerEagle3TargetModel


class VLLMEagle3TargetModel(RunnerEagle3TargetModel):
    """EAGLE-3 target backend whose runner is vLLM.

    Adds only vLLM construction; the supervision contract is inherited from
    :class:`RunnerEagle3TargetModel`.
    """

    @classmethod
    def from_pretrained(  # pragma: no cover - requires GPU + vLLM
        cls,
        model_path: str,
        *,
        aux_layer_ids: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        **vllm_kwargs,
    ) -> "VLLMEagle3TargetModel":
        """Build a vLLM runner for ``model_path`` and wrap it as a target backend.

        vLLM is imported here (not at module load) so this module stays
        importable in environments without vLLM; ``vllm_kwargs`` are passed
        through to ``vllm.LLM`` for endpoint / parallelism / memory tuning.
        """
        from nemo_automodel.components.speculative.eagle.vllm_runner import VLLMTargetRunner

        runner = VLLMTargetRunner.build(
            model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=trust_remote_code,
            **vllm_kwargs,
        )
        return cls(runner, aux_layer_ids=aux_layer_ids)
