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

"""SGLang adapter for the EAGLE-3 target backend.

A thin engine adapter on top of the engine-agnostic contract in
:mod:`nemo_automodel.components.speculative.eagle.target_runner`: it builds a
SGLang runner and wraps it in :class:`RunnerEagle3TargetModel`. SGLang is the
fastest serving path for mainstream architectures, so a remote target server
(:mod:`nemo_automodel.components.speculative.serve_target`) can hold the target
on dedicated GPUs while the draft trains elsewhere.

All supervision-contract logic (shift / aux-concatenation semantics, aux-layer
defaulting, embedding access) lives in ``target_runner`` and is shared with any
future engine adapter (e.g. vLLM). This module owns only SGLang construction,
and the SGLang-internal forward is isolated further in
:mod:`nemo_automodel.components.speculative.eagle.sglang_runner`, imported lazily
by :meth:`SGLangEagle3TargetModel.from_pretrained` so importing this module never
pulls in SGLang.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from nemo_automodel.components.speculative.eagle.target_runner import (
    RunnerEagle3TargetModel,
    TargetRunner,
)

#: Backwards-compatible alias. The runner surface is now engine-agnostic and
#: lives in ``target_runner`` as :class:`TargetRunner`; kept here so existing
#: imports of ``SGLangRunnerProtocol`` keep resolving.
SGLangRunnerProtocol = TargetRunner


class SGLangEagle3TargetModel(RunnerEagle3TargetModel):
    """EAGLE-3 target backend whose runner is SGLang.

    Adds only SGLang construction; the supervision contract is inherited from
    :class:`RunnerEagle3TargetModel`.
    """

    @classmethod
    def from_pretrained(  # pragma: no cover - requires GPU + SGLang
        cls,
        model_path: str,
        *,
        aux_layer_ids: Optional[Sequence[int]] = None,
        dtype: Optional[torch.dtype] = None,
        tp_size: int = 1,
        trust_remote_code: bool = False,
        **sglang_kwargs,
    ) -> "SGLangEagle3TargetModel":
        """Build a SGLang runner for ``model_path`` and wrap it as a target backend.

        SGLang is imported here (not at module load) so this module stays
        importable in environments without SGLang; ``sglang_kwargs`` are passed
        through to SGLang's ``ServerArgs`` for endpoint / parallelism tuning.
        """
        from nemo_automodel.components.speculative.eagle.sglang_runner import SGLangTargetRunner

        runner = SGLangTargetRunner.build(
            model_path,
            dtype=dtype,
            tp_size=tp_size,
            trust_remote_code=trust_remote_code,
            **sglang_kwargs,
        )
        return cls(runner, aux_layer_ids=aux_layer_ids)
