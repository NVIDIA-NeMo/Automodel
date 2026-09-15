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

"""Parallelization contract for the remote-code ``NemotronFlashForCausalLM``."""

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models.llama.parallelization import LLAMA_SEQUENCE_PARALLEL_PLAN, LLAMA_TP_PLAN


class NemotronFlashForCausalLM:
    """Contract for the remote-code ``NemotronFlashForCausalLM``: the Llama plan with a sharded-logits constraint.

    Its forward computes ``logits / self.lm_head.weight.norm(p=2, dim=1)``, so the logits and the weight norm
    must either both be plain tensors or both use the same vocab-sharded DTensor layout. A plan entry giving
    ``lm_head`` a replicated output would mix a plain logits tensor with a sharded weight norm, so such an entry
    (from any plan source) is dropped and the head left replicated; a vocab-sharded output keeps both operands
    aligned and is required under FSDP+TP.
    """

    parallel_spec: ParallelSpec = ParallelSpec(
        tp_plan=LLAMA_TP_PLAN,
        sequence_parallel_plan=LLAMA_SEQUENCE_PARALLEL_PLAN,
        sharded_output_only=("lm_head", "language_model.lm_head"),
    )
