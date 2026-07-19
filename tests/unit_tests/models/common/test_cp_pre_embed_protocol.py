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

"""Signature contract for the ``model(_pre_embed_only=True, ...)`` CP pre-embed call.

``prepare_cp_forward`` invokes hook-aware models through ``__call__`` with only
keyword arguments (``_pre_embed_only=True, _cp_batch=batch, num_chunks=n``) so
FSDP2 pre-forward hooks run. Any required positional parameter on ``forward``
breaks that call at signature-binding time, before the model's
``_pre_embed_only`` branch can run (DeepseekV4 crashed this way under pp4/cp8:
``TypeError: forward() missing 1 required positional argument: 'input_ids'``).
Every model that answers the pre-embed protocol must therefore accept a
kwargs-only ``forward`` call.
"""

import inspect

import pytest

PRE_EMBED_MODELS = [
    ("nemo_automodel.components.models.deepseek_v4.model", "DeepseekV4ForCausalLM"),
    ("nemo_automodel.components.models.glm_moe_dsa.model", "GlmMoeDsaForCausalLM"),
    ("nemo_automodel.components.models.gemma4_moe.model", "Gemma4ForConditionalGeneration"),
    ("nemo_automodel.components.models.step3p7.model", "Step3p7ForConditionalGeneration"),
    ("nemo_automodel.components.models.qwen3_5.model", "Qwen3_5ForConditionalGeneration"),
    ("nemo_automodel.components.models.qwen3_5_moe.model", "Qwen3_5MoeForConditionalGeneration"),
    ("nemo_automodel.components.models.minimax_m3_vl.model", "MiniMaxM3SparseForConditionalGeneration"),
    ("nemo_automodel.components.models.nemotron_omni.model", "NemotronOmniForConditionalGeneration"),
]


@pytest.mark.parametrize("module_path,class_name", PRE_EMBED_MODELS)
def test_forward_binds_pre_embed_kwargs_only_call(module_path, class_name):
    module = pytest.importorskip(module_path)
    forward = getattr(module, class_name).forward
    signature = inspect.signature(forward)
    try:
        signature.bind(object(), _pre_embed_only=True, _cp_batch={}, num_chunks=1)
    except TypeError as err:
        pytest.fail(
            f"{class_name}.forward cannot be called kwargs-only as "
            f"model(_pre_embed_only=True, _cp_batch=batch, num_chunks=n): {err}. "
            "Give every positional parameter a default (input_ids=None) so the "
            "CP pre-embed dispatch in prepare_cp_forward can reach the "
            "_pre_embed_only branch."
        )
