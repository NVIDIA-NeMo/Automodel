# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU regressions for model-owned FA4 packing and dispatch."""

from dataclasses import replace
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn
from nemo_automodel.components.datasets.packing import get_unpad_data
from nemo_automodel.components.datasets.utils import neat_packed_collater
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import configure_packing, get_model_attn_implementation
from tests.unit_tests.attention.test_attention_utils import _reference_varlen_sdpa
from tests.unit_tests.models.step3p5.test_step3p5_model import MockStep3p5Config


def _packed_sample():
    return dict(
        input_ids=[1, 2, 3, 4], labels=[2, -100, 4, -100], attention_mask=[1, 1, 2, 2], position_ids=[0, 1, 0, 1]
    )


@pytest.mark.parametrize("model_family", ["llama", "qwen2", "laguna"])
@pytest.mark.parametrize("backend", ["sdpa", "te", "fa4"])
def test_hf_dispatch_is_model_owned(model_family, backend):
    from nemo_automodel.components.models.laguna.model import LagunaForCausalLM
    from nemo_automodel.components.models.llama.model import LlamaForCausalLM
    from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM

    cls = {"llama": LlamaForCausalLM, "qwen2": Qwen2ForCausalLM, "laguna": LagunaForCausalLM}[model_family]
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn=backend)
    model.config = SimpleNamespace(_attn_implementation="flash_attention_4")
    expected = "te" if backend == "te" and model_family != "laguna" else "flash_attention_4"
    assert get_model_attn_implementation(model) == expected


def test_backend_dispatched_model_preserves_document_isolation():
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn="sdpa")
    model.config = SimpleNamespace(_attn_implementation="flash_attention_4")
    contract = configure_packing(get_model_attn_implementation(model), model=model, unpad_data=get_unpad_data)
    batch = neat_packed_collater([_packed_sample()], packing=contract)
    q = k = torch.zeros(1, 4, 1, 1)
    v = torch.tensor([10.0, 10.0, 0.0, 0.0]).reshape(1, 4, 1, 1).requires_grad_()
    q_sdpa, k_sdpa, v_sdpa, kwargs = preprocess_args_and_kwargs_for_attn(
        q, k, v, batch["attention_mask"], model.backend.attn
    )
    output = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, **kwargs)
    torch.testing.assert_close(output.flatten(), torch.tensor([10.0, 10.0, 0.0, 0.0]))
    upstream = torch.tensor([[[[0.0], [0.0], [2.0], [-3.0]]]])
    output.backward(upstream)
    torch.testing.assert_close(v.grad.flatten()[:2], torch.zeros(2))


def test_step3p5_model_derived_packing_matches_sdpa_forward_backward():
    from nemo_automodel.components.models.step3p5.model import Step3p5ForCausalLM

    torch.manual_seed(7)
    backend = BackendConfig(
        attn="fa4",
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch",
        enable_hf_state_dict_adapter=False,
    )
    config = MockStep3p5Config(torch_dtype="float32", num_hidden_layers=1, layer_types=["full_attention"])
    fa4 = ModuleType("flash_attn.cute")
    fa4.flash_attn_func = None
    fa4.flash_attn_varlen_func = _reference_varlen_sdpa
    with patch.dict("sys.modules", {"flash_attn.cute": fa4}):
        model = Step3p5ForCausalLM(config, backend=backend)
    reference = Step3p5ForCausalLM(config, backend=replace(backend, attn="sdpa"))
    reference.load_state_dict(model.state_dict())
    outputs = []
    for current in (model, reference):
        contract = configure_packing(get_model_attn_implementation(current), model=current)
        batch = neat_packed_collater([_packed_sample()], packing=contract)
        batch.pop("labels")
        outputs.append(current(**batch).logits)
    torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-5, atol=1e-5)
    upstream = torch.randn_like(outputs[0])
    for output in outputs:
        output.backward(upstream)
    for (name, param), (ref_name, ref_param) in zip(model.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if param.grad is not None or ref_param.grad is not None:
            torch.testing.assert_close(param.grad, ref_param.grad, rtol=1e-4, atol=1e-5, msg=name)


@pytest.mark.parametrize("with_metadata", [False, True])
def test_qwen3_next_rejects_packed_fa4_before_recurrent_layers(with_metadata):
    from nemo_automodel.components.models.qwen3_next.model import Qwen3NextForCausalLM

    model = Qwen3NextForCausalLM.__new__(Qwen3NextForCausalLM)
    torch.nn.Module.__init__(model)
    model.backend = BackendConfig(attn="fa4")
    metadata = {"cu_seqlens": torch.tensor([0, 2, 4], dtype=torch.int32)} if with_metadata else {}
    with pytest.raises(ValueError, match="Qwen3Next does not support packed FA4"):
        model(torch.ones(2, 4, dtype=torch.long), attention_mask=torch.tensor([[1, 1, 2, 2]]).expand(2, -1), **metadata)


def test_deepseek_v32_rejects_fa4_before_sparse_layers():
    from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
    from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM

    assert DeepseekV32ForCausalLM._uses_native_fa4 is False
    with pytest.raises(ValueError, match="sparse attention does not support FA4"):
        DeepseekV32ForCausalLM(DeepseekV32Config(), backend=BackendConfig(attn="fa4"))
