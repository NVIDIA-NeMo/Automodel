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

"""Real checkpoint interoperability for native models with HF-compatible tensors."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, PreTrainedModel

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM

# Over the default 5s budget on purpose: this module repeatedly saves, loads, and exports full model state.
# Shrink the checkpoint round trips before raising this further.
pytestmark = pytest.mark.timeout(60)


@pytest.fixture(params=[LlamaForCausalLM, Qwen2ForCausalLM, Qwen3ForCausalLM], ids=["llama", "qwen2", "qwen3"])
def native_model_class(request) -> type[PreTrainedModel]:
    return request.param


@pytest.fixture
def peft_process_group(tmp_path: Path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


@pytest.fixture
def model_pair(tmp_path: Path, native_model_class: type[PreTrainedModel], tie_word_embeddings: bool):
    torch.manual_seed(1234)
    config = native_model_class.config_class(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        attention_dropout=0.0,
        tie_word_embeddings=tie_word_embeddings,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    reference = AutoModelForCausalLM.from_config(config).eval()
    reference.save_pretrained(tmp_path / "source")
    config.dtype = torch.float32
    config.name_or_path = str(tmp_path / "source")
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
    model = native_model_class(config, backend=backend).eval()
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
        side_effect=AssertionError("Expected direct DCP loading"),
    ):
        checkpointer.load_model(model, str(tmp_path / "source"), is_init_step=True)
    return model, reference, checkpointer


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_native_hf_checkpoint_weight_gradient_and_export_parity(tmp_path: Path, model_pair, tie_word_embeddings: bool):
    """Loading, an optimizer update, HF export, and native resume preserve the model's weight contract."""
    model, reference, checkpointer = model_pair
    assert not hasattr(model, "state_dict_adapter")
    assert (model.get_input_embeddings().weight is model.get_output_embeddings().weight) is tie_word_embeddings
    assert model.state_dict().keys() == reference.state_dict().keys()
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)

    input_ids = torch.tensor([[1, 2, 3, 4]])
    actual = model(input_ids=input_ids, use_cache=False).logits
    expected = reference(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    upstream = torch.randn_like(expected)
    actual.backward(upstream)
    expected.backward(upstream)
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, reference_parameters[name].grad, atol=2e-6, rtol=2e-5)

    torch.optim.SGD(model.parameters(), lr=0.01).step()
    torch.optim.SGD(reference.parameters(), lr=0.01).step()
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, reference_parameters[name], atol=2e-6, rtol=2e-5)

    checkpointer.save_model(model, str(tmp_path / "saved"))
    exported = AutoModelForCausalLM.from_pretrained(tmp_path / "saved/model/consolidated").eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(exported.state_dict()[name], value, rtol=0, atol=0)

    reloaded = type(model)(model.config, backend=model.backend).eval()
    checkpointer.load_model(reloaded, str(tmp_path / "saved/model"))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[name], value, rtol=0, atol=0)
    assert (reloaded.get_input_embeddings().weight is reloaded.get_output_embeddings().weight) is tie_word_embeddings


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
def test_native_hf_peft_export_and_reload(tmp_path: Path, model_pair, tie_word_embeddings: bool, peft_process_group):
    """LoRA exports load in both HF PEFT and native models without a state-dict adapter."""
    model, reference, checkpointer = model_pair
    peft_config = PeftConfig(target_modules=["*.q_proj", "*.v_proj"], dim=2, alpha=4, use_memory_efficient_lora=False)
    assert apply_lora_to_linear_modules(model, peft_config) == 2
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.normal_(std=0.1)
    checkpointer = Checkpointer(
        replace(checkpointer.config, is_peft=True, model_save_format="safetensors", save_consolidated=False),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.save_model(model, str(tmp_path / "peft"), peft_config=peft_config)

    exported = PeftModel.from_pretrained(reference, tmp_path / "peft/model").eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(exported(input_ids=input_ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)

    reloaded = type(model)(model.config, backend=model.backend).eval()
    reloaded.load_state_dict(AutoModelForCausalLM.from_pretrained(tmp_path / "source").state_dict())
    assert apply_lora_to_linear_modules(reloaded, peft_config) == 2
    checkpointer.load_model(reloaded, str(tmp_path / "peft/model"))
    torch.testing.assert_close(reloaded(input_ids=input_ids, use_cache=False).logits, expected, atol=1e-6, rtol=1e-6)
