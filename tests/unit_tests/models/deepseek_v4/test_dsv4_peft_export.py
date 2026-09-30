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

"""DeepSeek V4 LoRA exports must load into the actual Transformers model."""

import json
from pathlib import Path

import pytest
import torch
from peft import PeftModel, get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config as HFConfig
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM as HFModel

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

# Real checkpoint save/load and PEFT construction exceed the default 5-second budget on cold CI workers.
pytestmark = pytest.mark.timeout(120)


@pytest.fixture
def peft_process_group(tmp_path: Path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("v4_compatible", [False, True])
def test_peft_export_loads_in_hf_and_automodel(tmp_path: Path, peft_process_group, v4_compatible: bool):
    """Exercise real metadata/tensor saves, HF consumption, and native checkpoint reload."""
    torch.manual_seed(42)
    config_values = dict(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        compress_ratios=[0, 4, 128],
        num_hash_layers=0,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        max_position_embeddings=32,
        num_nextn_predict_layers=0,
        torch_dtype="float32",
    )
    model = DeepseekV4ForCausalLM(
        DeepseekV4Config(**config_values),
        backend=BackendConfig(
            attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False, dispatcher="torch", experts="torch_mm"
        ),
    )
    reference = HFModel(HFConfig(**config_values))
    reference.save_pretrained(tmp_path / "source")
    peft_config = PeftConfig(
        target_modules=["*wq_a", "*wq_b", "*wkv", "*wo_b"],
        dim=2,
        alpha=4,
        use_memory_efficient_lora=False,
    )
    assert apply_lora_to_linear_modules(model, peft_config) > 0

    # These are the receiving model's public module names, independently of the export implementation.
    projections = {"wq_a": "q_a_proj", "wq_b": "q_b_proj", "wkv": "kv_proj", "wo_b": "o_b_proj"}
    module_pairs = []
    with torch.no_grad():
        for name, module in model.named_modules():
            if not hasattr(module, "lora_A"):
                continue
            prefix, leaf = name.rsplit(".", 1)
            hf_name = prefix + "." + projections[leaf]
            hf_module = reference.get_submodule(hf_name)
            module.weight.copy_(hf_module.weight)
            module.lora_A.weight.normal_(std=0.05)
            module.lora_B.weight.normal_(std=0.05)
            module_pairs.append((name, hf_name))
    assert any("compressor.indexer" in name for name, _ in module_pairs)
    native = {name: tensor.clone() for name, tensor in ModelState(model, is_peft=True).state_dict().items()}
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
            v4_compatible=v4_compatible,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.save_model(model, str(tmp_path / "peft"), peft_config=peft_config)
    adapter_dir = tmp_path / "peft" / "model"
    metadata = json.loads((adapter_dir / "adapter_config.json").read_text())
    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    streaming = dict(
        item
        for name, tensor in native.items()
        for item in model.state_dict_adapter.convert_single_tensor_to_hf(name, tensor, v4_compatible=v4_compatible)
    )
    assert set(exported) == set(streaming)
    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), is_trainable=True, autocast_adapter_dtype=False)
    assert set(metadata["target_modules"]) == {hf_name for _, hf_name in module_pairs}
    loaded_state = get_peft_model_state_dict(loaded, save_embedding_layers=False)
    assert set(exported) == set(loaded_state)
    for name, value in exported.items():
        torch.testing.assert_close(loaded_state[name], value, rtol=0, atol=0)
        torch.testing.assert_close(streaming[name], value, rtol=0, atol=0)

    # Check every adapted projection's forward and backward behavior with nonzero adapters.
    # Full-model parity is covered by the scoped checkpoint-robustness CI recipe.
    for native_name, hf_name in module_pairs:
        source = model.get_submodule(native_name)
        target = loaded.base_model.model.get_submodule(hf_name)
        inputs = torch.randn(2, 3, source.in_features, requires_grad=True)
        hf_inputs = inputs.detach().clone().requires_grad_()
        expected = source(inputs)
        actual = target(hf_inputs)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        torch.testing.assert_close(hf_inputs.grad, inputs.grad, rtol=1e-5, atol=1e-6)
        for side in ("lora_A", "lora_B"):
            torch.testing.assert_close(
                getattr(target, side)["default"].weight.grad,
                getattr(source, side).weight.grad,
                rtol=1e-5,
                atol=1e-6,
            )

    # New HF-named exports and old AutoModel-named adapters must both remain resumable.
    for saved in (exported, native):
        save_file(saved, str(adapter_dir / "adapter_model.safetensors"))
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if ".lora_" in name:
                    parameter.zero_()
        checkpointer.load_model(model, str(adapter_dir))
        restored = ModelState(model, is_peft=True).state_dict()
        assert set(restored) == set(native)
        for name in native:
            torch.testing.assert_close(restored[name], native[name], rtol=0, atol=0)
