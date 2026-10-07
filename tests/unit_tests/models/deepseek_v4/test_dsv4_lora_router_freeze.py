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

"""Attention-only LoRA must preserve pretrained routing across checkpoint resume."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

# Actual DCP save/load and cold model imports exceed the default 5-second budget.
pytestmark = pytest.mark.timeout(120)


def _make_model(*, freeze_router: bool) -> DeepseekV4ForCausalLM:
    recipe_path = (
        Path(__file__).resolve().parents[4] / "examples/llm_finetune/deepseek_v4/deepseek_v4_flash_hellaswag_lora.yaml"
    )
    recipe = yaml.safe_load(recipe_path.read_text())
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=8,
        o_lora_rank=8,
        o_groups=1,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        moe_intermediate_size=8,
        compress_ratios=[0, 0],
        num_hash_layers=1,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=2,
        max_position_embeddings=32,
        num_nextn_predict_layers=0,
        torch_dtype="float32",
    )
    model = DeepseekV4ForCausalLM(
        config,
        backend=BackendConfig(
            attn="sdpa",
            linear="torch",
            rms_norm="torch",
            rope_fusion=False,
            dispatcher="torch",
            experts="torch_mm",
        ),
        moe_overrides=recipe["model"]["moe_overrides"] if freeze_router else None,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model


def _train_step(model: DeepseekV4ForCausalLM, optimizer: torch.optim.Optimizer, tokens: torch.Tensor) -> torch.Tensor:
    """Run one next-token update, including the recipe's router-update hook.

    Args:
        model: Tiny CPU model in training mode.
        optimizer: Optimizer owning the trainable model parameters.
        tokens: Integer token IDs of shape [batch, sequence].

    Returns:
        Detached scalar loss tensor.
    """
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    model.update_moe_gate_bias()
    return loss.detach()


@pytest.mark.runtime_budget(
    45,
    reason="Six real CPU training updates plus model/optimizer DCP save and resume measured 29s on a cold worker.",
)
def test_lora_recipe_preserves_router_state_across_resume(tmp_path: Path):
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        torch.manual_seed(42)
        model = _make_model(freeze_router=True)
        bias_name = "model.layers.1.mlp.gate.e_score_correction_bias"
        # A frozen buffer must still accept the nonzero bias from the base checkpoint.
        original_bias = torch.tensor([0.125, -0.25])
        model.load_state_dict({bias_name: original_bias}, strict=False)
        torch.testing.assert_close(dict(model.named_buffers())[bias_name], original_bias, rtol=0, atol=0)
        peft = PeftConfig(
            target_modules=["*wq_a", "*wq_b", "*wkv", "*wo_b"],
            dim=2,
            alpha=4,
            use_memory_efficient_lora=False,
        )
        apply_lora_to_linear_modules(model, peft)
        initial_adapters = {
            name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad
        }
        resumed = copy.deepcopy(model)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        resumed_optimizer = torch.optim.AdamW([p for p in resumed.parameters() if p.requires_grad], lr=0.01)
        tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
        model.train()
        resumed.train()
        for _ in range(2):
            assert torch.isfinite(_train_step(model, optimizer, tokens))
            torch.testing.assert_close(dict(model.named_buffers())[bias_name], original_bias, rtol=0, atol=0)
        assert any(
            not torch.equal(initial_adapters[name], param)
            for name, param in model.named_parameters()
            if name in initial_adapters
        )

        checkpointer = Checkpointer(
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir=str(tmp_path),
                model_cache_dir=str(tmp_path),
                model_repo_id="tiny-source",
                model_save_format="safetensors",
                save_consolidated=False,
                is_peft=True,
            ),
            dp_rank=0,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )
        checkpoint = str(tmp_path / "saved")
        checkpointer.save_model(model, checkpoint, peft_config=peft)
        checkpointer.save_optimizer(optimizer, model, checkpoint)
        checkpointer.load_model(resumed, str(tmp_path / "saved" / "model"))
        checkpointer.load_optimizer(resumed_optimizer, resumed, checkpoint)

        for _ in range(2):
            expected_loss = _train_step(model, optimizer, tokens)
            actual_loss = _train_step(resumed, resumed_optimizer, tokens)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(dict(resumed.named_buffers())[bias_name], original_bias, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


def test_default_router_bias_updates_remain_enabled():
    model = _make_model(freeze_router=False)
    model.train()
    gate = model.model.layers["1"].mlp.gate
    with torch.no_grad():
        gate.weight.zero_()
    before = gate.e_score_correction_bias.clone()
    _, selected, _ = gate(torch.ones(4, 16), torch.ones(4, dtype=torch.bool), None)
    model.update_moe_gate_bias()
    expected = torch.full((2,), 1e-3)
    expected[selected[0, 0]] = -1e-3
    torch.testing.assert_close(gate.e_score_correction_bias - before, expected, rtol=0, atol=0)
