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

"""LoRA recipes must retain pretrained routing when checkpointing is enabled."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml
from transformers import PretrainedConfig
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
from nemo_automodel.components.models.deepseek_v32.config import DeepseekV32Config
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM
from nemo_automodel.components.models.glm_moe_dsa.model import GlmMoeDsaForCausalLM

MODEL_TYPES = [
    pytest.param(DeepseekV3Config, DeepseekV3ForCausalLM, id="deepseek-v3"),
    pytest.param(DeepseekV32Config, DeepseekV32ForCausalLM, id="deepseek-v32"),
    pytest.param(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, id="glm-dsa"),
]
RECIPE_CASES = [
    pytest.param(
        DeepseekV32Config, DeepseekV32ForCausalLM, "llm_benchmark/deepseek/dsv32_lora.yaml", id="deepseek-v32"
    ),
    pytest.param(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, "llm_benchmark/glm/glm5_lora.yaml", id="glm5"),
    pytest.param(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, "llm_finetune/glm/glm_5.1_lora.yaml", id="glm51"),
    pytest.param(GlmMoeDsaConfig, GlmMoeDsaForCausalLM, "llm_finetune/glm/glm_5.2_lora.yaml", id="glm52"),
]


def _make_model(
    config_type: type[PretrainedConfig],
    model_type: type[DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM],
    moe_overrides: dict[str, float | bool] | None = None,
) -> DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM:
    config = config_type(
        vocab_size=32,
        hidden_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=128,
        moe_intermediate_size=32,
        n_routed_experts=4,
        n_shared_experts=0,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        norm_topk_prob=False,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        attention_bias=False,
        q_lora_rank=32,
        kv_lora_rank=16,
        qk_head_dim=16,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        index_n_heads=2,
        index_head_dim=16,
        index_topk=2,
        first_k_dense_replace=0,
        mlp_layer_types=["sparse"],
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        torch_dtype="float32",
    )
    model = model_type(
        config,
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            rope_fusion=False,
            experts="torch_mm",
            dispatcher="torch",
            fake_balanced_gate=False,
        ),
        moe_overrides=moe_overrides,
    )
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    return model


def _train_step(
    model: DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM,
    optimizer: torch.optim.Optimizer,
    tokens: torch.Tensor,
) -> torch.Tensor:
    """Train adapters and invoke the same router update hook as the recipe.

    Args:
        model: Tiny native causal language model in training mode.
        optimizer: Optimizer owning the trainable adapter parameters.
        tokens: Integer token IDs of shape [batch, sequence].

    Returns:
        Detached scalar next-token loss tensor.
    """
    optimizer.zero_grad(set_to_none=True)
    logits = model(tokens).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
    loss.backward()
    optimizer.step()
    model.update_moe_gate_bias()
    return loss.detach()


@pytest.mark.parametrize("config_type,model_type,recipe_name", RECIPE_CASES)
@pytest.mark.runtime_budget(45, reason="Real model/optimizer DCP save/reload and six tiny CPU training updates.")
def test_lora_recipe_save_resume_keeps_pretrained_router(
    tmp_path: Path,
    config_type: type[PretrainedConfig],
    model_type: type[DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM],
    recipe_name: str,
) -> None:
    recipe = yaml.safe_load((Path(__file__).resolve().parents[3] / "examples" / recipe_name).read_text())
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        torch.manual_seed(42)
        model = _make_model(config_type, model_type, recipe["model"].get("moe_overrides"))
        bias_name = "model.layers.0.mlp.gate.e_score_correction_bias"
        original_bias = torch.tensor([0.125, -0.25, 0.375, -0.5])
        # Exercise actual base-weight loading, including retention of the nonzero buffer.
        model.load_state_dict({bias_name: original_bias}, strict=False)
        torch.testing.assert_close(dict(model.named_buffers())[bias_name], original_bias, rtol=0, atol=0)
        # Keep each recipe's target selection; use small ranks and portable CPU kernels.
        peft_options = {k: v for k, v in recipe["peft"].items() if k != "_target_"}
        peft_options.update(dim=4, alpha=8, use_triton=False, use_memory_efficient_lora=False)
        peft = PeftConfig(**peft_options)
        apply_lora_to_linear_modules(model, peft)
        initial_adapters = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}
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
            not torch.equal(initial_adapters[n], p) for n, p in model.named_parameters() if n in initial_adapters
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
        for name, value in model.state_dict().items():
            torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
        for _ in range(2):
            expected_loss = _train_step(model, optimizer, tokens)
            actual_loss = _train_step(resumed, resumed_optimizer, tokens)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(resumed.state_dict()[name], value, rtol=0, atol=0)
            torch.testing.assert_close(dict(resumed.named_buffers())[bias_name], original_bias, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("config_type,model_type", MODEL_TYPES)
@pytest.mark.parametrize("inner_hook", [False, True], ids=["causal-lm", "pipeline-stage"])
def test_disabled_router_hook_preserves_pretrained_bias(
    config_type: type[PretrainedConfig],
    model_type: type[DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM],
    inner_hook: bool,
) -> None:
    model = _make_model(
        config_type,
        model_type,
        {
            "gate_bias_update_factor": 0.0,
            "force_e_score_correction_bias": True,
        },
    )
    bias = torch.tensor([0.125, -0.25, 0.375, -0.5])
    model.load_state_dict({"model.layers.0.mlp.gate.e_score_correction_bias": bias}, strict=False)
    gate = model.model.layers["0"].mlp.gate
    gate(torch.ones(4, 64), torch.ones(4, dtype=torch.bool), None)
    (model.model if inner_hook else model).update_moe_gate_bias()
    torch.testing.assert_close(gate.e_score_correction_bias, bias, rtol=0, atol=0)


@pytest.mark.parametrize("config_type,model_type", MODEL_TYPES)
@pytest.mark.parametrize("inner_hook", [False, True], ids=["causal-lm", "pipeline-stage"])
def test_default_router_updates_remain_enabled(
    config_type: type[PretrainedConfig],
    model_type: type[DeepseekV3ForCausalLM | GlmMoeDsaForCausalLM],
    inner_hook: bool,
) -> None:
    model = _make_model(config_type, model_type)
    gate = model.model.layers["0"].mlp.gate
    with torch.no_grad():
        gate.weight.zero_()
    before = gate.e_score_correction_bias.clone()
    _, selected, _ = gate(torch.ones(4, 64), torch.ones(4, dtype=torch.bool), None)
    (model.model if inner_hook else model).update_moe_gate_bias()
    expected = torch.full((4,), 1e-3)
    expected[selected[0]] = -1e-3
    torch.testing.assert_close(gate.e_score_correction_bias - before, expected, rtol=0, atol=0)
