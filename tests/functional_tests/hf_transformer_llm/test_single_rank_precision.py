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

"""Exercise native DeepSeek routing through checkpoint loading and single-rank training."""

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import PreTrainedModel
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM
from transformers.models.deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from transformers.models.deepseek_v32.modeling_deepseek_v32 import DeepseekV32ForCausalLM

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM as NativeDeepseekV3ForCausalLM
from nemo_automodel.components.models.deepseek_v32.model import DeepseekV32ForCausalLM as NativeDeepseekV32ForCausalLM

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@dataclass
class _Checkpoint:
    path: Path
    model_class: type[PreTrainedModel]
    native_class: type[torch.nn.Module]


def _per_expert_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Expand grouped HF weights to the native loader's per-expert checkpoint layout.

    Args:
        state: HF state with gate/up weights [experts, 2 * intermediate, hidden]
            and down weights [experts, hidden, intermediate].

    Returns:
        State with individual two-dimensional gate, up and down projection weights.
    """
    result = {}
    for name, value in state.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert, weight in enumerate(value):
                gate, up = weight.chunk(2, dim=0)
                result[f"{prefix}.{expert}.gate_proj.weight"] = gate.clone()
                result[f"{prefix}.{expert}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert, weight in enumerate(value):
                result[f"{prefix}.{expert}.down_proj.weight"] = weight.clone()
        else:
            result[name] = value
    return result


@pytest.fixture(
    params=[
        (DeepseekV3Config, DeepseekV3ForCausalLM, NativeDeepseekV3ForCausalLM, {}),
        (
            DeepseekV32Config,
            DeepseekV32ForCausalLM,
            NativeDeepseekV32ForCausalLM,
            dict(index_n_heads=2, index_head_dim=32, index_topk=4),
        ),
    ],
    ids=["deepseek_v3", "deepseek_v32"],
)
def checkpoint(tmp_path, request):
    torch.manual_seed(42)
    config_type, model_class, native_class, index_kwargs = request.param
    config = config_type(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=16,
        q_lora_rank=16,
        qk_nope_head_dim=16,
        qk_rope_head_dim=16,
        v_head_dim=16,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        num_mtp_layers=0,
        use_cache=False,
        **index_kwargs,
    )
    config._attn_implementation = "sdpa"
    path = tmp_path / "checkpoint"
    model = model_class(config)
    model.save_pretrained(path, state_dict=_per_expert_state(model.state_dict()))
    return _Checkpoint(path, model_class, native_class)


@pytest.fixture
def single_rank_group(tmp_path):
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1, timeout=timedelta(seconds=120)
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def _check_router(module, inputs, output):
    # Native routing must retain the values and selections computed outside AMP.
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        expected = type(module).forward(module, *inputs)
    assert module.gate_precision == output[0].dtype == torch.float32
    for actual, oracle in zip(output, expected):
        if oracle is not None:
            torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


@pytest.mark.parametrize("checkpointing", [False, True])
def test_native_router_precision_and_training(checkpoint, single_rank_group, tmp_path, checkpointing):
    model = NeMoAutoModelForCausalLM.from_pretrained(
        str(checkpoint.path),
        dtype=torch.float32,
        attn_implementation="sdpa",
        use_liger_kernel=False,
        use_sdpa_patching=False,
        backend=BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False),
        moe_overrides=dict(
            router_weights_fp32=True, force_e_score_correction_bias=True, gate_bias_update_factor=0, aux_loss_coeff=0
        ),
        distributed_setup=DistributedSetup.build(
            strategy=FSDP2Config(), world_size=1, activation_checkpointing=checkpointing
        ),
    ).train()
    assert isinstance(model, checkpoint.native_class)
    source = checkpoint.model_class.from_pretrained(checkpoint.path, dtype=torch.float32).state_dict()
    expected_state = model.state_dict_adapter.from_hf(_per_expert_state(source))
    loaded = model.state_dict()
    for name, value in expected_state.items():
        assert loaded[name].dtype == value.dtype
        torch.testing.assert_close(loaded[name].cpu(), value.cpu(), rtol=0, atol=0)
    gate = model.model.layers["0"].mlp.gate
    original = gate.weight.detach().clone()
    buffers = {name: value.clone() for name, value in model.named_buffers()}
    router_calls, projection_dtypes = [], []
    handles = [
        gate.register_forward_hook(_check_router),
        gate.register_forward_hook(lambda module, inputs, output: router_calls.append(output[0].dtype)),
        model.lm_head.register_forward_hook(lambda module, inputs, output: projection_dtypes.append(output.dtype)),
    ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0)
    inputs = torch.tensor([[2, 4, 6, 8, 10, 12, 14, 16]], device="cuda")
    try:
        for _ in range(2):
            output = model(input_ids=inputs)
            assert output.logits.dtype == torch.bfloat16
            loss = torch.nn.functional.cross_entropy(
                output.logits[:, :-1].float().flatten(0, 1), inputs[:, 1:].flatten()
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert gate.weight.grad is not None and gate.weight.grad.abs().max() > 0
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            optimizer.step()
            optimizer.zero_grad()
    finally:
        for handle in handles:
            handle.remove()
    assert len(router_calls) >= (4 if checkpointing else 2)
    assert projection_dtypes == [torch.bfloat16, torch.bfloat16]
    assert not torch.equal(gate.weight, original)
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert all(
        state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32 for state in optimizer.state.values()
    )
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    # Export/reload must retain the ordinary HF key layout and exact weights.
    exported = tmp_path / "exported"
    checkpointer = Checkpointer(
        CheckpointingConfig(checkpoint_dir=str(tmp_path), model_save_format="safetensors", save_consolidated=True),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    model.save_pretrained(str(exported), checkpointer=checkpointer)
    reloaded = checkpoint.model_class.from_pretrained(exported / "model/consolidated", dtype=torch.float32)
    reloaded_state = model.state_dict_adapter.from_hf(_per_expert_state(reloaded.state_dict()))
    for name, value in reloaded_state.items():
        torch.testing.assert_close(value.cpu(), model.state_dict()[name].cpu(), rtol=0, atol=0)
