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

"""The HF router keeps FP32 scoring under the unsharded compute policy."""

import copy
import inspect

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3ForCausalLM

from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches
from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel.components.distributed import FSDP2Config, MeshContext
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model


@pytest.fixture
def hf_model():
    torch.manual_seed(42)
    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_nope_head_dim=8,
        qk_rope_head_dim=8,
        v_head_dim=8,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        n_group=2,
        topk_group=1,
        first_k_dense_replace=0,
        num_mtp_layers=0,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return _get_mixin_wrapped_class(DeepseekV3ForCausalLM)(config)


@pytest.fixture(
    params=[
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")),
    ]
)
def device_mesh(tmp_path, request):
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        yield init_device_mesh(request.param, (1,), mesh_dim_names=("dp",))
    finally:
        dist.destroy_process_group()


def _parallelize(model, compute_dtype, mesh):
    context = MeshContext(
        device_mesh=mesh,
        strategy_config=FSDP2Config(autocast_dtype=compute_dtype),
        activation_checkpointing=False,
        reapply_trainability=None,
    )
    model = apply_model_runtime_patches(model, context)
    return parallelize_model(model, context)


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16])
def test_hf_router_outputs_and_gradients_match_unmodified_fp32(hf_model, device_mesh, storage_dtype, compute_dtype):
    device = device_mesh.device_type
    model = hf_model.to(device=device, dtype=storage_dtype)
    gate = model.model.layers[0].mlp.gate
    reference = copy.deepcopy(gate)
    keys = tuple(model.state_dict())
    parameters = tuple(model.parameters())
    signature = inspect.signature(gate.forward)
    hf_forward = type(gate).forward
    assert _parallelize(model, compute_dtype, device_mesh) is model
    assert tuple(model.state_dict()) == keys
    assert all(actual is original for actual, original in zip(model.parameters(), parameters))
    assert inspect.signature(gate.forward) == signature
    assert type(gate).forward is hf_forward

    x = torch.randn(2, 1024, 32, device=device, dtype=storage_dtype, requires_grad=True)
    oracle_x = x.detach().clone().requires_grad_()
    expected_logits, expected_weights, expected_indices = reference(oracle_x)
    with torch.autocast(device, dtype=compute_dtype):
        logits, weights, indices = gate(x)
    assert logits.dtype == weights.dtype == torch.float32
    torch.testing.assert_close(logits, expected_logits, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=0, atol=0)
    torch.testing.assert_close(indices, expected_indices, rtol=0, atol=0)
    upstream = torch.randn_like(weights)
    (weights * upstream).sum().backward()
    (expected_weights * upstream).sum().backward()
    torch.testing.assert_close(x.grad, oracle_x.grad, rtol=0, atol=0)
    torch.testing.assert_close(gate.weight.grad, reference.weight.grad, rtol=0, atol=0)

    clone = copy.deepcopy(model)
    assert clone.model.layers[0].mlp.gate.forward.__self__ is clone.model.layers[0].mlp.gate
    projected_dtypes = []
    handle = clone.lm_head.register_forward_hook(lambda module, inputs, output: projected_dtypes.append(output.dtype))
    try:
        output = clone(input_ids=torch.tensor([[2, 4, 6, 8]], device=device))
    finally:
        handle.remove()
    assert output.logits.dtype == compute_dtype
    assert projected_dtypes == [compute_dtype]
    assert not torch.is_autocast_enabled(device)
