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

"""HF fallback routers preserve explicit FP32 math under root autocast."""

import copy
import importlib

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches
from nemo_automodel.components.distributed import FSDP2Config, MeshContext
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model

_ROUTERS = [
    ("axk2", "AXK2TopkRouter", "AXK2Config"),
    ("exaone_moe", "ExaoneMoeTopkRouter", "ExaoneMoeConfig"),
    ("glm5_next", "Glm5NextTextTopkRouter", "Glm5NextTextConfig"),
    ("deepseek_v3", "DeepseekV3TopkRouter", "DeepseekV3Config"),
    ("glm_moe_dsa", "GlmMoeDsaTopkRouter", "GlmMoeDsaConfig"),
    ("dots1", "Dots1TopkRouter", "Dots1Config"),
    ("hy_v3", "HYV3TopKRouter", "HYV3Config"),
    ("deepseek_v2", "DeepseekV2TopkRouter", "DeepseekV2Config"),
    ("glm4_moe", "Glm4MoeTopkRouter", "Glm4MoeConfig"),
    ("mimo_v2_flash", "MiMoV2FlashTopkRouter", "MiMoV2FlashConfig"),
    ("kimi_linear", "KimiLinearTopkRouter", "KimiLinearConfig"),
    ("nemotron_h", "NemotronHTopkRouter", "NemotronHConfig"),
    ("glm4v_moe", "Glm4vMoeTextTopkRouter", "Glm4vMoeTextConfig"),
    ("deepseek_ocr2", "DeepseekOcr2TextTopkRouter", "DeepseekOcr2TextConfig"),
    ("glm4_moe_lite", "Glm4MoeLiteTopkRouter", "Glm4MoeLiteConfig"),
    ("deepseek_v32", "DeepseekV32TopkRouter", "DeepseekV32Config"),
    ("solar_open", "SolarOpenTopkRouter", "SolarOpenConfig"),
    ("longcat_flash", "LongcatFlashTopkRouter", "LongcatFlashConfig"),
    ("hy_v4", "HYV4TopkRouter", "HYV4Config"),
    ("axk1", "AXK1TopkRouter", "AXK1Config"),
]


class _RouterModel(nn.Module):
    def __init__(self, gate, config, bias):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(32, 32, bias=False)
        self.gate = gate
        self.register_buffer("bias", bias)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Project tokens and route them.

        Args:
            hidden_states: Activations [tokens, hidden=32].

        Returns:
            Projected activations [tokens, hidden], then upstream router outputs:
            floating logits [tokens, experts] and/or weights [tokens, top_k],
            followed by integer expert indices [tokens, top_k].
        """
        projected = self.proj(hidden_states)
        args = () if self.bias is None else (self.bias,)
        return (projected, *self.gate(projected, *args))


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


@pytest.fixture(params=_ROUTERS, ids=[row[0] for row in _ROUTERS])
def router_model(request):
    family, router_class, config_class = request.param
    config_module = importlib.import_module(f"transformers.models.{family}.configuration_{family}")
    config = getattr(config_module, config_class)()
    overrides = dict(hidden_size=32, num_experts_per_tok=2, n_group=2, topk_group=1)
    for name, value in overrides.items():
        if hasattr(config, name):
            setattr(config, name, value)
    # All real config aliases are retained; no phantom fields are introduced.
    for name in ("n_routed_experts", "num_local_experts", "num_experts"):
        if hasattr(config, name):
            setattr(config, name, 8)
    if hasattr(config, "zero_expert_num"):
        config.zero_expert_num = 0
    modeling = importlib.import_module(f"transformers.models.{family}.modeling_{family}")
    gate = getattr(modeling, router_class)(config)
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in gate.parameters():
            parameter.normal_(0, 0.1)
    bias = torch.linspace(-0.01, 0.01, 8) if family == "hy_v3" else None
    return _RouterModel(gate, config, bias)


@pytest.mark.parametrize("storage_dtype", [torch.float32, torch.bfloat16])
def test_hf_router_policy_matches_fp32_reference(router_model, device_mesh, storage_dtype):
    device = device_mesh.device_type
    model = router_model.to(device=device, dtype=storage_dtype)
    oracle = copy.deepcopy(model)
    context = MeshContext(device_mesh=device_mesh, strategy_config=FSDP2Config())
    assert apply_model_runtime_patches(model, context) is model
    assert parallelize_model(model, context) is model
    inputs = torch.randn(2048, 32, device=device, dtype=storage_dtype, requires_grad=True)
    oracle_inputs = inputs.detach().clone().requires_grad_()
    # Ordinary projections use the compute policy; the untouched HF router runs
    # outside autocast. This oracle never receives the compatibility hook.
    with torch.autocast(device, dtype=torch.bfloat16):
        projected = oracle.proj(oracle_inputs)
    args = () if oracle.bias is None else (oracle.bias,)
    expected = (projected, *oracle.gate(projected, *args))
    actual = model(inputs)
    assert actual[0].dtype == torch.bfloat16
    for value in actual[1:]:
        if value.is_floating_point():
            assert value.dtype == torch.float32
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    floating = [(value, reference) for value, reference in zip(actual, expected) if value.is_floating_point()]
    upstream = [torch.randn_like(value) for value, _ in floating]
    sum((value * grad).sum() for (value, _), grad in zip(floating, upstream)).backward()
    sum((reference * grad).sum() for (_, reference), grad in zip(floating, upstream)).backward()
    torch.testing.assert_close(inputs.grad, oracle_inputs.grad, rtol=0, atol=0)
    for parameter, reference in zip(model.parameters(), oracle.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)
