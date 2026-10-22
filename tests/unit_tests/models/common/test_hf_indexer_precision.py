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

"""HF sparse score precision, cache semantics, and projection compute policy."""

import copy
import importlib
import inspect

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.overrides import TorchFunctionMode
from transformers.cache_utils import Cache, DynamicIndexedLayer, StaticIndexedLayer

from nemo_automodel._transformers.kernel_patches import apply_model_runtime_patches
from nemo_automodel.components.distributed import FSDP2Config, MeshContext
from nemo_automodel.components.distributed.model_parallelizer import parallelize_model


class _MatmulOracle(TorchFunctionMode):
    """Observe real operations without replacing operands or results."""

    def __init__(self):
        self.calls = 0

    def __torch_function__(self, func, types, args: tuple[object, ...] = (), kwargs=None):
        output = func(*args, **(kwargs or {}))
        if func is torch.matmul:
            left, right = args
            assert isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor)
            assert left.dtype == right.dtype == output.dtype == torch.float32
            assert not output.requires_grad
            with torch.no_grad(), torch.autocast(left.device.type, enabled=False):
                expected = torch.matmul(left, right)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            self.calls += 1
        return output


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


@pytest.fixture(
    params=[
        ("deepseek_v32", "DeepseekV32Config", "DeepseekV32Indexer", "bool"),
        ("deepseek_v32", "DeepseekV32Config", "DeepseekV32Indexer", "additive"),
        ("glm_moe_dsa", "GlmMoeDsaConfig", "GlmMoeDsaIndexer", "bool"),
        ("glm_moe_dsa", "GlmMoeDsaConfig", "GlmMoeDsaIndexer", "additive"),
        ("glm5_next", "Glm5NextTextConfig", "Glm5NextTextIndexer", "pooled"),
    ],
    ids=["deepseek-bool", "deepseek-additive", "glm-bool", "glm-additive", "glm-pooled"],
)
def indexer_case(request):
    family, config_class, indexer_class, mask_kind = request.param
    config_type = getattr(importlib.import_module(f"transformers.models.{family}.configuration_{family}"), config_class)
    kwargs = dict(
        hidden_size=32,
        q_lora_rank=8,
        index_n_heads=2,
        index_head_dim=16,
        qk_rope_head_dim=8,
        index_topk=4,
        num_hidden_layers=1,
    )
    if mask_kind == "pooled":
        kwargs.update(index_kpool=4, index_kpool_always_select_tail=True, qk_rope_head_dim=0)
    config = config_type(**kwargs)
    indexer_type = getattr(importlib.import_module(f"transformers.models.{family}.modeling_{family}"), indexer_class)
    indexer = indexer_type(config, layer_idx=0)
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in indexer.parameters():
            parameter.normal_(0, 0.1)
    return indexer, mask_kind


def _cache(kind):
    if kind == "dynamic":
        return Cache(layers=[DynamicIndexedLayer()])
    if kind == "static":
        return Cache(layers=[StaticIndexedLayer(max_cache_len=64)])
    return None


def _inputs(length, offset, cache_kind, mask_kind, device):
    x = torch.randn(1, length, 32, device=device, requires_grad=True)
    residual = torch.randn(1, length, 8, device=device, requires_grad=True)
    if mask_kind == "pooled":
        mask = torch.ones(1, length, device=device, dtype=torch.bool)
        mask[:, -1] = False
        return (x, residual, mask)
    key_length = 64 if cache_kind == "static" else offset + length
    positions = torch.arange(offset, offset + length, device=device).unsqueeze(0)
    keys = torch.arange(key_length, device=device)
    mask = (keys <= positions.unsqueeze(-1)) & (keys != offset + length - 1)
    if mask_kind == "additive":
        mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(~mask, -torch.inf)
    angles = positions.float().unsqueeze(-1) * torch.tensor([1.0, 0.1, 0.01, 0.001], device=device)
    angles = torch.cat((angles, angles), -1)
    return (x, residual, (angles.cos(), angles.sin()), mask, positions)


def _check_cache_parity(actual: Cache, expected: Cache) -> None:
    for layer, reference in zip(actual.layers, expected.layers):
        assert isinstance(layer, (DynamicIndexedLayer, StaticIndexedLayer))
        assert isinstance(reference, (DynamicIndexedLayer, StaticIndexedLayer))
        assert layer.indexer_keys is not None and reference.indexer_keys is not None
        torch.testing.assert_close(layer.indexer_keys, reference.indexer_keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.keys, reference.keys, rtol=0, atol=0)
        torch.testing.assert_close(layer.values, reference.values, rtol=0, atol=0)


@pytest.mark.parametrize("compute_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cache_kind", ["none", "dynamic", "static"])
def test_hf_indexer_scores_cache_and_signature(indexer_case, device_mesh, compute_dtype, cache_kind):
    indexer, mask_kind = indexer_case
    device = device_mesh.device_type
    indexer.to(device)
    reference = copy.deepcopy(indexer)
    context = MeshContext(
        device_mesh=device_mesh, strategy_config=FSDP2Config(mp_policy=MixedPrecisionPolicy(param_dtype=compute_dtype))
    )
    assert apply_model_runtime_patches(indexer, context) is indexer
    assert parallelize_model(indexer, context) is indexer
    assert inspect.signature(indexer.forward) == inspect.signature(reference.forward)
    clone = copy.deepcopy(indexer)
    assert clone.forward.__self__ is clone
    assert tuple(indexer.state_dict()) == tuple(reference.state_dict())
    cache, reference_cache = _cache(cache_kind), _cache(cache_kind)
    projected = []
    handles = [
        module.register_forward_hook(lambda module, inputs, output: projected.append(output.dtype))
        for module in (indexer.wq_b, indexer.wk, indexer.weights_proj)
    ]
    offset = 0
    try:
        for length in (17, 5):
            args = _inputs(length, offset, cache_kind, mask_kind, device)
            if cache is not None:
                kv = torch.randn(1, 1, length, 8, device=device)
                cache.update(kv, kv, 0)
                assert reference_cache is not None
                reference_cache.update(kv, kv, 0)
            observer = _MatmulOracle()
            with observer:
                actual = indexer(*args, cache)
            assert observer.calls == 2
            assert actual.dtype == torch.int32 and not actual.requires_grad
            if compute_dtype == torch.float32:
                expected = reference(*args, reference_cache)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                if cache is not None:
                    assert reference_cache is not None
                    _check_cache_parity(cache, reference_cache)
            if cache is not None:
                offset += length
                assert cache.get_seq_length() == offset
    finally:
        for handle in handles:
            handle.remove()
    assert projected == [compute_dtype] * 6
    assert {parameter.dtype for parameter in indexer.parameters()} == {torch.float32}
    assert not torch.is_autocast_enabled(device)


class _HyperModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

        self.config = config
        self.hc = Glm5NextTextHyperConnection(config)

    def forward(self, streams: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute residual-stream coefficients and collapsed states.

        Args:
            streams: Activations [batch, sequence, hc_mult, hidden].

        Returns:
            Post weights [batch, sequence, hc_mult], combination weights
            [batch, sequence, hc_mult, hc_mult], and collapsed states
            [batch, sequence, hidden].
        """
        return self.hc(streams)


def test_hf_hyperconnection_values_and_gradients(device_mesh):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig

    config = Glm5NextTextConfig(hidden_size=32, hc_mult=3, hc_sinkhorn_iters=4)
    model = _HyperModel(config).to(device_mesh.device_type)
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0, 0.1)
    reference = copy.deepcopy(model)
    context = MeshContext(device_mesh=device_mesh, strategy_config=FSDP2Config())
    apply_model_runtime_patches(model, context)
    parallelize_model(model, context)
    x = torch.randn(1, 5, 3, 32, device=device_mesh.device_type, dtype=torch.bfloat16, requires_grad=True)
    oracle_x = x.detach().clone().requires_grad_()
    expected, actual = reference(oracle_x), model(x)
    upstream = [torch.randn_like(value) for value in expected]
    for value, oracle in zip(actual, expected):
        torch.testing.assert_close(value, oracle, rtol=0, atol=0)
    torch.autograd.backward(actual, upstream)
    torch.autograd.backward(expected, upstream)
    torch.testing.assert_close(x.grad, oracle_x.grad, rtol=0, atol=0)
    for parameter, oracle in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, oracle.grad, rtol=0, atol=0)
