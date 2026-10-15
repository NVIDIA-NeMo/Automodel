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

"""Storage precision survives LoRA surgery, initialization, and checkpoint loading."""

import copy

import pytest
import torch
from torch import nn

from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, patch_linear_module, patch_moe_module
from nemo_automodel.components.models.common.utils import (
    cast_frozen_modules_to_compute_dtype,
    cast_model_to_dtype,
    yield_fp32_model,
)
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.shared.import_utils import is_torch_min_version
from tests.utils.lora_storage import expert_routing_inputs, randomize_and_copy, tiny_moe_config


@pytest.mark.parametrize("expert_cls", [GroupedExperts, GroupedExpertsDeepEP])
@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_expert_wrapper_preserves_actual_base_storage(expert_cls, device, dtype):
    # Deliberately disagree with config: wrapping must honor the existing tensor.
    with torch.device(device):
        original = expert_cls(tiny_moe_config(dtype=torch.float32, bias=True)).to(dtype)
    base_parameters = dict(original.named_parameters())
    wrapped = patch_moe_module(original, dim=4)
    for name, original_parameter in base_parameters.items():
        parameter = wrapped.get_parameter(name)
        assert parameter is original_parameter
        assert parameter.dtype == dtype
        assert parameter.device.type == device
        assert not parameter.requires_grad
    for name, parameter in wrapped.named_parameters():
        if name.startswith("lora_"):
            assert parameter.dtype == torch.float32
            assert parameter.device.type == device
            assert parameter.requires_grad


@pytest.mark.parametrize("expert_cls", [GroupedExperts, GroupedExpertsDeepEP])
def test_expert_constructor_uses_config_dtype(expert_cls):
    experts = expert_cls(tiny_moe_config(bias=True))
    assert all(parameter.dtype == torch.bfloat16 for parameter in experts.parameters())


@pytest.mark.parametrize("kind", ["grouped", "deepep", "linear"])
def test_storage_survives_materialization_init_and_load(kind):
    with torch.device("meta"):
        if kind == "linear":
            module = LinearLoRA(nn.Linear(16, 16, dtype=torch.bfloat16), dim=4)
        else:
            expert_cls = GroupedExperts if kind == "grouped" else GroupedExpertsDeepEP
            module = patch_moe_module(expert_cls(tiny_moe_config()), dim=4)
    module.to_empty(device="cpu")
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.fill_(0.1234567)
    original = {name: parameter.detach().clone() for name, parameter in module.named_parameters()}
    # DeepSeek V3 initializes in FP32 before restoring base storage.
    with yield_fp32_model(module, torch.bfloat16):
        assert all(p.dtype == torch.float32 for p in module.parameters())
    # Custom model initialization and the final frozen-module pass both run these casts.
    cast_model_to_dtype(module, torch.bfloat16)
    cast_frozen_modules_to_compute_dtype(module, torch.bfloat16)
    for name, parameter in module.named_parameters():
        torch.testing.assert_close(parameter, original[name], rtol=0, atol=0)
    # A BF16 checkpoint source must not downcast the configured FP32 adapters.
    module.load_state_dict({name: value.to(torch.bfloat16) for name, value in original.items()})
    for name, parameter in module.named_parameters():
        assert parameter.dtype == original[name].dtype


def test_mixed_storage_expert_forward_backward_and_step():
    torch.manual_seed(7)
    module = patch_moe_module(GroupedExperts(tiny_moe_config()), dim=4)
    reference = randomize_and_copy(module)
    x = torch.randn(8, 16, dtype=torch.bfloat16, requires_grad=True)
    mask, weights, indices = expert_routing_inputs(x)
    output = module(x, mask, weights, indices)
    # Explicit BF16 matmuls with FP32 adapter leaves form an independent reference.
    rows = []
    for row, expert in enumerate(indices.flatten().tolist()):
        gate_up = x[row] @ reference.gate_and_up_projs[expert]
        gate_up = (
            gate_up
            + (
                x[row]
                @ reference.lora_gate_and_up_A[expert].bfloat16()
                @ reference.lora_gate_and_up_B[expert].bfloat16()
            )
            * reference.scale
        )
        gate, up = gate_up.chunk(2)
        hidden = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
        rows.append(
            hidden @ reference.down_projs[expert]
            + (hidden @ reference.lora_down_A[expert].bfloat16() @ reference.lora_down_B[expert].bfloat16())
            * reference.scale
        )
    expected = torch.stack(rows)
    torch.testing.assert_close(output, expected, rtol=0.02, atol=0.001)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)
    frozen = {name: p.detach().clone() for name, p in module.named_parameters() if not p.requires_grad}
    for name, parameter in module.named_parameters():
        if parameter.requires_grad:
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            torch.testing.assert_close(parameter.grad, reference.get_parameter(name).grad, rtol=0.03, atol=0.001)
        else:
            assert parameter.grad is None
    optimizer = torch.optim.SGD([p for p in module.parameters() if p.requires_grad], lr=0.1)
    optimizer.step()
    for name, value in frozen.items():
        torch.testing.assert_close(module.get_parameter(name), value, rtol=0, atol=0)


def _distributed_storage_worker(rank, world_size, rendezvous):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor, Shard
    from torch.distributed.tensor.parallel import parallelize_module

    from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_by_dtype
    from nemo_automodel.components.moe.parallelizer import ExpertParallel

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=world_size)
    try:
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("fsdp", "ep"))
        flat_mesh = mesh["fsdp", "ep"]._flatten()
        for mode in ("fsdp", "ep", "ep_fsdp"):
            torch.manual_seed(17)
            model = patch_moe_module(GroupedExperts(tiny_moe_config()), dim=4)
            reference = randomize_and_copy(model)
            if mode != "fsdp":
                ep_mesh = flat_mesh if mode == "ep" else mesh["ep"]
                parallelize_module(model, ep_mesh, ExpertParallel())
            if mode == "fsdp":
                fully_shard_by_dtype(model, flat_mesh, policy, None, reshard_after_forward=True)
            elif mode == "ep_fsdp":
                fully_shard(
                    model,
                    mesh=mesh["fsdp"],
                    mp_policy=policy,
                    shard_placement_fn=lambda parameter: Shard(1),
                    reshard_after_forward=True,
                )
            cast_model_to_dtype(model, torch.bfloat16)
            cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
            for name, p in model.named_parameters():
                assert p.dtype == (torch.float32 if name.startswith("lora_") else torch.bfloat16)
            x = torch.randn(8, 16, dtype=torch.bfloat16)
            mask, weights, indices = expert_routing_inputs(x)
            compute_dtypes = []
            hook = model.register_forward_pre_hook(
                lambda module, inputs: compute_dtypes.extend(p.dtype for p in module.parameters())
            )
            actual = model(x, mask, weights, indices)
            hook.remove()
            if mode != "ep":
                assert compute_dtypes and set(compute_dtypes) == {torch.bfloat16}
            expected = reference(x, mask, weights, indices)
            torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.001)
            upstream = torch.randn_like(actual)
            ep_size = 1 if mode == "fsdp" else ep_mesh.size()
            (actual * upstream).float().sum().div(ep_size).backward()
            (expected * upstream).float().sum().backward()
            for name, p in model.named_parameters():
                if p.requires_grad:
                    assert p.grad.dtype == torch.float32
                    grad = p.grad.full_tensor() if isinstance(p.grad, DTensor) else p.grad
                    # Allow small absolute differences near zero in BF16 gradient comparisons.
                    torch.testing.assert_close(
                        grad,
                        reference.get_parameter(name).grad,
                        rtol=0.03,
                        atol=0.002,
                        msg=lambda message: f"{mode}/{name}: {message}",
                    )
                else:
                    assert p.grad is None
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1)
            ref_optimizer = torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.1)
            optimizer.step()
            ref_optimizer.step()
            for name, p in model.named_parameters():
                full = p.full_tensor() if isinstance(p, DTensor) else p
                torch.testing.assert_close(full, reference.get_parameter(name), rtol=0.03, atol=0.001)
        # TP rewrites adapter projection classes; their compute casts must survive that rewrite.
        from nemo_automodel.components.distributed.parallel_styles import ColwiseParallelLora, RowwiseParallelLora

        torch.manual_seed(29)
        model = nn.Sequential(
            LinearLoRA(nn.Linear(16, 32, bias=False, dtype=torch.bfloat16), dim=4),
            LinearLoRA(nn.Linear(32, 16, bias=False, dtype=torch.bfloat16), dim=4),
        )
        for layer in model:
            nn.init.normal_(layer.lora_B.weight, std=0.02)
        reference = copy.deepcopy(model)
        parallelize_module(model, flat_mesh, {"0": ColwiseParallelLora(), "1": RowwiseParallelLora()})
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16)
        actual, expected = model(x), reference(x)
        torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.004)
        actual.sum().backward()
        expected.sum().backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.dtype == p.grad.dtype == torch.float32
                torch.testing.assert_close(
                    p.grad.full_tensor(), reference.get_parameter(name).grad, rtol=0.04, atol=0.01
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.timeout(120)
@pytest.mark.run_only_on("GPU")
@pytest.mark.skipif(not is_torch_min_version("2.8.0"), reason="CPU FSDP validation needs PyTorch >= 2.8")
def test_storage_and_gradients_with_ep_and_fsdp(tmp_path):
    # Four CPU ranks exercise the smallest genuine EP=2 + FSDP=2 composition.
    import torch.multiprocessing as mp

    mp.spawn(_distributed_storage_worker, args=(4, f"file://{tmp_path / 'rendezvous'}"), nprocs=4, join=True)


@pytest.mark.run_only_on("GPU")
def test_qwen_moe_meta_initialization_and_checkpoint_round_trip(tmp_path):
    from functools import partial

    from transformers import Qwen3MoeConfig

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        num_hidden_layers=1,
        intermediate_size=32,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        torch_dtype=torch.bfloat16,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )

    def build():
        with torch.device("meta"):
            model = Qwen3MoeForCausalLM(config, backend=backend)
            apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*"], dim=4))
        # Production initialize_weights defaults to CUDA; supply its CPU buffer device.
        model.initialize_weights = partial(model.initialize_weights, buffer_device=torch.device("cpu"))
        Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier")
        cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
        return model

    model = build()
    inputs = torch.arange(8).view(1, -1)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    logits = model(inputs).logits
    assert logits.dtype == torch.bfloat16
    logits.float().square().mean().backward()
    for name, parameter in model.named_parameters():
        expected_dtype = torch.float32 if "lora_" in name else torch.bfloat16
        assert parameter.dtype == expected_dtype, name
        if parameter.requires_grad:
            assert parameter.grad is not None and parameter.grad.dtype == torch.float32, name
    optimizer.step()
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
    restored = build()
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.001)
    checkpoint = torch.load(path, weights_only=True)
    restored.load_state_dict(checkpoint["model"])
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for name, parameter in restored.named_parameters():
        torch.testing.assert_close(parameter, model.get_parameter(name), rtol=0, atol=0)
    torch.testing.assert_close(restored(inputs).logits, model(inputs).logits, rtol=0, atol=0)
    for state in restored_optimizer.state.values():
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.float32


@pytest.mark.parametrize(
    "options, expected",
    [({}, torch.float32), ({"lora_dtype": None}, torch.bfloat16), ({"lora_dtype": "bfloat16"}, torch.bfloat16)],
)
def test_lora_storage_default_and_overrides(options, expected):
    config = PeftConfig(**options)
    assert PeftConfig.from_dict({}).lora_dtype == torch.float32
    assert PeftConfig.from_dict(config.to_dict()).lora_dtype == config.lora_dtype
    linear = patch_linear_module(nn.Linear(16, 16, dtype=torch.bfloat16), use_triton=False, **options)
    assert linear.lora_A.weight.dtype == linear.lora_B.weight.dtype == expected
    for expert_cls in (GroupedExperts, GroupedExpertsDeepEP):
        expert = patch_moe_module(expert_cls(tiny_moe_config()), **options)
        assert all(p.dtype == expected for name, p in expert.named_parameters() if name.startswith("lora_"))


@pytest.mark.parametrize(
    "mode",
    ["memory_efficient", "plain", "dora", pytest.param("compile", marks=pytest.mark.run_only_on("GPU"))],
)
def test_default_dense_lora_bf16_compute_without_autocast(mode):
    from torch.utils._python_dispatch import TorchDispatchMode

    class MatmulDtypes(TorchDispatchMode):
        def __init__(self):
            self.calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.addmm.default):
                operands = args[1:3] if func == torch.ops.aten.addmm.default else args[:2]
                assert all(t.dtype == torch.bfloat16 for t in operands), (func, operands)
                self.calls += 1
            return func(*args, **(kwargs or {}))

    torch.manual_seed(11)
    device = "cuda" if mode == "compile" else "cpu"
    module = LinearLoRA(
        nn.Linear(16, 16, bias=False, dtype=torch.bfloat16, device=device),
        dim=4,
        alpha=8,
        use_dora=mode == "dora",
        use_memory_efficient_lora=mode != "plain",
    )
    nn.init.normal_(module.lora_B.weight, std=0.02)
    reference = copy.deepcopy(module).to(torch.bfloat16)
    x = torch.randn(2, 4, 16, dtype=torch.bfloat16, device=device, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_(True)
    expected = reference(ref_x)
    audit = MatmulDtypes()
    if mode == "compile":
        from functools import partial

        from functorch.compile import make_boxed_func
        from torch._dynamo.backends.common import aot_autograd

        graph_gemms = {"forward": 0, "backward": 0}

        def compile_graph(graph, example_inputs, *, phase):
            def execute(*args):
                # Enter dispatch mode inside the captured graph, never around Dynamo.
                before = audit.calls
                with audit:
                    result = graph(*args)
                graph_gemms[phase] += audit.calls - before
                return result

            return make_boxed_func(execute)

        backend = aot_autograd(
            fw_compiler=partial(compile_graph, phase="forward"),
            bw_compiler=partial(compile_graph, phase="backward"),
        )
        run = torch.compile(module, backend=backend, fullgraph=True)
        actual = run(x)
        actual.sum().backward()
        assert all(count > 0 for count in graph_gemms.values()), graph_gemms
    else:
        with audit:
            actual = module(x)
            actual.sum().backward()
    expected.sum().backward()
    assert audit.calls > 0
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.002)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0.02, atol=0.002)
    for name, p in module.named_parameters():
        if p.requires_grad:
            assert p.dtype == p.grad.dtype == torch.float32
            torch.testing.assert_close(p.grad.bfloat16(), reference.get_parameter(name).grad, rtol=0.02, atol=0.002)
        else:
            assert p.dtype == torch.bfloat16 and p.grad is None
