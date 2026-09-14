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

"""Real EP forward/backward parity against a CPU FP32 expert reference.

Run with pytest (two GPUs), or select a case for diagnosis with:
    torchrun --standalone --nproc-per-node=2 tests/functional_tests/moe/test_mxfp4_ep.py \
        --dispatcher torch --variants mxfp4_lora --cases ragged

Base weights are MXFP4-representable in both implementations, and LoRA B is
nonzero so adapter A gradients are exercised. No dispatch or collective is mocked.
"""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import parallelize_module
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components._peft.lora import patch_moe_module
from nemo_automodel.components.distributed.init_utils import destroy_global_state, initialize_distributed
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.parallelizer import ExpertParallel
from nemo_automodel.components.moe.quantized_experts import GroupedExpertsDeepEPMXFP4, GroupedExpertsMXFP4
from nemo_automodel.components.quantization.mxfp4 import dequantize_mxfp4, quantize_mxfp4

_VARIANTS = ("plain", "lora", "mxfp4", "mxfp4_lora")
_CASES = ("balanced", "ragged", "ragged_checkpoint", "empty_rank_checkpoint")
_HIDDEN = 512
_INTER = 256
_EXPERTS = 4


def _model(variant: str, dispatcher: str, device: torch.device) -> torch.nn.Module:
    """Construct identical rounded weights for the CPU reference and CUDA implementation."""
    config = MoEConfig(
        n_routed_experts=_EXPERTS,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="softmax",
        route_scale=1.0,
        dim=_HIDDEN,
        inter_dim=_INTER,
        moe_inter_dim=_INTER,
        norm_topk_prob=False,
        dtype=torch.float32 if device.type == "cpu" else torch.bfloat16,
    )
    backend = BackendConfig(experts="torch_mm", dispatcher=dispatcher)
    deep = device.type == "cuda" and dispatcher != "torch"
    with torch.device(device):
        orig = (
            GroupedExpertsDeepEP(
                config, backend, dispatcher_backend=dispatcher, dispatcher_share_token_dispatcher=False
            )
            if deep
            else GroupedExperts(config, backend if device.type == "cuda" else None)
        )
    generator = torch.Generator().manual_seed(4321)
    with torch.no_grad():
        for param in orig.parameters():
            source = (torch.randn(param.shape, generator=generator) * 0.04).to(torch.bfloat16)
            packed, scales = quantize_mxfp4(source.transpose(-2, -1).contiguous())
            source = dequantize_mxfp4(packed, scales, torch.bfloat16).transpose(-2, -1)
            param.copy_(source.to(device=device, dtype=param.dtype))
    if "lora" in variant:
        with torch.device(device):
            model = patch_moe_module(
                orig,
                dim=8,
                alpha=16,
                expert_weight_format="mxfp4" if device.type == "cuda" and variant == "mxfp4_lora" else "unquantized",
            )
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name.startswith("lora_"):
                    source = (torch.randn(param.shape, generator=generator) * 0.04).to(torch.bfloat16)
                    param.copy_(source.to(device=device, dtype=param.dtype))
    elif variant == "mxfp4":
        if device.type == "cpu":
            model = orig.requires_grad_(False)
        else:
            cls = GroupedExpertsDeepEPMXFP4 if deep else GroupedExpertsMXFP4
            with torch.device(device):
                model = cls(orig)
    else:
        model = orig
    return model


def _compare(actual: torch.Tensor, expected: torch.Tensor, name: str) -> dict[str, float]:
    """Compare equally shaped values using BF16 error bounds relative to reference scale.

    Args:
        actual: Tensor of arbitrary shape, containing CUDA outputs, gradients, or parameters.
        expected: Tensor of the same shape, containing the CPU FP32 reference values.
        name: Diagnostic name for a failed comparison.

    Returns:
        Maximum absolute error and relative L2 error. A two-percent scale bound permits
        BF16 GEMM/activation rounding while detecting missing gradients or EP scaling errors.
    """
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
    assert torch.isfinite(actual).all(), name
    delta = actual - expected
    relative_l2 = delta.norm().item() / max(expected.norm().item(), 1e-8)
    max_abs = delta.abs().max().item()
    bound = expected.abs().max().item() * 0.02 + 1e-7
    assert relative_l2 <= 0.02 and max_abs <= bound, (name, relative_l2, max_abs, bound)
    return {"relative_l2": relative_l2, "max_abs": max_abs}


def _run_case(variant: str, dispatcher: str, case: str) -> None:
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    lengths = [16] * world if case == "balanced" else [17 + 14 * r for r in range(world)]
    start, count = sum(lengths[:rank]), lengths[rank]
    generator = torch.Generator().manual_seed(2468)
    x = torch.randn(sum(lengths), _HIDDEN, generator=generator).to(torch.bfloat16).float().requires_grad_()
    weights = torch.rand(sum(lengths), 2, generator=generator)
    weights = (weights / weights.sum(-1, keepdim=True)).requires_grad_()
    indices = torch.stack([torch.randperm(_EXPERTS, generator=generator)[:2] for _ in range(sum(lengths))])
    if case == "empty_rank_checkpoint":
        indices[:] = torch.tensor([0, 1])  # rank 1's experts receive no tokens at EP=2
    mask = torch.ones(sum(lengths), dtype=torch.bool)
    mask[0] = False
    upstream = torch.randn(x.shape, generator=generator).to(torch.bfloat16).float()

    reference = _model(variant, "torch", torch.device("cpu"))
    expected = reference(x, mask, weights, indices)
    expected.backward(upstream)
    assert x.grad is not None and weights.grad is not None

    model = _model(variant, dispatcher, device)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("ep",))
    # Match checkpoint passthrough: shard meta parameters before loading their local values.
    # NCCL cannot broadcast float8_e8m0fnu scales from materialized full parameters.
    full_params = {name: param.detach().cpu().clone() for name, param in model.named_parameters()}
    model.to("meta")
    parallelize_module(model, mesh, ExpertParallel())
    model.to_empty(device=device)
    with torch.no_grad():
        for name, param in model.named_parameters():
            first = rank * (_EXPERTS // world)
            param.to_local().copy_(full_params[name][first : first + _EXPERTS // world])
    local_x = x.detach()[start : start + count].to(device=device, dtype=torch.bfloat16).requires_grad_()
    local_weights = weights.detach()[start : start + count].to(device).requires_grad_()
    args = (local_x, mask[start : start + count].to(device), local_weights, indices[start : start + count].to(device))
    output = checkpoint(model, *args, use_reentrant=False) if "checkpoint" in case else model(*args)
    checks = {"output": _compare(output, expected[start : start + count], "output")}
    output.backward(upstream[start : start + count].to(device=device, dtype=output.dtype))
    assert local_x.grad is not None and local_weights.grad is not None
    checks["input_grad"] = _compare(local_x.grad, x.grad[start : start + count], "input_grad")
    checks["router_grad"] = _compare(local_weights.grad, weights.grad[start : start + count], "router_grad")

    ref_params = dict(reference.named_parameters())
    expert_start = rank * (_EXPERTS // world)
    expert_end = expert_start + _EXPERTS // world
    trainable = [p for p in model.parameters() if p.requires_grad]
    local_norm_sq = torch.zeros((), device=device)
    expected_norm_sq = torch.zeros(())
    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None
            continue
        assert param.grad is not None, name
        ref_grad = ref_params[name].grad
        assert ref_grad is not None, name
        checks[name + ".grad"] = _compare(param.grad.to_local(), ref_grad[expert_start:expert_end], name)
        local_norm_sq += param.grad.to_local().float().square().sum()
        expected_norm_sq += ref_grad.float().square().sum()
    if trainable:
        dist.all_reduce(local_norm_sq)
        checks["grad_norm"] = _compare(local_norm_sq.sqrt(), expected_norm_sq.sqrt(), "grad_norm")
        torch.optim.SGD(trainable, lr=0.1).step()
        torch.optim.SGD([p for p in reference.parameters() if p.requires_grad], lr=0.1).step()
        for name, param in model.named_parameters():
            if param.requires_grad:
                checks[name + ".step"] = _compare(param.to_local(), ref_params[name][expert_start:expert_end], name)
    print(
        json.dumps({"rank": rank, "dispatcher": dispatcher, "variant": variant, "case": case, "checks": checks}),
        flush=True,
    )
    dist.barrier()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dispatcher", choices=("torch", "deepep", "hybridep"), default="torch")
    parser.add_argument("--variants", nargs="+", choices=_VARIANTS, default=list(_VARIANTS))
    parser.add_argument("--cases", nargs="+", choices=_CASES, default=list(_CASES))
    args = parser.parse_args()
    initialize_distributed("nccl", timeout_minutes=1)
    try:
        for variant in args.variants:
            for case in args.cases:
                _run_case(variant, args.dispatcher, case)
    finally:
        destroy_global_state()


@pytest.mark.parametrize("dispatcher", ("torch", "deepep", "hybridep"))
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices for real EP collectives")
def test_mxfp4_expert_parallel(dispatcher: str) -> None:
    """Exercise unequal shards and checkpointed empty routing with actual dispatchers."""
    package = {"deepep": "deep_ep", "hybridep": "hybrid_ep_cpp"}.get(dispatcher)
    if package is not None and importlib.util.find_spec(package) is None:
        pytest.skip(f"{dispatcher} requires the optional {package} runtime")
    env = os.environ.copy()
    repo_root = Path(__file__).resolve().parents[3]
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(repo_root), env.get("PYTHONPATH", ""))))
    visible = env.get("CUDA_VISIBLE_DEVICES", "0,1").split(",")
    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:2])
    env["NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN"] = "2"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc-per-node=2",
            str(Path(__file__).resolve()),
            "--dispatcher",
            dispatcher,
        ],
        env=env,
        check=True,
        timeout=300,
    )


if __name__ == "__main__":
    main()
