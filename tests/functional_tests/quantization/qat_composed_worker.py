# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Actual tiny-model FSDP2+EP2 QAT gate, not high-level recipe authorization.

Two ranks share the DP/FSDP domain and partition two experts along axis zero;
expert-DP is one. Only expert adapters train. The public MoE parallelizer owns
EP placement, dtype-aware decoder/root FSDP wrapping, and expert exclusion.
There is no manual sharding fallback, guard patch, custom gradient hook, or
extra gradient averaging. A production-path failure is a FAILED gate.

FP32 is true FP32 storage/compute. BF16 preserves model-owned FP32 islands and
uses BF16 expert storage/compute, without autocast or FP32-master/BF16 casts.
Both compare against an entire unsharded, same-device/dtype cloned QAT model
on the concatenated rank batches. This is parallel parity, not an independent
quantizer oracle, bitwise forward parity, or pretrained-model evidence.

One SGD step, no clipping/accumulation/checkpointing/TP/CP/PP, text-only GLM DSA,
top-2 of two real experts, six tokens per rank. Empty/uneven routes and separate
axes belong to qat_distributed_worker; they are not claimed by this gate.
Use an external 180s timeout for direct torchrun; pytest kills the process tree.
Each rank writes incremental JSON, including mismatches and failure traceback.
"""

from __future__ import annotations

import argparse
import copy
import inspect
import json
import os
import sys
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, OffloadPolicy, fully_shard
from torch.distributed.tensor import DTensor, Shard
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.moe.parallelizer import parallelize_model
from nemo_automodel.components.quantization.qat import QATConfig
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig
from tests.unit_tests._transformers.test_lora_qat_models import _tiny_model

_ARCHITECTURES = ("deepseek_v4", "glm5_next")
_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}
_ADAPTERS = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
_TOPOLOGY = {"world_size": 2, "fsdp_size": 2, "ep_size": 2, "expert_dp_size": 1, "tp": 1, "cp": 1, "pp": 1}
_SCOPE = "tiny-full-model-public-moe-parallelizer-expert-only-qat"
_LR = 4.0  # Probe updates must exceed BF16 storage ULPs; not a recommended training LR.
# Predetermined arithmetic bounds, not fitted to GPU results. Relative L2 is
# mandatory even for tiny gradients, and detects missing/extra factors of two.
# BF16 allows cancellation via an RMS-scaled atol; updates have their own L2
# bound because subtraction exposes BF16 storage rounding.
_BOUNDS = {
    "float32": (3e-4, 2e-6, 5e-4, 1e-3, 0.0),
    "bfloat16": (0.08, 0.003, 0.15, 0.25, 0.08),
}


def _paths(architecture: str) -> tuple[str, str]:
    if architecture not in _ARCHITECTURES:
        raise ValueError(f"Unknown architecture: {architecture}")
    block = "model.language_model.layers.0" if architecture == "glm5_next" else "model.layers.0"
    return block, block + ".mlp.experts"


def _quantization(architecture: str) -> WeightQuantizationConfig:
    _paths(architecture)
    if architecture == "glm5_next":
        return WeightQuantizationConfig("fp8", (128, 128), scale_format="float32")
    return WeightQuantizationConfig("mxfp4", (1, 32), scale_format="e8m0")


def _model(architecture: str, dtype: str, device: torch.device) -> nn.Module:
    """Prepare the actual initialized tiny model, with expert-only PEFT then public QAT."""
    torch.manual_seed(19)
    model = _tiny_model(architecture, _DTYPES[dtype]).to(device=device)
    _, expert_path = _paths(architecture)
    assert isinstance(model, MoEFSDPSyncMixin)
    assert model.backend.dispatcher == model.backend.experts == "torch"
    # The single-rank helper defaults this public backend setting to False.
    # Opt in explicitly so the production mixin's lifecycle methods really run.
    model.backend = replace(model.backend, enable_fsdp_optimizations=True)
    patched = apply_lora_to_linear_modules(
        model, PeftConfig(target_modules=[expert_path], dim=4, alpha=8, use_memory_efficient_lora=False)
    )
    assert patched == 1
    experts = model.get_submodule(expert_path)
    assert isinstance(experts, GroupedExpertsLoRA)
    assert experts.n_routed_experts == experts.config.n_activated_experts == 2
    assert experts.config.aux_loss_coeff == 0, "The gate measures CE only"
    assert not experts.use_torch_mm and not experts.use_mxfp8
    expected = {f"{expert_path}.{name}" for name in _ADAPTERS}
    assert {name for name, p in model.named_parameters() if p.requires_grad} == expected
    # Nonzero A and B exercise all four gradients in the first backward.
    with torch.no_grad():
        for name in _ADAPTERS:
            experts.get_parameter(name).normal_(std=0.08)
    identities = {name: id(p) for name, p in model.named_parameters()}
    assert QAT(QATConfig(target_modules=(expert_path,), weight=_quantization(architecture))).prepare(model) is model
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert isinstance(experts.weight_fake_quantizer, WeightFakeQuantizer)
    assert experts.weight_fake_quantizer.config == _quantization(architecture)
    assert sum(isinstance(module, WeightFakeQuantizer) for module in model.modules()) == 1
    assert all(p.dtype == _DTYPES[dtype] for p in experts.parameters())
    if dtype == "float32":
        assert all(p.dtype == torch.float32 for p in model.parameters())
    return model.train()


def _tokens(device: torch.device) -> torch.Tensor:
    """Return integer tokens [global_batch=2, sequence=6], one distinct row per rank."""
    return torch.tensor([[1, 5, 9, 13, 17, 21], [2, 6, 10, 14, 18, 22]], device=device)


def _loss(logits: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Sum local shifted CE divided by the GLOBAL ten supervised tokens.

    Args:
        logits: Local or global tensor [batch, sequence=6, vocab=64].
        tokens: Integer tensor [batch, sequence=6] on logits' device.

    Returns:
        FP32 scalar loss []; the two rank-local losses sum to the reference loss.
    """
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, 64), tokens[:, 1:].reshape(-1), reduction="sum") / 10


def _full(value: torch.Tensor) -> torch.Tensor:
    """Collect before checking; every rank must call this in the same order.

    Args:
        value: Parameter/gradient of arbitrary shape; DTensors have global
            shape and Shard(0) on a two-rank FSDP or EP one-dimensional mesh.

    Returns:
        Independent CPU tensor with the full logical shape and original dtype.
    """
    return (value.full_tensor() if isinstance(value, DTensor) else value).detach().cpu().clone()


def _check(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: str,
    label: str,
    measurements: list[dict],
) -> None:
    """Record and enforce elementwise AND relative-L2 bounds without collectives.

    Args:
        actual: Full local CPU/device tensor, arbitrary shape, not a DTensor.
        expected: Reference tensor of identical semantic shape, not a DTensor.
        dtype: Selects predeclared arithmetic bounds; zero references require exact zero.
        label: Quantity, with ``update/`` selecting the update L2 bound.
        measurements: JSON evidence list, appended before any numerical failure.
    """
    assert not isinstance(actual, DTensor) and not isinstance(expected, DTensor), label
    rtol, atol, l2_limit, update_limit, rms_atol = _BOUNDS[dtype]
    if label.startswith("update/"):
        l2_limit = update_limit
    a, b = actual.detach().cpu().double(), expected.detach().cpu().double()
    record = {"label": label, "status": "failed", "relative_l2_limit": l2_limit}
    measurements.append(record)
    assert a.shape == b.shape, f"{label}: shape {a.shape} != {b.shape}"
    assert torch.isfinite(a).all() and torch.isfinite(b).all(), f"{label}: non-finite tensor"
    norm = float(b.norm())
    error = float((a - b).norm())
    relative = error / norm if norm else error
    effective_atol = atol + rms_atol * float(b.square().mean().sqrt())
    record.update(
        relative_l2=relative,
        max_abs=float((a - b).abs().max()),
        actual_norm=float(a.norm()),
        expected_norm=norm,
        rtol=rtol,
        atol=effective_atol,
    )
    assert relative <= (l2_limit if norm else 0), f"{label}: relative L2={relative}, limit={l2_limit}"
    torch.testing.assert_close(a, b, rtol=rtol, atol=effective_atol, msg=lambda message: f"{label}: {message}")
    record["status"] = "passed"


@contextmanager
def _synchronized_checks(label: str) -> Generator[None, None, None]:
    """Exchange local failures at a fixed boundary; NEVER put collectives inside."""
    failure = None
    try:
        yield
    except Exception:
        failure = traceback.format_exc()
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    if any(failures):
        details = "\n".join(f"rank {rank}: {error}" for rank, error in enumerate(failures) if error)
        raise AssertionError(f"{label}: synchronized failure\n{details}")


def _forward(model: nn.Module, tokens: torch.Tensor, expert_path: str) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Observe real routing/QAT without replacing modules, tensors, or outputs.

    Args:
        model: Real QAT model, either whole unsharded reference or composed FSDP/EP.
        tokens: Integer tensor [batch, sequence=6], local batch=1 or global batch=2.
        expert_path: Exact expert module path for this tiny architecture.

    Returns:
        Differentiable logits [batch, sequence, vocab=64] and detached CPU
        observations: input/output [batch*sequence, hidden=128], routing_weights
        and indices [batch*sequence, top_k=2], mask [batch*sequence], quantized
        operands qdq_input_0 [owned_experts, 2*intermediate, hidden] and
        qdq_input_1 [owned_experts, hidden, intermediate]. owned_experts is one
        under EP, two for the reference; intermediate is 64 (V4) or 128 (GLM).
    """
    captured = {}
    experts = model.get_submodule(expert_path)
    calls = []

    def observe(module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        """Capture expert boundaries.

        Args:
            module: Actual expert module (EP parameters remain DTensors inside forward).
            args: Input [tokens, hidden], mask [tokens], routing weights and
                indices [tokens, top_k=2], all local tensors.
            output: Local tensor [tokens, hidden].
        """
        calls.append(True)
        for name, tensor in zip(("input", "mask", "routing_weights", "indices", "output"), (*args, output)):
            captured[name] = tensor.detach().cpu().clone()

    def observe_qdq(module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Capture canonical merged weights passed into real QDQ.

        Args:
            module: Configured fake quantizer.
            args: One local tensor [owned_experts, out, in], never a DTensor.
        """
        weight = args[0]
        assert not isinstance(weight, DTensor)
        captured[f"qdq_input_{sum(name.startswith('qdq_input_') for name in captured)}"] = weight.detach().cpu().clone()

    handles = [
        experts.register_forward_hook(observe),
        experts.weight_fake_quantizer.register_forward_pre_hook(observe_qdq),
    ]
    try:
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(tokens).logits
    finally:
        for handle in handles:
            handle.remove()
    assert len(calls) == 1 and len(captured) == 7, "Real expert and both QDQ forwards must execute once"
    return logits, captured


def _run_case(architecture: str, dtype: str, result: dict, save: Callable[[], None]) -> None:
    rank = dist.get_rank()
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    block_path, expert_path = _paths(architecture)
    checks = result["measurements"]
    with _synchronized_checks("model/reference preparation"):
        model = _model(architecture, dtype, device)
        reference = copy.deepcopy(model)
        assert not any(isinstance(p, DTensor) for p in reference.parameters())
        initial = {name: p.detach().cpu().clone() for name, p in reference.named_parameters()}
        adapter_names = [f"{expert_path}.{name}" for name in _ADAPTERS]

    result["stage"] = "public MoE parallelization"
    save()
    # Build production topology via its public config boundary. No high-level
    # AutoModel/QAT entry point is called or monkeypatched. MoEParallelizerConfig
    # currently has no build method; use the existing public parallelize_model.
    policy = MixedPrecisionPolicy(cast_forward_inputs=False)
    setup = DistributedSetup.build(
        strategy=FSDP2Config(mp_policy=policy, activation_checkpointing=False, enable_compile=False),
        parallelism_sizes=ParallelismSizes(dp_size=2, ep_size=2),
        world_size=2,
        timeout_minutes=1,
    )
    mesh = setup.mesh_context
    parallelize_model(
        model,
        mesh.device_mesh,
        mesh.moe_mesh,
        **mesh.parallelize_axis_kwargs(),
        mp_policy=policy,
        offload_policy=OffloadPolicy(),
        activation_checkpointing=False,
        reshard_after_forward=True,
        wrap_outer_model=True,
    )
    with _synchronized_checks("ownership"):
        assert mesh.ep_size == 2 and mesh.moe_mesh["ep_shard"].size() == 1
        assert isinstance(model, FSDPModule) and isinstance(model.get_submodule(block_path), FSDPModule)
        experts = model.get_submodule(expert_path)
        assert not isinstance(experts, FSDPModule), "EP-owned experts must not acquire child FSDP ownership"
        ep_names, fsdp_names = [], []
        for name, p in model.named_parameters():
            assert isinstance(p, DTensor) and p.placements == (Shard(0),), name
            assert p.device_mesh.ndim == 1 and p.device_mesh.size() == 2, name
            if name.startswith(expert_path + "."):
                assert p.shape[0] == 2 and p.to_local().shape[0] == 1, name
                assert p.device_mesh == mesh.moe_mesh["ep"], name
                ep_names.append(name)
            else:
                assert not p.requires_grad, f"Dense/shared/router parameter unexpectedly trains: {name}"
                fsdp_names.append(name)
        assert len(ep_names) == 6 and fsdp_names
        assert {name for name, p in model.named_parameters() if p.requires_grad} == set(adapter_names)
        result["ownership"] = {
            "ep_parameters": ep_names,
            "fsdp_parameters": fsdp_names,
            "fsdp_units": [name for name, module in model.named_modules() if isinstance(module, FSDPModule)],
            "trainable_parameters": adapter_names,
        }
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=_LR, foreach=False)
        reference_optimizer = torch.optim.SGD(
            (p for p in reference.parameters() if p.requires_grad), lr=_LR, foreach=False
        )
        assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
            id(p) for p in model.parameters() if p.requires_grad
        }

    # Gather ALL parameters before local checks so a mismatch never skips a collective.
    sharded_initial = {name: _full(p) for name, p in model.named_parameters()}
    with _synchronized_checks("initial state"):
        assert initial.keys() == sharded_initial.keys()
        for name in initial:
            torch.testing.assert_close(sharded_initial[name], initial[name], rtol=0, atol=0)

    result["stage"] = "full-model forward/backward"
    save()
    tokens = _tokens(device)
    with _synchronized_checks("reference forward/backward"):
        reference_logits, reference_trace = _forward(reference, tokens, expert_path)
        reference_loss = _loss(reference_logits, tokens)
        reference_loss.backward()
    # Invoke the production mixin exactly as a one-microbatch training window.
    # It configures FSDP lifecycle/sync, not EP gradient normalization. Frozen
    # dense FSDP units have no trainable grads to average. EP combine's SUM
    # backward assembles disjoint CE slices already divided by ten: no *2,
    # /2, manual all_reduce, clipping, or custom expert hooks are appropriate.
    model.prepare_for_grad_accumulation()
    logits, trace = _forward(model, tokens[rank : rank + 1], expert_path)
    loss = _loss(logits, tokens[rank : rank + 1])
    model.prepare_for_final_backward()
    loss.backward()
    global_loss = loss.detach().clone()
    dist.all_reduce(global_loss)  # Reporting only: never part of the training graph.
    with _synchronized_checks("gradient presence"):
        for name, p in model.named_parameters():
            other = reference.get_parameter(name)
            if p.requires_grad:
                assert p.grad is not None and other.grad is not None, name
                assert isinstance(p.grad, DTensor) and p.grad.placements == (Shard(0),), name
                assert p.grad.device_mesh == mesh.moe_mesh["ep"], name
                assert torch.isfinite(p.grad.to_local()).all() and torch.count_nonzero(p.grad.to_local()), name
            else:
                assert p.grad is None and other.grad is None, name
    gradients = {name: _full(model.get_parameter(name).grad) for name in adapter_names}
    with _synchronized_checks("numerical parity"):
        _check(logits, reference_logits[rank : rank + 1], dtype, "logits", checks)
        _check(global_loss, reference_loss, dtype, "loss", checks)
        for name in ("input", "output", "routing_weights"):
            _check(trace[name], reference_trace[name][rank * 6 : (rank + 1) * 6], dtype, name, checks)
        for name in ("mask", "indices"):
            assert torch.equal(trace[name], reference_trace[name][rank * 6 : (rank + 1) * 6]), name
        assert trace["mask"].all()
        assert torch.equal(trace["indices"].sort(-1).values, torch.tensor([0, 1]).expand(6, 2))
        result["routing"] = {"tokens_per_rank": 6, "top_k": 2, "expert_load": [6, 6], "matched": True}
        for index in range(2):
            key = f"qdq_input_{index}"
            assert trace[key].dtype == _DTYPES[dtype]
            # Merge happens independently per expert; sharding must not alter
            # the quantizer's operands at all, even in BF16.
            torch.testing.assert_close(trace[key], reference_trace[key][rank : rank + 1], rtol=0, atol=0)
        result["qat_calls"] = 2
        for name, grad in gradients.items():
            expected = reference.get_parameter(name).grad
            assert torch.count_nonzero(grad) and torch.count_nonzero(expected), name
            _check(grad, expected, dtype, f"gradient/{name}", checks)
        _check(
            torch.cat([g.double().flatten() for g in gradients.values()]).norm(),
            torch.cat([reference.get_parameter(name).grad.double().flatten() for name in adapter_names]).norm(),
            dtype,
            "expert_gradient_norm",
            checks,
        )

    result["stage"] = "optimizer step"
    save()
    optimizer.step()
    reference_optimizer.step()
    after = {name: _full(p) for name, p in model.named_parameters()}
    with _synchronized_checks("updates/frozen parameters"):
        for name, p in reference.named_parameters():
            if p.requires_grad:
                _check(after[name], p, dtype, f"parameter/{name}", checks)
                delta = after[name].float() - initial[name].float()
                expected_delta = p.detach().cpu().float() - initial[name].float()
                assert torch.count_nonzero(delta) and torch.count_nonzero(expected_delta), name
                _check(delta, expected_delta, dtype, f"update/{name}", checks)
            else:
                torch.testing.assert_close(after[name], initial[name], rtol=0, atol=0)
                torch.testing.assert_close(p.detach().cpu(), initial[name], rtol=0, atol=0)
        result["frozen_parameters_unchanged"] = True
    result.update(status="passed", stage="complete", supported_topology=dict(_TOPOLOGY))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=(*_ARCHITECTURES, "all"), default="all")
    parser.add_argument("--dtype", choices=(*_DTYPES, "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    """Run under two-rank torchrun; prerequisites or production failures are never skips."""
    args = _parser().parse_args()
    rank = int(os.environ.get("RANK", "0"))
    report = {
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "torch_version": torch.__version__,
        "scope": _SCOPE,
        "status": "failed",
        "high_level_guard_lifted": False,
        "requested_topology": dict(_TOPOLOGY),
        "supported_topology": None,
        "tests": [],
    }
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / f"rank-{rank}.json"

    def save() -> None:
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    save()
    try:
        assert report["world_size"] == 2, "Exactly two torchrun ranks required"
        assert "ignored_params" in inspect.signature(fully_shard).parameters, "Torch >=2.10 ignored_params API required"
        assert torch.cuda.is_available() and torch.cuda.device_count() >= 2, "Two visible CUDA GPUs required"
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dtypes = tuple(_DTYPES) if args.dtype == "all" else (args.dtype,)
        if "bfloat16" in dtypes:
            assert torch.cuda.is_bf16_supported(including_emulation=False), "Native BF16 GPU support required"
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
        architectures = _ARCHITECTURES if args.architecture == "all" else (args.architecture,)
        with torch.compiler.set_stance("force_eager"):
            for architecture in architectures:
                for dtype in dtypes:
                    quantization = _quantization(architecture)
                    result = {
                        "architecture": architecture,
                        "dtype": dtype,
                        "status": "failed",
                        "stage": "preparation",
                        "supported_topology": None,
                        "quantization": {
                            "format": quantization.format,
                            "block_size": list(quantization.block_size),
                            "scale_format": quantization.scale_format,
                        },
                        "reference": "whole-model-clone-global-batch-same-dtype",
                        "loss_denominator": 10,
                        "expert_gradient_rescale": 1,
                        "steps": 1,
                        "measurements": [],
                    }
                    report["tests"].append(result)
                    save()
                    _run_case(architecture, dtype, result, save)
                    save()
        report.update(status="passed", supported_topology=dict(_TOPOLOGY))
    except Exception:
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
