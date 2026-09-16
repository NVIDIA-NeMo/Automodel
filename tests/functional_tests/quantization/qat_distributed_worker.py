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

"""Real FSDP2 / gather-based EP2 QAT probes, without high-level guard bypasses.

Use two CUDA GPUs and Torch >= 2.10. No DeepEP, HybridEP, grouped_gemm, model
downloads, native FP8/FP4 GEMMs, autocast, or production clipping are involved.
The torch expert loop and real collectives run eagerly (including decorated
activation functions) to keep compilation outside this bounded numerical gate.

Parallel parity uses an unsharded, same-device/dtype actual QAT model. A separate
FP32 oracle uses independent linear/token-expert algebra and an identity STE,
but rounds the merge operations and decoded weights to the forward dtype BEFORE
FP32 algebra. Otherwise BF16 rounding can select different quantization codes.
Both controls start each step from the distributed pre-step parameters/momentum:
updates are checked before reconditioning, never hidden by copying after a step.
Unconditioned FP32 trajectory drift is a separate CPU precision regression.
FSDP and EP are separate axes here: NO composed-topology support is claimed.

Reports are incrementally written to OUTPUT/rank-{rank}.json, including failed
case names and tracebacks; failures propagate as nonzero exits, never skips.
The pytest launcher bounds the whole subprocess tree at 180 seconds. Direct
launchers should impose the same external timeout (collectives also have 60s).
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn.functional as F
from torch import nn
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor, Shard, distribute_tensor

# Direct script execution must work without an editable install/PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts
from nemo_automodel.components.quantization.qat import QATConfig
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig


@dataclass(frozen=True)
class _Tolerance:
    # Predeclared, never fitted to GPU observations. Elementwise checks guard
    # outliers; per-tensor relative L2 prevents small gradients passing via atol.
    rtol: float
    atol: float
    relative_l2: float
    update_relative_l2: float
    rms_atol: float = 0.0


_TOLERANCES = {
    torch.float32: _Tolerance(3e-4, 2e-6, 5e-4, 1e-3),
    # Bounds remain unchanged; quantization-bin changes are NOT arithmetic error.
    # They remain far below a missing/extra factor of two in gradient norms.
    # Near-zero outputs/gradients suffer cancellation: permit 8% of the
    # reference tensor RMS as absolute error, while retaining the L2 bound.
    torch.bfloat16: _Tolerance(0.08, 0.003, 0.15, 0.25, rms_atol=0.08),
}
_QUANTIZATIONS = {
    "fp8-e8m0": WeightQuantizationConfig("fp8", (32, 32)),
    "fp8-float32": WeightQuantizationConfig("fp8", (128, 128), scale_format="float32"),
    "mxfp4-e8m0": WeightQuantizationConfig("mxfp4"),
}
_EP_ROUTES = ("uneven", "before-down", "zero-rank", "empty-owner", "masked", "all-empty")


@dataclass(frozen=True)
class _Case:
    mode: str
    dtype: torch.dtype
    quantization: str
    route: str = "uneven"

    @property
    def name(self) -> str:
        return f"{self.mode}/{self.dtype}/{self.quantization}/{self.route}"

    @property
    def counts(self) -> tuple[int, int]:
        return (0, 0) if self.route == "all-empty" else (0, 5) if self.route == "zero-rank" else (3, 5)

    @property
    def width(self) -> int:
        return 128 if self.quantization == "fp8-float32" else 32

    @property
    def learning_rate(self) -> float:
        # Tiny expert gradients need a larger one-step probe update to exceed
        # BF16 storage ULPs; otherwise unchanged adapters can look "close".
        return 4.0 if self.mode == "ep" else 0.25


@dataclass
class _Batch:
    """Local or concatenated tokens, never DTensors; all fields share a device.

    Attributes:
        x: Floating tensor [tokens, hidden], an autograd leaf.
        target: FP32 tensor [tokens, hidden].
        mask: Boolean tensor [tokens].
        weights: FP32 routing probabilities [tokens, top_k=2], an autograd leaf.
        indices: Int64 global expert IDs [tokens, top_k=2], distinct per token.
    """

    x: torch.Tensor
    target: torch.Tensor
    mask: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor


def _batch(case: _Case, step: int, rank: int | None, device: torch.device) -> _Batch:
    """Generate deterministic rank-specific data, or its FP32 concatenation.

    Returns:
        Batch with layouts documented on _Batch. Rank=None concatenates ranks
        0,1 and widens already storage-rounded inputs to FP32 for the oracle.
    """
    xs, targets, masks, weights, indices = [], [], [], [], []
    for source_rank in range(2) if rank is None else (rank,):
        count = case.counts[source_rank]
        generator = torch.Generator().manual_seed(900 + 17 * step + source_rank)
        xs.append((torch.randn(count, case.width, generator=generator) * 0.25).to(case.dtype))
        targets.append(torch.randn(count, case.width, generator=generator) * 0.25)
        masks.append(torch.full((count,), case.route != "masked", dtype=torch.bool))
        weights.append(torch.rand(count, 2, generator=generator).softmax(-1))
        first = (torch.arange(count) + source_rank) % (1 if case.route == "empty-owner" else 4)
        indices.append(torch.stack((first, (first + 1) % 4), dim=-1))
    return _Batch(
        torch.cat(xs).to(device=device, dtype=torch.float32 if rank is None else case.dtype).requires_grad_(),
        torch.cat(targets).to(device),
        torch.cat(masks).to(device),
        torch.cat(weights).to(device).requires_grad_(),
        torch.cat(indices).to(device),
    )


def _model(case: _Case, seed: int = 123) -> nn.Module:
    """Create initialized/frozen real LoRA modules and prepare QAT before sharding."""
    torch.manual_seed(seed)
    if case.mode == "fsdp":
        model = LinearLoRA(nn.Linear(case.width, case.width, bias=True), dim=8, alpha=8)
    else:
        config = MoEConfig(
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=False,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=case.width,
            inter_dim=case.width,
            moe_inter_dim=case.width,
            norm_topk_prob=True,
            dtype=torch.float32,
            apply_router_weight_after_down=case.route != "before-down",
        )
        base = GroupedExperts(config)  # Default backend: ordinary torch per-expert loops.
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(std=0.1)
        model = GroupedExpertsLoRA(base, lora_dim=8, alpha=8)
    with torch.no_grad():
        # Nonzero A AND B exercise both gradients on the very first step.
        for parameter in model.parameters():
            parameter.normal_(std=0.08 if parameter.requires_grad else 0.1)
    model.to(dtype=case.dtype)
    identities = [id(parameter) for parameter in model.parameters()]
    QAT(QATConfig(target_modules=("",), weight=_QUANTIZATIONS[case.quantization])).prepare(model)
    assert identities == [id(parameter) for parameter in model.parameters()]
    assert isinstance(model.weight_fake_quantizer, WeightFakeQuantizer)
    assert model.weight_fake_quantizer.config == _QUANTIZATIONS[case.quantization].build().config
    assert not model.weight_fake_quantizer.state_dict()
    return model


def _shard(model: nn.Module, case: _Case, mesh: DeviceMesh) -> torch.optim.SGD:
    model.to(device=torch.device(mesh.device_type))
    if case.mode == "fsdp":
        # Shard the root, including its adapter children: forward must see
        # all-gathered plain parameters, not DTensors rejected by LinearLoRA.
        fully_shard(model, mesh=mesh, reshard_after_forward=True)
    else:
        for name, parameter in list(model.named_parameters()):
            assert "." not in name and parameter.ndim == 3 and parameter.shape[0] == 4
            model.register_parameter(
                name, nn.Parameter(distribute_tensor(parameter.detach(), mesh, [Shard(0)]), parameter.requires_grad)
            )
    for parameter in model.parameters():
        assert isinstance(parameter, DTensor) and parameter.placements == (Shard(0),)
        assert parameter.to_local().shape[0] * 2 == parameter.shape[0]
    optimizer = torch.optim.SGD(
        (p for p in model.parameters() if p.requires_grad), lr=case.learning_rate, momentum=0.9, foreach=False
    )
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    return optimizer


def _merge(
    base: torch.Tensor, left: torch.Tensor, right: torch.Tensor, scale: float, forward_dtype: torch.dtype
) -> torch.Tensor:
    """Condition forward rounding while retaining the independent FP32 merge VJP.

    Args:
        base: FP32 tensor [..., out, in] (or expert [..., in, out]).
        left: FP32 tensor [..., out, rank] in the same orientation as base.
        right: FP32 tensor [..., rank, in], with matching leading axes.
        scale: LoRA multiplier.
        forward_dtype: Storage/compute dtype; round matmul, multiply, then add.

    Returns:
        FP32 tensor shaped like base, with dtype-conditioned forward values and
        derivatives of base + scale * (left @ right). No production merge helper.
    """
    smooth = base + scale * (left @ right)
    with torch.no_grad():
        rounded = base.to(forward_dtype) + scale * (left.to(forward_dtype) @ right.to(forward_dtype))
    return rounded.float() + (smooth - smooth.detach())


def _qdq(
    weight: torch.Tensor, quantizer: WeightFakeQuantizer, forward_dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Encode/decode real configured scales, with an independent identity STE.

    Args:
        weight: FP32 canonical tensor [..., out, in], arbitrary leading axes.
        quantizer: The configured stateless numerical quantizer.
        forward_dtype: Decode dtype used by production before widening to FP32.

    Returns:
        Independent FP32 tensor of the same canonical shape, gradient identity.
    """
    decoded = quantizer.quantize(weight.detach()).dequantize(dtype=forward_dtype).float()
    return decoded + (weight - weight.detach())


def _reference(model: nn.Module, batch: _Batch, *, forward_dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Independent FP32 dense/token-expert algebra, not production forward.

    Args:
        model: Unsharded FP32 LoRA storage on batch's device (not its forward).
        batch: Concatenated input fields with layouts documented on _Batch.
        forward_dtype: Condition merge/QDQ on this dtype, retaining FP32 STE
            backward and FP32 activation algebra. FP32 selects the raw oracle.

    Returns:
        FP32 output [global_tokens, hidden], retaining input/adapter gradients.
    """
    quantizer = model.weight_fake_quantizer
    if isinstance(model, LinearLoRA):
        merged = _merge(model.weight, model.lora_B.weight, model.lora_A.weight, model.scale, forward_dtype)
        return F.linear(batch.x, _qdq(merged, quantizer, forward_dtype), model.bias)
    up = _merge(model.gate_and_up_projs, model.lora_gate_and_up_A, model.lora_gate_and_up_B, model.scale, forward_dtype)
    down = _merge(model.down_projs, model.lora_down_A, model.lora_down_B, model.scale, forward_dtype)
    up = _qdq(up.transpose(-1, -2), quantizer, forward_dtype).transpose(-1, -2)
    down = _qdq(down.transpose(-1, -2), quantizer, forward_dtype).transpose(-1, -2)
    # Keep unused experts and empty batches connected, independently of the
    # production empty-route implementation. No collectives or production kernels.
    zero = (up.sum() + down.sum() + batch.x.sum() + batch.weights.sum()) * 0
    tokens = []
    for token in range(batch.x.shape[0]):
        slots = []
        for slot in range(2):
            expert = int(batch.indices[token, slot])
            gate, value = (batch.x[token] @ up[expert]).chunk(2)
            result = (F.silu(gate) * value) @ down[expert]
            # In FP32, pre/post-down weighting is algebraically equivalent;
            # use post-down here rather than duplicating production dispatch.
            slots.append(result * batch.weights[token, slot] * batch.mask[token])
        tokens.append(slots[0] + slots[1])
    return (torch.stack(tokens) if tokens else batch.x * 0) + zero


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Collect a parameter-shaped tensor on every rank, as an independent CPU copy.

    Args:
        tensor: Local tensor of arbitrary shape, or DTensor with global parameter
            shape and Shard(0) placement on the two-rank FSDP/EP mesh.

    Returns:
        CPU tensor of the complete logical shape; never aliases live storage.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


def _check(actual: torch.Tensor, expected: torch.Tensor, tolerance: _Tolerance, label: str) -> float:
    """Check full tensors elementwise AND by relative L2, including exact zeros.

    Args:
        actual: CPU tensor of arbitrary complete logical shape (not a shard).
        expected: FP32 CPU tensor of the same semantic shape as actual.
        tolerance: Predeclared dtype-specific comparison bounds.
        label: Diagnostic including step, parameter name, and quantity.

    Returns:
        Scalar relative L2 error; zero-reference tensors must be exactly zero.
    """
    actual, expected = actual.detach().cpu().float(), expected.detach().cpu().float()
    assert actual.shape == expected.shape, label
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all(), label
    rms = float(expected.square().mean().sqrt()) if expected.numel() else 0.0
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance.rtol,
        atol=tolerance.atol + tolerance.rms_atol * rms,
        msg=lambda message: f"{label}\n{message}",
    )
    norm = torch.linalg.vector_norm(expected)
    error = torch.linalg.vector_norm(actual - expected)
    relative = float(error / norm) if norm else float(error)
    assert relative <= (tolerance.relative_l2 if norm else 0), f"{label}: relative L2={relative}"
    return relative


@contextmanager
def _synchronized_checks(label: str) -> Generator[None, None, None]:
    """Run LOCAL checks only, then exchange failures once at a fixed boundary.

    Never put collectives inside this context: a failed local check skips its
    remainder. All ranks enter once, regardless of token/parameter/check counts.
    Runtime failures inside production collectives still use launcher timeouts.
    """
    failure = None
    try:
        yield
    except Exception:
        failure = traceback.format_exc()
    failures = [None] * dist.get_world_size()
    dist.all_gather_object(failures, failure)
    if any(item is not None for item in failures):
        details = "\n".join(f"rank {rank}: {item}" for rank, item in enumerate(failures) if item is not None)
        raise AssertionError(f"{label}: synchronized check failure\n{details}")


def _backward(model: nn.Module, case: _Case, step: int) -> tuple[torch.Tensor, torch.Tensor, _Batch]:
    """Run rank-local data through real sharded forward/backward.

    Returns:
        Detached CPU output [local_tokens, hidden] and scalar device-local loss,
        normalized by global token count but NOT multiplied by FSDP world size,
        and local batch with gradients (layouts documented on _Batch).
    """
    batch = _batch(case, step, dist.get_rank(), next(model.parameters()).device)
    output = model(batch.x) if case.mode == "fsdp" else model(batch.x, batch.mask, batch.weights, batch.indices)
    loss = (output.float() - batch.target).square().sum() / max(sum(case.counts), 1)
    # FSDP averages reduce-scattered gradients: mean_r(2 * sum_r / N)
    # = sum_all / N. EP combine's autograd all_reduce SUM assembles the
    # disjoint local loss slices; owned expert gradients must NOT be divided
    # by EP size or manually averaged again.
    (loss * (dist.get_world_size() if case.mode == "fsdp" else 1)).backward()
    return output.detach().cpu(), loss.detach(), batch


def _step(
    model: nn.Module,
    optimizer: torch.optim.SGD,
    case: _Case,
    step: int,
) -> dict[str, object]:
    """Check one transition against actual same-dtype QAT AND independent FP32 STE.

    Controls use the full pre-step state, not their own drifting trajectories.
    The distributed optimizer is never reset; its second step and DCP replay
    still exercise real momentum. Gather everything before local assertions.
    """
    tolerance = _TOLERANCES[case.dtype]
    device = next(model.parameters()).device
    optimizer.zero_grad(set_to_none=True)
    before, momentum = {}, {}
    for name, parameter in model.named_parameters():
        before[name] = _full(parameter)
        if step and parameter.requires_grad:
            momentum[name] = _full(optimizer.state[parameter]["momentum_buffer"])
    controls = []
    with _synchronized_checks(f"{case.name}/step-{step}/controls"):
        same_dtype = _model(case).to(device)
        same_dtype.load_state_dict(before, strict=True)
        for label, reference in (
            ("same_dtype_qat", same_dtype),
            ("conditioned_fp32_ste", copy.deepcopy(same_dtype).float()),
        ):
            reference_optimizer = torch.optim.SGD(
                (p for p in reference.parameters() if p.requires_grad),
                lr=case.learning_rate,
                momentum=0.9,
                foreach=False,
            )
            for name, parameter in reference.named_parameters():
                if name in momentum:
                    reference_optimizer.state[parameter]["momentum_buffer"] = momentum[name].to(parameter).clone()
            batch = _batch(case, step, None, device)
            if label == "same_dtype_qat":
                batch.x = batch.x.detach().to(case.dtype).requires_grad_()
                expected_output = (
                    reference(batch.x)
                    if case.mode == "fsdp"
                    else reference(batch.x, batch.mask, batch.weights, batch.indices)
                )
            else:
                expected_output = _reference(reference, batch, forward_dtype=case.dtype)
            expected_loss = (expected_output.float() - batch.target).square().sum() / max(sum(case.counts), 1)
            expected_loss.backward()
            controls.append((label, reference, reference_optimizer, batch, expected_output, expected_loss))

    output, loss, local_batch = _backward(model, case, step)
    # Structural failures must also reach agreement BEFORE any grad.full_tensor().
    with _synchronized_checks(f"{case.name}/step-{step}/gradient-layout"):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert isinstance(parameter.grad, DTensor) and parameter.grad.placements == (Shard(0),), name
            else:
                assert parameter.grad is None, f"frozen {name}"
    dist.all_reduce(loss, op=dist.ReduceOp.SUM)
    gradients = {name: _full(p.grad) for name, p in model.named_parameters() if p.requires_grad}
    local_squared_norm = torch.zeros((), device=device, dtype=torch.float64)
    for parameter in model.parameters():
        if parameter.requires_grad:
            local_squared_norm += parameter.grad.to_local().double().square().sum()
    dist.all_reduce(local_squared_norm, op=dist.ReduceOp.SUM)
    shard_norm = local_squared_norm.sqrt().cpu()
    full_norm = sum(g.double().square().sum() for g in gradients.values()).sqrt()
    start = sum(case.counts[: dist.get_rank()])
    local_slice = slice(start, start + output.shape[0])
    results = {}
    with _synchronized_checks(f"{case.name}/step-{step}/forward-backward"):
        torch.testing.assert_close(shard_norm, full_norm, rtol=1e-12, atol=1e-12)
        for label, reference, _, batch, expected_output, expected_loss in controls:
            result = results[label] = {}
            result["output_relative_l2"] = _check(output, expected_output[local_slice], tolerance, f"{label}/output")
            result["loss_relative_l2"] = _check(loss, expected_loss, tolerance, f"{label}/global loss")
            assert local_batch.x.grad is not None and batch.x.grad is not None, label
            # FSDP loss was multiplied by two; input grads are not reduce-scattered.
            result["input_gradient_relative_l2"] = _check(
                local_batch.x.grad.float() / (2 if case.mode == "fsdp" else 1),
                batch.x.grad[local_slice],
                tolerance,
                f"{label}/input gradient",
            )
            if case.mode == "ep":
                assert local_batch.weights.grad is not None and batch.weights.grad is not None, label
                result["router_gradient_relative_l2"] = _check(
                    local_batch.weights.grad, batch.weights.grad[local_slice], tolerance, f"{label}/router gradient"
                )
            ref_parameters = dict(reference.named_parameters())
            result["gradient_relative_l2"] = {
                name: _check(gradient, ref_parameters[name].grad, tolerance, f"{label}/{name} gradient")
                for name, gradient in gradients.items()
            }
            reference_norm = sum(ref_parameters[name].grad.double().square().sum() for name in gradients).sqrt()
            _check(full_norm, reference_norm, tolerance, f"{label}/global gradient norm")
            result["reference_grad_norm"] = float(reference_norm)
    with _synchronized_checks(f"{case.name}/step-{step}/optimizer"):
        optimizer.step()  # SGD on Shard(0) parameters is local; no collectives.
        for _, _, reference_optimizer, _, _, _ in controls:
            reference_optimizer.step()
    after = _snapshot(model, optimizer)
    with _synchronized_checks(f"{case.name}/step-{step}/updates"):
        for label, reference, reference_optimizer, _, _, _ in controls:
            parameter_errors, update_errors, momentum_errors = {}, {}, {}
            for name, parameter in reference.named_parameters():
                actual, expected = after[f"parameter/{name}"], parameter.detach().cpu().float()
                if not parameter.requires_grad:
                    torch.testing.assert_close(actual, before[name], rtol=0, atol=0, msg=f"frozen {name}")
                    torch.testing.assert_close(expected, before[name].float(), rtol=0, atol=0)
                    continue
                parameter_errors[name] = _check(actual, expected, tolerance, f"{label}/{name} updated parameter")
                delta, ref_delta = actual.float() - before[name].float(), expected - before[name].float()
                update_tolerance = _Tolerance(
                    tolerance.rtol, tolerance.atol, tolerance.update_relative_l2, 0, tolerance.rms_atol
                )
                update_errors[name] = _check(delta, ref_delta, update_tolerance, f"{label}/{name} update delta")
                momentum_errors[name] = _check(
                    after[f"momentum/{name}"],
                    reference_optimizer.state[parameter]["momentum_buffer"],
                    tolerance,
                    f"{label}/{name} momentum",
                )
                if torch.count_nonzero(ref_delta):
                    assert torch.count_nonzero(delta), f"{label}/{name}: optimizer made no update"
            results[label].update(
                parameter_relative_l2=parameter_errors,
                update_relative_l2=update_errors,
                momentum_relative_l2=momentum_errors,
            )
    return {
        "step": step,
        "loss": float(loss),
        "controls": results,
        "full_grad_norm": float(full_norm),
        "shard_grad_norm": float(shard_norm),
    }


def _snapshot(model: nn.Module, optimizer: torch.optim.SGD, *, gradients: bool = False) -> dict[str, torch.Tensor]:
    """Snapshot complete distributed training state on every rank.

    Returns:
        CPU copies keyed by quantity/FQN: parameters, momentum and optional
        gradients all have their original full logical parameter shape; optimizer
        hyperparameters are a [3] FP64 vector (lr, momentum, weight_decay).
    """
    result = {}
    with _synchronized_checks("snapshot/layout"):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert "momentum_buffer" in optimizer.state[parameter], name
                if gradients:
                    assert parameter.grad is not None, name
    for name, parameter in model.named_parameters():
        result[f"parameter/{name}"] = _full(parameter)
        if parameter.requires_grad:
            result[f"momentum/{name}"] = _full(optimizer.state[parameter]["momentum_buffer"])
            if gradients:
                result[f"gradient/{name}"] = _full(parameter.grad)
    group = optimizer.param_groups[0]
    result["hyperparameters"] = torch.tensor(
        [group["lr"], group["momentum"], group["weight_decay"]], dtype=torch.float64
    )
    return result


def _assert_snapshot(model: nn.Module, optimizer: torch.optim.SGD, expected: dict[str, torch.Tensor]) -> None:
    """Require exact restore/replay, not BF16-versus-FP32 tolerances.

    Args:
        model: Live sharded module whose state is gathered on all ranks.
        optimizer: Live SGD state attached to model parameter identities.
        expected: Complete CPU tensors in the layout documented by _snapshot.
    """
    actual = _snapshot(model, optimizer, gradients=any(key.startswith("gradient/") for key in expected))
    with _synchronized_checks("snapshot/restore-replay"):
        assert actual.keys() == expected.keys()
        for name in expected:
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0, msg=f"resume {name}")


def _run_case(case: _Case, mesh: DeviceMesh, output: Path) -> dict[str, object]:
    model = _model(case)
    optimizer = _shard(model, case, mesh)
    steps = [_step(model, optimizer, case, 0)]
    if case.mode == "fsdp":
        saved = _snapshot(model, optimizer)
        checkpoint = [tempfile.mkdtemp(prefix="dcp-", dir=output) if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(checkpoint, src=0)
        try:
            model_state, optim_state = get_state_dict(model, optimizer)
            dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=checkpoint[0])
            steps.append(_step(model, optimizer, case, 1))
            uninterrupted = _snapshot(model, optimizer, gradients=True)
            resumed = _model(case, seed=321)
            resumed_optimizer = _shard(resumed, case, mesh)
            # Prove load restores both nontrivial tensors and optimizer settings.
            resumed_optimizer.param_groups[0]["lr"] = 0.123
            model_state, optim_state = get_state_dict(resumed, resumed_optimizer)
            dcp.load({"model": model_state, "optimizer": optim_state}, checkpoint_id=checkpoint[0])
            incompatible = set_state_dict(
                resumed, resumed_optimizer, model_state_dict=model_state, optim_state_dict=optim_state
            )
            with _synchronized_checks("checkpoint/keys"):
                assert not incompatible.missing_keys and not incompatible.unexpected_keys
            _assert_snapshot(resumed, resumed_optimizer, saved)
            resumed_optimizer.zero_grad(set_to_none=True)
            _backward(resumed, case, 1)
            resumed_optimizer.step()
            _assert_snapshot(resumed, resumed_optimizer, uninterrupted)
            dist.barrier()
        finally:
            if dist.get_rank() == 0:
                shutil.rmtree(checkpoint[0])
    else:
        steps.append(_step(model, optimizer, case, 1))
    return {
        "name": case.name,
        "status": "passed",
        "counts": case.counts,
        "quantizer": asdict(_QUANTIZATIONS[case.quantization].build().config),
        "tolerances": asdict(_TOLERANCES[case.dtype]),
        "reference_conditioning": "distributed-pre-step-parameters-and-momentum",
        "steps": steps,
        "dcp_model_optimizer_resume": case.mode == "fsdp",
    }


def main() -> None:
    """Run requested low-level probes; missing hardware and failures exit nonzero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fsdp", "ep", "all"), default="all")
    parser.add_argument(
        "--output", type=Path, required=True, help="Directory for per-rank JSON and temporary DCP state"
    )
    args = parser.parse_args()
    rank = int(os.environ.get("RANK", "0"))
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / f"rank-{rank}.json"
    report = {
        "rank": rank,
        "world_size": int(os.environ.get("WORLD_SIZE", "1")),
        "scope": "low-level-only",
        "tests": [],
    }
    started = time.monotonic()
    current = "preflight"
    try:
        if report["world_size"] != 2 or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            raise RuntimeError("Requires torchrun --nproc-per-node=2 and two visible CUDA GPUs; CPU collection only")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        if not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("Both ranks require native BF16 GPU support")
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        dist.init_process_group("nccl", timeout=timedelta(seconds=60))
        mesh = init_device_mesh("cuda", (2,))
        torch.cuda.reset_peak_memory_stats()
        modes = ("fsdp", "ep") if args.mode == "all" else (args.mode,)
        with torch.compiler.set_stance("force_eager"):
            for mode in modes:
                for dtype in (torch.float32, torch.bfloat16):
                    for quantization in _QUANTIZATIONS:
                        for route in _EP_ROUTES if mode == "ep" else ("uneven",):
                            case = _Case(mode, dtype, quantization, route)
                            current = case.name
                            report["tests"].append(_run_case(case, mesh, args.output))
                            report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        dist.barrier()
        report["status"] = "passed"
        report["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
    except Exception:
        report["status"] = "failed"
        report["tests"].append({"name": current, "status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - started
        report["torch_version"] = torch.__version__
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
