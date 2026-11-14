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

"""Two-GPU, low-level QAT gates; not high-level distributed recipe support.

CPU regressions cover precision conditioning and synchronized Gloo failures.
The GPU gate deliberately FAILS, never skips, if two CUDA/BF16 GPUs or required
dependencies are unavailable. Each GPU launch has a
180-second wall timeout, including startup, and kills its whole process group.
Do not launch pytest itself under torchrun. The sibling worker also supports
direct torchrun execution with --mode all --output DIRECTORY.
"""

import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.functional_tests.quantization import qat_distributed_worker as worker

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = Path(__file__).with_name("qat_distributed_worker.py")


@pytest.mark.gpu
@pytest.mark.timeout(200)
@pytest.mark.parametrize("mode,expected_cases", [("fsdp", 6), ("ep", 36)])
def test_lora_qat_distributed(tmp_path: Path, mode: str, expected_cases: int) -> None:
    """Run real collectives and require successful numerical reports from both ranks."""
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "Launch pytest normally, not under torchrun"
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0,1")
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", TORCH_NCCL_ASYNC_ERROR_HANDLING="1")
    env["PYTHONPATH"] = str(_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--max-restarts=0",
        str(_WORKER),
        "--mode",
        mode,
        "--output",
        str(tmp_path),
    ]
    log_path = tmp_path / "torchrun.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=180)
        except subprocess.TimeoutExpired:
            pytest.fail(f"{mode}: torchrun exceeded 180 seconds; log:\n{log_path.read_text()}")
        finally:
            # Also kill grandchildren if pytest is interrupted or the launcher exits early.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    assert returncode == 0, f"{mode}: torchrun exit {returncode}\n{log_path.read_text()}"
    reports = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    for rank, report in enumerate(reports):
        assert report["rank"] == rank and report["world_size"] == 2
        assert report["status"] == "passed", report
        assert report["scope"] == "low-level-only"
        assert len(report["tests"]) == expected_cases
        assert all(result["status"] == "passed" for result in report["tests"])
        for result in report["tests"]:
            assert result["reference_conditioning"] == "distributed-pre-step-parameters-and-momentum"
            assert [step["step"] for step in result["steps"]] == [0, 1]
            for step in result["steps"]:
                assert set(step["controls"]) == {"same_dtype_qat", "conditioned_fp32_ste"}
                for control in step["controls"].values():
                    assert control["gradient_relative_l2"]
                    assert control["update_relative_l2"].keys() == control["gradient_relative_l2"].keys()
                    assert control["momentum_relative_l2"].keys() == control["gradient_relative_l2"].keys()
    assert [test["name"] for test in reports[0]["tests"]] == [test["name"] for test in reports[1]["tests"]]


@pytest.mark.timeout(30)
def test_bf16_mxfp4_discontinuity_is_not_parallel_error() -> None:
    """Reproduce job 7591773's exact dense step-1 failure without FSDP or EP."""
    case = worker._Case("fsdp", torch.bfloat16, "mxfp4-e8m0")
    actual = worker._model(case)
    raw = copy.deepcopy(actual).float()
    optimizers = [
        torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=0.25, momentum=0.9, foreach=False)
        for model in (actual, raw)
    ]
    for step in range(2):
        batch = worker._batch(case, step, None, torch.device("cpu"))
        output = actual(batch.x.bfloat16())
        raw_output = worker._reference(raw, batch)
        conditioned = worker._reference(copy.deepcopy(actual).float(), batch, forward_dtype=case.dtype)
        worker._check(output[:3], conditioned[:3], worker._TOLERANCES[case.dtype], "conditioned")
        if step == 0:
            for value, optimizer in zip((output, raw_output), optimizers):
                ((value.float() - batch.target).square().sum() / 8).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
    # Do not hide or relax the old failure: it must still fail the ORIGINAL bounds.
    with pytest.raises(AssertionError, match="raw FP32"):
        worker._check(output[:3], raw_output[:3], worker._TOLERANCES[case.dtype], "raw FP32")
    assert float((output[:3].float() - raw_output[:3]).detach().abs().max()) == pytest.approx(0.02814483642578125)
    quantizer = actual.weight_fake_quantizer
    merged = actual.weight + actual.scale * (actual.lora_B.weight @ actual.lora_A.weight)
    raw_merged = raw.weight + raw.scale * (raw.lora_B.weight @ raw.lora_A.weight)
    encoded, raw_encoded = quantizer.quantize(merged), quantizer.quantize(raw_merged)
    assert torch.equal(encoded.scales, raw_encoded.scales)
    assert int((encoded.payload != raw_encoded.payload).sum()) == 3
    decoded = encoded.dequantize(dtype=torch.float32)
    raw_decoded = raw_encoded.dequantize(dtype=torch.float32)
    # Exact midpoint ties round to even; widening BF16 storage cannot restore
    # the pre-rounding FP32 values on the opposite sides of these boundaries.
    assert merged[13, 17] == -0.078125 and raw_merged[13, 17] < -0.078125
    assert merged[13, 28] == 0.15625 and raw_merged[13, 28] > 0.15625
    assert decoded[13, 17] == -0.0625 and raw_decoded[13, 17] == -0.09375
    assert decoded[13, 28] == 0.125 and raw_decoded[13, 28] == 0.1875


@pytest.mark.timeout(30)
@pytest.mark.parametrize("mode", ("fsdp", "ep"))
@pytest.mark.parametrize("quantization", worker._QUANTIZATIONS)
def test_conditioned_merge_qdq_matches_actual_forward_and_fp32_ste(mode: str, quantization: str) -> None:
    """Verify actual quantizer inputs, decode rounding, and independent merge VJP."""
    case = worker._Case(mode, torch.bfloat16, quantization)
    actual = worker._model(case)
    actual.scale = 2.5  # Exercise multiplication rounding, not only scale=1.
    reference = copy.deepcopy(actual).float()
    observed = []

    def capture(module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        """Record args[0], the canonical weight tensor [..., out, in], without mutation."""
        observed.append(args[0].detach().clone())

    hook = actual.weight_fake_quantizer.register_forward_pre_hook(capture)
    try:
        batch = worker._batch(case, 0, None, torch.device("cpu"))
        with torch.compiler.set_stance("force_eager"):
            if mode == "fsdp":
                actual(batch.x.bfloat16())
                operands = [(reference.weight, reference.lora_B.weight, reference.lora_A.weight)]
            else:
                actual(batch.x.bfloat16(), batch.mask, batch.weights, batch.indices)
                operands = [
                    (reference.gate_and_up_projs, reference.lora_gate_and_up_A, reference.lora_gate_and_up_B),
                    (reference.down_projs, reference.lora_down_A, reference.lora_down_B),
                ]
    finally:
        hook.remove()
    assert len(observed) == len(operands)
    for captured, (base, left, right) in zip(observed, operands):
        merged = worker._merge(base, left, right, reference.scale, case.dtype)
        canonical = merged.transpose(-1, -2) if mode == "ep" else merged
        torch.testing.assert_close(canonical.detach(), captured.float(), rtol=0, atol=0)
        encoded = actual.weight_fake_quantizer.quantize(captured)
        independent_encoded = actual.weight_fake_quantizer.quantize(canonical.detach())
        assert torch.equal(encoded.payload, independent_encoded.payload)
        assert torch.equal(encoded.scales, independent_encoded.scales)
        qdq = worker._qdq(canonical, reference.weight_fake_quantizer, case.dtype)
        torch.testing.assert_close(qdq.detach(), encoded.dequantize(dtype=case.dtype).float(), rtol=0, atol=0)
        upstream = torch.linspace(-1, 1, qdq.numel()).reshape(qdq.shape)
        left_grad, right_grad = torch.autograd.grad(qdq, (left, right), upstream)
        g = upstream.transpose(-1, -2) if mode == "ep" else upstream
        torch.testing.assert_close(left_grad, (g * reference.scale) @ right.detach().transpose(-1, -2))
        torch.testing.assert_close(right_grad, left.detach().transpose(-1, -2) @ (g * reference.scale))


_CPU_CASES = [
    worker._Case(mode, dtype, quantization, route)
    for mode in ("fsdp", "ep")
    for dtype in (torch.float32, torch.bfloat16)
    for quantization in worker._QUANTIZATIONS
    for route in (worker._EP_ROUTES if mode == "ep" else ("uneven",))
]


@pytest.mark.timeout(30)
@pytest.mark.parametrize("case", _CPU_CASES, ids=lambda case: case.name)
def test_conditioned_oracle_cpu(case: worker._Case) -> None:
    """Exercise two local steps of every oracle case, without claiming GPU parity."""
    actual = worker._model(case)
    optimizer = torch.optim.SGD(
        (p for p in actual.parameters() if p.requires_grad), lr=case.learning_rate, momentum=0.9, foreach=False
    )
    tolerance = worker._TOLERANCES[case.dtype]
    with torch.compiler.set_stance("force_eager"):
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference = copy.deepcopy(actual).float()
            reference.zero_grad(set_to_none=True)
            reference_optimizer = torch.optim.SGD(p for p in reference.parameters() if p.requires_grad)
            reference_optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            batch = worker._batch(case, step, None, torch.device("cpu"))
            local = worker._batch(case, step, None, torch.device("cpu"))
            local.x = local.x.detach().to(case.dtype).requires_grad_()
            output = (
                actual(local.x) if case.mode == "fsdp" else actual(local.x, local.mask, local.weights, local.indices)
            )
            expected = worker._reference(reference, batch, forward_dtype=case.dtype)
            worker._check(output, expected, tolerance, "output")
            for value, data in ((output, local), (expected, batch)):
                ((value.float() - data.target).square().sum() / max(sum(case.counts), 1)).backward()
            worker._check(local.x.grad, batch.x.grad, tolerance, "input gradient")
            if case.mode == "ep":
                worker._check(local.weights.grad, batch.weights.grad, tolerance, "router gradient")
            before = {name: p.detach().float().clone() for name, p in actual.named_parameters()}
            for name, parameter in reference.named_parameters():
                other = actual.get_parameter(name)
                if parameter.requires_grad:
                    worker._check(other.grad, parameter.grad, tolerance, f"{name} gradient")
                else:
                    assert other.grad is None and parameter.grad is None
            optimizer.step()
            reference_optimizer.step()
            for name, parameter in reference.named_parameters():
                other = actual.get_parameter(name)
                worker._check(other, parameter, tolerance, f"{name} parameter")
                update_tolerance = worker._Tolerance(
                    tolerance.rtol, tolerance.atol, tolerance.update_relative_l2, 0, tolerance.rms_atol
                )
                worker._check(
                    other.float() - before[name], parameter - before[name], update_tolerance, f"{name} update"
                )


@pytest.mark.timeout(90)
def test_synchronized_checks_gloo(tmp_path: Path) -> None:
    """Run real CPU EP and prove a rank-only mismatch cannot strand its peer."""
    code = """
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from tests.functional_tests.quantization import qat_distributed_worker as worker
torch.set_num_threads(2)
dist.init_process_group('gloo', timeout=timedelta(seconds=15))
try:
    for failing_rank, label in ((0, 'output'), (1, 'gradient'), (0, 'resume')):
        try:
            with worker._synchronized_checks(label):
                # Deliberately different local check counts; no per-check collectives.
                for _ in range(dist.get_rank() + 1):
                    worker._check(torch.zeros(3), torch.zeros(3), worker._TOLERANCES[torch.float32], 'zero')
                if dist.get_rank() == failing_rank:
                    worker._check(torch.ones(3), torch.zeros(3), worker._TOLERANCES[torch.float32], label)
        except AssertionError as error:
            assert f'rank {failing_rank}:' in str(error) and label in str(error)
        else:
            raise AssertionError('rank did not receive the mismatch')
        # This collective must complete on BOTH ranks after each caught mismatch.
        reached = torch.ones(())
        dist.all_reduce(reached)
        assert reached.item() == 2
    with worker._synchronized_checks('success'):
        pass
    # Exercise the worker's actual gather/check boundaries, not just its helper.
    # CPU EP is valid; this does not replace the CUDA/NCCL or FSDP gates.
    mesh = init_device_mesh('cpu', (2,))
    with torch.compiler.set_stance('force_eager'):
        original_check = worker._check
        for failing_rank, quantity in ((0, 'same_dtype_qat/output'), (1, 'same_dtype_qat/lora_down_B gradient')):
            def inject(actual, expected, tolerance, label):
                '''Compare full tensors of matching arbitrary shape, injecting one rank-local error.'''
                if dist.get_rank() == failing_rank and label == quantity:
                    actual = actual + 1
                return original_check(actual, expected, tolerance, label)

            try:
                with patch.object(worker, '_check', inject):
                    worker._run_case(worker._Case('ep', torch.bfloat16, 'mxfp4-e8m0'), mesh, Path('.'))
            except AssertionError as error:
                assert 'synchronized check failure' in str(error)
                assert quantity in str(error) and f'rank {failing_rank}:' in str(error)
            else:
                raise AssertionError('worker failed to propagate its numerical mismatch')
            dist.barrier()
        for dtype in (torch.float32, torch.bfloat16):
            for quantization in worker._QUANTIZATIONS:
                for route in worker._EP_ROUTES:
                    result = worker._run_case(worker._Case('ep', dtype, quantization, route), mesh, Path('.'))
                    assert result['status'] == 'passed' and len(result['steps']) == 2
finally:
    dist.destroy_process_group()
"""
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    log_path = tmp_path / "gloo.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "--max-restarts=0",
                "--no-python",
                sys.executable,
                "-c",
                code,
            ],
            cwd=_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=75)
        except subprocess.TimeoutExpired:
            pytest.fail(f"Gloo regression exceeded 75 seconds; log:\n{log_path.read_text()}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    assert returncode == 0, log_path.read_text()
