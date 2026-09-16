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

"""Fail-closed real two-GPU composed gate; run CPU contracts with ``-m 'not gpu'``.

No missing-dependency/GPU skips. Invoke pytest normally, not inside torchrun.
The gate is limited to the tiny expert-only topology documented in the worker;
passing does NOT authorize lifting production high-level QAT guards.
"""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.functional_tests.quantization import qat_composed_worker as worker

_ROOT = Path(__file__).resolve().parents[3]
_WORKER = Path(__file__).with_name("qat_composed_worker.py")


def _validate_report(report: dict, architecture: str, dtype: str, rank: int) -> None:
    assert report["rank"] == rank and report["world_size"] == 2
    assert report["status"] == "passed", report
    assert report["scope"] == worker._SCOPE
    assert report["high_level_guard_lifted"] is False
    assert report["supported_topology"] == report["requested_topology"] == worker._TOPOLOGY
    assert len(report["tests"]) == 1
    case = report["tests"][0]
    assert (case["architecture"], case["dtype"]) == (architecture, dtype)
    assert case["status"] == "passed" and case["stage"] == "complete", case
    assert case["supported_topology"] == worker._TOPOLOGY
    quantization = worker._quantization(architecture)
    assert case["quantization"] == {
        "format": quantization.format,
        "block_size": list(quantization.block_size),
        "scale_format": quantization.scale_format,
    }
    assert case["reference"] == "whole-model-clone-global-batch-same-dtype"
    assert case["loss_denominator"] == 10 and case["expert_gradient_rescale"] == 1 and case["steps"] == 1
    assert case["frozen_parameters_unchanged"] is True and case["qat_calls"] == 2
    assert case["routing"] == {"tokens_per_rank": 6, "top_k": 2, "expert_load": [6, 6], "matched": True}
    block, expert = worker._paths(architecture)
    ownership = case["ownership"]
    adapters = {f"{expert}.{name}" for name in worker._ADAPTERS}
    assert set(ownership["trainable_parameters"]) == adapters
    assert set(ownership["ep_parameters"]) == adapters | {f"{expert}.gate_and_up_projs", f"{expert}.down_projs"}
    assert ownership["fsdp_parameters"] and not set(ownership["fsdp_parameters"]) & set(ownership["ep_parameters"])
    assert "" in ownership["fsdp_units"] and block in ownership["fsdp_units"]
    assert expert not in ownership["fsdp_units"]
    expected_labels = {"logits", "loss", "input", "output", "routing_weights", "expert_gradient_norm"}
    expected_labels |= {f"{quantity}/{name}" for quantity in ("gradient", "update", "parameter") for name in adapters}
    measurements = case["measurements"]
    assert len(measurements) == len(expected_labels)
    assert {item["label"] for item in measurements} == expected_labels
    for item in measurements:
        assert item["status"] == "passed", item
        assert 0 <= item["relative_l2"] <= item["relative_l2_limit"], item
        if item["label"].startswith(("gradient/", "update/")):
            assert item["actual_norm"] > 0 and item["expected_norm"] > 0, item


@pytest.mark.gpu
@pytest.mark.timeout(200)
@pytest.mark.parametrize("architecture", worker._ARCHITECTURES)
@pytest.mark.parametrize("dtype", worker._DTYPES)
def test_lora_qat_composed(tmp_path: Path, architecture: str, dtype: str) -> None:
    """Require actual composed-model forward/backward/update evidence from both ranks."""
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
        "--architecture",
        architecture,
        "--dtype",
        dtype,
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
            pytest.fail(f"Composed gate exceeded 180s; partial rank reports: {tmp_path}\n{log_path.read_text()}")
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait(timeout=10)
    assert returncode == 0, f"torchrun exit {returncode}; rank reports: {tmp_path}\n{log_path.read_text()}"
    for rank in range(2):
        _validate_report(json.loads((tmp_path / f"rank-{rank}.json").read_text()), architecture, dtype, rank)


@pytest.mark.parametrize("architecture", worker._ARCHITECTURES)
@pytest.mark.parametrize("dtype", worker._DTYPES)
def test_composed_cli_and_quantization_contract(architecture: str, dtype: str) -> None:
    """CPU-only parser/config check, not distributed or model parity evidence."""
    args = worker._parser().parse_args(["--architecture", architecture, "--dtype", dtype, "--output", "/tmp/qat-gate"])
    assert args.architecture == architecture and args.dtype == dtype
    assert args.output == Path("/tmp/qat-gate")
    quantization = worker._quantization(architecture)
    assert quantization.build().config == quantization
    if architecture == "glm5_next":
        assert (quantization.format, quantization.block_size, quantization.scale_format) == (
            "fp8",
            (128, 128),
            "float32",
        )
    else:
        assert (quantization.format, quantization.block_size, quantization.scale_format) == ("mxfp4", (1, 32), "e8m0")


def test_composed_cli_rejects_unsupported_case() -> None:
    with pytest.raises(SystemExit):
        worker._parser().parse_args(["--architecture", "glm_moe_dsa", "--output", "/tmp/qat-gate"])
    with pytest.raises(SystemExit):
        worker._parser().parse_args(["--dtype", "float16", "--output", "/tmp/qat-gate"])


@pytest.mark.parametrize("dtype", worker._DTYPES)
@pytest.mark.parametrize("factor", [0.0, 0.5, 2.0, float("nan"), float("inf")])
def test_composed_comparison_rejects_wrong_gradient_scale(dtype: str, factor: float) -> None:
    """Tiny values must not conceal missing/double normalization through absolute tolerance."""
    measurements = []
    expected = torch.tensor([1e-8, -3e-8])
    with pytest.raises(AssertionError):
        worker._check(expected * factor, expected, dtype, "gradient/probe", measurements)
    assert len(measurements) == 1 and measurements[0]["status"] == "failed"
    json.dumps(measurements, allow_nan=False)


@pytest.mark.parametrize("dtype", worker._DTYPES)
def test_composed_comparison_zero_and_shape_contract(dtype: str) -> None:
    measurements = []
    worker._check(torch.zeros(2), torch.zeros(2), dtype, "zero", measurements)
    worker._check(torch.ones(2), torch.ones(2), dtype, "gradient/exact", measurements)
    assert all(item["status"] == "passed" for item in measurements)
    with pytest.raises(AssertionError):
        worker._check(torch.full((2,), 1e-12), torch.zeros(2), dtype, "zero", measurements)
    with pytest.raises(AssertionError, match="shape"):
        worker._check(torch.zeros(2), torch.zeros(3), dtype, "shape", measurements)


def test_composed_loss_is_global_mean_without_extra_ep_average() -> None:
    """CPU algebra: summed local losses/gradients equal one global CE mean."""
    torch.manual_seed(7)
    tokens = worker._tokens(torch.device("cpu"))
    assert tokens.shape == (2, 6) and not torch.equal(tokens[0], tokens[1])
    logits = torch.randn(2, 6, 64, requires_grad=True)
    local = sum(worker._loss(logits[rank : rank + 1], tokens[rank : rank + 1]) for rank in range(2))
    expected = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 64), tokens[:, 1:].reshape(-1))
    torch.testing.assert_close(local, expected)
    actual_grad = torch.autograd.grad(local, logits, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, logits)[0]
    torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.timeout(60)
@pytest.mark.parametrize("architecture", worker._ARCHITECTURES)
@pytest.mark.parametrize("dtype", worker._DTYPES)
def test_composed_reference_cpu(architecture: str, dtype: str) -> None:
    """Validate real-model preparation/reference helpers only; no distributed claim."""
    device = torch.device("cpu")
    with torch.compiler.set_stance("force_eager"):
        model = worker._model(architecture, dtype, device)
        assert model.backend.enable_fsdp_optimizations
        assert type(model).__module__.endswith(f"{architecture}.model")
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        tokens = worker._tokens(device)
        logits, trace = worker._forward(model, tokens, worker._paths(architecture)[1])
        assert logits.shape == (2, 6, 64) and torch.isfinite(logits).all()
        assert trace["input"].dtype == worker._DTYPES[dtype]
        assert torch.equal(trace["indices"].sort(-1).values, torch.tensor([0, 1]).expand(12, 2))
        assert trace["mask"].all()
        assert trace["qdq_input_0"].shape[0] == trace["qdq_input_1"].shape[0] == 2
        worker._loss(logits, tokens).backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None and torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad), name
            else:
                assert p.grad is None, name
        optimizer = torch.optim.SGD((p for p in model.parameters() if p.requires_grad), lr=worker._LR, foreach=False)
        optimizer.step()
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
            assert torch.equal(p, before[name]) != p.requires_grad, name


def test_composed_report_rejects_empty_success() -> None:
    """A success flag without actual model measurements cannot authorize the gate."""
    report = {
        "rank": 0,
        "world_size": 2,
        "status": "passed",
        "scope": worker._SCOPE,
        "high_level_guard_lifted": False,
        "supported_topology": dict(worker._TOPOLOGY),
        "requested_topology": dict(worker._TOPOLOGY),
        "tests": [],
    }
    with pytest.raises(AssertionError):
        _validate_report(report, "deepseek_v4", "float32", 0)


@pytest.mark.parametrize("status", ["failed", "skipped", "running"])
def test_composed_report_rejects_incomplete_gate(status: str) -> None:
    with pytest.raises(AssertionError):
        _validate_report({"rank": 0, "world_size": 2, "status": status}, "deepseek_v4", "float32", 0)
