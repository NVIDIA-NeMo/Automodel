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

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

from nemo_automodel.components.models.common.kf_triton_rms_norm import (
    KF_RMS_NORM_HIDDEN,
    KFTritonRMSNorm,
)
from nemo_automodel.components.models.common.utils import Float32RMSNorm, initialize_rms_norm_module

# The campaign tolerance from examples/scalable_ai/kernel_factory/moonlight-rmsnorm/workload.jsonl:
# both paths do the same fp32 arithmetic, so they can only disagree by one bf16 ulp.
FORWARD_RTOL, FORWARD_ATOL = 8e-3, 5e-3

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the kf Triton kernel requires a GPU")


def _reference(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Run the shipped torch_fp32 forward.

    Args:
        x: Tensor of shape [..., hidden], with arbitrary leading dimensions.
        weight: Tensor of shape [hidden].
        eps: Epsilon added to the mean square.

    Returns:
        Tensor of shape [..., hidden] in x's dtype.
    """
    return torch.nn.functional.rms_norm(x.float(), (x.shape[-1],), weight.float(), eps).to(x.dtype)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dim": 4096, "eps": 1e-6}, "hidden=2048"),
        ({"dim": KF_RMS_NORM_HIDDEN, "eps": 1e-5}, "got eps=1e-05"),
        ({"dim": KF_RMS_NORM_HIDDEN, "eps": 1e-6, "dtype": torch.float32}, "bf16 kernel"),
    ],
)
def test_rejects_contracts_the_campaign_did_not_search(kwargs, message):
    with pytest.raises(ValueError, match=message):
        KFTritonRMSNorm(**kwargs)


def test_backend_string_builds_the_adapter_and_keeps_the_checkpoint_key():
    module = initialize_rms_norm_module("kf_triton_h2048", KF_RMS_NORM_HIDDEN, eps=1e-6)
    assert isinstance(module, KFTritonRMSNorm)
    # The parameter name and shape are what the checkpoint keys depend on.
    assert list(dict(module.named_parameters())) == ["weight"]
    assert module.weight.shape == (KF_RMS_NORM_HIDDEN,)
    assert module.weight.dtype == torch.bfloat16
    reference = Float32RMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6)
    assert dict(module.state_dict()).keys() == dict(reference.state_dict()).keys()


def test_backend_string_rejects_a_non_moonlight_hidden_size():
    # The DeepSeek-V4 q/kv, compressor and indexer norms normalize other sizes; the
    # backend must refuse them rather than silently fall back.
    with pytest.raises(ValueError, match="hidden=2048"):
        initialize_rms_norm_module("kf_triton_h2048", 128, eps=1e-6)


@pytest.mark.parametrize(
    ("x_shape", "x_dtype", "weight_shape", "weight_dtype", "message"),
    [
        ((4, 1024), torch.bfloat16, (1024,), torch.bfloat16, "hidden=2048"),
        ((4, KF_RMS_NORM_HIDDEN), torch.float32, (KF_RMS_NORM_HIDDEN,), torch.bfloat16, "bf16 kernel"),
        ((4, KF_RMS_NORM_HIDDEN), torch.bfloat16, (1, KF_RMS_NORM_HIDDEN), torch.bfloat16, r"shape \[2048\]"),
    ],
)
def test_op_rejects_unsupported_inputs_before_launching(x_shape, x_dtype, weight_shape, weight_dtype, message):
    # Validation runs ahead of the kernel launch, so this is checkable without a GPU.
    x = torch.zeros(x_shape, dtype=x_dtype)
    weight = torch.ones(weight_shape, dtype=weight_dtype)
    with pytest.raises(ValueError, match=message):
        torch.ops.nemo_automodel.kf_triton_rms_norm(x, weight, 1e-6)


@requires_cuda
@pytest.mark.parametrize("shape", [(1024, KF_RMS_NORM_HIDDEN), (4, 2048, KF_RMS_NORM_HIDDEN), (0, KF_RMS_NORM_HIDDEN)])
def test_forward_matches_the_torch_fp32_baseline(shape):
    torch.manual_seed(17)
    x = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    module = KFTritonRMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6, device="cuda")
    with torch.no_grad():
        module.weight.normal_()
    actual = module(x)
    expected = _reference(x, module.weight, module.eps)
    assert actual.shape == x.shape
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)


@requires_cuda
def test_forward_handles_a_non_contiguous_input():
    torch.manual_seed(17)
    source = torch.randn(8, 2, KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16)
    x = source[:, 0]  # rows are contiguous, the row stride is 2 * hidden
    module = KFTritonRMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6, device="cuda")
    with torch.no_grad():
        module.weight.normal_()
    torch.testing.assert_close(
        module(x), _reference(x, module.weight, module.eps), rtol=FORWARD_RTOL, atol=FORWARD_ATOL
    )


@requires_cuda
@pytest.mark.parametrize("trainable", ["both", "input", "weight"])
def test_gradients_are_the_torch_fp32_reference_gradients(trainable):
    # The campaign is forward only, so the backward must be bit-identical to the
    # baseline's: both call the same reference gradient on the same saved tensors.
    torch.manual_seed(17)
    values = torch.randn(512, KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16)
    upstream = torch.randn_like(values)

    candidate = KFTritonRMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6, device="cuda")
    baseline = Float32RMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6, device="cuda")
    for module in (candidate, baseline):
        with torch.no_grad():
            module.weight.copy_(weights)
        module.weight.requires_grad_(trainable != "input")

    grads = []
    for module in (candidate, baseline):
        x = values.detach().clone().requires_grad_(trainable != "weight")
        with torch.compiler.set_stance("force_eager"):
            module(x).backward(upstream)
        grads.append((x.grad, module.weight.grad))

    (candidate_x, candidate_w), (baseline_x, baseline_w) = grads
    for candidate_grad, baseline_grad in ((candidate_x, baseline_x), (candidate_w, baseline_w)):
        if baseline_grad is None:
            assert candidate_grad is None
        else:
            torch.testing.assert_close(candidate_grad, baseline_grad, rtol=0, atol=0)


@requires_cuda
@pytest.mark.runtime_budget(30, hard_timeout=120, reason="opcheck runs the op through AOTAutograd compilation")
def test_custom_op_contract():
    x = torch.randn(8, KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    torch.library.opcheck(torch.ops.nemo_automodel.kf_triton_rms_norm.default, (x, weight, 1e-6))


@requires_cuda
@pytest.mark.runtime_budget(
    60, hard_timeout=180, reason="real Inductor compilation of the wrapper at two dynamic token counts"
)
def test_compiles_fullgraph_and_matches_eager(monkeypatch):
    monkeypatch.delenv("TORCH_COMPILE_DISABLE", raising=False)
    torch.manual_seed(17)
    module = KFTritonRMSNorm(KF_RMS_NORM_HIDDEN, eps=1e-6, device="cuda")
    with torch.no_grad():
        module.weight.normal_()
    compiled = torch.compile(module, fullgraph=True, dynamic=True)
    for tokens in (1024, 4096):
        x = torch.randn(tokens, KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        actual = compiled(x)
        with torch.no_grad():
            no_grad_output = compiled(x)
        # The forward is opaque, so grad and no_grad must run the identical kernel.
        torch.testing.assert_close(actual, no_grad_output, rtol=0, atol=0)
        torch.testing.assert_close(
            actual, _reference(x, module.weight, module.eps), rtol=FORWARD_RTOL, atol=FORWARD_ATOL
        )
        actual.backward(torch.randn_like(actual))
        assert torch.isfinite(x.grad).all()
        module.zero_grad()


def _dtensor_worker(rank: int, world_size: int, init_file: str) -> None:
    """Check the registered sharding rule on a token-sharded input.

    Args:
        rank: Global rank of this process.
        world_size: Number of processes in the mesh.
        init_file: Path used by the file:// rendezvous.
    """
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        torch.manual_seed(17)
        values = torch.randn(8 * world_size, KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16)
        weights = torch.randn(KF_RMS_NORM_HIDDEN, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn_like(values)
        expected = _reference(values, weights, 1e-6)
        for placement in (Replicate(), Shard(0)):
            x = distribute_tensor(values.clone(), mesh, [placement]).requires_grad_()
            weight = distribute_tensor(weights.clone(), mesh, [Replicate()]).requires_grad_()
            actual = torch.ops.nemo_automodel.kf_triton_rms_norm(x, weight, 1e-6)
            # The hidden axis must stay complete; the token axis may stay sharded.
            assert actual.placements == (placement,)
            torch.testing.assert_close(actual.full_tensor(), expected, rtol=FORWARD_RTOL, atol=FORWARD_ATOL)

            actual.backward(distribute_tensor(upstream, mesh, list(actual.placements)))
            assert torch.isfinite(x.grad.full_tensor()).all()
            # A token-sharded input leaves a Partial weight gradient; the training
            # gradient-sync boundary sums it before the optimizer sees it.
            assert torch.isfinite(weight.grad.redistribute(placements=[Replicate()]).to_local()).all()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="DTensor sharding needs 2 GPUs")
@pytest.mark.runtime_budget(
    60, hard_timeout=180, reason="two spawned workers initialize NCCL and run the kernel under DTensor"
)
def test_dtensor_sharding(tmp_path):
    mp.spawn(_dtensor_worker, args=(2, str(tmp_path / "init")), nprocs=2, join=True)
