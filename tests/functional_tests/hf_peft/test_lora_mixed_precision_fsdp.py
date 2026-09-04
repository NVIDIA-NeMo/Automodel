# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank FSDP2 mixed-precision coverage for the memory-efficient LoRA backward paths.

Regression for issue #3652. Under the FSDP2 policy that NeMo-RL's
``sft-llama3.1-8b-1n8g-fsdp2tp1-lora`` recipe uses, an FSDP unit's ``output_dtype=float32``
hands the next module an FP32 activation while ``param_dtype=bfloat16`` keeps its weights in
BF16. LoRA's memory-efficient autograd functions run their backward *outside* autocast, so
nothing re-applies the forward compute dtype and the recomputed matmuls raised
"expected m1 and m2 to have the same dtype".

Two independent paths reach that backward, and this covers both:

* ``LoRATritonFunction`` -- the per-linear path taken by every LoRA-patched ``nn.Linear``
  (the traceback in the issue);
* ``LoRASwiGLUMLPFunction`` / ``LoRAReLU2MLPFunction`` -- the fused MLP path that
  ``apply_lora_to_linear_modules()`` installs automatically once a whole gate/up/down (or
  up/down) MLP is LoRA-patched, which is what production execution actually enters.

Each case asserts the mixed dtype was really observed, so the test cannot quietly degrade
into uniform-dtype coverage, and compares gradients against the same model running on plain
autograd (``use_memory_efficient_lora=False``) under the identical sharded setup.

``tests/unit_tests/_peft`` already covers the dtype contract on CPU, but it constructs the
mixed dtype by hand. What cannot be covered there -- and is what actually broke in the field --
is that a real ``MixedPrecisionPolicy`` *produces* this layout: FSDP2's ``output_dtype`` cast is
the thing that puts an FP32 activation in front of BF16 weights, and it only exists inside a
sharded, multi-rank graph. Hence two ranks and real ``fully_shard``.
"""

import argparse
import contextlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft import lora as lora_module
from nemo_automodel.components._peft import lora_mlp as lora_mlp_module
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.moe.layers import MLP

_RESULT_PREFIX = "LORA_MIXED_PRECISION_FSDP_RESULT "

HIDDEN, INTER, RANK, TOKENS = 64, 96, 8, 16
# alpha != dim on purpose: PeftConfig.scale is alpha/dim, and scale==1.0 would hide every
# dropped-scale bug in the backward.
ALPHA = 3 * RANK

# The policy from issue #3652: parameters compute in BF16, gradients reduce in FP32, and each
# FSDP unit casts its output back to FP32 -- which is what puts an FP32 activation in front of
# BF16 LoRA weights.
_MP_POLICY = MixedPrecisionPolicy(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    output_dtype=torch.float32,
)


class _AttentionBlock(nn.Module):
    """Projections fed by their own FSDP unit -- the per-linear ``LoRATritonFunction`` path."""

    def __init__(self) -> None:
        super().__init__()
        # Its own FSDP unit, so output_dtype=float32 makes q_proj's input FP32.
        self.pre_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.q_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``o_proj(tanh(q_proj(pre_proj(x))))``.

        Args:
            x: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.o_proj(torch.tanh(self.q_proj(self.pre_proj(x))))


class _MLPBlock(nn.Module):
    """A whole MLP fed by its own FSDP unit -- the fused ``LoRA*MLPFunction`` path."""

    def __init__(self, activation: str) -> None:
        super().__init__()
        self.pre_proj = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.mlp = MLP(dim=HIDDEN, inter_dim=INTER, backend="torch", dtype=torch.float32, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run ``mlp(pre_proj(x))``.

        Args:
            x: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.mlp(self.pre_proj(x))


_CASES = {
    # case -> (block factory, LoRA target modules, autograd function expected to run, peft.use_triton)
    "per_linear": (lambda: _AttentionBlock(), ["*q_proj", "*o_proj"], "LoRATritonFunction", False),
    "fused_swiglu": (
        lambda: _MLPBlock("swiglu"),
        ["*gate_proj", "*up_proj", "*down_proj"],
        "LoRASwiGLUMLPFunction",
        False,
    ),
    "fused_relu2": (lambda: _MLPBlock("relu2"), ["*up_proj", "*down_proj"], "LoRAReLU2MLPFunction", False),
    # No `use_triton: true` case here on purpose: patch_linear_module force-disables Triton whenever
    # TransformerEngine is importable, and the CI image builds it (docker/Dockerfile TE_COMMIT), so
    # such a case would silently duplicate "per_linear". The Triton fallback is covered instead by
    # test_lora_kernel.py::test_memory_efficient_lora_declines_triton_on_mixed_dtype, which calls
    # apply_memory_efficient_lora directly and therefore runs regardless of TE.
}


@contextlib.contextmanager
def _observe_lora_dtypes():
    """Record ``(activation dtype, weight dtype)`` per memory-efficient LoRA autograd function.

    Without this the cases could pass on a uniform-dtype graph -- i.e. never exercise the bug --
    and still look green.
    """
    observed: dict[str, set] = {}
    originals = {
        "LoRATritonFunction": lora_module.LoRATritonFunction.forward,
        "LoRASwiGLUMLPFunction": lora_mlp_module.LoRASwiGLUMLPFunction.forward,
        "LoRAReLU2MLPFunction": lora_mlp_module.LoRAReLU2MLPFunction.forward,
    }

    def record(name: str, x: torch.Tensor, weight: torch.Tensor) -> None:
        """Record the dtype pair one autograd function saw.

        Args:
            name: Autograd function name.
            x: Activation tensor of shape [tokens, hidden] or [batch, sequence, hidden]; only its
                dtype is read.
            weight: The first weight the function multiplies ``x`` by; only its dtype is read.
        """
        observed.setdefault(name, set()).add((x.dtype, weight.dtype))

    def per_linear(x, lora_A, lora_B, scale, dtype, use_triton_kernel=True, res=None):
        record("LoRATritonFunction", x, lora_A)
        return originals["LoRATritonFunction"](x, lora_A, lora_B, scale, dtype, use_triton_kernel, res)

    def swiglu(ctx, x, gW, *rest):
        record("LoRASwiGLUMLPFunction", x, gW)
        return originals["LoRASwiGLUMLPFunction"](ctx, x, gW, *rest)

    def relu2(ctx, x, uW, *rest):
        record("LoRAReLU2MLPFunction", x, uW)
        return originals["LoRAReLU2MLPFunction"](ctx, x, uW, *rest)

    lora_module.LoRATritonFunction.forward = staticmethod(per_linear)
    lora_mlp_module.LoRASwiGLUMLPFunction.forward = staticmethod(swiglu)
    lora_mlp_module.LoRAReLU2MLPFunction.forward = staticmethod(relu2)
    try:
        yield observed
    finally:
        for name, fn in originals.items():
            owner = lora_module if name == "LoRATritonFunction" else lora_mlp_module
            getattr(owner, name).forward = staticmethod(fn)


def _build_sharded_block(case: str, memory_efficient: bool, mesh, device, reference_state=None):
    """LoRA-patch a block through the production entry point, then shard it with the issue's policy."""
    block_fn, target_modules, _, use_triton = _CASES[case]
    torch.manual_seed(20260825)
    block = block_fn()
    apply_lora_to_linear_modules(
        block,
        PeftConfig(
            target_modules=target_modules,
            dim=RANK,
            alpha=ALPHA,
            use_triton=use_triton,
            use_memory_efficient_lora=memory_efficient,
        ),
    )
    for module in block.modules():
        # lora_B initializes to zeros, which would leave the LoRA branch's gradients trivial.
        if hasattr(module, "lora_B"):
            nn.init.normal_(module.lora_B.weight, std=0.02)
    if reference_state is not None:
        block.load_state_dict(reference_state)
    state = {k: v.detach().clone() for k, v in block.state_dict().items()}

    block.to(device)
    # pre_proj as its own FSDP unit: output_dtype=float32 makes the LoRA modules' input FP32
    # while param_dtype=bfloat16 keeps their weights BF16.
    fully_shard(block.pre_proj, mesh=mesh, mp_policy=_MP_POLICY)
    fully_shard(block, mesh=mesh, mp_policy=_MP_POLICY)
    return block, state


def _lora_grads(block: nn.Module) -> dict[str, torch.Tensor]:
    """Collect every LoRA parameter gradient as a full (unsharded) fp32 tensor.

    Args:
        block: A sharded block whose LoRA gradients are ``DTensor``s sharded across the mesh.

    Returns:
        Parameter name to the gradient's global (all-rank) shape in fp32 -- ``[rank, in_features]``
        for ``lora_A`` and ``[out_features, rank]`` for ``lora_B``.
    """
    grads = {}
    for name, param in block.named_parameters():
        if "lora_" not in name:
            continue
        assert param.grad is not None, f"missing gradient for {name}"
        grad = param.grad
        grads[name] = (grad.full_tensor() if isinstance(grad, DTensor) else grad).detach().float()
    return grads


# Fused vs per-linear differ only by op ordering, so the floor here is bf16 GEMM rounding: measured
# at <=8.8e-3 (worst over cases and seeds, sm_89). CI also runs A100/H100/GB200, whose bf16 split-k
# differs, so allow ~3.4x headroom rather than risk a permanent red. Detection is unaffected -- the
# bugs this guards against are far larger: a dropped LoRA scale, a swapped base weight, or a missing
# gate term all land at >=1.3e-1, and a uniform 5% gradient error at 5.0e-2.
_TOLERANCE = 3e-2


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Max elementwise deviation normalized by ``expected``'s magnitude.

    Args:
        actual: Tensor of any shape.
        expected: Tensor broadcastable to ``actual``'s shape.

    Returns:
        The scalar relative error.
    """
    return (actual - expected).abs().max().item() / (expected.abs().max().item() + 1e-9)


def _assert_lora_mixed_precision(case: str, device: torch.device) -> None:
    """Run one LoRA path under the issue's FSDP2 policy and check it against plain autograd."""
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("dp",))
    _, _, expected_fn, _ = _CASES[case]

    generator = torch.Generator(device=device).manual_seed(4242 + dist.get_rank())
    inputs = torch.randn(2, TOKENS, HIDDEN, device=device, dtype=torch.float32, generator=generator)
    grad_out = torch.randn(2, TOKENS, HIDDEN, device=device, dtype=torch.float32, generator=generator)

    with _observe_lora_dtypes() as observed:
        efficient, state = _build_sharded_block(case, memory_efficient=True, mesh=mesh, device=device)
        x_efficient = inputs.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_efficient = efficient(x_efficient)
        (out_efficient.float() * grad_out).sum().backward()

    # The custom autograd path must actually have run, on the mixed dtype the issue describes.
    assert expected_fn in observed, f"{case}: {expected_fn} never ran (observed: {sorted(observed)})"
    assert (torch.float32, torch.bfloat16) in observed[expected_fn], (
        f"{case}: {expected_fn} never saw FP32 activations against BF16 weights "
        f"(observed: {sorted((str(a), str(b)) for a, b in observed[expected_fn])})"
    )

    # Guard the comparison itself: observe the baseline too and require that it did NOT enter any
    # memory-efficient autograd function. Without this, a refactor that made fusion unconditional
    # (or retired `use_memory_efficient_lora`) would turn every check below into self-vs-self, and
    # they would keep passing while the path under test was silently broken.
    with _observe_lora_dtypes() as baseline_observed:
        baseline, _ = _build_sharded_block(
            case, memory_efficient=False, mesh=mesh, device=device, reference_state=state
        )
        x_baseline = inputs.clone().requires_grad_(True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_baseline = baseline(x_baseline)
        (out_baseline.float() * grad_out).sum().backward()
    assert not baseline_observed, (
        f"{case}: baseline entered {sorted(baseline_observed)}; it must run on plain autograd or "
        "the parity check compares an implementation against itself"
    )

    # NB: do not assert gradient dtypes here. torch's autograd engine coerces a Function's returned
    # gradient to its input's dtype (verified on 2.10), so such an assertion can never fail and would
    # give false confidence. The compute-dtype half of the fix is what the parity checks below catch.
    efficient_grads, baseline_grads = _lora_grads(efficient), _lora_grads(baseline)
    assert efficient_grads.keys() == baseline_grads.keys()
    for name, grad in efficient_grads.items():
        assert torch.isfinite(grad).all(), f"{case}: non-finite gradient for {name}"

    assert _relative_error(out_efficient.float(), out_baseline.float()) < _TOLERANCE, case
    assert _relative_error(x_efficient.grad, x_baseline.grad) < _TOLERANCE, case
    for name, grad in efficient_grads.items():
        assert _relative_error(grad, baseline_grads[name]) < _TOLERANCE, f"{case}: gradient mismatch for {name}"


def _run_worker() -> None:
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"This regression requires exactly 2 ranks, got {dist.get_world_size()}.")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))

        for case in _CASES:
            _assert_lora_mixed_precision(case, device)
            dist.barrier()

        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + "PASS", flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least 2 CUDA devices")
def test_lora_mixed_precision_fsdp_two_rank() -> None:
    """Per-linear and fused-MLP LoRA backward work under FSDP2 FP32-activation mixed precision."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    # Inherit CUDA_VISIBLE_DEVICES: the skipif gate counts the parent's visible devices, so
    # hardcoding "0,1" here could push the workers onto GPUs this job was not assigned.
    env = os.environ.copy()
    repo_root = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=600,
        check=False,
    )
    print(completed.stdout)
    assert completed.returncode == 0, completed.stdout
    assert _RESULT_PREFIX + "PASS" in completed.stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    if _parse_args().worker:
        _run_worker()
