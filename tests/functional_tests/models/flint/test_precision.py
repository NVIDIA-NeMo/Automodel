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

"""One-GPU precision gates: independent CUDA reference, optimizer master states and cold compiled kernels."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.distributed.activation_checkpointing import make_selective_checkpoint_context_fn
from nemo_automodel.components.loss.loss import MaskedCrossEntropyConfig
from nemo_automodel.components.loss.utils import calculate_loss
from nemo_automodel.components.models.common.utils import Float32RMSNorm
from nemo_automodel.components.moe.megatron.moe_utils import weighted_bias_swiglu_impl
from nemo_automodel.components.optim.optimizer import AdamWConfig
from nemo_automodel.shared.import_utils import safe_import
from tests.functional_tests.models.flint.reference import (
    full_native_tensors,
    reference_logits,
    tiny_config,
    torch_backend,
)
from tests.functional_tests.models.flint.test_attention import _provenance

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


@pytest.mark.parametrize("precision", ["fp32", "bf16_master", "bf16_resident"])
def test_cuda_precision_matches_independent_reference(precision: str, tmp_path: Path) -> None:
    """Exercise actual CUDA math and precision hooks, without private checkpoint data."""
    if precision != "fp32" and not torch.cuda.is_bf16_supported():
        pytest.skip("requires CUDA BF16 support")
    torch.cuda.set_device(0)
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    handles = []
    try:
        torch.manual_seed(41)
        dtype = torch.bfloat16 if precision == "bf16_resident" else torch.float32
        compute = torch.float32 if precision == "fp32" else torch.bfloat16
        config = tiny_config(dtype)
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            dtype=dtype,
            attn_implementation="eager",
            backend=torch_backend(),
            use_liger_kernel=False,
            use_sdpa_patching=False,
        ).train()
        batch = {
            "input_ids": torch.tensor([[5, 2, 9, 4, 7], [8, 6, 2, 2, 2]], device="cuda"),
            "labels": torch.tensor([[2, 9, 4, 7, 3], [6, 2, 3, -100, -100]], device="cuda"),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device="cuda"),
        }
        original = model.state_dict_adapter.to_hf(full_native_tensors(model))
        reference = {name: tensor.clone().requires_grad_() for name, tensor in original.items()}
        expected, expected_routes = reference_logits(
            reference, config, batch, compute_dtype=compute, routed_precision_contract=precision == "bf16_master"
        )
        per_head_reference = None
        if precision == "bf16_master":
            per_head_reference = {name: tensor.clone().requires_grad_() for name, tensor in original.items()}
            per_head_expected, _ = reference_logits(per_head_reference, config, batch, compute_dtype=compute)
        sink_observations, head_observations, actual_routes = [], [], {}

        def capture_sink(module: torch.nn.Module, inputs: tuple, output: torch.Tensor) -> None:
            """Observe one sink call without altering its trainable FP32 tensor.

            Args:
                module: Sink parameter holder.
                inputs: Empty tuple; this module takes no activation inputs.
                output: FP32 CUDA tensor [query_heads] with gradients to its parameter.
            """
            sink_observations.append((output.dtype, torch.is_autocast_enabled("cuda")))

        def capture_head(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            """Observe head input precision without changing its activation.

            Args:
                module: Output linear module, owning [vocab,hidden] weight.
                inputs: One CUDA tensor [batch,sequence,hidden], FP32 for head compute.
            """
            head_observations.append((inputs[0].dtype, module.weight.dtype, torch.is_autocast_enabled("cuda")))

        def gate_capture(index: int):
            """Bind a layer index for a numerical route diagnostic."""

            def capture(module: torch.nn.Module, inputs: tuple[torch.Tensor, ...], output: tuple) -> None:
                """Record valid learned routing outputs without mutating the graph.

                Args:
                    module: Shared learned Gate, with gate_precision=None.
                    inputs: Activations [tokens,hidden], validity [tokens], and optional mesh.
                    output: FP32 weights and int64 IDs [tokens,top_k], plus None auxiliary loss.
                """
                assert module.gate_precision is None and module.e_score_correction_bias is None
                assert output[0].dtype == torch.float32 and output[2] is None
                actual_routes[index] = output[0].detach().clone(), output[1].detach().clone()

            return capture

        handles.append(model.lm_head.register_forward_pre_hook(capture_head))
        for index, layer in enumerate(model.model.layers):
            assert layer.self_attn._fp32_params.softmax_offset.dtype == torch.float32
            handles.append(layer.self_attn._fp32_params.register_forward_hook(capture_sink))
            if index not in config.mlp_only_layers:
                handles.append(layer.mlp.gate.register_forward_hook(gate_capture(index)))
        context = torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16_master" else nullcontext()
        with context:
            actual = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits
        assert actual.dtype == torch.float32
        assert sink_observations == [(torch.float32, False)] * config.num_hidden_layers
        assert head_observations == [(torch.float32, torch.float32, False)]
        valid = batch["attention_mask"].bool()
        atol, rtol = (3e-6, 3e-5) if precision == "fp32" else (1e-3, 5e-2)
        torch.testing.assert_close(actual[valid], expected[valid], atol=atol, rtol=rtol)
        if per_head_reference is not None:
            torch.testing.assert_close(actual[valid], per_head_expected[valid], atol=atol, rtol=rtol)
        for index, (expected_weights, expected_ids) in expected_routes.items():
            weights, ids = actual_routes[index]
            order, expected_order = ids.argsort(-1), expected_ids.argsort(-1)
            flat_valid = valid.flatten()
            torch.testing.assert_close(
                ids.gather(1, order)[flat_valid], expected_ids.gather(1, expected_order)[flat_valid]
            )
            torch.testing.assert_close(
                weights.gather(1, order)[flat_valid],
                expected_weights.gather(1, expected_order)[flat_valid],
                atol=1e-7 if precision == "fp32" else 2e-3,
                rtol=rtol,
            )
        loss = calculate_loss(
            MaskedCrossEntropyConfig().build(),
            logits=actual,
            labels=batch["labels"],
            num_label_tokens=(batch["labels"] != -100).sum(),
        )
        expected_loss = F.cross_entropy(expected.reshape(-1, config.vocab_size), batch["labels"].reshape(-1))
        torch.testing.assert_close(loss, expected_loss, atol=atol, rtol=rtol)
        loss.backward()
        expected_loss.backward()
        if per_head_reference is not None:
            per_head_loss = F.cross_entropy(
                per_head_expected.reshape(-1, config.vocab_size), batch["labels"].reshape(-1)
            )
            torch.testing.assert_close(loss, per_head_loss, atol=atol, rtol=rtol)
            per_head_loss.backward()
        gradients = model.state_dict_adapter.to_hf(full_native_tensors(model, gradients=True))
        gradient_atol = 3e-6 if precision == "fp32" else 3e-4
        gradient_rtol = 4e-4 if precision == "fp32" else 8e-2
        errors = {}
        per_head_errors = {}
        for name, value in reference.items():
            assert value.grad is not None and torch.isfinite(gradients[name]).all(), name
            torch.testing.assert_close(gradients[name], value.grad, atol=gradient_atol, rtol=gradient_rtol, msg=name)
            errors[name] = float((gradients[name].float() - value.grad.float()).abs().max())
            if per_head_reference is not None:
                per_head_grad = per_head_reference[name].grad
                assert per_head_grad is not None, name
                torch.testing.assert_close(
                    gradients[name], per_head_grad, atol=gradient_atol, rtol=gradient_rtol, msg=name
                )
                per_head_errors[name] = float((gradients[name].float() - per_head_grad.float()).abs().max())
        for index in range(config.num_hidden_layers):
            sink = model.model.layers[index].self_attn._fp32_params.softmax_offset
            assert sink.grad.dtype == torch.float32 and torch.count_nonzero(sink.grad) > 0
        assert torch.count_nonzero(model.lm_head.weight.grad) > 0
        reference_preclip = {name: value.grad.detach().clone() for name, value in reference.items()}
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        expected_norm = torch.nn.utils.clip_grad_norm_(list(reference.values()), 1.0)
        torch.testing.assert_close(norm.float(), expected_norm.float(), atol=gradient_atol, rtol=gradient_rtol)
        metrics = {
            "precision": precision,
            "device": torch.cuda.get_device_name(),
            "loss": loss.item(),
            "reference_loss": expected_loss.item(),
            "valid_logit_max_abs_error": float((actual[valid] - expected[valid]).detach().abs().max()),
            "gradient_max_abs_error": max(errors.values()),
            "gradient_error_by_name": errors,
            "global_norm": norm.item(),
            "reference_global_norm": expected_norm.item(),
            "compared_hf_tensors": len(reference),
            "optimizer_update_exercised": precision != "bf16_resident",
            "routed_update_oracle": "selected_fused_fp32_adjoint" if precision == "bf16_master" else "eager_autograd",
        }
        if per_head_reference is not None:
            per_head_norm = torch.nn.utils.clip_grad_norm_(list(per_head_reference.values()), 1.0)
            torch.testing.assert_close(norm.float(), per_head_norm.float(), atol=gradient_atol, rtol=gradient_rtol)
            metrics["per_head_gradient_max_abs_error"] = max(per_head_errors.values())
            metrics["per_head_valid_logit_max_abs_error"] = float(
                (actual[valid] - per_head_expected[valid]).detach().abs().max()
            )
            metrics["per_head_global_norm"] = per_head_norm.item()
        if precision != "bf16_resident":
            optimizer = AdamWConfig(lr=0.003, weight_decay=0.01, eps=1e-6).build(model)[0]
            expected_optimizer = torch.optim.AdamW(list(reference.values()), lr=0.003, weight_decay=0.01, eps=1e-6)
            optimizer.step()
            expected_optimizer.step()
            updated = model.state_dict_adapter.to_hf(full_native_tensors(model))
            diagnostics = []
            clipped_gradients = model.state_dict_adapter.to_hf(full_native_tensors(model, gradients=True))
            matched_gradient_error = 0.0
            for name, value in reference.items():
                # First-step AdamW has this closed form after bias correction.
                # This independent optimizer diagnostic supplements, and never
                # replaces, the separate full-model reference update gate.
                native_gradient = clipped_gradients[name]
                matched_update = original[name] * (1 - 0.003 * 0.01) - (
                    0.003 * native_gradient / (native_gradient.abs() + 1e-6)
                )
                torch.testing.assert_close(updated[name], matched_update, atol=1e-7, rtol=1e-5, msg=name)
                matched_gradient_error = max(
                    matched_gradient_error, float((updated[name] - matched_update).abs().max())
                )
                error = (updated[name] - value).detach().abs()
                offset = int(error.reshape(-1).argmax())
                diagnostics.append(
                    {
                        "name": name,
                        "max_abs_error": float(error.max()),
                        "flat_index": offset,
                        "initial": float(original[name].reshape(-1)[offset]),
                        "actual": float(updated[name].reshape(-1)[offset]),
                        "reference": float(value.detach().reshape(-1)[offset]),
                        "actual_clipped_grad": float(clipped_gradients[name].reshape(-1)[offset]),
                        "reference_clipped_grad": float(value.grad.reshape(-1)[offset]),
                        "actual_preclip_grad": float(gradients[name].reshape(-1)[offset]),
                        "reference_preclip_grad": float(reference_preclip[name].reshape(-1)[offset]),
                        "sign_mismatches": int((clipped_gradients[name].sign() != value.grad.sign()).sum()),
                        "native_nearzero_count_at_adam_eps": int((clipped_gradients[name].abs() < 1e-6).sum()),
                        "reference_nearzero_count_at_adam_eps": int((value.grad.abs() < 1e-6).sum()),
                    }
                )
            metrics["updated_parameter_max_abs_error"] = max(item["max_abs_error"] for item in diagnostics)
            metrics["matched_native_gradient_adam_max_abs_error"] = matched_gradient_error
            metrics["native_adam_steps"] = sorted({float(state["step"]) for state in optimizer.state.values()})
            metrics["reference_adam_steps"] = sorted(
                {float(state["step"]) for state in expected_optimizer.state.values()}
            )
            metrics["largest_update_errors"] = sorted(
                diagnostics, key=lambda item: item["max_abs_error"], reverse=True
            )[:8]
            metrics["update_diagnostics_by_name"] = {item["name"]: item for item in diagnostics}
            # Preserve diagnostics even if a numerical acceptance assertion fails.
            (tmp_path / f"cuda_{precision}.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
            for name, value in reference.items():
                torch.testing.assert_close(
                    updated[name],
                    value,
                    atol=1e-6 if precision == "fp32" else 5e-4,
                    rtol=rtol,
                    msg=lambda message, key=name: f"{key}\n{message}",
                )
            assert all(state["exp_avg"].dtype == torch.float32 for state in optimizer.state.values())
        evidence = tmp_path / f"cuda_{precision}.json"
        evidence.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"FLINT_CUDA_EVIDENCE={evidence}")
    finally:
        for handle in handles:
            handle.remove()
        torch.backends.cuda.matmul.allow_tf32 = old_tf32


def _gradients(parameters: list[torch.nn.Parameter], step: int) -> None:
    """Assign deterministic, varying gradients in each resident parameter dtype."""
    for index, parameter in enumerate(parameters):
        values = torch.arange(parameter.numel(), device=parameter.device, dtype=torch.float32)
        parameter.grad = (0.03 * (values * 0.17 + step * 0.3 + index).sin()).reshape_as(parameter).to(parameter.dtype)


def test_fused_fp32_adamw_preserves_master_states_and_updates() -> None:
    """Bound the memory-saving fused CUDA update against scalar AdamW for nine steps."""
    torch.manual_seed(4375)
    fused = [torch.nn.Parameter(torch.randn(37, 19, device="cuda") * 0.1) for _ in range(3)]
    scalar = [torch.nn.Parameter(parameter.detach().clone()) for parameter in fused]
    options = dict(lr=5e-6, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    actual = torch.optim.AdamW(fused, fused=True, **options)
    reference = torch.optim.AdamW(scalar, fused=False, foreach=False, **options)
    for step in range(9):
        _gradients(fused, step)
        _gradients(scalar, step)
        actual.step()
        reference.step()
        for observed, expected in zip(fused, scalar, strict=True):
            torch.testing.assert_close(observed, expected, atol=5e-8, rtol=2e-6)
            for name in ("exp_avg", "exp_avg_sq"):
                assert actual.state[observed][name].dtype == torch.float32
                torch.testing.assert_close(
                    actual.state[observed][name], reference.state[expected][name], atol=2e-9, rtol=2e-6
                )


def test_te_bf16_moments_and_fp32_masters_survive_real_checkpoint(tmp_path: Path) -> None:
    """Check mixed BF16 bulk/FP32 protected units and exact resumed CUDA updates."""
    available, optimizers = safe_import("transformer_engine.pytorch.optimizers")
    assert available, "The long-context precision gate requires actual Transformer Engine"
    torch.manual_seed(4375)
    parameters = [
        torch.nn.Parameter((torch.randn(37, 19, device="cuda") * 0.1).to(torch.bfloat16)),
        torch.nn.Parameter(torch.randn(32, device="cuda") * 0.01),
        torch.nn.Parameter(torch.randn(31, 19, device="cuda") * 0.1),
    ]
    options = dict(
        lr=5e-6,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        adam_w_mode=True,
        master_weights=True,
        master_weight_dtype=torch.float32,
        store_param_remainders=True,
        exp_avg_dtype=torch.bfloat16,
        exp_avg_sq_dtype=torch.bfloat16,
    )
    optimizer = optimizers.FusedAdam(parameters, **options)
    initial = [parameter.detach().float().clone() for parameter in parameters]
    _gradients(parameters, 0)
    first_gradients = [parameter.grad.float().clone() for parameter in parameters]
    optimizer.step()
    # Independent first-step AdamW equation. BF16 bulk compares after the same
    # resident-weight cast; FP32 protected weights retain full precision.
    for parameter, before, gradient in zip(parameters, initial, first_gradients, strict=True):
        expected = before * (1 - options["lr"] * options["weight_decay"])
        expected -= options["lr"] * gradient / (gradient.abs() + options["eps"])
        torch.testing.assert_close(parameter, expected.to(parameter.dtype), atol=2e-7, rtol=0)
    for step in (1, 2):
        _gradients(parameters, step)
        optimizer.step()
    for parameter in parameters:
        state = optimizer.state[parameter]
        assert state["exp_avg"].dtype == state["exp_avg_sq"].dtype == torch.bfloat16
        assert torch.isfinite(state["exp_avg"]).all() and torch.isfinite(state["exp_avg_sq"]).all()
    assert optimizer.state[parameters[0]]["master_param"].dtype == torch.int16
    checkpoint = tmp_path / "mixed_precision_optimizer.pt"
    torch.save(
        {"parameters": [p.detach().clone() for p in parameters], "optimizer": optimizer.state_dict()}, checkpoint
    )
    saved = torch.load(checkpoint, weights_only=True)
    resumed_parameters = [torch.nn.Parameter(value.clone()) for value in saved["parameters"]]
    resumed_optimizer = optimizers.FusedAdam(resumed_parameters, **options)
    resumed_optimizer.load_state_dict(saved["optimizer"])
    for step in range(3, 7):
        _gradients(parameters, step)
        _gradients(resumed_parameters, step)
        optimizer.step()
        resumed_optimizer.step()
        for actual, resumed in zip(parameters, resumed_parameters, strict=True):
            torch.testing.assert_close(actual, resumed, atol=0, rtol=0)
            for name, value in optimizer.state[actual].items():
                other = resumed_optimizer.state[resumed][name]
                if isinstance(value, torch.Tensor):
                    assert value.dtype == other.dtype
                    torch.testing.assert_close(value, other, atol=0, rtol=0)
                else:
                    assert value == other


class _CompiledBlock(nn.Module):
    """Use the production compiled kernels without copying their implementations."""

    def __init__(self, dtype: torch.dtype) -> None:
        """Create CUDA norm [32], fused projection [48,32] and down [32,24]."""
        super().__init__()
        self.norm = Float32RMSNorm(32, eps=1e-6, device="cuda", dtype=dtype)
        self.up = nn.Linear(32, 48, bias=False, device="cuda", dtype=dtype)
        self.down = nn.Linear(24, 32, bias=False, device="cuda", dtype=dtype)
        self.norm_calls = 0
        self.activation_calls = 0

    def _activation(self, projected: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Evaluate shared compiled weighted SwiGLU and count Python invocations.

        Args:
            projected: Fused gate/up activations [tokens,48], BF16 or FP32.
            weights: Differentiable FP32 route probabilities [tokens,1].

        Returns:
            Weighted activations [tokens,24] in projected's dtype.
        """
        self.activation_calls += 1
        return weighted_bias_swiglu_impl(projected, weights)

    def forward(self, inputs: torch.Tensor, weights: torch.Tensor, *, nested: bool = False) -> torch.Tensor:
        """Compute a residual projection using optional inner nonreentrant AC.

        Args:
            inputs: CUDA activations [13,32], matching the projection dtype.
            weights: Differentiable FP32 route probabilities [13,1].
            nested: Put the activation inside an additional nonreentrant checkpoint.

        Returns:
            Residual output [13,32] in inputs' dtype. Counters observe forward
            invocations, including recompute; SAC may retain expensive subops.
        """
        self.norm_calls += 1
        projected = self.up(self.norm(inputs))
        activated = (
            checkpoint(self._activation, projected, weights, use_reentrant=False)
            if nested
            else self._activation(projected, weights)
        )
        return self.down(activated) + inputs * 0.125


@pytest.mark.parametrize("mode", ["selective", "nested"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cold_compiled_kernels_selective_and_nested_checkpoint(mode: str, dtype: torch.dtype) -> None:
    """Require recomputation plus forward and every input/parameter adjoint parity."""
    torch.manual_seed(437527)
    block = _CompiledBlock(dtype)
    control = copy.deepcopy(block)
    inputs = (torch.randn(13, 32, device="cuda", dtype=torch.float32) * 0.3).to(dtype).requires_grad_()
    weights = torch.linspace(0.13, 0.87, 13, device="cuda")[:, None].requires_grad_()
    expected_inputs = inputs.detach().clone().requires_grad_()
    expected_weights = weights.detach().clone().requires_grad_()
    upstream = torch.linspace(-0.7, 1.3, inputs.numel(), device="cuda").reshape_as(inputs)

    # The checkpointed candidate makes the first calls after reset, before the
    # uncheckpointed control can warm either production compiled entry point.
    torch._dynamo.reset()
    from torch._inductor import metrics

    kernels_before = metrics.generated_kernel_count
    with set_checkpoint_early_stop(False):
        output = checkpoint(
            lambda x, w: block(x, w, nested=mode == "nested"),
            inputs,
            weights,
            use_reentrant=False,
            context_fn=make_selective_checkpoint_context_fn(),
        )
        (output.float() * upstream).sum().backward()
    candidate_kernels = metrics.generated_kernel_count - kernels_before
    assert block.norm_calls >= 2 and block.activation_calls >= 2, (block.norm_calls, block.activation_calls)
    assert candidate_kernels > 0, "Cold candidate must reach actual Inductor compilation"

    expected = control(expected_inputs, expected_weights)
    (expected.float() * upstream).sum().backward()
    pairs = {
        "output": (output, expected),
        "input_grad": (inputs.grad, expected_inputs.grad),
        "route_grad": (weights.grad, expected_weights.grad),
    }
    control_parameters = dict(control.named_parameters())
    for name, parameter in block.named_parameters():
        pairs[f"parameter_grad.{name}"] = (parameter.grad, control_parameters[name].grad)
    errors = {}
    for name, (observed, reference) in pairs.items():
        assert observed is not None and reference is not None, name
        assert observed.dtype == reference.dtype and torch.isfinite(observed).all(), name
        if name != "output":
            assert torch.count_nonzero(reference), name
        torch.testing.assert_close(observed, reference, atol=0, rtol=0, msg=name)
        errors[name] = float((observed.float() - reference.float()).abs().max())
    result = {
        "status": "passed",
        "mode": mode,
        "dtype": str(dtype),
        "provenance": _provenance(),
        "test_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "production_kernel_sha256": {
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "nemo_automodel/components/models/common/utils.py",
                "nemo_automodel/components/moe/megatron/moe_utils.py",
                "nemo_automodel/components/distributed/activation_checkpointing.py",
            )
        },
        "candidate_before_control": True,
        "candidate_generated_kernels": candidate_kernels,
        "norm_forward_invocations": block.norm_calls,
        "swiglu_forward_invocations": block.activation_calls,
        "recomputation_scope": "Checkpointed body and kernel wrappers re-entered; selective policy may save expensive compiled subops.",
        "all_forward_input_route_and_parameter_gradients_exact": True,
        "max_abs_errors": errors,
    }
    directory = Path(os.environ.get("FLINT_BACKEND_EVIDENCE", "/tmp/flint-compiled-checkpoint"))
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"compiled_checkpoint_{mode}_{dtype}.json").write_text(json.dumps(result, indent=2) + "\n")
