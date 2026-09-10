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

"""Real TE kernels are necessary to verify saved backward precision choices."""

import copy

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.utils.checkpoint import set_checkpoint_early_stop

from nemo_automodel.components.models.common.utils import TEFp8Config
from nemo_automodel.shared.import_utils import safe_import

HAVE_TE, te = safe_import("transformer_engine.pytorch")
pytestmark = pytest.mark.skipif(not HAVE_TE or not torch.cuda.is_available(), reason="requires TE and CUDA")


@pytest.mark.parametrize("recipe", ["current", "block", "mxfp8"])
def test_exclusions_preserve_bf16_forward_backward_and_nested_autocast(recipe):
    supported, reason = {
        "current": te.quantization.is_fp8_available,
        "block": te.quantization.is_fp8_block_scaling_available,
        "mxfp8": te.quantization.is_mxfp8_available,
    }[recipe](return_reason=True)
    if not supported:
        # TE's block recipe needs CUDA >= 12.9; MXFP8 needs Blackwell.
        pytest.skip(reason)
    torch.manual_seed(42)
    # Block scaling needs 128-wide GEMMs. Match a non-TE parent name to select
    # multiple TE children, not just one leaf, and catch incorrectly bound forwards.
    model = nn.ModuleDict(
        {
            "keep": nn.Sequential(
                te.Linear(128, 128, params_dtype=torch.bfloat16),
                te.Linear(128, 128, params_dtype=torch.bfloat16),
            ),
            "quantized": te.Linear(128, 128, params_dtype=torch.bfloat16),
            "norm": nn.LayerNorm(128, device="cuda", dtype=torch.bfloat16),
        }
    )
    reference = nn.Sequential(
        te.Linear(128, 128, params_dtype=torch.bfloat16),
        te.Linear(128, 128, params_dtype=torch.bfloat16),
    )
    reference.load_state_dict(model["keep"].state_dict())
    cfg = TEFp8Config(recipe=recipe, filter_fqns=["keep", "norm"])
    norm_forward = model["norm"].forward
    assert cfg.apply_filter_fqns(model) == ["keep.0", "keep.1"]
    forwards = [layer.forward for layer in model["keep"]]
    assert cfg.apply_filter_fqns(model) == ["keep.0", "keep.1"]
    assert [layer.forward for layer in model["keep"]] == forwards
    assert model["norm"].forward == norm_forward

    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x)
    expected = reference(reference_x)
    expected.backward(upstream)
    with torch.no_grad():
        unquantized = model["quantized"](x)

    # Repeated calls recreate the disabled context and leave both enclosing
    # enabled scopes intact, including for the sibling called after exclusion.
    for _ in range(2):
        model.zero_grad(set_to_none=True)
        x.grad = None
        with cfg.maybe_te_autocast():
            before = model["quantized"](x)
            with cfg.maybe_te_autocast():
                actual = model["keep"](x)
            after = model["quantized"](x)
        # Identical TE BF16 kernels and operands must be bit-exact: no FP8-sized
        # tolerance that could accidentally allow a quantized execution path.
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(before, after, atol=0, rtol=0)
        assert not torch.equal(after, unquantized), "unmatched sibling must still use quantized GEMM"
        # Additional coverage for backward outside autocast; the recipe keeps
        # backward inside the enabled region (covered by the checkpoint test).
        actual.backward(upstream)
        torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
        for parameter, reference_parameter in zip(model["keep"].parameters(), reference.parameters()):
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0)
        sibling_grads = torch.autograd.grad(after, (x, *model["quantized"].parameters()), upstream)
        assert all(torch.isfinite(gradient).all() for gradient in sibling_grads)


@pytest.mark.parametrize("recipe", ["current", "block", "mxfp8"])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_exclusions_preserve_bf16_through_checkpoint_recompute(recipe, checkpointed):
    supported, reason = {
        "current": te.quantization.is_fp8_available,
        "block": te.quantization.is_fp8_block_scaling_available,
        "mxfp8": te.quantization.is_mxfp8_available,
    }[recipe](return_reason=True)
    if not supported:
        # TE's block recipe needs CUDA >= 12.9; MXFP8 needs Blackwell.
        pytest.skip(reason)
    torch.manual_seed(44)

    class Projections(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = te.Linear(128, 128, params_dtype=torch.bfloat16)
            self.k_proj = te.Linear(128, 128, params_dtype=torch.bfloat16)
            self.outputs = []

        def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Run independent projections so sibling quantization cannot mask query parity.

            Args:
                query: BF16 CUDA tensor of shape [tokens, hidden].
                key: BF16 CUDA tensor of shape [tokens, hidden].

            Returns:
                Query and key projections, each of shape [tokens, hidden].
            """
            query_out = self.q_proj(query)
            key_out = self.k_proj(key)
            self.outputs.append((query_out.detach().clone(), key_out.detach().clone()))
            return query_out, key_out

    attention = Projections()
    model = nn.ModuleDict(
        {
            "self_attn": checkpoint_wrapper(attention, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
            if checkpointed
            else attention
        }
    )
    reference = te.Linear(128, 128, params_dtype=torch.bfloat16)
    reference.load_state_dict(attention.q_proj.state_dict())
    cfg = TEFp8Config(recipe=recipe, filter_fqns=["self_attn.q_proj"])
    assert cfg.apply_filter_fqns(model) == ["self_attn.q_proj"]
    query = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    reference_query = query.detach().clone().requires_grad_()
    upstream = torch.randn_like(query)
    sibling_upstream = torch.randn_like(key)
    expected = reference(reference_query)
    expected.backward(upstream)
    with torch.no_grad():
        sibling_bf16 = attention.k_proj(key)

    # Disable early stopping only to observe completed GEMMs in recompute.
    # Forward AND backward remain inside the enabled region, as in train_ft.
    with cfg.maybe_te_autocast():
        with set_checkpoint_early_stop(False):
            actual, sibling = model["self_attn"](query, key)
        assert len(attention.outputs) == 1
        torch.autograd.backward((actual, sibling), (upstream, sibling_upstream))
        with torch.no_grad():
            sibling_after = attention.k_proj(key)

    assert len(attention.outputs) == (2 if checkpointed else 1)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(query.grad, reference_query.grad, atol=0, rtol=0)
    for parameter, reference_parameter in zip(attention.q_proj.parameters(), reference.parameters()):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0)
    # Inspect actual GEMM results, not just the FP8 flag: both the original
    # forward and recompute must leave the unmatched sibling quantized.
    for query_out, key_out in attention.outputs:
        torch.testing.assert_close(query_out, expected, atol=0, rtol=0)
        torch.testing.assert_close(key_out, sibling, atol=0, rtol=0)
        assert not torch.equal(key_out, sibling_bf16)
    torch.testing.assert_close(sibling_after, sibling, atol=0, rtol=0)
    assert torch.isfinite(key.grad).all()
    assert all(torch.isfinite(parameter.grad).all() for parameter in attention.k_proj.parameters())


def test_exclusion_deepcopy_owns_weights_and_gradients():
    supported, reason = te.quantization.is_fp8_available(return_reason=True)
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(45)
    source = nn.ModuleDict({"proj": te.Linear(32, 32, params_dtype=torch.bfloat16)})
    cfg = TEFp8Config(filter_fqns=["proj"])
    cfg.apply_filter_fqns(source)
    cloned = copy.deepcopy(source)
    with torch.no_grad():
        source["proj"].weight.zero_()
        source["proj"].bias.zero_()
        cloned["proj"].weight.normal_()
        cloned["proj"].bias.fill_(2)
    reference = te.Linear(32, 32, params_dtype=torch.bfloat16)
    reference.load_state_dict(cloned["proj"].state_dict())
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    upstream = torch.randn_like(x)
    expected = reference(reference_x)
    expected.backward(upstream)
    with cfg.maybe_te_autocast():
        actual = cloned["proj"](x)
        actual.backward(upstream)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(x.grad, reference_x.grad, atol=0, rtol=0)
    for parameter, reference_parameter in zip(cloned["proj"].parameters(), reference.parameters()):
        assert torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, atol=0, rtol=0)
    assert all(parameter.grad is None for parameter in source.parameters())


def test_exclusion_restores_autocast_after_forward_failure():
    supported, reason = te.quantization.is_fp8_available(return_reason=True)
    if not supported:
        pytest.skip(reason)
    torch.manual_seed(43)
    model = nn.ModuleDict(
        {
            "keep": te.Linear(32, 32, params_dtype=torch.bfloat16),
            "quantized": te.Linear(32, 32, params_dtype=torch.bfloat16),
        }
    )
    cfg = TEFp8Config(filter_fqns=["keep"])
    cfg.apply_filter_fqns(model)
    x = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        baseline = model["quantized"](x)
        with cfg.maybe_te_autocast():
            before = model["quantized"](x)
            # A real TE forward failure must propagate without leaking the
            # nested disabled state into the remainder of the training step.
            with pytest.raises((RuntimeError, AssertionError)):
                model["keep"](x[:, :16])
            after = model["quantized"](x)
        restored = model["quantized"](x)
    torch.testing.assert_close(before, after, atol=0, rtol=0)
    assert not torch.equal(after, baseline)
    torch.testing.assert_close(restored, baseline, atol=0, rtol=0)
