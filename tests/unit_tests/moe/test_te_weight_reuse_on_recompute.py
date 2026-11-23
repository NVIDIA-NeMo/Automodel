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

"""Quantized weights of Transformer Engine GEMM modules across an activation-checkpoint recompute.

FP8 with current scaling runs on every GPU Transformer Engine quantizes on; MXFP8 and NVFP4 need Blackwell.
"""

from collections.abc import Callable
from contextlib import AbstractContextManager
from typing import Literal

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import (
    HAVE_TE,
    TENvfp4Config,
    _patch_te_modules,
    set_is_first_microbatch,
)
from nemo_automodel.shared.import_utils import is_te_min_version
from nemo_automodel.shared.recompute_replay import mark_checkpoint_recompute


def _block_scaling_capable() -> bool:
    return HAVE_TE and torch.cuda.is_available() and torch.cuda.get_device_capability() >= (10, 0)


pytestmark = pytest.mark.skipif(
    not (HAVE_TE and torch.cuda.is_available() and is_te_min_version("2.19.0")),
    reason="requires Transformer Engine 2.19+ and a GPU",
)

RecipeName = Literal["fp8", "mxfp8", "nvfp4"]
_blackwell_only = pytest.mark.skipif(not _block_scaling_capable(), reason="MXFP8 and NVFP4 GEMMs need Blackwell")
RECIPES = ["fp8", pytest.param("mxfp8", marks=_blackwell_only), pytest.param("nvfp4", marks=_blackwell_only)]
ModuleKind = Literal["grouped_linear", "linear"]
ContextFn = Callable[[], tuple[AbstractContextManager, AbstractContextManager]]

EXPERTS = 4
ROWS = 128  # per expert: the NVFP4 group alignment
IN_FEATURES = 128
OUT_FEATURES = 256


@pytest.fixture(autouse=True)
def _microbatch_flag():
    _patch_te_modules()
    yield
    set_is_first_microbatch(None)


def _recipe(name: RecipeName):
    """A recipe without random rounding, so that equal operands give equal bits."""
    from transformer_engine.common.recipe import Float8CurrentScaling, MXFP8BlockScaling

    if name == "fp8":
        return Float8CurrentScaling()
    if name == "mxfp8":
        return MXFP8BlockScaling()
    return TENvfp4Config(stochastic_rounding=False).build_recipe()


def _module(kind: ModuleKind) -> torch.nn.Module:
    import transformer_engine.pytorch as te

    torch.manual_seed(0)
    if kind == "grouped_linear":
        return te.GroupedLinear(EXPERTS, IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16).cuda()
    return te.Linear(IN_FEATURES, OUT_FEATURES, bias=False, params_dtype=torch.bfloat16).cuda()


def _count_weight_quantizations(kind: ModuleKind, monkeypatch: pytest.MonkeyPatch) -> list[bool]:
    """Record, per weight handed to Transformer Engine's weight quantization, whether it was quantized.

    Returns:
        The list the calls append to: ``True`` for a quantization (no stored copy, or a refresh of it),
        ``False`` for a reuse of the stored copy.
    """
    import transformer_engine.pytorch.module.grouped_linear as grouped_linear_module
    import transformer_engine.pytorch.module.linear as linear_module

    te_module = grouped_linear_module if kind == "grouped_linear" else linear_module
    quantize_weight = te_module.quantize_weight
    quantized: list[bool] = []

    def counting(**kwargs):
        quantized.append(kwargs["workspace"] is None or kwargs["update_workspace"])
        return quantize_weight(**kwargs)

    monkeypatch.setattr(te_module, "quantize_weight", counting)
    return quantized


def _checkpointed_step(
    module: torch.nn.Module,
    kind: ModuleKind,
    recipe_name: RecipeName,
    context_fn: ContextFn | None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Forward and backward of ``module`` through a non-reentrant checkpoint, both inside the autocast.

    Returns:
        Output of shape ``[EXPERTS * ROWS, OUT_FEATURES]``, input gradient of shape
        ``[EXPERTS * ROWS, IN_FEATURES]`` and one gradient of shape ``[OUT_FEATURES, IN_FEATURES]`` per weight.
    """
    from transformer_engine.pytorch.quantization import autocast as te_autocast

    generator = torch.Generator(device="cuda").manual_seed(1)
    rows = EXPERTS * ROWS
    inputs = torch.randn(rows, IN_FEATURES, device="cuda", generator=generator).bfloat16().requires_grad_(True)
    output_grad = torch.randn(rows, OUT_FEATURES, device="cuda", generator=generator).bfloat16()
    for parameter in module.parameters():
        parameter.grad = None

    def body(x: torch.Tensor) -> torch.Tensor:
        if kind == "grouped_linear":
            return module(x, [ROWS] * EXPERTS)
        return module(x)

    checkpoint_kwargs = {} if context_fn is None else {"context_fn": context_fn}
    with te_autocast(enabled=True, recipe=_recipe(recipe_name)):
        output = checkpoint(body, inputs, use_reentrant=False, **checkpoint_kwargs)
        output.backward(output_grad)
    return output.detach(), inputs.grad, [parameter.grad.clone() for parameter in module.parameters()]


def _assert_same(
    left: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
    right: tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]],
) -> None:
    assert torch.equal(left[0], right[0])
    assert torch.equal(left[1], right[1])
    assert len(left[2]) == len(right[2])
    for left_grad, right_grad in zip(left[2], right[2], strict=True):
        assert torch.equal(left_grad, right_grad)


@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("kind", ["grouped_linear", "linear"])
def test_recompute_of_a_first_microbatch_reuses_the_weights_its_forward_quantized(
    kind: ModuleKind, recipe_name: RecipeName, monkeypatch: pytest.MonkeyPatch
):
    module = _module(kind)
    num_weights = len(list(module.parameters()))
    quantized = _count_weight_quantizations(kind, monkeypatch)

    # Without the mark the recompute quantizes every weight a second time.
    set_is_first_microbatch(True)
    unmarked = _checkpointed_step(module, kind, recipe_name, None)
    assert quantized == [True] * num_weights + [True] * num_weights

    quantized.clear()
    marked = _checkpointed_step(module, kind, recipe_name, mark_checkpoint_recompute(None))
    assert quantized == [True] * num_weights + [False] * num_weights
    _assert_same(marked, unmarked)

    # Later micro-batches reuse the copy in the forward and in the recompute, as before.
    quantized.clear()
    set_is_first_microbatch(False)
    later = _checkpointed_step(module, kind, recipe_name, mark_checkpoint_recompute(None))
    assert quantized == [False] * (2 * num_weights)
    _assert_same(later, unmarked)


@pytest.mark.parametrize("recipe_name", RECIPES)
@pytest.mark.parametrize("kind", ["grouped_linear", "linear"])
def test_reused_weights_follow_an_optimizer_step(kind: ModuleKind, recipe_name: RecipeName):
    module = _module(kind)
    marked_context = mark_checkpoint_recompute(None)

    set_is_first_microbatch(True)
    before = _checkpointed_step(module, kind, recipe_name, marked_context)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.mul_(-1.5)

    # A forward that reuses the copy of the old weights is what a stale reuse would compute.
    set_is_first_microbatch(False)
    stale = _checkpointed_step(module, kind, recipe_name, marked_context)
    _assert_same(stale, before)

    # The first micro-batch after the update quantizes the new weights and its recompute reuses those.
    set_is_first_microbatch(True)
    after = _checkpointed_step(module, kind, recipe_name, marked_context)
    # Reference: no stored copy at all, every forward and recompute quantizes the weights it is given.
    set_is_first_microbatch(None)
    reference = _checkpointed_step(module, kind, recipe_name, None)
    _assert_same(after, reference)
    assert not torch.equal(after[0], before[0])


def test_recompute_without_a_microbatch_flag_keeps_quantizing(monkeypatch: pytest.MonkeyPatch):
    module = _module("grouped_linear")
    quantized = _count_weight_quantizations("grouped_linear", monkeypatch)
    set_is_first_microbatch(None)
    _checkpointed_step(module, "grouped_linear", "fp8", mark_checkpoint_recompute(None))
    assert quantized == [True] * (2 * EXPERTS)
