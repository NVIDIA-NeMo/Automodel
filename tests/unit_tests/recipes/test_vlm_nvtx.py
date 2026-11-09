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

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.checkpoint import checkpoint

import nemo_automodel.recipes.vlm.finetune as vlm_finetune
from nemo_automodel import autonvtx
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


@pytest.fixture
def nvtx_ranges(monkeypatch):
    active = []
    entered = []
    underflows = []

    def push(name):
        active.append(name)
        entered.append(name)

    def pop():
        # PyTorch warns instead of propagating always_call hook errors during
        # an existing forward exception. Record underflows for teardown instead.
        if active:
            active.pop()
        else:
            underflows.append(True)

    monkeypatch.setattr(torch.cuda.nvtx, "range_push", push)
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", pop)
    yield active, entered
    leaked = set(autonvtx._get_active_ranges())
    autonvtx._get_active_ranges().clear()
    assert not underflows
    assert not active
    assert not leaked


class _Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class _Pipeline:
    def __init__(self, parts):
        self.parts = parts


class _SetupComplete(Exception):
    """Stop setup after model construction, before dataset downloads."""


@pytest.mark.parametrize("nvtx", [None, False, True])
@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("owns_training", [False, True])
def test_vlm_setup_nvtx_preserves_forward_backward(monkeypatch, nvtx_ranges, nvtx, pipeline, owns_training):
    torch.manual_seed(7)
    parts = [torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU(inplace=True))]
    if pipeline:
        parts.append(torch.nn.Linear(3, 2))
    references = copy.deepcopy(parts)
    model = _Pipeline(parts) if pipeline else parts[0]
    recipe = object.__new__(FinetuneRecipeForVLM)
    recipe.cfg = _Config(
        wandb=None,
        trackio=None,
        mlflow=None,
        model=object(),
        loss_fn=SimpleNamespace(build=lambda: torch.nn.MSELoss()),
        checkpoint=SimpleNamespace(build=lambda **_: object(), load_metadata_with_gloo=False),
        optimizer=SimpleNamespace(build=lambda *a, **k: object()),
        vlm_dataloader=SimpleNamespace(is_packed=False, output_format="batched"),
    )
    if nvtx is not None:
        recipe.cfg.nvtx = nvtx
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(vlm_finetune, "initialize_distributed", lambda **_: SimpleNamespace(is_main=False))
    monkeypatch.setattr(vlm_finetune, "setup_logging", lambda: None)
    monkeypatch.setattr(vlm_finetune, "apply_cache_compatibility_patches", lambda: None)
    monkeypatch.setattr(vlm_finetune, "StatefulRNG", lambda **_: None)
    monkeypatch.setattr(recipe, "_create_distributed_setup", lambda: None)
    monkeypatch.setattr(recipe, "_should_setup_training_components", lambda: owns_training)
    monkeypatch.setattr(
        recipe,
        "_distributed_setup_attributes",
        lambda _: (
            None,
            SimpleNamespace(cp_size=1, pp_size=2),
            None,
            None,
            None,
            pipeline,
            SimpleNamespace(),
            None,
            None,
        ),
    )
    monkeypatch.setattr(vlm_finetune, "setup_magi", lambda *a, **k: None)
    for name in ("_log_experiment_details", "_log_library_versions"):
        monkeypatch.setattr(recipe, name, lambda: None)
    for name in ("_get_dp_rank", "_get_tp_rank", "_get_pp_rank", "_get_pp_group"):
        monkeypatch.setattr(recipe, name, lambda **_: None)
    monkeypatch.setattr(vlm_finetune, "build_model", lambda *a, **k: model)
    monkeypatch.setattr(vlm_finetune, "AutoPipeline", _Pipeline)
    monkeypatch.setattr(vlm_finetune, "apply_te_patches", lambda: None)
    monkeypatch.setattr(vlm_finetune, "shard_optimizers_for_megatron_fsdp", lambda *a, **k: None)

    def after_model_setup(*args):
        raise _SetupComplete

    monkeypatch.setattr(vlm_finetune, "_maybe_downgrade_loss_fn", after_model_setup)
    if pipeline:
        recipe.cfg.__dict__["step_scheduler.local_batch_size"] = 2

    if owns_training:
        with pytest.raises(_SetupComplete):
            recipe.setup()
        assert recipe.model_parts == parts
        assert recipe.pp is (model if pipeline else None)
    else:
        recipe.setup()
        assert not hasattr(recipe, "model_parts")
    active, entered = nvtx_ranges
    for index, (part, reference) in enumerate(zip(parts, references)):
        x = torch.randn(2, 3, requires_grad=True)
        reference_x = x.detach().clone().requires_grad_()
        actual = part(x)
        expected = reference(reference_x)
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        expected.sum().backward()
        torch.testing.assert_close(x.grad, reference_x.grad)
        for parameter, reference_parameter in zip(part.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter.grad, reference_parameter.grad)
        if nvtx and owns_training:
            expected_root = f"PipelineStage_{index}" if pipeline else type(part).__name__
            assert any(name.startswith(expected_root + ":") for name in entered)
        assert not part._backward_hooks
        assert not active
    assert bool(entered) == bool(nvtx and owns_training)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_nvtx_forward_ranges_balance_with_checkpoint_recomputation(nvtx_ranges, use_reentrant):
    torch.manual_seed(11)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU(inplace=True), torch.nn.Linear(3, 2))
    reference = copy.deepcopy(model)
    autonvtx.patch(model, add_backward_hooks=False)
    x = torch.randn(2, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = checkpoint(model, x, use_reentrant=use_reentrant)
    expected = checkpoint(reference, reference_x, use_reentrant=use_reentrant)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    active, entered = nvtx_ranges
    assert entered.count("Sequential") == 2
    assert not active


def test_nvtx_default_backward_hooks_preserve_forward_backward(nvtx_ranges):
    torch.manual_seed(13)
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU(), torch.nn.Linear(3, 2))
    reference = copy.deepcopy(model)
    autonvtx.patch(model)
    x = torch.randn(2, 3, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    actual = model(x)
    expected = reference(reference_x)
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, reference_x.grad)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad)
    active, entered = nvtx_ranges
    assert entered.count("Sequential") == 2
    assert not active


def test_nvtx_forward_error_closes_ranges_and_preserves_exception(nvtx_ranges):
    class FailingModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Reject a malformed input.

            Args:
                x: Input tensor of shape [features].
            """
            raise ValueError("invalid image input")

    model = torch.nn.Sequential(FailingModel())
    autonvtx.patch(model, add_backward_hooks=False)
    with pytest.raises(ValueError, match="invalid image input"):
        model(torch.ones(1))
    active, entered = nvtx_ranges
    assert entered == ["Sequential", "0: FailingModel"]
    assert not active


@pytest.mark.parametrize("run_before_failure", [False, True])
@pytest.mark.parametrize("global_hook", [False, True])
@pytest.mark.parametrize("add_backward_hooks", [False, True])
def test_nvtx_existing_pre_hook_error_closes_owned_ranges(
    nvtx_ranges, request, run_before_failure, global_hook, add_backward_hooks
):
    fail = False

    def existing_pre_hook(module, inputs):
        """Fail before a child forward.

        Args:
            module: Module about to execute.
            inputs: Positional inputs containing a tensor of shape [batch, features].
        """
        if fail and module is child:
            raise ValueError("invalid pre-hook input")

    child = torch.nn.Linear(3, 2)
    if global_hook:
        handle = torch.nn.modules.module.register_module_forward_pre_hook(existing_pre_hook)
    else:
        handle = child.register_forward_pre_hook(existing_pre_hook)
    request.addfinalizer(handle.remove)
    model = torch.nn.Sequential(child)
    autonvtx.patch(model, add_backward_hooks=add_backward_hooks)
    if run_before_failure:
        model(torch.ones(1, 3)).sum().backward()
    fail = True
    with pytest.raises(ValueError, match="invalid pre-hook input"):
        model(torch.ones(1, 3))
    active, entered = nvtx_ranges
    assert entered[-1:] == ["Sequential"]
    assert not active
