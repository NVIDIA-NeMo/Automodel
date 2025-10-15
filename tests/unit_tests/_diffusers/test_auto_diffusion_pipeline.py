# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch


MODULE_PATH = "nemo_automodel.components._diffusers.auto_diffusion_pipeline"


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()


class DummyPipeline:
    """
    Minimal stand-in for diffusers.DiffusionPipeline that supports the
    attributes/methods we exercise in tests.
    """

    def __init__(self, components=None):
        # components is a mapping name->object to emulate .components registry
        if components is None:
            components = {}
        # assign components first without triggering syncing logic
        object.__setattr__(self, "components", dict(components))
        # also expose each nn.Module as an attribute like real Diffusers pipelines
        for name, value in self.components.items():
            if isinstance(value, torch.nn.Module):
                object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        # Keep components dict synchronized when modules are set as attributes
        if name != "components" and "components" in self.__dict__ and isinstance(value, torch.nn.Module):
            self.components[name] = value
        object.__setattr__(self, name, value)


@patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False)
def test_choose_device_cpu_when_no_cuda(mock_is_available):
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _choose_device

    dev = _choose_device(None)
    assert dev.type == "cpu"


@patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=True)
@patch.dict("os.environ", {"LOCAL_RANK": "2"}, clear=False)
def test_choose_device_uses_cuda_and_local_rank(mock_is_available):
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _choose_device

    dev = _choose_device(None)
    assert dev.type == "cuda"
    # torch.device('cuda', index) string contains index; verify index property directly
    assert dev.index == 2


def test_choose_device_respects_explicit_device():
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _choose_device

    explicit = torch.device("cpu")
    dev = _choose_device(explicit)
    assert dev is explicit


def test_iter_pipeline_modules_prefers_components_registry():
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _iter_pipeline_modules

    m1, m2 = DummyModule(), DummyModule()
    pipe = DummyPipeline({"unet": m1, "text_encoder": m2, "scheduler": object()})

    names = [name for name, _ in _iter_pipeline_modules(pipe)]
    assert set(names) == {"unet", "text_encoder"}


def test_iter_pipeline_modules_fallback_attribute_scan():
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _iter_pipeline_modules

    class AttrPipe:
        def __init__(self):
            self.unet = DummyModule()
            self._private = DummyModule()  # should be ignored
            self.non_module = 3

    pipe = AttrPipe()
    out = list(_iter_pipeline_modules(pipe))
    assert out and out[0][0] == "unet" and isinstance(out[0][1], DummyModule)
    assert all(name != "_private" for name, _ in out)


@pytest.mark.parametrize("torch_dtype,expected_dtype", [("auto", None), (torch.float16, torch.float16), ("float32", torch.float32)])
def test_move_module_to_device_respects_dtype(torch_dtype, expected_dtype):
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import _move_module_to_device

    mod = DummyModule()
    dev = torch.device("cpu")

    with patch.object(torch.nn.Module, "to") as mock_to:
        _move_module_to_device(mod, dev, torch_dtype)

    # verify torch.nn.Module.to was called with expected args
    if expected_dtype is None:
        mock_to.assert_called_once_with(device=dev)
    else:
        mock_to.assert_called_once_with(device=dev, dtype=expected_dtype)


def test_from_pretrained_basic_flow_moves_modules_and_returns_pipeline(caplog):
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    # Prepare a real DummyPipeline instance containing two nn.Modules
    m1, m2 = DummyModule(), DummyModule()
    dummy_pipe = DummyPipeline({"unet": m1, "text_encoder": m2})

    with (
        patch("diffusers.DiffusionPipeline.from_pretrained", return_value=dummy_pipe) as mock_hf_from,
        patch.object(torch.nn.Module, "to") as mock_to,
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        caplog.set_level(logging.WARNING)
        out = NeMoAutoDiffusionPipeline.from_pretrained("dummy")

    assert out is dummy_pipe
    assert mock_hf_from.call_count == 1
    # Both modules should be moved to device once
    assert mock_to.call_count == 2


def test_from_pretrained_skips_move_when_flag_false():
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    dummy_pipe = DummyPipeline({"unet": DummyModule()})
    with (
        patch("diffusers.DiffusionPipeline.from_pretrained", return_value=dummy_pipe),
        patch.object(torch.nn.Module, "to") as mock_to,
    ):
        out = NeMoAutoDiffusionPipeline.from_pretrained("dummy", move_to_device=False)

    assert out is dummy_pipe
    mock_to.assert_not_called()


def test_from_pretrained_parallel_scheme_applies_managers_and_sets_attrs():
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    unet = DummyModule()
    text_encoder = DummyModule()
    dummy_pipe = DummyPipeline({"unet": unet, "text_encoder": text_encoder})

    # Manager returns a wrapped module for one component and same object for the other
    new_unet = DummyModule()
    mgr_unet = Mock()
    mgr_unet.parallelize.return_value = new_unet
    mgr_text = Mock()
    mgr_text.parallelize.return_value = text_encoder

    parallel_scheme = {"unet": mgr_unet, "text_encoder": mgr_text}

    with (
        patch("diffusers.DiffusionPipeline.from_pretrained", return_value=dummy_pipe),
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
    ):
        out = NeMoAutoDiffusionPipeline.from_pretrained("dummy", parallel_scheme=parallel_scheme, move_to_device=False)

    assert out is dummy_pipe
    # unet was replaced
    assert dummy_pipe.components["unet"] is new_unet
    # text_encoder unchanged
    assert dummy_pipe.components["text_encoder"] is text_encoder
    mgr_unet.parallelize.assert_called_once_with(unet)
    mgr_text.parallelize.assert_called_once_with(text_encoder)


def test_from_pretrained_parallel_scheme_logs_and_continues_on_errors(caplog):
    from nemo_automodel.components._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    comp = DummyModule()
    dummy_pipe = DummyPipeline({"unet": comp})

    mgr = Mock()
    mgr.parallelize.side_effect = RuntimeError("boom")

    with (
        patch("diffusers.DiffusionPipeline.from_pretrained", return_value=dummy_pipe),
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
        caplog.at_level(logging.WARNING),
    ):
        out = NeMoAutoDiffusionPipeline.from_pretrained("dummy", parallel_scheme={"unet": mgr}, move_to_device=False)

    assert out is dummy_pipe
    assert "parallelize failed" in caplog.text


