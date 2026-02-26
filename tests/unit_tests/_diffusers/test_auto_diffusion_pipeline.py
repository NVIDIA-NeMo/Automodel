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
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

# Check if diffusers can be imported properly (may fail due to peft/transformers incompatibility)
try:
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _choose_device
    DIFFUSERS_AVAILABLE = True
except Exception:
    DIFFUSERS_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DIFFUSERS_AVAILABLE, reason="diffusers not available or incompatible with current transformers version")

MODULE_PATH = "nemo_automodel._diffusers.auto_diffusion_pipeline"


class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


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


# =============================================================================
# _choose_device tests
# =============================================================================


@patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False)
def test_choose_device_cpu_when_no_cuda(mock_is_available):
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _choose_device

    dev = _choose_device(None)
    assert dev.type == "cpu"


@patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=True)
@patch.dict("os.environ", {"LOCAL_RANK": "2"}, clear=False)
def test_choose_device_uses_cuda_and_local_rank(mock_is_available):
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _choose_device

    dev = _choose_device(None)
    assert dev.type == "cuda"
    assert dev.index == 2


def test_choose_device_respects_explicit_device():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _choose_device

    explicit = torch.device("cpu")
    dev = _choose_device(explicit)
    assert dev is explicit


# =============================================================================
# _iter_pipeline_modules tests
# =============================================================================


def test_iter_pipeline_modules_prefers_components_registry():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _iter_pipeline_modules

    m1, m2 = DummyModule(), DummyModule()
    pipe = DummyPipeline({"unet": m1, "text_encoder": m2, "scheduler": object()})

    names = [name for name, _ in _iter_pipeline_modules(pipe)]
    assert set(names) == {"unet", "text_encoder"}


def test_iter_pipeline_modules_fallback_attribute_scan():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _iter_pipeline_modules

    class AttrPipe:
        def __init__(self):
            self.unet = DummyModule()
            self._private = DummyModule()  # should be ignored
            self.non_module = 3

    pipe = AttrPipe()
    out = list(_iter_pipeline_modules(pipe))
    assert out and out[0][0] == "unet" and isinstance(out[0][1], DummyModule)
    assert all(name != "_private" for name, _ in out)


# =============================================================================
# _move_module_to_device tests
# =============================================================================


@pytest.mark.parametrize("torch_dtype,expected_dtype", [("auto", None), (torch.float16, torch.float16), ("float32", torch.float32)])
def test_move_module_to_device_respects_dtype(torch_dtype, expected_dtype):
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _move_module_to_device

    mod = DummyModule()
    dev = torch.device("cpu")

    with patch.object(torch.nn.Module, "to") as mock_to:
        _move_module_to_device(mod, dev, torch_dtype)

    if expected_dtype is None:
        mock_to.assert_called_once_with(device=dev)
    else:
        mock_to.assert_called_once_with(device=dev, dtype=expected_dtype)


# =============================================================================
# _ensure_params_trainable tests
# =============================================================================


def test_ensure_params_trainable_sets_requires_grad():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _ensure_params_trainable

    mod = DummyModule()
    for p in mod.parameters():
        p.requires_grad = False

    count = _ensure_params_trainable(mod, module_name="test_mod")

    assert count > 0
    assert all(p.requires_grad for p in mod.parameters())


def test_ensure_params_trainable_returns_correct_count():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _ensure_params_trainable

    mod = DummyModule()
    expected = sum(p.numel() for p in mod.parameters())
    count = _ensure_params_trainable(mod)
    assert count == expected


# =============================================================================
# PipelineSpec tests
# =============================================================================


def test_pipeline_spec_from_dict_none():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import PipelineSpec

    spec = PipelineSpec.from_dict(None)
    assert spec.transformer_cls == ""
    assert spec.subfolder == "transformer"


def test_pipeline_spec_from_dict_with_values():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import PipelineSpec

    spec = PipelineSpec.from_dict({
        "transformer_cls": "FluxTransformer2DModel",
        "subfolder": "transformer",
        "load_full_pipeline": True,
        "unknown_field": "ignored",
    })
    assert spec.transformer_cls == "FluxTransformer2DModel"
    assert spec.load_full_pipeline is True


def test_pipeline_spec_validate_for_from_config_raises_on_empty_cls():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import PipelineSpec

    spec = PipelineSpec()
    with pytest.raises(ValueError, match="transformer_cls is required"):
        spec.validate_for_from_config()


def test_pipeline_spec_validate_for_from_config_passes_with_cls():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import PipelineSpec

    spec = PipelineSpec(transformer_cls="FluxTransformer2DModel")
    spec.validate_for_from_config()  # should not raise


# =============================================================================
# _create_parallel_manager tests
# =============================================================================


def test_create_parallel_manager_fsdp2_default():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _create_parallel_manager

    with patch(f"{MODULE_PATH}.FSDP2Manager") as MockFSDP2:
        MockFSDP2.return_value = Mock()
        manager = _create_parallel_manager({"some_arg": "value"})

    MockFSDP2.assert_called_once_with(some_arg="value")
    assert manager is MockFSDP2.return_value


def test_create_parallel_manager_ddp():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _create_parallel_manager

    with patch(f"{MODULE_PATH}.DDPManager") as MockDDP:
        MockDDP.return_value = Mock()
        manager = _create_parallel_manager({"_manager_type": "ddp", "some_arg": "value"})

    MockDDP.assert_called_once_with(some_arg="value")
    assert manager is MockDDP.return_value


def test_create_parallel_manager_explicit_fsdp2():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _create_parallel_manager

    with patch(f"{MODULE_PATH}.FSDP2Manager") as MockFSDP2:
        MockFSDP2.return_value = Mock()
        manager = _create_parallel_manager({"_manager_type": "fsdp2"})

    MockFSDP2.assert_called_once_with()


def test_create_parallel_manager_unknown_type_raises():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _create_parallel_manager

    with pytest.raises(ValueError, match="Unknown manager type"):
        _create_parallel_manager({"_manager_type": "unknown"})


def test_create_parallel_manager_does_not_mutate_input():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _create_parallel_manager

    original = {"_manager_type": "ddp", "key": "val"}
    original_copy = original.copy()

    with patch(f"{MODULE_PATH}.DDPManager") as MockDDP:
        MockDDP.return_value = Mock()
        _create_parallel_manager(original)

    assert original == original_copy


# =============================================================================
# _apply_parallelization tests
# =============================================================================


def test_apply_parallelization_returns_empty_when_scheme_is_none():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _apply_parallelization

    pipe = DummyPipeline({"unet": DummyModule()})
    result = _apply_parallelization(pipe, None)
    assert result == {}


def test_apply_parallelization_creates_managers_and_replaces_modules():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _apply_parallelization

    unet = DummyModule()
    new_unet = DummyModule()
    pipe = DummyPipeline({"unet": unet, "text_encoder": DummyModule()})

    mock_manager = Mock()
    mock_manager.parallelize.return_value = new_unet

    with (
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
        patch(f"{MODULE_PATH}._create_parallel_manager", return_value=mock_manager) as mock_create,
        patch(f"{MODULE_PATH}._init_parallelizer"),
    ):
        managers = _apply_parallelization(pipe, {"unet": {"_manager_type": "fsdp2"}})

    mock_create.assert_called_once_with({"_manager_type": "fsdp2"})
    mock_manager.parallelize.assert_called_once_with(unet)
    assert managers == {"unet": mock_manager}
    # unet was replaced on the pipeline
    assert pipe.unet is new_unet


def test_apply_parallelization_skips_components_not_in_scheme():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _apply_parallelization

    unet = DummyModule()
    text_encoder = DummyModule()
    pipe = DummyPipeline({"unet": unet, "text_encoder": text_encoder})

    mock_manager = Mock()
    mock_manager.parallelize.return_value = DummyModule()

    with (
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
        patch(f"{MODULE_PATH}._create_parallel_manager", return_value=mock_manager),
        patch(f"{MODULE_PATH}._init_parallelizer"),
    ):
        managers = _apply_parallelization(pipe, {"unet": {"_manager_type": "fsdp2"}})

    # Only unet should be parallelized
    assert "unet" in managers
    assert "text_encoder" not in managers
    # text_encoder should be unchanged
    assert pipe.text_encoder is text_encoder


# =============================================================================
# from_pretrained tests
# =============================================================================


def test_from_pretrained_returns_pipe_and_managers_tuple(caplog):
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    m1, m2 = DummyModule(), DummyModule()
    dummy_pipe = DummyPipeline({"unet": m1, "text_encoder": m2})

    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch.object(torch.nn.Module, "to") as mock_to,
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        caplog.set_level(logging.WARNING)
        result = NeMoAutoDiffusionPipeline.from_pretrained("dummy")

    # from_pretrained now returns (pipe, managers) tuple
    assert isinstance(result, tuple)
    assert len(result) == 2
    pipe, managers = result
    assert pipe is dummy_pipe
    assert isinstance(managers, dict)
    assert mock_diffusion_pipeline.from_pretrained.call_count == 1
    # Both modules should be moved to device once
    assert mock_to.call_count == 2


def test_from_pretrained_skips_move_when_flag_false():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    dummy_pipe = DummyPipeline({"unet": DummyModule()})
    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch.object(torch.nn.Module, "to") as mock_to,
    ):
        pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained("dummy", move_to_device=False)

    assert pipe is dummy_pipe
    mock_to.assert_not_called()


def test_from_pretrained_parallel_scheme_applies_managers_and_sets_attrs():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    unet = DummyModule()
    text_encoder = DummyModule()
    dummy_pipe = DummyPipeline({"unet": unet, "text_encoder": text_encoder})

    # Mock managers returned by _create_parallel_manager
    new_unet = DummyModule()
    mgr_unet = Mock()
    mgr_unet.parallelize.return_value = new_unet
    mgr_text = Mock()
    mgr_text.parallelize.return_value = text_encoder

    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    # _create_parallel_manager is called once per component; return different managers
    manager_sequence = [mgr_unet, mgr_text]

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
        patch(f"{MODULE_PATH}._create_parallel_manager", side_effect=manager_sequence),
        patch(f"{MODULE_PATH}._init_parallelizer"),
    ):
        # parallel_scheme values are now dicts (manager kwargs), not manager objects
        pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained(
            "dummy",
            parallel_scheme={"unet": {"_manager_type": "fsdp2"}, "text_encoder": {"_manager_type": "fsdp2"}},
            move_to_device=False,
        )

    assert pipe is dummy_pipe
    # unet was replaced
    assert dummy_pipe.components["unet"] is new_unet
    # text_encoder unchanged (mgr_text.parallelize returns same object)
    assert dummy_pipe.components["text_encoder"] is text_encoder
    mgr_unet.parallelize.assert_called_once_with(unet)
    mgr_text.parallelize.assert_called_once_with(text_encoder)
    assert managers == {"unet": mgr_unet, "text_encoder": mgr_text}


def test_from_pretrained_parallel_scheme_propagates_errors():
    """Parallelization errors propagate as exceptions (not silently logged)."""
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    comp = DummyModule()
    dummy_pipe = DummyPipeline({"unet": comp})

    mgr = Mock()
    mgr.parallelize.side_effect = RuntimeError("boom")

    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch(f"{MODULE_PATH}.torch.distributed.is_initialized", return_value=True),
        patch(f"{MODULE_PATH}._create_parallel_manager", return_value=mgr),
        patch(f"{MODULE_PATH}._init_parallelizer"),
    ):
        with pytest.raises(RuntimeError, match="boom"):
            NeMoAutoDiffusionPipeline.from_pretrained(
                "dummy",
                parallel_scheme={"unet": {"_manager_type": "fsdp2"}},
                move_to_device=False,
            )


def test_from_pretrained_load_for_training_makes_params_trainable():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    mod = DummyModule()
    for p in mod.parameters():
        p.requires_grad = False

    dummy_pipe = DummyPipeline({"transformer": mod})
    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        pipe, managers = NeMoAutoDiffusionPipeline.from_pretrained(
            "dummy", load_for_training=True, enable_gradient_checkpointing=False,
        )

    assert all(p.requires_grad for p in mod.parameters())


def test_from_pretrained_gradient_checkpointing_enabled():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    transformer = DummyModule()
    transformer.enable_gradient_checkpointing = Mock()

    dummy_pipe = DummyPipeline({"transformer": transformer})
    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained("dummy", enable_gradient_checkpointing=True)

    transformer.enable_gradient_checkpointing.assert_called_once()


def test_from_pretrained_raises_when_diffusers_unavailable():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    with patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="diffusers is required"):
            NeMoAutoDiffusionPipeline.from_pretrained("dummy")


def test_from_pretrained_components_to_load_filters_modules():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    unet = DummyModule()
    text_encoder = DummyModule()
    dummy_pipe = DummyPipeline({"unet": unet, "text_encoder": text_encoder})

    mock_diffusion_pipeline = MagicMock()
    mock_diffusion_pipeline.from_pretrained.return_value = dummy_pipe

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}.DiffusionPipeline", mock_diffusion_pipeline),
        patch(f"{MODULE_PATH}._move_module_to_device") as mock_move,
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            "dummy", components_to_load=["unet"],
        )

    # Only unet should be moved, not text_encoder
    assert mock_move.call_count == 1
    assert mock_move.call_args[0][0] is unet


# =============================================================================
# from_config tests
# =============================================================================


def test_from_config_transformer_only_mode():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    mock_transformer_cls = MagicMock()
    mock_transformer = DummyModule()
    mock_transformer_cls.load_config.return_value = {"some": "config"}
    mock_transformer_cls.from_config.return_value = mock_transformer

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}._import_diffusers_class", return_value=mock_transformer_cls),
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        pipe, managers = NeMoAutoDiffusionPipeline.from_config(
            "model-id",
            pipeline_spec={"transformer_cls": "FakeTransformer", "subfolder": "transformer"},
        )

    assert isinstance(pipe, NeMoAutoDiffusionPipeline)
    assert pipe.transformer is not None
    assert managers == {}
    mock_transformer_cls.load_config.assert_called_once_with("model-id", subfolder="transformer")
    mock_transformer_cls.from_config.assert_called_once()


def test_from_config_raises_when_diffusers_unavailable():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    with patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", False):
        with pytest.raises(RuntimeError, match="diffusers is required"):
            NeMoAutoDiffusionPipeline.from_config("dummy", pipeline_spec={"transformer_cls": "X"})


def test_from_config_raises_without_transformer_cls():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
    ):
        with pytest.raises(ValueError, match="transformer_cls is required"):
            NeMoAutoDiffusionPipeline.from_config("dummy", pipeline_spec={})


def test_from_config_full_pipeline_mode():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    mock_transformer_cls = MagicMock()
    mock_transformer = DummyModule()
    mock_transformer_cls.load_config.return_value = {}
    mock_transformer_cls.from_config.return_value = mock_transformer

    mock_pipeline_cls = MagicMock()
    full_pipe = DummyPipeline({"transformer": mock_transformer})
    mock_pipeline_cls.from_pretrained.return_value = full_pipe

    def import_class(name):
        if name == "FakeTransformer":
            return mock_transformer_cls
        if name == "FakePipeline":
            return mock_pipeline_cls
        raise ImportError(name)

    with (
        patch(f"{MODULE_PATH}.DIFFUSERS_AVAILABLE", True),
        patch(f"{MODULE_PATH}._import_diffusers_class", side_effect=import_class),
        patch(f"{MODULE_PATH}.torch.cuda.is_available", return_value=False),
    ):
        pipe, managers = NeMoAutoDiffusionPipeline.from_config(
            "model-id",
            pipeline_spec={
                "transformer_cls": "FakeTransformer",
                "pipeline_cls": "FakePipeline",
                "load_full_pipeline": True,
            },
        )

    assert pipe is full_pipe
    mock_pipeline_cls.from_pretrained.assert_called_once()


# =============================================================================
# _import_diffusers_class tests
# =============================================================================


def test_import_diffusers_class_success():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _import_diffusers_class

    with patch("diffusers.SomeClass", create=True, new="sentinel"):
        result = _import_diffusers_class("SomeClass")
    assert result == "sentinel"


def test_import_diffusers_class_missing_raises():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import _import_diffusers_class

    with pytest.raises(ImportError, match="not found in diffusers"):
        _import_diffusers_class("NonExistentClassName12345")


# =============================================================================
# NeMoAutoDiffusionPipeline wrapper tests
# =============================================================================


def test_pipeline_wrapper_init_and_components():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    transformer = DummyModule()
    pipe = NeMoAutoDiffusionPipeline(transformer=transformer)

    assert pipe.transformer is transformer
    assert "transformer" in pipe.components
    assert pipe.components["transformer"] is transformer


def test_pipeline_wrapper_components_excludes_none():
    from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline

    pipe = NeMoAutoDiffusionPipeline(transformer=None)
    assert "transformer" not in pipe.components
