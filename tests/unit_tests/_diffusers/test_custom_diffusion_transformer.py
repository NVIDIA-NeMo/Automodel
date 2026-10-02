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

"""Custom-model (Automodel) diffusion transformers: detection, dispatch and option validation."""

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel._diffusers import auto_diffusion_pipeline as adp
from nemo_automodel._diffusers.auto_diffusion_pipeline import (
    NeMoAutoDiffusionPipeline,
    _reject_diffusers_only_options,
    build_custom_transformer,
    custom_transformer_config_dir,
)
from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING, ModelRegistry, register_architecture
from nemo_automodel.components.models.common import BackendConfig
from tests.functional_tests.diffusion.toy_moe_dit import (
    TOY_MOE_DIT_ARCHITECTURE,
    ToyMoEDiTConfig,
    ToyMoEDiTForDiffusion,
    register_toy_moe_dit,
    toy_backend,
    write_toy_moe_dit_checkpoint,
)

# A built-in architecture of the static registry; detection only reads the name.
NATIVE_ARCH = "Qwen3MoeForCausalLM"
UNKNOWN_ARCH = "NotARegisteredDiffusionTransformer"


def _write_config(directory, architectures):
    os.makedirs(directory, exist_ok=True)
    config = {"model_type": "dummy"}
    if architectures is not None:
        config["architectures"] = architectures
    with open(os.path.join(directory, "config.json"), "w") as f:
        json.dump(config, f)


def _write_model_index(root):
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "model_index.json"), "w") as f:
        json.dump({"_class_name": "DummyPipeline", "transformer": ["dummy", "Dummy"]}, f)


def _diffusers_repo(tmp_path, architectures, subfolder="transformer"):
    root = str(tmp_path / "repo")
    _write_model_index(root)
    _write_config(os.path.join(root, subfolder), architectures)
    return root


class _FrozenTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        for param in self.parameters():
            param.requires_grad_(False)


def test_custom_arch_fixture_is_in_static_registry():
    assert NATIVE_ARCH in MODEL_ARCH_MAPPING
    assert UNKNOWN_ARCH not in MODEL_ARCH_MAPPING


# =============================================================================
# custom_transformer_config_dir
# =============================================================================


def test_config_dir_detects_transformer_subfolder_of_diffusers_repo(tmp_path):
    root = _diffusers_repo(tmp_path, [NATIVE_ARCH])

    assert custom_transformer_config_dir(root) == os.path.join(root, "transformer")


def test_config_dir_honors_custom_subfolder(tmp_path):
    root = _diffusers_repo(tmp_path, [NATIVE_ARCH], subfolder="dit")

    assert custom_transformer_config_dir(root, subfolder="dit") == os.path.join(root, "dit")
    assert custom_transformer_config_dir(root) is None


def test_config_dir_ignores_root_config_of_diffusers_repo(tmp_path):
    """With model_index.json only the transformer subfolder is consulted, never the repo root."""
    root = str(tmp_path / "repo")
    _write_model_index(root)
    _write_config(root, [NATIVE_ARCH])

    assert custom_transformer_config_dir(root) is None


def test_config_dir_detects_single_model_repo_at_root(tmp_path):
    root = str(tmp_path / "single")
    _write_config(root, [NATIVE_ARCH])
    # A transformer/ subfolder is irrelevant without model_index.json.
    _write_config(os.path.join(root, "transformer"), [UNKNOWN_ARCH])

    assert custom_transformer_config_dir(root) == root


@pytest.mark.parametrize("diffusers_layout", [True, False])
def test_config_dir_returns_none_for_unregistered_architecture(tmp_path, diffusers_layout):
    if diffusers_layout:
        root = _diffusers_repo(tmp_path, [UNKNOWN_ARCH])
    else:
        root = str(tmp_path / "single")
        _write_config(root, [UNKNOWN_ARCH])

    assert custom_transformer_config_dir(root) is None


@pytest.mark.parametrize("architectures", [None, []])
def test_config_dir_returns_none_without_architectures(tmp_path, architectures):
    """Diffusers transformer configs carry ``_class_name`` and no ``architectures``."""
    root = _diffusers_repo(tmp_path, architectures)

    assert custom_transformer_config_dir(root) is None


def test_config_dir_returns_none_without_config(tmp_path):
    root = str(tmp_path / "repo")
    _write_model_index(root)

    assert custom_transformer_config_dir(root) is None
    assert custom_transformer_config_dir(str(tmp_path / "empty_dir")) is None


def test_config_dir_only_considers_first_architecture(tmp_path):
    root = _diffusers_repo(tmp_path, [UNKNOWN_ARCH, NATIVE_ARCH])

    assert custom_transformer_config_dir(root) is None


@pytest.mark.parametrize("diffusers_layout", [True, False])
def test_config_dir_detects_registered_toy_moe_dit_checkpoint(tmp_path, diffusers_layout):
    register_toy_moe_dit()
    root = write_toy_moe_dit_checkpoint(str(tmp_path / "toy"), diffusers_layout=diffusers_layout)

    expected = os.path.join(root, "transformer") if diffusers_layout else root
    assert custom_transformer_config_dir(root) == expected


def test_config_dir_detects_runtime_registered_architecture(tmp_path):
    arch = "RuntimeOnlyCustomDiffusionTransformer"
    register_architecture(arch, ToyMoEDiTForDiffusion, exist_ok=True)
    try:
        assert ModelRegistry.has_custom_model(arch)
        root = _diffusers_repo(tmp_path, [arch])
        assert custom_transformer_config_dir(root) == os.path.join(root, "transformer")
    finally:
        ModelRegistry.model_arch_name_to_cls._extra.pop(arch, None)


# =============================================================================
# _reject_diffusers_only_options
# =============================================================================


def test_reject_options_accepts_disabled_options_without_mesh():
    _reject_diffusers_only_options(None, transformer_engine_linear=False, attention_backend=None, active_transformer="")


def test_reject_options_lists_every_enabled_option():
    with pytest.raises(ValueError) as excinfo:
        _reject_diffusers_only_options(
            None, transformer_engine_linear=True, fuse_qkv_projections=False, attention_backend="flash"
        )

    message = str(excinfo.value)
    assert "['attention_backend', 'transformer_engine_linear']" in message
    assert "fuse_qkv_projections" not in message


def test_reject_options_rejects_context_parallelism():
    _reject_diffusers_only_options(SimpleNamespace(cp_size=1))
    with pytest.raises(ValueError, match="cp_size > 1"):
        _reject_diffusers_only_options(SimpleNamespace(cp_size=2))


# =============================================================================
# build_custom_transformer
# =============================================================================


@pytest.fixture
def toy_config_dir(tmp_path):
    register_toy_moe_dit()
    root = write_toy_moe_dit_checkpoint(str(tmp_path / "toy"), num_hidden_layers=2)
    return os.path.join(root, "transformer")


def _patch_from_config(monkeypatch):
    from nemo_automodel._transformers import auto_model

    sentinel = nn.Linear(1, 1)
    from_config = MagicMock(return_value=sentinel)
    monkeypatch.setattr(auto_model.NeMoAutoModelForCausalLM, "from_config", from_config)
    return from_config, sentinel


def test_build_custom_transformer_forwards_mesh_overrides_and_backend(monkeypatch, toy_config_dir):
    from nemo_automodel.components.distributed import config as distributed_config

    from_config, sentinel = _patch_from_config(monkeypatch)
    setup_cls = MagicMock(side_effect=lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(distributed_config, "DistributedSetup", setup_cls)
    strategy_config, moe_parallel_config = object(), object()
    mesh_context = SimpleNamespace(
        cp_size=1,
        strategy_config=strategy_config,
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=True,
    )
    peft_cfg = object()

    model = build_custom_transformer(
        toy_config_dir,
        mesh_context=mesh_context,
        torch_dtype=torch.float32,
        load_base_model=True,
        peft_cfg=peft_cfg,
        backend={"experts": "torch", "dispatcher": "torch", "linear": "torch"},
        config_overrides={"num_hidden_layers": 1, "router_aux_loss_coef": 0.5},
    )

    assert model is sentinel
    (config,), kwargs = from_config.call_args
    assert isinstance(config, ToyMoEDiTConfig)
    assert config.architectures == [TOY_MOE_DIT_ARCHITECTURE]
    assert config.num_hidden_layers == 1
    assert config.router_aux_loss_coef == 0.5
    # The MeshContext policy is lifted onto the DistributedSetup so the model infrastructure
    # (FSDP / EP sharding, meta-device init, checkpoint loading) is instantiated.
    setup_cls.assert_called_once_with(
        mesh_context=mesh_context,
        strategy_config=strategy_config,
        moe_parallel_config=moe_parallel_config,
        activation_checkpointing=True,
    )
    assert kwargs["distributed_setup"].mesh_context is mesh_context
    assert kwargs["load_base_model"] is True
    assert kwargs["torch_dtype"] == torch.float32
    assert kwargs["peft_config"] is peft_cfg
    assert kwargs["trust_remote_code"] is False
    assert kwargs["use_liger_kernel"] is False
    assert kwargs["use_sdpa_patching"] is False
    assert isinstance(kwargs["backend"], BackendConfig)
    assert kwargs["backend"].experts == "torch"
    assert kwargs["backend"].dispatcher == "torch"


def test_build_custom_transformer_without_mesh_or_backend(monkeypatch, toy_config_dir):
    from_config, _ = _patch_from_config(monkeypatch)

    build_custom_transformer(toy_config_dir, mesh_context=None, torch_dtype=torch.bfloat16, load_base_model=False)

    (config,), kwargs = from_config.call_args
    assert config.num_hidden_layers == 2
    assert kwargs["distributed_setup"] is None
    assert kwargs["load_base_model"] is False
    assert kwargs["peft_config"] is None
    assert "backend" not in kwargs


def test_build_custom_transformer_passes_backend_config_through(monkeypatch, toy_config_dir):
    from_config, _ = _patch_from_config(monkeypatch)
    backend = toy_backend()

    build_custom_transformer(
        toy_config_dir, mesh_context=None, torch_dtype=torch.float32, load_base_model=False, backend=backend
    )

    assert from_config.call_args.kwargs["backend"] is backend


# =============================================================================
# NeMoAutoDiffusionPipeline.from_pretrained / from_config custom-model branch
# =============================================================================


@pytest.fixture
def custom_repo(tmp_path):
    return _diffusers_repo(tmp_path, [NATIVE_ARCH])


@pytest.fixture
def patched_custom_build(monkeypatch):
    """Replace the custom-model build and make any fallback to diffusers loading fail loudly."""
    transformer = _FrozenTransformer()
    build = MagicMock(return_value=transformer)
    monkeypatch.setattr(adp, "build_custom_transformer", build)
    monkeypatch.setattr(adp, "DIFFUSERS_AVAILABLE", True)
    diffusion_pipeline = SimpleNamespace(
        from_pretrained=MagicMock(side_effect=AssertionError("diffusers loading must not be used"))
    )
    monkeypatch.setattr(adp, "DiffusionPipeline", diffusion_pipeline)
    monkeypatch.setattr(
        adp, "_import_diffusers_class", MagicMock(side_effect=AssertionError("diffusers class must not be imported"))
    )
    return SimpleNamespace(build=build, transformer=transformer, diffusion_pipeline=diffusion_pipeline)


def test_from_pretrained_dispatches_to_custom_transformer(custom_repo, patched_custom_build):
    mesh_context = SimpleNamespace(cp_size=1)
    backend = {"experts": "torch"}
    overrides = {"num_hidden_layers": 1}

    pipe = NeMoAutoDiffusionPipeline.from_pretrained(
        custom_repo,
        torch_dtype=torch.float32,
        mesh_context=mesh_context,
        components_to_load=["transformer"],
        load_for_training=True,
        backend=backend,
        config_overrides=overrides,
    )

    assert isinstance(pipe, NeMoAutoDiffusionPipeline)
    assert pipe.transformer is patched_custom_build.transformer
    patched_custom_build.build.assert_called_once_with(
        os.path.join(custom_repo, "transformer"),
        mesh_context=mesh_context,
        torch_dtype=torch.float32,
        load_base_model=True,
        peft_cfg=None,
        backend=backend,
        config_overrides=overrides,
    )
    # Full finetuning: base weights are made trainable.
    assert all(param.requires_grad for param in pipe.transformer.parameters())


def test_from_pretrained_custom_keeps_base_weights_frozen_with_peft(custom_repo, patched_custom_build):
    peft_cfg = object()

    pipe = NeMoAutoDiffusionPipeline.from_pretrained(custom_repo, load_for_training=True, peft_cfg=peft_cfg)

    assert patched_custom_build.build.call_args.kwargs["peft_cfg"] is peft_cfg
    assert not any(param.requires_grad for param in pipe.transformer.parameters())


def test_from_pretrained_custom_without_training_keeps_params_frozen(custom_repo, patched_custom_build):
    pipe = NeMoAutoDiffusionPipeline.from_pretrained(custom_repo, load_for_training=False)

    assert not any(param.requires_grad for param in pipe.transformer.parameters())


@pytest.mark.parametrize(
    ("option", "value"),
    [
        ("active_transformer", "transformer_2"),
        ("transformer_engine_linear", True),
        ("fuse_qkv_projections", True),
        ("attention_backend", "flash"),
    ],
)
def test_from_pretrained_custom_rejects_diffusers_only_options(custom_repo, patched_custom_build, option, value):
    with pytest.raises(ValueError, match=option):
        NeMoAutoDiffusionPipeline.from_pretrained(custom_repo, **{option: value})

    patched_custom_build.build.assert_not_called()


def test_from_pretrained_custom_rejects_context_parallelism(custom_repo, patched_custom_build):
    with pytest.raises(ValueError, match="Context parallelism"):
        NeMoAutoDiffusionPipeline.from_pretrained(custom_repo, mesh_context=SimpleNamespace(cp_size=2))

    patched_custom_build.build.assert_not_called()


def test_from_pretrained_custom_rejects_non_transformer_components(custom_repo, patched_custom_build):
    with pytest.raises(ValueError, match="only the `transformer` component"):
        NeMoAutoDiffusionPipeline.from_pretrained(custom_repo, components_to_load=["transformer", "vae"])

    patched_custom_build.build.assert_not_called()


def test_from_pretrained_unregistered_architecture_uses_diffusers(tmp_path, patched_custom_build):
    root = _diffusers_repo(tmp_path, [UNKNOWN_ARCH])

    with pytest.raises(AssertionError, match="diffusers loading must not be used"):
        NeMoAutoDiffusionPipeline.from_pretrained(root)

    patched_custom_build.build.assert_not_called()
    patched_custom_build.diffusion_pipeline.from_pretrained.assert_called_once()


def test_from_config_dispatches_to_custom_transformer_with_random_init(tmp_path, patched_custom_build):
    root = _diffusers_repo(tmp_path, [NATIVE_ARCH], subfolder="dit")
    mesh_context = SimpleNamespace(cp_size=1)

    pipe = NeMoAutoDiffusionPipeline.from_config(
        root,
        # transformer_cls is a diffusers class name and is not needed for custom-model transformers.
        pipeline_spec={"subfolder": "dit"},
        torch_dtype=torch.float32,
        mesh_context=mesh_context,
        backend={"experts": "torch"},
        config_overrides={"num_hidden_layers": 1},
    )

    assert isinstance(pipe, NeMoAutoDiffusionPipeline)
    patched_custom_build.build.assert_called_once_with(
        os.path.join(root, "dit"),
        mesh_context=mesh_context,
        torch_dtype=torch.float32,
        load_base_model=False,
        backend={"experts": "torch"},
        config_overrides={"num_hidden_layers": 1},
    )
    # Pretraining always trains every parameter.
    assert all(param.requires_grad for param in pipe.transformer.parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"transformer_engine_linear": True}, "transformer_engine_linear"),
        ({"fuse_qkv_projections": True}, "fuse_qkv_projections"),
        ({"attention_backend": "flash"}, "attention_backend"),
        ({"mesh_context": SimpleNamespace(cp_size=2)}, "Context parallelism"),
    ],
)
def test_from_config_custom_rejects_unsupported_options(custom_repo, patched_custom_build, kwargs, match):
    with pytest.raises(ValueError, match=match):
        NeMoAutoDiffusionPipeline.from_config(custom_repo, pipeline_spec={"subfolder": "transformer"}, **kwargs)

    patched_custom_build.build.assert_not_called()


# =============================================================================
# Toy MoE DiT (test fixture) contract
# =============================================================================


def test_toy_moe_dit_matches_simple_adapter_contract_and_round_trips_hf_keys(tmp_path):
    from safetensors.torch import load_file

    from nemo_automodel.components.flow_matching.adapters.simple import SimpleAdapter
    from nemo_automodel.components.moe.layers import MoE

    register_toy_moe_dit()
    root = write_toy_moe_dit_checkpoint(str(tmp_path / "toy"), seed=7)
    config = ToyMoEDiTConfig.from_pretrained(os.path.join(root, "transformer"))
    model = ToyMoEDiTForDiffusion(config, backend=toy_backend())
    hf_state_dict = load_file(os.path.join(root, "transformer", "model.safetensors"))

    # Standard per-expert HF keys; grouped native experts after from_hf.
    assert "model.layers.0.mlp.experts.7.gate_proj.weight" in hf_state_dict
    assert "model.layers.1.mlp.experts.0.down_proj.weight" in hf_state_dict
    native = model.state_dict_adapter.from_hf(hf_state_dict)
    assert set(native) == set(model.state_dict())
    model.load_state_dict(native)
    assert all(isinstance(model.model.layers[str(i)].mlp, MoE) for i in range(config.num_hidden_layers))

    latents = torch.randn(2, config.in_channels, 2, 4, 4)
    adapter = SimpleAdapter()
    prediction = adapter.forward(
        model,
        {
            "hidden_states": latents,
            "timestep": torch.tensor([10.0, 900.0]),
            "encoder_hidden_states": torch.randn(2, 8, config.text_embed_dim),
            "attention_kwargs": {"scale": 1.0},
        },
    )
    assert prediction.shape == latents.shape
    prediction.float().pow(2).mean().backward()
    assert model.model.layers["0"].mlp.experts.gate_and_up_projs.grad is not None
