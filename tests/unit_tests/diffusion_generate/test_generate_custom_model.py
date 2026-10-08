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

"""CPU unit tests for generating with custom-model (non-diffusers) transformers."""

import json
import os
from types import SimpleNamespace

import pytest
import torch

import examples.diffusion.generate.generate as gen
from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.distributed.config import DistributedSetup
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hunyuan_image3.pipeline import HunyuanImage3Pipeline


class _Node(SimpleNamespace):
    """ConfigNode stand-in for the sections a custom-model load reads."""

    def to_dict(self):
        return dict(vars(self))


def _lora_dir(tmp_path):
    lora = tmp_path / "model"
    lora.mkdir()
    (lora / "automodel_peft_config.json").write_text(
        json.dumps({"target_modules": ["*.self_attn.qkv_proj"], "dropout": 0.0, "lora_dtype": "torch.bfloat16"})
    )
    (lora / "adapter_config.json").write_text(json.dumps({"r": 64, "lora_alpha": 32, "target_modules": ["x"]}))
    return str(lora)


def test_build_mesh_context_passes_expert_parallelism(monkeypatch):
    calls = []

    def build(cls, *args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(mesh_context="mesh")

    monkeypatch.setattr(DistributedSetup, "build", classmethod(build))
    assert gen._build_mesh_context(SimpleNamespace(ep_size=8), SimpleNamespace(world_size=8)) == "mesh"
    assert calls[0]["parallelism_sizes"].ep_size == 8 and calls[0]["world_size"] == 8
    gen._build_mesh_context(SimpleNamespace(), SimpleNamespace(world_size=8))
    assert calls[1]["parallelism_sizes"].ep_size == 1


def test_custom_model_backend():
    assert gen._custom_model_backend(SimpleNamespace(model=SimpleNamespace())) is None
    fields = _Node(attn="sdpa", experts="torch_mm", gate_precision="float32")
    backend = gen._custom_model_backend(SimpleNamespace(model=SimpleNamespace(backend=fields)))
    assert isinstance(backend, BackendConfig) and backend.experts == "torch_mm"
    sentinel = object()
    targeted = _Node(_target_="x.BackendConfig", instantiate=lambda: sentinel)
    targeted.to_dict = lambda: {"_target_": "x.BackendConfig"}
    assert gen._custom_model_backend(SimpleNamespace(model=SimpleNamespace(backend=targeted))) is sentinel


def test_custom_model_sampler_outputs_images():
    assert gen.detect_output_type(HunyuanImage3Pipeline.__new__(HunyuanImage3Pipeline)) == "image"


def _patch_load_pipeline(monkeypatch, custom):
    monkeypatch.setattr(
        "nemo_automodel._diffusers._hf_cache.resolve_diffusion_model_dir", lambda name: f"/local/{name}"
    )
    monkeypatch.setattr(
        "nemo_automodel._diffusers.auto_diffusion_pipeline.has_custom_transformer", lambda model_dir: custom
    )
    monkeypatch.setattr(gen, "patch_t5_layer_norm", lambda: None)
    return SimpleNamespace(
        model=SimpleNamespace(pretrained_model_name_or_path="org/model"), inference=SimpleNamespace()
    )


def test_load_pipeline_routes_custom_transformers(monkeypatch):
    cfg = _patch_load_pipeline(monkeypatch, custom=True)
    calls = []
    monkeypatch.setattr(gen, "_load_custom_model_pipeline", lambda *args: calls.append(args) or "pipe")
    assert gen.load_pipeline(cfg, None) == "pipe"
    assert calls == [(cfg, "/local/org/model", None, torch.bfloat16)]


def test_load_pipeline_loads_diffusers_checkpoint_and_lora(monkeypatch):
    cfg = _patch_load_pipeline(monkeypatch, custom=False)
    pipe, events = SimpleNamespace(), []
    monkeypatch.setattr(
        NeMoAutoDiffusionPipeline,
        "from_pretrained",
        classmethod(lambda cls, model_id, **kwargs: events.append(model_id) or pipe),
    )
    monkeypatch.setattr(gen, "load_checkpoint_into_pipeline", lambda p, c: events.append(("checkpoint", p, c)))
    monkeypatch.setattr(gen, "load_lora_weights_into_pipeline", lambda p, c: events.append(("lora", p, c)))
    assert gen.load_pipeline(cfg, None) is pipe
    assert events == ["/local/org/model", ("checkpoint", pipe, cfg), ("lora", pipe, cfg)]


class _Transformer:
    config = SimpleNamespace(architectures=["HunyuanImage3ForCausalMM"])


def _patch_custom_load(monkeypatch, transformer):
    built, loads = [], []

    def from_pretrained(cls, model_id, **kwargs):
        built.append((model_id, kwargs))
        return SimpleNamespace(transformer=transformer)

    def build(self, **kwargs):
        return SimpleNamespace(load_model=lambda model, path: loads.append((self.is_peft, model, path, kwargs)))

    monkeypatch.setattr(NeMoAutoDiffusionPipeline, "from_pretrained", classmethod(from_pretrained))
    monkeypatch.setattr(CheckpointingConfig, "build", build)
    monkeypatch.setattr(
        HunyuanImage3Pipeline,
        "from_transformer",
        classmethod(lambda cls, model, model_dir: ("sampler", model, model_dir)),
    )
    return built, loads


def test_load_custom_model_pipeline_loads_checkpoint_and_lora(tmp_path, monkeypatch):
    transformer = _Transformer()
    built, loads = _patch_custom_load(monkeypatch, transformer)
    lora = _lora_dir(tmp_path)
    cfg = SimpleNamespace(model=SimpleNamespace(checkpoint="/ckpt/step_10", lora_weights=lora))
    mesh_context = SimpleNamespace(moe_mesh="moe")

    pipe = gen._load_custom_model_pipeline(cfg, str(tmp_path), mesh_context, torch.bfloat16)

    assert pipe == ("sampler", transformer, str(tmp_path))
    ((model_id, kwargs),) = built
    assert model_id == str(tmp_path) and kwargs["mesh_context"] is mesh_context and kwargs["backend"] is None
    assert kwargs["components_to_load"] == ["transformer"]
    peft = kwargs["peft_cfg"]
    assert peft.target_modules == ["*.self_attn.qkv_proj"] and (peft.dim, peft.alpha) == (64, 32)
    assert [(is_peft, path) for is_peft, _, path, _ in loads] == [
        (False, os.path.join("/ckpt/step_10", "model")),
        (True, lora),
    ]
    assert all(model is transformer and build_kwargs["moe_mesh"] == "moe" for _, model, _, build_kwargs in loads)


def test_load_custom_model_pipeline_without_weights_or_known_sampler(monkeypatch):
    transformer = _Transformer()
    built, loads = _patch_custom_load(monkeypatch, transformer)
    cfg = SimpleNamespace(model=SimpleNamespace())
    gen._load_custom_model_pipeline(cfg, "ckpt", None, torch.bfloat16)
    assert built[0][1]["peft_cfg"] is None and loads == []

    transformer.config = SimpleNamespace(architectures=["SomethingElse"])
    with pytest.raises(ValueError, match="No sampling pipeline"):
        gen._load_custom_model_pipeline(cfg, "ckpt", None, torch.bfloat16)
