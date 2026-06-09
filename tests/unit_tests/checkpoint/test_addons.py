# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os

import torch
from torch import nn

from nemo_automodel.components.checkpoint.addons import (
    _extract_target_modules,
    _maybe_save_custom_model_code,
    _maybe_strip_quantization_config,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState


def _write(path: str, content: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_maybe_save_custom_model_code_copies_py_files_and_structure(tmp_path):
    # Arrange: create a nested source tree with .py and non-.py files
    src_root = tmp_path / "src_model_code"
    dst_root = tmp_path / "hf_meta"
    src_root.mkdir(parents=True)
    dst_root.mkdir(parents=True)

    files = {
        "main.py": "print('main')\n",
        "pkg/__init__.py": "# pkg init\n",
        "pkg/subpkg/module.py": "def foo():\n    return 1\n",
        "pkg/readme.txt": "do not copy\n",
    }
    for rel, content in files.items():
        _write(os.path.join(src_root, rel), content)

    # Act
    _maybe_save_custom_model_code(str(src_root), str(dst_root))

    # Assert: .py files copied with preserved structure; non-.py and __init__.py ignored
    assert (dst_root / "main.py").exists()
    assert not (dst_root / "pkg" / "__init__.py").exists()
    assert (dst_root / "pkg" / "subpkg" / "module.py").exists()
    assert not (dst_root / "pkg" / "readme.txt").exists()

    # Verify contents match
    with open(dst_root / "pkg" / "subpkg" / "module.py", "r") as f:
        assert "def foo()" in f.read()


def test_maybe_save_custom_model_code_noop_for_none_or_non_dir(tmp_path):
    dst_root = tmp_path / "hf_meta"
    dst_root.mkdir(parents=True)

    # None input should be a no-op
    _maybe_save_custom_model_code(None, str(dst_root))
    assert list(dst_root.rglob("*.py")) == []

    # Non-directory input should be a no-op
    some_file = tmp_path / "not_a_dir.txt"
    some_file.write_text("hello")
    _maybe_save_custom_model_code(str(some_file), str(dst_root))
    assert list(dst_root.rglob("*.py")) == []


def test_maybe_save_custom_model_code_copies_transitive_relative_imports(tmp_path):
    """A consolidated checkpoint must carry transitively-imported custom modules.

    Mirrors Nemotron-Flash: ``modeling_*.py`` imports ``.fused_mha_with_cache`` which
    imports ``.triton_attention``. ``triton_attention.py`` is a *transitive* (second-level)
    relative import, so copying only the modeling file and its direct imports leaves it out
    and reloading the consolidated dir would die with FileNotFoundError. The transitive
    closure pass must pull it in.
    """
    src_root = tmp_path / "src_model_code"
    dst_root = tmp_path / "hf_meta"
    src_root.mkdir(parents=True)
    dst_root.mkdir(parents=True)

    _write(str(src_root / "configuration_x.py"), "class XConfig:\n    pass\n")
    _write(
        str(src_root / "modeling_x.py"),
        "from .configuration_x import XConfig\nfrom .fused_mha import mha\n",
    )
    # Direct import of modeling that itself imports the transitive module.
    _write(str(src_root / "fused_mha.py"), "from .triton_attention import attn\n\ndef mha():\n    return attn()\n")
    # Transitive (second-level) import.
    _write(str(src_root / "triton_attention.py"), "def attn():\n    return 1\n")

    _maybe_save_custom_model_code(str(src_root), str(dst_root))

    assert (dst_root / "modeling_x.py").exists()
    assert (dst_root / "configuration_x.py").exists()
    assert (dst_root / "fused_mha.py").exists()
    # The key assertion: the transitively-imported module must be present.
    assert (dst_root / "triton_attention.py").exists()


def test_maybe_save_custom_model_code_fills_transitive_gap_from_snapshot(tmp_path, monkeypatch):
    """When the resolved custom-code (module-cache) dir is missing a transitive import,
    the model's HF snapshot dir must supply it.

    Reproduces the CI failure path: HF's dynamic-module cache holds only the modeling file
    plus its *direct* relative imports (no ``triton_attention.py``), but the full snapshot
    dir resolved from ``config.name_or_path`` carries every repo ``.py``. ``original_model_path``
    is None so only the model_part fallback runs.
    """
    import inspect as _inspect

    import nemo_automodel.components.checkpoint.addons as addons

    # Full HF snapshot dir (has everything, incl. the transitive triton_attention.py).
    snap = tmp_path / "hub" / "models--acme--Custom-1B" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    _write(str(snap / "configuration_x.py"), "class XConfig:\n    pass\n")
    _write(str(snap / "modeling_x.py"), "from .configuration_x import XConfig\nfrom .fused_mha import mha\n")
    _write(str(snap / "fused_mha.py"), "from .triton_attention import attn\n\ndef mha():\n    return attn()\n")
    _write(str(snap / "triton_attention.py"), "def attn():\n    return 1\n")

    # Partial module-cache dir: modeling + direct imports only (NO triton_attention.py).
    modcache = tmp_path / "modcache"
    modcache.mkdir()
    for name in ("modeling_x.py", "configuration_x.py", "fused_mha.py"):
        _write(str(modcache / name), (snap / name).read_text())
    _write(str(modcache / "__init__.py"), "")
    assert not (modcache / "triton_attention.py").exists()

    dst_root = tmp_path / "hf_meta"
    dst_root.mkdir()

    class _Cls:
        __module__ = "transformers_modules.acme.modeling_x"

    class _Config:
        name_or_path = "acme/Custom-1B"

    class _Model(nn.Module):
        name_or_path = "acme/Custom-1B"
        config = _Config()

    # Point the custom-code class resolution at the partial module-cache dir.
    monkeypatch.setattr(addons, "_iter_custom_code_classes", lambda mp: iter([_Cls]))
    orig_getfile = _inspect.getfile
    monkeypatch.setattr(
        _inspect,
        "getfile",
        lambda c: str(modcache / "modeling_x.py") if c is _Cls else orig_getfile(c),
    )
    # Resolve the snapshot via the fake hub cache root (read inside _resolve_hf_snapshot_dir).
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path / "hub"), raising=False)

    _maybe_save_custom_model_code(None, str(dst_root), model_part=_Model())

    # triton_attention.py was absent from the module cache but present in the snapshot;
    # the transitive-closure pass must have copied it from the snapshot.
    assert (dst_root / "triton_attention.py").exists()
    assert (dst_root / "modeling_x.py").exists()
    assert (dst_root / "fused_mha.py").exists()


def test_model_state_disables_tied_embeddings_for_non_tied_models():
    """
    Ensure ModelState explicitly disables tied embeddings for models listed in
    the non_tied_lm_head_models filter (e.g., Qwen3 Omni Moe Thinker).
    """

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    _DummyModel.__name__ = "Qwen3OmniMoeThinkerForConditionalGeneration"

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is False
    assert state.has_local_tied_lm_head is False
    assert not hasattr(state, "lm_head_param_name")

    state_dict = state.state_dict()
    assert "lm_head.weight" in state_dict


# _extract_target_modules tests
def _make_model_with_named_modules(module_names):
    """Build a dummy model whose ``named_modules`` yields the given names.

    We simulate LoRA sub-modules by adding ``nn.Identity`` leaves under
    the requested paths.  ``_extract_target_modules`` looks for any
    module whose name contains "lora", so we add leaves like
    ``<target>.lora_A``.
    """
    root = nn.Module()
    for name in module_names:
        parts = name.split(".")
        parent = root
        for part in parts[:-1]:
            if not hasattr(parent, part):
                setattr(parent, part, nn.Module())
            parent = getattr(parent, part)
        setattr(parent, parts[-1], nn.Identity())
    return root


class TestExtractTargetModules:
    """Tests for _extract_target_modules with combined-projection expansion."""

    def test_simple_non_combined_modules(self):
        """Non-combined module names pass through unchanged."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_B",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.o_proj" in result
        assert "model.layers.0.mlp.down_proj" in result

    def test_qkv_proj_expanded(self):
        """qkv_proj is expanded to q_proj, k_proj, v_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        # Combined name should NOT appear
        assert all("qkv_proj" not in m for m in result)

    def test_gate_up_proj_expanded(self):
        """gate_up_proj is expanded to gate_proj, up_proj."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_B",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.gate_proj" in result
        assert "model.layers.0.mlp.up_proj" in result
        assert all("gate_up_proj" not in m for m in result)

    def test_mixed_combined_and_regular(self):
        """Mixed combined and regular module names."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        expected = {
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.self_attn.k_proj",
            "model.layers.0.self_attn.v_proj",
            "model.layers.0.self_attn.o_proj",
            "model.layers.0.mlp.gate_proj",
            "model.layers.0.mlp.up_proj",
            "model.layers.0.mlp.down_proj",
        }
        assert set(result) == expected

    def test_torch_compile_prefix_stripped(self):
        """_orig_mod. prefix from torch.compile is stripped before expansion."""
        model = _make_model_with_named_modules(
            [
                "_orig_mod.model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        assert all("_orig_mod" not in m for m in result)

    def test_result_is_sorted(self):
        """Return value is sorted."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert result == sorted(result)

    def test_encoder_target_modules_remapped(self):
        """Encoder model.* target modules have model. prefix stripped."""
        from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.q_proj.lora_A",
                "model.layers.0.self_attn.k_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        model.state_dict_adapter = EncoderStateDictAdapter()
        result = _extract_target_modules(model)
        assert "layers.0.self_attn.q_proj" in result
        assert "layers.0.self_attn.k_proj" in result
        assert "layers.0.mlp.down_proj" in result
        assert all(not m.startswith("model.") for m in result)


class TestMaybeStripQuantizationConfig:
    """Tests for _maybe_strip_quantization_config."""

    @staticmethod
    def _make_config_with_quant():
        cfg = type("Config", (), {})()
        cfg.quantization_config = {"quant_method": "mxfp4"}
        return cfg

    def test_strips_quantization_config_when_all_params_bf16(self):
        """quantization_config is removed when all params are standard floating-point."""
        model = nn.Linear(4, 4, dtype=torch.bfloat16)
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_keeps_quantization_config_when_uint8_params_exist(self):
        """quantization_config is preserved when quantized (uint8) parameters exist."""
        model = nn.Module()
        model.register_parameter("weight", nn.Parameter(torch.ones(4, 4, dtype=torch.uint8), requires_grad=False))
        model.config = self._make_config_with_quant()

        _maybe_strip_quantization_config(model)
        assert hasattr(model.config, "quantization_config")

    def test_noop_when_no_quantization_config(self):
        """No error when config has no quantization_config attribute."""
        model = nn.Linear(4, 4)
        model.config = type("Config", (), {})()

        _maybe_strip_quantization_config(model)
        assert not hasattr(model.config, "quantization_config")

    def test_noop_when_no_config(self):
        """No error when model has no config attribute."""
        model = nn.Linear(4, 4)
        _maybe_strip_quantization_config(model)
