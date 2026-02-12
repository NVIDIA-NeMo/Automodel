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
from types import SimpleNamespace

import torch
from torch import nn

from nemo_automodel.components.checkpoint.addons import (
    _extract_target_modules,
    _maybe_save_custom_model_code,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


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

    # Assert: .py files copied with preserved structure; non-.py ignored
    assert (dst_root / "main.py").exists()
    assert (dst_root / "pkg" / "__init__.py").exists()
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

    assert state.is_tied_lm_head is False
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


def _llama_style_adapter():
    """Create a CombinedProjectionStateDictAdapter with default Llama-style mapping."""
    config = SimpleNamespace(
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_size=256,
    )
    return CombinedProjectionStateDictAdapter(config)


class TestExtractTargetModules:
    """Tests for _extract_target_modules with combined-projection expansion.

    Expansion only happens when the model has a ``state_dict_adapter`` with
    a non-identity ``fused_modules_mapping``.  Tests that verify expansion
    attach a Llama-style adapter to the dummy model.
    """

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
        """qkv_proj is expanded to q_proj, k_proj, v_proj when adapter is present."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.qkv_proj.lora_B",
            ]
        )
        model.state_dict_adapter = _llama_style_adapter()
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.q_proj" in result
        assert "model.layers.0.self_attn.k_proj" in result
        assert "model.layers.0.self_attn.v_proj" in result
        # Combined name should NOT appear
        assert all("qkv_proj" not in m for m in result)

    def test_gate_up_proj_expanded(self):
        """gate_up_proj is expanded to gate_proj, up_proj when adapter is present."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_B",
            ]
        )
        model.state_dict_adapter = _llama_style_adapter()
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.gate_proj" in result
        assert "model.layers.0.mlp.up_proj" in result
        assert all("gate_up_proj" not in m for m in result)

    def test_mixed_combined_and_regular(self):
        """Mixed combined and regular module names with adapter present."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.self_attn.o_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
                "model.layers.0.mlp.down_proj.lora_A",
            ]
        )
        model.state_dict_adapter = _llama_style_adapter()
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
        model.state_dict_adapter = _llama_style_adapter()
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
        model.state_dict_adapter = _llama_style_adapter()
        result = _extract_target_modules(model)
        assert result == sorted(result)

    def test_no_expansion_without_adapter(self):
        """Without a state_dict_adapter, fused names should NOT be expanded."""
        model = _make_model_with_named_modules(
            [
                "model.layers.0.self_attn.qkv_proj.lora_A",
                "model.layers.0.mlp.gate_up_proj.lora_A",
            ]
        )
        result = _extract_target_modules(model)
        assert "model.layers.0.self_attn.qkv_proj" in result
        assert "model.layers.0.mlp.gate_up_proj" in result
        assert all("q_proj" not in m for m in result)


