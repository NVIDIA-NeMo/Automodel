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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.addons import (
    ConsolidatedHFAddon,
    PeftAddon,
    _check_modules_to_save_cover_exported_tensors,
    _extract_modules_to_save,
    _extract_target_modules,
    _get_automodel_peft_metadata,
    _get_hf_peft_config,
    _group_barrier,
    _is_group_rank_0,
    _maybe_save_custom_model_code,
    _maybe_strip_quantization_config,
    load_automodel_peft_config_dict,
)
from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, validate_peft_trainable_modules
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing


def test_group_barrier_uses_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.barrier") as barrier,
    ):
        _group_barrier(group)
    barrier.assert_called_once_with(group=group)


def test_group_rank_zero_is_relative_to_model_process_group():
    group = object()
    with (
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.is_initialized", return_value=True),
        patch("nemo_automodel.components.checkpoint.addons.torch.distributed.get_rank", return_value=0) as get_rank,
    ):
        assert _is_group_rank_0(group)
    get_rank.assert_called_once_with(group=group)


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


@pytest.mark.parametrize("use_ddp", [False, True])
@pytest.mark.parametrize("v4_compatible", [False, True])
def test_consolidated_hf_addon_delegates_to_model_metadata_exporter(
    tmp_path: Path, use_ddp: bool, v4_compatible: bool
) -> None:
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    exporter = SimpleNamespace(validate=MagicMock(), save=MagicMock())
    model = nn.Module()
    model._get_consolidated_hf_metadata_exporter = lambda *, tokenizer, original_model_path: exporter
    wrapped_model = object.__new__(DistributedDataParallel)
    nn.Module.__init__(wrapped_model)
    wrapped_model.module = model
    tokenizer = MagicMock()

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[wrapped_model if use_ddp else model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping=None,
        original_model_path="/source",
        v4_compatible=v4_compatible,
    )

    exporter.validate.assert_called_once_with(tokenizer=tokenizer, original_model_path="/source")
    exporter.save.assert_called_once_with(
        hf_metadata_dir=str(metadata_dir),
        tokenizer=tokenizer,
        original_model_path="/source",
        v4_compatible=v4_compatible,
    )


def test_consolidated_hf_addon_validates_model_exporter_on_nonzero_rank(tmp_path):
    exporter = SimpleNamespace(
        validate=MagicMock(side_effect=ValueError("invalid export")),
        save=MagicMock(),
    )
    model = nn.Module()
    model._get_consolidated_hf_metadata_exporter = lambda *, tokenizer, original_model_path: exporter

    with (
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.barrier") as barrier,
        pytest.raises(ValueError, match="invalid export"),
    ):
        ConsolidatedHFAddon().pre_save(
            model_state=SimpleNamespace(model=[model]),
            hf_metadata_dir=str(tmp_path),
            tokenizer=None,
            fqn_to_file_index_mapping={"w": 1},
            fqn_to_dtype_mapping=None,
            original_model_path=None,
            v4_compatible=False,
        )

    exporter.save.assert_not_called()
    barrier.assert_not_called()


def test_consolidated_hf_addon_uses_standard_metadata_when_exporter_declines_without_tokenizer(tmp_path):
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    model = nn.Module()
    model.config = {"model_type": "dummy"}
    get_metadata_exporter = MagicMock(return_value=None)
    model._get_consolidated_hf_metadata_exporter = get_metadata_exporter

    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(metadata_dir),
        tokenizer=None,
        fqn_to_file_index_mapping={"w": 1},
        fqn_to_dtype_mapping=None,
        original_model_path=None,
        v4_compatible=False,
    )

    get_metadata_exporter.assert_called_once_with(tokenizer=None, original_model_path=None)
    assert (metadata_dir / "config.json").is_file()


def test_model_state_keeps_lm_head_when_storage_not_shared():
    """Config-tied model with a separate lm_head keeps lm_head.weight on save.

    The resolver reports the top-level config intent (so ``uses_tied_lm_head`` is
    True here), but ModelState gates lm_head dropping on the storage-based
    ``has_local_tied_lm_head``. With a separate lm_head and no shared embedding,
    the save path must keep lm_head.weight. This is the safety that previously came
    from a force-untied exclusion list, now provided by the storage check.
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

    assert state.uses_tied_lm_head is True  # follows the top-level config flag
    assert state.has_local_tied_lm_head is False  # but the tensors do not actually share storage

    state_dict = state.state_dict()
    assert "lm_head.weight" in state_dict  # so the head is kept (storage-gated safety)


def test_model_state_drops_lm_head_when_storage_shared():
    """Config-tied model whose lm_head shares storage with the embedding drops lm_head.weight on save."""

    class _DummyConfig:
        tie_word_embeddings = True

    class _DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DummyConfig()
            self.model = torch.nn.Module()
            self.model.embed_tokens = torch.nn.Embedding(4, 2)
            self.lm_head = torch.nn.Linear(2, 4, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight  # genuine tie

    model = _DummyModel()
    state = ModelState([model])

    assert state.uses_tied_lm_head is True
    assert state.has_local_tied_lm_head is True

    state_dict = state.state_dict()
    assert "lm_head.weight" not in state_dict  # dropped because storage is shared
    assert "model.embed_tokens.weight" in state_dict


def test_peft_model_state_can_skip_default_group_broadcast():
    """Subset-mesh ranks already load PEFT state and must not broadcast globally."""

    class _DummyConfig:
        tie_word_embeddings = False

    model = nn.Linear(2, 2)
    model.config = _DummyConfig()
    state = ModelState(model, is_peft=True)

    with patch("nemo_automodel.components.checkpoint.stateful_wrappers.set_model_state_dict") as set_state:
        state.load_state_dict({}, strict=False, broadcast_from_rank0=False)

    options = set_state.call_args.kwargs["options"]
    assert options.full_state_dict is True
    assert options.broadcast_from_rank0 is False


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


class _StubStateDictAdapter(StateDictAdapter):
    """Minimal concrete StateDictAdapter, no renames of its own."""

    def to_hf(self, state_dict, **kwargs):
        """Identity pass-through; tensor values keep their native layouts."""
        return state_dict

    def from_hf(self, hf_state_dict, device_mesh=None, **kwargs):
        """Identity pass-through; tensor values keep their HF layouts."""
        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn, tensor, **kwargs):
        """Identity pass-through; ``tensor`` keeps its layout, whatever it is."""
        return [(fqn, tensor)]


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

    def test_adapter_target_module_renames_applied(self):
        """An adapter overriding map_peft_target_module_to_hf rewrites the entries.

        Adapters that rename modules between the native and checkpoint layouts
        (e.g. Kimi K3's expert projections) convert the saved keys, so the
        target_modules metadata has to follow or PEFT can't resolve it.
        """

        class _RenamingAdapter(_StubStateDictAdapter):
            def map_peft_target_module_to_hf(self, name: str, *, v4_compatible: bool = False) -> str:
                return name.replace(".mlp.experts.", ".block_sparse_moe.experts.")

        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.experts.0.gate_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_A",
            ]
        )
        model.state_dict_adapter = _RenamingAdapter()
        result = _extract_target_modules(model)
        assert "model.layers.0.block_sparse_moe.experts.0.gate_proj" in result
        assert "model.layers.0.self_attn.q_proj" in result
        assert all(".mlp.experts." not in m for m in result)

    def test_adapter_without_override_keeps_target_modules_unchanged(self):
        """Adapters that don't rename modules inherit the identity default."""
        adapter = _StubStateDictAdapter()
        assert adapter.map_peft_target_module_to_hf("model.layers.0.self_attn.q_proj") == (
            "model.layers.0.self_attn.q_proj"
        )

        model = _make_model_with_named_modules(
            [
                "model.layers.0.mlp.experts.0.gate_proj.lora_A",
                "model.layers.0.self_attn.q_proj.lora_A",
            ]
        )
        model.state_dict_adapter = adapter
        result = _extract_target_modules(model)
        assert "model.layers.0.mlp.experts.0.gate_proj" in result
        assert "model.layers.0.self_attn.q_proj" in result

    def test_duck_typed_adapter_without_the_method_is_tolerated(self):
        """Adapters that don't subclass StateDictAdapter must not crash the save.

        llama, qwen2/3, and kimivl attach plain classes as state_dict_adapter,
        and some tests attach mixin-only adapters; they have no
        map_peft_target_module_to_hf and just keep their names.
        """

        class _BareAdapter:
            pass

        model = _make_model_with_named_modules(["model.layers.0.self_attn.q_proj.lora_A"])
        model.state_dict_adapter = _BareAdapter()
        result = _extract_target_modules(model)
        assert result == ["model.layers.0.self_attn.q_proj"]


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


class TestParamWrapperLayoutStamp:
    """The PEFT save stamps the fused expert LoRA layout into
    automodel_peft_config.json (peft flipped ParamWrapper in 0.19.1,
    huggingface/peft#3165) so loads resolve it from metadata, not shapes."""

    def test_v5_adapter_gets_the_modern_stamp_by_default(self):
        from nemo_automodel.components.checkpoint.addons import _get_paramwrapper_layout_stamp

        adapter = SimpleNamespace(_v5_peft_target_parameters=("mlp.experts.gate_up_proj",))
        assert _get_paramwrapper_layout_stamp(adapter, False, False) == "peft-0.19.1"
        assert _get_paramwrapper_layout_stamp(adapter, False, True) == "peft-0.18"

    def test_non_v5_or_v4_saves_get_no_stamp(self):
        from nemo_automodel.components.checkpoint.addons import _get_paramwrapper_layout_stamp

        v5_adapter = SimpleNamespace(_v5_peft_target_parameters=("mlp.experts.gate_up_proj",))
        plain_adapter = SimpleNamespace(_v5_peft_target_parameters=())
        assert _get_paramwrapper_layout_stamp(plain_adapter, False, False) is None
        assert _get_paramwrapper_layout_stamp(None, False, False) is None
        assert _get_paramwrapper_layout_stamp(v5_adapter, True, False) is None


class TestParamWrapperLayoutHintPlumbing:
    def test_metadata_reader_round_trips(self, tmp_path):
        import json as json_module

        from nemo_automodel.components.checkpoint.checkpointing import _read_paramwrapper_layout_metadata

        assert _read_paramwrapper_layout_metadata(tmp_path) is None
        (tmp_path / "automodel_peft_config.json").write_text(json_module.dumps({"paramwrapper_layout": "peft-0.18"}))
        assert _read_paramwrapper_layout_metadata(tmp_path) == "peft-0.18"

    def test_hint_is_set_during_from_hf_and_cleared_after(self):
        from nemo_automodel.components.checkpoint.checkpointing import _maybe_adapt_state_dict_from_hf

        seen = {}

        class _Adapter:
            def from_hf(self, state_dict, device_mesh=None, **kwargs):
                seen["hint"] = self._paramwrapper_layout_hint
                return state_dict

        model = nn.Module()
        model.state_dict_adapter = _Adapter()
        _maybe_adapt_state_dict_from_hf(model, {}, paramwrapper_layout_hint="peft-0.18")

        assert seen["hint"] == "peft-0.18"
        assert model.state_dict_adapter._paramwrapper_layout_hint is None


@pytest.mark.parametrize("legacy_defaults", [False, True])
def test_load_automodel_peft_config_dict_round_trips_the_saved_metadata(tmp_path, legacy_defaults: bool):
    import json

    from nemo_automodel.components._peft.lora import PeftConfig

    peft = PeftConfig(
        target_modules=["*.qkv_proj"], dim=16, alpha=8, lora_dtype=torch.bfloat16, use_memory_efficient_lora=False
    )
    metadata = _get_automodel_peft_metadata(peft)
    assert "dim" not in metadata and "alpha" not in metadata
    assert metadata["use_memory_efficient_lora"] is False
    metadata.update({"paramwrapper_layout": "peft-0.19.1+", "future_checkpoint_field": True})
    if legacy_defaults:
        metadata.pop("use_memory_efficient_lora")
    (tmp_path / "automodel_peft_config.json").write_text(json.dumps(metadata))
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 16, "lora_alpha": 8}))
    kwargs = load_automodel_peft_config_dict(str(tmp_path))
    assert "paramwrapper_layout" not in kwargs and "future_checkpoint_field" not in kwargs
    loaded = PeftConfig(**kwargs)
    assert loaded.target_modules == ["*.qkv_proj"] and (loaded.dim, loaded.alpha) == (16, 8)
    assert loaded.lora_dtype == "torch.bfloat16"
    assert loaded.use_memory_efficient_lora is legacy_defaults


# ----------------------------------------------- PEFT with modules trained in full


@pytest.fixture
def single_rank_process_group(tmp_path):
    """DCP's full-state-dict collection, used by the PEFT save path, needs a process group."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


class _TinyEncoderClassifier(nn.Module):
    """LoRA targets in the encoder, a two-layer classification head, and buffers.

    The head owns a persistent and a non-persistent buffer; the root owns a persistent one.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.query = nn.Linear(4, 4)
        self.encoder.value = nn.Linear(4, 4)
        self.encoder.norm = nn.LayerNorm(4)
        self.classifier = nn.Module()
        self.classifier.dense = nn.Linear(4, 4)
        self.classifier.out_proj = nn.Linear(4, 2)
        self.classifier.register_buffer("label_bias", torch.full((2,), 0.5))
        self.classifier.register_buffer("scratch", torch.zeros(2), persistent=False)
        self.register_buffer("position_ids", torch.arange(4))


_LORA_KEYS = {
    f"base_model.model.encoder.{name}.lora_{side}.weight" for name in ("query", "value") for side in ("A", "B")
}
_PEFT = PeftConfig(target_modules=["*.query", "*.value"], dim=2, alpha=4, use_triton=False)


def _peft_model(unfreeze_modules: list[str]) -> _TinyEncoderClassifier:
    """The tiny model with LoRA on the encoder and ``unfreeze_modules`` trained in full."""
    model = _TinyEncoderClassifier()
    assert apply_lora_to_linear_modules(model, _PEFT) == 2
    if unfreeze_modules:
        apply_parameter_freezing(model, {"unfreeze_modules": [{"path": path} for path in unfreeze_modules]})
    return model


_FULL_MODULE_KEYS = {
    f"base_model.model.{name}"
    for name in (
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
        "classifier.label_bias",
        "encoder.norm.weight",
        "encoder.norm.bias",
    )
}


@pytest.mark.parametrize("local_collection", [False, True], ids=["dcp", "local"])
def test_peft_model_state_keeps_fully_trained_modules_with_their_persistent_buffers(
    single_rank_process_group, local_collection: bool
):
    """A PEFT checkpoint holds every trainable parameter and the full modules' persistent buffers.

    HF PEFT restores a ``modules_to_save`` module from all of its state-dict entries, so the
    head's persistent buffer goes with it. Non-persistent buffers and buffers of other modules
    stay out. The local collection serves pipeline-parallel, expert-parallel and quantized models.
    """
    model = _peft_model(["classifier", "encoder.norm"])
    pp_group = torch.distributed.group.WORLD if local_collection else None

    state = ModelState(model, is_peft=True, pp_group=pp_group).state_dict()

    assert set(state) == _LORA_KEYS | _FULL_MODULE_KEYS
    expected = model.state_dict()
    for key in _FULL_MODULE_KEYS:
        torch.testing.assert_close(state[key], expected[key.removeprefix("base_model.model.")], rtol=0, atol=0)


def _wrap_ddp(model: nn.Module) -> nn.Module:
    return DistributedDataParallel(model)


def _wrap_head_for_activation_checkpointing(model: nn.Module) -> nn.Module:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    model.classifier = checkpoint_wrapper(model.classifier)
    return model


def _compile_head(model: nn.Module) -> nn.Module:
    model.classifier = torch.compile(model.classifier, backend="eager")
    return model


@pytest.mark.parametrize(
    ("wrap", "local_collection"),
    [
        pytest.param(_wrap_ddp, False, id="ddp"),
        pytest.param(_wrap_head_for_activation_checkpointing, False, id="activation-checkpoint-dcp"),
        pytest.param(_wrap_head_for_activation_checkpointing, True, id="activation-checkpoint-local"),
        pytest.param(_compile_head, False, id="torch-compile-dcp"),
        pytest.param(_compile_head, True, id="torch-compile-local"),
    ],
)
def test_peft_export_names_ignore_training_wrappers(single_rank_process_group, wrap, local_collection: bool):
    """Wrappers training adds around modules do not change the exported names or modules_to_save.

    A trainable parameter whose name does not match its state-dict key fails the save, so the
    names have to follow DCP's normalization through DDP, activation checkpointing and torch.compile.
    """
    model = wrap(_peft_model(["classifier", "encoder.norm"]))
    pp_group = torch.distributed.group.WORLD if local_collection else None

    state = ModelState(model, is_peft=True, pp_group=pp_group).state_dict()
    config = _get_hf_peft_config(_PEFT, ModelState(model, is_peft=True, pp_group=pp_group))

    assert set(state) == _LORA_KEYS | _FULL_MODULE_KEYS
    assert config["modules_to_save"] == ["classifier", "encoder.norm"]


def test_lora_only_peft_model_state_is_unchanged(single_rank_process_group):
    """Without modules trained in full, the adapter keeps exactly the LoRA weights."""
    assert set(ModelState(_peft_model([]), is_peft=True).state_dict()) == _LORA_KEYS


def test_peft_model_state_refuses_to_drop_a_trainable_parameter(single_rank_process_group):
    """A trainable parameter missing from the DCP state dict fails the save instead of vanishing."""
    model = _peft_model(["classifier"])
    from torch.distributed.checkpoint.state_dict import get_model_state_dict

    def renamed_state_dict(module, options=None):
        # Stands in for a wrapper whose prefix DCP keeps, so the names no longer match.
        state = get_model_state_dict(module, options=options)
        return {f"wrapper.{key}" if key.startswith("classifier.") else key: value for key, value in state.items()}

    with patch(
        "nemo_automodel.components.checkpoint.stateful_wrappers.get_model_state_dict", side_effect=renamed_state_dict
    ):
        with pytest.raises(RuntimeError, match="would drop trainable parameters"):
            ModelState(model, is_peft=True).state_dict()


@pytest.mark.parametrize(
    "wrap", [pytest.param(lambda model: model, id="plain"), pytest.param(_compile_head, id="torch-compile")]
)
def test_peft_load_without_dcp_restores_the_buffers_of_modules_stored_whole(single_rank_process_group, wrap):
    """The expert-parallel save and load paths, which bypass DCP, restore the tensors the checkpoint stores.

    A module trained in full comes back with its persistent buffers, for example an MoE router's
    score-correction bias, as it does on the DCP path. Buffers outside those modules are left alone.
    Both paths name tensors as DCP does, so a compiled module round-trips too.
    """
    trained = wrap(_peft_model(["classifier", "encoder.norm"]))
    with torch.no_grad():
        trained.classifier.label_bias.add_(1.0)
        for parameter in trained.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter))
    state = ModelState(trained, is_peft=True, pp_group=torch.distributed.group.WORLD).state_dict()
    assert set(state) == _LORA_KEYS | _FULL_MODULE_KEYS
    state["base_model.model.position_ids"] = torch.full((4,), 7)
    resumed = wrap(_peft_model(["classifier", "encoder.norm"]))

    ModelState(resumed, is_peft=True, has_expert_parallelism=True).load_state_dict(state)

    torch.testing.assert_close(resumed.classifier.label_bias, trained.classifier.label_bias, rtol=0, atol=0)
    resumed_parameters = dict(resumed.named_parameters())
    for name, parameter in trained.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(resumed_parameters[name], parameter, rtol=0, atol=0)
    torch.testing.assert_close(resumed.position_ids, torch.arange(4), rtol=0, atol=0)


def test_hf_peft_config_lists_the_largest_fully_trained_modules():
    """modules_to_save names whole trained modules, which is what HF PEFT wraps and restores.

    Listing ``classifier.dense`` and ``classifier.out_proj`` instead of ``classifier`` breaks
    loading a sequence-classification adapter, because PEFT always wraps ``classifier`` too.
    """
    model = _peft_model(["classifier", "encoder.norm"])

    config = _get_hf_peft_config(_PEFT, ModelState(model, is_peft=True))

    assert config["modules_to_save"] == ["classifier", "encoder.norm"]
    assert config["target_modules"] == ["encoder.query", "encoder.value"]


def test_lora_only_hf_peft_config_has_no_modules_to_save():
    """Adapters without modules trained in full keep the previous adapter_config.json."""
    assert "modules_to_save" not in _get_hf_peft_config(_PEFT, ModelState(_peft_model([]), is_peft=True))


@pytest.mark.cpu_only
@pytest.mark.parametrize("mask_ids", [None, [63, 61, 62]])
def test_peft_export_round_trips_invocation_tokens(tmp_path, mask_ids):
    """Both PEFT and AutoModel recover the ordered gate IDs from an actual export."""
    import json

    from peft import PeftConfig as HFPeftConfig

    config = replace(_PEFT, alora_invocation_tokens=mask_ids)
    PeftAddon().pre_save(
        model_path=str(tmp_path),
        model_state=ModelState(_peft_model([]), is_peft=True),
        peft_config=config,
        original_model_path=None,
    )

    loaded = HFPeftConfig.from_pretrained(tmp_path)
    assert loaded.alora_invocation_tokens == mask_ids
    restored = PeftConfig(**load_automodel_peft_config_dict(str(tmp_path)))
    assert restored.alora_invocation_tokens == mask_ids
    exported = json.loads((tmp_path / "adapter_config.json").read_text())
    assert ("alora_invocation_tokens" in exported) == (mask_ids is not None)


def test_modules_to_save_split_combined_projections_and_follow_adapter_renames():
    """modules_to_save uses the same HF names as target_modules and the exported tensors."""

    class _RenamingAdapter(_StubStateDictAdapter):
        def map_peft_target_module_to_hf(self, name, *, v4_compatible=False):
            return f"language_model.{name}"

    model = nn.Module()
    model.layers = nn.ModuleList([nn.Module()])
    model.layers[0].qkv_proj = nn.Linear(4, 12)
    model.layers[0].o_proj = nn.Linear(4, 4)
    model.layers[0].o_proj.requires_grad_(False)
    model.state_dict_adapter = _RenamingAdapter()

    assert _extract_modules_to_save([model]) == [
        "language_model.layers.0.k_proj",
        "language_model.layers.0.q_proj",
        "language_model.layers.0.v_proj",
    ]


def _model_with_blocks_in(container: type[nn.Module]) -> nn.Module:
    """LoRA on the head only, and a backbone whose two blocks sit in ``container`` next to a norm."""
    model = nn.Module()
    model.backbone = nn.Module()
    blocks = [nn.LayerNorm(4), nn.LayerNorm(4)]
    model.backbone.layers = (
        nn.ModuleDict({str(index): block for index, block in enumerate(blocks)})
        if container is nn.ModuleDict
        else nn.ModuleList(blocks)
    )
    model.backbone.norm = nn.LayerNorm(4)
    model.head = nn.Linear(4, 2)
    assert apply_lora_to_linear_modules(model, PeftConfig(target_modules=["head"], dim=2, alpha=4)) == 1
    return model


@pytest.mark.parametrize("container", [nn.ModuleList, nn.ModuleDict], ids=["module-list", "module-dict"])
@pytest.mark.parametrize(
    ("unfreeze_modules", "modules_to_save"),
    [
        pytest.param(["backbone.layers"], ["backbone.layers.0", "backbone.layers.1"], id="whole-container"),
        pytest.param(["backbone.layers.1"], ["backbone.layers.1"], id="one-block"),
        pytest.param(["backbone"], ["backbone"], id="enclosing-module"),
    ],
)
def test_modules_to_save_never_names_a_module_container(container, unfreeze_modules, modules_to_save):
    """HF PEFT refuses to wrap a ModuleList or ModuleDict, so a trained one is listed through its blocks.

    A trained enclosing module is still listed whole, the same way a trained classifier is.
    """
    model = _model_with_blocks_in(container)
    apply_parameter_freezing(model, {"unfreeze_modules": [{"path": path} for path in unfreeze_modules]})

    validate_peft_trainable_modules(model)
    assert _extract_modules_to_save([model]) == modules_to_save


def _add_a_trained_parameter_list(model: nn.Module) -> None:
    model.encoder.scales = nn.ParameterList([nn.Parameter(torch.ones(4))])


def _tie_classifier_to_a_second_head(model: nn.Module) -> None:
    model.head = nn.Linear(4, 2)
    model.head.weight = model.classifier.out_proj.weight


def _add_root_parameter(model: nn.Module) -> None:
    model.register_parameter("logit_scale", nn.Parameter(torch.ones(1)))


def _unfreeze_only_the_norm_bias(model: nn.Module) -> None:
    apply_parameter_freezing(model, {"unfreeze_parameters": ["encoder.norm.bias"]})


def _freeze_the_head_output_bias(model: nn.Module) -> None:
    apply_parameter_freezing(model, {"freeze_parameters": ["classifier.out_proj.bias"]})


@pytest.mark.parametrize(
    ("unfreeze_modules", "modify", "message"),
    [
        pytest.param(["encoder.query"], None, "module also holds LoRA adapters", id="lora-module-base-weight"),
        pytest.param(["encoder"], None, "module also holds LoRA adapters", id="module-around-lora"),
        pytest.param(["classifier"], _tie_classifier_to_a_second_head, "registered under several names", id="tied"),
        pytest.param([], _add_root_parameter, "belong to the root module", id="root-parameter"),
        pytest.param([], _unfreeze_only_the_norm_bias, "also have frozen parameters", id="unfreeze-part-of-a-module"),
        pytest.param(
            ["classifier"], _freeze_the_head_output_bias, "also have frozen parameters", id="freeze-part-of-a-module"
        ),
        pytest.param([], _add_a_trained_parameter_list, "belong to a ParameterList", id="parameter-list"),
    ],
)
def test_peft_rejects_trainable_modules_hf_peft_cannot_restore(unfreeze_modules, modify, message):
    """Layouts the adapter checkpoint cannot restore fail at startup and on save, never silently."""
    model = _peft_model(unfreeze_modules)
    if modify is not None:
        modify(model)

    with pytest.raises(ValueError, match=message):
        validate_peft_trainable_modules(model)
    with pytest.raises(ValueError, match=message):
        _get_hf_peft_config(_PEFT, ModelState(model, is_peft=True))


def test_modules_to_save_must_cover_the_exported_full_weights():
    """The save path rejects an export whose modules_to_save and tensors name different modules."""
    exported = [
        "base_model.model.model.norm_f.weight",
        "base_model.model.model.layers.0.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.q_proj.lora_B.weight",
    ]
    _check_modules_to_save_cover_exported_tensors(["model.norm_f"], exported)

    with pytest.raises(ValueError, match="tensors outside modules_to_save: \\['model.norm_f.weight'\\]"):
        _check_modules_to_save_cover_exported_tensors(["model.norm"], exported)
    with pytest.raises(ValueError, match="entries without tensors: \\['lm_head'\\]"):
        _check_modules_to_save_cover_exported_tensors(["model.norm_f", "lm_head"], exported)


@pytest.mark.cpu_only
def test_mtp_mask_rows_round_trip_through_native_and_peft_loaders(tmp_path, single_rank_process_group):
    """The resize-first PEFT path and native resume restore trained mask rows and draft logits."""
    from copy import deepcopy

    from peft import PeftModel
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import AutoTokenizer, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

    from nemo_automodel.components._peft.lora import lora_token_gate
    from nemo_automodel.components.checkpoint.config import CheckpointingConfig

    base = Qwen3ForCausalLM(
        Qwen3Config(
            vocab_size=16,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            tie_word_embeddings=True,
        )
    ).eval()
    trained, resumed = deepcopy(base), deepcopy(base)
    ids = [12, 10]
    peft = PeftConfig(target_modules=["*q_proj"], dim=2, alpha=4, alora_invocation_tokens=ids)
    for model in (trained, resumed):
        apply_lora_to_linear_modules(model, peft)
    frozen_rows = base.get_input_embeddings().weight.detach().clone()
    with torch.no_grad():
        trained.get_input_embeddings().weight[ids] += 0.15
        for name, param in trained.named_parameters():
            if "lora_B" in name:
                param.fill_(0.125)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({f"token{i}": i for i in range(8)}, unk_token="token0")),
        unk_token="token0",
    )
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="safetensors",
        is_peft=True,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    checkpointer.save_model(trained, str(tmp_path / "saved"), peft_config=peft, tokenizer=tokenizer)
    path = tmp_path / "saved" / "model"
    checkpointer.load_model(resumed, str(path))
    assert len(tokenizer) == 8
    exported = AutoTokenizer.from_pretrained(path)
    assert len(exported) == 16
    assert exported.convert_tokens_to_ids(["<|automodel_mtp_mask_1|>", "<|automodel_mtp_mask_2|>"]) == ids

    base.resize_token_embeddings(len(exported))
    external = PeftModel.from_pretrained(base, path).eval()
    intact = [i for i in range(16) if i not in ids]
    inputs = torch.tensor([[1, 2, *ids]])
    gate = torch.tensor([[False, False, True, True]])
    with torch.inference_mode():
        with lora_token_gate(trained, gate):
            expected = trained(inputs).logits
        with lora_token_gate(resumed, gate):
            torch.testing.assert_close(resumed(inputs).logits, expected, atol=0, rtol=0)
        torch.testing.assert_close(external(inputs).logits, expected, atol=1e-6, rtol=1e-5)
        rows = trained.get_input_embeddings().weight[ids]
        torch.testing.assert_close(external.get_input_embeddings()(torch.tensor(ids)), rows, atol=0, rtol=0)
        torch.testing.assert_close(external.get_output_embeddings().weight[ids], rows, atol=0, rtol=0)
        torch.testing.assert_close(resumed.get_input_embeddings().weight[intact], frozen_rows[intact], atol=0, rtol=0)
    checkpointer.close()
