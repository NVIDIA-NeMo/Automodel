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

"""Real SafeTensors reads, exact decode oracles, mapping and storage-lifetime tests."""

import gc
import inspect
import json
import math
import weakref
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file

from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint._backports.hf_storage import _HuggingFaceStorageReader
from nemo_automodel.components.checkpoint.addons import _maybe_strip_quantization_config
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.inkling2.config import (
    Inkling2Config,
    Inkling2TextConfig,
    load_quantization_config,
)
from nemo_automodel.components.models.inkling2.model import Inkling2ForCausalLM
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.components.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def single_cpu_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint_config() -> Inkling2TextConfig:
    return Inkling2TextConfig(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        d_rel=3,
        rel_extent=4,
        sliding_window_size=7,
        local_layer_ids=[0],
        n_layers_per_attnres_block=1,
        n_routed_experts=4,
        num_experts_per_tok=2,
        intermediate_size=32,
        vocab_size=40,
        unpadded_vocab_size=37,
        dtype="float32",
        quantization_config={"quant_method": "inkling2", "nvfp4_layer_count": 1},
        architectures=["Inkling2ForCausalLM"],
    )


def decode_oracle(
    weight: torch.Tensor,
    scales: torch.Tensor,
    amax: torch.Tensor | None = None,
    denominator: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode [E,O,I] source data independently with scalar exponent/mantissa arithmetic."""
    fp4 = amax is not None
    width = weight.shape[-1] * (2 if fp4 else 1)
    result = torch.empty((*weight.shape[:-1], width), dtype=torch.float32)
    for e in range(weight.shape[0]):
        for row in range(weight.shape[1]):
            for col in range(width):
                if fp4:
                    byte = int(weight[e, row, col // 2])
                    code = (byte >> (4 * (col % 2))) & 15
                    exponent, mantissa = (code >> 1) & 3, code & 1
                    value = mantissa * 0.5 if exponent == 0 else math.ldexp(1 + mantissa * 0.5, exponent - 1)
                    if code & 8:
                        value = -value
                    # Explicit FP32 rounding at the same two serialized scale boundaries.
                    scaled = torch.tensor(value * float(scales[e, row, col // 16]), dtype=torch.float32)
                    outer = amax[e] / denominator[0]
                    result[e, row, col] = scaled * outer
                else:
                    result[e, row, col] = float(weight[e, row, col]) * math.ldexp(
                        1.0, int(scales[e, row, col // 32]) - 127
                    )
    return result


def mixed_fixture() -> tuple[Inkling2ForCausalLM, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    torch.manual_seed(905)
    model = Inkling2ForCausalLM(checkpoint_config())
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = {
        key: value.detach().contiguous().clone() for key, value in model.state_dict_adapter.to_hf(expected).items()
    }
    for layer in range(2):
        for kind, native in [("w13", "gate_and_up_projs"), ("w2", "down_projs")]:
            key = f"model.llm.layers.{layer}.mlp.experts.{kind}_weight"
            shape = checkpoint[key].shape
            if layer == 0:
                # All 16 E2M1 codes appear, with asymmetric adjacent columns.
                packed = (
                    torch.arange(math.prod(shape) // 2)
                    .remainder(256)
                    .to(torch.uint8)
                    .reshape(*shape[:-1], shape[-1] // 2)
                )
                scale = torch.randint(1, 9, (*shape[:-1], shape[-1] // 16)).to(torch.float8_e4m3fn)
                amax = torch.tensor([0.0, 0.125, 0.75, 2.0])
                denominator = torch.tensor([1536.0 if kind == "w13" else 2304.0])
                checkpoint.update(
                    {
                        key: packed,
                        key + "_scale": scale,
                        key + "_amax": amax,
                        key + "_outer_scale_denominator": denominator,
                    }
                )
                decoded = decode_oracle(packed, scale, amax, denominator)
            else:
                packed = (torch.randint(-12, 13, shape).float() / 4).to(torch.float8_e4m3fn)
                scale = torch.randint(121, 130, (*shape[:-1], shape[-1] // 32), dtype=torch.uint8)
                checkpoint.update({key: packed, key + "_scale": scale})
                decoded = decode_oracle(packed, scale)
            expected[f"layers.{layer}.mlp.experts.{native}"] = decoded.transpose(-1, -2)
    return model, checkpoint, expected


def write_checkpoint(checkpoint: dict[str, torch.Tensor], folder: Path) -> None:
    save_file({k: v.contiguous() for k, v in checkpoint.items()}, str(folder / "model.safetensors"))
    (folder / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {k: "model.safetensors" for k in checkpoint}})
    )


def test_mixed_decode_exact_and_source_not_mutated() -> None:
    model, checkpoint, expected = mixed_fixture()
    original_keys = set(checkpoint)
    actual = model.state_dict_adapter.from_hf(checkpoint)
    assert set(checkpoint) == original_keys
    assert set(actual) == set(expected)
    for key, value in actual.items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0, msg=key)
    assert checkpoint["model.llm.layers.0.mlp.experts.w13_weight"].dtype == torch.uint8


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_safetensors_load_targets_decode_into_existing_storage(tmp_path: Path, dtype: torch.dtype) -> None:
    """Packed expert staging from to_hf is read by DCP and decoded by from_hf without replacing storage."""
    model, checkpoint, expected = mixed_fixture()
    write_checkpoint(checkpoint, tmp_path)
    adapter = model.state_dict_adapter
    native = {k: torch.full_like(v, float("nan"), dtype=dtype) for k, v in expected.items()}
    pointers = {k: v.untyped_storage().data_ptr() for k, v in native.items()}
    expert_keys = {k for k in native if ".mlp.experts." in k}
    targets = adapter.to_hf(native, quantization=True, for_checkpoint_load=True)
    assert set(targets) == set(checkpoint)
    for key, value in targets.items():
        # Packed staging mirrors the source dtypes exactly; floating aliases keep the model dtype and DCP casts.
        assert value.shape == checkpoint[key].shape, key
        assert value.dtype == (checkpoint[key].dtype if ".mlp.experts." in key else dtype), key
    staged = [weakref.ref(v) for k, v in targets.items() if ".mlp.experts." in k]
    dcp.load(targets, storage_reader=_HuggingFaceStorageReader(str(tmp_path), key_mapping=None), no_dist=True)
    restored = adapter.from_hf(targets)
    assert adapter.view_loaded_native_keys == expert_keys
    assert set(restored) == set(native) - expert_keys
    for key, value in restored.items():
        # Local fused projections are re-concatenated from their six source pieces; everything else is a view.
        if ".attn.qkvr." in key:
            torch.testing.assert_close(value, expected[key].to(dtype), rtol=0, atol=0, msg=key)
        else:
            assert value.untyped_storage().data_ptr() == pointers[key], key
    for key, value in native.items():
        assert value.untyped_storage().data_ptr() == pointers[key]
        torch.testing.assert_close(value, expected[key].to(dtype), rtol=0, atol=0, msg=key)
    del targets, restored
    gc.collect()
    assert all(ref() is None for ref in staged), "packed staging remains live after from_hf"
    with pytest.raises(ValueError, match="checkpoint loading only"):
        adapter.to_hf(native, quantization=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_public_checkpointer_load_forward_backward_and_float_reload(tmp_path: Path, dtype: torch.dtype) -> None:
    model, checkpoint, expected = mixed_fixture()
    if dtype == torch.bfloat16:
        config = checkpoint_config()
        config.dtype = dtype
        model = Inkling2ForCausalLM(config)
    write_checkpoint(checkpoint, tmp_path)
    model.config.save_pretrained(tmp_path)
    checkpointer = CheckpointingConfig(
        is_async=False,
        model_save_format="safetensors",
        save_consolidated="final",
        model_repo_id=str(tmp_path),
        original_model_root_dir=str(tmp_path),
        model_cache_dir=str(tmp_path),
    ).build(0, 0, 0)
    checkpointer.load_base_model(model, torch.device("cpu"), str(tmp_path), str(tmp_path))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[key].to(value.dtype), rtol=0, atol=0, msg=key)
    ids = torch.tensor([[3, 8, 12, 20]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def step(target: Inkling2ForCausalLM, optim: torch.optim.Optimizer) -> None:
        optim.zero_grad(set_to_none=True)
        logits = target(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in target.parameters())
        optim.step()

    try:
        step(model, optimizer)
        saved = tmp_path / "saved"
        model.save_pretrained(str(saved), checkpointer=checkpointer, is_final_checkpoint=True)
        checkpointer.save_optimizer(optimizer, model, str(saved))
        floating = saved / "model/consolidated"
        cfg = json.loads((floating / "config.json").read_text())
        assert "quantization_config" not in cfg
        exported = model.state_dict_adapter.to_hf(model.state_dict())
        index = json.loads((floating / "model.safetensors.index.json").read_text())["weight_map"]
        assert set(index) == set(exported)
        for filename in set(index.values()):
            with safe_open(floating / filename, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    actual = handle.get_tensor(key)
                    assert actual.dtype == exported[key].dtype
                    torch.testing.assert_close(actual, exported[key], rtol=0, atol=0, msg=key)
        restored = Inkling2ForCausalLM(Inkling2TextConfig.from_pretrained(floating))
        checkpointer.load_base_model(restored, torch.device("cpu"), str(floating), str(floating))
        torch.testing.assert_close(restored(ids), model(ids), rtol=0, atol=0)
        assert restored.state_dict_adapter.supports_low_memory_dcp_load
        # Unsharded floating storage loads through aliases only: nothing is staged or completed in from_hf.
        aliases = restored.state_dict_adapter.to_hf(restored.state_dict(), for_checkpoint_load=True)
        assert restored.state_dict_adapter.view_loaded_native_keys == set()
        assert all(not key.endswith(("_scale", "_amax", "_outer_scale_denominator")) for key in aliases)
        resumed_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-4)
        checkpointer.load_optimizer(resumed_optimizer, restored, str(saved))
        step(model, optimizer)
        step(restored, resumed_optimizer)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0, msg=key)
    finally:
        checkpointer.close()


def test_floating_aliases_and_residual_query_indexing() -> None:
    from .test_reference import tiny_config

    model = Inkling2ForCausalLM(tiny_config())
    state = model.state_dict()
    adapter = model.state_dict_adapter
    sources = adapter.to_hf(state)
    for key, value in sources.items():
        assert value.untyped_storage().data_ptr() in {v.untyped_storage().data_ptr() for v in state.values()}, key
    # Serialized attention query zero belongs to decoder layer ONE, not layer zero.
    assert sources["model.llm.attn_res.attn_queries.0"].shape == (16,)
    assert sources["model.llm.attn_res.attn_queries.4"].shape == (3, 16)
    assert sources["model.llm.attn_res.attn_queries.10"].shape == (3, 16)
    sources["model.llm.layers.5.attn.wk_dv.weight"].fill_(13)
    assert torch.all(state["layers.5.attn.qkvr.weight"][32:48] == 13)
    sources["model.llm.layers.0.mlp.experts.w13_weight"][1, 0, 2] = 17
    assert state["layers.0.mlp.experts.gate_and_up_projs"][1, 2, 0] == 17
    restored = adapter.from_hf(sources)
    for key, value in state.items():
        torch.testing.assert_close(value, restored[key], rtol=0, atol=0, msg=key)


def test_load_initializes_native_only_projection_padding() -> None:
    settings = checkpoint_config().to_dict()
    settings.update(num_attention_heads=3, num_key_value_heads=1, swa_num_attention_heads=3, swa_num_key_value_heads=1)
    model = Inkling2ForCausalLM(Inkling2TextConfig(**settings))
    state = {key: torch.full_like(value, float("nan")) for key, value in model.state_dict().items()}
    adapter = model.state_dict_adapter
    sources = adapter.to_hf(state)
    assert torch.isnan(state["layers.0.attn.qkvr.weight"]).all(), "export must not alter model storage"
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    for key, value in state.items():
        if key.endswith("attn.qkvr.weight"):
            assert torch.isnan(value[:76]).all()
            assert torch.equal(value[76:], torch.zeros_like(value[76:]))
    assert set(sources) == set(destinations)


@pytest.mark.parametrize(
    "case", ["missing_scale", "orphan_scale", "nan_scale", "bad_denominator", "unknown", "incomplete_projection"]
)
def test_invalid_checkpoint_is_rejected(case: str) -> None:
    model, checkpoint, _ = mixed_fixture()
    key = "model.llm.layers.0.mlp.experts.w13_weight"
    if case == "missing_scale":
        del checkpoint[key + "_scale"]
    elif case == "orphan_scale":
        checkpoint[key] = torch.zeros(4, 64, 32)
    elif case == "nan_scale":
        checkpoint["model.llm.layers.1.mlp.experts.w2_weight_scale"].fill_(255)
    elif case == "bad_denominator":
        checkpoint[key + "_outer_scale_denominator"].zero_()
    elif case == "unknown":
        checkpoint["model.llm.nonexistent.weight"] = torch.zeros(1)
    else:
        del checkpoint["model.llm.layers.0.attn.wq_du.weight"]
    with pytest.raises(ValueError):
        model.state_dict_adapter.from_hf(checkpoint)


def write_released_checkpoint_metadata(folder: Path, nvfp4_layers: list[int], mxfp8_layers: list[int]) -> None:
    """Write the released config.json and hf_quant_config.json pair for a two-layer text model."""
    text = {key: value for key, value in checkpoint_config().to_dict().items() if key != "quantization_config"}
    config = {
        "architectures": ["Inkling2ForConditionalGeneration"],
        "model_type": "inkling2_mm_model",
        "text_config": text,
        "vision_config": {},
    }
    groups = [
        {
            "compute_dtype": dtype,
            "tensors": [f"model.llm.layers.{i}.mlp.experts.{p}" for i in layers for p in ("w13_weight", "w2_weight")],
        }
        for dtype, layers in (("nvfp4", nvfp4_layers), ("mxfp8", mxfp8_layers))
    ]
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "hf_quant_config.json").write_text(
        json.dumps({"quant_method": "inkling2", "model": {"parameter_groups": groups}})
    )


@pytest.mark.parametrize("nvfp4_layer_count", [0, 1])
def test_models_read_the_expert_layout_from_the_checkpoint_sidecar(tmp_path: Path, nvfp4_layer_count: int) -> None:
    write_released_checkpoint_metadata(
        tmp_path, nvfp4_layers=list(range(nvfp4_layer_count)), mxfp8_layers=list(range(nvfp4_layer_count, 2))
    )
    # The registry loader hands the checkpoint path to from_dict; config.json itself has no declaration.
    config = Inkling2Config.from_dict(json.loads((tmp_path / "config.json").read_text()), name_or_path=str(tmp_path))
    assert config.quantization_config is None and config.text_config.quantization_config is None
    declaration = {"quant_method": "inkling2", "nvfp4_layer_count": nvfp4_layer_count}
    text_model = Inkling2ForCausalLM.from_config(config)
    assert not hasattr(text_model, "visual")
    assert text_model.config is config.text_config and text_model.config.quantization_config == declaration
    assert text_model.config.architectures == ["Inkling2ForCausalLM"] and text_model.name_or_path == str(tmp_path)
    assert (
        text_model.state_dict_adapter.source_quantized
        and text_model.state_dict_adapter.nvfp4_layer_count == nvfp4_layer_count
    )
    config = Inkling2Config.from_dict(json.loads((tmp_path / "config.json").read_text()), name_or_path=str(tmp_path))
    image_model = Inkling2ForConditionalGeneration.from_config(config)
    assert image_model.config.quantization_config == declaration
    assert (
        image_model.state_dict_adapter.multimodal
        and image_model.state_dict_adapter.nvfp4_layer_count == nvfp4_layer_count
    )
    # Floating checkpoints, such as the ones training saves, ship no sidecar.
    (tmp_path / "hf_quant_config.json").unlink()
    assert load_quantization_config(str(tmp_path), 2) is None
    config = Inkling2Config.from_dict(json.loads((tmp_path / "config.json").read_text()), name_or_path=str(tmp_path))
    assert not Inkling2ForCausalLM.from_config(config).state_dict_adapter.source_quantized


@pytest.mark.parametrize("case", ["incomplete_projection", "gap", "unknown_dtype"])
def test_incomplete_expert_precision_declaration_is_rejected(tmp_path: Path, case: str) -> None:
    write_released_checkpoint_metadata(tmp_path, nvfp4_layers=[0], mxfp8_layers=[1])
    path = tmp_path / "hf_quant_config.json"
    sidecar = json.loads(path.read_text())
    groups = sidecar["model"]["parameter_groups"]
    if case == "incomplete_projection":
        groups[0]["tensors"].pop()
    elif case == "gap":
        groups[1]["tensors"].clear()
    else:
        groups[1]["compute_dtype"] = "fp8"
    path.write_text(json.dumps(sidecar))
    with pytest.raises(ValueError, match="both projections" if case != "unknown_dtype" else "Unsupported"):
        load_quantization_config(str(tmp_path), 2)


def test_quantization_config_boundary() -> None:
    cfg = checkpoint_config()
    with pytest.raises(ValueError, match="quant_method"):
        Inkling2TextConfig(quantization_config={"quant_method": "mxfp8", "nvfp4_layer_count": 1})
    with pytest.raises(ValueError, match="nvfp4_layer_count"):
        Inkling2TextConfig(quantization_config={"quant_method": "inkling2", "nvfp4_layer_count": 49})
    with pytest.raises(ValueError, match="metadata fields"):
        Inkling2TextConfig(quantization_config={"quant_method": "inkling2"})
    adapter = Inkling2StateDictAdapter(cfg)
    with pytest.raises(ValueError, match="checkpoint loading only"):
        adapter.to_hf({}, quantization=True)
    with pytest.raises(TypeError):
        adapter.to_hf({}, unsupported=True)
    with pytest.raises(ValueError, match="legacy"):
        adapter.to_hf({}, v4_compatible=True)
    assert adapter.forced_hf_dtype_mapping(
        {"pseudo_query": torch.zeros(3), "expert": torch.zeros(3, dtype=torch.bfloat16)}
    ) == {"pseudo_query": "F32", "expert": "BF16"}


def released_image_text_config(folder: Path, name_or_path: str) -> Inkling2Config:
    """Load the written released metadata with a small vision tower."""
    raw = json.loads((folder / "config.json").read_text())
    raw["vision_config"] = {
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "mlp_ratio": 2.0,
        "posemb_grid_size": 4,
        "projection_hidden_dim": 24,
    }
    return Inkling2Config.from_dict(raw, name_or_path=name_or_path)


def test_constructor_accepts_the_released_configuration_like_the_factory(tmp_path: Path) -> None:
    """The shared factory constructs ``architectures[0]`` with the released outer config directly."""
    write_released_checkpoint_metadata(tmp_path, nvfp4_layers=[0], mxfp8_layers=[1])
    declaration = {"quant_method": "inkling2", "nvfp4_layer_count": 1}
    config = released_image_text_config(tmp_path, str(tmp_path))
    config.architectures = ["Inkling2ForCausalLM"]
    model_cls = ModelRegistry.resolve_custom_model_cls(config.architectures[0], config)
    assert model_cls is Inkling2ForCausalLM
    model = model_cls(config)
    assert not hasattr(model, "visual")
    assert model.config is config.text_config and model.vocab_size == config.text_config.vocab_size
    assert model.config.architectures == ["Inkling2ForCausalLM"] and model.name_or_path == str(tmp_path)
    assert model.config.quantization_config == declaration
    assert model.state_dict_adapter.source_quantized and model.state_dict_adapter.nvfp4_layer_count == 1
    ids = torch.tensor([[3, 8, 12, 20]])
    logits = model(ids)
    loss = torch.nn.functional.cross_entropy(logits[:, :-1].flatten(0, 1), ids[:, 1:].flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # Text configurations and the image/text class keep their contracts.
    assert isinstance(Inkling2ForCausalLM(checkpoint_config()).config, Inkling2TextConfig)
    config = released_image_text_config(tmp_path, str(tmp_path))
    image_model = Inkling2ForConditionalGeneration(config)
    assert image_model.config is config and image_model.visual is not None
    assert image_model.config.quantization_config == image_model.text_config.quantization_config == declaration


def test_quantization_sidecar_follows_the_resolved_revision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A Hub checkpoint's sidecar is read from the snapshot the weights come from."""
    import huggingface_hub

    write_released_checkpoint_metadata(tmp_path, nvfp4_layers=[0], mxfp8_layers=[1])
    calls: list[tuple[str, str, str | None]] = []

    def fake_download(repo_id: str, filename: str, *, revision: str | None = None, **kwargs: object) -> str:
        calls.append((repo_id, filename, revision))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    commit = "0123456789abcdef0123456789abcdef01234567"
    # The shared factory forwards the resolved revision only to constructors that declare it.
    for cls in (Inkling2ForCausalLM, Inkling2ForConditionalGeneration):
        assert "revision" in inspect.signature(cls.__init__).parameters
        model = cls(released_image_text_config(tmp_path, "org/inkling2-small"), revision=commit)
        assert calls[-1] == ("org/inkling2-small", "hf_quant_config.json", commit)
        assert model.state_dict_adapter.nvfp4_layer_count == 1
    assert len(calls) == 2
    # Without a resolved revision the default branch is read; a local directory reads its own file.
    Inkling2ForCausalLM(released_image_text_config(tmp_path, "org/inkling2-small"))
    assert len(calls) == 3 and calls[-1][2] is None
    Inkling2ForCausalLM(released_image_text_config(tmp_path, str(tmp_path)), revision=commit)
    assert len(calls) == 3


def test_floating_export_strips_the_nested_quantization_declaration(tmp_path: Path) -> None:
    """A dequantized image/text export declares no source quantization at either level."""
    write_released_checkpoint_metadata(tmp_path, nvfp4_layers=[0], mxfp8_layers=[1])
    declaration = {"quant_method": "inkling2", "nvfp4_layer_count": 1}
    model = Inkling2ForConditionalGeneration(released_image_text_config(tmp_path, str(tmp_path)))
    declared = json.loads(model.config.to_json_string(use_diff=False))
    assert declared["quantization_config"] == declared["text_config"]["quantization_config"] == declaration
    _maybe_strip_quantization_config(model, config=model.config)
    for exported in (
        json.loads(model.config.to_json_string(use_diff=False)),
        json.loads(model.config.to_json_string()),
    ):
        assert "quantization_config" not in exported and "quantization_config" not in exported["text_config"]
    # The live import layout is untouched; only the serialized export changes.
    assert model.config.text_config.quantization_config == declaration and model.state_dict_adapter.source_quantized
    floating = tmp_path / "floating"
    model.config.save_pretrained(floating)
    reloaded = Inkling2Config.from_pretrained(floating)
    assert reloaded.quantization_config is None and reloaded.text_config.quantization_config is None
    assert not Inkling2ForConditionalGeneration.from_config(reloaded).state_dict_adapter.source_quantized
