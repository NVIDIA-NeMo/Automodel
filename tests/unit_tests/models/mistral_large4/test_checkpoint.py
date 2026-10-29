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

"""Source FP8 loading, grouped LoRA and floating training checkpoint lifecycle."""

import copy
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from nemo_automodel._transformers import registry
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.models.mistral_large4.state_dict_adapter import (
    MistralLarge4StateDictAdapter,
    _decode_into,
)
from nemo_automodel.components.models.mistral_large4.vision import MistralLarge4VisionConfig
from nemo_automodel.components.training.rng import StatefulRNG
from tests.unit_tests.models.mistral_large4.math_reference import tiny_config
from tests.unit_tests.models.mistral_large4.test_model import make_model


def step(
    model: MistralLarge4ForCausalLM,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    ids: torch.Tensor,
) -> torch.Tensor:
    """Apply a real supervised update.

    Args:
        model: Floating causal model with matrix [output, input] parameters.
        optimizer: AdamW holding tensor states in the corresponding parameter layouts.
        scheduler: Learning-rate scheduler.
        ids: Integer [batch, sequence] tokens/next-token targets.

    Returns:
        Detached scalar loss tensor.
    """
    optimizer.zero_grad(set_to_none=True)
    loss = model(ids, labels=ids).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    scheduler.step()
    return loss.detach()


def peft_config() -> PeftConfig:
    return PeftConfig(
        target_modules=[
            "*.self_attn.q_proj",
            "*.self_attn.kv_b_proj",
            "*.self_attn.o_proj",
            "*.mlp.experts",
        ],
        dim=2,
        alpha=4,
        use_memory_efficient_lora=False,
        lora_dtype=None,
        dropout=0.1,
    )


def make_optimizer(model: MistralLarge4ForCausalLM) -> tuple[torch.optim.AdamW, torch.optim.lr_scheduler.StepLR]:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.003, weight_decay=0.01)
    return optimizer, torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.7)


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_sft_lora_native_dcp_resume_matches_uninterrupted(tmp_path: Path, lora: bool, dtype: torch.dtype) -> None:
    model = make_model(dtype=dtype)
    base = copy.deepcopy(model.state_dict())
    peft = peft_config() if lora else None
    if lora:
        count = apply_lora_to_linear_modules(model, peft)
        assert count == 15
        assert all("lora_" in name for name, p in model.named_parameters() if p.requires_grad)
    optimizer, scheduler = make_optimizer(model)
    state = StatefulRNG(seed=781)
    ids = torch.tensor([[1, 5, 7, 2, 8, 3, 6], [8, 2, 5, 3, 7, 4, 1]])
    initial = model.eval()(ids, labels=ids).loss.item()
    model.train()
    for _ in range(3):
        step(model, optimizer, scheduler, ids)
    checkpointer = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path),
        model_save_format="torch_save",
        save_consolidated=False,
        is_peft=lora,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    saved = str(tmp_path / "step3")
    try:
        checkpointer.save_model(model, saved, peft_config=peft)
        checkpointer.save_optimizer(optimizer, model, saved, scheduler)
        # Framework RNG persistence is a separate existing public hook.
        checkpointer.save_on_global_ranks(state, "training_state", saved)
        expected_losses = [step(model, optimizer, scheduler, ids) for _ in range(3)]
        reloaded = make_model(dtype=dtype)
        reloaded.load_state_dict(base)
        if lora:
            apply_lora_to_linear_modules(reloaded, peft)
        new_optimizer, new_scheduler = make_optimizer(reloaded)
        checkpointer.load_model(reloaded, str(Path(saved) / "model"), is_init_step=False)
        checkpointer.load_optimizer(new_optimizer, reloaded, saved, new_scheduler)
        restored = StatefulRNG(seed=0)
        checkpointer.load_on_global_ranks(restored, "training_state", saved)
        actual_losses = [step(reloaded, new_optimizer, new_scheduler, ids) for _ in range(3)]
        torch.testing.assert_close(torch.stack(actual_losses), torch.stack(expected_losses), atol=0, rtol=0)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(reloaded.state_dict()[key], value, atol=0, rtol=0, msg=key)
        assert new_scheduler.state_dict() == scheduler.state_dict()
        for old_state, new_state in zip(optimizer.state.values(), new_optimizer.state.values()):
            for key in old_state:
                torch.testing.assert_close(new_state[key], old_state[key], atol=0, rtol=0)
        assert model.eval()(ids, labels=ids).loss.item() < initial
        if lora:
            # KV-B and routed expert LoRA must have nonzero updates; base stays exact.
            changed = [
                name
                for name, p in model.named_parameters()
                if ("lora_B" in name or name.endswith("_B")) and torch.count_nonzero(p)
            ]
            assert any("kv_b_proj" in name for name in changed)
            assert any("mlp.experts" in name for name in changed)
            for name, value in base.items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    finally:
        checkpointer.close()


def test_fp8_initialization_float_export_and_reinitialization(tmp_path: Path) -> None:
    model = make_model()
    # Build a small dependency-complete native checkpoint, exercising every family.
    raw = model.state_dict_adapter.to_hf(model.state_dict())
    quantized = {}
    for name, value in raw.items():
        should_quantize = any(token in name for token in (".wq.", ".wo.", ".wq_b.", ".w1.", ".w2.", ".w3."))
        if should_quantize:
            quantized[name] = value.to(torch.float8_e4m3fn).contiguous()
            quantized[name.removesuffix(".weight") + ".qscale_weight"] = torch.full(
                tuple((d + 127) // 128 for d in value.shape), 0.25, dtype=torch.float32
            )
        else:
            quantized[name] = value.contiguous()
    source = tmp_path / "source"
    source.mkdir()
    save_file(quantized, source / "model.safetensors")
    expected = model.state_dict_adapter.from_hf(quantized)
    config = tiny_config(quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]})
    config.save_pretrained(source)
    model = MistralLarge4ForCausalLM(config)
    cp = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        model_save_format="safetensors",
        save_consolidated=True,
        model_cache_dir=str(tmp_path),
        model_repo_id=str(source),
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    try:
        cp.load_model(model, str(source), is_init_step=True)
        for key, value in expected.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
        ids = torch.tensor([[1, 2, 3, 4]])
        model(ids, labels=ids).loss.backward()
        torch.optim.SGD(model.parameters(), lr=0.01).step()
        cp.save_model(model, str(tmp_path / "export"))
        metadata = list((tmp_path / "export").rglob("config.json"))
        assert metadata
        assert "quantization_config" not in json.loads(metadata[0].read_text())
        shards = list((tmp_path / "export").rglob("*.safetensors"))
        assert shards
        for shard in shards:
            saved = load_file(shard)
            assert all(v.dtype in (torch.float32, torch.bfloat16) for v in saved.values())
            assert not any("qscale" in k for k in saved)
        reloaded = MistralLarge4ForCausalLM(tiny_config())
        cp.load_model(reloaded, str(tmp_path / "export/model"), is_init_step=False)
        torch.testing.assert_close(reloaded(ids).logits, model(ids).logits, rtol=0, atol=0)
        # Exported metadata must also be usable as a fresh floating base checkpoint.
        config = type(model.config).from_pretrained(metadata[0].parent)
        fresh = MistralLarge4ForCausalLM(config)
        cp.load_model(fresh, str(tmp_path / "export/model/consolidated"), is_init_step=True)
        torch.testing.assert_close(fresh(ids).logits, model(ids).logits, rtol=0, atol=0)
    finally:
        cp.close()


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_initialization_preserves_resident_body_storage(lora: bool, dtype: torch.dtype) -> None:
    """Production meta initialization must not replace same-dtype body storage."""
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(tiny_config(dtype=dtype))
        if lora:
            apply_lora_to_linear_modules(model, peft_config())
    model.to_empty(device=torch.device("cpu"))
    before = {
        name: (parameter.dtype, parameter.data_ptr(), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(float("nan"))
    Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier" if lora else None)
    protected = ("rotary_emb", "indexer.k_norm", "mlp.gate.weight")
    for name, parameter in model.named_parameters():
        original_dtype, pointer, trainable = before[name]
        assert parameter.dtype == original_dtype, name
        assert parameter.requires_grad == trainable, name
        assert torch.isfinite(parameter).all(), name
        if any(part in name for part in protected):
            assert parameter.dtype == torch.float32, name
        else:
            assert parameter.data_ptr() == pointer, f"initializer replaced resident storage: {name}"
        if lora and ("lora_B" in name or name.endswith("_B")):
            assert torch.count_nonzero(parameter) == 0, name
    ids = torch.tensor([[1, 5, 7, 2, 8]])
    loss = model(ids, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name
        if not parameter.requires_grad:
            assert parameter.grad is None, name


def test_grouped_lora_zero_update_preserves_activation_and_logits():
    model = make_model(dtype=torch.bfloat16)
    ids = torch.tensor([[1, 5, 7, 2, 8]])
    expected = model(ids).logits.detach()
    activations = {
        name: module.expert_activation_grouped
        for name, module in model.named_modules()
        if hasattr(module, "expert_activation_grouped")
    }
    count = apply_lora_to_linear_modules(
        model,
        PeftConfig(
            target_modules=["*.mlp.experts"],
            dim=4,
            alpha=8,
            dropout=0.0,
            use_memory_efficient_lora=False,
        ),
    )
    assert count == 3
    for name, function in activations.items():
        assert model.get_submodule(name).expert_activation_grouped is function
    torch.testing.assert_close(model(ids).logits, expected, atol=0, rtol=0)


def test_config_registry_round_trip(tmp_path):
    config = MistralLarge4Config()
    config.save_pretrained(tmp_path)
    restored = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(restored, MistralLarge4Config)
    assert registry.resolve_custom_config_cls("mistral_large4") is MistralLarge4Config
    assert restored.kv_source_layers[39::4] == [35] * 9
    assert restored.nope_layer_flags == [i % 4 == 3 for i in range(72)]
    assert restored.index_sharing[16:24] == ["full", "shared", "shared", "full", "full", "shared", "shared", "full"]
    assert restored.tie_word_embeddings is False
    assert restored.architectures == ["MistralLarge4ForCausalLM"]
    assert restored.model_type != "mistral4"


def test_prefix_keeps_original_sharing():
    config = MistralLarge4Config(num_hidden_layers=40)
    assert config.kv_source_layers[39] == 35
    assert config.nope_layer_flags[35] and config.nope_layer_flags[39]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tie_word_embeddings": True},
        {"q_lora_rank": 8},
        {"head_dim": 64},
        {"intermediate_size": -1},
        {"norm_topk_prob": False},
        {"num_hidden_layers": 0},
        {"num_experts_per_tok": 129},
        {"qk_rope_head_dim": 63},
        {"num_hidden_layers": 1, "kv_source_layers": [-1]},
        {"num_hidden_layers": 1, "index_sharing": ["shared"]},
    ],
)
def test_invalid_configuration_fails(kwargs):
    with pytest.raises(ValueError):
        MistralLarge4Config(**kwargs)


def _source():
    # Multiple scales and odd dimensions expose block-axis and edge errors.
    values = (torch.arange(257 * 259).reshape(257, 259) % 29 - 14).float() / 4
    weights = values.to(torch.float8_e4m3fn)
    scales = torch.tensor([[0.5, 2.0, 4.0], [8.0, 0.25, 16.0], [32.0, 64.0, 0.125]])
    expected = torch.empty_like(values)
    # Independent oracle: block slices, rather than the decoder's index gather.
    for i in range(3):
        for j in range(3):
            expected[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] = (
                values[i * 128 : (i + 1) * 128, j * 128 : (j + 1) * 128] * scales[i, j]
            )
    return weights, scales, expected


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fp8_exact_reconstruction_and_storage(dtype):
    weights, scales, expected = _source()
    destination = torch.empty((259, 257), dtype=dtype).t()
    pointer = destination.data_ptr()
    _decode_into(weights, scales, destination)
    torch.testing.assert_close(destination, expected.to(dtype), rtol=0, atol=0)
    assert destination.data_ptr() == pointer
    assert not destination.is_contiguous()


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_scales_rejected(bad_scale):
    weights, scales, expected = _source()
    scales[0, 0] = bad_scale
    with pytest.raises(ValueError, match="positive and finite"):
        _decode_into(weights, scales, expected)


def test_missing_and_orphaned_scales_rejected():
    weights, scales, _ = _source()
    adapter = MistralLarge4StateDictAdapter()
    key = "layers.1.experts.127.w3.weight"
    scale_key = key.removesuffix(".weight") + ".qscale_weight"
    with pytest.raises(ValueError, match="Missing FP8 scale"):
        adapter.from_hf({key: weights})
    with pytest.raises(ValueError, match="Orphaned"):
        adapter.from_hf({key: weights.float(), scale_key: scales})


def test_float_round_trip_does_not_reuse_source_scales():
    weights, scales, expected = _source()
    adapter = MistralLarge4StateDictAdapter()
    source = {
        "layers.1.experts.127.w3.weight": weights,
        "layers.1.experts.127.w3.qscale_weight": scales,
        "tok_embeddings.weight": torch.ones(17, 8, dtype=torch.bfloat16),
        "output.weight": torch.zeros(17, 8, dtype=torch.bfloat16),
        "vision_encoder.norm.weight": torch.ones(8, dtype=torch.bfloat16),
        "mtp_dspark.test.weight": torch.ones(8, 8, dtype=torch.bfloat16),
    }
    original_keys = set(source)
    native = adapter.from_hf(source)
    assert set(source) == original_keys
    torch.testing.assert_close(
        native["model.layers.1.mlp.experts.127.up_proj.weight"], expected.bfloat16(), rtol=0, atol=0
    )
    assert native["model.embed_tokens.weight"].data_ptr() != native["lm_head.weight"].data_ptr()
    native["model.layers.1.mlp.experts.127.up_proj.weight"].add_(1)
    exported = adapter.to_hf(native)
    assert not any("qscale" in key for key in exported)
    reloaded = adapter.from_hf(exported)
    for name, tensor in native.items():
        torch.testing.assert_close(reloaded[name], tensor, rtol=0, atol=0)
    with pytest.raises(NotImplementedError, match="floating training checkpoints"):
        adapter.to_hf(native, quantization=True)


def test_real_dcp_populates_parts(tmp_path):
    from torch.distributed.checkpoint import load
    from torch.distributed.checkpoint.hf_storage import HuggingFaceStorageReader

    weights, scales, expected = _source()
    dense = torch.arange(24).reshape(3, 8).bfloat16()
    source = {
        "layers.0.attention.wq.weight": weights,
        "layers.0.attention.wq.qscale_weight": scales,
        "output.weight": dense,
    }
    save_file(source, str(tmp_path / "model-00001-of-00001.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(x.numel() * x.element_size() for x in source.values())},
                "weight_map": {key: "model-00001-of-00001.safetensors" for key in source},
            }
        )
    )
    state = {
        "model.layers.0.self_attn.q_proj.weight": torch.empty_like(expected, dtype=torch.bfloat16),
        "lm_head.weight": torch.empty_like(dense),
    }
    pointers = {k: v.data_ptr() for k, v in state.items()}
    seen = set()
    reader = HuggingFaceStorageReader(str(tmp_path))
    for part in MistralLarge4StateDictAdapter().iter_checkpoint_load_parts(state):
        assert not seen.intersection(part.model_keys)
        seen.update(part.model_keys)
        if "output.weight" in part.checkpoint_tensors:
            assert part.checkpoint_tensors["output.weight"].data_ptr() == state["lm_head.weight"].data_ptr()
        load(part.checkpoint_tensors, storage_reader=reader)
        part.finish()
    assert seen == set(state)
    assert pointers == {k: v.data_ptr() for k, v in state.items()}
    torch.testing.assert_close(state["model.layers.0.self_attn.q_proj.weight"], expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(state["lm_head.weight"], dense, rtol=0, atol=0)


def _released_config() -> dict:
    # Released Transformers config.json; it contains no weights or tokenizer.
    return json.loads((Path(__file__).parent / "config.json").read_text())


def test_released_config_maps_to_native_fields(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_released_config()))
    config = AutoConfig.from_pretrained(tmp_path, local_files_only=True)
    assert isinstance(config, MistralLarge4Config)
    assert config.architectures == ["MistralLarge4ForConditionalGeneration"]
    assert config.nope_layer_flags == [layer % 4 == 3 for layer in range(72)]
    assert config.indexer_nope_layer_flags == config.nope_layer_flags
    assert config.kv_source_layers == [35 if layer >= 39 and layer % 4 == 3 else layer for layer in range(72)]
    assert config.index_sharing == ["shared" if layer >= 17 and layer % 4 in (1, 2) else "full" for layer in range(72)]
    assert (config.yarn_factor, config.yarn_original_max_position_embeddings) == (2048, 512)
    assert (config.n_routed_experts, config.first_k_dense_replace, config.head_dim) == (128, 1, 128)
    assert config.quantization_config["quant_method"] == "fp8"
    assert config.transformers_weights == "consolidated.safetensors.index.json"
    assert MistralLarge4VisionConfig(**config.vision_config) == MistralLarge4VisionConfig()
    config.save_pretrained(tmp_path / "saved")
    restored = AutoConfig.from_pretrained(tmp_path / "saved", local_files_only=True).to_dict()
    assert {**restored, "_name_or_path": None} == {**config.to_dict(), "_name_or_path": None}


@pytest.mark.parametrize(
    "section,changes",
    [
        ("text_config", {"rope_parameters": {"rope_type": "default", "rope_theta": 10000.0}}),
        ("text_config", {"n_group": 8}),
        ("text_config", {"attention_bias": True}),
        ("text_config", {"mlp_layer_types": ["sparse"] * 72}),
        ("text_config", {"no_rope_layers": [2] * 72}),
        ("text_config", {"tie_word_embeddings": True}),
    ],
)
def test_released_config_rejects_unsupported_variants(section, changes):
    config = _released_config()
    config[section].update(changes)
    with pytest.raises(ValueError):
        MistralLarge4Config.from_dict(config)
