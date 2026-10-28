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

"""Exercise nested HF checkpoint renames through the real base-model loader."""

import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from transformers import (
    InternVLConfig,
    InternVLForConditionalGeneration,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
)

from nemo_automodel.components.checkpoint._backports.hf_storage import _get_key_renaming_mapping
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.conversion_mapping import get_combined_key_mapping


@pytest.fixture
def reference() -> InternVLForConditionalGeneration:
    config = InternVLConfig(
        vision_config={
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "image_size": 8,
            "patch_size": 2,
        },
        text_config={
            "model_type": "qwen3",
            "vocab_size": 32,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "max_position_embeddings": 32,
            "tie_word_embeddings": False,
        },
        image_token_id=31,
    )
    config._attn_implementation = "eager"
    return InternVLForConditionalGeneration(config).eval()


@pytest.fixture
def checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


@pytest.mark.parametrize("legacy_keys", [True, False], ids=["published", "native"])
@pytest.mark.parametrize("checkpoint_format", ["safetensors", "bin"])
def test_internvl_base_checkpoint_restores_weights_and_logits(
    tmp_path: Path,
    reference: InternVLForConditionalGeneration,
    checkpointer: Checkpointer,
    legacy_keys: bool,
    checkpoint_format: str,
) -> None:
    """Both published and already-converted keys restore every weight through the production loader."""
    checkpoint = {}
    for key, value in reference.state_dict().items():
        if legacy_keys:
            if key.startswith("model.language_model."):
                key = "language_model.model." + key.removeprefix("model.language_model.")
            elif key.startswith("lm_head."):
                key = "language_model." + key
            else:
                key = key.removeprefix("model.")
        checkpoint[key] = value.clone()
    source = tmp_path / "source"
    source.mkdir()
    if checkpoint_format == "safetensors":
        save_file(checkpoint, source / "model.safetensors")
    else:
        torch.save(checkpoint, source / "pytorch_model.bin")

    model = type(reference)(reference.config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpointer.load_base_model(model, torch.device("cpu"), str(tmp_path), str(source))

    for key, expected in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    inputs = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        expected = reference(input_ids=inputs, use_cache=False).logits
        actual = model(input_ids=inputs, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_internvl_missing_checkpoint_weight_still_fails(
    tmp_path: Path, reference: InternVLForConditionalGeneration, checkpointer: Checkpointer
) -> None:
    """Name conversion must not turn a genuinely missing weight into a partial load."""
    checkpoint = {key: value.clone() for key, value in reference.state_dict().items()}
    missing_key = "model.language_model.layers.0.input_layernorm.weight"
    del checkpoint[missing_key]
    source = tmp_path / "source"
    source.mkdir()
    save_file(checkpoint, source / "model.safetensors")

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: " + missing_key):
        checkpointer.load_base_model(reference, torch.device("cpu"), str(tmp_path), str(source))


def test_nested_key_mapping_preserves_explicit_rule_precedence(reference: InternVLForConditionalGeneration) -> None:
    """An explicit complete rename wins over overlapping root rules and is not renamed twice."""
    mapping = get_combined_key_mapping(
        "internvl",
        {
            r"^language_model\.model\.": "model.language_model.",
            r"^language_model\.": "unused.",
        },
        model=reference,
    )
    expected = "model.language_model.layers.0.input_layernorm.weight"
    assert _get_key_renaming_mapping("language_model.model.layers.0.input_layernorm.weight", mapping) == expected
    assert _get_key_renaming_mapping(expected, mapping) == expected


def test_tensor_merging_rejects_callable_mapping(
    tmp_path: Path, reference: InternVLForConditionalGeneration, checkpointer: Checkpointer
) -> None:
    """Tensor converters require explicit regex rules rather than a rename-only callable."""
    reference.config.model_type = "mixtral"
    with pytest.raises(ValueError, match="Tensor-merging checkpoint loads require a regex key mapping"):
        checkpointer.load_model(reference, str(tmp_path), is_init_step=True, key_mapping=lambda key: key)


@pytest.fixture
def qwen_reference() -> Qwen2_5_VLForConditionalGeneration:
    config = Qwen2_5_VLConfig(
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 16,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 2,
            "window_size": 4,
            "fullatt_block_indexes": [0],
        },
        text_config={
            "vocab_size": 32,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "max_position_embeddings": 32,
            "rope_parameters": {"rope_type": "default", "mrope_section": [1, 1, 2]},
        },
        image_token_id=28,
        video_token_id=29,
        vision_start_token_id=30,
        vision_end_token_id=31,
        tie_word_embeddings=True,
    )
    config._attn_implementation = "eager"
    return Qwen2_5_VLForConditionalGeneration(config).eval()


@pytest.mark.parametrize("legacy_keys", [True, False], ids=["published", "native"])
@pytest.mark.parametrize("checkpoint_format", ["safetensors", "sharded", "bin"])
def test_qwen25vl_base_checkpoint_restores_weights_and_logits(
    tmp_path: Path,
    qwen_reference: Qwen2_5_VLForConditionalGeneration,
    checkpointer: Checkpointer,
    legacy_keys: bool,
    checkpoint_format: str,
) -> None:
    """Restore the Qwen2.5-VL published layout, including its omitted tied output head."""
    checkpoint = {}
    for key, value in qwen_reference.state_dict().items():
        if key == "lm_head.weight":
            continue
        if legacy_keys:
            if key.startswith("model.language_model."):
                key = "model." + key.removeprefix("model.language_model.")
            elif key.startswith("model.visual."):
                key = key.removeprefix("model.")
        checkpoint[key] = value.clone()
    source = tmp_path / "source"
    source.mkdir()
    if checkpoint_format == "sharded":
        weight_map = {}
        for shard_id in range(2):
            shard = dict(list(checkpoint.items())[shard_id::2])
            filename = f"model-{shard_id + 1:05d}-of-00002.safetensors"
            save_file(shard, source / filename)
            weight_map.update({key: filename for key in shard})
        (source / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    elif checkpoint_format == "safetensors":
        save_file(checkpoint, source / "model.safetensors")
    else:
        torch.save(checkpoint, source / "pytorch_model.bin")

    model = type(qwen_reference)(qwen_reference.config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    checkpointer.load_base_model(model, torch.device("cpu"), str(tmp_path), str(source))

    assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
    for key, expected in qwen_reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)
    inputs = torch.tensor([[1, 30, 28, 31, 2]])
    pixels = torch.arange(48, dtype=torch.float32).reshape(4, 12) / 48
    grid = torch.tensor([[1, 2, 2]])
    with torch.no_grad():
        expected = qwen_reference(input_ids=inputs, pixel_values=pixels, image_grid_thw=grid, use_cache=False).logits
        actual = model(input_ids=inputs, pixel_values=pixels, image_grid_thw=grid, use_cache=False).logits
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("missing_suffix", ["embed_tokens.weight", "layers.0.input_layernorm.weight"])
def test_qwen25vl_missing_checkpoint_weight_still_fails(
    tmp_path: Path,
    qwen_reference: Qwen2_5_VLForConditionalGeneration,
    checkpointer: Checkpointer,
    missing_suffix: str,
) -> None:
    """A genuinely missing embedding or layer weight must still fail a strict load."""
    checkpoint = {key: value.clone() for key, value in qwen_reference.state_dict().items()}
    missing_key = "model.language_model." + missing_suffix
    del checkpoint[missing_key]
    del checkpoint["lm_head.weight"]
    source = tmp_path / "source"
    source.mkdir()
    save_file(checkpoint, source / "model.safetensors")

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict: " + missing_key):
        checkpointer.load_base_model(qwen_reference, torch.device("cpu"), str(tmp_path), str(source))


def test_qwen25vl_explicit_mapping_overrides_class_rules(qwen_reference: Qwen2_5_VLForConditionalGeneration) -> None:
    """Explicit root overrides win without suppressing unrelated class-specific renames."""
    mapping = get_combined_key_mapping(
        "qwen2_5_vl",
        {r"^model\.embed_tokens\.": "custom_embeddings."},
        model=qwen_reference,
    )
    assert _get_key_renaming_mapping("model.embed_tokens.weight", mapping) == "custom_embeddings.weight"
    expected = "model.language_model.layers.0.input_layernorm.weight"
    assert _get_key_renaming_mapping("model.layers.0.input_layernorm.weight", mapping) == expected
    assert _get_key_renaming_mapping(expected, mapping) == expected
    assert (
        _get_key_renaming_mapping("visual.patch_embed.proj.weight", mapping) == "model.visual.patch_embed.proj.weight"
    )


def test_qwen25vl_fsdp_preserves_class_key_mapping(
    tmp_path: Path, qwen_reference: Qwen2_5_VLForConditionalGeneration
) -> None:
    """Real FSDP wrapping must preserve the HF class's root checkpoint renames."""
    dist.init_process_group("gloo", init_method=f"file://{tmp_path / 'rendezvous'}", rank=0, world_size=1)
    try:
        fully_shard(qwen_reference, mesh=init_device_mesh("cpu", (1,)))
        assert type(qwen_reference) is not Qwen2_5_VLForConditionalGeneration
        mapping = get_combined_key_mapping("qwen2_5_vl", model=qwen_reference)
        expected = "model.language_model.embed_tokens.weight"
        assert _get_key_renaming_mapping("model.embed_tokens.weight", mapping) == expected
        assert _get_key_renaming_mapping(expected, mapping) == expected
        assert (
            _get_key_renaming_mapping("visual.patch_embed.proj.weight", mapping)
            == "model.visual.patch_embed.proj.weight"
        )
    finally:
        dist.destroy_process_group()
