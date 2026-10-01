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

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch.distributed.checkpoint.api import CheckpointException
from transformers import InternVLConfig, InternVLForConditionalGeneration

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
