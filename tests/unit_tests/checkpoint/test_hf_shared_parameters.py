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

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.api import CheckpointException
from transformers import AutoModelForSeq2SeqLM, BartConfig, PretrainedConfig, T5Config

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig


@pytest.fixture(params=["t5", "bart"])
def model_config(request) -> PretrainedConfig:
    if request.param == "t5":
        return T5Config(
            vocab_size=32,
            d_model=16,
            d_ff=32,
            d_kv=8,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            decoder_start_token_id=0,
            dropout_rate=0.0,
        )
    return BartConfig(
        vocab_size=32,
        d_model=16,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=32,
        decoder_ffn_dim=32,
        max_position_embeddings=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )


@pytest.fixture
def checkpointer(tmp_path: Path) -> Checkpointer:
    return Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path / "cache"),
            model_repo_id="test/shared-parameters",
            model_save_format="safetensors",
            save_consolidated=False,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
@pytest.mark.parametrize("mapped_keys", [False, True])
def test_hf_shared_embeddings_load_with_weight_and_gradient_parity(
    tmp_path: Path,
    model_config: PretrainedConfig,
    checkpointer: Checkpointer,
    tie_word_embeddings: bool,
    mapped_keys: bool,
) -> None:
    """Real HF saves omit shared aliases; DCP must restore the full model without copying it to CPU."""
    torch.manual_seed(1234)
    model_config.tie_word_embeddings = tie_word_embeddings
    reference = AutoModelForSeq2SeqLM.from_config(model_config).eval()
    reference.save_pretrained(tmp_path / "model")
    checkpoint = load_file(tmp_path / "model" / "model.safetensors")
    missing_aliases = set(reference.state_dict()) - checkpoint.keys()
    if tie_word_embeddings:
        assert missing_aliases
    key_mapping = None
    if mapped_keys:
        save_file(
            {f"checkpoint.{key}": value for key, value in checkpoint.items()}, tmp_path / "model" / "model.safetensors"
        )
        key_mapping = {r"^checkpoint\.": ""}

    model = AutoModelForSeq2SeqLM.from_config(model_config).eval()
    original_parameters = dict(model.named_parameters(remove_duplicate=False))
    with patch(
        "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
        side_effect=AssertionError("Expected DCP loading"),
    ):
        checkpointer.load_model(model, str(tmp_path / "model"), is_init_step=True, key_mapping=key_mapping)

    for name, parameter in model.named_parameters(remove_duplicate=False):
        assert parameter is original_parameters[name]
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
    assert (model.get_encoder().embed_tokens.weight is model.get_decoder().embed_tokens.weight) == (
        reference.get_encoder().embed_tokens.weight is reference.get_decoder().embed_tokens.weight
    )
    assert (model.get_output_embeddings().weight is model.get_input_embeddings().weight) == tie_word_embeddings

    inputs = torch.tensor([[3, 4, 5, 2]])
    labels = torch.tensor([[6, 7, 2]])
    expected = reference(input_ids=inputs, labels=labels)
    actual = model(input_ids=inputs, labels=labels)
    torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
    torch.testing.assert_close(actual.loss, expected.loss, rtol=0, atol=0)
    actual.loss.backward()
    expected.loss.backward()
    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        expected_gradient = reference_parameters[name].grad
        if expected_gradient is None:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None
            torch.testing.assert_close(parameter.grad, expected_gradient, rtol=0, atol=0)


def test_hf_checkpoint_can_keep_only_the_lm_head_alias(
    tmp_path: Path, model_config: PretrainedConfig, checkpointer: Checkpointer
) -> None:
    """A saved alias absent from ModelState's initial destinations can still supply all shared embeddings."""
    reference = AutoModelForSeq2SeqLM.from_config(model_config)
    reference.save_pretrained(tmp_path / "model")
    checkpoint = load_file(tmp_path / "model" / "model.safetensors")
    source_name = "shared.weight" if model_config.model_type == "t5" else "model.shared.weight"
    checkpoint["lm_head.weight"] = checkpoint.pop(source_name)
    save_file(checkpoint, tmp_path / "model" / "model.safetensors")

    model = AutoModelForSeq2SeqLM.from_config(model_config)
    checkpointer.load_model(model, str(tmp_path / "model"), is_init_step=True)
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("missing_parameter", ["shared", "independent", "untied_alias"])
def test_shared_alias_handling_does_not_hide_missing_weights(
    tmp_path: Path, model_config: PretrainedConfig, checkpointer: Checkpointer, missing_parameter: str
) -> None:
    """Only aliases of the same live parameter may be omitted from DCP destinations."""
    reference = AutoModelForSeq2SeqLM.from_config(model_config)
    reference.save_pretrained(tmp_path / "model")
    checkpoint = load_file(tmp_path / "model" / "model.safetensors")
    model = AutoModelForSeq2SeqLM.from_config(model_config)
    if missing_parameter == "untied_alias":
        # Leave the config's tying declaration intact, but make the encoder embedding independent.
        model.get_encoder().embed_tokens = torch.nn.Embedding(model_config.vocab_size, model_config.d_model)
    else:
        source_name = "shared.weight" if model_config.model_type == "t5" else "model.shared.weight"
        missing_name = (
            source_name if missing_parameter == "shared" else next(key for key in checkpoint if key != source_name)
        )
        del checkpoint[missing_name]
        save_file(checkpoint, tmp_path / "model" / "model.safetensors")

    with pytest.raises(CheckpointException, match="Missing key in checkpoint state_dict"):
        checkpointer.load_model(model, str(tmp_path / "model"), is_init_step=True)
