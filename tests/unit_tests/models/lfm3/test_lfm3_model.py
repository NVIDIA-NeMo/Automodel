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

"""LFM3 and LFM3-VL against goldens from the native Transformers 5.23.0 implementation."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.lfm3.config import Lfm3Config, Lfm3VlConfig
from nemo_automodel.components.models.lfm3.model import Lfm3ForCausalLM, rotary_factors
from nemo_automodel.components.models.lfm3.vl_model import Lfm3VlForConditionalGeneration

REFERENCE = Path(__file__).with_name("fixtures") / "reference.npz"
TOLERANCE = {"atol": 3e-6, "rtol": 3e-5}
MEDIA = ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw")


@pytest.fixture(scope="module")
def reference():
    """Golden arrays and the configs they were generated with."""
    metadata = json.loads(REFERENCE.with_suffix(".json").read_text())
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == metadata["archive_sha256"]
    with np.load(REFERENCE, allow_pickle=False) as archive:
        arrays = {name: torch.from_numpy(archive[name]) for name in archive.files}
    assert len(arrays) == metadata["array_count"]
    return arrays, metadata


def _state(arrays):
    return {name.removeprefix("state."): value for name, value in arrays.items() if name.startswith("state.")}


def _text_model(reference, **config_overrides):
    arrays, metadata = reference
    with torch.device("cpu"):
        model = Lfm3ForCausalLM(Lfm3Config(**{**metadata["text_config"], **config_overrides})).float().eval()
    prefix = "model.language_model."
    state = {f"model.{name.removeprefix(prefix)}": value for name, value in _state(arrays).items() if prefix in name}
    assert model.load_state_dict(state, strict=False).unexpected_keys == []
    return model


def _vl_model(reference):
    arrays, metadata = reference
    config = Lfm3VlConfig(
        text_config=metadata["text_config"], vision_config=metadata["vision_config"], **metadata["vl_config"]
    )
    with torch.device("cpu"):
        model = Lfm3VlForConditionalGeneration(config).float().eval()
    # The golden weights use the names Transformers writes to disk; the tied head is not stored.
    loaded = model.load_state_dict(_state(arrays), strict=False)
    assert (loaded.missing_keys, loaded.unexpected_keys) == (["lm_head.weight"], [])
    return model


def _vl_inputs(arrays):
    return {"input_ids": arrays["input.ids"], "attention_mask": arrays["input.mask"]} | {
        name: arrays[f"input.{name}"] for name in MEDIA
    }


def test_text_model_matches_native_logits_loss_and_generation(reference):
    arrays, metadata = reference
    model = _text_model(reference)
    ids, mask = arrays["input.text_ids"], arrays["input.mask"]

    output = model(input_ids=ids, attention_mask=mask, labels=ids)
    generated = model.generate(input_ids=ids, attention_mask=mask, **metadata["generate"])

    torch.testing.assert_close(output.logits, arrays["expected.text.logits"], **TOLERANCE)
    torch.testing.assert_close(output.loss, arrays["expected.text.loss"], **TOLERANCE)
    assert torch.equal(generated, arrays["expected.text.generated"])


def test_published_rope_label_gives_the_native_proportional_result(reference):
    arrays, _ = reference
    published = {
        "full_attention": {"rope_type": "default", "rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10_000},
    }
    model = _text_model(reference, rope_parameters=published)

    logits = model(input_ids=arrays["input.text_ids"], attention_mask=arrays["input.mask"]).logits

    torch.testing.assert_close(logits, arrays["expected.text.logits"], **TOLERANCE)


def test_vl_model_matches_native_logits_loss_gradients_and_generation(reference):
    arrays, metadata = reference
    model = _vl_model(reference)
    inputs = _vl_inputs(arrays)

    output = model(**inputs, labels=inputs["input_ids"])
    output.loss.backward()
    gradients = dict(model.named_parameters())

    torch.testing.assert_close(output.logits, arrays["expected.vl.logits"], **TOLERANCE)
    torch.testing.assert_close(output.loss, arrays["expected.vl.loss"], **TOLERANCE)
    expected_gradients = {name: value for name, value in arrays.items() if name.startswith("expected.vl.grad.")}
    assert len(expected_gradients) == 8
    for name, expected in expected_gradients.items():
        torch.testing.assert_close(gradients[name.removeprefix("expected.vl.grad.")].grad, expected, **TOLERANCE)
    with torch.no_grad():
        assert torch.equal(model.generate(**inputs, **metadata["generate"]), arrays["expected.vl.generated"])


def test_partial_rotation_leaves_the_tail_of_each_head_unrotated():
    positions = torch.arange(50)[None]

    cosine, sine = rotary_factors(positions, {"rope_theta": 1_000_000.0, "partial_rotary_factor": 0.5}, head_dim=8)

    assert cosine.dtype == sine.dtype == torch.float32 and cosine.shape == (1, 50, 8)
    # Two of the four frequency pairs rotate; the pair layout is (first half, second half) of the head.
    expected = positions[..., None] * torch.tensor([1.0, 1_000_000.0 ** (-2 / 8)])
    torch.testing.assert_close(cosine[..., :2], expected.cos())
    torch.testing.assert_close(sine[..., 4:6], expected.sin())
    assert torch.all(cosine[..., [2, 3, 6, 7]] == 1) and torch.all(sine[..., [2, 3, 6, 7]] == 0)


def test_model_wide_bfloat16_cast_keeps_full_precision_rotary_frequencies(reference):
    arrays, _ = reference
    ids = arrays["input.text_ids"]
    cast_after_load = _text_model(reference).to(torch.bfloat16)
    with torch.device("cpu"):
        built_in_bfloat16 = Lfm3ForCausalLM(cast_after_load.config).to(torch.bfloat16).eval()
    built_in_bfloat16.load_state_dict(cast_after_load.state_dict())

    with torch.no_grad():
        assert torch.equal(cast_after_load(input_ids=ids).logits, built_in_bfloat16(input_ids=ids).logits)
    assert not [name for name, _ in cast_after_load.named_buffers()]


def test_cached_generation_matches_recomputing_the_full_sequence(reference):
    arrays, _ = reference
    model = _text_model(reference)
    ids = arrays["input.text_ids"][:, :9]

    with torch.no_grad():
        cached = model.generate(input_ids=ids, max_new_tokens=10, do_sample=False, eos_token_id=None, pad_token_id=0)
        uncached = ids
        for _ in range(10):
            uncached = torch.cat(
                [uncached, model(input_ids=uncached, use_cache=False).logits[:, -1:].argmax(-1)], dim=1
            )

    assert torch.equal(cached, uncached)


@pytest.mark.parametrize("tied", [True, False])
def test_vocabulary_head_follows_the_config_tie_setting(reference, tied):
    model = _text_model(reference, tie_word_embeddings=tied)
    vl_config = Lfm3VlConfig(
        text_config=reference[1]["text_config"], vision_config=reference[1]["vision_config"], tie_word_embeddings=tied
    )
    with torch.device("cpu"):
        vl_model = Lfm3VlForConditionalGeneration(vl_config)

    assert (model.lm_head.weight is model.model.embed_tokens.weight) is tied
    assert (vl_model.lm_head.weight is vl_model.model.language_model.embed_tokens.weight) is tied


def test_checkpointer_load_export_and_resume_keep_released_names_weights_and_tie(reference, tmp_path):
    state = _state(reference[0])
    config = _vl_model(reference).config
    config.dtype, config.name_or_path = torch.float32, str(tmp_path / "source")
    config.save_pretrained(tmp_path / "source")
    save_file(state, tmp_path / "source" / "model.safetensors")
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )

    def load(path, **kwargs):
        with torch.device("cpu"):
            model = Lfm3VlForConditionalGeneration(config).eval()
        checkpointer.load_model(model, str(path), **kwargs)
        assert model.lm_head.weight is model.model.language_model.embed_tokens.weight
        return model

    model = load(tmp_path / "source", is_init_step=True)
    checkpointer.save_model(model, str(tmp_path / "saved"))
    exported = load_file(next((tmp_path / "saved/model/consolidated").glob("*.safetensors")))
    resumed = load(tmp_path / "saved/model")

    assert not hasattr(model, "state_dict_adapter")
    assert exported.keys() - {"lm_head.weight"} == state.keys()
    for name, value in state.items():
        assert torch.equal(model.state_dict()[name], value), name
        assert torch.equal(exported[name], value), name
        assert torch.equal(resumed.state_dict()[name], value), name


def test_frozen_low_precision_vision_tower_feeds_a_full_precision_projector(reference):
    arrays, _ = reference
    model = _vl_model(reference)
    expected = model.model.get_image_features(arrays["input.pixel_values"], arrays["input.image_grid_thw"])
    model.model.vision_tower.to(torch.bfloat16)

    with torch.no_grad():
        features = model.model.get_image_features(
            arrays["input.pixel_values"].to(torch.bfloat16), arrays["input.image_grid_thw"]
        )

    assert [tokens.shape[0] for tokens in features.pooler_output] == [6, 2]
    assert features.pooler_output[0].dtype == torch.float32
    torch.testing.assert_close(torch.cat(features.pooler_output), torch.cat(expected.pooler_output), atol=0.5, rtol=0.1)


def test_media_inputs_that_cannot_be_placed_are_rejected(reference):
    arrays, _ = reference
    model = _vl_model(reference)
    inputs = _vl_inputs(arrays)
    images = {name: inputs[name] for name in ("pixel_values", "image_grid_thw")}

    with pytest.raises(ValueError, match="6 placeholder tokens with id 10 for 8 vision features"):
        model(input_ids=inputs["input_ids"][:1], **images)
    with pytest.raises(ValueError, match="need `input_ids`"):
        model(inputs_embeds=model.get_input_embeddings()(inputs["input_ids"]), **images)
    with pytest.raises(ValueError, match="not both"):
        model(input_ids=inputs["input_ids"], mm_encoder_outputs={}, **images)


def test_a_few_optimizer_steps_reduce_the_loss(reference):
    arrays, _ = reference
    model = _vl_model(reference).train()
    inputs = _vl_inputs(arrays)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for _ in range(5):
        loss = model(**inputs, labels=inputs["input_ids"]).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())

    assert losses[-1] < losses[0] and all(np.isfinite(losses))
