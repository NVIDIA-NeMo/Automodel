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

"""Capture-parity tests for HFDSparkTargetModel (CPU; target forward only).

These pin the wrapper's captured features to the HuggingFace
``output_hidden_states`` offset-1 convention used by ``extract_context_feature``
and the functional tests -- including the last layer (post-norm) and ``-1``
(embedding output), the two cases a forward hook gets wrong.
"""

import logging

import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from nemo_automodel.components.speculative.dspark.common import validate_target_layer_ids
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel

VOCAB = 256
HIDDEN = 64
NUM_LAYERS = 4


def _tiny_target():
    cfg = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )
    return AutoModelForCausalLM.from_config(cfg).to(dtype=torch.float32).eval()


def test_capture_matches_output_hidden_states_convention():
    target = _tiny_target()
    # -1 (embedding), an interior layer, and the last layer (post-norm).
    target_layer_ids = [-1, 1, NUM_LAYERS - 1]
    wrapper = HFDSparkTargetModel(target, target_layer_ids=target_layer_ids)

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (2, 16))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.uint8)

    batch = wrapper.generate_batch(input_ids, attention_mask, loss_mask)

    with torch.no_grad():
        ref = target(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    hs = ref.hidden_states  # (embed, layer_0, ..., post_norm); length NUM_LAYERS + 1
    expected = torch.cat([hs[0 if lid == -1 else lid + 1] for lid in target_layer_ids], dim=-1)

    assert batch.target_hidden_states.shape == (2, 16, len(target_layer_ids) * HIDDEN)
    assert torch.allclose(batch.target_hidden_states, expected, atol=1e-5)
    # Last hidden state is the post-norm final hidden (what lm_head consumes).
    assert torch.allclose(batch.target_last_hidden_states, hs[-1], atol=1e-5)


def test_negative_one_layer_id_is_accepted():
    target = _tiny_target()
    wrapper = HFDSparkTargetModel(target, target_layer_ids=[-1])
    input_ids = torch.randint(0, VOCAB, (1, 8))
    batch = wrapper.generate_batch(input_ids, torch.ones_like(input_ids), torch.ones_like(input_ids, dtype=torch.uint8))
    assert batch.target_hidden_states.shape == (1, 8, HIDDEN)


def _spy_on_forward(target):
    """Wrap ``target``'s bound ``forward`` to record the kwargs it was called
    with, while still delegating to the real forward (Qwen3's forward accepts
    arbitrary ``**kwargs``, so passing e.g. ``pixel_values`` does not error)."""
    received = {}
    original_forward = target.forward

    def _recording_forward(**kwargs):
        received.update(kwargs)
        return original_forward(**kwargs)

    target.forward = _recording_forward
    return received


def test_generate_batch_forwards_multimodal_kwargs_when_present():
    target = _tiny_target()
    received = _spy_on_forward(target)
    wrapper = HFDSparkTargetModel(target, target_layer_ids=[-1])

    input_ids = torch.randint(0, VOCAB, (1, 8))
    attention_mask = torch.ones_like(input_ids)
    loss_mask = torch.ones_like(input_ids, dtype=torch.uint8)
    pixel_values = torch.randn(2, 3, 4, 4)
    image_grid_thw = torch.tensor([[1, 2, 2]])

    wrapper.generate_batch(
        input_ids,
        attention_mask,
        loss_mask,
        pixel_values=pixel_values,
        image_grid_thw=image_grid_thw,
    )

    assert received["pixel_values"] is pixel_values
    assert received["image_grid_thw"] is image_grid_thw
    assert "pixel_values_videos" not in received
    assert "video_grid_thw" not in received


def test_generate_batch_backward_compatible_without_multimodal_kwargs():
    target = _tiny_target()
    received = _spy_on_forward(target)
    wrapper = HFDSparkTargetModel(target, target_layer_ids=[-1])

    input_ids = torch.randint(0, VOCAB, (1, 8))
    wrapper.generate_batch(input_ids, torch.ones_like(input_ids), torch.ones_like(input_ids, dtype=torch.uint8))

    # None of the multimodal keys are ever added -- not even as `None` -- so a
    # plain text-only target's forward call is byte-for-byte what it was before.
    for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
        assert key not in received
    assert set(received) == {"input_ids", "attention_mask", "use_cache"}


def test_validate_target_layer_ids_warns_on_sglang_unservable_ids(caplog):
    # -1 (embedding) and the last layer (post-norm) have no standard-SGLang aux
    # capture point, so validate_target_layer_ids warns (but still accepts them,
    # since AutoModel's own spec_generate serves them).
    with caplog.at_level(logging.WARNING, logger="nemo_automodel.components.speculative.dspark.common"):
        validate_target_layer_ids([-1, 1, NUM_LAYERS - 1], NUM_LAYERS)
    assert "SGLang" in caplog.text
    assert str(NUM_LAYERS - 1) in caplog.text
    # The ids are still returned unchanged (warning, not error).
    assert validate_target_layer_ids([-1, 1, NUM_LAYERS - 1], NUM_LAYERS) == [-1, 1, NUM_LAYERS - 1]


def test_validate_target_layer_ids_no_warn_within_range(caplog):
    # Ids within [0, N-2] are captured identically by AutoModel and SGLang -> no warning.
    with caplog.at_level(logging.WARNING, logger="nemo_automodel.components.speculative.dspark.common"):
        validate_target_layer_ids([0, 1, NUM_LAYERS - 2], NUM_LAYERS)
    assert "SGLang" not in caplog.text
