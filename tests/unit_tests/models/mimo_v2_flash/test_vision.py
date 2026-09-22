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

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import patch

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mimo_v2_flash.config import MiMoV2Config
from nemo_automodel.components.models.mimo_v2_flash.cp import (
    _MIMO_GLOBAL_IMAGE_MASK,
    _MIMO_GLOBAL_VIDEO_MASK,
    _MIMO_THD_LOCAL_INDICES,
    shard_batch_for_mimo_te,
)
from nemo_automodel.components.models.mimo_v2_flash.model import MiMoV2ForCausalLM
from nemo_automodel.components.models.mimo_v2_flash.vision import (
    MiMoVisionRotaryEmbedding,
    MiMoVisionTransformer,
)


def _vision_config() -> dict:
    return {
        "depth": 2,
        "fullatt_block_indexes": [0],
        "hidden_act": "silu",
        "hidden_size": 8,
        "in_chans": 3,
        "intermediate_size": 16,
        "num_heads": 2,
        "num_key_value_heads": 1,
        "qk_channels": 4,
        "out_hidden_size": 16,
        "patch_size": 2,
        "spatial_merge_size": 2,
        "temporal_patch_size": 1,
        "use_sink": True,
        "visual_token_window_size": 2,
        "vit_window_attn_types": [-1, 0],
    }


def _model_config() -> MiMoV2Config:
    return MiMoV2Config(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        v_head_dim=4,
        swa_num_attention_heads=4,
        swa_num_key_value_heads=2,
        swa_head_dim=4,
        swa_v_head_dim=4,
        hybrid_layer_pattern=[0],
        moe_layer_freq=[0],
        n_routed_experts=None,
        partial_rotary_factor=1.0,
        torch_dtype="float32",
        vision_config=_vision_config(),
        image_token_id=62,
        video_token_id=63,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )


def test_vision_checkpoint_names_and_shapes():
    vision = MiMoVisionTransformer(_vision_config(), dtype=torch.float32)
    state = vision.state_dict()

    assert len(state) == 29
    assert state["patch_embed.proj.weight"].shape == (8, 3, 1, 2, 2)
    assert state["blocks.0.attn.qkv.weight"].shape == (16, 8)
    assert state["blocks.0.attn.proj.weight"].shape == (8, 8)
    assert "blocks.0.attn.sinks" not in state
    assert state["blocks.1.attn.sinks"].shape == (2,)
    assert state["merger.mlp.0.weight"].shape == (32, 32)
    assert state["merger.mlp.2.weight"].shape == (16, 32)
    assert "merger.ln_q.bias" not in state
    assert "merger.mlp.0.bias" not in state
    assert "merger.mlp.2.bias" not in state


def test_vision_forward_merges_each_spatial_group():
    torch.manual_seed(7)
    vision = MiMoVisionTransformer(_vision_config(), dtype=torch.float32)
    grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])
    pixel_values = torch.randn(8, 3 * 1 * 2 * 2)

    output = vision(pixel_values, grid_thw)

    assert output.shape == (2, 16)
    assert torch.isfinite(output).all()


def test_vision_rotary_rebuilds_meta_buffer_on_materialized_device():
    with torch.device("meta"):
        rotary = MiMoVisionRotaryEmbedding(4)
    rotary.to_empty(device="cpu")
    rotary.inv_freq.fill_(float("nan"))

    output = rotary(3, device=torch.device("cpu"))

    expected_inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 4, 2, dtype=torch.float32) / 4))
    torch.testing.assert_close(rotary.inv_freq, expected_inv_freq)
    torch.testing.assert_close(output, torch.outer(torch.arange(3, dtype=torch.float32), expected_inv_freq))


def test_multimodal_embeddings_replace_processor_token_slots():
    model = MiMoV2ForCausalLM(_model_config(), backend=_backend())
    input_ids = torch.tensor([[3, 62, 4, 63]])
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_embed = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    video_embed = -image_embed

    output = model._get_multimodal_embeds(
        input_ids,
        inputs_embeds,
        image_embeds=image_embed,
        video_embeds=video_embed,
    )

    torch.testing.assert_close(output[0, 1], image_embed[0].to(output.dtype))
    torch.testing.assert_close(output[0, 3], video_embed[0].to(output.dtype))
    torch.testing.assert_close(output[0, 0], inputs_embeds[0, 0])


def test_causal_lm_forward_accepts_qwen_processor_fields():
    torch.manual_seed(11)
    model = MiMoV2ForCausalLM(_model_config(), backend=_backend()).eval()
    input_ids = torch.tensor([[3, 62, 4]])
    pixel_values = torch.randn(4, 3 * 1 * 2 * 2)

    output = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        image_grid_thw=torch.tensor([[1, 2, 2]]),
    )

    assert output.logits.shape == (1, 3, 64)
    assert torch.isfinite(output.logits).all()


def test_text_only_forward_enters_inner_root_before_embedding():
    model = MiMoV2ForCausalLM(_model_config(), backend=_backend()).eval()
    entered_inner_root = False

    def mark_inner_root(_module, _args):
        nonlocal entered_inner_root
        entered_inner_root = True

    def check_embedding_order(_module, _args):
        assert entered_inner_root

    root_handle = model.model.register_forward_pre_hook(mark_inner_root)
    embedding_handle = model.model.embed_tokens.register_forward_pre_hook(check_embedding_order)
    try:
        output = model(input_ids=torch.tensor([[3, 4, 5]]))
    finally:
        root_handle.remove()
        embedding_handle.remove()

    assert output.logits.shape == (1, 3, 64)


def test_pipeline_hook_keeps_top_level_visual_only_on_stage_zero():
    model = MiMoV2ForCausalLM(_model_config(), backend=_backend())
    stages = [
        ["model.embed_tokens", "model.layers.0", "model.visual", "model.rotary_emb"],
        ["model.norm", "lm_head", "model.visual", "model.rotary_emb"],
    ]

    output = model.customize_pipeline_stage_modules(stages, layers_prefix="model.")

    assert "visual" in output[0]
    assert "model.visual" not in output[0]
    assert "visual" not in output[1]
    assert "model.visual" not in output[1]
    assert "model.swa_rotary_emb" in output[0]
    assert "model.swa_rotary_emb" in output[1]


def test_pipeline_media_side_channel_accepts_hw_grid_alias():
    model = MiMoV2ForCausalLM(_model_config(), backend=_backend())
    pixels = torch.randn(4, 12)
    model._vlm_pixel_values_chunks = [pixels]
    model._vlm_image_grid_hws_chunks = [torch.tensor([[2, 2]])]
    model._vlm_chunk_idx = 0

    pixel_values, grid_thw, pixel_values_videos, video_grid_thw = model._pull_pipeline_media(
        torch.tensor([[62, 1]]), None, None, None, None
    )

    assert pixel_values is pixels
    torch.testing.assert_close(grid_thw, torch.tensor([[1, 2, 2]]))
    assert pixel_values_videos is None
    assert video_grid_thw is None
    assert model._vlm_chunk_idx == 1


def test_vlm_te_cp_maps_global_media_to_local_dual_chunk_tokens():
    class FakeMesh:
        def size(self):
            return 2

        def get_group(self):
            return object()

        def get_local_rank(self):
            return 1

    input_ids = torch.tensor([[62, 63, 7, 62, 8, 63, 63, 62]])
    batch = {
        "input_ids": input_ids.clone(),
        "labels": input_ids.clone(),
        "position_ids": torch.arange(input_ids.shape[1]).unsqueeze(0),
        "seq_lens": torch.tensor([[8]]),
        "seq_lens_padded": torch.tensor([[8]]),
        "qkv_format": "thd",
    }
    local_indices = torch.tensor([2, 3, 4, 5])
    fake_tex = SimpleNamespace(
        thd_get_partitioned_indices=lambda _cu, _tokens, _size, _rank: local_indices,
    )

    with (
        patch.dict(sys.modules, {"transformer_engine_torch": fake_tex}),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        _, local_batch, layout = shard_batch_for_mimo_te(
            FakeMesh(),
            None,
            batch,
            image_token_id=62,
            video_token_id=63,
        )

    torch.testing.assert_close(local_batch["input_ids"], torch.tensor([7, 62, 8, 63]))
    torch.testing.assert_close(local_batch[_MIMO_THD_LOCAL_INDICES], local_indices)
    torch.testing.assert_close(local_batch[_MIMO_GLOBAL_IMAGE_MASK], input_ids.reshape(-1).eq(62))
    torch.testing.assert_close(local_batch[_MIMO_GLOBAL_VIDEO_MASK], input_ids.reshape(-1).eq(63))
    torch.testing.assert_close(layout.local_token_global_indices, local_indices)

    model = MiMoV2ForCausalLM(_model_config(), backend=_backend())
    inputs_embeds = model.get_input_embeddings()(local_batch["input_ids"])
    image_embeds = torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16)
    video_embeds = -torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16)
    image_feature_indices = model._local_modal_feature_indices(
        local_batch[_MIMO_GLOBAL_IMAGE_MASK],
        local_batch[_MIMO_THD_LOCAL_INDICES],
    )
    video_feature_indices = model._local_modal_feature_indices(
        local_batch[_MIMO_GLOBAL_VIDEO_MASK],
        local_batch[_MIMO_THD_LOCAL_INDICES],
    )

    output = model._get_multimodal_embeds(
        local_batch["input_ids"],
        inputs_embeds,
        image_embeds=image_embeds,
        image_feature_indices=image_feature_indices,
        video_embeds=video_embeds,
        video_feature_indices=video_feature_indices,
    )

    torch.testing.assert_close(image_feature_indices, torch.tensor([1]))
    torch.testing.assert_close(video_feature_indices, torch.tensor([1]))
    torch.testing.assert_close(output[1], image_embeds[1].to(output.dtype))
    torch.testing.assert_close(output[3], video_embeds[1].to(output.dtype))
    torch.testing.assert_close(output[[0, 2]], inputs_embeds[[0, 2]])


def test_pp_cp_global_masks_keep_vision_execution_and_local_mapping_in_lockstep(monkeypatch):
    global_input_ids = torch.tensor([[62, 1, 63, 3, 4, 5, 6, 7]])
    global_image_mask = global_input_ids.eq(62)
    global_video_mask = global_input_ids.eq(63)
    image_pixels = torch.randn(4, 12)
    video_pixels = torch.randn(4, 12)
    image_grid = torch.tensor([[1, 2, 2]])
    video_grid = torch.tensor([[1, 2, 2]])
    image_feature = torch.arange(16, dtype=torch.float32).unsqueeze(0)
    video_feature = -torch.arange(1, 17, dtype=torch.float32).unsqueeze(0)

    # These are TE's two CP=2 DualChunkSwap partitions. Only rank 0 owns the
    # image placeholder and only rank 1 owns the video placeholder.
    rank_local_indices = (
        torch.tensor([[0, 1, 6, 7]]),
        torch.tensor([[2, 3, 4, 5]]),
    )
    for local_indices in rank_local_indices:
        model = MiMoV2ForCausalLM(_model_config(), backend=_backend())
        model._pp_return_hidden_states = True
        model._vlm_pixel_values_chunks = [image_pixels]
        model._vlm_image_grid_hws_chunks = [image_grid]
        model._vlm_pixel_values_videos_chunks = [video_pixels]
        model._vlm_video_grid_thw_chunks = [video_grid]
        model._vlm_chunk_idx = 0

        visual_inputs = []

        def fake_visual(pixel_values, _grid_thw):
            visual_inputs.append(pixel_values)
            if pixel_values is image_pixels:
                return image_feature
            assert pixel_values is video_pixels
            return video_feature

        def text_passthrough(*, input_ids, inputs_embeds, **_kwargs):
            assert input_ids is None
            assert inputs_embeds is not None
            return inputs_embeds

        monkeypatch.setattr(model.visual, "forward", fake_visual)
        monkeypatch.setattr(model.model, "forward", text_passthrough)

        local_input_ids = global_input_ids.reshape(-1).index_select(0, local_indices.reshape(-1)).reshape(1, -1)
        original_embeds = model.get_input_embeddings()(local_input_ids).detach().clone()
        output = model(
            input_ids=local_input_ids,
            qkv_format="thd",
            cp_size=2,
            **{
                _MIMO_GLOBAL_IMAGE_MASK: global_image_mask,
                _MIMO_GLOBAL_VIDEO_MASK: global_video_mask,
                _MIMO_THD_LOCAL_INDICES: local_indices,
            },
        )

        assert len(visual_inputs) == 2
        assert visual_inputs[0] is image_pixels
        assert visual_inputs[1] is video_pixels
        assert model._vlm_chunk_idx == 1

        expected = original_embeds.clone()
        local_image_mask = local_input_ids.eq(62)
        local_video_mask = local_input_ids.eq(63)
        if bool(local_image_mask.any()):
            expected[local_image_mask] = image_feature.to(expected)
        if bool(local_video_mask.any()):
            expected[local_video_mask] = video_feature.to(expected)
        torch.testing.assert_close(output, expected)


def test_state_adapter_keeps_visual_weights_in_checkpoint_dtype():
    config = _model_config()
    backend = _backend()
    backend.enable_hf_state_dict_adapter = True
    model = MiMoV2ForCausalLM(config, backend=backend)
    weight = model.visual.blocks[0].attn.qkv.weight

    converted = model.state_dict_adapter.convert_single_tensor_to_hf(
        "visual.blocks.0.attn.qkv.weight",
        weight,
        quantization=True,
        for_checkpoint_load=True,
    )

    assert converted == [("visual.blocks.0.attn.qkv.weight", weight)]
    assert converted[0][1].dtype == torch.bfloat16
