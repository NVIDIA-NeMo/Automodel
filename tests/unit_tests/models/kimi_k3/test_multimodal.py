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

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig, KimiK3VisionConfig
from nemo_automodel.components.models.kimi_k3.cp import shard_batch_for_kimi_vlm_cp
from nemo_automodel.components.models.kimi_k3.multimodal import (
    KimiK3ForConditionalGeneration,
    _positions_within_documents,
)
from nemo_automodel.components.models.kimi_k3.vision import (
    Learnable2DInterpPosEmbDivided_fixed,
    get_1d_sincos_pos_embed,
)


def _tiny_vlm(
    *,
    vocab_size: int = 64,
    media_placeholder_token_id: int = 63,
    pad_token_id: int = 0,
    patch_size: int = 2,
    init_pos_emb_time: int = 1,
) -> KimiK3ForConditionalGeneration:
    text_config = KimiK3TextConfig(
        vocab_size=vocab_size,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=2,
        num_experts_per_token=1,
        num_shared_experts=0,
        first_k_dense_replace=2,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=None,
    )
    vision_config = KimiK3VisionConfig(
        patch_size=patch_size,
        init_pos_emb_height=4,
        init_pos_emb_width=2,
        init_pos_emb_time=init_pos_emb_time,
        vt_num_attention_heads=2,
        vt_num_hidden_layers=1,
        vt_hidden_size=16,
        vt_intermediate_size=32,
        merge_kernel_size=(2, 2),
        attn_implementation="eager",
        mm_projector_type="patchmergerv2",
        mm_hidden_size=16,
        qkv_hidden_size=16,
        norm_type="rmsnorm",
        text_hidden_size=text_config.hidden_size,
    )
    config = KimiK3Config(
        text_config=text_config,
        vision_config=vision_config,
        media_placeholder_token_id=media_placeholder_token_id,
        pad_token_id=pad_token_id,
    )
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    model = KimiK3ForConditionalGeneration(config, backend=backend)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model.eval()


def test_k3_vlm_forward_expands_image_placeholder():
    torch.manual_seed(23)
    model = _tiny_vlm()
    input_ids = torch.tensor([[1, 63, 2, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0]])
    labels = torch.tensor([[1, -100, 2, -100]])
    pixel_values = torch.randn(8, 3, 2, 2)
    grid_thws = torch.tensor([[1, 4, 2]])

    with torch.no_grad():
        image_features = model._extract_image_features(pixel_values, grid_thws)
        inputs_embeds = model.get_input_embeddings()(input_ids)
        merged_embeddings, merged_attention_mask, merged_labels, position_ids = (
            model._merge_input_ids_with_image_features(
                image_features,
                inputs_embeds,
                input_ids,
                attention_mask,
                labels,
            )
        )
        output = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            attention_mask=attention_mask,
            labels=labels,
        )

    assert [features.shape for features in image_features] == [(2, 32)]
    assert merged_embeddings.shape == (1, 5, 32)
    assert merged_attention_mask.tolist() == [[1, 1, 1, 1, 0]]
    assert merged_labels is not None
    assert merged_labels.tolist() == [[1, -100, -100, 2, -100]]
    assert position_ids.tolist() == [[0, 1, 2, 3, 1]]
    torch.testing.assert_close(merged_embeddings[:, 1:3], image_features[0].unsqueeze(0))
    assert output.logits.shape == (1, 5, 64)
    assert torch.isfinite(output.logits).all()
    assert output.loss is not None and torch.isfinite(output.loss)


def test_k3_vlm_rejects_placeholder_feature_count_mismatch():
    model = _tiny_vlm()
    input_ids = torch.tensor([[1, 63, 2]])
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_features = [torch.randn(2, 32), torch.randn(1, 32)]

    with pytest.raises(ValueError, match="Received 1 image placeholders for 2 image feature groups"):
        model._merge_input_ids_with_image_features(
            image_features,
            inputs_embeds,
            input_ids,
            torch.ones_like(input_ids),
        )


def test_preexpanded_image_tokens_match_hf_style_expansion_and_gradients():
    torch.manual_seed(24)
    model = _tiny_vlm()
    pixels = torch.randn(8, 3, 2, 2)
    grid = torch.tensor([[1, 4, 2]])
    compact = model(
        input_ids=torch.tensor([[1, 63, 2]]),
        attention_mask=torch.ones(1, 3, dtype=torch.long),
        pixel_values=pixels,
        grid_thws=grid,
    ).logits
    compact.square().mean().backward()
    expected_grads = {
        name: parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if name.startswith(("vision_tower.", "mm_projector.")) and parameter.grad is not None
    }
    model.zero_grad(set_to_none=True)
    expanded = model(
        input_ids=torch.tensor([[1, 63, 63, 2]]),
        attention_mask=torch.ones(1, 4, dtype=torch.long),
        pixel_values=pixels,
        image_grid_hws=grid[:, 1:],
    ).logits
    torch.testing.assert_close(expanded, compact, atol=1e-6, rtol=1e-5)
    expanded.square().mean().backward()
    assert expected_grads
    for name, parameter in model.named_parameters():
        if name in expected_grads:
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
            torch.testing.assert_close(parameter.grad, expected_grads[name], atol=1e-6, rtol=1e-5)


def test_vision_initialization_after_meta_materialization():
    model = _tiny_vlm()
    model.to_empty(device="cpu")
    for parameter in model.parameters():
        with torch.no_grad():
            parameter.fill_(float("nan"))
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), name
    output = model(
        input_ids=torch.tensor([[1, 63, 2]]),
        pixel_values=torch.randn(8, 3, 2, 2),
        grid_thws=torch.tensor([[1, 4, 2]]),
    ).logits
    assert torch.isfinite(output).all()


def _independent_sincos(dim: int, frames: int) -> torch.Tensor:
    """Return the configuration-derived temporal table [frames, 1, dim] in FP64."""
    omega = 1.0 / 10000 ** (torch.arange(dim // 2, dtype=torch.float64) / (dim / 2))
    angles = torch.arange(frames, dtype=torch.float64)[:, None] * omega[None, :]
    return torch.cat((angles.sin(), angles.cos()), dim=-1).unsqueeze(1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_two_frame_temporal_positions_follow_tower_dtype_after_initialization(dtype: torch.dtype) -> None:
    torch.manual_seed(25)
    model = _tiny_vlm(init_pos_emb_time=2)
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=dtype)
    (pos_emb,) = [module for module in model.modules() if isinstance(module, Learnable2DInterpPosEmbDivided_fixed)]
    assert pos_emb.time_weight.dtype == pos_emb.weight.dtype == dtype
    assert pos_emb.time_weight.device == pos_emb.weight.device
    expected = _independent_sincos(pos_emb.dim, 2)
    if dtype == torch.float32:
        torch.testing.assert_close(pos_emb.time_weight.double(), expected, atol=1e-7, rtol=0)
    else:
        torch.testing.assert_close(pos_emb.time_weight, expected.to(dtype), atol=0, rtol=0)
    # The production table also matches the independently derived one.
    torch.testing.assert_close(
        torch.from_numpy(get_1d_sincos_pos_embed(pos_emb.dim, 2)).double().unsqueeze(1), expected, atol=1e-7, rtol=0
    )
    pixels = torch.randn(16, 3, 2, 2, dtype=dtype, requires_grad=True)
    output = model(
        input_ids=torch.tensor([[1, 63, 2]]),
        pixel_values=pixels,
        grid_thws=torch.tensor([[2, 4, 2]]),
    ).logits
    assert output.dtype == dtype
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all() and pixels.grad.abs().sum() > 0


@pytest.mark.parametrize("left_padding", [False, True])
def test_compact_ragged_images_keep_text_and_padding_order(left_padding):
    model = _tiny_vlm()
    ids = torch.tensor([[0, 1, 63, 2], [1, 63, 3, 4]]) if left_padding else torch.tensor([[1, 63, 2, 0], [1, 63, 3, 4]])
    mask = (ids != 0).long()
    embeds = model.get_input_embeddings()(ids)
    features = [torch.full((2, 32), 7.0), torch.full((4, 32), 9.0)]
    merged, final_mask, labels, positions = model._merge_input_ids_with_image_features(
        features, embeds, ids, mask, ids.masked_fill(mask == 0, -100)
    )
    expected_mask = [[0, 0, 0, 1, 1, 1, 1], [1] * 7] if left_padding else [[1, 1, 1, 1, 0, 0, 0], [1] * 7]
    assert final_mask.tolist() == expected_mask
    for row, feature in enumerate(features):
        expected = torch.cat(
            (embeds[row, ids[row] == 1], feature, embeds[row, (ids[row] != 0) & (ids[row] != 1) & (ids[row] != 63)])
        )
        actual = merged[row, final_mask[row].bool()]
        torch.testing.assert_close(actual, expected)
        assert positions[row, final_mask[row].bool()].tolist() == list(range(len(expected)))
    assert (labels[final_mask == 0] == -100).all()


def _packed_case():
    """Three documents: image, text-only, and two images, plus tail padding."""
    ids = torch.tensor([[1, 63, 63, 2, 3, 4, 5, 6, 63, 63, 7, 63, 8, 9, 0, 0]])
    docs = torch.tensor([[1] * 5 + [2] * 2 + [3] * 7 + [0] * 2])
    positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 0, 1, 2, 3, 4, 5, 6, 0, 0]])
    return dict(
        input_ids=ids,
        attention_mask=docs,
        position_ids=positions,
        pixel_values=torch.randn(20, 3, 2, 2),
        grid_thws=torch.tensor([[1, 4, 2], [1, 4, 2], [1, 2, 2]]),
    )


def test_packed_images_match_independent_documents_and_isolate_gradients():
    torch.manual_seed(31)
    model = _tiny_vlm()
    packed = _packed_case()
    actual = model(**packed).logits
    separate = []
    for start, end, image_start, image_end in [(0, 5, 0, 8), (5, 7, 8, 8), (7, 14, 8, 20)]:
        batch = dict(
            input_ids=packed["input_ids"][:, start:end],
            attention_mask=torch.ones(1, end - start, dtype=torch.long),
            position_ids=torch.arange(end - start).unsqueeze(0),
        )
        if image_start != image_end:
            batch["pixel_values"] = packed["pixel_values"][image_start:image_end]
            batch["grid_thws"] = packed["grid_thws"][:1] if start == 0 else packed["grid_thws"][1:]
        separate.append(model(**batch).logits)
    torch.testing.assert_close(actual[:, :14], torch.cat(separate, dim=1), atol=2e-6, rtol=2e-5)
    changed = {**packed, "pixel_values": packed["pixel_values"].clone()}
    changed["pixel_values"][:8] = torch.randn_like(changed["pixel_values"][:8]) * 100
    changed["input_ids"] = packed["input_ids"].clone()
    changed["input_ids"][:, 0] = 27
    perturbed = model(**changed).logits
    torch.testing.assert_close(actual[:, 5:14], perturbed[:, 5:14], atol=0, rtol=0)
    # A deliberately erased boundary must make the perturbation observable.
    broken = {**packed, "attention_mask": (packed["attention_mask"] > 0).long()}
    broken_changed = {**changed, "attention_mask": broken["attention_mask"]}
    assert (model(**broken).logits[:, 7:14] - model(**broken_changed).logits[:, 7:14]).abs().max() > 1e-5
    pixels = packed["pixel_values"].clone().requires_grad_()
    logits = model(**{**packed, "pixel_values": pixels}).logits
    logits[:, 7:14].square().sum().backward()
    assert torch.count_nonzero(pixels.grad[:8]) == 0
    assert pixels.grad[8:].abs().sum() > 0


def test_vlm_cp_keeps_full_image_stream_and_shards_auxiliary_tokens():
    from tests.unit_tests.models.kimi_k3.test_context_parallel import _FakeCPMesh

    model = _tiny_vlm()
    model.cp_mesh = _FakeCPMesh()
    batch = _packed_case()
    batch["labels"] = batch["input_ids"].clone()
    original_ids = batch["input_ids"].clone()
    sharder = model.prepare_model_inputs_for_cp(batch)["cp_sharder"]
    assert sharder.shard_batch is shard_batch_for_kimi_vlm_cp
    _, local, layout = sharder.shard_batch(model.cp_mesh, None, batch)
    assert local["input_ids"].shape == (1, 16)
    torch.testing.assert_close(local["input_ids"], original_ids)
    assert local["position_ids"].tolist() == [[1, 2, 3, 4, 5, 6, 0, 0]]
    assert local["labels"].shape == (1, 8)
    assert local["kimi_packed_doc_ids"].tolist() == [[1] * 5 + [2] * 2 + [3] * 7 + [0] * 2]
    assert "kimi_vlm_token_order" not in local
    assert layout.padded_seq_len == 16


@pytest.mark.parametrize("fault", ["document", "padding", "text_gap", "row"])
def test_image_span_cannot_cross_ownership_boundary(fault):
    model = _tiny_vlm()
    ids = torch.tensor([[1, 63, 63, 2]])
    docs = torch.ones_like(ids)
    if fault == "document":
        docs[0, 2:] = 2
    elif fault == "padding":
        docs[0, 2] = 0
    elif fault == "text_gap":
        ids = torch.tensor([[1, 63, 2, 63]])
    else:
        ids = torch.tensor([[1, 63], [63, 2]])
        docs = torch.ones_like(ids)
    with pytest.raises(ValueError, match="image token span crosses"):
        model(
            input_ids=ids,
            attention_mask=docs,
            pixel_values=torch.randn(8, 3, 2, 2),
            grid_thws=torch.tensor([[1, 4, 2]]),
        )


def test_pipeline_media_cursor_advances_over_text_only_microbatch():
    model = _tiny_vlm()
    model._vlm_pixel_values_chunks = [torch.empty(0, 3, 2, 2), torch.randn(8, 3, 2, 2)]
    model._vlm_image_grid_hws_chunks = [torch.empty(0, 2, dtype=torch.long), torch.tensor([[4, 2]])]
    model._vlm_chunk_idx = 0
    model(input_ids=torch.tensor([[1, 2, 3]]))
    assert model._vlm_chunk_idx == 1
    output = model(input_ids=torch.tensor([[1, 63, 63, 2]]))
    assert model._vlm_chunk_idx == 2
    assert torch.isfinite(output.logits).all()


@pytest.mark.parametrize("expanded", [False, True])
def test_merge_preserves_document_positions(expanded: bool) -> None:
    model = _tiny_vlm()
    image_tokens = [63, 63] if expanded else [63]
    ids = torch.tensor([[1, *image_tokens, 2, 3, 4, 0]])
    mask = torch.tensor([[1] * (len(image_tokens) + 2) + [2, 2, 0]])
    _, merged_mask, _, positions = model._merge_input_ids_with_image_features(
        [torch.randn(2, 32)], model.get_input_embeddings()(ids), ids, mask
    )
    assert merged_mask.tolist() == [[1, 1, 1, 1, 2, 2, 0]]
    assert positions.tolist() == [[0, 1, 2, 3, 0, 1, 1]]


@pytest.mark.parametrize("is_first", [False, True])
def test_vlm_pipeline_metadata_retains_full_input_and_shards_outputs(is_first: bool) -> None:
    from unittest.mock import Mock

    from torch.distributed.device_mesh import DeviceMesh

    model = _tiny_vlm()
    model.cp_mesh = Mock(spec=DeviceMesh)
    model.cp_mesh.size.return_value = 2
    inputs, outputs = model.get_pipeline_stage_metas(
        is_first=is_first, microbatch_size=1, seq_len=287, dtype=torch.bfloat16
    )
    assert inputs[0].shape == ((1, 287) if is_first else (1, 144, 32))
    assert outputs[0].shape == (1, 144, 64)


@pytest.mark.parametrize("expanded", [False, True])
def test_masked_fake_image_span_stays_masked_and_skips_positions(expanded: bool) -> None:
    """Fake-image injection masks whole image spans; they must stay masked holes."""
    model = _tiny_vlm()
    image_tokens = [63, 63] if expanded else [63]
    ids = torch.tensor([[5, *image_tokens, 7, 8, 0]])
    mask = torch.tensor([[1, *[0] * len(image_tokens), 1, 1, 0]])
    labels = torch.tensor([[-100, *[-100] * len(image_tokens), 7, 8, -100]])
    features = [torch.randn(2, 32)]
    embeddings, merged_mask, merged_labels, positions = model._merge_input_ids_with_image_features(
        features, model.get_input_embeddings()(ids), ids, mask, labels
    )
    assert merged_mask.tolist() == [[1, 0, 0, 1, 1, 0]]
    assert merged_labels is not None
    assert merged_labels.tolist() == [[-100, -100, -100, 7, 8, -100]]
    # Masked image tokens neither advance positions nor start a new document.
    assert positions.tolist() == [[0, 1, 1, 1, 2, 1]]
    torch.testing.assert_close(embeddings[0, 1:3], features[0])


def test_positions_skip_masked_holes_within_documents() -> None:
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 0], [0, 0, 1, 2, 0, 2, 2], [1, 2, 2, 0, 3, 3, 0]])
    assert _positions_within_documents(mask).tolist() == [
        [0, 1, 1, 1, 2, 3, 1],
        [1, 1, 0, 0, 1, 1, 2],
        [0, 0, 1, 1, 0, 1, 1],
    ]


def test_packed_document_ids_cannot_reappear_after_another_document() -> None:
    from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext

    with pytest.raises(ValueError, match="contiguous"):
        KimiPackedContext(doc_ids=torch.tensor([[1, 1, 2, 2, 1, 1, 0]]))


def test_k3_vlm_staged_media_matches_direct_after_text_only_microbatch():
    model = _tiny_vlm()
    text_ids = torch.tensor([[1, 2, 3]])
    image_ids = torch.tensor([[1, 63, 2]])
    pixels = torch.randn(8, 3, 2, 2)
    grid = torch.tensor([[1, 4, 2]])
    with torch.no_grad():
        direct_text = model(text_ids)
        direct_image = model(image_ids, pixel_values=pixels, grid_thws=grid)
        model._vlm_pixel_values_chunks = [pixels[:0], pixels]
        model._vlm_image_grid_hws_chunks = [grid[:0], grid[:, 1:]]
        model._vlm_chunk_idx = 0
        staged_text = model(text_ids)
        assert model._vlm_chunk_idx == 1
        staged_image = model(image_ids)
    assert model._vlm_chunk_idx == 2
    torch.testing.assert_close(staged_text.logits, direct_text.logits)
    torch.testing.assert_close(staged_image.logits, direct_image.logits)
