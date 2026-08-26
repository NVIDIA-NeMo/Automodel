# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
)
from tests.unit_tests.models.glm5_next.conftest import tiny_glm5_next_model


def test_tiny_hybrid_packed_forward_backward_is_finite():
    torch.manual_seed(7)
    model = tiny_glm5_next_model().train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    document_ids = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32)

    logits = model(input_ids=input_ids, attention_mask=document_ids).logits
    logits.square().mean().backward()

    assert logits.shape == (1, 6, 64)
    assert torch.isfinite(logits).all()
    assert model.model.language_model.layers["0"].self_attn.q_proj.weight.grad is not None
    assert model.model.language_model.layers["3"].self_attn.q_a_proj.weight.grad is not None


def test_text_config_exposes_hidden_states_for_fused_linear_ce():
    model = tiny_glm5_next_model().eval()
    model.config.text_config.output_hidden_states = True

    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1, 2, 3]]), logits_to_keep=1)

    assert output.logits.shape == (1, 1, 64)
    assert output.hidden_states.shape == (1, 3, 16)


def test_packed_documents_are_attention_isolated():
    torch.manual_seed(11)
    model = tiny_glm5_next_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = torch.tensor([[1, 2, 3, 13, 14, 15]])
    document_ids = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32)

    with torch.inference_mode():
        baseline = model(input_ids=input_ids, attention_mask=document_ids).logits
        perturbed = model(input_ids=changed, attention_mask=document_ids).logits

    torch.testing.assert_close(baseline[:, :3], perturbed[:, :3], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(baseline[:, 3:], perturbed[:, 3:])


def test_sparse_indexer_prepares_document_pools_once_across_query_chunks():
    model = tiny_glm5_next_model().eval()
    sparse = model.model.language_model.layers["3"].self_attn
    sparse.query_chunk_size = 2
    calls = 0

    def count_key_projection(_module, _inputs, _output):
        nonlocal calls
        calls += 1

    handle = sparse.indexer.wk.register_forward_hook(count_key_projection)
    try:
        with torch.inference_mode():
            output = sparse._forward_document(torch.randn(1, 6, 16), 0, 6)
    finally:
        handle.remove()

    assert output.shape == (1, 6, 16)
    assert calls == 1


def test_image_features_replace_exactly_the_placeholder_tokens():
    torch.manual_seed(17)
    model = tiny_glm5_next_model().eval()
    input_ids = torch.tensor([[1, 63, 2, 3]])
    pixel_values = torch.randn(4, 3 * 2 * 2 * 2)
    grid_thw = torch.tensor([[1, 2, 2]])

    with torch.inference_mode():
        features = model.get_image_features(pixel_values, grid_thw).pooler_output
        embeddings = model._embed_and_splice(input_ids, pixel_values, grid_thw)
        logits = model(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
        ).logits

    assert features.shape == (1, 16)
    torch.testing.assert_close(embeddings[:, 1], features)
    assert logits.shape == (1, 4, 64)
    assert torch.isfinite(logits).all()


def test_hybrid_layer_pattern_and_parallel_capabilities():
    model = tiny_glm5_next_model()
    layers = model.model.language_model.layers
    assert all(isinstance(layers[str(index)].self_attn, Glm5NextLinearAttention) for index in range(3))
    assert isinstance(layers["3"].self_attn, Glm5NextSparseAttention)

    supports = ModelSupports(model, None)
    assert supports.supports_cp
    assert supports.supports_sequence_packing
    assert supports.supports_cp_with_sequence_packing


def test_hyperconnection_fp32_parameters_have_a_dedicated_fsdp_holder():
    hyperconnection = tiny_glm5_next_model().model.language_model.layers["0"].attn_hc

    assert set(dict(hyperconnection.named_parameters(recurse=False))) == {"fn"}
    assert set(dict(hyperconnection._fp32_params.named_parameters())) == {"base", "scale"}
    assert all(parameter.dtype is torch.float32 for parameter in hyperconnection._fp32_params.parameters())
