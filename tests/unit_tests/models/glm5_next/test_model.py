# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.models.glm5_next import layers as glm5_next_layers
from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.layers import (
    Glm5NextLinearAttention,
    Glm5NextSparseAttention,
)
from nemo_automodel.components.models.glm5_next.model import build_glm5_next_moe_config
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config, tiny_glm5_next_model


def _torch_sparse_latent_attention(
    query: torch.Tensor,
    latent_kv: torch.Tensor,
    indices: torch.Tensor,
    softmax_scale: float,
    *,
    all_rows_nonempty: bool,
) -> torch.Tensor:
    """Reference absorbed sparse attention used by the GLM-5.3 dispatch test.

    Args:
        query: Absorbed query tensor with shape ``[queries, heads, latent_dim]``.
        latent_kv: Shared latent K/V tensor with shape ``[keys, 1, latent_dim]``.
        indices: Document-local sparse indices with shape
            ``[queries, 1, sparse_width]``.
        softmax_scale: Scale applied to query-key scores.
        all_rows_nonempty: Whether every query has at least one selected key.

    Returns:
        Latent attention output with shape ``[queries, heads, latent_dim]``.
    """
    assert all_rows_nonempty
    rows = []
    for row in range(query.shape[0]):
        selected = indices[row, 0]
        selected = selected[(selected >= 0) & (selected < latent_kv.shape[0])].unique(sorted=True)
        assert selected.numel() > 0
        keys = latent_kv.index_select(0, selected.long()).squeeze(1)
        scores = torch.matmul(query[row].float(), keys.float().transpose(0, 1)) * softmax_scale
        rows.append(torch.matmul(scores.softmax(dim=-1), keys.float()).to(query.dtype))
    return torch.stack(rows)


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


def test_torch_kda_decay_scales_the_key_dimension():
    q = torch.tensor([[[[0.0, 0.0]], [[1.0, 0.0]]]])
    k = torch.tensor([[[[1.0, 0.0]], [[0.0, 0.0]]]])
    v = torch.tensor([[[[1.0, 2.0]], [[0.0, 0.0]]]])
    g = torch.tensor([[[[1.0, 1.0]], [[0.5, 0.25]]]]).log()
    beta = torch.tensor([[[1.0], [0.0]]])

    output = glm5_next_layers._torch_recurrent_kda(q, k, v, g, beta, cu_seqlens=None)

    expected = torch.tensor([0.5, 1.0]) / torch.sqrt(torch.tensor(2.0))
    torch.testing.assert_close(output[0, 1, 0], expected)


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


@pytest.mark.parametrize(
    ("length", "query_start", "query_end"),
    [(1, 0, 1), (6, 0, 6), (69, 0, 69), (69, 5, 64)],
)
def test_cudnn_sparse_attention_matches_sdpa_absorbed_math(monkeypatch, length, query_start, query_end):
    """Check absorbed math, sparse tails and nonzero local query offsets."""
    torch.manual_seed(19)
    config = tiny_glm5_next_config().text_config
    config.kv_lora_rank = 512
    sdpa_backend = tiny_backend()
    cudnn_backend = tiny_backend()
    cudnn_backend.attn = "cudnn"
    sdpa = Glm5NextSparseAttention(config, layer_idx=3, backend=sdpa_backend)
    cudnn = Glm5NextSparseAttention(config, layer_idx=3, backend=cudnn_backend)
    sdpa.init_weights(torch.device("cpu"), init_std=0.02)
    cudnn.load_state_dict(sdpa.state_dict())

    monkeypatch.setattr(glm5_next_layers, "is_cudnn_sparse_attention_available", lambda: True)
    monkeypatch.setattr(glm5_next_layers, "cudnn_sparse_attention", _torch_sparse_latent_attention)

    cudnn.indexer_query_chunk_size = 17
    sdpa_input = torch.randn(1, length, config.hidden_size, requires_grad=True)
    cudnn_input = sdpa_input.detach().clone().requires_grad_(True)
    sdpa_output = sdpa._forward_document(sdpa_input, query_start, query_end)
    with (
        patch.object(cudnn.q_a_proj, "forward", wraps=cudnn.q_a_proj.forward) as q_a_projection,
        patch.object(cudnn.q_b_proj, "forward", wraps=cudnn.q_b_proj.forward) as q_b_projection,
        patch.object(cudnn.indexer, "select", wraps=cudnn.indexer.select) as selections,
    ):
        cudnn_output = cudnn._forward_document(cudnn_input, query_start, query_end)
    upstream = torch.randn_like(sdpa_output)
    (sdpa_output * upstream).sum().backward()
    (cudnn_output * upstream).sum().backward()

    torch.testing.assert_close(cudnn_output, sdpa_output, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(cudnn_input.grad, sdpa_input.grad, rtol=5e-4, atol=2e-6)
    for name, sdpa_parameter in sdpa.named_parameters():
        cudnn_parameter = dict(cudnn.named_parameters())[name]
        if name.startswith("indexer."):
            assert sdpa_parameter.grad is None
            assert cudnn_parameter.grad is None
        else:
            assert sdpa_parameter.grad is not None, name
            assert cudnn_parameter.grad is not None, name
            torch.testing.assert_close(cudnn_parameter.grad, sdpa_parameter.grad, rtol=5e-4, atol=2e-6)

    # The numerical assertions above stay valid for the old implementation.
    # These dispatch assertions detect a regression in the intended batching.
    assert q_b_projection.call_count == 1
    assert q_a_projection.call_count == 1
    positions = [call.args[2] for call in selections.call_args_list]
    assert max(position.numel() for position in positions) <= cudnn.indexer_query_chunk_size
    torch.testing.assert_close(torch.cat(positions), torch.arange(query_start, query_end))


def test_cudnn_sparse_attention_packed_documents_match_separate_execution(monkeypatch):
    """Packed cuDNN dispatch preserves each document's outputs and gradients."""
    torch.manual_seed(91)
    config = tiny_glm5_next_config().text_config
    config.kv_lora_rank = 512
    backend = tiny_backend()
    backend.attn = "cudnn"
    attention = Glm5NextSparseAttention(config, layer_idx=3, backend=backend)
    attention.init_weights(torch.device("cpu"), init_std=0.02)
    attention.indexer_query_chunk_size = 17
    monkeypatch.setattr(glm5_next_layers, "is_cudnn_sparse_attention_available", lambda: True)
    monkeypatch.setattr(glm5_next_layers, "cudnn_sparse_attention", _torch_sparse_latent_attention)
    lengths = [5, 69, 7]
    document_ids = torch.cat(
        [torch.full((length,), index + 1, dtype=torch.int32) for index, length in enumerate(lengths)]
    )
    context = Glm5NextPackedContext(doc_ids=document_ids.unsqueeze(0))
    packed_input = torch.randn(1, sum(lengths), config.hidden_size, requires_grad=True)
    separate_input = packed_input.detach().clone().requires_grad_(True)
    selected = []
    original_select = attention.indexer.select

    def record_selection(
        query_hidden: torch.Tensor,
        query_resid: torch.Tensor,
        query_positions: torch.Tensor,
        pool_keys: torch.Tensor,
        pool_indices: torch.Tensor,
        key_length: int,
    ) -> torch.Tensor:
        """Record actual document-local selections and check their boundaries.

        Args:
            query_hidden: Input states [1, queries, hidden].
            query_resid: Low-rank queries [1, queries, q_lora_rank].
            query_positions: Document-local query positions [queries].
            pool_keys: Compressed keys [complete_pools, index_head_dim].
            pool_indices: Raw document-local key positions [complete_pools, index_kpool].
            key_length: Unpadded document length.

        Returns:
            Selected raw document-local key indices [1, queries, sparse_width].
        """
        indices = original_select(query_hidden, query_resid, query_positions, pool_keys, pool_indices, key_length)
        assert (indices >= -1).all()
        assert (indices < key_length).all()
        assert (indices <= query_positions[None, :, None]).all()
        selected.append(indices.detach().clone())
        return indices

    monkeypatch.setattr(attention.indexer, "select", record_selection)
    packed_output = attention(packed_input, packed_context=context)
    packed_selected = torch.cat(selected, dim=1)
    selected.clear()
    separate_output = torch.cat(
        [
            attention._forward_document(document, 0, document.shape[1])
            for document in separate_input.split(lengths, dim=1)
        ],
        dim=1,
    )
    separate_selected = torch.cat(selected, dim=1)
    torch.testing.assert_close(packed_selected.sort(dim=-1).values, separate_selected.sort(dim=-1).values)
    local_positions = torch.cat([torch.arange(length) for length in lengths])
    assert ((packed_selected[0] >= 0).sum(-1) < local_positions + 1).any()
    upstream = torch.randn_like(packed_output)
    (packed_output * upstream).sum().backward()
    (separate_output * upstream).sum().backward()
    torch.testing.assert_close(packed_output, separate_output, rtol=2e-4, atol=2e-6)
    torch.testing.assert_close(packed_input.grad, separate_input.grad, rtol=5e-4, atol=2e-6)

    perturbed = packed_input.detach().clone()
    middle_document = slice(lengths[0], lengths[0] + lengths[1])
    perturbed[:, middle_document] += torch.randn_like(perturbed[:, middle_document])
    with torch.no_grad():
        changed_output = attention(perturbed, packed_context=context)
    torch.testing.assert_close(changed_output[:, : lengths[0]], packed_output[:, : lengths[0]], rtol=0, atol=0)
    torch.testing.assert_close(changed_output[:, -lengths[-1] :], packed_output[:, -lengths[-1] :], rtol=0, atol=0)
    assert not torch.allclose(changed_output[:, middle_document], packed_output[:, middle_document])


def test_cudnn_sparse_attention_rejects_dropout():
    config = tiny_glm5_next_config().text_config
    config.kv_lora_rank = 512
    config.attention_dropout = 0.1
    backend = tiny_backend()
    backend.attn = "cudnn"

    with pytest.raises(ValueError, match="does not support attention dropout"):
        Glm5NextSparseAttention(config, layer_idx=3, backend=backend)


def test_sparse_attention_empty_cp_shard_keeps_gather_backward_live(monkeypatch):
    sparse = tiny_glm5_next_model().model.language_model.layers["3"].self_attn
    backward_calls = 0

    class FakeGather(torch.autograd.Function):
        @staticmethod
        def forward(ctx, local_hidden):
            ctx.local_length = local_hidden.shape[1]
            return torch.cat((local_hidden, local_hidden), dim=1)

        @staticmethod
        def backward(ctx, full_grad):
            nonlocal backward_calls
            backward_calls += 1
            return full_grad[:, : ctx.local_length] + full_grad[:, ctx.local_length :]

    class FakeCPMesh:
        @staticmethod
        def get_group():
            return None

    monkeypatch.setattr(
        glm5_next_layers,
        "all_gather_sequence",
        lambda local_hidden, _group, dim=1: FakeGather.apply(local_hidden),
    )
    sparse.setup_cp_attention(FakeCPMesh())
    local_hidden = torch.randn(1, 2, 16, requires_grad=True)
    context = Glm5NextPackedContext(
        doc_ids=torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
        seq_start=2,
        cp_size=2,
    )

    output = sparse(local_hidden, packed_context=context)
    output.sum().backward()

    assert output.requires_grad
    assert backward_calls == 1
    torch.testing.assert_close(local_hidden.grad, torch.zeros_like(local_hidden))


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
    assert "cudnn" in model._packed_cp_attn_backends


def test_hyperconnection_fp32_parameters_have_a_dedicated_fsdp_holder():
    hyperconnection = tiny_glm5_next_model().model.language_model.layers["0"].attn_hc

    assert set(dict(hyperconnection.named_parameters(recurse=False))) == {"fn"}
    assert set(dict(hyperconnection._fp32_params.named_parameters())) == {"base", "scale"}
    assert all(parameter.dtype is torch.float32 for parameter in hyperconnection._fp32_params.parameters())


def test_moe_router_correction_bias_only_controls_expert_selection():
    config = build_glm5_next_moe_config(tiny_glm5_next_config().text_config, torch.float32)

    assert config.score_func == "sigmoid_with_bias"
    assert not config.router_weight_uses_score_correction_bias


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the FLA kernel branch runs only for CUDA tensors")
def test_short_sequence_training_uses_chunk_kda_and_reaches_kda_parameters(monkeypatch):
    calls = []

    def fake_chunk_kda(q, k, v, g, beta, **kwargs):
        calls.append(q.shape[1])
        output = (q * k + v) * beta.unsqueeze(-1).to(v.dtype) + g.float().mean().to(v.dtype)
        return output, None

    monkeypatch.setattr(glm5_next_layers, "_CHUNK_KDA_OK", True)
    monkeypatch.setattr(glm5_next_layers, "_chunk_kda", fake_chunk_kda)
    # PyTorch short convolution and gate keep the test free of Triton compilation.
    monkeypatch.setattr(glm5_next_layers, "_SHORT_CONV_OK", False)
    monkeypatch.setattr(glm5_next_layers, "_KDA_GATE_OK", False)
    config = tiny_glm5_next_config().text_config
    config.hidden_size, config.linear_head_dim, config.linear_num_heads = 128, 64, 2
    config.linear_conv_kernel_dim = 4
    config.torch_dtype = "bfloat16"
    torch.manual_seed(0)
    layer = Glm5NextLinearAttention(config, layer_idx=0).cuda().train()
    hidden_states = torch.randn(1, 32, config.hidden_size, device="cuda", dtype=torch.bfloat16)

    layer(hidden_states).float().sum().backward()

    assert calls == [32]
    missing = [name for name, param in layer.named_parameters() if param.grad is None]
    assert missing == []
