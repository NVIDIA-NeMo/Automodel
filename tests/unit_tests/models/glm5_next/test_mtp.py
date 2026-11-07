# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.models.glm5_next.mtp import Glm5NextMTP
from tests.unit_tests.models.glm5_next.conftest import tiny_backend, tiny_glm5_next_config

_IDS = torch.tensor([[5, 9, 2, 7, 11, 4, 8, 3]])
_DOCS = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2]], dtype=torch.int32)


def _model(
    num_mtp_layers: int = 1, *, seed: int = 0, select_every_key: bool = False
) -> Glm5NextForConditionalGeneration:
    torch.manual_seed(seed)
    config = tiny_glm5_next_config()
    config.text_config.num_nextn_predict_layers = num_mtp_layers
    if select_every_key:
        # Every complete pool plus the tail: the sparse layer attends to all causal keys of a document.
        config.text_config.index_topk = 64
    model = Glm5NextForConditionalGeneration(config, backend=tiny_backend())
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def _trainable_mtp_parameters(model: Glm5NextForConditionalGeneration) -> list[torch.nn.Parameter]:
    """MTP parameters on the loss path; the KPool indexer only selects keys, under ``no_grad``."""
    return [parameter for name, parameter in model.mtp.named_parameters() if ".indexer." not in name]


def _rms(states: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    return weight * states * torch.rsqrt(states.square().mean(-1, keepdim=True) + eps)


def _reference_mtp(model: Glm5NextForConditionalGeneration, h_prev: torch.Tensor) -> torch.Tensor:
    """GLM-5.3-Flash MTP depth 1 written from the released model's definition (fp32, one row).

    ``fused = eh_proj([enorm(embed(x[t+1])), hnorm(h_prev[t])])``, causal NoPE MLA inside the
    document, MoE, residual adds and the final norm. The routed MoE is the decoder's shared
    module; attention is spelled out and is exact here because ``index_topk`` covers every key.
    """
    text = model.config.text_config
    eps = text.rms_norm_eps
    layer = model.mtp.layers[0]
    attn = layer.self_attn
    embeddings = model.get_input_embeddings().weight
    length = _IDS.shape[1]

    future = torch.zeros(1, length, text.hidden_size)
    for position in range(length - 1):
        if _DOCS[0, position + 1] == _DOCS[0, position]:
            future[0, position] = embeddings[_IDS[0, position + 1]]
    fused = torch.cat((_rms(future, layer.enorm.weight, eps), _rms(h_prev, layer.hnorm.weight, eps)), dim=-1)
    fused = fused @ layer.eh_proj.weight.T

    x = _rms(fused, layer.input_layernorm.weight, eps)
    heads, nope, value_dim = text.num_attention_heads, text.qk_nope_head_dim, text.v_head_dim
    query = _rms(x @ attn.q_a_proj.weight.T, attn.q_a_layernorm.weight, eps) @ attn.q_b_proj.weight.T
    latent = _rms(x @ attn.kv_a_proj_with_mqa.weight.T, attn.kv_a_layernorm.weight, eps)
    key, value = (latent @ attn.kv_b_proj.weight.T).view(length, heads, nope + value_dim).split([nope, value_dim], -1)
    query = query.view(length, heads, nope)
    scores = torch.einsum("qhd,khd->hqk", query, key) * nope**-0.5
    positions = torch.arange(length)
    visible = (positions[None, :] <= positions[:, None]) & (_DOCS[0][None, :] == _DOCS[0][:, None])
    probs = scores.masked_fill(~visible, float("-inf")).softmax(-1)
    attention = torch.einsum("hqk,khd->qhd", probs, value).reshape(1, length, heads * value_dim)
    hidden = fused + attention @ attn.o_proj.weight.T

    no_padding = torch.zeros(1, length, dtype=torch.bool)
    hidden = hidden + layer.mlp(_rms(hidden, layer.post_attention_layernorm.weight, eps), no_padding)
    return _rms(hidden, layer.final_layernorm.weight, eps)


def test_mtp_head_follows_num_nextn_predict_layers():
    model = _model(1)

    assert isinstance(model.mtp, Glm5NextMTP)
    assert [layer.layer_idx for layer in model.mtp.layers] == [model.config.text_config.num_hidden_layers]
    supports = ModelSupports(model, None)
    assert supports.mtp_enabled
    assert supports.supports_mtp_cp
    assert _model(0).mtp is None


def test_checkpoint_free_training_step_is_finite_and_reaches_the_mtp_head():
    model = _model(1).train()
    assert all(torch.isfinite(parameter).all() for parameter in model.mtp.parameters())

    output = model(input_ids=_IDS, attention_mask=_DOCS)
    (mtp_hidden,) = output.mtp_per_depth_h
    loss = output.logits.square().mean() + model.lm_head(mtp_hidden).square().mean()
    loss.backward()

    assert mtp_hidden.shape == (1, 8, 16)
    assert output.mtp_loss_scaling_factor == 0.1
    layer = model.mtp.layers[0]
    for parameter in _trainable_mtp_parameters(model):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert layer.eh_proj.weight.grad.abs().sum() > 0
    assert layer.self_attn.q_a_proj.weight.grad.abs().sum() > 0
    assert any(p.grad.abs().sum() > 0 for p in layer.mlp.experts.parameters())


def test_mtp_runs_only_in_training():
    model = _model(1).eval()

    with torch.no_grad():
        output = model(input_ids=_IDS, attention_mask=_DOCS)

    assert output.mtp_per_depth_h is None
    assert output.mtp_loss_scaling_factor is None


def test_mtp_forward_and_backward_match_the_reference_definition():
    model = _model(1, seed=3, select_every_key=True).train()
    model.config.text_config.output_hidden_states = True
    params = _trainable_mtp_parameters(model)

    output = model(input_ids=_IDS, attention_mask=_DOCS)
    expected = _reference_mtp(model, output.hidden_states)
    probe = torch.randn_like(expected)
    actual_grads = torch.autograd.grad((output.mtp_per_depth_h[0] * probe).sum(), params, retain_graph=True)
    expected_grads = torch.autograd.grad((expected * probe).sum(), params, allow_unused=True)

    torch.testing.assert_close(output.mtp_per_depth_h[0], expected, rtol=1e-5, atol=1e-5)
    for actual, reference in zip(actual_grads, expected_grads):
        reference = torch.zeros_like(actual) if reference is None else reference
        torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_mtp_does_not_read_the_next_document():
    model = _model(1, seed=5).train()
    changed = _IDS.clone()
    changed[0, 4:] = torch.tensor([13, 14, 15, 16])

    baseline = model(input_ids=_IDS, attention_mask=_DOCS).mtp_per_depth_h[0]
    perturbed = model(input_ids=changed, attention_mask=_DOCS).mtp_per_depth_h[0]

    torch.testing.assert_close(baseline[:, :4], perturbed[:, :4], rtol=0.0, atol=0.0)
    assert not torch.allclose(baseline[:, 4:], perturbed[:, 4:])


def test_cp_rank_receives_its_slice_of_the_globally_shifted_embeddings():
    model = _model(1)
    embeddings = torch.arange(8, dtype=torch.float32).view(1, 8, 1).expand(1, 8, 2)
    context = Glm5NextPackedContext(doc_ids=_DOCS, seq_start=4, cp_size=2, original_seq_len=8)

    (shifted,) = model._mtp_embed_inputs(embeddings, context)

    assert shifted[..., 0].tolist() == [[5.0, 6.0, 7.0, 0.0]]


def test_cp_targets_use_the_glm_document_map():
    model = _model(1)
    batch = {
        "input_ids": _IDS[0].clone(),
        "labels": torch.arange(10, 18),
        "attention_mask": _DOCS[0].clone(),
    }

    inputs = model.prepare_mtp_inputs_for_cp(batch, ignore_index=-100)

    assert inputs.targets[0].tolist() == [[11, 12, 13, -100, 15, 16, 17, -100]]
    assert inputs.valid_masks[0].tolist() == [[True, True, True, False, True, True, True, False]]
    assert batch["input_ids"].shape == (1, 8)


def test_router_bias_update_reaches_the_mtp_gate():
    model = _model(1).train()
    model(input_ids=_IDS, attention_mask=_DOCS)
    gate = model.mtp.layers[0].mlp.gate
    before = gate.e_score_correction_bias.clone()

    model.update_moe_gate_bias()

    assert not torch.equal(gate.e_score_correction_bias, before)


def test_single_rank_thd_batch_keeps_document_ids_for_the_mtp_loss():
    model = _model(1).train()
    labels = torch.arange(20, 28).unsqueeze(0)
    batch = {
        "input_ids": _IDS.clone(),
        "labels": labels.clone(),
        "_packed_seq_ids": _DOCS.clone(),
        "cu_seqlens_padded": torch.tensor([0, 4, 8]),
        "qkv_format": "thd",
    }

    _, batch = ContextParallelSharder(model, None, batch).shard(batch)
    output = model(input_ids=batch["input_ids"], _packed_seq_ids=batch["_packed_seq_ids"])
    per_depth = calculate_mtp_loss(
        MaskedCrossEntropy(reduction="sum"),
        mtp_per_depth_h=output.mtp_per_depth_h,
        labels=labels,
        model=model,
        scaling_factor=1.0,
        seq_idx=batch["_packed_seq_ids"],
        return_per_depth=True,
    )

    assert torch.equal(batch["_packed_seq_ids"], _DOCS)
    logits = model.lm_head(output.mtp_per_depth_h[0])
    targets = torch.tensor([[21, 22, 23, -100, 25, 26, 27, -100]])
    expected = torch.nn.functional.cross_entropy(logits.view(-1, 64), targets.view(-1), reduction="sum")
    torch.testing.assert_close(per_depth.per_depth_losses[0], expected)
