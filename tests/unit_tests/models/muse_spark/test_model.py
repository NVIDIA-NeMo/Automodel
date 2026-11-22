# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Muse Spark Open model, capability, and pipeline tests."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel._transformers.capabilities import ModelSupports
from nemo_automodel.components.distributed.pipelining.functional import (
    calculate_virtual_stages,
    generate_hf_model_fqn_per_model_part,
)
from tests.unit_tests.models.muse_spark.conftest import tiny_muse_spark_model


class _EchoTextModel(nn.Module):
    """Return the residual unchanged so PP/CP boundary behavior is isolated."""

    def forward(self, *, inputs_embeds, **kwargs):
        del kwargs
        return SimpleNamespace(last_hidden_state=inputs_embeds)


def test_tiny_text_forward_backward_is_finite() -> None:
    """The compressed MoE, gated residuals, and attention compose end to end."""
    torch.manual_seed(7)
    model = tiny_muse_spark_model().train()
    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])

    logits = model(input_ids=input_ids).logits
    logits.square().mean().backward()

    assert logits.shape == (1, 6, 64)
    assert torch.isfinite(logits).all()
    assert model.model.layers[0].self_attn.q_proj.weight.grad is not None
    assert model.model.layers[0].feed_forward.experts.gate_and_up_projs.grad is not None


def test_parallel_capabilities_include_packed_cp_and_pp() -> None:
    """Recipe capability checks accept the supported Muse topology."""
    model = tiny_muse_spark_model()
    # The portable fixture constructs SDPA modules; Muse's verified packed CP
    # transport is the production Transformer Engine attention path.
    model.backend.attn = "te"
    supports = ModelSupports(model, SimpleNamespace(cp_size=2, tp_size=1, pp_size=2, ep_size=2))

    assert supports.supports_cp
    assert supports.supports_pp
    assert supports.supports_ep
    assert supports.supports_sequence_packing
    assert supports.supports_cp_with_sequence_packing


def test_pipeline_metadata_uses_global_input_length_for_every_chunk() -> None:
    """Both input formats use global ids and CP-local activations on every PP chunk."""
    model = tiny_muse_spark_model()
    model.cp_mesh = SimpleNamespace(size=lambda: 2)
    model.lm_head = None

    _, outputs = model.get_pipeline_stage_metas(
        is_first=True,
        microbatch_size=1,
        seq_len=64,
        dtype=torch.bfloat16,
    )
    assert outputs[0].shape == (1, 32, 16)
    assert outputs[0].dtype is torch.float32

    inputs, outputs = model.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=1,
        seq_len=64,
        dtype=torch.bfloat16,
    )
    assert inputs[0].shape == outputs[0].shape == (1, 32, 16)
    assert inputs[0].dtype is outputs[0].dtype is torch.float32


def test_nonpacked_cp_keeps_received_pipeline_residual_local(monkeypatch) -> None:
    """A non-first PP stage must not CP-shard an already-local BSHD residual again."""
    model = tiny_muse_spark_model().eval()
    model.model = _EchoTextModel()
    model.lm_head = None
    model.cp_mesh = SimpleNamespace(size=lambda: 2)
    residual = torch.randn(1, 4, model.config.hidden_size)

    def fail_on_reshard(*args, **kwargs):
        del args, kwargs
        raise AssertionError("received PP residual was context-sharded twice")

    monkeypatch.setattr(
        "nemo_automodel.components.models.muse_spark.model.shard_sequence_for_cp_round_robin",
        fail_on_reshard,
    )

    output = model(input_ids=residual)

    assert output.logits.shape == residual.shape
    torch.testing.assert_close(output.logits, residual)


def test_full_pp8_split_retains_all_62_layers() -> None:
    """The production interleaved split has two chunks/rank without rounding layers away."""
    num_stages, stages_per_rank = calculate_virtual_stages(
        num_layers=62,
        layers_per_stage=4,
        pp_size=8,
        is_single_stage_schedule=False,
    )
    stages = generate_hf_model_fqn_per_model_part(
        num_stages=num_stages,
        num_layers=62,
        include_embeddings=True,
        include_lm_head=True,
        include_rotary_emb=True,
    )

    decoder_layers = [name for stage in stages for name in stage if ".layers." in name]
    assert num_stages == 16
    assert stages_per_rank == 2
    assert len(decoder_layers) == len(set(decoder_layers)) == 62
    assert [sum(".layers." in name for name in stage) for stage in stages] == [4] * 14 + [3, 3]


def test_pipeline_module_hook_moves_multimodal_towers_to_first_stage() -> None:
    """The splitter uses top-level Muse tower names and retains embed normalization."""
    model = tiny_muse_spark_model()
    generated = [
        ["model.embed_tokens", "model.vision_model", "model.audio_encoder", "model.layers.0"],
        ["model.layers.1", "model.norm", "lm_head"],
    ]

    stages = model.customize_pipeline_stage_modules(generated, layers_prefix="model.")

    assert "vision_model" in stages[0]
    assert "audio_encoder" in stages[0]
    assert "model.embed_norm" in stages[0]
    assert "vision_projection" in stages[0]
    assert "model.vision_model" not in stages[0]


def test_pp_media_cursor_advances_when_cp_shard_has_no_local_placeholder(monkeypatch) -> None:
    """A CP rank must consume the global PP media chunk even with no local image token."""
    model = tiny_muse_spark_model().eval()
    model._vlm_pixel_values_chunks = [torch.ones(1, 1), torch.full((1, 1), 2.0)]
    model._vlm_image_grid_hws_chunks = [torch.ones(1, 3, dtype=torch.long)] * 2
    model._vlm_chunk_idx = 0
    monkeypatch.setattr(model, "_vision_features", lambda pixels, grid: torch.ones(1, 16))

    # The image placeholder belongs to the other CP shard in microbatch 0.
    model(
        input_ids=torch.tensor([[1, 2]]),
        _thd_local_indices=torch.tensor([0, 1]),
        _muse_spark_global_image_mask=torch.tensor([[False, False, True, False]]),
    )
    assert model._vlm_chunk_idx == 1

    # Microbatch 1 has a local placeholder and must receive chunk 1, not the
    # stale chunk 0 skipped by the old rank-local cursor condition.
    captured = {}

    def capture_second_chunk(pixels, grid):
        captured["pixel"] = float(pixels.item())
        return torch.ones(1, 16)

    monkeypatch.setattr(model, "_vision_features", capture_second_chunk)
    model(
        input_ids=torch.tensor([[model.config.patch_token_id, 3]]),
        _thd_local_indices=torch.tensor([0, 1]),
        _muse_spark_global_image_mask=torch.tensor([[True, False]]),
    )

    assert captured["pixel"] == 2.0
    assert model._vlm_chunk_idx == 2


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_packed_pp_media_indices_without_cp(dtype: torch.dtype) -> None:
    """Each unsharded PP chunk keeps its own identity map and media features."""
    model = tiny_muse_spark_model()
    model.lm_head = None
    patch = model.config.patch_token_id
    ids = torch.tensor([[1, patch, 2, 0, 3, 4, 5, 0], [6, 7, patch, 8, 9, 10, 11, 0]], dtype=dtype)
    batch = {
        "input_ids": ids,
        "labels": ids.clone(),
        "position_ids": torch.arange(8).expand(2, -1),
        "seq_lens": torch.tensor([[3, 3], [7, -1000]]),
        "seq_lens_padded": torch.tensor([[4, 4], [8, -1000]]),
        "qkv_format": "thd",
        "_vlm_pp_media_chunks": {"pixel_values": [torch.ones(1, 1), torch.ones(1, 1)]},
    }
    prepared = model.prepare_model_inputs_for_cp(batch, num_chunks=2)
    sharder = prepared.pop("cp_sharder")
    batch.update(prepared)
    _, sharded, layout = sharder.shard_batch(None, None, batch)
    assert layout is None
    torch.testing.assert_close(sharded["input_ids"], ids.to(torch.long))
    torch.testing.assert_close(sharded["_thd_local_indices"], torch.arange(8).expand(2, -1))
    torch.testing.assert_close(sharded["_muse_spark_global_image_mask"], ids == patch)
    for chunk, value in enumerate((3.0, 9.0)):
        features = torch.full((1, 2), value, requires_grad=True)
        selected = model._local_media_features(
            features,
            sharded["input_ids"][chunk] == patch,
            sharded["_muse_spark_global_image_mask"][chunk],
            sharded["_thd_local_indices"][chunk],
        )
        torch.testing.assert_close(selected, features)
        selected.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones_like(features))


def test_packed_pp_first_stage_selects_local_ids_before_embedding(monkeypatch) -> None:
    """The public PP input stays global while embedding sees only its owned tokens."""
    model = tiny_muse_spark_model()
    model.lm_head = None
    model.model = _EchoTextModel()
    embeddings = nn.Embedding(64, model.config.hidden_size)

    def embed(input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """Embed the selected local tokens.

        Args:
            input_ids: Selected token ids [local_tokens].
            kwargs: Unused media arguments.

        Returns:
            Embeddings [local_tokens, hidden].
        """
        del kwargs
        return embeddings(input_ids)

    monkeypatch.setattr(model, "_embed_and_splice", embed)
    ids = torch.tensor([[2, 5, 7, 11]])
    output = model(input_ids=ids, qkv_format="thd", _thd_local_indices=torch.tensor([[0, 3]]))
    torch.testing.assert_close(output.logits, embeddings(torch.tensor([[2, 11]])))
    output.logits.sum().backward()
    assert embeddings.weight.grad[2].sum() == model.config.hidden_size
    assert embeddings.weight.grad[11].sum() == model.config.hidden_size
    assert embeddings.weight.grad[5].count_nonzero() == 0
    assert embeddings.weight.grad[7].count_nonzero() == 0

    with pytest.raises(ValueError, match="local-token indices"):
        model(input_ids=ids, qkv_format="thd")


@pytest.mark.parametrize("via_checkpointer", [False, True])
def test_meta_random_initialization_produces_finite_model(via_checkpointer: bool) -> None:
    """Fresh benchmark models must initialize parameters and persistent RoPE buffers."""
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
    from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
    from tests.unit_tests.models.muse_spark.conftest import tiny_backend, tiny_muse_spark_config

    reference = tiny_muse_spark_model()
    expected_freq = reference.model.rotary_emb.inv_freq.detach().clone()
    config = tiny_muse_spark_config()
    config.dtype = torch.float32
    with torch.device("meta"):
        model = MuseSpark1xOmniForCausalLM(config, backend=tiny_backend(adapter=False))

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        if via_checkpointer:
            Checkpointer.initialize_model_weights(model, torch.device("cpu"))
        else:
            model.to_empty(device="cpu")
            model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic, warn_only=previous_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill

    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
    for name, buffer in model.named_buffers():
        assert torch.isfinite(buffer).all(), name
    torch.testing.assert_close(model.model.rotary_emb.inv_freq, expected_freq, rtol=0, atol=0)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    output = model(input_ids=ids, labels=ids)
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.loss)
    output.loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("qkv_format", ["bshd", "thd"])
def test_pipeline_fused_head_preserves_scaled_loss_and_gradients(qkv_format: str, monkeypatch) -> None:
    """The last PP stage can skip the head without losing MetaP scaling or layout."""
    torch.manual_seed(19)
    model = tiny_muse_spark_model().train()
    model.model = _EchoTextModel()
    model.config.metap["base_width"] = 0.5
    model.config.output_soft_cap_temp = 1.5
    residual = (torch.randn(1, 7, model.config.hidden_size) * 3).requires_grad_()
    labels = torch.tensor([[2, 3, -100, 5, 6, 7, 8]])
    weight = model.lm_head.weight
    with torch.no_grad():
        weight.mul_(20)
    reference_logits = model(input_ids=residual, qkv_format=qkv_format).logits
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.flatten(0, 1), labels.flatten(), reduction="sum"
    )
    reference_grads = torch.autograd.grad(reference_loss, (residual, weight))

    model._pp_return_hidden_states = True

    def fail_on_head(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the fused PP path materialized logits inside the model")

    monkeypatch.setattr(model.lm_head, "forward", fail_on_head)
    hidden = model(input_ids=residual, qkv_format=qkv_format)
    assert hidden.shape == residual.shape
    logits = torch.nn.functional.linear(hidden, weight)
    logits = torch.tanh(logits / model.config.output_soft_cap_temp) * model.config.output_soft_cap_temp
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")
    gradients = torch.autograd.grad(loss, (residual, weight))
    torch.testing.assert_close(loss, reference_loss, atol=2e-6, rtol=2e-6)
    for actual, reference in zip(gradients, reference_grads):
        assert reference.count_nonzero() > 0
        torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    assert gradients[0][:, 2].count_nonzero() == 0


def test_pipeline_fused_head_metadata_uses_cp_local_hidden_size() -> None:
    """The final PP stage declares the hidden tensor consumed by fused CE."""
    model = tiny_muse_spark_model()
    model.cp_mesh = SimpleNamespace(size=lambda: 2)
    _, ordinary = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=64, dtype=torch.bfloat16)
    assert ordinary[0].shape == (1, 32, model.config.vocab_size)
    assert model._pp_return_hidden_states_supported
    model._pp_return_hidden_states = True
    _, fused = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=64, dtype=torch.bfloat16)
    assert fused[0].shape == (1, 32, model.config.hidden_size)
    assert fused[0].dtype == torch.bfloat16
    with pytest.raises(ValueError, match="labels to be passed to the pipeline loss"):
        model(input_ids=torch.ones(1, 4, dtype=torch.long), labels=torch.ones(1, 4, dtype=torch.long))


def test_vlm_pipeline_fused_loss_receives_hidden_states() -> None:
    """VLM setup must bypass the model's full logits projection for fused CE."""
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.components.loss.mtp import MTPLossConfig
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    model = tiny_muse_spark_model().train()
    model.model = _EchoTextModel()
    residual = torch.randn(1, 4, model.config.hidden_size, requires_grad=True)
    schedule = SimpleNamespace()
    recipe = SimpleNamespace(
        pp=SimpleNamespace(
            info=SimpleNamespace(has_last_stage=True, stages=[SimpleNamespace(is_last=True)], schedule=schedule)
        ),
        model_parts=[model],
        loss_fn=FusedLinearCrossEntropy(reduction="sum"),
        cfg=SimpleNamespace(mtp=MTPLossConfig()),
        _get_dp_group=lambda **kwargs: None,
    )

    FinetuneRecipeForVLM._configure_pipeline_loss_fn(recipe)
    output = model(inputs_embeds=residual)

    assert isinstance(output, torch.Tensor)
    assert output.shape == residual.shape
    torch.testing.assert_close(output, residual * model.config.output_multiplier)
    assert schedule._loss_fn.loss_fn is recipe.loss_fn
    output.sum().backward()
    torch.testing.assert_close(residual.grad, torch.full_like(residual, model.config.output_multiplier))
    assert model.lm_head.weight.grad is None


@pytest.mark.parametrize("qkv_format", ["bshd", "thd"])
@pytest.mark.parametrize("compute_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_pipeline_residual_preserves_fp32_values_and_gradients(qkv_format, compute_dtype) -> None:
    """Pipeline boundaries must preserve values and gradients below BF16 precision."""
    model = tiny_muse_spark_model().train()
    model.model = _EchoTextModel()
    model.lm_head = None
    inputs_meta, outputs_meta = model.get_pipeline_stage_metas(
        is_first=False, microbatch_size=1, seq_len=4, dtype=compute_dtype
    )
    residual = (
        torch.linspace(1.0001, 1.001, 4 * model.config.hidden_size)
        .reshape(1, 4, model.config.hidden_size)
        .requires_grad_()
    )
    actual = model(input_ids=residual, qkv_format=qkv_format).logits
    assert actual.dtype == inputs_meta[0].dtype == outputs_meta[0].dtype == torch.float32
    assert actual.shape == outputs_meta[0].shape
    assert not torch.equal(residual, residual.bfloat16().float())
    torch.testing.assert_close(actual, residual, rtol=0, atol=0)
    actual.square().sum().backward()
    torch.testing.assert_close(residual.grad, 2 * residual.detach(), rtol=0, atol=0)


@pytest.mark.parametrize("fused_head", [False, True])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_pipeline_final_stage_receives_fp32_residual_with_separate_output_dtype(fused_head, fp32_output) -> None:
    """The final stage receives the residual dtype even when its head uses BF16."""
    model = tiny_muse_spark_model()
    model.config.use_fp32_for_output = fp32_output
    model._pp_return_hidden_states = fused_head
    inputs, outputs = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=4, dtype=torch.bfloat16)
    assert inputs[0].dtype == torch.float32
    assert outputs[0].dtype == (torch.float32 if fp32_output else torch.bfloat16)
    assert outputs[0].shape[-1] == (model.config.hidden_size if fused_head else model.config.vocab_size)


def test_packed_targets_match_fused_head_hidden_state_layout() -> None:
    """Packed loss targets retain the head batch axis and masked token order."""
    model = tiny_muse_spark_model()
    ids = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 0]])
    labels = torch.tensor([[2, 3, -100, -100, 5, 6, -100, -100]])
    batch = {
        "input_ids": ids,
        "labels": labels.clone(),
        "loss_mask": (labels != -100).to(torch.float32),
        "position_ids": torch.tensor([[0, 1, 2, 0, 0, 1, 2, 0]]),
        "seq_lens": torch.tensor([[3, 3]]),
        "seq_lens_padded": torch.tensor([[4, 4]]),
        "qkv_format": "thd",
    }
    prepared = model.prepare_model_inputs_for_cp(batch)
    sharder = prepared.pop("cp_sharder")
    batch.update(prepared)
    _, sharded, layout = sharder.shard_batch(None, None, batch)

    model.model = _EchoTextModel()
    hidden = torch.randn(8, model.config.hidden_size, requires_grad=True)
    output = model(inputs_embeds=hidden, qkv_format="thd", logits_to_keep=1, output_hidden_states=True)
    assert output.hidden_states.shape[:-1] == sharded["labels"].shape == (1, 8)
    assert sharded["input_ids"].shape == sharded["position_ids"].shape == (8,)
    torch.testing.assert_close(sharded["labels"], labels)
    torch.testing.assert_close(sharded["loss_mask"], (labels != -100).to(torch.float32))
    torch.testing.assert_close(layout.local_token_global_indices, torch.arange(8))
    logits = torch.nn.functional.linear(output.hidden_states, model.lm_head.weight)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), sharded["labels"].flatten())
    loss.backward()
    assert torch.isfinite(hidden.grad).all()
    assert hidden.grad[labels.flatten() == -100].count_nonzero() == 0
    assert hidden.grad[labels.flatten() != -100].count_nonzero() > 0


def test_packed_cp_accepts_document_longer_than_sliding_window() -> None:
    """The SWA window limits attention history, not the length of a document."""
    model = tiny_muse_spark_model()
    model.cp_mesh = object()
    length = model.config.sliding_window + 1
    assert model._validate_thd_documents({"seq_lens": torch.tensor([[length, 3, -1000]])}) == length
