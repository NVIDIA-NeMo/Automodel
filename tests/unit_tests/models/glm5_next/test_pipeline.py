# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GLM pipeline stage, packed microbatch and vision regression tests."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch.distributed.pipelining.microbatch import split_args_kwargs_into_chunks

from nemo_automodel.components.models.glm5_next.cp import Glm5NextPackedContext
from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.glm5_next.conftest import tiny_glm5_next_model


def split_stages(model: Glm5NextForConditionalGeneration, count: int) -> list[Glm5NextForConditionalGeneration]:
    """Partition the four-layer test model, preserving global layer names."""
    stages = []
    for index in range(count):
        stage = copy.deepcopy(model)
        text = stage.language_model
        text.layers = torch.nn.ModuleDict(
            {name: layer for name, layer in text.layers.items() if int(name) // (4 // count) == index}
        )
        if index:
            text.embed_tokens = None
            stage.model.visual = None
        if index != count - 1:
            text.norm = None
            stage.lm_head = None
        stages.append(stage)
    return stages


@pytest.mark.parametrize("activation_checkpointing", [False, True])
@pytest.mark.parametrize("stage_count", [2, 4])
@pytest.mark.parametrize("vision", [False, True])
@pytest.mark.parametrize("fused_loss", [False, True])
def test_pipeline_outputs_and_all_parameter_gradients(stage_count, vision, fused_loss, activation_checkpointing):
    """Compare packed [B,S,V] logits and gradients with the unsplit model."""
    torch.manual_seed(31)
    reference = tiny_glm5_next_model()
    stages = split_stages(reference, stage_count)
    stages[-1]._pp_return_hidden_states = fused_loss
    if activation_checkpointing:
        for stage in stages:
            apply_ac(stage, ignore_router=True)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    docs = torch.tensor([[1] * 6 + [2] * 6], dtype=torch.int32)
    media = {}
    if vision:
        ids[0, 1] = reference.config.image_token_id
        media = {"pixel_values": torch.randn(4, 24), "image_grid_thw": torch.tensor([[1, 2, 2]])}
    expected = reference(ids, attention_mask=docs, **media).logits
    value = ids
    for index, stage in enumerate(stages):
        output = stage(value, attention_mask=docs, **(media if index == 0 else {}))
        value = output if isinstance(output, torch.Tensor) else output.logits
        if index < stage_count - 1:
            assert value.shape == (1, 12, 2, 16)
    if fused_loss:
        assert value.shape == (1, 12, 16)
        value = stages[-1].lm_head(value)
    torch.testing.assert_close(value, expected, rtol=1e-5, atol=1e-6)
    upstream = torch.randn_like(expected)
    (expected * upstream).sum().backward()
    (value * upstream).sum().backward()
    reference_parameters = dict(reference.named_parameters())
    seen = set()
    for stage in stages:
        for name, parameter in stage.named_parameters():
            name = name.replace("_checkpoint_wrapped_module.", "")
            assert name not in seen
            seen.add(name)
            expected_grad = reference_parameters[name].grad
            assert (parameter.grad is None) == (expected_grad is None), name
            if expected_grad is not None:
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-4, atol=1e-6, msg=name)
    assert seen == set(reference_parameters)


def test_packed_metadata_follows_microbatch_rows():
    """Exercise Torch's actual kwargs splitter with different [B,S] document maps."""
    model = tiny_glm5_next_model()
    ids = torch.arange(1, 25).reshape(2, 12)
    docs = torch.tensor([[1] * 6 + [2] * 6, [1] * 3 + [2] * 9], dtype=torch.int32)
    sharder = model.prepare_model_inputs_for_cp({}, num_chunks=2)["cp_sharder"]
    _, batch, _ = sharder.shard_batch(None, None, {"input_ids": ids, "_packed_seq_ids": docs})
    # Negative control: the old dataclass is replicated, retaining both rows.
    legacy = Glm5NextPackedContext(docs)
    _, legacy_chunks = split_args_kwargs_into_chunks((), {"context": legacy}, 2)
    assert legacy_chunks[1]["context"] is legacy
    assert legacy_chunks[1]["context"].doc_ids.shape[0] == 2
    _, chunks = split_args_kwargs_into_chunks((), batch, 2)
    for index, chunk in enumerate(chunks):
        context = chunk["glm5_next_packed_context"]
        assert isinstance(context, tuple)
        assert context[0].shape == (1, 12)
        torch.testing.assert_close(context[0], docs[index : index + 1])
        actual = model(**chunk).logits
        expected = model(ids[index : index + 1], attention_mask=docs[index : index + 1]).logits
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("cp_size", [1, 2, 4])
@pytest.mark.parametrize("fused_loss", [False, True])
def test_stage_metas_and_initialization(cp_size, fused_loss):
    """Metadata uses padded local sequence lengths and keeps mHC streams."""
    model = tiny_glm5_next_model()
    stages = split_stages(model, 4)
    local_length = (13 + (-13) % cp_size) // cp_size
    for index, stage in enumerate(stages):
        stage.cp_mesh = SimpleNamespace(size=lambda: cp_size)
        stage._pp_return_hidden_states = fused_loss and index == 3
        stage.initialize_weights(torch.device("cpu"), dtype=torch.float32)
        inputs, outputs = stage.get_pipeline_stage_metas(
            is_first=index == 0, microbatch_size=2, seq_len=13, dtype=torch.float32
        )
        assert inputs[0].shape == ((2, 13) if index == 0 else (2, local_length, 2, 16))
        assert inputs[0].dtype == (torch.long if index == 0 else torch.float32)
        expected = (2, local_length, 2, 16) if index < 3 else (2, local_length, 16 if fused_loss else 64)
        assert outputs[0].shape == expected


def test_nonfirst_stage_does_not_slice_cp_activations_again():
    """A later CP rank receives already-local [B,Slocal,HC,H] activations."""
    stage = split_stages(tiny_glm5_next_model(), 4)[1]
    stage.language_model.layers = torch.nn.ModuleDict()
    hidden = torch.randn(1, 6, 2, 16)
    context = Glm5NextPackedContext(torch.ones(1, 12, dtype=torch.int32), seq_start=6, cp_size=2)
    torch.testing.assert_close(stage(hidden, glm5_next_packed_context=context).logits, hidden)


def test_vision_microbatch_cursor_consumes_empty_slots():
    """Stage-zero media chunks map to image/text/image samples exactly once."""
    torch.manual_seed(47)
    reference = tiny_glm5_next_model()
    stage = split_stages(reference, 2)[0]
    first, second = torch.randn(4, 24), torch.randn(4, 24)
    grid = torch.tensor([[1, 2, 2]])
    stage._vlm_pixel_values_chunks = [first, torch.empty(0, 24), second]
    stage._vlm_image_grid_hws_chunks = [grid, torch.empty(0, 3, dtype=torch.long), grid]
    stage._vlm_chunk_idx = 0
    for index, pixels in enumerate([first, None, second]):
        ids = torch.tensor([[1, 63 if pixels is not None else 2, 3, 4, 5, 6]])
        direct = split_stages(reference, 2)[0]
        expected = direct(ids, pixel_values=pixels, image_grid_thw=grid if pixels is not None else None).logits
        actual = stage(ids).logits
        torch.testing.assert_close(actual, expected)
        assert stage._vlm_chunk_idx == index + 1


def test_pipeline_packed_documents_match_separate_execution():
    """Packed [1,12,V] outputs equal two independent six-token documents."""
    torch.manual_seed(61)
    stages = split_stages(tiny_glm5_next_model().eval(), 4)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]])
    docs = torch.tensor([[1] * 6 + [2] * 6], dtype=torch.int32)

    def forward(tokens: torch.Tensor, documents: torch.Tensor) -> torch.Tensor:
        """Pass ids [1,S] through all stages and return logits [1,S,V]."""
        value = tokens
        for stage in stages:
            value = stage(value, attention_mask=documents).logits
        return value

    with torch.inference_mode():
        packed = forward(ids, docs)
        separate = torch.cat(
            [forward(ids[:, :6], docs[:, :6]), forward(ids[:, 6:], torch.ones_like(docs[:, 6:]))], dim=1
        )
        no_boundary = forward(ids, torch.ones_like(docs))
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    # Negative control: ignoring the boundary must change the second document.
    assert not torch.allclose(no_boundary[:, 6:], separate[:, 6:], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_cp_microbatch_metadata_retains_global_map_and_local_offsets(cp_size):
    """Chunk batch rows while retaining CP's padded [B,Sglobal] document map."""
    model = tiny_glm5_next_model()
    mesh = SimpleNamespace(size=lambda: cp_size, get_local_rank=lambda: cp_size - 1)
    ids = torch.arange(1, 27).reshape(2, 13)
    docs = torch.tensor([[1] * 6 + [2] * 7, [1] * 3 + [2] * 10], dtype=torch.int32)
    sharder = model.prepare_model_inputs_for_cp({}, num_chunks=2)["cp_sharder"]
    _, batch, _ = sharder.shard_batch(mesh, None, {"input_ids": ids, "labels": ids.clone(), "_packed_seq_ids": docs})
    _, chunks = split_args_kwargs_into_chunks((), batch, 2)
    padded_length = 13 + (-13) % cp_size
    local_length = padded_length // cp_size
    start = (cp_size - 1) * local_length
    for row, chunk in enumerate(chunks):
        document_map, seq_start, size, original_length = chunk["glm5_next_packed_context"]
        assert (seq_start, size, original_length) == (start, cp_size, 13)
        assert document_map.shape == (1, padded_length)
        torch.testing.assert_close(document_map[:, :13], docs[row : row + 1])
        assert not document_map[:, 13:].any()
        assert chunk["input_ids"].shape == (1, 13)
        assert chunk["labels"].shape == chunk["padding_mask"].shape == (1, local_length)
        expected_positions = torch.cat((torch.arange(13), torch.zeros(padded_length - 13, dtype=torch.long)))
        torch.testing.assert_close(chunk["position_ids"][0], expected_positions[start:])
        assert chunk["padding_mask"][0, -1]
        assert chunk["labels"][0, -1] == -100


def test_kda_checkpoint_metadata_is_private_to_each_invocation():
    """Checkpoint replay gets fresh metadata while Q/K/V share each call's map."""
    stage = split_stages(tiny_glm5_next_model(), 2)[0]
    attention = stage.language_model.layers["0"].self_attn
    seen = []
    handles = [
        conv.register_forward_pre_hook(lambda _m, _a, kwargs: seen.append(kwargs["cu_seqlens"]), with_kwargs=True)
        for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    ]
    apply_ac(stage, ignore_router=True)
    try:
        outputs = [stage(torch.tensor([[1, 2, 3, 4, 5, 6]])).logits for _ in range(2)]
        for output in outputs:
            output.square().sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    # Two forward calls and their two checkpoint replays; observation uses only
    # Python references and adds no tensor operations to the checkpoint trace.
    assert len(seen) == 12
    invocations = seen[::3]
    assert len({id(boundaries) for boundaries in invocations}) == 4
    for q, k, v in zip(seen[::3], seen[1::3], seen[2::3], strict=True):
        assert q is k and k is v


def test_kda_reuses_packed_context_cpu_boundaries():
    """Every convolution receives cached CPU boundaries and fresh device metadata."""
    stage = split_stages(tiny_glm5_next_model(), 2)[0]
    attention = stage.language_model.layers["0"].self_attn
    context = Glm5NextPackedContext(torch.tensor([[1, 1, 1, 2, 2, 3]], dtype=torch.int32))
    boundaries, boundaries_cpu = context.row_cu_seqlens(0)
    seen = []
    handles = [
        conv.register_forward_pre_hook(
            lambda _m, _a, kwargs: seen.append((kwargs["cu_seqlens"], kwargs["cu_seqlens_cpu"])),
            with_kwargs=True,
        )
        for conv in (attention.q_conv1d, attention.k_conv1d, attention.v_conv1d)
    ]
    try:
        for _ in range(2):
            attention(torch.randn(1, 6, attention.hidden_size), packed_context=context)
    finally:
        for handle in handles:
            handle.remove()
    assert len(seen) == 6
    assert seen[0][0] is not seen[3][0]
    for device_copy, cpu_copy in seen:
        assert device_copy is not boundaries
        assert cpu_copy is boundaries_cpu
        assert cpu_copy.device.type == "cpu"
        torch.testing.assert_close(device_copy, torch.tensor([0, 3, 5, 6]))
