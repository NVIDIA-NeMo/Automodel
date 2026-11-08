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

"""Numerical multistage coverage of the forty-layer CSA2 graph and both Engram tables."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v41.cp import shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.test_vision_packing import _batch, _model, _sample

pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]


def _stages(reference: DeepseekV41ForCausalLM, cuts: tuple[int, ...] = (14,)) -> tuple[DeepseekV41ForCausalLM, ...]:
    """Trim model copies at supplied cuts, retaining original parameter names."""
    bounds = (0, *cuts, reference.config.text_config.num_hidden_layers)
    stages = []
    for start, end in zip(bounds, bounds[1:]):
        stage = copy.deepcopy(reference)
        for name in list(stage.model.layers):
            if not start <= int(name) < end:
                del stage.model.layers[name]
        if end != bounds[-1]:
            stage.model.norm = stage.lm_head = None
        if start != 0:
            stage.model.embed_tokens = stage.model.vision = stage.model.aligner = None
            for name in ("image_start", "image_end", "image_newline"):
                stage.model.register_parameter(name, None)
        stages.append(stage)
    return tuple(stages)


@pytest.mark.parametrize("cuts", [(14,), (3,), (24,), (3, 9, 25), (1, 3, 9, 15, 21, 24, 29)])
@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("images", [False, True])
def test_pipeline_matches_full_forward_gradients_and_update(
    cuts: tuple[int, ...], checkpoint: bool, images: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(483)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True, engram=True)
        stages = _stages(reference, cuts)
        first, last = stages[0], stages[-1]
        if checkpoint:
            for model in (reference, *stages):
                apply_ac(model, ignore_router=True)
        samples = [
            _sample(((3, 5),) if images else (), prefix=3),
            _sample(()),
            _sample(((1, 1), (2, 3)) if images else (), prefix=2),
        ]
        batch = _batch(samples, length=80, alignment=1)
        _, batch, _ = shard_cp_batch(None, None, batch, packed_alignment=2)
        labels = batch.pop("labels")
        batch.pop("n_images_per_sample", None)
        expected = reference(**batch).logits
        payload = first(**batch)
        assert payload[0].shape == (1, 80, reference.config.text_config.hc_mult, 16)
        assert payload[1].dtype == torch.float32 and payload[1].requires_grad
        assert payload[2].dtype == torch.long and not payload[2].requires_grad
        # No kwargs are supplied to the second stage: its wire payload must
        # preserve packing, visual routing and Engram token history by itself.
        for stage in stages[1:-1]:
            payload = stage(*payload)
        actual = last(*payload).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten())
        actual_loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten())
        expected_loss.backward()
        actual_loss.backward()
        reference_params = dict(reference.named_parameters())
        stage_params = {name: parameter for stage in stages for name, parameter in stage.named_parameters()}
        assert set(stage_params) == set(reference_params)
        assert reference.model.layers["14"].engram.embed.weight.grad.abs().sum() > 0
        for name, parameter in stage_params.items():
            expected_grad = reference_params[name].grad
            if expected_grad is None:
                assert parameter.grad is None, name
            else:
                assert torch.isfinite(parameter.grad).all(), name
                torch.testing.assert_close(parameter.grad, expected_grad, rtol=1e-5, atol=1e-7, msg=name)
        reference_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.3)
        actual_norm = torch.nn.utils.clip_grad_norm_(list(stage_params.values()), 0.3)
        torch.testing.assert_close(actual_norm, reference_norm, rtol=1e-6, atol=1e-7)
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        torch.optim.SGD(list(stage_params.values()), lr=0.01).step()
        for name, parameter in stage_params.items():
            torch.testing.assert_close(parameter, reference_params[name], rtol=1e-6, atol=1e-7, msg=name)


def test_pp_partition_preserves_supplied_layers_and_vision_ownership() -> None:
    model = _model(full_schedule=True, engram=True)
    bounds = (0, 3, 9, 25, 40)
    supplied = [[f"model.layers.{i}" for i in range(a, b)] for a, b in zip(bounds, bounds[1:])]
    plan = model.customize_pipeline_stage_modules(supplied, layers_prefix="model.")
    for before, after in zip(supplied, plan):
        assert [s for s in after if s.startswith("model.layers.")] == before
        assert "model.engram_hash" in after
    assert "model.vision" in plan[0]
    assert all("model.vision" not in stage for stage in plan[1:])
    assert all("model.engram_hash" not in stage for stage in supplied)


def test_stage_metadata_preserves_fp32_coefficients_and_local_sequence() -> None:
    first, last = _stages(_model(full_schedule=True))
    _, outputs = first.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=4096, dtype=torch.bfloat16)
    inputs, logits = last.get_pipeline_stage_metas(
        is_first=False, microbatch_size=1, seq_len=4096, dtype=torch.bfloat16
    )
    assert [(x.shape, x.dtype) for x in inputs] == [(x.shape, x.dtype) for x in outputs]
    assert outputs[0].shape == (1, 4096, 4, 16)
    assert outputs[1].dtype == torch.float32
    assert outputs[2].shape == (1, 4096, 5) and outputs[2].dtype == torch.long
    assert logits[0].dtype == torch.float32
    last._pp_return_hidden_states = True
    _, hidden = last.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=4096, dtype=torch.bfloat16)
    assert hidden[0].shape == (1, 4096, 16)


def test_pruned_stages_initialize_all_retained_weights() -> None:
    first, last = _stages(_model(full_schedule=True, engram=True))
    for stage in (first, last):
        stage.to_empty(device="cpu")
        stage.initialize_weights(torch.device("cpu"), dtype=torch.float32)
        assert all(torch.isfinite(p).all() for p in stage.parameters())


@pytest.mark.parametrize("padding", ["none", "right", "empty"])
@pytest.mark.parametrize("length", [12, 13])
def test_unpacked_pipeline_metadata_matches_unsplit(padding: str, length: int) -> None:
    """An implicit single document must preserve unpacked attention semantics."""
    torch.set_num_threads(1)
    torch.manual_seed(93)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True, engram=True)
        first, last = _stages(reference)
        ids = torch.arange(2 * length).reshape(2, length) % 7 + 1
        mask = torch.ones_like(ids)
        if padding == "right":
            mask[0, 9:] = 0
        elif padding == "empty":
            mask.zero_()
        expected = reference(ids, attention_mask=mask).logits
        actual = last(*first(ids, attention_mask=mask)).logits
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_pipeline_rejects_missing_or_rounded_pre_mix() -> None:
    first, last = _stages(_model(full_schedule=True))
    ids = torch.ones(1, 8, dtype=torch.long)
    payload = first(ids)
    with pytest.raises(ValueError, match="requires pre_mix"):
        last(payload[0])
    with pytest.raises(ValueError, match="FP32"):
        last(payload[0], payload[1].bfloat16(), payload[2])


def test_unsplit_model_retains_positional_mask_and_positions() -> None:
    model = _model(full_schedule=True)
    ids = torch.ones(1, 8, dtype=torch.long)
    mask = torch.ones_like(ids)
    positions = torch.arange(8).expand_as(ids)
    torch.testing.assert_close(
        model(ids, mask, positions).logits,
        model(ids, attention_mask=mask, position_ids=positions).logits,
        atol=0,
        rtol=0,
    )


@pytest.mark.parametrize("cuts", [(3, 9, 25), (1, 3, 9, 15, 21, 24, 29)])
def test_shared_state_wire_matches_metadata_and_preserves_frozen_keys(cuts: tuple[int, ...]) -> None:
    torch.set_num_threads(1)
    with torch.compiler.set_stance("force_eager"):
        stages = _stages(_model(full_schedule=True, engram=True), cuts)
        ids = torch.arange(16).reshape(1, 16) % 7 + 1
        payload = stages[0](ids)
        for previous, stage in zip(stages, stages[1:]):
            _, output_meta = previous.get_pipeline_stage_metas(
                is_first=previous.model.embed_tokens is not None,
                microbatch_size=1,
                seq_len=16,
                dtype=torch.float32,
            )
            input_meta, _ = stage.get_pipeline_stage_metas(
                is_first=False,
                microbatch_size=1,
                seq_len=16,
                dtype=torch.float32,
            )
            assert [(t.shape, t.dtype) for t in payload] == [(t.shape, t.dtype) for t in output_meta]
            assert [(t.shape, t.dtype) for t in payload] == [(t.shape, t.dtype) for t in input_meta]
            if len(payload) > 3:
                assert payload[3].requires_grad
                assert payload[4].dtype == torch.uint8 and not payload[4].requires_grad
                assert all(not t.requires_grad for t in payload[4:])
                with pytest.raises(ValueError, match="CSA2 payload"):
                    stage(*payload[:-1])
            result = stage(*payload)
            if stage.lm_head is None:
                payload = result


@pytest.mark.parametrize("padding", ["right", "empty"])
@pytest.mark.parametrize("length", [1, 13])
def test_shared_state_with_padding_and_incomplete_compression(padding: str, length: int) -> None:
    torch.set_num_threads(1)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True, engram=True)
        stages = _stages(reference, (3, 9, 25))
        ids = torch.ones(2, length, dtype=torch.long)
        mask = torch.ones_like(ids)
        mask[0, -1:] = 0
        if padding == "empty":
            mask.zero_()
        expected = reference(ids, attention_mask=mask).logits
        payload = stages[0](ids, attention_mask=mask)
        for stage in stages[1:-1]:
            payload = stage(*payload)
        actual = stages[-1](*payload).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.sum().backward()
        actual.sum().backward()
        reference_params = dict(reference.named_parameters())
        for stage in stages:
            for name, param in stage.named_parameters():
                grad = reference_params[name].grad
                if grad is None:
                    assert param.grad is None, name
                else:
                    torch.testing.assert_close(param.grad, grad, rtol=1e-5, atol=1e-6, msg=name)


def test_shared_state_accepts_int32_documents_and_broadcast_positions() -> None:
    torch.set_num_threads(1)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True)
        first, last = _stages(reference, (3,))
        ids = torch.ones(2, 16, dtype=torch.int32)
        positions = torch.arange(16, dtype=torch.int32).unsqueeze(0)
        documents = torch.ones_like(ids)
        mask = torch.ones_like(ids)
        expected = reference(ids, position_ids=positions, attention_mask=mask, packed_seq_ids=documents).logits
        payload = first(ids, position_ids=positions, attention_mask=mask, packed_seq_ids=documents)
        actual = last(*payload).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("length,blocks", [(16, 2), (17, 3)])
def test_decoder_candidates_use_compact_block_coordinates(length: int, blocks: int) -> None:
    """The fixture has ratio-one decoder keys and eight keys per candidate block."""
    torch.set_num_threads(1)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True)
        first, last = _stages(reference, (24,))
        ids = torch.ones(1, length, dtype=torch.long)
        payload = first(ids)
        _, metas = first.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=length, dtype=torch.float32)
        assert payload[-1].shape == (1, length, blocks)
        assert metas[-1].shape == payload[-1].shape
        torch.testing.assert_close(last(*payload).logits, reference(ids).logits, rtol=0, atol=0)
