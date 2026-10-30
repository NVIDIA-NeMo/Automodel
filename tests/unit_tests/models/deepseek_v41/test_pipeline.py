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

"""Numerical PP2 coverage with all forty CSA2 layers and both Engram tables."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.deepseek_v41.cp import shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.deepseek_v41.test_vision_packing import _batch, _model, _sample

pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]


def _stages(reference: DeepseekV41ForCausalLM) -> tuple[DeepseekV41ForCausalLM, DeepseekV41ForCausalLM]:
    """Trim independent model copies at layer 14, keeping original parameter names."""
    first, last = copy.deepcopy(reference), copy.deepcopy(reference)
    for name in list(first.model.layers):
        if int(name) >= 14:
            del first.model.layers[name]
    for name in list(last.model.layers):
        if int(name) < 14:
            del last.model.layers[name]
    first.model.norm = first.lm_head = None
    last.model.embed_tokens = last.model.vision = last.model.aligner = None
    for name in ("image_start", "image_end", "image_newline"):
        last.model.register_parameter(name, None)
    return first, last


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("images", [False, True])
def test_pp2_matches_full_forward_gradients_and_update(checkpoint: bool, images: bool) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(483)
    with torch.compiler.set_stance("force_eager"):
        reference = _model(full_schedule=True, engram=True)
        first, last = _stages(reference)
        if checkpoint:
            for model in (reference, first, last):
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
        assert len(payload) == 3
        assert payload[0].shape == (1, 80, reference.config.text_config.hc_mult, 16)
        assert payload[1].dtype == torch.float32 and payload[1].requires_grad
        assert payload[2].dtype == torch.long and not payload[2].requires_grad
        # No kwargs are supplied to the second stage: its wire payload must
        # preserve packing, visual routing and Engram token history by itself.
        actual = last(*payload).logits
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        expected_loss = F.cross_entropy(expected.flatten(0, 1), labels.flatten())
        actual_loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten())
        expected_loss.backward()
        actual_loss.backward()
        reference_params = dict(reference.named_parameters())
        stage_params = dict(first.named_parameters()) | dict(last.named_parameters())
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


def test_pp_partition_keeps_source_boundary_and_vision_ownership() -> None:
    model = _model(full_schedule=True, engram=True)
    plan = model.customize_pipeline_stage_modules([[], []], layers_prefix="model.")
    assert [s for s in plan[0] if s.startswith("model.layers.")] == [f"model.layers.{i}" for i in range(14)]
    assert [s for s in plan[1] if s.startswith("model.layers.")] == [f"model.layers.{i}" for i in range(14, 40)]
    assert "model.vision" in plan[0] and "model.vision" not in plan[1]
    assert "model.engram_hash" in plan[0] and "model.engram_hash" in plan[1]
    assert "lm_head" in plan[1] and "lm_head" not in plan[0]
    with pytest.raises(ValueError, match="two pipeline stages"):
        model.customize_pipeline_stage_modules([[], [], []], layers_prefix="model.")


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
