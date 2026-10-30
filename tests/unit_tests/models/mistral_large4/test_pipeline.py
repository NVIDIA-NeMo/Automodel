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

"""Compare pipeline partitions with the unsplit model, including every gradient."""

import copy
from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.models.mistral_large4.multimodal import MistralLarge4ForConditionalGeneration
from nemo_automodel.components.models.mistral_large4.vision import MistralLarge4VisionConfig
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
from tests.unit_tests.models.mistral_large4.math_reference import tiny_config
from tests.unit_tests.models.mistral_large4.test_model import make_model
from tests.unit_tests.models.mistral_large4.test_multimodal import image_batch


def make_pipeline_model(*, dtype: torch.dtype = torch.float32, multimodal: bool = False) -> MistralLarge4ForCausalLM:
    """Create a four-layer sharing fixture with matrix [out,in] parameters."""
    torch.manual_seed(417)
    vision = MistralLarge4VisionConfig(
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_attention_heads=2,
        patch_size=2,
        image_size=16,
    )
    config = tiny_config(
        dtype=dtype,
        n_routed_experts=4,
        nope_layer_flags=[False, True, False, True],
        indexer_nope_layer_flags=[False, True, False, True],
        kv_source_layers=[0, 1, 0, 1],
        index_sharing=["full", "full", "shared", "full"],
        vision_config=asdict(vision) if multimodal else None,
    )
    model = (MistralLarge4ForConditionalGeneration if multimodal else MistralLarge4ForCausalLM)(config)
    model.initialize_weights()
    with torch.no_grad():
        model.model.rotary_emb.alpha.copy_(torch.tensor([0.03, -0.07]))
    return model


def partition_model(
    model: MistralLarge4ForCausalLM, cuts: tuple[int, ...], *, ac: bool = False
) -> list[MistralLarge4ForCausalLM]:
    """Copy contiguous model stages while preserving global parameter identities."""
    parts = []
    for start, stop in zip(cuts, cuts[1:]):
        part = copy.deepcopy(model)
        part.model.layers = nn.ModuleDict({str(i): part.model.layers[i] for i in range(start, stop)})
        if start:
            part.model.embed_tokens = None
            part.model.rotary_emb = None
            if isinstance(part, MistralLarge4ForConditionalGeneration):
                for name in ("vision_encoder", "pre_mm_projector_norm", "patch_merger", "vision_language_adapter"):
                    setattr(part, name, None)
        if stop < model.config.num_hidden_layers:
            part.model.norm = None
            part.lm_head = None
        if ac:
            for name, block in part.model.layers.items():
                part.model.layers[name] = checkpoint_wrapper(block)
        parts.append(part)
    return parts


def run_partition(parts: list[MistralLarge4ForCausalLM], ids: torch.Tensor, **kwargs) -> torch.Tensor:
    """Execute stage forwards with IDs [batch, sequence] and optional image/mask tensors.

    Args:
        parts: Ordered decoder stages.
        ids: Integer token IDs [batch, sequence].
        kwargs: attention_mask/position_ids [batch, sequence], or pixel_values
            containing complete [channels, height, width] images.

    Returns:
        FP32 capped logits [batch, sequence, vocab].
    """
    value = parts[0](ids, **kwargs)
    for part in parts[1:]:
        value = part(*value)
    return value


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ac", [False, True])
@pytest.mark.parametrize("cuts", [(0, 2, 4), (0, 1, 2, 3, 4)])
@pytest.mark.parametrize("multimodal", [False, True])
def test_partition_logits_every_gradient_norm_and_adamw_update(dtype, ac, cuts, multimodal):
    reference = make_pipeline_model(dtype=dtype, multimodal=multimodal)
    parts = partition_model(reference, cuts, ac=ac)
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9]])
    kwargs = {}
    if multimodal:
        ids, images = image_batch()
        kwargs["pixel_values"] = images
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, -1] = False
    kwargs.update(attention_mask=mask, position_ids=torch.arange(ids.shape[1]).flip(0)[None].expand_as(ids))
    expected = reference(ids, **kwargs).logits
    actual = run_partition(parts, ids, **kwargs)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    # Several live microbatches must own independent KV/index/rotary graphs.
    later_expected = reference(ids, **kwargs).logits
    later_actual = run_partition(parts, ids, **kwargs)
    upstream = torch.randn_like(actual)
    later_upstream = torch.randn_like(actual)
    later_actual.backward(later_upstream)
    later_expected.backward(later_upstream)
    actual.backward(upstream)
    expected.backward(upstream)
    parameters = {}
    for part in parts:
        for name, parameter in part.named_parameters():
            name = canonical_parameter_fqn(name)
            assert name not in parameters, f"Duplicated PP parameter {name}"
            parameters[name] = parameter
    reference_parameters = dict(reference.named_parameters())
    assert parameters.keys() == reference_parameters.keys()
    assert sum(name == "model.rotary_emb.alpha" for name in parameters) == 1
    # BF16 shared-source gradient branches may combine in a different order.
    # Allow one-to-two BF16 rounding steps; FP32 checks remain strict.
    atol, rtol = (3e-6, 5e-5) if dtype == torch.float32 else (0.004, 0.02)
    for name, parameter in parameters.items():
        ref = reference_parameters[name]
        if not parameter.requires_grad:
            assert parameter.grad is None and ref.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            torch.testing.assert_close(parameter.grad, ref.grad, atol=atol, rtol=rtol, msg=name)
    for name in (
        "model.rotary_emb.alpha",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.1.self_attn.kv_a_proj_with_mqa.weight",
    ):
        assert parameters[name].grad.float().norm() > 0, name
    actual_norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), 0.25, foreach=False)
    expected_norm = torch.nn.utils.clip_grad_norm_(reference.parameters(), 0.25, foreach=False)
    torch.testing.assert_close(actual_norm, expected_norm, atol=atol, rtol=rtol)
    actual_optim = torch.optim.AdamW(parameters.values(), lr=1e-5)
    expected_optim = torch.optim.AdamW(reference.parameters(), lr=1e-5)
    actual_optim.step()
    expected_optim.step()
    for name, parameter in parameters.items():
        torch.testing.assert_close(
            parameter, reference_parameters[name], atol=2e-6 if dtype == torch.float32 else 2e-4, rtol=rtol, msg=name
        )


def test_stage_metadata_matches_native_mixed_dtypes_and_shared_kv():
    model = make_pipeline_model(dtype=torch.bfloat16)
    parts = partition_model(model, (0, 1, 2, 3, 4))
    value = (torch.tensor([[1, 2, 3, 4, 5, 6]]),)
    for index, part in enumerate(parts):
        inputs, outputs = part.get_pipeline_stage_metas(
            is_first=index == 0, microbatch_size=1, seq_len=6, dtype=torch.bfloat16
        )
        assert [(t.shape, t.dtype) for t in value] == [(t.shape, t.dtype) for t in inputs]
        actual = part(*value)
        value = actual if isinstance(actual, tuple) else (actual,)
        assert [(t.shape, t.dtype) for t in value] == [(t.shape, t.dtype) for t in outputs]
        if index != len(parts) - 1:
            assert value[1].dtype == torch.float32 and value[1].requires_grad


def test_stage_checkpoint_key_union_and_exact_roundtrip():
    model = make_pipeline_model(multimodal=True)
    parts = partition_model(model, (0, 2, 4))
    state = {}
    exported = {}
    for part in parts:
        local = part.state_dict()
        assert not state.keys() & local.keys()
        state.update(local)
        converted = part.state_dict_adapter.to_hf(local)
        assert not exported.keys() & converted.keys()
        exported.update(converted)
    model.load_state_dict(state, strict=True)
    expected = model.state_dict_adapter.to_hf(model.state_dict())
    assert exported.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(exported[name], expected[name], atol=0, rtol=0)


def test_final_fused_loss_input_is_uncapped_hidden_state():
    model = make_pipeline_model()
    parts = partition_model(model, (0, 2, 4))
    parts[-1]._pp_return_hidden_states = True
    ids = torch.tensor([[1, 2, 3, 4]])
    expected = model(ids, output_hidden_states=True)
    hidden = run_partition(parts, ids)
    torch.testing.assert_close(hidden, expected.hidden_states, atol=0, rtol=0)
    _, outputs = parts[-1].get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=4, dtype=torch.float32)
    assert outputs[0].shape == hidden.shape


def test_invalid_stage_ownership_missing_transport_and_packed_inputs_fail():
    model = make_pipeline_model()
    first, last = partition_model(model, (0, 2, 4))
    ids = torch.tensor([[1, 2, 3, 4]])
    with pytest.raises(ValueError, match="incomplete"):
        last(torch.randn(1, 4, model.config.hidden_size))
    with pytest.raises(ValueError, match="model-prepared packed THD"):
        first(ids, qkv_format="thd", cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(ValueError, match="labels in the pipeline loss"):
        first(ids, labels=ids)
    last.model.rotary_emb = copy.deepcopy(first.model.rotary_emb)
    with pytest.raises(ValueError, match="exclusively own learnable LeRoPE"):
        last.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=4, dtype=torch.float32)


def test_generated_stage_modules_own_rotary_and_vision_once():
    model = make_pipeline_model(multimodal=True)
    proposed = [
        ["model.embed_tokens", "model.rotary_emb", "model.layers.0", "model.layers.1"],
        ["model.rotary_emb", "model.layers.2", "model.layers.3", "model.norm", "lm_head"],
    ]
    result = model.customize_pipeline_stage_modules(proposed, layers_prefix="model.")
    assert "model.rotary_emb" in proposed[1]  # Input descriptions remain untouched.
    assert "model.rotary_emb" in result[0] and "model.rotary_emb" not in result[1]
    for name in ("vision_encoder", "pre_mm_projector_norm", "patch_merger", "vision_language_adapter"):
        assert name in result[0] and name not in result[1]


def test_media_cursor_consumes_text_only_and_image_microbatches():
    model = make_pipeline_model(multimodal=True)
    first, last = partition_model(model, (0, 2, 4))
    image_ids, images = image_batch()
    text_ids = torch.tensor([[1, 2, 3, 4]])
    first._vlm_pixel_values_chunks = [[], images]
    first._vlm_chunk_idx = 0
    for ids, pixels in ((text_ids, []), (image_ids, images)):
        expected = model(ids, pixel_values=pixels).logits
        actual = last(*first(ids))
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert first._vlm_chunk_idx == 2
    with pytest.raises(ValueError, match="cursor exceeds"):
        first(text_ids)


@pytest.mark.parametrize("cuts", [(0, 36, 72), (0, 18, 36, 54, 72)])
def test_released_layer_schedule_transports_source35_through_final_consumer(cuts):
    """Retain all 72 layer identities and 35→39..71 sharing at reduced width."""
    reference = make_model(num_hidden_layers=72, nope_layer_flags=None, kv_source_layers=None, index_sharing=None)
    parts = partition_model(reference, cuts, ac=True)
    ids = torch.tensor([[1, 5, 7, 2, 8]])
    expected = reference(ids).logits
    actual = run_partition(parts, ids)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    refs = dict(reference.named_parameters())
    for part in parts:
        for name, parameter in part.named_parameters():
            if parameter.requires_grad:
                name = canonical_parameter_fqn(name)
                torch.testing.assert_close(parameter.grad, refs[name].grad, atol=4e-6, rtol=5e-5, msg=name)
    assert parts[-1]._pipeline_live_sources(cuts[-2]) == [35]
    assert refs["model.layers.35.self_attn.kv_a_proj_with_mqa.weight"].grad.norm() > 0
