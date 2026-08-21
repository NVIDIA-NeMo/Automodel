# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import types

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.distributed.pipelining.model_parts import split_model_into_parts
from nemo_automodel.components.distributed.pipelining.module_plan import (
    PipelineStagePlanError,
    generate_hf_model_fqn_per_model_part,
)
from nemo_automodel.components.distributed.pipelining.schedules import resolve_pipeline_schedule


class DummyRotaryEmb(nn.Module):
    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        """Return a zero rotary embedding with the input layout.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            position_ids: Tensor of shape [batch, sequence].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        del position_ids
        return torch.zeros_like(hidden_states)


class DummyDecoderLayer(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False, device="meta")

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor]:
        """Pass hidden states through unchanged.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            **kwargs: Additional non-tensor layer arguments.

        Returns:
            One tensor of shape [batch, sequence, hidden].
        """
        del kwargs
        return (hidden_states,)


class DummyInnerModel(nn.Module):
    def __init__(self, *, vocab_size: int = 128, hidden_size: int = 64, num_layers: int = 8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size, device="meta")
        self.layers = nn.ModuleList([DummyDecoderLayer() for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_size, device="meta")
        self.rotary_emb = DummyRotaryEmb()


class DummyQwenForCausalLM(nn.Module):
    def __init__(self, *, num_layers: int = 8):
        super().__init__()
        self.model = DummyInnerModel(num_layers=num_layers)
        self.lm_head = nn.Linear(64, 128, device="meta")
        self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_cache=False)


class FakePPMesh:
    def __init__(self, *, size: int, rank: int):
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


@pytest.mark.parametrize(("pp_size", "rank"), [(2, 0), (2, 1), (4, 0), (4, 3)])
def test_split_model_materializes_only_local_parts(pp_size, rank):
    model = DummyQwenForCausalLM(num_layers=8)
    num_stages = pp_size * 2
    module_fqns = generate_hf_model_fqn_per_model_part(num_stages=num_stages, num_layers=8)

    parts = split_model_into_parts(
        model,
        FakePPMesh(size=pp_size, rank=rank),
        "interleaved1f1b",
        module_fqns,
    )

    expected_indices = (rank, rank + pp_size)
    assert tuple(part.stage_index for part in parts) == expected_indices
    assert all(part.num_stages == num_stages for part in parts)
    for part, stage_index in zip(parts, expected_indices):
        expected_layers = {
            name.rsplit(".", 1)[-1] for name in module_fqns[stage_index] if name.startswith("model.layers.")
        }
        assert isinstance(part.module.model.layers, nn.ModuleDict)
        assert set(part.module.model.layers) == expected_layers
        assert (part.module.model.embed_tokens is not None) is (stage_index == 0)
        assert (part.module.model.norm is not None) is (stage_index == num_stages - 1)
        assert (part.module.lm_head is not None) is (stage_index == num_stages - 1)
        assert part.module.model.rotary_emb is not None


def test_split_model_preserves_meta_parameters():
    model = DummyQwenForCausalLM(num_layers=4)
    parts = split_model_into_parts(
        model,
        FakePPMesh(size=2, rank=0),
        "PipelineScheduleSingle",
        generate_hf_model_fqn_per_model_part(num_stages=2, num_layers=4),
    )

    assert len(parts) == 1
    assert all(parameter.device.type == "meta" for parameter in parts[0].module.parameters())


@pytest.mark.parametrize(
    ("schedule", "layers_per_stage", "expected_parts"),
    [
        ("PipelineScheduleSingle", 4, 1),
        ("PipelineScheduleMulti", 2, 2),
        ("interleaved1f1b", 2, 2),
    ],
)
def test_split_model_derives_virtual_stage_count(schedule, layers_per_stage, expected_parts):
    parts = split_model_into_parts(
        DummyQwenForCausalLM(num_layers=8),
        FakePPMesh(size=2, rank=0),
        schedule,
        layers_per_stage=layers_per_stage,
        round_to_pp_multiple="up",
    )

    assert len(parts) == expected_parts
    assert all(part.num_stages == expected_parts * 2 for part in parts)


def test_zero_bubble_uses_v_stage_assignment():
    module_fqns = generate_hf_model_fqn_per_model_part(num_stages=4, num_layers=8)
    parts = split_model_into_parts(
        DummyQwenForCausalLM(num_layers=8),
        FakePPMesh(size=2, rank=0),
        "ZBVZeroBubble",
        module_fqns,
    )

    assert tuple(part.stage_index for part in parts) == (0, 3)


class DummyVisionTower(nn.Module):
    def __init__(self, hidden_size: int = 64):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, device="meta")

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Project image features.

        Args:
            pixel_values: Tensor of shape [batch, patches, hidden].

        Returns:
            Tensor of shape [batch, patches, hidden].
        """
        return self.proj(pixel_values)


class DummyVlmInnerModel(DummyInnerModel):
    def __init__(self, *, num_layers: int = 4):
        super().__init__(num_layers=num_layers)
        self.vision_tower = DummyVisionTower()


class DummyVlmForConditionalGeneration(nn.Module):
    def __init__(self, *, num_layers: int = 4):
        super().__init__()
        self.model = DummyVlmInnerModel(num_layers=num_layers)
        self.lm_head = nn.Linear(64, 128, device="meta")
        self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_cache=False)


class DummyTextWrapper(nn.Module):
    """Container whose text stack is one level deeper and which owns ``lm_head``."""

    def __init__(self, *, num_layers: int = 4):
        super().__init__()
        self.model = DummyInnerModel(num_layers=num_layers)
        self.lm_head = nn.Linear(64, 128, device="meta")


class DummyNestedForCausalLM(nn.Module):
    """``model.model.layers`` layout with no ``language_model``/``text_model`` attribute."""

    def __init__(self, *, num_layers: int = 4):
        super().__init__()
        self.model = DummyTextWrapper(num_layers=num_layers)
        self.config = types.SimpleNamespace(output_attentions=False, output_hidden_states=False, use_cache=False)


def _text_only_plan(num_stages: int, num_layers: int) -> list[list[str]]:
    return generate_hf_model_fqn_per_model_part(
        num_stages=num_stages,
        num_layers=num_layers,
        include_multimodal_encoders=False,
    )


def test_generated_plan_drops_multimodal_fqns_absent_from_a_text_only_model():
    parts = split_model_into_parts(
        DummyQwenForCausalLM(num_layers=4),
        FakePPMesh(size=2, rank=0),
        "PipelineScheduleSingle",
        layers_per_stage=2,
    )

    assert len(parts) == 1
    assert parts[0].module.model.embed_tokens is not None
    assert not hasattr(parts[0].module.model, "vision_tower")


def test_generated_plan_keeps_a_multimodal_encoder_the_model_actually_owns():
    parts = split_model_into_parts(
        DummyVlmForConditionalGeneration(num_layers=4),
        FakePPMesh(size=1, rank=0),
        "PipelineScheduleMulti",
        layers_per_stage=2,
    )

    assert [part.stage_index for part in parts] == [0, 1]
    assert parts[0].module.model.vision_tower is not None
    assert parts[1].module.model.vision_tower is None


def test_split_model_rejects_an_fqn_that_resolves_to_nothing():
    plan = _text_only_plan(num_stages=2, num_layers=4)
    plan[1].append("model.layers.4.mlp")

    with pytest.raises(PipelineStagePlanError) as excinfo:
        split_model_into_parts(
            DummyQwenForCausalLM(num_layers=4),
            FakePPMesh(size=2, rank=0),
            "PipelineScheduleSingle",
            plan,
        )

    message = str(excinfo.value)
    assert "stage 1" in message
    assert "model.layers.4.mlp" in message


def test_split_model_rejects_a_stage_without_transformer_layers():
    plan = [
        ["model.embed_tokens", *[f"model.layers.{index}" for index in range(4)]],
        ["model.norm", "lm_head"],
    ]

    with pytest.raises(PipelineStagePlanError, match=r"stages \[1\] own no transformer layer"):
        split_model_into_parts(
            DummyQwenForCausalLM(num_layers=4),
            FakePPMesh(size=2, rank=0),
            "PipelineScheduleSingle",
            plan,
        )


def test_split_model_rejects_a_layer_owned_by_no_stage():
    plan = [
        ["model.embed_tokens", "model.layers.0", "model.layers.1"],
        ["model.layers.3", "model.norm", "lm_head"],
    ]

    with pytest.raises(PipelineStagePlanError, match="layers assigned to no stage") as excinfo:
        split_model_into_parts(
            DummyQwenForCausalLM(num_layers=4),
            FakePPMesh(size=2, rank=0),
            "PipelineScheduleSingle",
            plan,
        )

    assert "model.layers.2" in str(excinfo.value)


def test_split_model_rejects_a_layer_owned_by_several_stages():
    plan = [
        ["model.embed_tokens", "model.layers.0", "model.layers.1", "model.layers.2"],
        ["model.layers.2", "model.layers.3", "model.norm", "lm_head"],
    ]

    with pytest.raises(PipelineStagePlanError, match="layers assigned to several stages") as excinfo:
        split_model_into_parts(
            DummyQwenForCausalLM(num_layers=4),
            FakePPMesh(size=2, rank=0),
            "PipelineScheduleSingle",
            plan,
        )

    assert "model.layers.2 -> stages [0, 1]" in str(excinfo.value)


def test_nested_text_model_keeps_its_lm_head():
    """A ``model.model.layers`` layout must not produce the ``model..lm_head`` FQN."""
    parts = split_model_into_parts(
        DummyNestedForCausalLM(num_layers=4),
        FakePPMesh(size=2, rank=1),
        "PipelineScheduleSingle",
        layers_per_stage=2,
    )

    assert len(parts) == 1 and parts[0].stage_index == 1
    assert parts[0].module.model.lm_head is not None
    assert parts[0].module.model.model.norm is not None


def test_explicit_plan_skips_virtual_stage_computation():
    """An explicit plan must not be rejected by the plan-generation arithmetic."""
    plan = _text_only_plan(num_stages=2, num_layers=8)

    parts = split_model_into_parts(
        DummyQwenForCausalLM(num_layers=8),
        FakePPMesh(size=2, rank=0),
        "interleaved1f1b",
        plan,
        layers_per_stage=8,
    )

    assert [part.stage_index for part in parts] == [0]


def test_module_list_to_module_dict_preserves_numeric_layer_order():
    plan = [
        ["model.embed_tokens", *[f"model.layers.{index}" for index in (0, 1, 3, 4, 5)]],
        [f"model.layers.{index}" for index in (2, 10, 11)],
        [*[f"model.layers.{index}" for index in (6, 7, 8, 9)], "model.norm", "lm_head"],
    ]

    parts = split_model_into_parts(
        DummyQwenForCausalLM(num_layers=12),
        FakePPMesh(size=3, rank=1),
        "PipelineScheduleMulti",
        plan,
    )

    assert len(parts) == 1 and parts[0].stage_index == 1
    layers = parts[0].module.model.layers
    assert isinstance(layers, nn.ModuleDict)
    assert list(layers) == ["2", "10", "11"]
    assert [key for key in parts[0].module.state_dict() if key.startswith("model.layers.")] == [
        "model.layers.2.proj.weight",
        "model.layers.10.proj.weight",
        "model.layers.11.proj.weight",
    ]


@pytest.mark.parametrize(
    ("schedule", "expected_style"),
    [("1f1b", "loop"), ("interleaved1f1b", "loop"), ("ZBVZeroBubble", "v")],
)
def test_resolve_pipeline_schedule_returns_the_stage_assignment_style(schedule, expected_style):
    _, style = resolve_pipeline_schedule(schedule)
    assert style == expected_style


@pytest.mark.parametrize("schedule", ["zero_bubble", "v_schedule", "looped_bfs", "dfs"])
def test_resolve_pipeline_schedule_rejects_unregistered_names(schedule):
    with pytest.raises(ValueError, match="Unknown pipeline schedule") as excinfo:
        resolve_pipeline_schedule(schedule)

    assert "Interleaved1F1B" in str(excinfo.value)


def test_resolve_pipeline_schedule_rejects_schedules_without_a_stage_mapping():
    with pytest.raises(ValueError, match="stage assignment is not implemented"):
        resolve_pipeline_schedule("dualpipev")


def test_split_model_rejects_schedules_without_a_stage_mapping():
    with pytest.raises(ValueError, match="stage assignment is not implemented"):
        split_model_into_parts(
            DummyQwenForCausalLM(num_layers=4),
            FakePPMesh(size=2, rank=0),
            "dualpipev",
            _text_only_plan(num_stages=4, num_layers=4),
        )
