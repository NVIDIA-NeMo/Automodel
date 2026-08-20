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
from nemo_automodel.components.distributed.pipelining.module_plan import generate_hf_model_fqn_per_model_part


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
