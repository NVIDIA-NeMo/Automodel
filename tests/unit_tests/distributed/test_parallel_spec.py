# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""ParallelSpec is data: nothing in it points at a model attribute; model-derived specs come from a classmethod."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch.nn as nn
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel

from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import validate_tp_mesh


def test_spec_fields_are_plans_constraints_and_a_strategy():
    assert [f.name for f in dataclasses.fields(ParallelSpec)] == [
        "tp_plan",
        "sequence_parallel_plan",
        "layer_groups",
        "sharded_output_only",
        "strategy",
    ]


class _HFLike(nn.Module):
    """The shape transformers gives a model: an assembled ``_tp_plan`` and ``get_input_embeddings()``."""

    def __init__(self, plan, with_embedding_api: bool = True):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(4, 4)
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Linear(4, 4)
        self._tp_plan = plan
        if not with_embedding_api:
            self.get_input_embeddings = None  # a module without the transformers API

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_from_hf_model_builds_a_data_spec_from_the_assembled_plan():
    spec = ParallelSpec.from_hf_model(_HFLike({"model.layers.*.mlp": "colwise"}))
    assert isinstance(spec, ParallelSpec)
    assert type(spec.tp_plan["model.layers.*.mlp"]) is ColwiseParallel
    assert type(spec.tp_plan["model.embed_tokens"]) is RowwiseParallel  # located via get_input_embeddings()
    assert spec.sequence_parallel_plan is None and spec.strategy is None


def test_from_hf_model_is_none_when_nothing_translates():
    moe_only = {"model.layers.*.experts": "local_rowwise"}
    assert ParallelSpec.from_hf_model(_HFLike(moe_only, with_embedding_api=False)) is None
    # with the embedding API the row-sharded embedding alone is a (tiny) plan
    assert set(ParallelSpec.from_hf_model(_HFLike(moe_only)).tp_plan) == {"model.embed_tokens"}


def test_from_hf_model_requires_a_plan():
    with pytest.raises(AssertionError, match="Hugging Face tp plan is not supported"):
        ParallelSpec.from_hf_model(_HFLike(None))


class _CompositeConfig:
    """A VLM-style config: head counts live on the text config, reached through transformers' API."""

    def __init__(self, heads, kv_heads):
        self.text_config = SimpleNamespace(num_attention_heads=heads, num_key_value_heads=kv_heads)

    def get_text_config(self):
        return self.text_config


def _model_with(config):
    model = nn.Module()
    model.config = config
    return model


def test_validate_tp_mesh_reads_the_text_config_through_the_config_api():
    mesh = SimpleNamespace(size=lambda: 4)
    validate_tp_mesh(_model_with(_CompositeConfig(heads=8, kv_heads=4)), mesh)
    with pytest.raises(AssertionError, match="num_key_value_heads"):
        validate_tp_mesh(_model_with(_CompositeConfig(heads=8, kv_heads=2)), mesh)


def test_validate_tp_mesh_falls_back_to_the_config_itself():
    mesh = SimpleNamespace(size=lambda: 2)
    validate_tp_mesh(_model_with(SimpleNamespace(num_attention_heads=8, num_key_value_heads=2)), mesh)
    validate_tp_mesh(_model_with(SimpleNamespace(num_attention_heads=8, num_key_value_heads=None)), mesh)
    with pytest.raises(AssertionError, match="num_attention_heads"):
        validate_tp_mesh(_model_with(SimpleNamespace(num_attention_heads=3, num_key_value_heads=None)), mesh)
