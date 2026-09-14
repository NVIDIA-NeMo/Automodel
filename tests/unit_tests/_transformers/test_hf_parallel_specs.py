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

"""The HF bridge binds ParallelSpecs for classes Automodel does not own."""

import pytest
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

from nemo_automodel._transformers.hf_parallel_specs import (
    HF_PARALLEL_SPECS,
    MISTRAL3_VLM_PARALLEL_SPEC,
    parallel_spec_for,
    register_parallel_strategy,
    validate_tp_mesh_for_nemotron_nas,
)
from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.distributed.parallelizer import ParallelizationStrategy, get_parallelization_strategy
from nemo_automodel.components.models.llama.model import LlamaForCausalLM as NativeLlamaForCausalLM
from nemo_automodel.components.models.llama.parallelization import LLAMA_PARALLEL_SPEC


def test_hf_only_architecture_uses_bridge_table():
    assert parallel_spec_for(Gemma3ForConditionalGeneration) is HF_PARALLEL_SPECS["Gemma3ForConditionalGeneration"]


def test_native_implementation_is_authoritative_for_its_transformers_twin():
    assert NativeLlamaForCausalLM.parallel_spec is LLAMA_PARALLEL_SPEC
    assert parallel_spec_for(LlamaForCausalLM) is LLAMA_PARALLEL_SPEC


def test_explicit_entry_wins_over_registry_twin():
    """``Mistral3ForConditionalGeneration`` maps to the mistral4 port in the registry, but the
    transformers class must keep the Mistral3 VLM contract."""
    assert parallel_spec_for(Mistral3ForConditionalGeneration) is MISTRAL3_VLM_PARALLEL_SPEC


def test_subclasses_inherit_through_the_mro():
    remote_subclass = type("Mistral3FP8Custom", (Mistral3ForConditionalGeneration,), {})
    assert parallel_spec_for(remote_subclass) is MISTRAL3_VLM_PARALLEL_SPEC


def test_remote_code_nemotron_nas_binds_its_validator():
    spec = parallel_spec_for(type("DeciLMForCausalLM", (), {}))
    assert spec.validate_tp is validate_tp_mesh_for_nemotron_nas
    assert spec.tp_plan is not None


def test_unknown_class_has_no_contract():
    assert parallel_spec_for(type("SomethingElseForCausalLM", (nn.Module,), {})) is None


def test_wrapper_class_carries_the_spec_and_keeps_declared_ones():
    wrapped = _get_mixin_wrapped_class(LlamaForCausalLM)
    assert wrapped.parallel_spec is LLAMA_PARALLEL_SPEC
    assert wrapped.__name__ == "LlamaForCausalLM"

    declared = ParallelSpec(tp_plan=lambda m, sp: {})
    owner = type("LlamaForCausalLM", (nn.Module,), {"parallel_spec": declared})
    assert _get_mixin_wrapped_class(owner).parallel_spec is declared


@pytest.fixture
def _restore_bridge_table():
    original = dict(HF_PARALLEL_SPECS)
    yield
    HF_PARALLEL_SPECS.clear()
    HF_PARALLEL_SPECS.update(original)


def test_register_parallel_strategy_binds_by_class_name(_restore_bridge_table):
    @register_parallel_strategy(name="OutOfTreeModel")
    class OutOfTreeModelStrat(ParallelizationStrategy):
        def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
            return model

    model_cls = type("OutOfTreeModel", (nn.Module,), {})
    model = model_cls()
    model.__class__ = _get_mixin_wrapped_class(model_cls)

    strategy = get_parallelization_strategy(model)
    assert strategy is HF_PARALLEL_SPECS["OutOfTreeModel"].strategy
    assert isinstance(strategy, OutOfTreeModelStrat)
    assert strategy.parallelize(model) is model


def test_register_parallel_strategy_keeps_existing_tp_plan(_restore_bridge_table):
    def tp_plan(model, sequence_parallel):
        return {}

    HF_PARALLEL_SPECS["MergedModel"] = ParallelSpec(tp_plan=tp_plan)

    @register_parallel_strategy(name="MergedModel")
    class MergedStrat(ParallelizationStrategy):
        def parallelize(self, model: nn.Module, *args, **kwargs) -> nn.Module:
            return model

    assert HF_PARALLEL_SPECS["MergedModel"].tp_plan is tp_plan
    assert isinstance(HF_PARALLEL_SPECS["MergedModel"].strategy, MergedStrat)


def test_register_parallel_strategy_requires_a_strategy_and_a_name():
    with pytest.raises(ValueError):

        @register_parallel_strategy
        class NotAStrategy:
            pass

    with pytest.raises(AssertionError):

        @register_parallel_strategy(name="Whatever")
        class StillNotAStrategy:
            pass
