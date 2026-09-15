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

"""The loader binds model-owned ParallelSpec declarations onto classes NeMo AutoModel does not re-implement."""

from types import SimpleNamespace

import pytest
import torch.nn as nn
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class, parallel_spec_for
from nemo_automodel.components.distributed.parallel_spec import ParallelSpec
from nemo_automodel.components.models import declared_parallel_spec, model_family
from nemo_automodel.components.models.gemma3 import parallelization as gemma3_parallelization
from nemo_automodel.components.models.llama.model import LlamaForCausalLM as NativeLlamaForCausalLM
from nemo_automodel.components.models.llama.parallelization import LLAMA_PARALLEL_SPEC
from nemo_automodel.components.models.mistral3.parallelization import MISTRAL3_VLM_PARALLEL_SPEC
from nemo_automodel.components.models.nemotron_nas.parallelization import validate_tp_mesh_for_nemotron_nas


def test_declaration_resolves_from_the_model_type_family():
    """Both Gemma 3 heads live in ``components/models/gemma3`` although their ``model_type`` values differ."""
    assert model_family(Gemma3ForCausalLM) == model_family(Gemma3ForConditionalGeneration) == "gemma3"
    causal = parallel_spec_for(Gemma3ForCausalLM)
    vlm = parallel_spec_for(Gemma3ForConditionalGeneration)
    assert causal is gemma3_parallelization.Gemma3ForCausalLM.parallel_spec
    assert vlm is gemma3_parallelization.Gemma3ForConditionalGeneration.parallel_spec
    assert causal is not vlm
    assert causal.tp_plan is vlm.tp_plan is gemma3_parallelization.gemma3_tp_plan


def test_native_implementation_is_authoritative_for_its_transformers_twin():
    assert NativeLlamaForCausalLM.parallel_spec is LLAMA_PARALLEL_SPEC
    assert parallel_spec_for(LlamaForCausalLM) is LLAMA_PARALLEL_SPEC


def test_declaration_wins_over_registry_twin():
    """``Mistral3ForConditionalGeneration`` maps to the mistral4 port in the registry, but the
    transformers class keeps the Mistral3 VLM contract declared in ``components/models/mistral3``."""
    assert parallel_spec_for(Mistral3ForConditionalGeneration) is MISTRAL3_VLM_PARALLEL_SPEC


def test_subclasses_inherit_through_the_mro():
    remote_subclass = type("Mistral3FP8Custom", (Mistral3ForConditionalGeneration,), {})
    assert parallel_spec_for(remote_subclass) is MISTRAL3_VLM_PARALLEL_SPEC


def test_remote_code_class_resolves_from_its_config_model_type():
    """trust_remote_code classes carry a snapshot-hashed module path; only ``config_class.model_type`` is stable."""
    remote = type("DeciLMForCausalLM", (), {"config_class": SimpleNamespace(model_type="nemotron-nas")})
    remote.__module__ = "transformers_modules.nemotron_nas.modeling_decilm"
    spec = parallel_spec_for(remote)
    assert spec.validate_tp is validate_tp_mesh_for_nemotron_nas
    assert spec.tp_plan is not None


def test_classes_without_a_declaration_have_no_contract():
    assert parallel_spec_for(type("SomethingElseForCausalLM", (nn.Module,), {})) is None
    unknown_head = type("LlamaForOddHead", (nn.Module,), {"config_class": SimpleNamespace(model_type="llama")})
    assert parallel_spec_for(unknown_head) is None


def test_wrapper_class_carries_the_spec_and_keeps_declared_ones():
    wrapped = _get_mixin_wrapped_class(LlamaForCausalLM)
    assert wrapped.parallel_spec is LLAMA_PARALLEL_SPEC
    assert wrapped.__name__ == "LlamaForCausalLM"

    declared = ParallelSpec(tp_plan=lambda m, sp: {})
    owner = type("LlamaForCausalLM", (nn.Module,), {"parallel_spec": declared})
    assert _get_mixin_wrapped_class(owner).parallel_spec is declared


def test_attribute_set_on_an_upstream_class_is_the_out_of_tree_hook():
    """A class you cannot ship a package for gets the attribute directly; the wrapper inherits it."""
    declared = ParallelSpec(tp_plan=lambda m, sp: {})
    upstream = type("ThirdPartyForCausalLM", (nn.Module,), {})
    upstream.parallel_spec = declared
    assert _get_mixin_wrapped_class(upstream).parallel_spec is declared


@pytest.mark.parametrize("model_type", ["does_not_exist", "..evil", "components.models", ""])
def test_declared_parallel_spec_ignores_missing_or_malformed_families(model_type):
    double = type("Anything", (), {"config_class": SimpleNamespace(model_type=model_type)})
    assert declared_parallel_spec(double) is None


def test_declared_parallel_spec_reads_the_class_named_after_the_architecture():
    assert declared_parallel_spec(Gemma3ForCausalLM) is gemma3_parallelization.Gemma3ForCausalLM.parallel_spec
    other_head = type("Gemma3Model", (), {"config_class": Gemma3ForCausalLM.config_class})
    assert declared_parallel_spec(other_head) is None
