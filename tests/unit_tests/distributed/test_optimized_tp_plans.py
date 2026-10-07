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

"""Layout and selection regressions for model-owned tensor-parallel plans."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, ParallelStyle, RowwiseParallel, SequenceParallel
from torch.distributed.tensor.placement_types import Replicate, Shard
from transformers.models.gemma3.modeling_gemma3 import Gemma3ForCausalLM, Gemma3ForConditionalGeneration
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForSequenceClassification

from nemo_automodel.components.distributed.parallel_styles import ReplicatedWithGradAllReduce
from nemo_automodel.components.distributed.tp_styles import RotaryEmbedParallel
from nemo_automodel.components.models.falcon_h1 import _parallelize_falcon_h1
from nemo_automodel.components.models.gemma3 import _parallelize_gemma3
from nemo_automodel.components.models.mistral3_vlm.parallelization import _parallelize_mistral3_vlm
from nemo_automodel.components.models.muse_glimmer.parallelization import _parallelize_muse_glimmer
from nemo_automodel.components.models.parallelization import (
    MODEL_PARALLELIZERS,
    _get_class_qualname,
    resolve_model_parallelizer,
)
from nemo_automodel.components.models.phi import _parallelize_phi
from nemo_automodel.components.models.qwen2.model import Qwen2ForCausalLM as CustomQwen2ForCausalLM
from nemo_automodel.components.models.qwen2.parallelization import _parallelize_qwen
from nemo_automodel.components.models.qwen3.model import Qwen3ForCausalLM as CustomQwen3ForCausalLM

QWEN_MODELS = (Qwen2ForCausalLM, CustomQwen2ForCausalLM, Qwen3ForCausalLM, CustomQwen3ForCausalLM)
GEMMA_MODELS = (Gemma3ForCausalLM, Gemma3ForConditionalGeneration)
CAUSAL_MODELS = (LlamaForCausalLM, *QWEN_MODELS, *GEMMA_MODELS)


def _model(model_class, tied=False):
    model = Mock(spec=model_class)
    model.config = SimpleNamespace(tie_word_embeddings=tied)
    if model_class is Qwen3ForSequenceClassification:
        del model.lm_head
        model.score = torch.nn.Linear(4, 2)
    return model


def _sidecar_for_key(key):
    module, _, name = key.rpartition(".")
    return resolve_model_parallelizer(type(name or key, (), {"__module__": module or "transformers_modules.snapshot"}))


def _assert_decoder(plan, prefix="model", mlp="mlp", extra_columns=()):
    columns = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", f"{mlp}.up_proj", f"{mlp}.gate_proj")
    for name in (*columns, *extra_columns):
        assert isinstance(plan[f"{prefix}.layers.*.{name}"], ColwiseParallel)
    for name in ("self_attn.o_proj", f"{mlp}.down_proj"):
        assert isinstance(plan[f"{prefix}.layers.*.{name}"], RowwiseParallel)
    assert isinstance(plan[f"{prefix}.embed_tokens"], RowwiseParallel)
    assert isinstance(plan["lm_head"], ColwiseParallel)


def test_rotary_keeps_dtensor_inputs():
    inputs = (Mock(spec=DTensor), Mock(spec=DTensor))
    result = RotaryEmbedParallel._prepare_input_fn([Shard(1)], Mock(), inputs, Mock())
    assert type(result) is type(inputs)
    assert result == inputs


def test_rotary_converts_local_inputs():
    inputs, mesh = (torch.randn(4, 8), torch.randn(4, 8)), Mock()
    converted = (Mock(spec=DTensor), Mock(spec=DTensor))
    with patch.object(DTensor, "from_local", side_effect=converted) as from_local:
        result = RotaryEmbedParallel._prepare_input_fn([Shard(1)], Mock(), inputs, mesh)
    assert result == converted
    assert from_local.call_count == 2
    for call, tensor, placements, run_check in zip(
        from_local.call_args_list, inputs, ([Shard(1)], (Replicate(),)), (True, False)
    ):
        assert call.kwargs["local_tensor"] is tensor
        assert call.kwargs["device_mesh"] is mesh
        assert call.kwargs["placements"] == placements
        assert call.kwargs["run_check"] is run_check


def test_rotary_input_error_has_shape_and_rank():
    with (
        patch.object(DTensor, "from_local", side_effect=ValueError("Shape mismatch")),
        patch("torch.distributed.get_rank", return_value=1),
    ):
        with pytest.raises(ValueError, match="Failed to shard tensor for sequence parallelism.*rank 1.*Shape mismatch"):
            RotaryEmbedParallel._prepare_input_fn([Shard(1)], Mock(), (torch.randn(4, 8), torch.randn(4, 8)), Mock())


@pytest.mark.parametrize("local", [False, True])
def test_rotary_output_conversion(local):
    outputs = (Mock(spec=DTensor), Mock(spec=DTensor))
    result = RotaryEmbedParallel._prepare_output_fn(local, Mock(), outputs, Mock())
    assert type(result) is type(outputs)
    for actual, original in zip(result, outputs):
        assert original.to_local.called is local
        assert actual is (original.to_local.return_value if local else original)


@pytest.mark.parametrize("model_class", CAUSAL_MODELS)
@pytest.mark.parametrize("sequence_parallel", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_decoder_plans(model_class, sequence_parallel, tied):
    """Exercise each implementation, prefix, SP setting, and tied-weight configuration."""
    assert _get_class_qualname(model_class) in MODEL_PARALLELIZERS
    factory = resolve_model_parallelizer(model_class).tp_plan
    model = _model(model_class, tied)
    plan = factory(model, sequence_parallel)
    prefix = "model.language_model" if model_class is Gemma3ForConditionalGeneration else "model"
    _assert_decoder(plan, prefix)
    assert all(isinstance(key, str) and key and isinstance(style, ParallelStyle) for key, style in plan.items())
    if sequence_parallel:
        assert factory(model, False).keys() <= plan.keys()
        for name in ("norm", "layers.*.input_layernorm", "layers.*.post_attention_layernorm"):
            assert isinstance(plan[f"{prefix}.{name}"], SequenceParallel)
        assert plan[f"{prefix}.embed_tokens"].output_layouts == (Shard(1),)
        assert plan["lm_head"].input_layouts == (Shard(1),)
        if model_class in GEMMA_MODELS:
            for name in ("rotary_emb", "rotary_emb_local"):
                assert isinstance(plan[f"{prefix}.{name}"], RotaryEmbedParallel)
    if model_class in (*QWEN_MODELS, *GEMMA_MODELS):
        for name in ("q_norm", "k_norm"):
            style = plan[f"{prefix}.layers.*.self_attn.{name}"]
            assert isinstance(style, ReplicatedWithGradAllReduce)
            norm = torch.nn.LayerNorm(4)
            style._apply(norm, Mock())
            assert norm._nemo_tp_replica_grad_reduction == "sum"


@pytest.mark.parametrize("qk_layernorm", [False, True])
@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_phi_optional_qk_norms(qk_layernorm, sequence_parallel):
    plan = _parallelize_phi(SimpleNamespace(config=SimpleNamespace(qk_layernorm=qk_layernorm)), sequence_parallel)
    for name in ("q_layernorm", "k_layernorm"):
        key = f"model.layers.*.self_attn.{name}"
        assert (key in plan) is qk_layernorm
        if qk_layernorm:
            assert isinstance(plan[key], ReplicatedWithGradAllReduce)


@pytest.mark.parametrize("model_class", (*CAUSAL_MODELS, Qwen3ForSequenceClassification))
def test_registered_factory_returns_dict(model_class):
    assert _get_class_qualname(model_class) in MODEL_PARALLELIZERS
    assert isinstance(resolve_model_parallelizer(model_class).tp_plan(_model(model_class), False), dict)


def test_all_sidecars_implement_parallelize():
    for key in MODEL_PARALLELIZERS:
        assert callable(_sidecar_for_key(key).parallelize)


@pytest.mark.parametrize("model_class", QWEN_MODELS)
def test_qwen_implementations_share_plan(model_class):
    assert resolve_model_parallelizer(model_class).tp_plan is _parallelize_qwen


@pytest.mark.parametrize("model_class", GEMMA_MODELS)
def test_gemma_variants_share_plan(model_class):
    assert resolve_model_parallelizer(model_class).tp_plan is _parallelize_gemma3


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_mistral_vlm_language_scope(sequence_parallel):
    plan = _parallelize_mistral3_vlm(None, sequence_parallel)
    _assert_decoder(plan, "model.language_model")
    assert all(key == "lm_head" or key.startswith("model.language_model.") for key in plan)


def test_mistral_vlm_native_and_hf_registration():
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    from nemo_automodel.components.models.mistral3_vlm.model import Mistral3FP8VLMForConditionalGeneration

    for cls in (Mistral3ForConditionalGeneration, Mistral3FP8VLMForConditionalGeneration):
        assert _get_class_qualname(cls) in MODEL_PARALLELIZERS
        assert resolve_model_parallelizer(cls).tp_plan is _parallelize_mistral3_vlm


@pytest.mark.parametrize("sequence_parallel", [False, True])
def test_falcon_leaves_mamba_replicated(sequence_parallel):
    plan = _parallelize_falcon_h1(None, sequence_parallel)
    _assert_decoder(plan, mlp="feed_forward")
    assert not any(".mlp." in key or ".mamba" in key for key in plan)
    baseline = _parallelize_falcon_h1(None, False)
    assert {k: (type(v), vars(v)) for k, v in plan.items()} == {k: (type(v), vars(v)) for k, v in baseline.items()}


@pytest.mark.parametrize(
    "key", ["transformers.models.falcon_h1.modeling_falcon_h1.FalconH1ForCausalLM", "FalconH1ForCausalLM"]
)
def test_falcon_hf_and_remote_code_registration(key):
    assert key in MODEL_PARALLELIZERS
    assert _sidecar_for_key(key).tp_plan is _parallelize_falcon_h1


def test_muse_glimmer_shards_full_language_backbone():
    plan = _parallelize_muse_glimmer(None)
    _assert_decoder(plan, extra_columns=("self_attn.output_gate_proj",))
    assert not any("vision" in key for key in plan)
    with pytest.warns(UserWarning, match="not yet supported for MuseGlimmer"):
        plan_sp = _parallelize_muse_glimmer(None, True)
    assert {k: (type(v), vars(v)) for k, v in plan.items()} == {k: (type(v), vars(v)) for k, v in plan_sp.items()}


def test_muse_glimmer_registration_and_capability():
    from nemo_automodel._transformers.capabilities import _has_optimized_tp_plan
    from nemo_automodel.components.models.muse_glimmer.model import MuseGlimmerForConditionalGeneration

    key = "nemo_automodel.components.models.muse_glimmer.model.MuseGlimmerForConditionalGeneration"
    assert _sidecar_for_key(key).tp_plan is _parallelize_muse_glimmer
    assert _has_optimized_tp_plan(MuseGlimmerForConditionalGeneration)
