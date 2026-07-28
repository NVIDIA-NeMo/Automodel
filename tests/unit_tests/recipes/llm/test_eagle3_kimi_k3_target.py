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

"""Unit tests for the Kimi K3 frozen-target build inside the EAGLE-3 recipe.

The K3 target has no HuggingFace implementation and its routed experts do not fit
on one GPU, so the colocated path builds it from the (possibly nested) text config
with an explicit expert-parallel backend. These tests pin the backend defaults,
the unsupported-parallelism gates, and the resulting ``from_config`` call.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.recipes.llm.train_eagle3 import (
    KIMI_K3_MODEL_TYPE,
    TrainEagle3Recipe,
    _build_kimi_k3_target_backend,
    _validate_kimi_k3_gates,
)

_FROM_CONFIG = "nemo_automodel.recipes.llm.train_eagle3.NeMoAutoModelForCausalLM.from_config"


class _Cfg(SimpleNamespace):
    """Config-node stand-in: attribute access plus the ConfigNode-style ``get``."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _text_config() -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=8,
        num_experts_per_token=2,
        num_shared_experts=1,
        linear_attn_config={"kda_layers": [1, 2, 3], "full_attn_layers": [4]},
    )


def _gate_kwargs(**overrides):
    kwargs = dict(
        backend="colocated",
        cp_size=1,
        tp_size=1,
        pp_size=1,
        packed_sequence_size=0,
        parallel_drafting=False,
        target_force_hf=False,
    )
    kwargs.update(overrides)
    return kwargs


def _make_recipe(device: str = "cuda", dist_setup=object(), **cfg_values) -> TrainEagle3Recipe:
    recipe = TrainEagle3Recipe.__new__(TrainEagle3Recipe)
    recipe.device = torch.device(device)
    recipe.compute_dtype = torch.bfloat16
    recipe.cfg = _Cfg(**cfg_values)
    recipe.dist_setup = dist_setup
    return recipe


def test_text_config_model_type_is_the_dispatch_key():
    """The recipe dispatches on the text config's model_type, so it must match the constant."""
    assert _text_config().model_type == KIMI_K3_MODEL_TYPE


def test_backend_defaults_to_expert_parallel_fp32_norm():
    backend = _build_kimi_k3_target_backend(_Cfg())
    assert (backend.attn, backend.dispatcher, backend.experts) == ("eager", "torch", "torch_mm")
    assert backend.rms_norm == "torch_fp32"
    assert backend.enable_hf_state_dict_adapter and not backend.rope_fusion


def test_backend_honors_overrides():
    backend = _build_kimi_k3_target_backend(
        _Cfg(target_dispatcher="deepep", target_experts="gmm", target_enable_fsdp_optimizations=False)
    )
    assert (backend.dispatcher, backend.experts) == ("deepep", "gmm")
    assert not backend.enable_fsdp_optimizations


def test_requires_cuda():
    recipe = _make_recipe(device="cpu")
    with pytest.raises(RuntimeError, match="requires CUDA"):
        recipe._load_kimi_k3_target(_Cfg(), "moonshotai/Kimi-K3", _text_config())


def test_requires_distributed_section():
    recipe = _make_recipe(dist_setup=None)
    with pytest.raises(ValueError, match="requires a `distributed:` section"):
        recipe._load_kimi_k3_target(_Cfg(), "moonshotai/Kimi-K3", _text_config())


def test_gates_allow_the_supported_combination():
    _validate_kimi_k3_gates(**_gate_kwargs())


@pytest.mark.parametrize("backend", ["sglang", "vllm", "remote"])
def test_gates_reject_non_colocated_backends(backend):
    with pytest.raises(NotImplementedError, match="requires target_model_backend='colocated'"):
        _validate_kimi_k3_gates(**_gate_kwargs(backend=backend))


def test_gates_reject_force_hf():
    with pytest.raises(ValueError, match="target_force_hf"):
        _validate_kimi_k3_gates(**_gate_kwargs(target_force_hf=True))


@pytest.mark.parametrize("key", ["cp_size", "tp_size"])
def test_gates_reject_context_and_tensor_parallelism(key):
    with pytest.raises(NotImplementedError, match="neither context nor tensor parallelism"):
        _validate_kimi_k3_gates(**_gate_kwargs(**{key: 2}))


def test_gates_reject_pipeline_parallelism():
    with pytest.raises(NotImplementedError, match="Pipeline parallelism"):
        _validate_kimi_k3_gates(**_gate_kwargs(pp_size=8))


def test_gates_reject_sequence_packing():
    with pytest.raises(NotImplementedError, match="Sequence packing"):
        _validate_kimi_k3_gates(**_gate_kwargs(packed_sequence_size=4096))


def test_gates_reject_parallel_drafting():
    with pytest.raises(NotImplementedError, match="Parallel drafting"):
        _validate_kimi_k3_gates(**_gate_kwargs(parallel_drafting=True))


def test_builds_text_only_causal_lm_from_config():
    """A VL checkpoint's text config is loaded as the text-only causal LM, weights included."""
    recipe = _make_recipe()
    text_config = _text_config()
    architectures, name_or_path = text_config.architectures, text_config.name_or_path
    target = object()
    with patch(_FROM_CONFIG, MagicMock(return_value=target)) as from_config:
        assert recipe._load_kimi_k3_target(_Cfg(), "moonshotai/Kimi-K3", text_config) is target

    kwargs = from_config.call_args.kwargs
    assert kwargs["config"].architectures == ["KimiK3ForCausalLM"]
    assert kwargs["config"].name_or_path == "moonshotai/Kimi-K3"
    # The draft base config the recipe serializes into the draft config.json is untouched.
    assert (text_config.architectures, text_config.name_or_path) == (architectures, name_or_path)
    assert kwargs["load_base_model"] is True
    assert kwargs["torch_dtype"] is torch.bfloat16
    assert kwargs["distributed_setup"] is recipe.dist_setup
    assert kwargs["backend"].dispatcher == "torch"
