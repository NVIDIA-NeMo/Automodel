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

"""Tests of the registered ERNIE 5.5 model: construction, input checks, forward, backward and initialization."""

import copy
import inspect
import json
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel._transformers import capabilities, model_init
from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate
from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.datasets.loader import ThdPackingConfig
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.models.common.tie_word_embeddings import TieSupport
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.ernie5_5 import model as model_module
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, main_rope_theta
from nemo_automodel.components.models.ernie5_5.dsa import Ernie5_5DSAAttention
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.model import (
    Ernie5_5_MoeForCausalLM,
    Ernie5_5Block,
    Ernie5_5Model,
    ModelClass,
)
from nemo_automodel.components.moe import layers as shared_moe_layers
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.fsdp_mixin import MoEFSDPSyncMixin
from nemo_automodel.components.utils import model_utils
from tests.unit_tests.models.ernie5_5.conftest import checkpoint_config, checkpoint_style_config

# No test of this file uses CUDA. The CUDA comparison of the indexer rotation is in test_dsa.py.
pytestmark = pytest.mark.cpu_only

VOCAB, HIDDEN, STREAMS = 64, 16, 2
ALL_KINDS = (128, 128, -2)  # HCA + dense MLP, HCA + MoE, DSA + MoE
# Hugging Face keys of the released layout: 8 model-level tensors; per layer 2 norms and 2 x 5 hyper-connection
# tensors; 15 tensors of an HCA layer or 16 of a DSA layer; 3 tensors of the dense MLP or 10 + 3 per expert of a MoE.
COMMON_KEYS, LAYER_KEYS, HCA_KEYS, DSA_KEYS, DENSE_KEYS, MOE_KEYS = 8, 12, 15, 16, 3, 10


def _eager_backend(attn: str = "eager") -> BackendConfig:
    return BackendConfig(attn=attn, linear="torch", experts="torch", dispatcher="torch")


def _cudnn_backend() -> BackendConfig:
    return BackendConfig(attn="cudnn", linear="torch", experts="torch", dispatcher="torch")


def _two_layer_checkpoint_config(ratios: tuple[int, int] = (128, 128), **overrides) -> Ernie5_5Config:
    """Return the dimensions of the checkpoint-style mapping with two decoder layers and no MTP layer."""
    fields = dict(num_hidden_layers=2, num_nextn_predict_layers=0, compress_ratios=list(ratios))
    return checkpoint_config(**{**fields, **overrides})


def _tiny_config(ratios: tuple[int, ...] = ALL_KINDS, dtype: str = "float32", **overrides) -> Ernie5_5Config:
    """Return a configuration small enough for CPU forward and backward passes."""
    fields = dict(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=24,
        num_hidden_layers=len(ratios),
        max_position_embeddings=1024,
        compress_ratios=list(ratios),
        num_nextn_predict_layers=0,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=STREAMS,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        pad_token_id=0,
        torch_dtype=dtype,
    )
    return Ernie5_5Config(**{**fields, **overrides})


def _tiny_model(ratios: tuple[int, ...] = ALL_KINDS, dtype: str = "float32", seed: int = 5579, **overrides):
    """Build a tiny model on the eager attention path with seeded weights from the model's initializer.

    The weights are drawn in FP32 and, for another dtype, loaded into a model of that dtype, so that FP32 and BF16
    models built with the same seed hold the same values up to rounding.
    """
    model = Ernie5_5_MoeForCausalLM(_tiny_config(ratios, "float32", **overrides), _eager_backend())
    torch.manual_seed(seed)
    model.initialize_weights(dtype=torch.float32)
    if dtype == "float32":
        return model
    target = Ernie5_5_MoeForCausalLM(_tiny_config(ratios, dtype, **overrides), _eager_backend())
    target.load_state_dict(model.state_dict())
    return target


def _tokens(length: int) -> torch.Tensor:
    """Return token IDs of shape [1, length] that avoid the padding token 0."""
    return (torch.arange(length) % (VOCAB - 1) + 1)[None]


def _expected_hf_keys(config: Ernie5_5Config) -> int:
    """Count the Hugging Face keys of the decoder described by a configuration without MTP entries."""
    count = COMMON_KEYS
    for index, ratio in enumerate(config.compress_ratios[: config.num_hidden_layers]):
        count += LAYER_KEYS + (HCA_KEYS if ratio == 128 else DSA_KEYS)
        count += DENSE_KEYS if index < config.first_k_dense_replace else MOE_KEYS + 3 * config.n_routed_experts
    return count


# Registration and class-level declarations


def test_registered_class_declares_capabilities_and_tie_support_itself():
    assert ModelClass is Ernie5_5_MoeForCausalLM
    assert ModelRegistry.get_model_cls_from_model_arch("Ernie5_5_MoeForCausalLM") is Ernie5_5_MoeForCausalLM
    assert "ModelCapabilities" in Ernie5_5_MoeForCausalLM.__dict__
    assert "get_capabilities" not in Ernie5_5_MoeForCausalLM.__dict__
    assert "tie_word_embeddings_support" in Ernie5_5_MoeForCausalLM.__dict__
    assert Ernie5_5_MoeForCausalLM.tie_word_embeddings_support is TieSupport.UNTIED_ONLY
    capabilities = Ernie5_5_MoeForCausalLM.ModelCapabilities()
    assert capabilities.supports_ep
    assert not (capabilities.supports_tp or capabilities.supports_cp or capabilities.supports_pp)
    assert not capabilities.supports_thd
    assert issubclass(Ernie5_5_MoeForCausalLM, (HFCheckpointingMixin, MoEFSDPSyncMixin))
    package = model_module.__name__.rpartition(".")[0]
    assert [cls for cls in Ernie5_5_MoeForCausalLM.__mro__ if cls.__module__.startswith(package)] == [
        Ernie5_5_MoeForCausalLM
    ]


def test_framework_constructor_parallelizer_and_expert_parallel_capability():
    config = _two_layer_checkpoint_config()
    with torch.device("meta"):
        custom, model = model_init._init_model(
            object, config, None, torch.bfloat16, None, False, backend=_cudnn_backend()
        )
    assert custom and type(model) is Ernie5_5_MoeForCausalLM
    assert isinstance(get_model_parallelizer(model), type(Ernie5_5_MoeForCausalLM.parallelizer))
    model = attach_capabilities_and_validate(model, SimpleNamespace(tp_size=1, cp_size=1, pp_size=1, ep_size=4))
    assert model.supports.supports_ep
    keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    assert len(keys) == len(set(keys)) == _expected_hf_keys(model.config) == 1611
    assert not model.backend.enable_fsdp_optimizations


def test_from_config_and_default_backend():
    config = _two_layer_checkpoint_config(ratios=(128, -2))
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM.from_config(config)
        explicit = Ernie5_5_MoeForCausalLM.from_config(config, _cudnn_backend())
    assert type(model) is Ernie5_5_MoeForCausalLM
    assert (model.backend.attn, model.backend.experts, model.backend.dispatcher, model.backend.linear) == (
        "cudnn",
        "torch",
        "torch",
        "torch",
    )
    assert list(model.state_dict()) == list(explicit.state_dict())


def test_adapter_is_built_only_when_the_backend_enables_it():
    backend = BackendConfig(
        attn="eager", linear="torch", experts="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
    )
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(), backend)
    assert not hasattr(model, "state_dict_adapter")


def test_model_and_blocks_hold_only_the_ernie_configuration():
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(), _eager_backend())
    assert type(model.config) is Ernie5_5Config and model.model.config is model.config
    owners = [model, model.model, *model.model.layers]
    foreign = [
        (type(owner).__name__, name)
        for owner in owners
        for name, value in vars(owner).items()
        if isinstance(value, DeepseekV4Config)
    ]
    assert foreign == []


# MTP: the model is built from a decoder-only configuration


def test_checkpoint_style_config_is_rejected_until_mtp_is_disabled(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_style_config()))
    config = model_init.get_hf_config(str(tmp_path), None, local_files_only=True)
    assert config.num_nextn_predict_layers == 1
    with torch.device("meta"), pytest.raises(ValueError, match=r"`model\.num_nextn_predict_layers: 0`"):
        model_init._init_model(object, config, None, torch.bfloat16, None, False, backend=_cudnn_backend())


def test_checkpoint_style_config_with_mtp_disabled_builds_the_decoder_and_its_key_plan(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_style_config()))
    config = model_init.get_hf_config(str(tmp_path), None, local_files_only=True, num_nextn_predict_layers=0)
    original = copy.deepcopy(config.to_dict())
    with torch.device("meta"):
        custom, model = model_init._init_model(
            object, config, None, torch.bfloat16, None, False, backend=_cudnn_backend()
        )
    assert custom and type(model) is Ernie5_5_MoeForCausalLM
    # The caller's configuration is left as it was; the model holds a normalized copy.
    # (The framework itself writes the requested dtype into the configuration it is given.)
    dtype_keys = ("dtype", "torch_dtype")
    assert {key: value for key, value in config.to_dict().items() if key not in dtype_keys} == {
        key: value for key, value in original.items() if key not in dtype_keys
    }
    assert len(config.compress_ratios) == 44 and config.mtp_shared_last_layer is True
    assert model.config is not config
    assert model.config.num_hidden_layers == 43 and model.config.num_nextn_predict_layers == 0
    assert len(model.config.compress_ratios) == 43 and model.config.mtp_shared_last_layer is False
    assert len(model.model.layers) == 43

    state = model.state_dict()
    assert len(state) == 1516
    keys = model.state_dict_adapter.get_hf_state_dict_keys(state)
    assert len(keys) == len(set(keys)) == _expected_hf_keys(model.config) == 66110
    assert not any(key.startswith("model.layers.43.") for key in keys)
    with pytest.raises(NotImplementedError, match="streaming export"):
        model.state_dict_adapter.convert_single_tensor_to_hf(
            "model.embed_tokens.weight", state["model.embed_tokens.weight"]
        )


def test_generation_config_is_initialized_from_model_config():
    from transformers.generation import GenerationConfig

    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(bos_token_id=1, eos_token_id=2), _eager_backend())

    assert isinstance(model.generation_config, GenerationConfig)
    assert model.generation_config.eos_token_id == 2
    assert model.generation_config.bos_token_id == 1
    assert model.generation_config.pad_token_id == 0


def _export_metadata_of_a_pretrained_build(tmp_path, generation_config: dict | None):
    """Build the tiny model from a hand-written checkpoint directory as the loader does and export its metadata."""
    from nemo_automodel.components.checkpoint.addons import ConsolidatedHFAddon

    checkpoint_dir = tmp_path / "checkpoint"
    checkpoint_dir.mkdir()
    config = _tiny_config(bos_token_id=1, eos_token_id=2)
    config.architectures = ["Ernie5_5_MoeForCausalLM"]
    config.save_pretrained(checkpoint_dir)
    if generation_config is not None:
        (checkpoint_dir / "generation_config.json").write_text(json.dumps(generation_config))

    with torch.device("meta"):
        custom, model = model_init._init_model(
            object, str(checkpoint_dir), None, torch.float32, None, False, backend=_eager_backend()
        )
    assert custom and type(model) is Ernie5_5_MoeForCausalLM

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    ConsolidatedHFAddon().pre_save(
        model_state=SimpleNamespace(model=[model]),
        hf_metadata_dir=str(export_dir),
        fqn_to_file_index_mapping={},
        tokenizer=None,
        original_model_path=str(checkpoint_dir),
        v4_compatible=False,
    )
    return export_dir


def test_pretrained_build_preserves_generation_config_in_export(tmp_path):
    """The end-of-sequence ids of the checkpoint's generation_config.json reach the export."""
    from transformers.generation import GenerationConfig

    export_dir = _export_metadata_of_a_pretrained_build(
        tmp_path, {"bos_token_id": 1, "eos_token_id": [2, 8], "pad_token_id": 0}
    )

    assert (export_dir / "generation_config.json").is_file()
    exported = GenerationConfig.from_pretrained(export_dir)
    assert exported.eos_token_id == [2, 8]
    assert exported.bos_token_id == 1
    assert exported.pad_token_id == 0


def test_pretrained_build_without_generation_config_exports_the_ids_of_the_model_config(tmp_path):
    """Without a generation_config.json in the checkpoint, loaders resolve the same ids as from config.json."""
    from transformers.generation import GenerationConfig

    export_dir = _export_metadata_of_a_pretrained_build(tmp_path, None)

    assert (export_dir / "generation_config.json").is_file()
    exported = GenerationConfig.from_pretrained(export_dir)
    from_model_config = GenerationConfig.from_pretrained(
        export_dir, config_file_name="config.json", _from_model_config=True
    )
    assert exported.eos_token_id == from_model_config.eos_token_id == 2
    assert exported.bos_token_id == from_model_config.bos_token_id == 1
    assert exported.pad_token_id == from_model_config.pad_token_id == 0


def test_default_configuration_with_a_schedule_builds_without_an_override():
    decoder_ratios = checkpoint_style_config()["compress_ratios"][:43]
    config = Ernie5_5Config(compress_ratios=decoder_ratios, rope_interleave=True)
    assert config.num_nextn_predict_layers == 0
    original = copy.deepcopy(config.to_dict())
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _cudnn_backend())
    assert config.to_dict() == original
    assert len(model.model.layers) == 43 and len(model.state_dict()) == 1516
    keys = model.state_dict_adapter.get_hf_state_dict_keys(model.state_dict())
    assert len(keys) == _expected_hf_keys(model.config) == 66110
    # A hand-written configuration with DSA layers must still declare the rotary pairing of its checkpoint.
    undeclared = Ernie5_5Config(compress_ratios=decoder_ratios)
    with torch.device("meta"), pytest.raises(ValueError, match="explicitly declare a boolean RoPE pairing"):
        Ernie5_5_MoeForCausalLM(undeclared, _cudnn_backend())
    with torch.device("meta"):
        hca_only = Ernie5_5_MoeForCausalLM(Ernie5_5Config(num_hidden_layers=2, compress_ratios=[128, 128]))
    assert len(hca_only.model.layers) == 2


def test_mtp_schedule_without_the_override_is_rejected_for_a_small_model():
    config = _two_layer_checkpoint_config(compress_ratios=[128, 128, 128], num_nextn_predict_layers=1)
    with torch.device("meta"), pytest.raises(ValueError, match="MTP depth 0"):
        Ernie5_5_MoeForCausalLM(config, _cudnn_backend())


# Configuration checks applied by the constructor


@pytest.mark.parametrize("explicit,fallback", [(False, True), (None, False)])
def test_dsa_rejects_nonadjacent_rope_layout(explicit, fallback):
    config = _two_layer_checkpoint_config(
        ratios=(128, -2), mqa_latent_rope_adjacent_pairing=explicit, rope_interleave=fallback
    )
    with torch.device("meta"), pytest.raises(NotImplementedError, match="Non-adjacent DSA RoPE"):
        Ernie5_5_MoeForCausalLM(config, _cudnn_backend())


def test_accepts_released_adjacent_pairing():
    config = _two_layer_checkpoint_config(ratios=(128, -2), mqa_latent_rope_adjacent_pairing=None, rope_interleave=True)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _cudnn_backend())
    assert len(model.model.layers) == 2


def test_hca_only_does_not_require_dsa_checkpoint_layout():
    config = _two_layer_checkpoint_config(mqa_latent_rope_adjacent_pairing=False)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _cudnn_backend())
    assert len(model.model.layers) == 2


def test_tied_embeddings_are_rejected():
    config = _two_layer_checkpoint_config(tie_word_embeddings=True)
    with torch.device("meta"), pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        Ernie5_5_MoeForCausalLM(config, _cudnn_backend())


def test_hca_options_and_yarn_settings_are_checked_at_construction():
    with torch.device("meta"), pytest.raises(NotImplementedError, match="YaRN"):
        Ernie5_5_MoeForCausalLM(_tiny_config(hca_rope_type="rope"), _eager_backend())
    config = _tiny_config()
    del config.rope_parameters["beta_fast"]
    with torch.device("meta"), pytest.raises(ValueError, match="YaRN"):
        Ernie5_5_MoeForCausalLM(config, _eager_backend())


# Backends and dtypes


@pytest.mark.parametrize("attn", ["eager", "torch", "sdpa"])
def test_eager_backend_names_select_the_eager_path_in_both_attention_kinds(attn):
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(), _eager_backend(attn))
    hca, dsa = model.model.layers[0].attention, model.model.layers[2].attention
    assert isinstance(hca, Ernie5_5HCAAttention) and hca.backend.attn == "eager"
    assert isinstance(dsa, Ernie5_5DSAAttention) and not dsa.uses_cudnn


def test_cudnn_backend_selects_cudnn_dsa_and_eager_hca():
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_two_layer_checkpoint_config(ratios=(128, -2)), _cudnn_backend())
    hca, dsa = model.model.layers[0].attention, model.model.layers[1].attention
    assert isinstance(hca, Ernie5_5HCAAttention) and hca.backend.attn == "eager"
    assert isinstance(dsa, Ernie5_5DSAAttention) and dsa.uses_cudnn
    assert model.backend.attn == "cudnn"


@pytest.mark.parametrize("attn", ["te", "flex", "tilelang", "fa4", "magi"])
def test_other_attention_backends_are_rejected(attn):
    with torch.device("meta"), pytest.raises(ValueError, match=rf"backend\.attn.*got attn='{attn}'"):
        Ernie5_5_MoeForCausalLM(_tiny_config(), _eager_backend(attn))


@pytest.mark.parametrize(
    "field,value,supported",
    [
        ("experts", "torch_mm", "torch"),
        ("dispatcher", "deepep", "torch"),
        # A tuning field of HybridEP: the model implements none of them, with either dispatcher.
        ("dispatcher_hybridep_max_tokens_per_rank", 256, None),
        ("linear", "te", "torch"),
        ("rope", "quack", "torch"),
        ("rope", "triton", "torch"),
        ("rms_norm", "torch", "torch_fp32"),
        ("sparse_attn", "msa", "generic"),
        ("dispatcher_num_sms", 24, 20),
        ("fake_balanced_gate", True, False),
        # BackendConfig turns the string into the dtype.
        ("gate_precision", torch.float32, None),
        ("compile_attn", True, False),
        ("compile_situ", True, False),
        ("compile_norm", True, False),
        ("compile_hc", True, False),
        ("compile_router_weight", True, False),
        ("shared_expert_overlap", True, False),
        # Honoured by MoEFSDPSyncMixin, but never validated with this model.
        ("enable_fsdp_optimizations", True, False),
    ],
)
def test_backend_fields_the_model_does_not_implement_are_rejected_by_name(field, value, supported):
    names = {"attn": "eager", "linear": "torch", "experts": "torch", "dispatcher": "torch", field: value}
    expected = re.escape(f"{field}={value!r} (supported: {supported!r})")
    with torch.device("meta"), pytest.raises(ValueError, match=expected):
        Ernie5_5_MoeForCausalLM(_tiny_config(), BackendConfig(**names))


def test_every_unsupported_backend_field_is_named_and_the_adapter_switch_is_accepted():
    unsupported = BackendConfig(attn="eager", linear="te", experts="torch", dispatcher="torch", compile_hc=True)
    with torch.device("meta"), pytest.raises(ValueError, match=r"linear='te' \(supported: 'torch'\), compile_hc=True"):
        Ernie5_5_MoeForCausalLM(_tiny_config(), unsupported)
    switches = BackendConfig(
        attn="eager",
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(), switches)
    assert model.backend is switches and not hasattr(model, "state_dict_adapter")
    # The HCA and MoE layers receive the model's backend with the eager attention name; nothing else differs.
    assert model.model.layers[0].attention.backend == replace(switches, attn="eager")
    assert model.model.layers[1].moe.backend == replace(switches, attn="eager")


def _hybridep_backend(**fields: str | int | float | bool | None) -> BackendConfig:
    """Return the grouped-GEMM experts with the HybridEP token dispatcher on the eager attention path.

    Args:
        **fields: ``BackendConfig`` fields that replace or extend those four.
    """
    values = {"attn": "eager", "linear": "torch", "experts": "torch_mm", "dispatcher": "hybridep", **fields}
    return BackendConfig(**values)


@pytest.mark.parametrize("ranks", [1, 2])
def test_hybridep_pair_is_accepted_and_keeps_the_state_and_the_checkpoint_keys(monkeypatch, ranks):
    """A checkpoint written with one dispatcher has the names and shapes that the other loads.

    In one process the shared MoE holds ``GroupedExperts`` and warns; in a job it holds ``GroupedExpertsDeepEP``.
    """
    with torch.device("meta"):
        reference = Ernie5_5_MoeForCausalLM(_tiny_config(), _eager_backend())
    backend = _hybridep_backend()
    monkeypatch.setattr(shared_moe_layers, "get_world_size_safe", lambda: ranks)
    with torch.device("meta"):
        if ranks == 1:
            with pytest.warns(UserWarning, match="'hybridep' dispatcher is enabled .* world size is 1"):
                model = Ernie5_5_MoeForCausalLM(_tiny_config(), backend)
        else:
            model = Ernie5_5_MoeForCausalLM(_tiny_config(), backend)
    assert model.backend is backend
    moe = model.model.layers[1].moe
    assert moe.backend == replace(backend, attn="eager")
    assert type(moe.experts) is (GroupedExperts if ranks == 1 else GroupedExpertsDeepEP)
    expected, state = reference.state_dict(), model.state_dict()
    assert list(state) == list(expected)
    assert [(value.shape, value.dtype) for value in state.values()] == [
        (value.shape, value.dtype) for value in expected.values()
    ]
    expected_keys = reference.state_dict_adapter.get_hf_state_dict_keys(expected)
    assert model.state_dict_adapter.get_hf_state_dict_keys(state) == expected_keys


@pytest.mark.parametrize(
    "field,value,supported",
    [
        ("experts", "torch", "torch_mm"),
        ("experts", "te", "torch_mm"),
        ("experts", "torch_mm_mxfp8", "torch_mm"),
        ("dispatcher_num_sms", 32, 20),
        ("dispatcher_share_token_dispatcher", False, True),
        ("dispatcher_async_dispatch", True, False),
        ("dispatcher_capacity_factor", 1.5, None),
        # One unpadded document per rank: the ranks of an expert-parallel group hold different token counts.
        ("dispatcher_equal_token_counts", True, False),
        ("dispatcher_recompute_cache", "combine", "none"),
        ("dispatcher_hybridep_permute_fusion", True, False),
        ("dispatcher_hybridep_compact_routing", True, False),
        ("dispatcher_hybridep_num_sms_preprocessing", 8, None),
        ("dispatcher_hybridep_num_blocks_permute", 2, None),
        ("dispatcher_hybridep_num_blocks_unpermute", 2, None),
        ("dispatcher_hybridep_max_tokens_per_rank", 256, None),
    ],
)
def test_hybridep_backend_fields_the_model_does_not_implement_are_rejected_by_name(field, value, supported):
    expected = re.escape(f"{field}={value!r} (supported: {supported!r})")
    with torch.device("meta"), pytest.raises(ValueError, match=expected):
        Ernie5_5_MoeForCausalLM(_tiny_config(), _hybridep_backend(**{field: value}))


@pytest.mark.parametrize(
    "experts,dispatcher,named",
    [
        ("torch", "hybridep", "experts='torch' (supported: 'torch_mm')"),
        ("torch_mm", "torch", "experts='torch_mm' (supported: 'torch')"),
        ("torch_mm", "deepep", "experts='torch_mm' (supported: 'torch'), dispatcher='deepep' (supported: 'torch')"),
    ],
)
def test_rejected_expert_and_dispatcher_pair_names_the_supported_pairs(experts, dispatcher, named):
    backend = BackendConfig(attn="eager", linear="torch", experts=experts, dispatcher=dispatcher)
    pairs = "`experts` and `dispatcher` must be 'torch' and 'torch', or 'torch_mm' and 'hybridep'"
    with torch.device("meta"), pytest.raises(ValueError, match=re.escape(f"{named}. {pairs}")):
        Ernie5_5_MoeForCausalLM(_tiny_config(), backend)


def test_cudnn_requires_bf16_and_the_released_dsa_dimensions():
    with torch.device("meta"), pytest.raises(ValueError, match="cuDNN DSA attention requires BF16"):
        Ernie5_5_MoeForCausalLM(_two_layer_checkpoint_config(ratios=(128, -2), torch_dtype="float32"), _cudnn_backend())
    with torch.device("meta"), pytest.raises(ValueError, match="index_head_dim == 128"):
        Ernie5_5_MoeForCausalLM(_tiny_config(dtype="bfloat16"), _cudnn_backend())
    # The default backend is cuDNN; the message names the backend that runs these dimensions.
    with torch.device("meta"), pytest.raises(ValueError, match=r"Set `model\.backend\.attn: eager`"):
        Ernie5_5_MoeForCausalLM(_tiny_config())


def test_dtype_comes_from_the_configuration_and_must_be_fp32_or_bf16():
    with torch.device("meta"):
        fp32 = Ernie5_5_MoeForCausalLM(_tiny_config(dtype="float32"), _eager_backend())
        bf16 = Ernie5_5_MoeForCausalLM(_tiny_config(dtype="bfloat16"), _eager_backend())
        default = Ernie5_5_MoeForCausalLM(_tiny_config(torch_dtype=None), _eager_backend())
    assert {parameter.dtype for parameter in fp32.parameters()} == {torch.float32}
    assert bf16.model.embed_tokens.weight.dtype == torch.bfloat16
    assert default.model.embed_tokens.weight.dtype == torch.bfloat16
    with torch.device("meta"), pytest.raises(ValueError, match="FP32 and BF16 only"):
        Ernie5_5_MoeForCausalLM(_tiny_config(dtype="float16"), _eager_backend())


# FP32 names


@pytest.mark.parametrize("initialized", [False, True])
def test_strict_fp32_names_select_exactly_the_fp32_tensors_of_a_bf16_model(initialized):
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype="bfloat16"), _eager_backend())
    if initialized:
        Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    names = Ernie5_5_MoeForCausalLM._keep_in_fp32_modules_strict
    assert "_keep_in_fp32_modules_strict" in Ernie5_5_MoeForCausalLM.__dict__ and len(names) == 13
    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    selected = {name for name in tensors if any(keyword in name for keyword in names)}
    fp32 = {name for name, tensor in tensors.items() if tensor.dtype == torch.float32}
    assert selected == fp32
    assert {tensor.dtype for name, tensor in tensors.items() if name not in fp32} == {torch.bfloat16}
    # Every keyword selects something in a model that has all layer kinds.
    assert all(any(keyword in name for name in tensors) for keyword in names)
    # The list lives on the registered class only; no layer instance carries its own copy.
    assert not any("_keep_in_fp32_modules_strict" in vars(module) for module in model.modules())


# Decoder block


@pytest.mark.parametrize("layer_index,child", [(0, "mlp"), (1, "moe")])
def test_block_keeps_fp32_streams_rounds_sublayer_inputs_and_backpropagates(layer_index, child):
    model = _tiny_model(dtype="bfloat16")
    block = model.model.layers[layer_index]
    assert isinstance(block, Ernie5_5Block) and block.is_moe == (child == "moe")
    seen = {}
    handles = [
        block.attention.register_forward_pre_hook(lambda _, args: seen.update(attention=args[0].dtype)),
        getattr(block, child).register_forward_pre_hook(lambda _, args: seen.update(feed_forward=args[0].dtype)),
    ]
    tokens = _tokens(6)
    streams = torch.randn(1, 6, STREAMS, HIDDEN, requires_grad=True)
    output = block(streams, model.model.attention_inputs(tokens))
    for handle in handles:
        handle.remove()
    assert output.shape == streams.shape and output.dtype == torch.float32
    assert seen == {"attention": torch.bfloat16, "feed_forward": torch.bfloat16}
    output.square().mean().backward()
    assert torch.isfinite(streams.grad).all() and torch.count_nonzero(streams.grad)
    attention_grads = [p.grad for p in block.attention.parameters() if p.grad is not None]
    feed_forward_grads = [p.grad for p in getattr(block, child).parameters() if p.grad is not None]
    assert attention_grads and feed_forward_grads
    assert all(torch.isfinite(grad).all() for grad in attention_grads + feed_forward_grads)


def test_block_rejects_non_fp32_streams_and_wrong_shapes():
    model = _tiny_model(ratios=(128,))
    block, inputs = model.model.layers[0], model.model.attention_inputs(_tokens(3))
    with pytest.raises(ValueError, match="must be FP32"):
        block(torch.zeros(1, 3, STREAMS, HIDDEN, dtype=torch.bfloat16), inputs)
    with pytest.raises(ValueError, match="Expected streams"):
        block(torch.zeros(1, 3, HIDDEN), inputs)


@pytest.mark.parametrize("layer_index,child", [(0, "mlp"), (1, "moe")])
def test_block_registers_the_feed_forward_under_its_native_name(layer_index, child):
    model = _tiny_model()
    block = model.model.layers[layer_index]
    names = list(block.state_dict())
    assert any(name.startswith(f"{child}.") for name in names)
    assert not any(name.startswith("feed_forward.") for name in names)
    assert not hasattr(block, "moe" if child == "mlp" else "mlp")

    # A module installed in place of the child, as activation checkpointing does, is used by the next forward.
    class _Zero(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            """Return zeros with the shape [batch, sequence, hidden] of the input."""
            return torch.zeros_like(hidden)

    tokens = _tokens(5)
    streams = torch.randn(1, 5, STREAMS, HIDDEN)
    inputs = model.model.attention_inputs(tokens)
    before = block(streams, inputs)
    setattr(block, child, _Zero())
    after = block(streams, inputs)
    assert getattr(block, child).__class__ is _Zero
    assert not torch.equal(before, after)


# Attention inputs


def test_hca_inputs_are_causal_and_window_bounded():
    model = _tiny_model(ratios=(128,), sliding_window=2)
    inputs = model.model.attention_inputs(_tokens(4))
    keep = inputs.hca_mask[0, 0] == 0
    expected = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]], dtype=torch.bool)
    assert inputs.hca_mask.shape == (1, 1, 4, 4) and torch.equal(keep, expected)
    cos, sin = inputs.hca_rope
    assert cos.shape == sin.shape == (1, 4, 4)
    assert torch.isfinite(cos).all() and torch.isfinite(sin).all()
    assert torch.equal(inputs.position_ids, torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(inputs.document_ids, torch.zeros(1, 4, dtype=torch.int64))
    assert torch.equal(inputs.cu_seqlens, torch.tensor([0, 4], dtype=torch.int32))
    # The rotary module is created per call and is not stored on the model.
    assert not any("inv_freq" in name for name, _ in model.named_buffers())


class _Captured(Exception):
    """Raised by a forward pre-hook after it recorded the arguments of a layer."""


def _capture_attention_call(model: Ernie5_5Model, layer_index: int, tokens: torch.Tensor) -> dict:
    """Return the keyword arguments that a block passes to its attention layer.

    Args:
        model: Decoder whose block is observed.
        layer_index: Index of the block.
        tokens: Token IDs of shape [1, sequence].

    Returns:
        The keyword arguments of the attention call. The call itself is aborted before the layer runs.
    """
    captured = {}

    def hook(_module, _args, kwargs):
        captured.update(kwargs)
        raise _Captured

    handle = model.layers[layer_index].attention.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with pytest.raises(_Captured):
            model.layers[layer_index](torch.zeros(1, tokens.shape[1], STREAMS, HIDDEN), model.attention_inputs(tokens))
    finally:
        handle.remove()
    return captured


def test_each_attention_kind_receives_the_inputs_of_its_path():
    tokens = _tokens(4)
    eager = _tiny_model().model
    hca = _capture_attention_call(eager, 0, tokens)
    assert set(hca) == {"position_embeddings", "attention_mask", "position_embeddings_compress", "rotary_compress"}
    assert hca["position_embeddings"] is hca["position_embeddings_compress"]
    dsa = _capture_attention_call(eager, 2, tokens)
    assert set(dsa) == {"position_ids", "document_ids"}
    assert torch.equal(dsa["document_ids"], torch.zeros_like(tokens))

    # The cuDNN layer is only constructed here; its kernels need a GPU.
    config = _tiny_config(dtype="bfloat16")
    cudnn = Ernie5_5Model(config, _cudnn_backend(), eager.moe_config)
    assert cudnn.layers[2].attention.uses_cudnn
    packed = _capture_attention_call(cudnn, 2, tokens)
    assert set(packed) == {"position_ids", "cu_seqlens", "max_seqlen"}
    assert torch.equal(packed["position_ids"], torch.tensor([[0, 1, 2, 3]]))
    assert torch.equal(packed["cu_seqlens"], torch.tensor([0, 4], dtype=torch.int32))
    assert packed["max_seqlen"] == 4


def test_attention_inputs_use_the_window_captured_at_construction():
    model = _tiny_model(sliding_window=2)
    assert model.model.sliding_window == model.model.layers[2].attention.window == 2
    before = model.model.attention_inputs(_tokens(6)).hca_mask
    model.config.sliding_window = 4
    after = model.model.attention_inputs(_tokens(6)).hca_mask
    assert torch.equal(before, after)


def test_overridden_rope_base_reaches_the_dsa_layers_and_leaves_the_configuration_unchanged():
    config = _tiny_config(rope_theta=10_000)
    assert main_rope_theta(config) == 10_000
    original = copy.deepcopy(config.to_dict())
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(config, _eager_backend())
    assert config.to_dict() == original
    assert model.model.layers[2].attention.rope_theta == 10_000
    assert model.config.to_dict()["rope_theta"] == 10_000
    default = _tiny_config()
    with torch.device("meta"):
        assert Ernie5_5_MoeForCausalLM(default, _eager_backend()).model.layers[2].attention.rope_theta == 8_000_000
    # A base that only one of the two keys carries is rejected at construction.
    conflicting = _tiny_config(rope_theta=10_000, rotary_base=8_000_000)
    with torch.device("meta"), pytest.raises(ValueError, match="Conflicting RoPE base values"):
        Ernie5_5_MoeForCausalLM(conflicting, _eager_backend())


# Input checks of the registered forward


@pytest.fixture(scope="module")
def hca_model():
    model = _tiny_model(ratios=(128, 128))
    model.eval()
    return model


def test_accepts_one_unpadded_row(hca_model):
    ids = torch.tensor([[4, 2, 9]])
    plain = hca_model(ids).logits
    assert plain.shape == (1, 3, VOCAB)
    explicit = hca_model(
        ids,
        attention_mask=torch.ones_like(ids),
        padding_mask=torch.zeros_like(ids, dtype=torch.bool),
        position_ids=torch.arange(3, dtype=torch.int32)[None],
        use_cache=False,
    ).logits
    assert torch.equal(plain, explicit)


@pytest.mark.parametrize(
    "ids", [torch.empty(1, 0, dtype=torch.int64), torch.ones(2, 2, dtype=torch.int64), torch.ones(1, 2)]
)
def test_rejects_bad_input_shape_or_dtype(hca_model, ids):
    with pytest.raises(ValueError):
        hca_model(ids)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"attention_mask": torch.tensor([[1, 0, 1]])}, "Padding is unsupported"),
        ({"padding_mask": torch.tensor([[False, True, False]])}, "Padding is unsupported"),
        ({"attention_mask": torch.ones(1, 2)}, "attention_mask must match"),
        ({"padding_mask": torch.zeros(1, 2, dtype=torch.bool)}, "padding_mask must match"),
        # Packed rows carry the document index in the attention mask and restart the positions.
        ({"attention_mask": torch.tensor([[1, 1, 2]])}, "Padding is unsupported"),
        ({"position_ids": torch.tensor([[0, 1, 0]])}, "one document"),
        ({"position_ids": torch.tensor([[0.0, 1.0, 2.0]])}, "position_ids"),
        ({"use_cache": True}, "KV cache is unsupported"),
    ],
)
def test_rejects_padding_packed_rows_restarted_positions_and_a_cache(hca_model, kwargs, message):
    with pytest.raises(ValueError, match=message):
        hca_model(torch.tensor([[4, 2, 9]]), **kwargs)


def test_forward_declares_no_packed_sequence_or_context_parallel_input(hca_model):
    parameters = inspect.signature(Ernie5_5_MoeForCausalLM.forward).parameters
    assert list(parameters) == ["self", "input_ids", "attention_mask", "padding_mask", "position_ids", "use_cache"]
    assert parameters["use_cache"].kind is inspect.Parameter.KEYWORD_ONLY
    # The framework probes the signature for ``seq_lens``; without it THD packing is refused at loader construction.
    assert not model_utils._supports_seq_lens(hca_model)
    assert not capabilities._supports_seq_lens(hca_model)
    with pytest.raises(ValueError, match="THD packing requires a model that consumes seq_lens"):
        ThdPackingConfig(packed_sequence_size=8).build([], supports_seq_lens=model_utils._supports_seq_lens(hca_model))
    # Batch keys that the signature does not declare are dropped by the recipe before the call.
    batch = {
        "input_ids": torch.tensor([[4, 2, 9]]),
        "seq_lens": torch.tensor([[3]]),
        "labels": torch.tensor([[2, 9, 1]]),
    }
    assert set(model_utils.filter_forward_kwargs(hca_model, batch)) == {"input_ids"}


def test_rejects_the_padding_token_and_unknown_inputs(hca_model):
    assert hca_model.config.pad_token_id == 0
    with pytest.raises(ValueError, match="pad_token_id=0"):
        hca_model(torch.tensor([[1, 0, 2]]))
    for name in ("packed_seq_params", "seq_lens", "cu_seqlens", "cp_group", "past_key_values"):
        with pytest.raises(TypeError, match=name):
            hca_model(torch.tensor([[4, 2, 9]]), **{name: None})


def test_a_token_equal_to_the_padding_id_is_accepted_when_no_padding_id_is_configured():
    model = _tiny_model(ratios=(128,), pad_token_id=None)
    assert torch.isfinite(model(torch.tensor([[1, 0, 2]])).logits.float()).all()


# Forward and backward on the eager path


def _loss(model: Ernie5_5_MoeForCausalLM, tokens: torch.Tensor) -> torch.Tensor:
    """Return the next-token cross entropy of one document.

    Args:
        model: Model under test.
        tokens: Token IDs of shape [1, sequence].

    Returns:
        Scalar FP32 loss.
    """
    logits = model(tokens).logits
    assert logits.shape == (1, tokens.shape[1], VOCAB)
    assert logits.dtype == model.model.embed_tokens.weight.dtype
    assert torch.isfinite(logits.float()).all()
    return F.cross_entropy(logits[0, :-1].float(), tokens[0, 1:])


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize(
    "ratios", [(128, 128), (128, 128, -2)], ids=["hca_dense_and_hca_moe", "hca_dense_hca_moe_and_dsa_moe"]
)
@pytest.mark.parametrize("length", [12, 140], ids=["short", "one_compression_window"])
def test_tiny_model_forward_and_backward_on_cpu(ratios, dtype, length):
    model = _tiny_model(ratios, dtype)
    tokens = _tokens(length)
    loss = _loss(model, tokens)
    assert torch.isfinite(loss)
    loss.backward()
    frozen = {name for name, parameter in model.named_parameters() if not parameter.requires_grad}
    assert frozen == {name for name, _ in model.named_parameters() if ".attention.indexer." in name}
    assert len(frozen) == (5 if -2 in ratios else 0)
    without_gradient = set()
    for name, parameter in model.named_parameters():
        if name in frozen:
            assert parameter.grad is None, name
        elif parameter.grad is None:
            without_gradient.add(name)
        else:
            assert torch.isfinite(parameter.grad).all(), name
    # A document shorter than one HCA compression window (128 tokens) has no compressed key, so the compressor
    # takes no part in the loss; every other trainable parameter always receives a gradient.
    assert all(".attention.compressor." in name for name in without_gradient), sorted(without_gradient)
    assert bool(without_gradient) == (length < 128)
    for name in ("model.embed_tokens.weight", "model.hyper_head.hc_fn", "model.lm_head.weight"):
        assert torch.count_nonzero(model.get_parameter(name).grad), name
    for index in range(len(ratios)):
        assert torch.count_nonzero(model.get_parameter(f"model.layers.{index}.attn_hc.fn").grad), index
        assert torch.count_nonzero(model.get_parameter(f"model.layers.{index}.ffn_hc.fn").grad), index


def test_tiny_model_one_optimizer_step():
    model = _tiny_model(ratios=(-2,), dtype="bfloat16", first_k_dense_replace=1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    labels = torch.tensor([[-100, -100, -100, -100, -100, 7, 8, 9]])
    logits = model(tokens).logits
    assert logits.shape == (1, 8, VOCAB) and logits.dtype == torch.bfloat16
    loss = F.cross_entropy(logits.float().reshape(-1, VOCAB), labels.reshape(-1))
    assert torch.isfinite(loss)
    loss.backward()
    for value in (model.model.embed_tokens.weight, model.model.hyper_head.hc_fn):
        assert value.grad is not None and torch.isfinite(value.grad).all() and torch.count_nonzero(value.grad)
    # The indexer only selects keys: the language-model loss gives it no gradient.
    indexer = model.model.layers[0].attention.indexer
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in indexer.parameters())
    before = model.model.lm_head.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.model.lm_head.weight)


def test_shared_trainability_policy_leaves_exactly_the_indexers_frozen():
    # The shared policy has no ERNIE step: the freeze of the constructor keeps the indexers out of training.
    model = _tiny_model()
    indexer = {name for name, _ in model.named_parameters() if ".attention.indexer." in name}
    assert len(indexer) == 5
    _apply_trainability_policy(model, peft_enabled=False, freeze_config=None, strict=True)
    assert {name for name, parameter in model.named_parameters() if not parameter.requires_grad} == indexer


def test_training_forward_rejects_an_unfreeze_selector_that_covers_an_indexer():
    model = _tiny_model(ratios=(-2,), first_k_dense_replace=1)
    tokens = _tokens(8)
    name = "model.layers.0.attention.indexer.wk.weight"
    freeze_config = model_utils.FreezeConfig(unfreeze_parameters=(name,))
    _apply_trainability_policy(model, peft_enabled=False, freeze_config=freeze_config, strict=True)
    assert model.training and model.get_parameter(name).requires_grad
    with pytest.raises(ValueError, match=r"model\.layers\.0\.attention\.indexer has trainable parameters"):
        model(tokens)
    # Evaluation updates no parameter, so the flags are not checked there.
    model.eval()
    assert model(tokens).logits.shape == (1, 8, VOCAB)


def test_eager_dsa_path_is_limited_to_512_tokens():
    model = _tiny_model(ratios=(-2,))
    with pytest.raises(ValueError, match="max_eager_sequence"):
        model(_tokens(513))


def test_native_state_round_trips_through_the_adapter():
    model = _tiny_model()
    state = model.state_dict()
    converted = model.state_dict_adapter.to_hf(state)
    assert len(converted) == _expected_hf_keys(model.config)
    assert list(converted) == model.state_dict_adapter.get_hf_state_dict_keys(state)
    restored = model.state_dict_adapter.from_hf(converted)
    assert set(restored) == set(state)
    assert all(torch.equal(restored[name], value) for name, value in state.items())


# Initialization without a checkpoint


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_production_initializer_fills_every_tensor_of_a_meta_built_model(dtype):
    torch.manual_seed(7)
    with torch.device("meta"):
        model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype=dtype), _eager_backend())
    frozen = [name for name, parameter in model.named_parameters() if not parameter.requires_grad]
    dtypes = {name: parameter.dtype for name, parameter in model.named_parameters()}
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))

    tensors = dict(model.named_parameters()) | dict(model.named_buffers())
    assert all(tensor.device.type == "cpu" for tensor in tensors.values())
    assert all(torch.isfinite(tensor.float()).all() for tensor in tensors.values())
    assert {name: parameter.dtype for name, parameter in model.named_parameters()} == dtypes
    assert [name for name, parameter in model.named_parameters() if not parameter.requires_grad] == frozen

    zeros = ("sinks_param.weight", "ape_param.weight", "vha_postmix_V", "softmax_offset", "_hc.base", "hc_base")
    zeros += ("multimax_ranges", "multimax_ts", "e_score_correction_bias")
    ones = ("_hc.scale", "hc_scale", "layernorm.weight", "norm.weight", "k_norm.weight")
    for name, tensor in tensors.items():
        if name.endswith(zeros) or name.endswith("k_norm.bias"):
            assert torch.count_nonzero(tensor) == 0, name
        elif name.endswith(ones):
            assert torch.equal(tensor.float(), torch.ones_like(tensor, dtype=torch.float32)), name
        elif name.endswith("routed_scaling_factor_param"):
            assert torch.equal(tensor, torch.full_like(tensor, model.config.routed_scaling_factor)), name
        else:
            assert torch.count_nonzero(tensor), name
            assert tensor.float().std() < 4 * model.config.initializer_range, name

    loss = _loss(model, _tokens(140))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)


def test_initialize_weights_without_a_dtype_keeps_the_dtype_of_the_configuration():
    model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype="float32"), _eager_backend())
    model.initialize_weights()
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_initialize_weights_casts_an_unsharded_model_to_an_explicit_dtype():
    model = Ernie5_5_MoeForCausalLM(_tiny_config(dtype="float32"), _eager_backend())
    model.initialize_weights(dtype=torch.bfloat16)
    names = Ernie5_5_MoeForCausalLM._keep_in_fp32_modules_strict
    for name, parameter in model.named_parameters():
        expected = torch.float32 if any(keyword in name for keyword in names) else torch.bfloat16
        assert parameter.dtype == expected, name


def test_hca_rotary_width_does_not_depend_on_a_floating_point_ratio():
    # int(44 * (30 / 44)) is 29: the width must come from qk_rope_head_dim itself.
    model = _tiny_model(ratios=(128,), head_dim=44, qk_rope_head_dim=30)
    cos, sin = model.model.attention_inputs(_tokens(6)).hca_rope
    assert cos.shape == sin.shape == (1, 6, 30)
    assert torch.isfinite(model(_tokens(6)).logits).all()


def test_initialize_weights_overwrites_poisoned_values_and_depends_on_the_seed():
    model = _tiny_model(seed=1)
    first = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with torch.no_grad():
        for tensor in list(model.parameters()) + list(model.buffers()):
            assert tensor.is_floating_point()
            tensor.fill_(float("nan"))
    torch.manual_seed(1)
    model.initialize_weights(dtype=torch.float32)
    repeated = model.state_dict()
    assert all(torch.equal(repeated[name], value) for name, value in first.items())
    torch.manual_seed(2)
    model.initialize_weights(dtype=torch.float32)
    other = model.state_dict()
    assert not torch.equal(other["model.embed_tokens.weight"], first["model.embed_tokens.weight"])
    assert not torch.equal(
        other["model.layers.1.moe.experts.gate_and_up_projs"], first["model.layers.1.moe.experts.gate_and_up_projs"]
    )
