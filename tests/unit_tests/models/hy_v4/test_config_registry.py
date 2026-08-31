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

import json

import pytest

from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel._transformers.registry import (
    _CUSTOM_CONFIG_REGISTRATIONS,
    MODEL_ARCH_MAPPING,
    resolve_custom_config_cls,
)
from nemo_automodel.components.models.hy_v4.config import HyV4Config
from nemo_automodel.components.models.hy_v4.model import HyV4ForCausalLM


def test_checkpoint_defaults_capture_hy_v4_topology():
    config = HyV4Config()

    assert config.model_type == "hy_v4"
    assert config.num_hidden_layers == 78
    assert config.mlp_layer_types[:2] == ["dense", "sparse"]
    assert config.indexer_types[:10] == [
        "full",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
        "shared",
        "shared",
        "shared",
        "full",
    ]
    assert config.enable_ihc and config.hc_mult == 4
    assert config.learnable_sink and config.gated_mla
    assert config.num_nextn_predict_layers == 1
    assert config.enable_lm_head_fp32


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mlp_layer_types", ["dense"], "one entry per hidden layer"),
        ("indexer_types", ["full"], "one entry per hidden layer"),
        ("indexer_types", ["invalid", "full"], "'full' or 'shared'"),
        ("gating_type", "invalid", "elementwise"),
    ],
)
def test_config_rejects_invalid_layer_contracts(field, value, message):
    with pytest.raises(ValueError, match=message):
        HyV4Config(num_hidden_layers=2, **{field: value})


def test_registry_resolves_automodel_owned_config_and_model():
    assert MODEL_ARCH_MAPPING["HYV4ForCausalLM"] == (
        "nemo_automodel.components.models.hy_v4.model",
        "HyV4ForCausalLM",
    )
    assert _CUSTOM_CONFIG_REGISTRATIONS["hy_v4"] == (
        "nemo_automodel.components.models.hy_v4.config",
        "HyV4Config",
    )
    assert resolve_custom_config_cls("hy_v4") is HyV4Config

    capabilities = HyV4ForCausalLM.ModelCapabilities()
    assert capabilities.supports_cp and capabilities.supports_ep and capabilities.supports_thd
    assert capabilities.supports_mtp_cp
    assert not capabilities.supports_tp and capabilities.supports_pp
    assert HyV4ForCausalLM._pp_return_hidden_states_supported
    assert HyV4ForCausalLM._pp_fused_linear_ce_mtp_supported


def test_checkpoint_style_config_resolves_without_remote_transformers_code(tmp_path):
    """The generic loader selects AutoModel's config from a local checkpoint JSON."""
    payload = HyV4Config().to_dict()
    payload["architectures"] = ["HYV4ForCausalLM"]
    (tmp_path / "config.json").write_text(json.dumps(payload), encoding="utf-8")

    config = get_hf_config(str(tmp_path), attn_implementation="eager")

    assert type(config) is HyV4Config
    assert config.architectures == ["HYV4ForCausalLM"]
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 10_000_000.0}


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"use_dsa": False}, "use_dsa"),
        ({"attention_dropout": 0.1}, "attention_dropout"),
        ({"hidden_act": "gelu"}, "hidden_act"),
        ({"gated_mla": False}, "gated_mla"),
        ({"rope_parameters": {"rope_type": "linear", "rope_theta": 10_000_000.0}}, "default interleaved"),
    ],
)
def test_config_rejects_forward_paths_absent_from_the_pinned_vllm_oracle(override, message):
    """Only the public HY4-preview forward contract is accepted."""
    with pytest.raises(ValueError, match=message):
        HyV4Config(**override)
