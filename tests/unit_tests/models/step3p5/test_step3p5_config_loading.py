# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load Step-3.5 trunk metadata without its remote config validation failure."""

import json
from pathlib import Path
from typing import TypedDict

import pytest
import yaml

from nemo_automodel import NeMoAutoConfig
from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7TextConfig

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[4]
RECIPE_PATHS = [
    "examples/llm_pretrain/step/step_3.5_flash_te_deepep.yaml",
    "examples/llm_benchmark/step/step_3.5_flash_te_deepep.yaml",
    "examples/llm_benchmark/step/step35flash_lora.yaml",
]


class CheckpointConfig(TypedDict):
    """Checkpoint metadata needed to distinguish trunk and MTP layers."""

    model_type: str
    architectures: list[str]
    auto_map: dict[str, str]
    num_hidden_layers: int
    num_nextn_predict_layers: int
    layer_types: list[str]
    rope_theta: list[float]
    partial_rotary_factors: list[float]
    swiglu_limits: list[float]
    swiglu_limits_shared: list[float]
    use_rope_layers: list[bool]
    rope_scaling: dict[str, str | float]
    yarn_only_types: list[str]
    tie_word_embeddings: bool
    share_expert_dim: int
    max_position_embeddings: int
    vocab_size: int


@pytest.fixture
def checkpoint(tmp_path: Path) -> tuple[Path, CheckpointConfig]:
    # Shape/identity from the failing checkpoint's config.json:
    # https://huggingface.co/stepfun-ai/Step-3.5-Flash/blob/
    # ab446a3de5e171ea341227e24bb1f090e1b771f7/config.json
    # It describes 45 trunk layers followed by three MTP layers in each list.
    data: CheckpointConfig = {
        "model_type": "step3p5",
        "architectures": ["Step3p5ForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_step3p5.Step3p5Config"},
        "num_hidden_layers": 45,
        "num_nextn_predict_layers": 3,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "sliding_attention"] * 12,
        "rope_theta": [5000000.0, 10000.0, 10000.0, 10000.0] * 12,
        "partial_rotary_factors": [0.5, 1.0, 1.0, 1.0] * 12,
        "swiglu_limits": [0.0] * 43 + [7.0, 7.0, 2.0, 3.0, 4.0],
        "swiglu_limits_shared": [0.0] * 44 + [16.0, 5.0, 6.0, 7.0],
        "use_rope_layers": [],
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 131072,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        },
        "yarn_only_types": ["full_attention"],
        "tie_word_embeddings": False,
        "share_expert_dim": 1280,
        "max_position_embeddings": 262144,
        "vocab_size": 128896,
    }
    (tmp_path / "config.json").write_text(json.dumps(data))
    return tmp_path, data


def _assert_checkpoint_metadata(config: Step3p7TextConfig, data: CheckpointConfig) -> None:
    assert config.model_type == "step3p5"
    assert config.architectures == ["Step3p5ForCausalLM"]
    assert config.num_hidden_layers == 45
    assert config.num_nextn_predict_layers == 3
    for trunk, mtp, source in (
        (config.layer_types, config.mtp_layer_types, data["layer_types"]),
        (config.rope_theta, config.mtp_rope_theta, data["rope_theta"]),
        (config.partial_rotary_factors, config.mtp_partial_rotary_factors, data["partial_rotary_factors"]),
        (config.swiglu_limits, config.mtp_swiglu_limits, data["swiglu_limits"]),
        (config.swiglu_limits_shared, config.mtp_swiglu_limits_shared, data["swiglu_limits_shared"]),
    ):
        assert trunk == source[:45]
        assert mtp == source[45:]
    assert config.use_rope_layers == []
    for name, value in data["rope_scaling"].items():
        assert config.rope_scaling[name] == value
    assert config.yarn_only_types == data["yarn_only_types"]
    assert config.share_expert_dim == 1280
    assert config.tie_word_embeddings is False
    assert config.vocab_size == 128896


def test_step3p5_reuses_registered_text_backbone_config() -> None:
    assert resolve_custom_config_cls("step3p5") is Step3p7TextConfig


@pytest.mark.parametrize("recipe_path", RECIPE_PATHS)
def test_recipe_config_loads_checkpoint_with_mtp_layer_metadata(
    recipe_path: str, checkpoint: tuple[Path, CheckpointConfig]
) -> None:
    path, data = checkpoint
    recipe = yaml.safe_load((ROOT / recipe_path).read_text())
    config_node = ConfigNode(recipe["model"]["config"])
    config_node.pretrained_model_name_or_path = str(path)
    config = config_node.instantiate()
    _assert_checkpoint_metadata(config, data)


def test_from_pretrained_config_routing_preserves_mtp_metadata(checkpoint: tuple[Path, CheckpointConfig]) -> None:
    path, data = checkpoint
    config = get_hf_config(str(path), "sdpa", trust_remote_code=True)
    _assert_checkpoint_metadata(config, data)


def test_config_save_reload_preserves_trunk_and_mtp_metadata(
    checkpoint: tuple[Path, CheckpointConfig], tmp_path: Path
) -> None:
    path, data = checkpoint
    config = get_hf_config(str(path), "sdpa", trust_remote_code=True)
    config.save_pretrained(tmp_path / "roundtrip")
    restored = NeMoAutoConfig.from_pretrained(tmp_path / "roundtrip", trust_remote_code=False)
    _assert_checkpoint_metadata(restored, data)
