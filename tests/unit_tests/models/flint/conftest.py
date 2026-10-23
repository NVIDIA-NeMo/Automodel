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

"""Shared tiny and checkpoint-style Flint configurations for the unit tests."""

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.flint.config import FlintConfig

_NUM_LAYERS = 49
_FULL_ATTENTION_LAYERS = (0, 1, *range(5, 46, 4), 48)


def checkpoint_style_config() -> dict[str, object]:
    """Return a fresh ``config.json``-style mapping with the released 49-layer Flint schedule.

    The dimensions and layer schedule match the source checkpoint. The mapping is built here
    instead of copying the checkpoint's own file, and carries only the fields the tests use.
    """
    return {
        "architectures": ["FlintForCausalLM"],
        "auto_map": {"AutoConfig": "configuration_flint.FlintConfig"},
        "model_type": "flint",
        "torch_dtype": "bfloat16",
        "vocab_size": 150144,
        "hidden_size": 2048,
        "intermediate_size": 6144,
        "num_hidden_layers": _NUM_LAYERS,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "hidden_act": "silu",
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-06,
        "rope_theta": 10000.0,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "use_sliding_window": True,
        "sliding_window": 513,
        "layer_types": [
            "full_attention" if index in _FULL_ATTENTION_LAYERS else "sliding_attention" for index in range(_NUM_LAYERS)
        ],
        "no_rope_layer_indices": list(_FULL_ATTENTION_LAYERS),
        "softmax_type": "learnable",
        "decoder_sparse_step": 1,
        "mlp_only_layers": [0, 1],
        "moe_intermediate_size": 512,
        "shared_expert_intermediate_size": 1024,
        "num_experts": 256,
        "num_experts_per_tok": 6,
        "norm_topk_prob": True,
        "router_score_function": "sigmoid",
        "router_aux_loss_coef": 0.0005,
        "output_router_logits": False,
        "tie_word_embeddings": False,
    }


def tiny_config(**overrides: object) -> FlintConfig:
    """Return a tiny asymmetric projection/FFN config with all three layer kinds."""
    values = dict(
        hidden_size=12,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_hidden_layers=3,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        mlp_only_layers=[0, 1, 2],
        intermediate_size=20,
        moe_intermediate_size=5,
        shared_expert_intermediate_size=7,
        num_experts=4,
        num_experts_per_tok=2,
        sliding_window=3,
        vocab_size=31,
    )
    values.update(overrides)
    return FlintConfig(**values)


def tiny_backend(attn: str = "eager") -> BackendConfig:
    """Use actual shared Torch operators, with explicit attention selection."""
    return BackendConfig(
        attn=attn,
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
    )


@contextmanager
def cpu_loader() -> Iterator[None]:
    """Build through the public loader on CPU.

    ``_build_model`` unconditionally queries the current CUDA device. This hardware-only
    adaptation follows the MuseGlimmer and DeepseekV41 tests and is scoped to the loader
    call so that compiled helpers still see the real device afterwards.
    """
    with patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")):
        yield
