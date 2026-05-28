# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Shared fixtures for Falcon-H1 unit tests.

Design notes
------------
* Two tiny configs are provided. ``tiny_config`` keeps the absolute-minimum
  dims for fast shape/round-trip checks; ``layer_config`` uses *realistic*
  per-layer dims (hidden_size=256, 8 attn heads, 4 kv heads) because the
  SKILL.md warns that extremely small dims can mask numerical divergence in
  layer-equivalence tests.
* dtype follows the model. The reference Falcon-H1 checkpoints run in
  bfloat16, but the Mamba fast-path kernel (mamba-ssm / Triton) is CUDA-only,
  so CPU-only shape tests build in float32 and GPU equivalence tests build in
  the config's declared dtype. Tests never silently override dtype.
* Both ``mamba_rms_norm`` variants are exposed via params so the gated-norm
  path (and its extra ``mamba.norm.weight`` checkpoint key) is actually
  exercised — this is the single most error-prone part of the port.
"""

import pytest

torch = pytest.importorskip("torch")

try:
    from transformers import FalconH1Config

    HAS_FALCON_H1 = True
except Exception:  # pragma: no cover - exercised only on old transformers
    HAS_FALCON_H1 = False


requires_falcon_h1 = pytest.mark.skipif(
    not HAS_FALCON_H1,
    reason="transformers build does not ship FalconH1Config",
)
requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Mamba fast-path kernel requires CUDA + mamba-ssm",
)


def _has_mamba_ssm() -> bool:
    try:
        import mamba_ssm  # noqa: F401

        return True
    except Exception:
        return False


requires_mamba_ssm = pytest.mark.skipif(
    not _has_mamba_ssm(),
    reason="mamba-ssm (Triton ssd_combined) not installed",
)


def _base_kwargs():
    """Config fields common to every tiny build.

    Values are deliberately *consistent* with each other:
    ``mamba_d_ssm == mamba_n_heads * mamba_d_head`` and
    ``head_dim * num_attention_heads == hidden_size`` so the projections line
    up the way the real checkpoints do.
    """
    return dict(
        # --- transformer backbone ---
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        vocab_size=256,
        max_position_embeddings=128,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        attention_bias=False,
        attention_dropout=0.0,
        mlp_bias=False,
        rope_theta=10000.0,
        # --- mamba mixer ---
        mamba_d_ssm=128,          # == n_heads * d_head
        mamba_n_heads=16,
        mamba_d_head=8,
        mamba_n_groups=1,
        mamba_d_state=16,
        mamba_d_conv=4,
        mamba_chunk_size=16,
        mamba_expand=2,
        mamba_conv_bias=True,
        mamba_proj_bias=False,
        mamba_norm_before_gate=False,
        time_step_min=0.001,
        time_step_max=0.1,
        time_step_limit=(0.0, float("inf")),
        projectors_bias=False,
        # --- muP multipliers (non-trivial so bugs in wiring surface) ---
        embedding_multiplier=2.0,
        lm_head_multiplier=0.5,
        key_multiplier=0.9,
        attention_in_multiplier=1.1,
        attention_out_multiplier=0.8,
        ssm_in_multiplier=1.2,
        ssm_out_multiplier=0.7,
        mlp_multipliers=[1.3, 0.6],
        ssm_multipliers=[1.1, 1.2, 0.9, 0.8, 1.05],
    )


@pytest.fixture(params=[False, True], ids=["no_rms_norm", "rms_norm"])
def mamba_rms_norm(request):
    """Both gated-norm settings. The True case adds mamba.norm.weight keys."""
    return request.param


@pytest.fixture
def tiny_config(mamba_rms_norm):
    """Tiny but self-consistent FalconH1Config, untied embeddings."""
    pytest.importorskip("transformers")
    from transformers import FalconH1Config

    return FalconH1Config(
        **_base_kwargs(),
        mamba_rms_norm=mamba_rms_norm,
        tie_word_embeddings=False,
        dtype="float32",
    )


@pytest.fixture
def tied_config():
    """Tied-embedding variant — checkpoint has no separate lm_head.weight."""
    pytest.importorskip("transformers")
    from transformers import FalconH1Config

    return FalconH1Config(
        **_base_kwargs(),
        mamba_rms_norm=False,
        tie_word_embeddings=True,
        dtype="float32",
    )


def _abstract_methods(cls) -> set:
    """Names of still-abstract methods on a class (empty == concrete)."""
    return set(getattr(cls, "__abstractmethods__", frozenset()))


def make_adapter(config):
    """Construct FalconH1StateDictAdapter, failing with a clear message if the
    class is still abstract.

    The repo's StateDictAdapter base declares three abstract methods
    (to_hf, from_hf, convert_single_tensor_to_hf). If the subclass omits any,
    instantiation raises a bare ``TypeError: Can't instantiate abstract class``
    that, repeated across many tests, hides the single root cause. This wrapper
    turns that into one actionable failure naming the missing method(s).
    """
    from nemo_automodel.components.models.falcon_h1.state_dict_adapter import (
        FalconH1StateDictAdapter,
    )

    missing = _abstract_methods(FalconH1StateDictAdapter)
    if missing:
        pytest.fail(
            "FalconH1StateDictAdapter is abstract — missing implementations for: "
            f"{sorted(missing)}. The adapter cannot be instantiated and no "
            "checkpoint can be loaded until these are implemented.",
            pytrace=False,
        )
    return FalconH1StateDictAdapter(config)


@pytest.fixture
def adapter(tiny_config):
    return make_adapter(tiny_config)
