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

"""Shared fixtures: a tiny Beam config covering every layer flavour of the architecture."""

import os

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig

transformers = pytest.importorskip("transformers")
# Load the HF Beam config and modeling modules at collection time: the first import pays transformers' package metadata
# scan (~5 s cold), which would otherwise count against the runtime budget of whichever test runs first.
pytest.importorskip("transformers.models.beam.configuration_beam")
pytest.importorskip("transformers.models.beam.modeling_beam")


@pytest.fixture(autouse=True)
def _bounded_cpu_threads():
    """Cap intra-op CPU threads for the tiny models used here.

    With one thread per core, the CPU variants spend most of their time scheduling threads on large hosts (5-6 s per
    dropout test on a 104-core node versus 0.2-0.9 s with 4 threads), which breaks the unit-test runtime budget.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def _eager_moe_swiglu(monkeypatch):
    """Run the shared ``torch.compile``d MoE SwiGLU forward/backward helpers eagerly.

    These tests check Beam semantics (dropout, checkpointing, HF parity, FSDP/RoPE precision), not kernel compilation.
    Compiling the helpers on first use costs a cold Inductor build that lands on whichever test runs first in a
    process. ``weighted_bias_swiglu_impl`` resolves them through the module globals patched here; FlexAttention keeps
    its own compilation. ``TestSharedExpertActivationBoundary`` overrides this fixture to pin the compiled kernel.
    """
    from nemo_automodel.components.moe.megatron import moe_utils

    for name in ("swiglu", "weighted_swiglu", "swiglu_back", "weighted_swiglu_back"):
        monkeypatch.setattr(moe_utils, name, getattr(moe_utils, name)._torchdynamo_orig_callable)


@pytest.fixture
def beam_config():
    """4 layers: dense+sliding+RoPE, MoE+full+NoPE(SSMax), MoE+sliding+RoPE, MoE+full+NoPE(SSMax)."""
    from transformers.models.beam import BeamConfig

    cfg = BeamConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=256,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse", "sparse"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": None,
        },
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        n_group=2,
        topk_group=1,
        routed_scaling_factor=2.5,
        norm_topk_prob=True,
        final_logit_softcapping=30.0,
        tie_word_embeddings=False,
        attention_bias=False,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    cfg.torch_dtype = torch.float32
    return cfg


def _backend(attn: str, linear: str = "torch", rms_norm: str = "torch") -> BackendConfig:
    return BackendConfig(
        attn=attn,
        linear=linear,
        rms_norm=rms_norm,
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=False,
    )


@pytest.fixture
def backend_config():
    return _backend("sdpa")


@pytest.fixture
def flex_backend_config():
    return _backend("flex")


def _te_available() -> bool:
    try:
        import transformer_engine.pytorch  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture
def te_backend_config():
    if not _te_available():
        pytest.skip("TransformerEngine not installed")
    return _backend("te")


@pytest.fixture
def te_full_backend_config():
    """TE attention plus TE Linear and TE RMSNorm (the production backend)."""
    if not _te_available():
        pytest.skip("TransformerEngine not installed")
    return _backend("te", linear="te", rms_norm="te")


@pytest.fixture
def fp32_matmul():
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous)
