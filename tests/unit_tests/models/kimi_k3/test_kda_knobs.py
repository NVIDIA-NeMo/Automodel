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
"""KimiK3 KDA kernel knobs: conv backend, FLA in-backward recompute, state layout, discarded final state.

The layer runs with FLA's kernels replaced by recording stubs: these tests check the keyword arguments
Automodel passes to FLA, not FLA itself, and a first call into the real kernels would JIT-compile them
for minutes inside the unit-test shard.
"""

from __future__ import annotations

import warnings

import pytest
import torch
from torch import nn

from nemo_automodel.components.models.kimi_k3 import model as kmod
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig

pytest.importorskip("fla")


def _small_config(**overrides) -> KimiK3TextConfig:
    kwargs = dict(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        linear_attn_config={
            "head_dim": 16,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1, 2, 3],
            "full_attn_layers": [4],
        },
        # The fused decay gate is a Triton kernel; the torch gate keeps these tests on the CPU.
        kda_use_fused_gate=False,
    )
    kwargs.update(overrides)
    return KimiK3TextConfig(**kwargs)


class _RecordingConv(nn.Module):
    """Stand-in for FLA's short convolution: records its keyword arguments, passes the input through."""

    def __init__(self, seen: dict) -> None:
        super().__init__()
        self.seen = seen

    def forward(self, x: torch.Tensor, **kwargs):
        self.seen.update({key: value for key, value in kwargs.items() if not torch.is_tensor(value)})
        return x, None


class _PassThroughNorm(nn.Module):
    """Stand-in for FLA's gated RMSNorm."""

    def forward(self, o: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        del gate
        return o


def _run_layer_with_stubbed_kernels(monkeypatch, cfg: KimiK3TextConfig) -> tuple[dict, dict]:
    """Run one CPU forward of ``KimiDeltaAttention`` with FLA's kernels stubbed out.

    Returns the non-tensor keyword arguments the layer passed to ``chunk_kda`` and to the
    short convolutions. The real kernels are never called: a first call would JIT-compile
    them for minutes, and the tests only check what Automodel asks FLA to do.
    """
    kernel_seen: dict = {}
    conv_seen: dict = {}

    def fake_chunk_kda(*, q, k, v, g, beta, **kwargs):
        del q, k, g, beta
        kernel_seen.update({key: value for key, value in kwargs.items() if not torch.is_tensor(value)})
        return torch.zeros_like(v), None

    monkeypatch.setattr(kmod, "chunk_kda", fake_chunk_kda)
    layer = kmod.KimiDeltaAttention(cfg, layer_idx=0)
    with torch.no_grad():
        layer.init_weights(torch.device("cpu"), init_std=0.02)
    layer.q_conv1d = _RecordingConv(conv_seen)
    layer.k_conv1d = _RecordingConv(conv_seen)
    layer.v_conv1d = _RecordingConv(conv_seen)
    layer.o_norm = _PassThroughNorm()

    x = torch.randn(1, 8, cfg.hidden_size, dtype=torch.bfloat16)
    out = layer(x)
    assert out.shape == x.shape
    return kernel_seen, conv_seen


def test_config_defaults_keep_the_reference_kernel_path():
    cfg = _small_config()
    assert cfg.kda_disable_recompute is False
    assert cfg.kda_conv_backend == "triton"
    assert cfg.kda_transpose_state_layout is True


def test_config_rejects_unknown_conv_backend():
    with pytest.raises(ValueError, match="kda_conv_backend"):
        _small_config(kda_conv_backend="cudnn")


def test_short_conv_backend_kwargs_only_for_non_default():
    assert kmod._short_conv_backend_kwargs("triton") == {}
    kw = kmod._short_conv_backend_kwargs("cuda")
    # FLA >= 0.4 exposes ``backend``; older releases get no keyword at all.
    assert kw in ({}, {"backend": "cuda"})


def test_conv_backend_reaches_the_short_convolutions():
    if kmod._short_conv_backend_kwargs("cuda") != {"backend": "cuda"}:
        pytest.skip("the installed FLA ShortConvolution has no backend parameter")
    reference = kmod.KimiDeltaAttention(_small_config(), layer_idx=0)
    assert reference.q_conv1d._fp32_params.backend == "triton"

    try:
        import causal_conv1d  # noqa: F401

        expected = "cuda"
    except ImportError:
        # FLA warns and falls back to its Triton kernels when causal_conv1d is not installed.
        expected = "triton"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        knobbed = kmod.KimiDeltaAttention(_small_config(kda_conv_backend="cuda"), layer_idx=0)
    for conv in (knobbed.q_conv1d, knobbed.k_conv1d, knobbed.v_conv1d):
        assert conv._fp32_params.backend == expected


def test_default_options_keep_the_reference_path_and_never_request_the_final_state(monkeypatch):
    kernel_seen, conv_seen = _run_layer_with_stubbed_kernels(monkeypatch, _small_config())
    assert kernel_seen["output_final_state"] is False
    assert kernel_seen["transpose_state_layout"] is True
    assert "disable_recompute" not in kernel_seen
    assert conv_seen["output_final_state"] is False


def test_kernel_options_follow_the_config(monkeypatch):
    cfg = _small_config(kda_disable_recompute=True, kda_transpose_state_layout=False)
    kernel_seen, _ = _run_layer_with_stubbed_kernels(monkeypatch, cfg)
    assert kernel_seen["output_final_state"] is False
    assert kernel_seen["transpose_state_layout"] is False
    if kmod._CHUNK_KDA_HAS_DISABLE_RECOMPUTE:
        assert kernel_seen["disable_recompute"] is True
    else:
        assert "disable_recompute" not in kernel_seen
