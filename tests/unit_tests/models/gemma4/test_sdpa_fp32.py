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

"""Unit tests for the fp32 SDPA attention fix (issue #2208).

The fused bf16 SDPA kernels are gradient-unstable for Gemma4 on Hopper. These
tests cover ``sdpa_fp32_attention_forward`` (runs SDPA with fp32 query/key/value
and casts the output back) and ``enable_gemma4_sdpa_fp32`` (installs it as the
``"sdpa"`` attention). CPU-only -- the behaviour under test is dtype handling and
registration, so no GPU is required.
"""

import torch

from nemo_automodel.components.models.gemma4_moe import sdpa_fp32
from nemo_automodel.components.models.gemma4_moe.sdpa_fp32 import (
    enable_gemma4_sdpa_fp32,
    sdpa_fp32_attention_forward,
)


class _RecordingSDPA:
    """Stand-in for ``transformers`` ``sdpa_attention_forward`` that records the
    dtypes it is called with and returns an output of a chosen dtype."""

    def __init__(self, out_dtype=torch.float32):
        self.calls = []
        self.out_dtype = out_dtype

    def __call__(self, module, query, key, value, attention_mask=None, **kwargs):
        self.calls.append(
            {
                "q": query.dtype,
                "k": key.dtype,
                "v": value.dtype,
                "mask": None if attention_mask is None else attention_mask.dtype,
            }
        )
        return torch.zeros_like(query, dtype=self.out_dtype), None


def _qkv(dtype=torch.bfloat16):
    t = torch.randn(1, 2, 4, 8, dtype=dtype)
    return t, t.clone(), t.clone()


def test_upcasts_qkv_to_fp32(monkeypatch):
    rec = _RecordingSDPA()
    monkeypatch.setattr(sdpa_fp32, "sdpa_attention_forward", rec)

    q, k, v = _qkv(torch.bfloat16)
    sdpa_fp32_attention_forward(torch.nn.Module(), q, k, v)

    call = rec.calls[0]
    assert call["q"] == call["k"] == call["v"] == torch.float32


def test_casts_output_back_to_input_dtype(monkeypatch):
    # The underlying SDPA runs in fp32; the wrapper must cast back to the input dtype.
    monkeypatch.setattr(sdpa_fp32, "sdpa_attention_forward", _RecordingSDPA(out_dtype=torch.float32))

    q, k, v = _qkv(torch.bfloat16)
    out, weights = sdpa_fp32_attention_forward(torch.nn.Module(), q, k, v)

    assert out.dtype == torch.bfloat16
    assert weights is None


def test_upcasts_floating_point_mask(monkeypatch):
    rec = _RecordingSDPA()
    monkeypatch.setattr(sdpa_fp32, "sdpa_attention_forward", rec)

    q, k, v = _qkv(torch.bfloat16)
    mask = torch.zeros(1, 1, 4, 4, dtype=torch.bfloat16)
    sdpa_fp32_attention_forward(torch.nn.Module(), q, k, v, attention_mask=mask)

    assert rec.calls[0]["mask"] == torch.float32


def test_non_float_mask_left_untouched(monkeypatch):
    rec = _RecordingSDPA()
    monkeypatch.setattr(sdpa_fp32, "sdpa_attention_forward", rec)

    q, k, v = _qkv(torch.bfloat16)
    mask = torch.ones(1, 1, 4, 4, dtype=torch.bool)
    sdpa_fp32_attention_forward(torch.nn.Module(), q, k, v, attention_mask=mask)

    assert rec.calls[0]["mask"] == torch.bool


def _restore_sdpa(registry, fn):
    try:
        registry.register("sdpa", fn)
    except Exception:
        pass
    for attr in ("_global_mapping", "_local_mapping"):
        mapping = getattr(registry, attr, None)
        if isinstance(mapping, dict) and "sdpa" in mapping:
            mapping["sdpa"] = fn


def test_enable_registers_fp32_sdpa():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    try:
        enable_gemma4_sdpa_fp32()
        assert ALL_ATTENTION_FUNCTIONS["sdpa"] is sdpa_fp32_attention_forward
    finally:
        _restore_sdpa(ALL_ATTENTION_FUNCTIONS, original)


# ---------------------------------------------------------------------------
# Usage: Gemma4ForConditionalGeneration.from_pretrained installs the fp32 SDPA
# attention only when attn_implementation="sdpa".
# ---------------------------------------------------------------------------
def _from_pretrained_calls(monkeypatch, attn_implementation):
    """Run the from_pretrained hook with model construction stubbed out and
    return the list recording whether enable_gemma4_sdpa_fp32 was called."""
    import pytest

    try:
        from nemo_automodel.components.models.gemma4_moe import model as g4model
    except ImportError as exc:  # e.g. older torch without FSDPModule, or no gemma4
        pytest.skip(f"gemma4_moe model unavailable: {exc}")

    if not g4model._GEMMA4_HF_AVAILABLE:
        pytest.skip("transformers gemma4 is not available")

    calls = []
    monkeypatch.setattr(g4model, "enable_gemma4_sdpa_fp32", lambda: calls.append(True))
    monkeypatch.setattr(g4model.Gemma4Config, "from_pretrained", classmethod(lambda cls, *a, **k: object()))
    monkeypatch.setattr(
        g4model.Gemma4ForConditionalGeneration, "from_config", classmethod(lambda cls, *a, **k: "MODEL")
    )

    result = g4model.Gemma4ForConditionalGeneration.from_pretrained(
        "dummy", attn_implementation=attn_implementation
    )
    assert result == "MODEL"
    return calls


def test_from_pretrained_enables_fp32_for_sdpa(monkeypatch):
    assert _from_pretrained_calls(monkeypatch, "sdpa") == [True]


def test_from_pretrained_skips_fp32_for_eager(monkeypatch):
    assert _from_pretrained_calls(monkeypatch, "eager") == []
