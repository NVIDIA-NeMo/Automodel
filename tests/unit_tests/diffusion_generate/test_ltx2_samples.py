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

"""Unit tests for the LTX-2 finetune evaluation sampler helpers."""

import copy

import numpy as np
import pytest
import torch

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from tools.diffusion.generate_ltx2_samples import (
    frames_to_uint8,
    merge_lora_into_transformer,
    waveform_to_2d,
)


class _Attn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.to_q = torch.nn.Linear(32, 32)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(32, 32), torch.nn.Dropout(0.0)])

    def forward(self, x):
        return self.to_out[0](self.to_q(x))


class _Block(torch.nn.Module):
    """Deliberately contains both attn1 and audio_attn1 — their names collide
    under naive suffix matching, the exact bug the merge matcher must avoid."""

    def __init__(self):
        super().__init__()
        self.attn1 = _Attn()
        self.audio_attn1 = _Attn()

    def forward(self, x):
        return self.attn1(x) + self.audio_attn1(x)


def _make_lora_pair():
    torch.manual_seed(0)
    base = _Block()
    lora_model = copy.deepcopy(base)
    n = apply_lora_to_linear_modules(
        lora_model, PeftConfig(target_modules=["*.to_q", "*.to_out.0"], dim=4, alpha=16, dropout=0.0)
    )
    state = {}
    for name, p in lora_model.named_parameters():
        if "lora_B" in name:
            with torch.no_grad():
                p.copy_(torch.randn_like(p) * 0.05)
        if "lora_" in name:
            state[name] = p.detach().clone()
    return base, lora_model, state, n


class TestMergeLora:
    def test_merged_forward_matches_lora_forward(self):
        base, lora_model, state, n = _make_lora_pair()
        merged = copy.deepcopy(base)
        count = merge_lora_into_transformer(merged, state, scale=16 / 4)
        assert count == n == 4
        x = torch.randn(2, 32)
        lora_model.eval()
        merged.eval()
        torch.testing.assert_close(merged(x), lora_model(x), atol=1e-5, rtol=1e-5)

    def test_prefixed_checkpoint_keys(self):
        base, lora_model, state, _ = _make_lora_pair()
        merged = copy.deepcopy(base)
        count = merge_lora_into_transformer(merged, {f"model.{k}": v for k, v in state.items()}, scale=16 / 4)
        assert count == 4
        x = torch.randn(2, 32)
        torch.testing.assert_close(merged.eval()(x), lora_model.eval()(x), atol=1e-5, rtol=1e-5)

    def test_audio_video_name_collision_not_cross_merged(self):
        # Merging ONLY the audio_attn1 deltas must leave attn1 weights untouched.
        base, _, state, _ = _make_lora_pair()
        audio_only = {k: v for k, v in state.items() if "audio" in k}
        merged = copy.deepcopy(base)
        merge_lora_into_transformer(merged, audio_only, scale=16 / 4)
        assert torch.equal(merged.attn1.to_q.weight, base.attn1.to_q.weight)
        assert not torch.equal(merged.audio_attn1.to_q.weight, base.audio_attn1.to_q.weight)

    def test_missing_module_raises(self):
        base, _, _, _ = _make_lora_pair()
        with pytest.raises(KeyError, match="no matching base weight"):
            merge_lora_into_transformer(
                base,
                {"nonexistent.lora_A.weight": torch.zeros(4, 32), "nonexistent.lora_B.weight": torch.zeros(32, 4)},
                scale=1.0,
            )


class TestOutputConversion:
    def test_frames_from_float_ndarray(self):
        frames = frames_to_uint8(np.random.rand(1, 5, 8, 8, 3).astype(np.float32))
        assert frames.shape == (5, 8, 8, 3) and frames.dtype == np.uint8

    def test_frames_from_tchw_tensor(self):
        frames = frames_to_uint8(torch.rand(5, 3, 8, 8))
        assert frames.shape == (5, 8, 8, 3) and frames.dtype == np.uint8

    def test_waveform_shapes(self):
        assert waveform_to_2d(torch.randn(1, 2, 100)).shape == (2, 100)
        assert waveform_to_2d(torch.randn(100)).shape == (1, 100)
        assert waveform_to_2d(np.random.randn(2, 100)).shape == (2, 100)
