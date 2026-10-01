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

"""Unit tests for HunyuanImage3Adapter: per-sample conditioning, CFG dropout and the per-sample model calls."""

from types import SimpleNamespace

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.hunyuan_image3 import (
    HunyuanImage3Adapter,
    text_causal_image_bidirectional_mask,
)
from nemo_automodel.components.flow_matching.pipeline import create_adapter


def _conditioning(seq_len, image_start, token_h, token_w, offset):
    def one(shift):
        image_mask = torch.zeros(seq_len + shift, dtype=torch.bool)
        image_mask[image_start + shift : image_start + shift + token_h * token_w] = True
        return {
            "input_ids": torch.arange(seq_len + shift, dtype=torch.int32) + offset,
            "image_mask": image_mask,
            "timestep_scatter_index": torch.tensor([image_start + shift - 1]),
            "image_start": torch.tensor(image_start + shift),
        }

    cond, uncond = one(0), one(1)
    out = dict(cond, token_h=torch.tensor(token_h), token_w=torch.tensor(token_w))
    out.update({f"uncond_{k}": v for k, v in uncond.items()})
    return out


def _context(cfg_dropout_prob=0.0, token_h=2, token_w=3):
    latents = torch.randn(2, 4, token_h, token_w)
    return FlowMatchingContext(
        noisy_latents=torch.randn_like(latents),
        latents=latents,
        timesteps=torch.tensor([250, 750]),
        sigma=torch.rand(2),
        task_type="t2v",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch={"conditioning": [_conditioning(12, 4, 2, 3, 0), _conditioning(12, 4, 2, 3, 100)]},
        cfg_dropout_prob=cfg_dropout_prob,
    )


class _RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, input_ids, **kwargs):
        self.calls.append(dict(kwargs, input_ids=input_ids))
        return SimpleNamespace(diffusion_prediction=kwargs["images"] * 2)


def test_registered():
    assert isinstance(create_adapter("hunyuan_image3"), HunyuanImage3Adapter)


def test_mask_is_causal_with_bidirectional_image_block():
    mask = text_causal_image_bidirectional_mask(6, 2, 3, "cpu")[0, 0]
    expected = torch.ones(6, 6, dtype=torch.bool).tril()
    expected[2:5, 2:5] = True
    torch.testing.assert_close(mask, expected)


def test_prepare_inputs_uses_conditional_sequence():
    inputs = HunyuanImage3Adapter().prepare_inputs(_context())
    assert [s["token_hw"] for s in inputs["samples"]] == [(2, 3), (2, 3)]
    assert inputs["samples"][1]["input_ids"][0] == 100
    assert int(inputs["samples"][0]["image_start"]) == 4
    assert inputs["timesteps"].dtype == torch.float32


def test_cfg_dropout_swaps_in_unconditional_sequence():
    inputs = HunyuanImage3Adapter().prepare_inputs(_context(cfg_dropout_prob=1.0))
    assert all(int(s["image_start"]) == 5 for s in inputs["samples"])
    assert inputs["samples"][0]["input_ids"].shape == (13,)


def test_prepare_inputs_requires_conditioning():
    ctx = _context()
    ctx.batch.pop("conditioning")
    with pytest.raises(ValueError, match="conditioning"):
        HunyuanImage3Adapter().prepare_inputs(ctx)


def test_forward_runs_one_call_per_sample():
    adapter = HunyuanImage3Adapter()
    ctx = _context()
    model = _RecordingModel()
    pred = adapter.forward(model, adapter.prepare_inputs(ctx))
    torch.testing.assert_close(pred, ctx.noisy_latents * 2)
    assert len(model.calls) == 2
    call = model.calls[0]
    assert call["mode"] == "gen_image"
    assert call["input_ids"].shape == (1, 12) and call["input_ids"].dtype == torch.long
    assert call["rope_positions"].shape == (1, 12, 2)
    assert call["attention_mask"].shape == (1, 1, 12, 12)
    assert call["image_mask"].sum() == 6
    torch.testing.assert_close(call["timestep"], torch.tensor([250.0]))


def test_forward_rejects_latents_that_do_not_match_the_image_block():
    adapter = HunyuanImage3Adapter()
    inputs = adapter.prepare_inputs(_context())
    inputs["samples"][0]["token_hw"] = (3, 2)
    with pytest.raises(ValueError, match="does not match"):
        adapter.forward(_RecordingModel(), inputs)
