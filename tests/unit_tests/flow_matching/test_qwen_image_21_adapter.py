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

"""Unit tests for QwenImage21Adapter: pack/unpack latents, prepare_inputs, forward."""

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.qwen_image_21 import QwenImage21Adapter
from nemo_automodel.components.flow_matching.pipeline import create_adapter


def _make_context(
    batch_size=2,
    channels=64,
    height=8,
    width=6,
    text_len=16,
    text_dim=32,
    text_attention_mask=None,
    cfg_dropout_prob=0.0,
):
    batch = {"text_embeddings": torch.randn(batch_size, text_len, text_dim)}
    if text_attention_mask is not None:
        batch["text_attention_mask"] = text_attention_mask
    latents = torch.randn(batch_size, channels, height, width)
    return FlowMatchingContext(
        noisy_latents=torch.randn_like(latents),
        latents=latents,
        timesteps=torch.tensor([250.0, 750.0][:batch_size] + [500.0] * max(0, batch_size - 2)),
        sigma=torch.rand(batch_size),
        task_type="t2v",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
        cfg_dropout_prob=cfg_dropout_prob,
    )


class TestPackUnpack:
    def test_pack_is_row_major_flatten(self):
        latents = torch.arange(2 * 3 * 4 * 6, dtype=torch.float32).reshape(2, 3, 4, 6)
        packed = QwenImage21Adapter._pack_latents(latents)
        assert packed.shape == (2, 24, 3)
        # Token (row 1, col 2) holds every channel of that pixel.
        torch.testing.assert_close(packed[:, 1 * 6 + 2], latents[:, :, 1, 2])

    def test_roundtrip(self):
        latents = torch.randn(2, 64, 8, 6)
        packed = QwenImage21Adapter._pack_latents(latents)
        torch.testing.assert_close(QwenImage21Adapter._unpack_latents(packed, 8, 6), latents)

    def test_unpack_rejects_wrong_token_count(self):
        with pytest.raises(ValueError, match="target tokens"):
            QwenImage21Adapter._unpack_latents(torch.randn(1, 10, 64), 4, 4)


class TestPrepareInputs:
    def test_shapes(self):
        adapter = QwenImage21Adapter()
        ctx = _make_context(batch_size=2, height=8, width=6, text_len=16)
        inputs = adapter.prepare_inputs(ctx)

        assert inputs["hidden_states"].shape == (2, 48, 64)
        assert inputs["encoder_hidden_states"].shape == (2, 16, 32)
        assert inputs["img_shapes"] == [[(1, 8, 6)], [(1, 8, 6)]]
        assert inputs["_text_lengths"] == [16, 16]
        torch.testing.assert_close(inputs["timestep"], torch.tensor([0.25, 0.75]))

    def test_records_per_sample_lengths_and_trims_shared_padding(self):
        adapter = QwenImage21Adapter()
        mask = torch.zeros(2, 16, dtype=torch.bool)
        mask[0, :9] = True
        mask[1, :13] = True
        ctx = _make_context(text_len=16, text_attention_mask=mask)
        inputs = adapter.prepare_inputs(ctx)

        assert inputs["_text_lengths"] == [9, 13]
        torch.testing.assert_close(inputs["encoder_hidden_states"], ctx.batch["text_embeddings"][:, :13])

    def test_cfg_dropout_zeroes_text(self):
        adapter = QwenImage21Adapter()
        inputs = adapter.prepare_inputs(_make_context(cfg_dropout_prob=1.0))
        assert torch.count_nonzero(inputs["encoder_hidden_states"]) == 0

    def test_rejects_odd_latent_sides(self):
        with pytest.raises(ValueError, match="must be even"):
            QwenImage21Adapter().prepare_inputs(_make_context(height=7, width=6))

    def test_rejects_5d_latents(self):
        ctx = _make_context()
        ctx.noisy_latents = ctx.noisy_latents.unsqueeze(2)
        with pytest.raises(ValueError, match="4D"):
            QwenImage21Adapter().prepare_inputs(ctx)


class _RecordingModel:
    """Fake transformer: text outputs are NaN, target outputs are 2 * input latents."""

    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        batch, text_len = kwargs["encoder_hidden_states"].shape[:2]
        text_out = torch.full((batch, text_len, kwargs["hidden_states"].shape[-1]), float("nan"))
        return (torch.cat([text_out, kwargs["hidden_states"] * 2], dim=1),)


class TestForward:
    def test_one_unpadded_call_per_sample(self):
        adapter = QwenImage21Adapter()
        ctx = _make_context(batch_size=2, height=4, width=4, text_len=5)
        model = _RecordingModel()

        pred = adapter.forward(model, adapter.prepare_inputs(ctx))

        assert pred.shape == (2, 64, 4, 4)
        torch.testing.assert_close(pred, ctx.noisy_latents * 2)
        # Equal prompt lengths still give one call per sample: under FSDP every rank must make the same
        # number of calls, so the count must not depend on the prompt lengths a rank happens to receive.
        assert len(model.calls) == 2
        for call in model.calls:
            assert call["encoder_hidden_states_mask"] is None
            assert call["return_dict"] is False
            assert "guidance" not in call
            # 5 text positions followed by 16 / 4 target slots.
            assert call["img_mask"].shape == (1, 5 + 4)
            assert not call["img_mask"][:, :5].any()
            assert call["img_mask"][:, 5:].all()

    def test_ragged_batch_trims_each_sample(self):
        adapter = QwenImage21Adapter()
        mask = torch.zeros(3, 8, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :8] = True
        mask[2, :5] = True
        ctx = _make_context(batch_size=3, height=4, width=4, text_len=8, text_attention_mask=mask)
        model = _RecordingModel()

        pred = adapter.forward(model, adapter.prepare_inputs(ctx))

        torch.testing.assert_close(pred, ctx.noisy_latents * 2)
        assert len(model.calls) == 3
        text = ctx.batch["text_embeddings"]
        for i, (call, length) in enumerate(zip(model.calls, [5, 8, 5])):
            torch.testing.assert_close(call["encoder_hidden_states"], text[i : i + 1, :length])
            torch.testing.assert_close(call["timestep"], ctx.timesteps[i : i + 1] / 1000)
            assert call["encoder_hidden_states_mask"] is None
            assert call["img_mask"].shape == (1, length + 4)

    def test_factory(self):
        assert isinstance(create_adapter("qwen_image_21"), QwenImage21Adapter)


def _tiny_transformer():
    diffusers = pytest.importorskip("diffusers")
    if not hasattr(diffusers, "QwenImage21Transformer2DModel"):
        pytest.skip("diffusers build does not ship QwenImage21Transformer2DModel")
    torch.manual_seed(0)
    return diffusers.QwenImage21Transformer2DModel(
        in_channels=8,
        out_channels=8,
        num_layers=2,
        attention_head_dim=16,
        num_attention_heads=2,
        context_in_dim=24,
        mlp_ratio=2,
        axes_dims_rope=(4, 6, 6),
    )


class TestWithDiffusersTransformer:
    def test_forward_backward(self):
        model = _tiny_transformer()
        adapter = QwenImage21Adapter()
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, :5] = True
        mask[1, :7] = True
        ctx = _make_context(channels=8, height=4, width=6, text_len=8, text_dim=24, text_attention_mask=mask)

        pred = adapter.forward(model, adapter.prepare_inputs(ctx))
        assert pred.shape == (2, 8, 4, 6)
        assert torch.isfinite(pred).all()

        pred.float().pow(2).mean().backward()
        assert model.img_in.weight.grad is not None
        assert model.txt_in.in_layer.weight.grad is not None

    def test_ragged_batch_matches_single_sample_calls(self):
        """Each sample in a ragged batch must get what the pipeline gives it alone, unpadded."""
        model = _tiny_transformer().eval()
        adapter = QwenImage21Adapter()
        lengths = [4, 9, 4]
        mask = torch.zeros(3, 9, dtype=torch.bool)
        for i, length in enumerate(lengths):
            mask[i, :length] = True
        ctx = _make_context(
            batch_size=3, channels=8, height=4, width=6, text_len=9, text_dim=24, text_attention_mask=mask
        )
        text = ctx.batch["text_embeddings"]

        with torch.no_grad():
            pred = adapter.forward(model, adapter.prepare_inputs(ctx))
            for i, length in enumerate(lengths):
                ref = _pipeline_style_call(
                    model, ctx.noisy_latents[i : i + 1], ctx.timesteps[i : i + 1], text[i : i + 1, :length]
                )
                torch.testing.assert_close(pred[i : i + 1], ref)

            # Sensitivity check: padding the short prompt to the batch length (masked) shifts its RoPE layout.
            padded = _pipeline_style_call(model, ctx.noisy_latents[:1], ctx.timesteps[:1], text[:1], text_mask=mask[:1])
            assert not torch.allclose(pred[:1], padded)

    def test_matches_pipeline_style_call(self):
        """The adapter must feed the transformer exactly what QwenImage21Pipeline does for T2I."""
        model = _tiny_transformer().eval()
        adapter = QwenImage21Adapter()
        ctx = _make_context(batch_size=1, channels=8, height=4, width=6, text_len=7, text_dim=24)

        with torch.no_grad():
            pred = adapter.forward(model, adapter.prepare_inputs(ctx))
            ref = _pipeline_style_call(model, ctx.noisy_latents, ctx.timesteps, ctx.batch["text_embeddings"])

        torch.testing.assert_close(pred, ref)


def _pipeline_style_call(model, noisy_latents, timesteps, text, text_mask=None):
    """Call the transformer the way QwenImage21Pipeline does for T2I; returns [B, C, H, W] target output."""
    b, c, h, w = noisy_latents.shape
    latents = noisy_latents.reshape(b, c, h * w).transpose(1, 2)
    image_pad_mask = torch.zeros(b, text.shape[1], dtype=torch.bool)
    img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(b, latents.shape[1] // 4)], dim=1)
    out = model(
        hidden_states=latents,
        timestep=timesteps / 1000,
        encoder_hidden_states=text,
        encoder_hidden_states_mask=text_mask,
        img_shapes=[[(1, h, w)]] * b,
        img_mask=img_mask,
        return_dict=False,
    )[0][:, -latents.size(1) :]
    return out.transpose(1, 2).reshape(b, c, h, w)
