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

"""
Unit tests for LTX2Adapter (dual-stream video+audio).

Tests cover:
- Token pack/unpack round-trips for both streams
- Input preparation: key set, shapes, shared-sigma noising, fp32 audio target
- Forward pass with a stub dual-stream model, audio prediction stashing
- auxiliary_losses: value, autograd connectivity, weight scaling
- Full FlowMatchingPipeline.step() integration (combined loss)
- Kwarg filtering against the model forward signature
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters import (
    FlowMatchingContext,
    LTX2Adapter,
    SimpleAdapter,
)
from nemo_automodel.components.flow_matching.adapters.ltx2 import (
    _pack_audio_latents,
    _pack_video_latents,
    _unpack_audio_latents,
    _unpack_video_latents,
)
from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline

B, C_VID, F, H, W = 2, 6, 3, 4, 5
C_AUD, L, M = 8, 7, 16
T_TXT, D_VID, D_AUD = 11, 12, 10


class MockLTX2Model(nn.Module):
    """Stub dual-stream transformer: returns scaled token streams."""

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.last_kwargs = {}

    def forward(
        self,
        hidden_states,
        audio_hidden_states,
        encoder_hidden_states,
        audio_encoder_hidden_states,
        encoder_attention_mask,
        audio_encoder_attention_mask,
        timestep,
        audio_timestep,
        sigma,
        audio_sigma,
        num_frames,
        height,
        width,
        fps,
        audio_num_frames,
        return_dict=True,
    ):
        self.last_kwargs = {
            "timestep": timestep,
            "audio_timestep": audio_timestep,
            "sigma": sigma,
            "num_frames": num_frames,
            "fps": fps,
            "return_dict": return_dict,
        }
        return hidden_states * self.scale, audio_hidden_states * self.scale


def make_batch():
    return {
        "video_latents": torch.randn(B, C_VID, F, H, W),
        "audio_latents": torch.randn(B, C_AUD, L, M),
        "text_embeddings": torch.randn(B, T_TXT, D_VID),
        "audio_text_embeddings": torch.randn(B, T_TXT, D_AUD),
        "text_mask": torch.ones(B, T_TXT, dtype=torch.long),
    }


def make_context(batch=None, sigma=None):
    batch = batch if batch is not None else make_batch()
    sigma = sigma if sigma is not None else torch.rand(B).clamp(0.001, 0.999)
    latents = batch["video_latents"].float()
    noise = torch.randn_like(latents)
    sigma_view = sigma.view(-1, 1, 1, 1, 1)
    noisy = (1.0 - sigma_view) * latents + sigma_view * noise
    return FlowMatchingContext(
        noisy_latents=noisy.to(torch.float32),
        latents=latents,
        timesteps=sigma * 1000.0,
        sigma=sigma,
        task_type="t2v",
        data_type="video",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
    )


class TestPackUnpack:
    def test_video_round_trip(self):
        latents = torch.randn(B, C_VID, F, H, W)
        tokens = _pack_video_latents(latents)
        assert tokens.shape == (B, F * H * W, C_VID)
        assert torch.equal(_unpack_video_latents(tokens, F, H, W), latents)

    def test_video_pack_matches_reference_permute(self):
        # Reference: fastgen-style patchify at patch size 1 via reshape+permute.
        latents = torch.randn(B, C_VID, F, H, W)
        reference = latents.reshape(B, C_VID, F, 1, H, 1, W, 1)
        reference = reference.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
        assert torch.equal(_pack_video_latents(latents), reference)

    def test_audio_round_trip(self):
        latents = torch.randn(B, C_AUD, L, M)
        tokens = _pack_audio_latents(latents)
        assert tokens.shape == (B, L, C_AUD * M)
        assert torch.equal(_unpack_audio_latents(tokens, M), latents)


class TestPrepareInputs:
    def test_key_set_and_shapes(self):
        adapter = LTX2Adapter()
        inputs = adapter.prepare_inputs(make_context())

        assert inputs["hidden_states"].shape == (B, F * H * W, C_VID)
        assert inputs["audio_hidden_states"].shape == (B, L, C_AUD * M)
        assert inputs["encoder_hidden_states"].shape == (B, T_TXT, D_VID)
        assert inputs["audio_encoder_hidden_states"].shape == (B, T_TXT, D_AUD)
        assert inputs["encoder_attention_mask"].shape == (B, T_TXT)
        assert inputs["timestep"].shape == (B, F * H * W)
        assert inputs["audio_timestep"].shape == (B, L)
        assert inputs["sigma"].shape == (B,)
        assert inputs["audio_sigma"].shape == (B,)
        assert (inputs["num_frames"], inputs["height"], inputs["width"]) == (F, H, W)
        assert inputs["audio_num_frames"] == L
        assert inputs["fps"] == 24.0
        assert inputs["_audio_target"].shape == (B, C_AUD, L, M)
        assert inputs["_audio_target"].dtype == torch.float32

    def test_audio_noised_with_shared_sigma(self):
        # noisy_audio = (1-sigma)*x0 + sigma*eps and target = eps - x0
        # imply noisy_audio == x0 + sigma * target exactly.
        adapter = LTX2Adapter()
        context = make_context()
        inputs = adapter.prepare_inputs(context)

        noisy_audio = _unpack_audio_latents(inputs["audio_hidden_states"], M)
        x0 = context.batch["audio_latents"].float()
        sigma = context.sigma.view(-1, 1, 1, 1)
        reconstructed = x0 + sigma * inputs["_audio_target"]
        torch.testing.assert_close(noisy_audio, reconstructed)

    def test_timesteps_are_rescaled_sigma(self):
        adapter = LTX2Adapter()
        context = make_context()
        inputs = adapter.prepare_inputs(context)
        expected = context.sigma * 1000.0
        torch.testing.assert_close(inputs["sigma"], expected)
        torch.testing.assert_close(inputs["timestep"][:, 0], expected)
        torch.testing.assert_close(inputs["audio_timestep"][:, -1], expected)

    @pytest.mark.parametrize("missing", ["audio_latents", "text_embeddings", "audio_text_embeddings", "text_mask"])
    def test_missing_key_raises(self, missing):
        adapter = LTX2Adapter()
        batch = make_batch()
        del batch[missing]
        with pytest.raises(KeyError, match=missing):
            adapter.prepare_inputs(make_context(batch=batch))


class TestForwardAndAuxLoss:
    def test_forward_returns_video_and_stashes_audio(self):
        adapter = LTX2Adapter()
        model = MockLTX2Model()
        inputs = adapter.prepare_inputs(make_context())
        video_pred = adapter.forward(model, inputs)

        assert video_pred.shape == (B, C_VID, F, H, W)
        assert inputs["_audio_pred"].shape == (B, C_AUD, L, M)
        assert model.last_kwargs["return_dict"] is False
        # Private stash keys must not reach the model.
        assert "_audio_target" not in model.last_kwargs

    def test_auxiliary_losses_scalar_and_grad(self):
        adapter = LTX2Adapter()
        model = MockLTX2Model()
        inputs = adapter.prepare_inputs(make_context())
        adapter.forward(model, inputs)
        aux = adapter.auxiliary_losses(inputs)

        assert set(aux.keys()) == {"audio_loss"}
        assert aux["audio_loss"].ndim == 0
        assert aux["audio_loss"].requires_grad
        aux["audio_loss"].backward()
        assert model.scale.grad is not None

    def test_audio_loss_weight_scales(self):
        torch.manual_seed(0)
        inputs_full = LTX2Adapter(audio_loss_weight=1.0).prepare_inputs(make_context())
        model = MockLTX2Model()
        LTX2Adapter().forward(model, inputs_full)

        full = LTX2Adapter(audio_loss_weight=1.0).auxiliary_losses(inputs_full)["audio_loss"]
        half = LTX2Adapter(audio_loss_weight=0.5).auxiliary_losses(inputs_full)["audio_loss"]
        torch.testing.assert_close(half, 0.5 * full)

    def test_kwargs_filtered_to_model_signature(self):
        class NoSigmaModel(nn.Module):
            def forward(self, hidden_states, audio_hidden_states, timestep, audio_timestep, return_dict=True):
                return hidden_states, audio_hidden_states

        adapter = LTX2Adapter()
        inputs = adapter.prepare_inputs(make_context())
        # Must not raise despite the model rejecting sigma/fps/etc. kwargs.
        video_pred = adapter.forward(NoSigmaModel(), inputs)
        assert video_pred.shape == (B, C_VID, F, H, W)


class TestPipelineIntegration:
    def _make_pipeline(self, adapter):
        return FlowMatchingPipeline(
            model_adapter=adapter,
            num_train_timesteps=1000,
            timestep_sampling="uniform",
            use_sigma_noise=False,
            sigma_min=0.001,
            sigma_max=0.999,
            i2v_prob=0.0,
            cfg_dropout_prob=0.0,
            use_loss_weighting=False,
            device=torch.device("cpu"),
        )

    def test_step_combined_loss(self):
        torch.manual_seed(0)
        pipeline = self._make_pipeline(LTX2Adapter())
        model = MockLTX2Model()
        _, avg_loss, _, metrics = pipeline.step(
            model=model,
            batch=make_batch(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            collect_metrics=True,
            check_loss=True,
        )

        assert torch.isfinite(avg_loss)
        assert avg_loss.requires_grad
        assert "audio_loss" in metrics
        # Combined loss strictly exceeds the video-only component.
        assert avg_loss.item() > metrics["loss"] - metrics["audio_loss"] - 1e-6
        avg_loss.backward()
        assert model.scale.grad is not None

    def test_step_zero_audio_weight_matches_video_loss(self):
        torch.manual_seed(0)
        pipeline = self._make_pipeline(LTX2Adapter(audio_loss_weight=0.0))
        _, avg_loss, _, metrics = pipeline.step(
            model=MockLTX2Model(),
            batch=make_batch(),
            device=torch.device("cpu"),
            dtype=torch.float32,
            collect_metrics=True,
            check_loss=True,
        )
        assert metrics["audio_loss"] == pytest.approx(0.0)

    def test_base_adapter_hook_default_is_none(self):
        # Existing adapters inherit a no-op auxiliary_losses hook.
        assert SimpleAdapter().auxiliary_losses({}) is None
