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

"""Unit tests for Flux3Adapter (flux3.py): pack/unpack latents, positional IDs, per-sample timesteps, forward."""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.flux3 import Flux3Adapter
from nemo_automodel.components.flow_matching.pipeline import create_adapter

# =============================================================================
# TestFlux3AdapterPackUnpackLatents
# =============================================================================


class TestFlux3AdapterPackUnpackLatents:
    """Tests for Flux3Adapter._pack_latents / _unpack_latents."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 128, 8, 8),
            (2, 128, 16, 16),
            (4, 128, 8, 16),
        ],
    )
    def test_pack_shape(self, b, c, h, w):
        adapter = Flux3Adapter()
        latents = torch.randn(b, c, h, w)
        packed = adapter._pack_latents(latents)
        assert packed.shape == (b, h * w, c)

    def test_pack_matches_flatten_transpose(self):
        # The upstream reference packs via latents.flatten(2).transpose(1, 2).
        adapter = Flux3Adapter()
        latents = torch.randn(2, 8, 4, 6)
        packed = adapter._pack_latents(latents)
        assert torch.equal(packed, latents.flatten(2).transpose(1, 2))

    def test_pack_unpack_roundtrip(self):
        adapter = Flux3Adapter()
        latents = torch.randn(2, 128, 8, 8)
        packed = adapter._pack_latents(latents)
        unpacked = adapter._unpack_latents(packed, 8, 8)
        assert torch.allclose(unpacked, latents, atol=1e-6)

    def test_unpack_contiguous(self):
        adapter = Flux3Adapter()
        packed = torch.randn(2, 64, 128)
        unpacked = adapter._unpack_latents(packed, 8, 8)
        assert unpacked.is_contiguous()


# =============================================================================
# TestFlux3PrepareLatentIds
# =============================================================================


class TestFlux3PrepareLatentIds:
    """Tests for Flux3Adapter._prepare_latent_ids (unbatched, RoPE-scaled)."""

    def test_shape_unbatched(self):
        adapter = Flux3Adapter()
        ids = adapter._prepare_latent_ids(4, 6, device=torch.device("cpu"))
        assert ids.shape == (4 * 6, 4)

    def test_dtype_float32(self):
        adapter = Flux3Adapter()
        ids = adapter._prepare_latent_ids(4, 4, device=torch.device("cpu"))
        assert ids.dtype == torch.float32

    def test_t_and_l_are_zero(self):
        adapter = Flux3Adapter()
        ids = adapter._prepare_latent_ids(2, 3, device=torch.device("cpu"))
        assert (ids[:, 0] == 0).all()
        assert (ids[:, 3] == 0).all()

    def test_h_w_coords_scaled_by_half(self):
        adapter = Flux3Adapter()
        h_p, w_p = 3, 2
        ids = adapter._prepare_latent_ids(h_p, w_p, device=torch.device("cpu"))
        expected_h = torch.arange(h_p, dtype=torch.float32).repeat_interleave(w_p) * 0.5
        expected_w = torch.arange(w_p, dtype=torch.float32).repeat(h_p) * 0.5
        assert torch.equal(ids[:, 1], expected_h)
        assert torch.equal(ids[:, 2], expected_w)

    def test_matches_upstream_cartesian_prod(self):
        # Upstream reference: cartesian_prod(arange(1), arange(h), arange(w),
        # arange(1)).float() with H/W axes scaled by 0.5.
        adapter = Flux3Adapter()
        h_p, w_p = 4, 5
        expected = torch.cartesian_prod(torch.arange(1), torch.arange(h_p), torch.arange(w_p), torch.arange(1)).to(
            dtype=torch.float32
        )
        expected[:, 1:3] = expected[:, 1:3] * 0.5
        ids = adapter._prepare_latent_ids(h_p, w_p, device=torch.device("cpu"))
        assert torch.equal(ids, expected)


# =============================================================================
# TestFlux3PrepareTextIds
# =============================================================================


class TestFlux3PrepareTextIds:
    """Tests for Flux3Adapter._prepare_text_ids (unbatched, l-axis positions)."""

    def test_shape_unbatched(self):
        adapter = Flux3Adapter()
        ids = adapter._prepare_text_ids(seq_len=64, device=torch.device("cpu"))
        assert ids.shape == (64, 4)

    def test_dtype_float32(self):
        adapter = Flux3Adapter()
        ids = adapter._prepare_text_ids(seq_len=32, device=torch.device("cpu"))
        assert ids.dtype == torch.float32

    def test_t_h_w_are_zero_and_l_advances(self):
        adapter = Flux3Adapter()
        seq_len = 10
        ids = adapter._prepare_text_ids(seq_len=seq_len, device=torch.device("cpu"))
        assert (ids[:, 0] == 0).all()
        assert (ids[:, 1] == 0).all()
        assert (ids[:, 2] == 0).all()
        assert torch.equal(ids[:, 3], torch.arange(seq_len, dtype=torch.float32))


# =============================================================================
# TestFlux3PrepareInputs
# =============================================================================


class TestFlux3PrepareInputs:
    """Tests for Flux3Adapter.prepare_inputs."""

    def _make_context(self, noisy_latents, batch, **kwargs):
        defaults = dict(
            noisy_latents=noisy_latents,
            latents=torch.randn_like(noisy_latents),
            timesteps=torch.tensor([500.0, 500.0]),
            sigma=torch.tensor([0.5, 0.5]),
            task_type="t2i",
            data_type="image",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        defaults.update(kwargs)
        return FlowMatchingContext(**defaults)

    def test_output_keys(self):
        adapter = Flux3Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 64, 20480)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        for key in (
            "hidden_states",
            "img_ids",
            "timestep",
            "encoder_hidden_states",
            "txt_ids",
            "_h_p",
            "_w_p",
        ):
            assert key in inputs

    def test_no_guidance_or_pooled_keys(self):
        # FLUX.3 is not guidance-distilled and passes no pooled projections.
        adapter = Flux3Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 20480)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert "guidance" not in inputs
        assert "pooled_projections" not in inputs

    def test_3d_latent_raises(self):
        adapter = Flux3Adapter()
        noisy = torch.randn(2, 128, 64)
        batch = {"text_embeddings": torch.randn(2, 64, 20480)}
        ctx = self._make_context(noisy, batch)
        with pytest.raises(ValueError, match="Flux3Adapter expects 4D"):
            adapter.prepare_inputs(ctx)

    def test_image_cond_stream_from_context_latents(self):
        adapter = Flux3Adapter()
        b, c, h, w = 2, 128, 4, 4
        noisy = torch.randn(b, c, h, w)
        cond = torch.randn(b, c, 6, 8)
        batch = {"text_embeddings": torch.randn(b, 16, 20480), "context_latents": [cond]}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["image_cond_hidden_states"].shape == (b, 6 * 8, c)
        # cond ids: l = -1, H/W scaled by 0.5
        cond_ids = inputs["image_cond_ids"]
        assert cond_ids.shape == (6 * 8, 4)
        assert (cond_ids[:, 3] == -1.0).all()
        expected = torch.cartesian_prod(torch.arange(1), torch.arange(6), torch.arange(8), torch.tensor([-1]))
        expected = expected.to(dtype=torch.float32)
        expected[:, 1:3] = expected[:, 1:3] * 0.5
        assert torch.equal(cond_ids, expected)
        assert "image_cond_timestep" not in inputs

    def test_multiple_context_latents_raises(self):
        adapter = Flux3Adapter()
        noisy = torch.randn(1, 128, 4, 4)
        batch = {
            "text_embeddings": torch.randn(1, 16, 20480),
            "context_latents": [torch.randn(1, 128, 4, 4), torch.randn(1, 128, 4, 4)],
        }
        ctx = self._make_context(noisy, batch, timesteps=torch.tensor([500.0]), sigma=torch.tensor([0.5]))
        with pytest.raises(ValueError, match="exactly one conditioning image stream"):
            adapter.prepare_inputs(ctx)

    def test_image_hidden_states_shape(self):
        adapter = Flux3Adapter()
        b, c, h, w = 2, 128, 8, 8
        noisy = torch.randn(b, c, h, w)
        batch = {"text_embeddings": torch.randn(b, 64, 20480)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["hidden_states"].shape == (b, h * w, c)

    def test_ids_unbatched(self):
        adapter = Flux3Adapter()
        b, c, h_p, w_p, seq_len = 2, 128, 4, 6, 32
        noisy = torch.randn(b, c, h_p, w_p)
        batch = {"text_embeddings": torch.randn(b, seq_len, 20480)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["img_ids"].shape == (h_p * w_p, 4)
        assert inputs["txt_ids"].shape == (seq_len, 4)
        assert inputs["img_ids"].dtype == torch.float32
        assert inputs["txt_ids"].dtype == torch.float32

    def test_per_sample_timestep(self):
        adapter = Flux3Adapter()
        b, c, h_p, w_p = 2, 128, 4, 4
        noisy = torch.randn(b, c, h_p, w_p)
        batch = {"text_embeddings": torch.randn(b, 32, 20480)}
        sigma = torch.tensor([0.25, 0.75])
        ctx = self._make_context(noisy, batch, sigma=sigma, timesteps=sigma * 1000.0)
        inputs = adapter.prepare_inputs(ctx)
        timestep = inputs["timestep"]
        assert timestep.shape == (b,)
        assert torch.equal(timestep, sigma)

    def test_cfg_dropout_zeroing(self):
        adapter = Flux3Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 20480) + 1.0}
        ctx = self._make_context(noisy, batch, cfg_dropout_prob=1.0)
        inputs = adapter.prepare_inputs(ctx)
        assert (inputs["encoder_hidden_states"] == 0).all()

    def test_h_p_w_p_stored(self):
        adapter = Flux3Adapter()
        h_p, w_p = 6, 10
        noisy = torch.randn(1, 128, h_p, w_p)
        batch = {"text_embeddings": torch.randn(1, 32, 20480)}
        ctx = self._make_context(noisy, batch, timesteps=torch.tensor([500.0]), sigma=torch.tensor([0.5]))
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["_h_p"] == h_p
        assert inputs["_w_p"] == w_p


# =============================================================================
# TestFlux3Forward
# =============================================================================


class TestFlux3Forward:
    """Tests for Flux3Adapter.forward."""

    def _make_inputs(self, b, c, h_p, w_p, seq_len=32):
        return {
            "hidden_states": torch.randn(b, h_p * w_p, c),
            "img_ids": torch.zeros(h_p * w_p, 4),
            "timestep": torch.full((b,), 0.5),
            "encoder_hidden_states": torch.randn(b, seq_len, 20480),
            "txt_ids": torch.zeros(seq_len, 4),
            "_h_p": h_p,
            "_w_p": w_p,
        }

    def test_unpacked_output_shape(self):
        adapter = Flux3Adapter()
        b, c, h_p, w_p = 2, 128, 8, 8

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c),)

        pred = adapter.forward(mock_model, self._make_inputs(b, c, h_p, w_p))
        assert pred.shape == (b, c, h_p, w_p)
        assert pred.is_contiguous()

    def test_forward_calls_model_with_correct_kwargs(self):
        adapter = Flux3Adapter()
        b, c, h_p, w_p = 1, 128, 4, 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c),)

        adapter.forward(mock_model, self._make_inputs(b, c, h_p, w_p))
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        for key in ("hidden_states", "img_ids", "timestep", "encoder_hidden_states", "txt_ids"):
            assert key in call_kwargs
        assert call_kwargs["return_dict"] is False
        assert "guidance" not in call_kwargs

    def test_prediction_space_matches_target_space(self):
        # forward() must return prediction in the unpacked latent space so
        # MSE against (noise - latents) is computed elementwise-aligned.
        adapter = Flux3Adapter()
        b, c, h_p, w_p = 1, 8, 2, 3
        packed_pred = torch.randn(b, h_p * w_p, c)

        mock_model = MagicMock()
        mock_model.return_value = (packed_pred,)

        pred = adapter.forward(mock_model, self._make_inputs(b, c, h_p, w_p))
        assert torch.equal(pred, packed_pred.transpose(1, 2).reshape(b, c, h_p, w_p))


# =============================================================================
# TestFlux3AdapterRegistration
# =============================================================================


class TestFlux3AdapterRegistration:
    """Tests for the adapter factory registration."""

    def test_create_adapter_flux3(self):
        adapter = create_adapter("flux3")
        assert isinstance(adapter, Flux3Adapter)
