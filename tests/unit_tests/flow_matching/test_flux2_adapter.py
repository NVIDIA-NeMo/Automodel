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

"""Unit tests for Flux2Adapter (flux2.py): pack/unpack latents, positional IDs, prepare_inputs, forward."""

from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.flux2 import Flux2Adapter

# =============================================================================
# TestFlux2AdapterPackLatents
# =============================================================================


class TestFlux2AdapterPackLatents:
    """Tests for Flux2Adapter._pack_latents."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 128, 8, 8),
            (2, 128, 16, 16),
            (4, 128, 8, 16),
        ],
    )
    def test_pack_shape(self, b, c, h, w):
        adapter = Flux2Adapter()
        latents = torch.randn(b, c, h, w)
        packed = adapter._pack_latents(latents)
        assert packed.shape == (b, h * w, c)

    def test_pack_contiguous(self):
        adapter = Flux2Adapter()
        latents = torch.randn(2, 128, 8, 8)
        packed = adapter._pack_latents(latents)
        assert packed.is_contiguous()

    def test_pack_values_deterministic(self):
        adapter = Flux2Adapter()
        torch.manual_seed(42)
        latents = torch.randn(1, 128, 4, 4)
        packed = adapter._pack_latents(latents)
        assert torch.isfinite(packed).all()
        assert packed.shape == (1, 16, 128)


# =============================================================================
# TestFlux2AdapterUnpackLatents
# =============================================================================


class TestFlux2AdapterUnpackLatents:
    """Tests for Flux2Adapter._unpack_latents."""

    @pytest.mark.parametrize(
        "b, c, h, w",
        [
            (1, 128, 8, 8),
            (2, 128, 16, 16),
            (4, 128, 8, 16),
        ],
    )
    def test_unpack_shape(self, b, c, h, w):
        packed = torch.randn(b, h * w, c)
        adapter = Flux2Adapter()
        unpacked = adapter._unpack_latents(packed, h, w)
        assert unpacked.shape == (b, c, h, w)

    def test_unpack_contiguous(self):
        adapter = Flux2Adapter()
        packed = torch.randn(2, 64, 128)
        unpacked = adapter._unpack_latents(packed, 8, 8)
        assert unpacked.is_contiguous()

    def test_pack_unpack_roundtrip(self):
        adapter = Flux2Adapter()
        latents = torch.randn(2, 128, 8, 8)
        packed = adapter._pack_latents(latents)
        unpacked = adapter._unpack_latents(packed, 8, 8)
        assert torch.allclose(unpacked, latents, atol=1e-6)


# =============================================================================
# TestPrepareLatentIds
# =============================================================================


class TestPrepareLatentIds:
    """Tests for Flux2Adapter._prepare_latent_ids."""

    def test_shape(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_latent_ids(4, 6, batch_size=2, device=torch.device("cpu"))
        assert ids.shape == (2, 4 * 6, 4)

    def test_dtype_long(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_latent_ids(4, 4, batch_size=1, device=torch.device("cpu"))
        assert ids.dtype == torch.long

    def test_t_and_l_are_zero(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_latent_ids(2, 3, batch_size=1, device=torch.device("cpu"))
        # T (dim 0) and L (dim 3) should be zero
        assert (ids[:, :, 0] == 0).all()
        assert (ids[:, :, 3] == 0).all()

    def test_h_coords(self):
        adapter = Flux2Adapter()
        h_p, w_p = 3, 2
        ids = adapter._prepare_latent_ids(h_p, w_p, batch_size=1, device=torch.device("cpu"))
        # ids[0, :, 1] should be [0,0, 1,1, 2,2] (repeat_interleave)
        expected_h = torch.arange(h_p).repeat_interleave(w_p)
        assert torch.equal(ids[0, :, 1], expected_h)

    def test_w_coords(self):
        adapter = Flux2Adapter()
        h_p, w_p = 3, 2
        ids = adapter._prepare_latent_ids(h_p, w_p, batch_size=1, device=torch.device("cpu"))
        # ids[0, :, 2] should be [0,1, 0,1, 0,1] (repeat)
        expected_w = torch.arange(w_p).repeat(h_p)
        assert torch.equal(ids[0, :, 2], expected_w)

    def test_batch_expand(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_latent_ids(4, 4, batch_size=3, device=torch.device("cpu"))
        # All batch items should be identical
        assert ids.shape[0] == 3
        assert torch.equal(ids[0], ids[1])
        assert torch.equal(ids[1], ids[2])


# =============================================================================
# TestPrepareTextIds
# =============================================================================


class TestPrepareTextIds:
    """Tests for Flux2Adapter._prepare_text_ids."""

    def test_shape(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_text_ids(seq_len=64, batch_size=2, device=torch.device("cpu"))
        assert ids.shape == (2, 64, 4)

    def test_dtype_long(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_text_ids(seq_len=32, batch_size=1, device=torch.device("cpu"))
        assert ids.dtype == torch.long

    def test_t_h_w_are_zero(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_text_ids(seq_len=8, batch_size=1, device=torch.device("cpu"))
        assert (ids[:, :, 0] == 0).all()
        assert (ids[:, :, 1] == 0).all()
        assert (ids[:, :, 2] == 0).all()

    def test_position_indices(self):
        adapter = Flux2Adapter()
        seq_len = 10
        ids = adapter._prepare_text_ids(seq_len=seq_len, batch_size=1, device=torch.device("cpu"))
        expected = torch.arange(seq_len)
        assert torch.equal(ids[0, :, 3], expected)

    def test_batch_expand(self):
        adapter = Flux2Adapter()
        ids = adapter._prepare_text_ids(seq_len=16, batch_size=3, device=torch.device("cpu"))
        assert ids.shape[0] == 3
        assert torch.equal(ids[0], ids[1])


# =============================================================================
# TestFlux2PrepareInputs
# =============================================================================


class TestFlux2PrepareInputs:
    """Tests for Flux2Adapter.prepare_inputs."""

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
        adapter = Flux2Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 64, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        for key in ("hidden_states", "encoder_hidden_states", "timestep", "img_ids", "txt_ids", "guidance", "_h_p", "_w_p"):
            assert key in inputs

    def test_5d_latent_raises(self):
        adapter = Flux2Adapter()
        noisy = torch.randn(2, 128, 4, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 64, 15360)}
        ctx = self._make_context(noisy, batch, timesteps=torch.tensor([500.0, 500.0]), sigma=torch.tensor([0.5, 0.5]))
        with pytest.raises(ValueError, match="Flux2Adapter expects 4D"):
            adapter.prepare_inputs(ctx)

    def test_hidden_states_shape(self):
        adapter = Flux2Adapter()
        b, c, h, w = 2, 128, 8, 8
        noisy = torch.randn(b, c, h, w)
        batch = {"text_embeddings": torch.randn(b, 64, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["hidden_states"].shape == (b, h * w, c)

    def test_img_ids_shape_and_dtype(self):
        adapter = Flux2Adapter()
        b, c, h_p, w_p = 2, 128, 4, 6
        noisy = torch.randn(b, c, h_p, w_p)
        batch = {"text_embeddings": torch.randn(b, 32, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["img_ids"].shape == (b, h_p * w_p, 4)
        assert inputs["img_ids"].dtype == torch.long

    def test_txt_ids_shape_and_dtype(self):
        adapter = Flux2Adapter()
        b, seq_len = 2, 64
        noisy = torch.randn(b, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(b, seq_len, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["txt_ids"].shape == (b, seq_len, 4)
        assert inputs["txt_ids"].dtype == torch.long

    def test_timestep_normalization(self):
        adapter = Flux2Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 15360)}
        timesteps = torch.tensor([500.0, 1000.0])
        ctx = self._make_context(noisy, batch, timesteps=timesteps, sigma=torch.tensor([0.5, 1.0]))
        inputs = adapter.prepare_inputs(ctx)
        expected = timesteps / 1000.0
        assert torch.allclose(inputs["timestep"], expected)

    def test_cfg_dropout_zeroing(self):
        adapter = Flux2Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 15360) + 1.0}
        ctx = self._make_context(noisy, batch, cfg_dropout_prob=1.0)
        inputs = adapter.prepare_inputs(ctx)
        assert (inputs["encoder_hidden_states"] == 0).all()

    def test_no_pooled_projections_key(self):
        # Flux2 has no pooled_projections — ensure it is not in the returned dict
        adapter = Flux2Adapter()
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert "pooled_projections" not in inputs

    def test_guidance_embedding_present(self):
        adapter = Flux2Adapter(guidance_scale=7.0)
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["guidance"].shape == (2,)
        assert torch.allclose(inputs["guidance"], torch.tensor([7.0, 7.0]))

    def test_guidance_none_when_disabled(self):
        adapter = Flux2Adapter(use_guidance_embeds=False)
        noisy = torch.randn(2, 128, 8, 8)
        batch = {"text_embeddings": torch.randn(2, 32, 15360)}
        ctx = self._make_context(noisy, batch)
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["guidance"] is None

    def test_h_p_w_p_stored(self):
        adapter = Flux2Adapter()
        h_p, w_p = 6, 10
        noisy = torch.randn(1, 128, h_p, w_p)
        batch = {"text_embeddings": torch.randn(1, 32, 15360)}
        ctx = self._make_context(
            noisy, batch, timesteps=torch.tensor([500.0]), sigma=torch.tensor([0.5])
        )
        inputs = adapter.prepare_inputs(ctx)
        assert inputs["_h_p"] == h_p
        assert inputs["_w_p"] == w_p


# =============================================================================
# TestFlux2Forward
# =============================================================================


class TestFlux2Forward:
    """Tests for Flux2Adapter.forward."""

    def test_unpacked_output_shape(self):
        adapter = Flux2Adapter()
        b, c, h_p, w_p = 2, 128, 8, 8

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c),)

        inputs = {
            "hidden_states": torch.randn(b, h_p * w_p, c),
            "encoder_hidden_states": torch.randn(b, 64, 15360),
            "timestep": torch.tensor([0.5, 0.5]),
            "img_ids": torch.zeros(b, h_p * w_p, 4, dtype=torch.long),
            "txt_ids": torch.zeros(b, 64, 4, dtype=torch.long),
            "guidance": torch.tensor([3.5, 3.5]),
            "_h_p": h_p,
            "_w_p": w_p,
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h_p, w_p)

    def test_tuple_output_handling(self):
        adapter = Flux2Adapter()
        b, c, h_p, w_p = 1, 128, 4, 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c), "extra")

        inputs = {
            "hidden_states": torch.randn(b, h_p * w_p, c),
            "encoder_hidden_states": torch.randn(b, 32, 15360),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(b, h_p * w_p, 4, dtype=torch.long),
            "txt_ids": torch.zeros(b, 32, 4, dtype=torch.long),
            "guidance": torch.tensor([3.5]),
            "_h_p": h_p,
            "_w_p": w_p,
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.shape == (b, c, h_p, w_p)

    def test_forward_calls_model_with_correct_kwargs(self):
        adapter = Flux2Adapter()
        b, c, h_p, w_p = 1, 128, 4, 4

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c),)

        inputs = {
            "hidden_states": torch.randn(b, h_p * w_p, c),
            "encoder_hidden_states": torch.randn(b, 32, 15360),
            "timestep": torch.tensor([0.5]),
            "img_ids": torch.zeros(b, h_p * w_p, 4, dtype=torch.long),
            "txt_ids": torch.zeros(b, 32, 4, dtype=torch.long),
            "guidance": torch.tensor([3.5]),
            "_h_p": h_p,
            "_w_p": w_p,
        }

        adapter.forward(mock_model, inputs)
        mock_model.assert_called_once()
        call_kwargs = mock_model.call_args[1]
        assert "hidden_states" in call_kwargs
        assert "encoder_hidden_states" in call_kwargs
        assert "return_dict" in call_kwargs
        assert call_kwargs["return_dict"] is False

    def test_forward_output_contiguous(self):
        adapter = Flux2Adapter()
        b, c, h_p, w_p = 2, 128, 4, 8

        mock_model = MagicMock()
        mock_model.return_value = (torch.randn(b, h_p * w_p, c),)

        inputs = {
            "hidden_states": torch.randn(b, h_p * w_p, c),
            "encoder_hidden_states": torch.randn(b, 32, 15360),
            "timestep": torch.tensor([0.5, 0.5]),
            "img_ids": torch.zeros(b, h_p * w_p, 4, dtype=torch.long),
            "txt_ids": torch.zeros(b, 32, 4, dtype=torch.long),
            "guidance": torch.tensor([3.5, 3.5]),
            "_h_p": h_p,
            "_w_p": w_p,
        }

        pred = adapter.forward(mock_model, inputs)
        assert pred.is_contiguous()
