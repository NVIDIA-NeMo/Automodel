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
Unit tests for LTXAdapter.

LTXAdapter supports LTX-Video models with:
- 3D latent packing/unpacking ([B, C, F, H, W] <-> [B, S, D])
- encoder_attention_mask for text conditioning
- num_frames, height, width for RoPE positional embeddings

Tests cover:
- Pack/unpack roundtrip
- Input preparation
- Forward pass
- CFG dropout
- Shape handling
"""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters import (
    FlowMatchingContext,
    LTXAdapter,
)


class MockLTXModel(nn.Module):
    """Mock model that mimics LTXVideoTransformer3DModel forward interface."""

    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.last_inputs = {}

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        timestep,
        encoder_attention_mask,
        num_frames=None,
        height=None,
        width=None,
        return_dict=False,
    ):
        self.call_count += 1
        self.last_inputs = {
            "hidden_states": hidden_states,
            "encoder_hidden_states": encoder_hidden_states,
            "timestep": timestep,
            "encoder_attention_mask": encoder_attention_mask,
            "num_frames": num_frames,
            "height": height,
            "width": width,
            "return_dict": return_dict,
        }
        # Return prediction with same shape as input (packed)
        output = torch.randn_like(hidden_states)
        return (output,)


@pytest.fixture
def ltx_adapter():
    """Create an LTXAdapter instance with default patch sizes."""
    return LTXAdapter(patch_size=1, patch_size_t=1)


@pytest.fixture
def mock_model():
    """Create a mock LTX model."""
    return MockLTXModel()


@pytest.fixture
def sample_context():
    """Create a sample FlowMatchingContext with LTX-compatible batch."""
    batch = {
        "video_latents": torch.randn(2, 128, 4, 8, 8),
        "text_embeddings": torch.randn(2, 128, 4096),
        "text_mask": torch.ones(2, 128),
    }
    return FlowMatchingContext(
        noisy_latents=torch.randn(2, 128, 4, 8, 8),
        latents=batch["video_latents"],
        timesteps=torch.rand(2) * 1000,
        sigma=torch.rand(2),
        task_type="t2v",
        data_type="video",
        device=torch.device("cpu"),
        dtype=torch.float32,
        cfg_dropout_prob=0.0,
        batch=batch,
    )


class TestLTXAdapterInit:
    """Test LTXAdapter initialization."""

    def test_adapter_creation(self):
        adapter = LTXAdapter()
        assert adapter is not None

    def test_default_patch_sizes(self):
        adapter = LTXAdapter()
        assert adapter.patch_size == 1
        assert adapter.patch_size_t == 1

    def test_custom_patch_sizes(self):
        adapter = LTXAdapter(patch_size=2, patch_size_t=2)
        assert adapter.patch_size == 2
        assert adapter.patch_size_t == 2


class TestPackUnpack:
    """Test latent packing and unpacking roundtrip."""

    def test_pack_shape(self):
        latents = torch.randn(2, 128, 4, 8, 8)
        packed = LTXAdapter._pack_latents(latents, patch_size=1, patch_size_t=1)
        # [B, F*H*W, C] = [2, 4*8*8, 128] = [2, 256, 128]
        assert packed.shape == (2, 256, 128)

    def test_unpack_shape(self):
        packed = torch.randn(2, 256, 128)
        unpacked = LTXAdapter._unpack_latents(packed, num_frames=4, height=8, width=8)
        assert unpacked.shape == (2, 128, 4, 8, 8)

    def test_roundtrip_identity(self):
        latents = torch.randn(2, 128, 4, 8, 8)
        packed = LTXAdapter._pack_latents(latents, 1, 1)
        unpacked = LTXAdapter._unpack_latents(packed, 4, 8, 8, 1, 1)
        assert torch.allclose(latents, unpacked)

    def test_roundtrip_various_shapes(self):
        shapes = [
            (1, 128, 2, 4, 4),
            (2, 128, 4, 8, 8),
            (4, 128, 8, 16, 16),
            (1, 128, 1, 8, 12),
        ]
        for shape in shapes:
            latents = torch.randn(shape)
            packed = LTXAdapter._pack_latents(latents, 1, 1)
            unpacked = LTXAdapter._unpack_latents(packed, shape[2], shape[3], shape[4], 1, 1)
            assert torch.allclose(latents, unpacked), f"Roundtrip failed for shape {shape}"

    def test_pack_with_patch_size_2(self):
        latents = torch.randn(2, 128, 4, 8, 8)
        packed = LTXAdapter._pack_latents(latents, patch_size=2, patch_size_t=2)
        # [B, F//2 * H//2 * W//2, C * 2 * 2 * 2] = [2, 2*4*4, 128*8] = [2, 32, 1024]
        assert packed.shape == (2, 32, 1024)

    def test_roundtrip_with_patch_size_2(self):
        latents = torch.randn(2, 128, 4, 8, 8)
        packed = LTXAdapter._pack_latents(latents, 2, 2)
        unpacked = LTXAdapter._unpack_latents(packed, 2, 4, 4, 2, 2)
        assert torch.allclose(latents, unpacked)


class TestLTXAdapterPrepareInputs:
    """Test LTXAdapter.prepare_inputs method."""

    def test_prepare_inputs_keys(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        expected_keys = {
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
            "num_frames",
            "height",
            "width",
            "_original_shape",
        }
        assert set(inputs.keys()) == expected_keys

    def test_hidden_states_is_packed(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        # [2, 128, 4, 8, 8] packed to [2, 4*8*8, 128] = [2, 256, 128]
        assert inputs["hidden_states"].shape == (2, 256, 128)

    def test_encoder_hidden_states_shape(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        assert inputs["encoder_hidden_states"].shape == (2, 128, 4096)

    def test_encoder_attention_mask_shape(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        assert inputs["encoder_attention_mask"].shape == (2, 128)

    def test_timestep_dtype(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        assert inputs["timestep"].dtype == sample_context.dtype

    def test_rope_dimensions(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        assert inputs["num_frames"] == 4
        assert inputs["height"] == 8
        assert inputs["width"] == 8

    def test_original_shape_stored(self, ltx_adapter, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        assert inputs["_original_shape"] == (2, 128, 4, 8, 8)

    def test_2d_text_embeddings_unsqueezed(self, ltx_adapter):
        batch = {
            "text_embeddings": torch.randn(128, 4096),  # 2D
            "text_mask": torch.ones(128),
        }
        context = FlowMatchingContext(
            noisy_latents=torch.randn(1, 128, 4, 8, 8),
            latents=torch.randn(1, 128, 4, 8, 8),
            timesteps=torch.rand(1) * 1000,
            sigma=torch.rand(1),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        inputs = ltx_adapter.prepare_inputs(context)
        assert inputs["encoder_hidden_states"].ndim == 3

    def test_missing_text_mask_creates_ones(self, ltx_adapter):
        batch = {
            "text_embeddings": torch.randn(2, 128, 4096),
            # No text_mask
        }
        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 128, 4, 8, 8),
            latents=torch.randn(2, 128, 4, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        inputs = ltx_adapter.prepare_inputs(context)
        assert inputs["encoder_attention_mask"].shape == (2, 128)
        assert (inputs["encoder_attention_mask"] == 1.0).all()

    def test_cfg_dropout_zeros_embeddings(self, ltx_adapter):
        batch = {
            "text_embeddings": torch.randn(2, 128, 4096),
            "text_mask": torch.ones(2, 128),
        }
        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 128, 4, 8, 8),
            latents=torch.randn(2, 128, 4, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=1.0,  # Always drop
            batch=batch,
        )
        inputs = ltx_adapter.prepare_inputs(context)
        assert (inputs["encoder_hidden_states"] == 0).all()
        assert (inputs["encoder_attention_mask"] == 0).all()

    def test_different_batch_sizes(self, ltx_adapter):
        for batch_size in [1, 2, 4]:
            batch = {
                "text_embeddings": torch.randn(batch_size, 128, 4096),
                "text_mask": torch.ones(batch_size, 128),
            }
            context = FlowMatchingContext(
                noisy_latents=torch.randn(batch_size, 128, 4, 8, 8),
                latents=torch.randn(batch_size, 128, 4, 8, 8),
                timesteps=torch.rand(batch_size) * 1000,
                sigma=torch.rand(batch_size),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )
            inputs = ltx_adapter.prepare_inputs(context)
            assert inputs["hidden_states"].shape[0] == batch_size

    def test_different_dtypes(self, ltx_adapter):
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            batch = {
                "text_embeddings": torch.randn(2, 128, 4096),
                "text_mask": torch.ones(2, 128),
            }
            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 128, 4, 8, 8),
                latents=torch.randn(2, 128, 4, 8, 8),
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=dtype,
                cfg_dropout_prob=0.0,
                batch=batch,
            )
            inputs = ltx_adapter.prepare_inputs(context)
            assert inputs["timestep"].dtype == dtype
            assert inputs["encoder_hidden_states"].dtype == dtype

    def test_rejects_non_5d_latents(self, ltx_adapter):
        batch = {
            "text_embeddings": torch.randn(2, 128, 4096),
        }
        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 128, 8, 8),  # 4D
            latents=torch.randn(2, 128, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        with pytest.raises(ValueError, match="5D"):
            ltx_adapter.prepare_inputs(context)


class TestLTXAdapterForward:
    """Test LTXAdapter.forward method."""

    def test_forward_output_shape_matches_input_latents(self, ltx_adapter, mock_model, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        output = ltx_adapter.forward(mock_model, inputs)
        # Output should be unpacked back to [B, C, F, H, W]
        assert output.shape == sample_context.noisy_latents.shape

    def test_forward_calls_model_correctly(self, ltx_adapter, mock_model, sample_context):
        inputs = ltx_adapter.prepare_inputs(sample_context)
        ltx_adapter.forward(mock_model, inputs)

        assert mock_model.call_count == 1
        assert mock_model.last_inputs["return_dict"] is False
        assert mock_model.last_inputs["num_frames"] == 4
        assert mock_model.last_inputs["height"] == 8
        assert mock_model.last_inputs["width"] == 8

    def test_forward_various_shapes(self, ltx_adapter, mock_model):
        shapes = [
            (1, 128, 2, 4, 4),
            (2, 128, 4, 8, 8),
            (4, 128, 8, 16, 16),
        ]
        for shape in shapes:
            batch = {
                "text_embeddings": torch.randn(shape[0], 128, 4096),
                "text_mask": torch.ones(shape[0], 128),
            }
            context = FlowMatchingContext(
                noisy_latents=torch.randn(shape),
                latents=torch.randn(shape),
                timesteps=torch.rand(shape[0]) * 1000,
                sigma=torch.rand(shape[0]),
                task_type="t2v",
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )
            inputs = ltx_adapter.prepare_inputs(context)
            output = ltx_adapter.forward(mock_model, inputs)
            assert output.shape == shape, f"Shape mismatch for input {shape}"

    def test_forward_with_tuple_output(self, ltx_adapter, sample_context):
        class TupleOutputModel(nn.Module):
            def forward(
                self,
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask,
                num_frames=None,
                height=None,
                width=None,
                return_dict=False,
            ):
                return (torch.randn_like(hidden_states), "extra", {"k": "v"})

        model = TupleOutputModel()
        inputs = ltx_adapter.prepare_inputs(sample_context)
        output = ltx_adapter.forward(model, inputs)
        assert output.shape == sample_context.noisy_latents.shape

    def test_forward_with_tensor_output(self, ltx_adapter, sample_context):
        class TensorOutputModel(nn.Module):
            def forward(
                self,
                hidden_states,
                encoder_hidden_states,
                timestep,
                encoder_attention_mask,
                num_frames=None,
                height=None,
                width=None,
                return_dict=False,
            ):
                return torch.randn_like(hidden_states)

        model = TensorOutputModel()
        inputs = ltx_adapter.prepare_inputs(sample_context)
        output = ltx_adapter.forward(model, inputs)
        assert output.shape == sample_context.noisy_latents.shape


class TestLTXAdapterEndToEnd:
    """End-to-end tests for LTXAdapter."""

    def test_full_workflow(self, ltx_adapter, mock_model):
        batch = {
            "text_embeddings": torch.randn(2, 128, 4096),
            "text_mask": torch.ones(2, 128),
        }
        context = FlowMatchingContext(
            noisy_latents=torch.randn(2, 128, 4, 8, 8),
            latents=torch.randn(2, 128, 4, 8, 8),
            timesteps=torch.rand(2) * 1000,
            sigma=torch.rand(2),
            task_type="t2v",
            data_type="video",
            device=torch.device("cpu"),
            dtype=torch.float32,
            cfg_dropout_prob=0.0,
            batch=batch,
        )
        inputs = ltx_adapter.prepare_inputs(context)
        output = ltx_adapter.forward(mock_model, inputs)

        assert output.shape == context.noisy_latents.shape
        assert torch.isfinite(output).all()

    def test_with_different_task_types(self, ltx_adapter, mock_model):
        for task_type in ["t2v", "i2v"]:
            batch = {
                "text_embeddings": torch.randn(2, 128, 4096),
                "text_mask": torch.ones(2, 128),
            }
            context = FlowMatchingContext(
                noisy_latents=torch.randn(2, 128, 4, 8, 8),
                latents=torch.randn(2, 128, 4, 8, 8),
                timesteps=torch.rand(2) * 1000,
                sigma=torch.rand(2),
                task_type=task_type,
                data_type="video",
                device=torch.device("cpu"),
                dtype=torch.float32,
                cfg_dropout_prob=0.0,
                batch=batch,
            )
            inputs = ltx_adapter.prepare_inputs(context)
            output = ltx_adapter.forward(mock_model, inputs)
            assert output.shape == context.noisy_latents.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
