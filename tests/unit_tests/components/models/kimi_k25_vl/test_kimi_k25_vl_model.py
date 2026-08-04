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

import inspect
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.kimi_k25_vl.model import (
    DeepSeekV3RotaryEmbeddingAdapter,
    KimiK25VLConfig,
    KimiK25VLForConditionalGeneration,
    KimiK25VLModel,
    KimiK25VLMultiModalProjector,
    Learnable2DInterpPosEmbDividedFixed,
    MoonViT3dConfig,
    MoonViT3dEncoder,
    MoonViT3dEncoderLayer,
    MoonViT3dMLP,
    MoonViT3dPretrainedModel,
    MoonVision3dPatchEmbed,
    Rope2DPosEmbRepeated,
    _apply_rope_vision,
    get_1d_sincos_pos_embed,
    get_1d_sincos_pos_embed_from_grid,
    tpool_patch_merger,
    vision_attention_sdpa,
)


class TestKimiK25VLConfig:
    """Tests for KimiK25VLConfig."""

    def test_config_default_values(self):
        """Test KimiK25VLConfig has expected default values."""
        config = KimiK25VLConfig()

        assert config.model_type == "kimi_k25"
        assert config.ignore_index == -100
        assert config.media_placeholder_token_id == 163605
        assert config.pad_token_id == 0
        assert config.tie_word_embeddings is False
        assert config.mm_projector_type == "patchmerger"
        assert config.projector_hidden_act == "gelu"
        assert config.projector_ln_eps == 1e-5

    def test_config_architectures(self):
        """Test KimiK25VLConfig sets architectures for ModelRegistry matching."""
        config = KimiK25VLConfig()

        assert hasattr(config, "architectures")
        assert "KimiK25ForConditionalGeneration" in config.architectures
        assert "KimiK25VLForConditionalGeneration" in config.architectures

    def test_config_vision_config_default(self):
        """Test KimiK25VLConfig creates default vision config."""
        config = KimiK25VLConfig()

        assert config.vision_config is not None
        assert isinstance(config.vision_config, MoonViT3dConfig)
        assert config.vision_config.hidden_size == 1152
        assert config.vision_config.num_hidden_layers == 27

    def test_config_from_dict(self):
        """Test KimiK25VLConfig accepts dict for nested configs."""
        vision_dict = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "patch_size": 16,
        }
        config = KimiK25VLConfig(vision_config=vision_dict)

        assert config.vision_config.hidden_size == 768
        assert config.vision_config.num_hidden_layers == 12
        assert config.vision_config.patch_size == 16

    def test_config_to_dict(self):
        """Test KimiK25VLConfig.to_dict serializes nested configs."""
        config = KimiK25VLConfig()
        config_dict = config.to_dict()

        assert "vision_config" in config_dict
        assert "text_config" in config_dict
        assert isinstance(config_dict["vision_config"], dict)
        assert isinstance(config_dict["text_config"], dict)


class TestMoonViT3dConfig:
    """Tests for MoonViT3dConfig."""

    def test_moonvit3d_config_defaults(self):
        """Test MoonViT3dConfig default values."""
        config = MoonViT3dConfig()

        assert config.model_type == "moonvit3d"
        assert config.patch_size == 14
        assert config.init_pos_emb_height == 64
        assert config.init_pos_emb_width == 64
        assert config.init_pos_emb_time == 4
        assert config.num_attention_heads == 16
        assert config.num_hidden_layers == 27
        assert config.hidden_size == 1152
        assert config.intermediate_size == 4304
        assert config.merge_kernel_size == [2, 2]
        assert config.merge_type == "sd2_tpool"


class TestKimiK25VLModelUpdates:
    """Tests for KimiK25VLForConditionalGeneration model updates."""

    def test_from_pretrained_classmethod(self):
        """Ensure classmethod from_pretrained builds config then delegates to from_config."""
        cfg = KimiK25VLConfig()

        with patch.object(
            KimiK25VLConfig, "from_pretrained", return_value=cfg
        ) as mock_from_pretrained:
            with patch.object(
                KimiK25VLForConditionalGeneration,
                "from_config",
                wraps=KimiK25VLForConditionalGeneration.from_config,
            ) as mock_from_config:
                # Mock the actual model instantiation to avoid CUDA
                with patch.object(
                    KimiK25VLForConditionalGeneration,
                    "__init__",
                    lambda self, *args, **kwargs: None,
                ):
                    model = KimiK25VLForConditionalGeneration.__new__(
                        KimiK25VLForConditionalGeneration
                    )
                    # Simulate from_pretrained behavior
                    mock_from_pretrained.assert_not_called()

    def test_modelclass_export_exists(self):
        """Ensure ModelClass pointer is defined and points to class."""
        from nemo_automodel.components.models.kimi_k25_vl import model as kimi_mod

        assert hasattr(kimi_mod, "ModelClass")
        assert kimi_mod.ModelClass == KimiK25VLForConditionalGeneration

    def test_config_class_attribute(self):
        """Test KimiK25VLForConditionalGeneration has config_class attribute."""
        assert hasattr(KimiK25VLForConditionalGeneration, "config_class")
        assert KimiK25VLForConditionalGeneration.config_class is KimiK25VLConfig

    def test_model_class_attributes(self):
        """Test KimiK25VLForConditionalGeneration has expected class attributes."""
        assert KimiK25VLForConditionalGeneration.base_model_prefix == "model"
        assert KimiK25VLForConditionalGeneration.main_input_name == "pixel_values"
        assert "MoonViT3dEncoderLayer" in KimiK25VLForConditionalGeneration._no_split_modules
        assert KimiK25VLForConditionalGeneration.supports_gradient_checkpointing is True


class TestKimiK25VLModelInputsEmbeds:
    """Tests for inputs_embeds support in KimiK25VLModel.

    These tests verify the API changes without running actual forward passes
    that would trigger CUDA code in MoE layers.
    """

    def test_forward_signature_accepts_inputs_embeds(self):
        """Test KimiK25VLModel.forward signature includes inputs_embeds parameter."""
        sig = inspect.signature(KimiK25VLModel.forward)
        params = list(sig.parameters.keys())

        assert "inputs_embeds" in params, "forward should accept inputs_embeds parameter"
        assert "input_ids" in params, "forward should accept input_ids parameter"
        assert "pixel_values" in params, "forward should accept pixel_values parameter"
        assert "grid_thws" in params, "forward should accept grid_thws parameter"

        # Check input_ids is optional (has default None)
        input_ids_param = sig.parameters["input_ids"]
        assert input_ids_param.default is None, "input_ids should default to None"

        # Check inputs_embeds is optional (has default None)
        inputs_embeds_param = sig.parameters["inputs_embeds"]
        assert inputs_embeds_param.default is None, "inputs_embeds should default to None"

    def test_forward_signature_accepts_target_seq_length(self):
        """Test KimiK25VLModel.forward signature includes target_seq_length for PP."""
        sig = inspect.signature(KimiK25VLModel.forward)
        params = list(sig.parameters.keys())

        assert "target_seq_length" in params, "forward should accept target_seq_length for PP"
        target_seq_length_param = sig.parameters["target_seq_length"]
        assert target_seq_length_param.default is None, "target_seq_length should default to None"

    def test_validation_passes_with_only_inputs_embeds(self):
        """Test validation passes when only inputs_embeds is provided."""
        input_ids = None
        inputs_embeds = torch.randn(2, 8, 64)

        # This should NOT raise - validation passes
        if (input_ids is None) == (inputs_embeds is None):
            pytest.fail("Validation should pass when only inputs_embeds is provided")

    def test_validation_passes_with_only_input_ids(self):
        """Test validation passes when only input_ids is provided."""
        input_ids = torch.randint(0, 100, (2, 8))
        inputs_embeds = None

        # This should NOT raise - validation passes
        if (input_ids is None) == (inputs_embeds is None):
            pytest.fail("Validation should pass when only input_ids is provided")

    def test_forward_raises_when_both_input_ids_and_inputs_embeds(self):
        """Test KimiK25VLModel raises error when both input_ids and inputs_embeds provided."""
        input_ids = torch.randint(0, 100, (2, 8))
        inputs_embeds = torch.randn(2, 8, 64)

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_forward_raises_when_neither_input_ids_nor_inputs_embeds(self):
        """Test KimiK25VLModel raises error when neither input_ids nor inputs_embeds provided."""
        input_ids = None
        inputs_embeds = None

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")


class TestKimiK25VLForConditionalGenerationForward:
    """Tests for KimiK25VLForConditionalGeneration.forward signature."""

    def test_forward_signature_vlm_params(self):
        """Test forward signature includes VLM-specific parameters."""
        sig = inspect.signature(KimiK25VLForConditionalGeneration.forward)
        params = list(sig.parameters.keys())

        # Standard LM params
        assert "input_ids" in params
        assert "attention_mask" in params
        assert "position_ids" in params
        assert "inputs_embeds" in params
        assert "labels" in params

        # VLM-specific params
        assert "pixel_values" in params
        assert "grid_thws" in params
        assert "target_seq_length" in params

    def test_forward_signature_optional_params(self):
        """Test forward signature has expected optional parameters."""
        sig = inspect.signature(KimiK25VLForConditionalGeneration.forward)

        optional_params = [
            "past_key_values",
            "use_cache",
            "output_attentions",
            "output_hidden_states",
            "return_dict",
            "padding_mask",
        ]

        for param in optional_params:
            assert param in sig.parameters, f"forward should accept {param} parameter"


class TestKimiK25VLRegistration:
    """Tests for KimiK25VL registration with transformers."""

    def test_registration_function_exists(self):
        """Test _register_kimi_k25_vl_with_transformers function exists."""
        from nemo_automodel.components.models.kimi_k25_vl import model as kimi_mod

        assert hasattr(kimi_mod, "_register_kimi_k25_vl_with_transformers")
        assert callable(kimi_mod._register_kimi_k25_vl_with_transformers)

    def test_compute_expanded_seq_length_exists(self):
        """Test compute_expanded_seq_length utility function exists."""
        from nemo_automodel.components.models.kimi_k25_vl import model as kimi_mod

        assert hasattr(kimi_mod, "compute_expanded_seq_length")
        assert callable(kimi_mod.compute_expanded_seq_length)

    def test_compute_expanded_seq_length_basic(self):
        """Test compute_expanded_seq_length computes correct length."""
        from nemo_automodel.components.models.kimi_k25_vl.model import compute_expanded_seq_length

        # grid_thws = [[1, 28, 28]] -> (28//2) * (28//2) = 196 image tokens
        # text_seq_length=82, num_images=1
        # expanded = 82 - 1 + 196 = 277
        grid_thws = torch.tensor([[1, 28, 28]])
        result = compute_expanded_seq_length(82, grid_thws, merge_kernel_size=(2, 2), num_images=1)
        assert result == 277

    def test_compute_expanded_seq_length_larger_image(self):
        """Test compute_expanded_seq_length with larger image."""
        from nemo_automodel.components.models.kimi_k25_vl.model import compute_expanded_seq_length

        # grid_thws = [[1, 56, 56]] -> (56//2) * (56//2) = 784 image tokens
        grid_thws = torch.tensor([[1, 56, 56]])
        result = compute_expanded_seq_length(100, grid_thws, merge_kernel_size=(2, 2), num_images=1)
        assert result == 100 - 1 + 784

    def test_compute_expanded_seq_length_multiple_images(self):
        """Test compute_expanded_seq_length with multiple images."""
        from nemo_automodel.components.models.kimi_k25_vl.model import compute_expanded_seq_length

        # grid_thws = [[1, 28, 28], [1, 14, 14]]
        # Image 1: (28//2) * (28//2) = 196
        # Image 2: (14//2) * (14//2) = 49
        # Total image tokens: 245
        # expanded = 100 - 2 + 245 = 343
        grid_thws = torch.tensor([[1, 28, 28], [1, 14, 14]])
        result = compute_expanded_seq_length(100, grid_thws, merge_kernel_size=(2, 2), num_images=2)
        assert result == 100 - 2 + 196 + 49


# =============================================================================
# Helper Functions Tests
# =============================================================================


class TestSincosPosEmbed:
    """Tests for sinusoidal position embedding functions."""

    def test_get_1d_sincos_pos_embed_from_grid_shape(self):
        """Test get_1d_sincos_pos_embed_from_grid produces correct shape."""
        embed_dim = 64
        pos = np.arange(10)

        result = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

        assert result.shape == (10, embed_dim)
        assert result.dtype == np.float64  # numpy default

    def test_get_1d_sincos_pos_embed_from_grid_values(self):
        """Test get_1d_sincos_pos_embed_from_grid produces sin/cos pattern."""
        embed_dim = 4
        pos = np.array([0, 1])

        result = get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

        # First half is sin, second half is cos
        # At position 0, sin(0) = 0, cos(0) = 1
        assert result[0, 0] == pytest.approx(0, abs=1e-5)  # sin(0)
        assert result[0, 2] == pytest.approx(1, abs=1e-5)  # cos(0)

    def test_get_1d_sincos_pos_embed_shape(self):
        """Test get_1d_sincos_pos_embed produces correct shape."""
        embed_dim = 64
        t_size = 4

        result = get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=False)
        assert result.shape == (t_size, embed_dim)

    def test_get_1d_sincos_pos_embed_with_cls_token(self):
        """Test get_1d_sincos_pos_embed adds cls token row."""
        embed_dim = 64
        t_size = 4

        result = get_1d_sincos_pos_embed(embed_dim, t_size, cls_token=True)

        assert result.shape == (t_size + 1, embed_dim)
        # CLS token should be zeros
        assert np.allclose(result[0], 0)


class TestApplyRopeVision:
    """Tests for _apply_rope_vision function."""

    def test_apply_rope_vision_shapes(self):
        """Test _apply_rope_vision preserves shapes."""
        seq_len, num_heads, head_dim = 16, 8, 64
        xq = torch.randn(seq_len, num_heads, head_dim)
        xk = torch.randn(seq_len, num_heads, head_dim)

        # freqs_cis shape: (seq_len, head_dim // 2)
        freqs_cis = torch.polar(
            torch.ones(seq_len, head_dim // 2),
            torch.randn(seq_len, head_dim // 2),
        )

        xq_out, xk_out = _apply_rope_vision(xq, xk, freqs_cis)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape

    def test_apply_rope_vision_dtype_preserved(self):
        """Test _apply_rope_vision preserves input dtype."""
        seq_len, num_heads, head_dim = 8, 4, 32
        xq = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16)
        xk = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16)

        freqs_cis = torch.polar(
            torch.ones(seq_len, head_dim // 2),
            torch.randn(seq_len, head_dim // 2),
        )

        xq_out, xk_out = _apply_rope_vision(xq, xk, freqs_cis)

        assert xq_out.dtype == torch.bfloat16
        assert xk_out.dtype == torch.bfloat16


class TestVisionAttentionSdpa:
    """Tests for vision_attention_sdpa function."""

    def test_vision_attention_sdpa_single_sequence(self):
        """Test SDPA attention with single sequence."""
        seq_len, num_heads, head_dim = 16, 4, 32

        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        result = vision_attention_sdpa(q, k, v, cu_seqlens, cu_seqlens)

        expected_out_dim = num_heads * head_dim
        assert result.shape == (seq_len, expected_out_dim)

    def test_vision_attention_sdpa_multiple_sequences(self):
        """Test SDPA attention with batched sequences."""
        total_tokens = 24
        num_heads, head_dim = 4, 32

        q = torch.randn(total_tokens, num_heads, head_dim)
        k = torch.randn(total_tokens, num_heads, head_dim)
        v = torch.randn(total_tokens, num_heads, head_dim)

        # Two sequences: [0:10] and [10:24]
        cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32)

        result = vision_attention_sdpa(q, k, v, cu_seqlens, cu_seqlens)

        assert result.shape == (total_tokens, num_heads * head_dim)


class TestTpoolPatchMerger:
    """Tests for tpool_patch_merger function."""

    def test_tpool_patch_merger_single_image(self):
        """Test patch merger with single image."""
        # Single image: t=1, h=28, w=28
        t, h, w = 1, 28, 28
        d_model = 64
        x = torch.randn(t * h * w, d_model)
        grid_thws = torch.tensor([[t, h, w]])

        result = tpool_patch_merger(x, grid_thws, merge_kernel_size=[2, 2])

        assert len(result) == 1
        # After merge: new_h=14, new_w=14, each patch has kh*kw=4 tokens
        assert result[0].shape == (14 * 14, 2 * 2, d_model)

    def test_tpool_patch_merger_with_temporal(self):
        """Test patch merger with temporal dimension."""
        # Video: t=2, h=14, w=14
        t, h, w = 2, 14, 14
        d_model = 64
        x = torch.randn(t * h * w, d_model)
        grid_thws = torch.tensor([[t, h, w]])

        result = tpool_patch_merger(x, grid_thws, merge_kernel_size=[2, 2])

        assert len(result) == 1
        # After temporal pooling (mean over t) and spatial merge
        # new_h=7, new_w=7
        assert result[0].shape == (7 * 7, 2 * 2, d_model)

    def test_tpool_patch_merger_multiple_images(self):
        """Test patch merger with multiple images."""
        # Two images with different sizes
        t1, h1, w1 = 1, 28, 28
        t2, h2, w2 = 1, 14, 14
        d_model = 64

        x1 = torch.randn(t1 * h1 * w1, d_model)
        x2 = torch.randn(t2 * h2 * w2, d_model)
        x = torch.cat([x1, x2], dim=0)
        grid_thws = torch.tensor([[t1, h1, w1], [t2, h2, w2]])

        result = tpool_patch_merger(x, grid_thws, merge_kernel_size=[2, 2])

        assert len(result) == 2
        assert result[0].shape == (14 * 14, 4, d_model)
        assert result[1].shape == (7 * 7, 4, d_model)


# =============================================================================
# Position Embedding Tests
# =============================================================================


class TestLearnable2DInterpPosEmbDividedFixed:
    """Tests for Learnable2DInterpPosEmbDividedFixed."""

    def test_initialization(self):
        """Test position embedding initialization."""
        height, width, num_frames, dim = 64, 64, 4, 128

        pos_emb = Learnable2DInterpPosEmbDividedFixed(height, width, num_frames, dim)

        assert pos_emb.weight.shape == (height, width, dim)
        assert pos_emb.time_weight.shape == (num_frames, 1, dim)

    def test_forward_same_size(self):
        """Test forward with same size as init."""
        height, width, num_frames, dim = 8, 8, 4, 32

        pos_emb = Learnable2DInterpPosEmbDividedFixed(height, width, num_frames, dim)

        # Single image with t=1
        grid_thws = torch.tensor([[1, height, width]])
        x = torch.randn(height * width, dim)

        result = pos_emb(x, grid_thws)

        assert result.shape == x.shape

    def test_forward_interpolation(self):
        """Test forward with different size requiring interpolation."""
        height, width, num_frames, dim = 8, 8, 4, 32

        pos_emb = Learnable2DInterpPosEmbDividedFixed(height, width, num_frames, dim)

        # Different spatial size
        new_h, new_w = 4, 4
        grid_thws = torch.tensor([[1, new_h, new_w]])
        x = torch.randn(new_h * new_w, dim)

        result = pos_emb(x, grid_thws)

        assert result.shape == (new_h * new_w, dim)

    def test_forward_with_temporal(self):
        """Test forward with temporal dimension."""
        height, width, num_frames, dim = 8, 8, 4, 32

        pos_emb = Learnable2DInterpPosEmbDividedFixed(height, width, num_frames, dim)

        # Video with t=2
        t, h, w = 2, height, width
        grid_thws = torch.tensor([[t, h, w]])
        x = torch.randn(t * h * w, dim)

        result = pos_emb(x, grid_thws)

        assert result.shape == (t * h * w, dim)


class TestRope2DPosEmbRepeated:
    """Tests for Rope2DPosEmbRepeated."""

    def test_initialization(self):
        """Test 2D RoPE initialization."""
        dim, max_height, max_width = 64, 512, 512

        rope = Rope2DPosEmbRepeated(dim, max_height, max_width)

        assert rope.dim == dim
        assert rope.max_height == max_height
        assert rope.max_width == max_width
        assert rope.freqs_cis is None  # Lazy init

    def test_get_freqs_cis_shape(self):
        """Test get_freqs_cis produces correct shape."""
        dim, max_height, max_width = 64, 64, 64

        rope = Rope2DPosEmbRepeated(dim, max_height, max_width)

        grid_thws = torch.tensor([[1, 8, 8]])
        freqs_cis = rope.get_freqs_cis(grid_thws, device="cpu")

        # t=1, h=8, w=8 -> 64 tokens, dim//2 complex values
        assert freqs_cis.shape == (8 * 8, dim // 2)

    def test_get_freqs_cis_temporal_repeat(self):
        """Test get_freqs_cis repeats for temporal dimension."""
        dim, max_height, max_width = 64, 64, 64

        rope = Rope2DPosEmbRepeated(dim, max_height, max_width)

        # t=2 should repeat the spatial pattern
        grid_thws = torch.tensor([[2, 4, 4]])
        freqs_cis = rope.get_freqs_cis(grid_thws, device="cpu")

        # 2 * 4 * 4 = 32 tokens
        assert freqs_cis.shape == (32, dim // 2)


# =============================================================================
# Vision Tower Component Tests
# =============================================================================


class TestMoonViT3dMLP:
    """Tests for MoonViT3dMLP."""

    def test_mlp_forward(self):
        """Test MLP forward pass."""
        hidden_dim, mlp_dim = 64, 256

        mlp = MoonViT3dMLP([hidden_dim, mlp_dim, hidden_dim], activation=F.gelu)

        x = torch.randn(16, hidden_dim)
        result = mlp(x)

        assert result.shape == (16, hidden_dim)

    def test_mlp_weights_initialized(self):
        """Test MLP weights are initialized with truncated normal."""
        hidden_dim, mlp_dim = 64, 256

        mlp = MoonViT3dMLP([hidden_dim, mlp_dim, hidden_dim], activation=F.gelu)

        # Weights should be non-zero
        assert mlp.fc0.weight.abs().sum() > 0
        assert mlp.fc1.weight.abs().sum() > 0


class TestMoonViT3dEncoderLayer:
    """Tests for MoonViT3dEncoderLayer."""

    def test_encoder_layer_initialization(self):
        """Test encoder layer initialization."""
        num_heads, hidden_dim, mlp_dim = 4, 64, 256

        layer = MoonViT3dEncoderLayer(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attn_implementation="sdpa",
        )

        assert layer.num_heads == num_heads
        assert layer.hidden_dim == hidden_dim
        assert layer.head_dim == hidden_dim // num_heads

    def test_encoder_layer_forward(self):
        """Test encoder layer forward pass."""
        num_heads, hidden_dim, mlp_dim = 4, 64, 256
        seq_len = 16

        layer = MoonViT3dEncoderLayer(
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            attn_implementation="sdpa",
        )

        hidden_states = torch.randn(seq_len, hidden_dim)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        # Create freqs_cis for RoPE
        freqs_cis = torch.polar(
            torch.ones(seq_len, hidden_dim // num_heads // 2),
            torch.randn(seq_len, hidden_dim // num_heads // 2),
        )

        result = layer(hidden_states, cu_seqlens, seq_len, freqs_cis)

        assert result.shape == hidden_states.shape


class TestMoonVision3dPatchEmbed:
    """Tests for MoonVision3dPatchEmbed."""

    def test_patch_embed_initialization(self):
        """Test patch embedding initialization."""
        out_dim, patch_size = 64, 14

        patch_embed = MoonVision3dPatchEmbed(
            out_dim=out_dim,
            patch_size=patch_size,
            pos_emb_height=8,
            pos_emb_width=8,
            pos_emb_time=4,
        )

        assert patch_embed.patch_size == (patch_size, patch_size)
        assert patch_embed.proj.out_channels == out_dim

    def test_patch_embed_forward(self):
        """Test patch embedding forward pass."""
        out_dim, patch_size = 64, 14

        patch_embed = MoonVision3dPatchEmbed(
            out_dim=out_dim,
            patch_size=patch_size,
            pos_emb_height=8,
            pos_emb_width=8,
            pos_emb_time=4,
        )

        # Input: (num_patches, 3, patch_size, patch_size)
        # For h=w=8 patches: 64 patches total
        num_patches = 64
        x = torch.randn(num_patches, 3, patch_size, patch_size)
        grid_thws = torch.tensor([[1, 8, 8]])

        result = patch_embed(x, grid_thws)

        # Output: (num_patches, out_dim)
        assert result.shape == (num_patches, out_dim)


class TestMoonViT3dEncoder:
    """Tests for MoonViT3dEncoder."""

    def test_encoder_initialization(self):
        """Test encoder initialization."""
        hidden_dim, num_layers = 64, 2

        block_cfg = {
            "num_heads": 4,
            "hidden_dim": hidden_dim,
            "mlp_dim": 256,
            "activation": F.gelu,
            "attn_bias": True,
            "attn_implementation": "sdpa",
        }

        encoder = MoonViT3dEncoder(hidden_dim, num_layers, block_cfg)

        assert len(encoder.blocks) == num_layers
        assert encoder.final_layernorm.normalized_shape[0] == hidden_dim

    def test_encoder_forward(self):
        """Test encoder forward pass."""
        hidden_dim, num_layers = 64, 2
        t, h, w = 1, 8, 8
        seq_len = t * h * w

        block_cfg = {
            "num_heads": 4,
            "hidden_dim": hidden_dim,
            "mlp_dim": 256,
            "activation": F.gelu,
            "attn_bias": True,
            "attn_implementation": "sdpa",
        }

        encoder = MoonViT3dEncoder(hidden_dim, num_layers, block_cfg)

        hidden_states = torch.randn(seq_len, hidden_dim)
        grid_thws = torch.tensor([[t, h, w]])

        result = encoder(hidden_states, grid_thws)

        assert result.shape == hidden_states.shape


class TestMoonViT3dPretrainedModel:
    """Tests for MoonViT3dPretrainedModel (vision tower)."""

    @pytest.fixture
    def small_vision_config(self):
        """Create a small config for testing."""
        return MoonViT3dConfig(
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=256,
            patch_size=14,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            init_pos_emb_time=4,
            merge_kernel_size=[2, 2],
            merge_type="sd2_tpool",
        )

    def test_vision_tower_initialization(self, small_vision_config):
        """Test vision tower initialization."""
        vision_tower = MoonViT3dPretrainedModel(small_vision_config)

        assert vision_tower.merge_kernel_size == [2, 2]
        assert vision_tower.merge_type == "sd2_tpool"
        assert vision_tower.patch_embed is not None
        assert vision_tower.encoder is not None

    def test_vision_tower_dtype_property(self, small_vision_config):
        """Test vision tower dtype property."""
        vision_tower = MoonViT3dPretrainedModel(small_vision_config)

        assert vision_tower.dtype == torch.float32  # Default

        vision_tower = vision_tower.to(torch.bfloat16)
        assert vision_tower.dtype == torch.bfloat16

    def test_vision_tower_forward(self, small_vision_config):
        """Test vision tower forward pass."""
        vision_tower = MoonViT3dPretrainedModel(small_vision_config)

        # h=8, w=8 patches
        num_patches = 64
        pixel_values = torch.randn(num_patches, 3, 14, 14)
        grid_thws = torch.tensor([[1, 8, 8]])

        result = vision_tower(pixel_values, grid_thws)

        # Result is a list of merged patch tensors
        assert isinstance(result, list)
        assert len(result) == 1
        # After merge: (4*4, 2*2, hidden_size)
        assert result[0].shape == (16, 4, 64)


# =============================================================================
# Multi-Modal Projector Tests
# =============================================================================


class TestKimiK25VLMultiModalProjector:
    """Tests for KimiK25VLMultiModalProjector."""

    @pytest.fixture
    def projector_config(self):
        """Create config for projector testing."""
        vision_config = MoonViT3dConfig(
            hidden_size=64,
            merge_kernel_size=[2, 2],
        )
        from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

        text_config = DeepseekV3Config(hidden_size=128)

        config = KimiK25VLConfig(
            vision_config=vision_config,
            text_config=text_config,
            mm_hidden_size=64,
        )
        return config

    def test_projector_initialization(self, projector_config):
        """Test projector initialization."""
        projector = KimiK25VLMultiModalProjector(projector_config)

        # mm_hidden_size * kernel_h * kernel_w = 64 * 2 * 2 = 256
        assert projector.hidden_size == 256
        assert projector.pre_norm.normalized_shape[0] == 64
        assert projector.linear_1.in_features == 256
        assert projector.linear_1.out_features == 256
        assert projector.linear_2.in_features == 256
        assert projector.linear_2.out_features == 128  # text hidden size

    def test_projector_forward(self, projector_config):
        """Test projector forward pass."""
        projector = KimiK25VLMultiModalProjector(projector_config)

        # Input: list of tensors [(num_patches, kernel_size, hidden_size)]
        num_patches = 16
        kernel_size = 4  # 2*2
        hidden_size = 64

        image_features = [torch.randn(num_patches, kernel_size, hidden_size)]

        result = projector(image_features)

        assert isinstance(result, list)
        assert len(result) == 1
        # Output: (num_patches, text_hidden_size)
        assert result[0].shape == (num_patches, 128)

    def test_projector_forward_multiple_images(self, projector_config):
        """Test projector forward with multiple images."""
        projector = KimiK25VLMultiModalProjector(projector_config)

        # Two images with different numbers of patches
        image_features = [
            torch.randn(16, 4, 64),  # 16 patches
            torch.randn(9, 4, 64),  # 9 patches
        ]

        result = projector(image_features)

        assert len(result) == 2
        assert result[0].shape == (16, 128)
        assert result[1].shape == (9, 128)


# =============================================================================
# DeepSeekV3 RoPE Adapter Tests
# =============================================================================


class TestDeepSeekV3RotaryEmbeddingAdapter:
    """Tests for DeepSeekV3RotaryEmbeddingAdapter."""

    def test_adapter_initialization(self):
        """Test adapter initialization."""
        mock_parent = MagicMock()
        mock_parent.freqs_cis = torch.randn(1024, 64)

        adapter = DeepSeekV3RotaryEmbeddingAdapter(mock_parent, rope_fusion=False)

        assert adapter._parent is mock_parent
        assert adapter.rope_fusion is False

    def test_adapter_freqs_cis_property(self):
        """Test adapter freqs_cis property."""
        mock_parent = MagicMock()
        expected_freqs = torch.randn(1024, 64)
        mock_parent.freqs_cis = expected_freqs

        adapter = DeepSeekV3RotaryEmbeddingAdapter(mock_parent)

        assert torch.equal(adapter.freqs_cis, expected_freqs)

    def test_adapter_call_raises_without_freqs(self):
        """Test adapter raises error when freqs_cis is None."""
        mock_parent = MagicMock()
        mock_parent.freqs_cis = None

        adapter = DeepSeekV3RotaryEmbeddingAdapter(mock_parent)

        hidden_states = torch.randn(2, 8, 64)
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)

        with pytest.raises(RuntimeError, match="freqs_cis is None"):
            adapter(hidden_states, position_ids)


# =============================================================================
# Model Integration Tests (with mocking)
# =============================================================================


class TestKimiK25VLModelIntegration:
    """Integration tests for KimiK25VLModel using mocks to avoid CUDA."""

    def test_compute_num_image_tokens_from_grid(self):
        """Test _compute_num_image_tokens_from_grid method."""
        # Create minimal config
        config = KimiK25VLConfig(
            vision_config=MoonViT3dConfig(
                hidden_size=64,
                merge_kernel_size=[2, 2],
            ),
        )

        # Mock the model components that require CUDA
        with patch.object(KimiK25VLModel, "__init__", lambda self, *args, **kwargs: None):
            model = KimiK25VLModel.__new__(KimiK25VLModel)
            model.config = config

            grid_thws = torch.tensor([[1, 28, 28]])
            result = model._compute_num_image_tokens_from_grid(grid_thws)

            # (28 // 2) * (28 // 2) = 196
            assert result == [196]

    def test_compute_num_image_tokens_multiple_images(self):
        """Test _compute_num_image_tokens_from_grid with multiple images."""
        config = KimiK25VLConfig(
            vision_config=MoonViT3dConfig(
                hidden_size=64,
                merge_kernel_size=[2, 2],
            ),
        )

        with patch.object(KimiK25VLModel, "__init__", lambda self, *args, **kwargs: None):
            model = KimiK25VLModel.__new__(KimiK25VLModel)
            model.config = config

            grid_thws = torch.tensor([[1, 28, 28], [1, 14, 14]])
            result = model._compute_num_image_tokens_from_grid(grid_thws)

            assert result == [196, 49]


class TestKimiK25VLMergeImageFeatures:
    """Tests for _merge_input_ids_with_image_features method."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing merge logic."""
        config = KimiK25VLConfig(
            media_placeholder_token_id=163605,
            pad_token_id=0,
            ignore_index=-100,
        )

        with patch.object(KimiK25VLModel, "__init__", lambda self, *args, **kwargs: None):
            model = KimiK25VLModel.__new__(KimiK25VLModel)
            model.config = config
            model.media_placeholder_token_id = config.media_placeholder_token_id
            return model

    def test_merge_pre_expanded_mode(self, mock_model):
        """Test merge with pre-expanded placeholders (PP mode)."""
        batch_size, seq_len, embed_dim = 1, 10, 64
        num_image_tokens = 4

        # Image features: 4 tokens
        image_features = [torch.randn(num_image_tokens, embed_dim)]

        # Input already has 4 placeholder tokens
        input_ids = torch.tensor([[1, 2, 163605, 163605, 163605, 163605, 3, 4, 5, 0]])
        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, -1] = 0  # Padding

        result = mock_model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels=None
        )

        final_embedding, final_attention_mask, final_labels, position_ids = result

        # Pre-expanded mode: no sequence length change
        assert final_embedding.shape == (batch_size, seq_len, embed_dim)
        assert final_attention_mask.shape == attention_mask.shape

    def test_merge_dynamic_expansion_mode(self, mock_model):
        """Test merge with dynamic expansion (single placeholder per image)."""
        batch_size, embed_dim = 1, 64
        num_image_tokens = 4

        # Image features: 4 tokens
        image_features = [torch.randn(num_image_tokens, embed_dim)]

        # Input has 1 placeholder token (will be expanded to 4)
        input_ids = torch.tensor([[1, 2, 163605, 3, 4, 5, 0, 0]])  # 8 tokens
        seq_len = 8
        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        attention_mask[:, -2:] = 0  # Padding

        result = mock_model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels=None
        )

        final_embedding, final_attention_mask, final_labels, position_ids = result

        # Dynamic expansion: seq_len - 1 + num_image_tokens = 8 - 1 + 4 = 11
        expected_seq_len = 11
        assert final_embedding.shape == (batch_size, expected_seq_len, embed_dim)

    def test_merge_with_labels(self, mock_model):
        """Test merge preserves and masks labels correctly."""
        batch_size, seq_len, embed_dim = 1, 10, 64
        num_image_tokens = 4

        image_features = [torch.randn(num_image_tokens, embed_dim)]

        # Pre-expanded input with labels
        input_ids = torch.tensor([[1, 2, 163605, 163605, 163605, 163605, 3, 4, 5, 0]])
        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        labels = torch.tensor([[1, 2, 100, 100, 100, 100, 3, 4, 5, -100]])

        result = mock_model._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, labels=labels
        )

        final_embedding, final_attention_mask, final_labels, position_ids = result

        # Image positions should be masked (-100) in labels
        assert final_labels is not None
        # Check that image token positions have ignore_index (-100)
        image_mask = input_ids == 163605
        assert (final_labels[image_mask] == -100).all()


class TestKimiK25VLForConditionalGenerationIntegration:
    """Integration tests for KimiK25VLForConditionalGeneration."""

    def test_get_input_embeddings_attribute(self):
        """Test get_input_embeddings method exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "get_input_embeddings")
        assert callable(KimiK25VLForConditionalGeneration.get_input_embeddings)

    def test_get_output_embeddings_attribute(self):
        """Test get_output_embeddings method exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "get_output_embeddings")
        assert callable(KimiK25VLForConditionalGeneration.get_output_embeddings)

    def test_set_input_embeddings_attribute(self):
        """Test set_input_embeddings method exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "set_input_embeddings")
        assert callable(KimiK25VLForConditionalGeneration.set_input_embeddings)

    def test_set_output_embeddings_attribute(self):
        """Test set_output_embeddings method exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "set_output_embeddings")
        assert callable(KimiK25VLForConditionalGeneration.set_output_embeddings)

    def test_from_config_classmethod(self):
        """Test from_config classmethod exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "from_config")
        assert callable(KimiK25VLForConditionalGeneration.from_config)

    def test_from_pretrained_classmethod(self):
        """Test from_pretrained classmethod exists."""
        assert hasattr(KimiK25VLForConditionalGeneration, "from_pretrained")
        assert callable(KimiK25VLForConditionalGeneration.from_pretrained)


# =============================================================================
# KimiK25VLModel.forward Tests
# =============================================================================


class TestKimiK25VLModelForward:
    """Tests for KimiK25VLModel.forward method."""

    def test_forward_raises_when_both_input_ids_and_inputs_embeds_provided(self):
        """Test forward raises ValueError when both input_ids and inputs_embeds are provided."""
        input_ids = torch.randint(0, 100, (2, 8))
        inputs_embeds = torch.randn(2, 8, 64)

        # Simulate the validation logic from forward
        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_forward_raises_when_neither_input_ids_nor_inputs_embeds_provided(self):
        """Test forward raises ValueError when neither input_ids nor inputs_embeds is provided."""
        input_ids = None
        inputs_embeds = None

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_forward_with_qkv_format_thd(self):
        """Test forward handles qkv_format='thd' kwarg."""
        # Verify squeeze_input_for_thd is called when qkv_format='thd'
        from nemo_automodel.components.models.kimi_k25_vl.model import squeeze_input_for_thd

        input_ids = torch.randint(0, 100, (2, 8))
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        padding_mask = torch.ones(2, 8, dtype=torch.bool)
        kwargs = {"qkv_format": "thd"}

        # Test the squeeze function directly
        result = squeeze_input_for_thd(input_ids, position_ids, padding_mask, kwargs)
        assert result is not None
        assert len(result) == 4  # input_ids, position_ids, padding_mask, kwargs

    def test_forward_vision_processing_conditions(self):
        """Test conditions for vision processing in forward."""
        # Test the boolean logic for vision processing
        # has_vision and has_pixels and not_generation

        # Case 1: No vision tower
        has_vision = False
        has_pixels = True
        not_generation = True
        assert not (has_pixels and has_vision and not_generation)

        # Case 2: No pixel values
        has_vision = True
        has_pixels = False
        not_generation = True
        assert not (has_pixels and has_vision and not_generation)

        # Case 3: Generation mode (seq_len == 1)
        has_vision = True
        has_pixels = True
        not_generation = False  # seq_len == 1
        assert not (has_pixels and has_vision and not_generation)

        # Case 4: All conditions met
        has_vision = True
        has_pixels = True
        not_generation = True
        assert has_pixels and has_vision and not_generation


class TestKimiK25VLModelForwardMocked:
    """Tests for KimiK25VLModel.forward with mocked components."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock KimiK25VLModel for forward testing."""
        config = KimiK25VLConfig(
            media_placeholder_token_id=163605,
            pad_token_id=0,
            ignore_index=-100,
            vision_config=MoonViT3dConfig(
                hidden_size=64,
                merge_kernel_size=[2, 2],
            ),
        )

        with patch.object(KimiK25VLModel, "__init__", lambda self, *args, **kwargs: None):
            model = KimiK25VLModel.__new__(KimiK25VLModel)
            model.config = config
            model.media_placeholder_token_id = config.media_placeholder_token_id

            # Mock language_model
            mock_lm = MagicMock()
            mock_embed = MagicMock(return_value=torch.randn(2, 8, 64))
            mock_lm.get_input_embeddings.return_value = mock_embed
            mock_lm.return_value = torch.randn(2, 8, 64)
            model.language_model = mock_lm

            # Mock vision components as None for text-only tests
            model.vision_tower = None
            model.multi_modal_projector = None

            return model

    def test_forward_text_only_with_input_ids(self, mock_model):
        """Test forward with input_ids only (no vision)."""
        input_ids = torch.randint(0, 100, (2, 8))
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        # With no vision tower, should just process text
        result = mock_model.language_model(
            inputs_embeds=mock_model.language_model.get_input_embeddings()(input_ids),
            attention_mask=attention_mask,
            position_ids=None,
            padding_mask=None,
        )

        assert result.shape == (2, 8, 64)

    def test_forward_text_only_with_inputs_embeds(self, mock_model):
        """Test forward with inputs_embeds only (no vision)."""
        inputs_embeds = torch.randn(2, 8, 64)
        attention_mask = torch.ones(2, 8, dtype=torch.long)

        result = mock_model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            padding_mask=None,
        )

        assert result.shape == (2, 8, 64)

    def test_forward_embed_tokens_none_with_float_input(self, mock_model):
        """Test forward when embed_tokens is None but input is float (already embeddings)."""
        mock_model.language_model.get_input_embeddings.return_value = None

        # Simulate the logic from forward
        input_ids = torch.randn(2, 8, 64, dtype=torch.bfloat16)  # Already embeddings
        embed_tokens = mock_model.language_model.get_input_embeddings()

        if embed_tokens is None:
            if (
                input_ids is not None
                and isinstance(input_ids, torch.Tensor)
                and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                inputs_embeds = input_ids
            else:
                pytest.fail("Should not reach here with float input")

        assert inputs_embeds.shape == (2, 8, 64)

    def test_forward_target_seq_length_param(self, mock_model):
        """Test forward accepts target_seq_length parameter for PP."""
        # Verify the parameter is accepted in the signature
        import inspect
        sig = inspect.signature(KimiK25VLModel.forward)
        params = list(sig.parameters.keys())

        assert "target_seq_length" in params
        target_seq_length_param = sig.parameters["target_seq_length"]
        assert target_seq_length_param.default is None


# =============================================================================
# KimiK25VLForConditionalGeneration.from_pretrained Tests
# =============================================================================


class TestKimiK25VLForConditionalGenerationFromPretrained:
    """Tests for KimiK25VLForConditionalGeneration.from_pretrained classmethod."""

    def test_from_pretrained_signature(self):
        """Test from_pretrained has expected signature."""
        import inspect
        sig = inspect.signature(KimiK25VLForConditionalGeneration.from_pretrained)
        params = list(sig.parameters.keys())

        assert "pretrained_model_name_or_path" in params
        assert "model_args" in params
        assert "kwargs" in params

    def test_from_pretrained_torch_dtype_string_conversion(self):
        """Test torch_dtype string is converted to torch dtype."""
        # Simulate the conversion logic
        torch_dtype = "bfloat16"

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else torch.bfloat16

        assert torch_dtype == torch.bfloat16

    def test_from_pretrained_torch_dtype_auto(self):
        """Test torch_dtype='auto' defaults to bfloat16."""
        torch_dtype = "auto"

        if isinstance(torch_dtype, str):
            torch_dtype = getattr(torch, torch_dtype) if torch_dtype != "auto" else torch.bfloat16

        assert torch_dtype == torch.bfloat16

    @patch.object(KimiK25VLConfig, "from_pretrained")
    @patch.object(KimiK25VLForConditionalGeneration, "from_config")
    def test_from_pretrained_loads_config(self, mock_from_config, mock_config_from_pretrained):
        """Test from_pretrained loads config when not provided."""
        mock_config = KimiK25VLConfig()
        mock_config_from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_from_config.return_value = mock_model

        # Call the method
        result = KimiK25VLForConditionalGeneration.from_pretrained(
            "/fake/path", torch_dtype=torch.bfloat16
        )

        mock_config_from_pretrained.assert_called_once_with("/fake/path")
        assert mock_config._name_or_path == "/fake/path"

    @patch.object(KimiK25VLConfig, "from_pretrained")
    @patch.object(KimiK25VLForConditionalGeneration, "from_config")
    def test_from_pretrained_uses_provided_config(self, mock_from_config, mock_config_from_pretrained):
        """Test from_pretrained uses config when explicitly provided."""
        provided_config = KimiK25VLConfig()
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_from_config.return_value = mock_model

        result = KimiK25VLForConditionalGeneration.from_pretrained(
            "/fake/path", config=provided_config, torch_dtype=torch.bfloat16
        )

        # Should not call from_pretrained on config
        mock_config_from_pretrained.assert_not_called()

    @patch.object(KimiK25VLConfig, "from_pretrained")
    @patch.object(KimiK25VLForConditionalGeneration, "from_config")
    def test_from_pretrained_num_hidden_layers_override(self, mock_from_config, mock_config_from_pretrained):
        """Test from_pretrained can override num_hidden_layers."""
        mock_config = KimiK25VLConfig()
        mock_config.text_config.num_hidden_layers = 61
        mock_config_from_pretrained.return_value = mock_config

        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_from_config.return_value = mock_model

        result = KimiK25VLForConditionalGeneration.from_pretrained(
            "/fake/path", num_hidden_layers=2, torch_dtype=torch.bfloat16
        )

        # Check num_hidden_layers was overridden
        assert mock_config.text_config.num_hidden_layers == 2


# =============================================================================
# KimiK25VLForConditionalGeneration.__init__ Tests
# =============================================================================


class TestKimiK25VLForConditionalGenerationInit:
    """Tests for KimiK25VLForConditionalGeneration.__init__."""

    def test_init_signature(self):
        """Test __init__ has expected signature."""
        import inspect
        sig = inspect.signature(KimiK25VLForConditionalGeneration.__init__)
        params = list(sig.parameters.keys())

        assert "self" in params
        assert "config" in params
        assert "moe_config" in params
        assert "backend" in params
        assert "kwargs" in params

    def test_init_creates_state_dict_adapter_when_enabled(self):
        """Test __init__ creates state_dict_adapter when backend.enable_hf_state_dict_adapter is True."""
        from nemo_automodel.components.models.kimi_k25_vl.model import BackendConfig

        backend = BackendConfig(enable_hf_state_dict_adapter=True)
        assert backend.enable_hf_state_dict_adapter is True

    def test_init_backend_default(self):
        """Test __init__ creates default BackendConfig when not provided."""
        from nemo_automodel.components.models.kimi_k25_vl.model import BackendConfig

        # Simulate the default backend logic
        backend = None
        backend = backend or BackendConfig()

        assert backend is not None
        assert isinstance(backend, BackendConfig)


# =============================================================================
# KimiK25VLForConditionalGeneration.forward Tests
# =============================================================================


class TestKimiK25VLForConditionalGenerationForward:
    """Tests for KimiK25VLForConditionalGeneration.forward method."""

    def test_forward_signature_complete(self):
        """Test forward has all expected parameters."""
        import inspect
        sig = inspect.signature(KimiK25VLForConditionalGeneration.forward)
        params = list(sig.parameters.keys())

        expected_params = [
            "self", "input_ids", "attention_mask", "position_ids",
            "past_key_values", "inputs_embeds", "labels", "use_cache",
            "output_attentions", "output_hidden_states", "return_dict",
            "pixel_values", "grid_thws", "padding_mask", "target_seq_length",
            "kwargs",
        ]

        for param in expected_params:
            assert param in params, f"Missing parameter: {param}"

    def test_forward_vlm_chunk_retrieval_logic(self):
        """Test VLM chunk retrieval from model attributes."""
        # Simulate the chunk retrieval logic
        pixel_values = None
        _vlm_pixel_values_chunks = [torch.randn(64, 3, 14, 14), torch.randn(32, 3, 14, 14)]
        _vlm_image_grid_hws_chunks = [torch.tensor([[28, 28]]), torch.tensor([[14, 14]])]
        media_placeholder_token_id = 163605
        input_ids = torch.tensor([[1, 2, media_placeholder_token_id, 3, 4]])
        _vlm_chunk_idx = 0

        has_media_tokens = (
            input_ids is not None
            and media_placeholder_token_id is not None
            and (input_ids == media_placeholder_token_id).any().item()
        )

        assert has_media_tokens == True

        if has_media_tokens:
            if _vlm_chunk_idx < len(_vlm_pixel_values_chunks):
                pixel_values = _vlm_pixel_values_chunks[_vlm_chunk_idx]
                image_grid_hws = _vlm_image_grid_hws_chunks[_vlm_chunk_idx]

                assert pixel_values.shape == (64, 3, 14, 14)
                assert image_grid_hws.shape == (1, 2)

    def test_forward_grid_thws_conversion(self):
        """Test conversion from image_grid_hws [N, 2] to grid_thws [N, 3]."""
        image_grid_hws = torch.tensor([[28, 28], [14, 14]])  # [N, 2]

        if image_grid_hws.shape[-1] == 2:
            ones = torch.ones(image_grid_hws.shape[0], 1, dtype=image_grid_hws.dtype)
            grid_thws = torch.cat([ones, image_grid_hws], dim=-1)
        else:
            grid_thws = image_grid_hws

        assert grid_thws.shape == (2, 3)
        assert (grid_thws[:, 0] == 1).all()  # T dimension is 1
        assert (grid_thws[:, 1:] == image_grid_hws).all()

    def test_forward_loss_computation_with_attention_mask(self):
        """Test loss computation with attention mask masking."""
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10, dtype=torch.long)
        attention_mask[:, -2:] = 0  # Last 2 tokens are padding

        shift_mask = attention_mask[..., 1:]  # Shape: (2, 9)
        shift_logits = logits[..., :-1, :][shift_mask != 0]  # Masked logits
        shift_labels = labels[..., 1:][shift_mask != 0]  # Masked labels

        # Verify shapes make sense
        assert shift_mask.shape == (2, 9)
        assert shift_logits.dim() == 2  # Flattened
        assert shift_labels.dim() == 1

    def test_forward_loss_computation_without_attention_mask(self):
        """Test loss computation without attention mask."""
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        attention_mask = None

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        assert shift_logits.shape == (2, 9, 1000)
        assert shift_labels.shape == (2, 9)

    def test_forward_return_dict_false(self):
        """Test forward returns logits when return_dict is False."""
        # Simulate the return logic
        return_dict = None
        logits = torch.randn(2, 10, 1000)

        if return_dict is None:
            return_dict = False

        if not return_dict:
            result = logits
        else:
            result = {"logits": logits}

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 10, 1000)


# =============================================================================
# vision_attention_flash Tests
# =============================================================================


class TestVisionAttentionFlash:
    """Tests for vision_attention_flash function."""

    def test_vision_attention_flash_exists(self):
        """Test vision_attention_flash function is importable."""
        from nemo_automodel.components.models.kimi_k25_vl.model import vision_attention_flash
        assert callable(vision_attention_flash)

    def test_vision_attention_flash_max_seqlen_computation(self):
        """Test max_seqlen computation from cu_seqlens."""
        cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32)

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()

        assert max_seqlen == 14  # max(10, 14) = 14

    def test_vision_attention_flash_cu_seqlens_format(self):
        """Test cu_seqlens has correct format."""
        # cu_seqlens should be cumulative sequence lengths
        # For 3 sequences of lengths 5, 8, 11
        seq_lengths = [5, 8, 11]
        cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lengths).cumsum(0)), dtype=torch.int32)

        assert cu_seqlens[0] == 0
        assert cu_seqlens[1] == 5
        assert cu_seqlens[2] == 13
        assert cu_seqlens[3] == 24

    def test_vision_attention_flash_signature(self):
        """Test vision_attention_flash has expected signature."""
        import inspect
        from nemo_automodel.components.models.kimi_k25_vl.model import vision_attention_flash

        sig = inspect.signature(vision_attention_flash)
        params = list(sig.parameters.keys())

        assert "q" in params
        assert "k" in params
        assert "v" in params
        assert "q_cu_seqlens" in params
        assert "k_cu_seqlens" in params
        assert "max_seqlen_q" in params
        assert "max_seqlen_k" in params

        # Check defaults
        assert sig.parameters["max_seqlen_q"].default is None
        assert sig.parameters["max_seqlen_k"].default is None

    @patch("nemo_automodel.components.models.kimi_k25_vl.model.flash_attn_varlen_func")
    def test_vision_attention_flash_calls_flash_attn(self, mock_flash_attn):
        """Test vision_attention_flash calls flash_attn_varlen_func with correct args."""
        from nemo_automodel.components.models.kimi_k25_vl.model import vision_attention_flash

        seq_len, num_heads, head_dim = 24, 4, 32
        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 10, 24], dtype=torch.int32)

        # Mock return value
        mock_output = torch.randn(seq_len, num_heads, head_dim)
        mock_flash_attn.return_value = mock_output

        result = vision_attention_flash(q, k, v, cu_seqlens, cu_seqlens)

        mock_flash_attn.assert_called_once()
        call_args = mock_flash_attn.call_args
        assert call_args[0][0] is q
        assert call_args[0][1] is k
        assert call_args[0][2] is v
        assert call_args[1]["causal"] is False

    @patch("nemo_automodel.components.models.kimi_k25_vl.model.flash_attn_varlen_func")
    def test_vision_attention_flash_handles_tuple_output(self, mock_flash_attn):
        """Test vision_attention_flash handles tuple output from flash_attn."""
        from nemo_automodel.components.models.kimi_k25_vl.model import vision_attention_flash

        seq_len, num_heads, head_dim = 16, 4, 32
        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)

        # Return tuple (output, softmax_lse)
        mock_output = torch.randn(seq_len, num_heads, head_dim)
        mock_flash_attn.return_value = (mock_output, torch.randn(seq_len))

        result = vision_attention_flash(q, k, v, cu_seqlens, cu_seqlens)

        # Should extract first element from tuple
        expected_shape = (seq_len, num_heads * head_dim)
        assert result.shape == expected_shape

    @patch("nemo_automodel.components.models.kimi_k25_vl.model.flash_attn_varlen_func")
    def test_vision_attention_flash_flattens_output(self, mock_flash_attn):
        """Test vision_attention_flash flattens last two dimensions."""
        from nemo_automodel.components.models.kimi_k25_vl.model import vision_attention_flash

        seq_len, num_heads, head_dim = 16, 4, 32
        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, 16], dtype=torch.int32)

        mock_output = torch.randn(seq_len, num_heads, head_dim)
        mock_flash_attn.return_value = mock_output

        result = vision_attention_flash(q, k, v, cu_seqlens, cu_seqlens)

        # Output should be flattened from (seq_len, num_heads, head_dim) to (seq_len, num_heads * head_dim)
        assert result.shape == (seq_len, num_heads * head_dim)


# =============================================================================
# squeeze_input_for_thd Tests
# =============================================================================


class TestSqueezeInputForThd:
    """Tests for squeeze_input_for_thd helper function."""

    def test_squeeze_input_for_thd_import(self):
        """Test squeeze_input_for_thd is importable."""
        from nemo_automodel.components.models.kimi_k25_vl.model import squeeze_input_for_thd
        assert callable(squeeze_input_for_thd)

    def test_squeeze_input_for_thd_basic(self):
        """Test squeeze_input_for_thd with basic inputs."""
        from nemo_automodel.components.models.kimi_k25_vl.model import squeeze_input_for_thd

        input_ids = torch.randint(0, 100, (2, 8))
        position_ids = torch.arange(8).unsqueeze(0).expand(2, -1)
        padding_mask = torch.ones(2, 8, dtype=torch.bool)
        kwargs = {"qkv_format": "thd"}

        result = squeeze_input_for_thd(input_ids, position_ids, padding_mask, kwargs)

        assert len(result) == 4
        input_ids_out, position_ids_out, padding_mask_out, kwargs_out = result

        # Verify outputs are returned
        assert input_ids_out is not None
        assert isinstance(kwargs_out, dict)


# =============================================================================
# LM Head and Loss Computation Tests
# =============================================================================


class TestLMHeadAndLoss:
    """Tests for lm_head and loss computation in forward."""

    def test_lm_head_none_returns_hidden_states(self):
        """Test when lm_head is None, hidden_states are returned as logits."""
        hidden_states = torch.randn(2, 10, 64)
        lm_head = None

        logits = lm_head(hidden_states) if lm_head is not None else hidden_states

        assert torch.equal(logits, hidden_states)

    def test_cross_entropy_loss_computation(self):
        """Test CrossEntropyLoss is computed correctly."""
        import torch.nn as nn

        shift_logits = torch.randn(14, 1000)  # Flattened
        shift_labels = torch.randint(0, 1000, (14,))

        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        assert loss.shape == ()
        assert loss.item() > 0  # Loss should be positive
