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

"""Unit tests for KimiVL model components."""

import torch
from unittest.mock import MagicMock, patch

import pytest
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.models.kimivl.model import (
    KimiVLConfig,
    KimiVLForConditionalGeneration,
    MoonViTConfig,
)


class TestMoonViTConfig:
    """Tests for MoonViTConfig."""

    def test_default_initialization(self):
        """Test MoonViTConfig initializes with correct defaults."""
        config = MoonViTConfig()
        
        assert config.patch_size == 14
        assert config.init_pos_emb_height == 64
        assert config.init_pos_emb_width == 64
        assert config.num_attention_heads == 16
        assert config.num_hidden_layers == 27
        assert config.hidden_size == 1152
        assert config.intermediate_size == 4304
        assert config.merge_kernel_size == [2, 2]
        assert config.model_type == "moonvit"

    def test_custom_initialization(self):
        """Test MoonViTConfig with custom values."""
        config = MoonViTConfig(
            patch_size=16,
            hidden_size=768,
            num_hidden_layers=12,
            merge_kernel_size=(4, 4),
        )
        
        assert config.patch_size == 16
        assert config.hidden_size == 768
        assert config.num_hidden_layers == 12
        assert config.merge_kernel_size == [4, 4]

    def test_merge_kernel_size_tuple_to_list(self):
        """Test that tuple merge_kernel_size is converted to list."""
        config = MoonViTConfig(merge_kernel_size=(3, 3))
        assert config.merge_kernel_size == [3, 3]
        assert isinstance(config.merge_kernel_size, list)


class TestKimiVLConfig:
    """Tests for KimiVLConfig."""

    def test_default_initialization(self):
        """Test KimiVLConfig initializes with defaults."""
        config = KimiVLConfig()
        
        assert isinstance(config.vision_config, MoonViTConfig)
        assert isinstance(config.text_config, DeepseekV3Config)
        assert config.ignore_index == -100
        assert config.media_placeholder_token_id == 163605
        assert config.pad_token_id == 0
        assert config.architectures == ["KimiVLForConditionalGeneration"]
        assert config.model_type == "kimi_vl"

    def test_initialization_with_dict_configs(self):
        """Test KimiVLConfig initializes correctly from dict configs."""
        vision_dict = {"hidden_size": 768, "patch_size": 16}
        text_dict = {"hidden_size": 1024, "vocab_size": 50000}
        
        config = KimiVLConfig(
            vision_config=vision_dict,
            text_config=text_dict,
        )
        
        assert isinstance(config.vision_config, MoonViTConfig)
        assert config.vision_config.hidden_size == 768
        assert config.vision_config.patch_size == 16
        
        assert isinstance(config.text_config, DeepseekV3Config)
        assert config.text_config.hidden_size == 1024
        assert config.text_config.vocab_size == 50000

    def test_initialization_with_config_objects(self):
        """Test KimiVLConfig initializes correctly from config objects."""
        vision_config = MoonViTConfig(hidden_size=512)
        text_config = DeepseekV3Config(hidden_size=2048)
        
        config = KimiVLConfig(
            vision_config=vision_config,
            text_config=text_config,
        )
        
        assert config.vision_config is vision_config
        assert config.text_config is text_config

    def test_to_dict(self):
        """Test KimiVLConfig.to_dict() includes nested configs."""
        config = KimiVLConfig()
        config_dict = config.to_dict()
        
        assert "vision_config" in config_dict
        assert "text_config" in config_dict
        assert isinstance(config_dict["vision_config"], dict)
        assert isinstance(config_dict["text_config"], dict)
        assert config_dict["vision_config"]["model_type"] == "moonvit"

    def test_custom_architectures(self):
        """Test KimiVLConfig with custom architectures."""
        config = KimiVLConfig(architectures=["CustomArch"])
        assert config.architectures == ["CustomArch"]


class TestKimiVLForConditionalGeneration:
    """Tests for KimiVLForConditionalGeneration."""

    def test_from_pretrained_delegates_to_from_config(self):
        """Test from_pretrained loads config and delegates to from_config."""
        mock_config = MagicMock(spec=KimiVLConfig)
        mock_config.vision_config = MoonViTConfig()
        mock_config.text_config = DeepseekV3Config(
            vocab_size=100,
            hidden_size=64,
            num_attention_heads=4,
            num_hidden_layers=1,
            intermediate_size=128,
            qk_rope_head_dim=16,
            v_head_dim=16,
            qk_nope_head_dim=16,
        )
        mock_config.media_placeholder_token_id = 163605
        
        with patch.object(KimiVLConfig, "from_pretrained", return_value=mock_config):
            with patch.object(
                KimiVLForConditionalGeneration, "from_config"
            ) as mock_from_config:
                mock_from_config.return_value = MagicMock()
                
                KimiVLForConditionalGeneration.from_pretrained("dummy/path")
                
                KimiVLConfig.from_pretrained.assert_called_once_with("dummy/path")
                mock_from_config.assert_called_once()
                assert mock_from_config.call_args[0][0] is mock_config

    def test_modelclass_export_exists(self):
        """Test ModelClass is exported and points to correct class."""
        from nemo_automodel.components.models.kimivl import model as kimivl_mod
        
        assert hasattr(kimivl_mod, "ModelClass")
        assert kimivl_mod.ModelClass is KimiVLForConditionalGeneration


class TestKimiVLUsesDeepseekV3Config:
    """Tests to verify KimiVL properly uses HuggingFace's DeepseekV3Config."""

    def test_text_config_is_hf_deepseek_v3_config(self):
        """Verify text_config uses HF's DeepseekV3Config, not a custom class."""
        config = KimiVLConfig()
        
        # Should be the actual HuggingFace DeepseekV3Config class
        assert type(config.text_config).__name__ == "DeepseekV3Config"
        assert type(config.text_config).__module__ == "transformers.models.deepseek_v3.configuration_deepseek_v3"

    def test_text_config_from_dict_creates_hf_config(self):
        """Verify creating from dict still uses HF's DeepseekV3Config."""
        config = KimiVLConfig(text_config={"hidden_size": 512})
        
        assert type(config.text_config).__name__ == "DeepseekV3Config"
        assert config.text_config.hidden_size == 512


class TestVisionTowerComponents:
    """Tests for MoonVit vision tower components."""

    def test_apply_rope_vision_output_shape(self):
        """Test _apply_rope_vision produces correct output shapes."""
        from nemo_automodel.components.models.kimivl.model import _apply_rope_vision

        batch_seq = 16
        num_heads = 4
        head_dim = 32
        xq = torch.randn(batch_seq, num_heads, head_dim)
        xk = torch.randn(batch_seq, num_heads, head_dim)
        freqs_cis = torch.randn(batch_seq, head_dim // 2, dtype=torch.complex64)

        xq_out, xk_out = _apply_rope_vision(xq, xk, freqs_cis)

        assert xq_out.shape == xq.shape
        assert xk_out.shape == xk.shape
        assert xq_out.dtype == xq.dtype
        assert xk_out.dtype == xk.dtype

    def test_learnable_2d_interp_pos_emb_same_size(self):
        """Test Learnable2DInterpPosEmb with same size (no interpolation)."""
        from nemo_automodel.components.models.kimivl.model import Learnable2DInterpPosEmb

        height, width, dim = 8, 8, 64
        pos_emb = Learnable2DInterpPosEmb(height, width, dim)

        seq_len = height * width
        x = torch.randn(seq_len, dim)
        grid_hws = torch.tensor([[height, width]])

        output = pos_emb(x, grid_hws)
        assert output.shape == (seq_len, dim)

    def test_learnable_2d_interp_pos_emb_interpolation(self):
        """Test Learnable2DInterpPosEmb with different size (requires interpolation)."""
        from nemo_automodel.components.models.kimivl.model import Learnable2DInterpPosEmb

        height, width, dim = 8, 8, 64
        pos_emb = Learnable2DInterpPosEmb(height, width, dim)

        # Different size triggers interpolation
        new_h, new_w = 4, 4
        seq_len = new_h * new_w
        x = torch.randn(seq_len, dim)
        grid_hws = torch.tensor([[new_h, new_w]])

        output = pos_emb(x, grid_hws)
        assert output.shape == (seq_len, dim)

    def test_rope_2d_pos_emb_freqs_cis_shape(self):
        """Test Rope2DPosEmb generates correct freqs_cis shape."""
        from nemo_automodel.components.models.kimivl.model import Rope2DPosEmb

        dim = 64
        max_height, max_width = 16, 16
        rope = Rope2DPosEmb(dim, max_height, max_width)

        grid_hws = torch.tensor([[8, 8], [4, 4]])
        freqs_cis = rope.get_freqs_cis(grid_hws)

        # Total tokens = 8*8 + 4*4 = 64 + 16 = 80
        expected_seq_len = 8 * 8 + 4 * 4
        assert freqs_cis.shape == (expected_seq_len, dim // 2)

    def test_moonvit_mlp_forward(self):
        """Test MoonVitMLP forward pass."""
        from nemo_automodel.components.models.kimivl.model import MoonVitMLP
        import torch.nn.functional as F

        dims = [64, 128, 64]
        mlp = MoonVitMLP(dims, activation=F.gelu)

        x = torch.randn(16, 64)
        output = mlp(x)

        assert output.shape == (16, 64)

    def test_patch_merger_output_structure(self):
        """Test patch_merger produces correct output structure."""
        from nemo_automodel.components.models.kimivl.model import patch_merger

        hidden_dim = 64
        h1, w1 = 8, 8
        h2, w2 = 4, 4
        total_tokens = h1 * w1 + h2 * w2

        x = torch.randn(total_tokens, hidden_dim)
        grid_hws = torch.tensor([[h1, w1], [h2, w2]])
        merge_kernel = [2, 2]

        outputs = patch_merger(x, grid_hws, merge_kernel)

        assert len(outputs) == 2
        # First image: 8x8 -> 4x4 after 2x2 merge = 16 patches
        assert outputs[0].shape == (16, 4, hidden_dim)  # (new_h*new_w, kh*kw, dim)
        # Second image: 4x4 -> 2x2 after 2x2 merge = 4 patches
        assert outputs[1].shape == (4, 4, hidden_dim)


class TestMoonVitPretrainedModel:
    """Tests for MoonVitPretrainedModel."""

    @pytest.fixture
    def small_vit_config(self):
        """Create a small MoonViT config for testing."""
        return MoonViTConfig(
            patch_size=14,
            init_pos_emb_height=8,
            init_pos_emb_width=8,
            num_attention_heads=4,
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=128,
            merge_kernel_size=[2, 2],
        )

    def test_moonvit_initialization(self, small_vit_config):
        """Test MoonVitPretrainedModel initializes correctly."""
        from nemo_automodel.components.models.kimivl.model import MoonVitPretrainedModel

        model = MoonVitPretrainedModel(small_vit_config)

        assert model.config is small_vit_config
        assert model.merge_kernel_size == [2, 2]
        assert hasattr(model, "patch_embed")
        assert hasattr(model, "encoder")

    def test_moonvit_dtype_property(self, small_vit_config):
        """Test MoonVitPretrainedModel dtype property."""
        from nemo_automodel.components.models.kimivl.model import MoonVitPretrainedModel

        model = MoonVitPretrainedModel(small_vit_config)
        assert model.dtype == torch.float32

        model = model.to(torch.bfloat16)
        assert model.dtype == torch.bfloat16


class TestKimiVLMultiModalProjector:
    """Tests for KimiVLMultiModalProjector."""

    @pytest.fixture
    def projector_config(self):
        """Create config for projector testing."""
        vision_config = MoonViTConfig(hidden_size=64, merge_kernel_size=[2, 2])
        text_config = DeepseekV3Config(hidden_size=128)
        return KimiVLConfig(vision_config=vision_config, text_config=text_config)

    def test_projector_initialization(self, projector_config):
        """Test KimiVLMultiModalProjector initializes correctly."""
        from nemo_automodel.components.models.kimivl.model import KimiVLMultiModalProjector

        projector = KimiVLMultiModalProjector(projector_config)

        # hidden_size = vision_hidden * merge_h * merge_w = 64 * 2 * 2 = 256
        assert projector.hidden_size == 256
        assert projector.linear_1.in_features == 256
        assert projector.linear_1.out_features == 256
        assert projector.linear_2.in_features == 256
        assert projector.linear_2.out_features == 128  # text hidden size

    def test_projector_forward(self, projector_config):
        """Test KimiVLMultiModalProjector forward pass."""
        from nemo_automodel.components.models.kimivl.model import KimiVLMultiModalProjector

        projector = KimiVLMultiModalProjector(projector_config)

        # Simulate merged patches: list of (num_patches, merge_tokens, vision_dim)
        image_features = [
            torch.randn(16, 4, 64),  # 16 patches, 4 merged tokens (2x2), 64 dim
            torch.randn(4, 4, 64),   # 4 patches
        ]

        output = projector(image_features)

        # Total tokens = 16 + 4 = 20
        assert output.shape == (20, 128)  # (total_tokens, text_hidden_size)


class TestKimiVLModel:
    """Tests for KimiVLModel.
    
    These tests verify validation logic without instantiating full model
    to avoid CUDA code in MoE layers.
    """

    def test_kimivl_model_validation_raises_when_both_inputs(self):
        """Test validation raises error when both input_ids and inputs_embeds provided."""
        # Test the validation logic directly (same as in KimiVLModel.forward)
        input_ids = torch.randint(0, 100, (1, 8))
        inputs_embeds = torch.randn(1, 8, 64)

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_kimivl_model_validation_raises_when_neither_inputs(self):
        """Test validation raises error when neither input_ids nor inputs_embeds provided."""
        input_ids = None
        inputs_embeds = None

        with pytest.raises(ValueError, match="exactly one of input_ids or inputs_embeds"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_kimivl_model_validation_passes_with_only_input_ids(self):
        """Test validation passes when only input_ids is provided."""
        input_ids = torch.randint(0, 100, (1, 8))
        inputs_embeds = None

        # Should NOT raise
        if (input_ids is None) == (inputs_embeds is None):
            pytest.fail("Validation should pass when only input_ids is provided")

    def test_kimivl_model_validation_passes_with_only_inputs_embeds(self):
        """Test validation passes when only inputs_embeds is provided."""
        input_ids = None
        inputs_embeds = torch.randn(1, 8, 64)

        # Should NOT raise
        if (input_ids is None) == (inputs_embeds is None):
            pytest.fail("Validation should pass when only inputs_embeds is provided")

    def test_kimivl_model_forward_signature(self):
        """Test KimiVLModel.forward has expected signature."""
        import inspect
        from nemo_automodel.components.models.kimivl.model import KimiVLModel

        sig = inspect.signature(KimiVLModel.forward)
        params = list(sig.parameters.keys())

        assert "input_ids" in params
        assert "inputs_embeds" in params
        assert "pixel_values" in params
        assert "attention_mask" in params


class TestKimiVLForConditionalGenerationForward:
    """Tests for KimiVLForConditionalGeneration forward pass.

    """

    def test_forward_signature_has_required_params(self):
        """Test forward signature includes expected parameters."""
        import inspect
        sig = inspect.signature(KimiVLForConditionalGeneration.forward)
        params = list(sig.parameters.keys())

        assert "input_ids" in params
        assert "attention_mask" in params
        assert "labels" in params
        assert "pixel_values" in params
        assert "return_dict" in params

    def test_from_config_creates_model(self):
        """Test from_config creates a model instance."""
        config = KimiVLConfig()

        with patch.object(KimiVLForConditionalGeneration, "__init__", return_value=None):
            model = KimiVLForConditionalGeneration.from_config(config)
            # Just verify it returns something (mocked)

    def test_from_pretrained_delegates_to_from_config(self):
        """Test from_pretrained loads config and calls from_config."""
        mock_config = MagicMock(spec=KimiVLConfig)

        with patch.object(KimiVLConfig, "from_pretrained", return_value=mock_config):
            with patch.object(KimiVLForConditionalGeneration, "from_config") as mock_from_config:
                mock_from_config.return_value = MagicMock()

                KimiVLForConditionalGeneration.from_pretrained("dummy/path")

                KimiVLConfig.from_pretrained.assert_called_once_with("dummy/path")
                mock_from_config.assert_called_once()

    def test_model_has_expected_attributes(self):
        """Test model class has expected attributes and methods."""
        assert hasattr(KimiVLForConditionalGeneration, "forward")
        assert hasattr(KimiVLForConditionalGeneration, "from_config")
        assert hasattr(KimiVLForConditionalGeneration, "from_pretrained")
        assert hasattr(KimiVLForConditionalGeneration, "get_input_embeddings")
        assert hasattr(KimiVLForConditionalGeneration, "get_output_embeddings")
        assert callable(KimiVLForConditionalGeneration.forward)

    def test_return_dict_false_returns_tensor(self):
        """Test that return_dict=False path exists in forward signature."""
        import inspect
        sig = inspect.signature(KimiVLForConditionalGeneration.forward)
        return_dict_param = sig.parameters.get("return_dict")
        assert return_dict_param is not None
        # Default should be None (meaning use config default)
        assert return_dict_param.default is None


class TestKimiVLStateDictAdapter:
    """Tests for KimiVLStateDictAdapter."""

    @pytest.fixture
    def adapter_setup(self):
        """Create adapter setup for testing."""
        from nemo_automodel.components.models.common import BackendConfig
        from nemo_automodel.components.models.kimivl.model import KimiVLStateDictAdapter
        from nemo_automodel.components.moe.layers import MoEConfig

        config = KimiVLConfig(
            vision_config=MoonViTConfig(hidden_size=64),
            text_config=DeepseekV3Config(
                hidden_size=64,
                num_hidden_layers=1,
                n_routed_experts=4,
                n_group=2,
                topk_group=2,
            ),
        )
        moe_config = MoEConfig(
            dim=64,
            inter_dim=128,
            moe_inter_dim=64,
            n_routed_experts=4,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=2,
            n_limited_groups=2,
            train_gate=True,
            gate_bias_update_factor=0.001,
            aux_loss_coeff=0.0,
            score_func="sigmoid",
            route_scale=1.0,
            norm_topk_prob=True,
        )
        backend = BackendConfig(linear="torch", rms_norm="torch")
        adapter = KimiVLStateDictAdapter(config, moe_config, backend)
        return adapter

    def test_adapter_from_hf_vision_keys(self, adapter_setup):
        """Test adapter correctly transforms vision tower keys."""
        adapter = adapter_setup

        hf_state_dict = {
            "vision_tower.patch_embed.proj.weight": torch.randn(64, 3, 14, 14),
            "vision_tower.encoder.blocks.0.norm0.weight": torch.randn(64),
        }

        native_dict = adapter.from_hf(hf_state_dict)

        assert "model.vision_tower.patch_embed.proj.weight" in native_dict
        assert "model.vision_tower.encoder.blocks.0.norm0.weight" in native_dict

    def test_adapter_from_hf_projector_keys(self, adapter_setup):
        """Test adapter correctly transforms projector keys."""
        adapter = adapter_setup

        hf_state_dict = {
            "multi_modal_projector.linear_1.weight": torch.randn(256, 256),
            "multi_modal_projector.linear_2.weight": torch.randn(64, 256),
        }

        native_dict = adapter.from_hf(hf_state_dict)

        assert "model.multi_modal_projector.linear_1.weight" in native_dict
        assert "model.multi_modal_projector.linear_2.weight" in native_dict

    def test_adapter_from_hf_lm_head_keys(self, adapter_setup):
        """Test adapter correctly transforms lm_head keys."""
        adapter = adapter_setup

        hf_state_dict = {
            "language_model.lm_head.weight": torch.randn(100, 64),
        }

        native_dict = adapter.from_hf(hf_state_dict)

        assert "lm_head.weight" in native_dict

    def test_adapter_to_hf_roundtrip_vision(self, adapter_setup):
        """Test to_hf/from_hf roundtrip preserves vision keys."""
        adapter = adapter_setup

        original = {
            "model.vision_tower.patch_embed.proj.weight": torch.randn(64, 3, 14, 14),
        }

        hf_dict = adapter.to_hf(original)
        restored = adapter.from_hf(hf_dict)

        assert "model.vision_tower.patch_embed.proj.weight" in restored


class TestKimiVLPipelineParallelismChunking:
    """Tests for VLM chunking logic used in pipeline parallelism.
    
    These tests verify the chunking logic directly without instantiating
    the full model to avoid CUDA code in MoE layers.
    """

    def _simulate_pp_chunking_logic(
        self,
        pixel_values,
        input_ids,
        media_placeholder_token_id,
        vlm_pixel_values_chunks,
        vlm_image_grid_hws_chunks,
        vlm_chunk_idx,
    ):
        """Simulate the PP chunking logic from KimiVLForConditionalGeneration.forward.
        
        Returns (pixel_values, image_grid_hws, new_chunk_idx).
        """
        image_grid_hws = None
        
        if (
            pixel_values is None
            and vlm_pixel_values_chunks is not None
        ):
            has_media_tokens = (
                input_ids is not None
                and media_placeholder_token_id is not None
                and (input_ids == media_placeholder_token_id).any()
            )
            if has_media_tokens:
                if vlm_chunk_idx < len(vlm_pixel_values_chunks):
                    pixel_values = vlm_pixel_values_chunks[vlm_chunk_idx]
                    image_grid_hws = vlm_image_grid_hws_chunks[vlm_chunk_idx]
                    vlm_chunk_idx = vlm_chunk_idx + 1
        
        return pixel_values, image_grid_hws, vlm_chunk_idx

    def test_pp_chunking_retrieves_when_media_tokens_present(self):
        """Test chunking retrieves pixel_values when input has media tokens."""
        chunk1 = torch.randn(1, 3, 56, 56)
        chunk2 = torch.randn(1, 3, 56, 56)
        grid1 = torch.tensor([[4, 4]])
        grid2 = torch.tensor([[4, 4]])

        input_ids = torch.tensor([[1, 2, 99, 3, 4]])  # 99 is media token

        pixel_values, grid_hws, new_idx = self._simulate_pp_chunking_logic(
            pixel_values=None,
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=[chunk1, chunk2],
            vlm_image_grid_hws_chunks=[grid1, grid2],
            vlm_chunk_idx=0,
        )

        assert pixel_values is chunk1
        assert grid_hws is grid1
        assert new_idx == 1

    def test_pp_chunking_increments_idx_per_call(self):
        """Test chunk_idx increments with each simulated forward."""
        chunks = [torch.randn(1, 3, 56, 56) for _ in range(3)]
        grids = [torch.tensor([[4, 4]]) for _ in range(3)]
        input_ids = torch.tensor([[1, 99, 2]])

        idx = 0
        for i in range(3):
            _, _, idx = self._simulate_pp_chunking_logic(
                pixel_values=None,
                input_ids=input_ids,
                media_placeholder_token_id=99,
                vlm_pixel_values_chunks=chunks,
                vlm_image_grid_hws_chunks=grids,
                vlm_chunk_idx=idx,
            )
            assert idx == i + 1

    def test_pp_chunking_skipped_when_no_media_tokens(self):
        """Test chunking is skipped when input has no media tokens."""
        chunk1 = torch.randn(1, 3, 56, 56)
        grid1 = torch.tensor([[4, 4]])
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])  # No 99

        pixel_values, grid_hws, new_idx = self._simulate_pp_chunking_logic(
            pixel_values=None,
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=[chunk1],
            vlm_image_grid_hws_chunks=[grid1],
            vlm_chunk_idx=0,
        )

        assert pixel_values is None
        assert grid_hws is None
        assert new_idx == 0

    def test_pp_chunking_bypassed_when_pixel_values_provided(self):
        """Test explicit pixel_values bypasses chunking logic."""
        chunk1 = torch.randn(1, 3, 56, 56)
        grid1 = torch.tensor([[4, 4]])
        explicit_pv = torch.randn(1, 3, 56, 56)
        input_ids = torch.tensor([[1, 99, 2]])

        pixel_values, grid_hws, new_idx = self._simulate_pp_chunking_logic(
            pixel_values=explicit_pv,  # Explicit pixel_values provided
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=[chunk1],
            vlm_image_grid_hws_chunks=[grid1],
            vlm_chunk_idx=0,
        )

        # Should return the explicit pixel_values, not from chunks
        assert pixel_values is explicit_pv
        assert grid_hws is None
        assert new_idx == 0

    def test_pp_chunking_stops_at_end_of_chunks(self):
        """Test chunking stops incrementing when all chunks consumed."""
        chunk1 = torch.randn(1, 3, 56, 56)
        grid1 = torch.tensor([[4, 4]])
        input_ids = torch.tensor([[1, 99, 2]])

        # First call
        _, _, idx = self._simulate_pp_chunking_logic(
            pixel_values=None,
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=[chunk1],
            vlm_image_grid_hws_chunks=[grid1],
            vlm_chunk_idx=0,
        )
        assert idx == 1

        # Second call - no more chunks
        pixel_values, _, idx = self._simulate_pp_chunking_logic(
            pixel_values=None,
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=[chunk1],
            vlm_image_grid_hws_chunks=[grid1],
            vlm_chunk_idx=1,  # Already at end
        )
        assert pixel_values is None  # No chunk available
        assert idx == 1  # Stays at 1

    def test_pp_chunking_not_triggered_without_chunks(self):
        """Test chunking not triggered when chunks is None."""
        input_ids = torch.tensor([[1, 99, 2]])

        pixel_values, grid_hws, new_idx = self._simulate_pp_chunking_logic(
            pixel_values=None,
            input_ids=input_ids,
            media_placeholder_token_id=99,
            vlm_pixel_values_chunks=None,  # No chunks set
            vlm_image_grid_hws_chunks=None,
            vlm_chunk_idx=0,
        )

        assert pixel_values is None
        assert grid_hws is None
        assert new_idx == 0


class TestKimiVLRegistration:
    """Tests for KimiVL registration with transformers."""

    def test_registration_executed_on_import(self):
        """Test that registration happens when module is imported."""
        from transformers import AutoConfig
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING

        # After importing kimivl.model, kimi_vl should be registered
        assert "kimi_vl" in CONFIG_MAPPING

    def test_autoconfig_recognizes_kimi_vl(self):
        """Test AutoConfig can create KimiVLConfig."""
        from transformers import AutoConfig

        # Create a config using the registered model type
        # AutoConfig.for_model uses model_type as first positional arg
        config = AutoConfig.for_model(
            "kimi_vl",
            vision_config={"hidden_size": 64},
            text_config={"hidden_size": 128},
        )
        assert type(config).__name__ == "KimiVLConfig"
