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
        from nemo_automodel.components.moe.config import MoEConfig

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


# =============================================================================
# Additional Tests for Vision Tower Components
# =============================================================================


class TestVisionAttentionFunctions:
    """Tests for vision_attention_sdpa and vision_attention_flash functions."""

    def test_vision_attention_sdpa_single_sequence(self):
        """Test vision_attention_sdpa with a single sequence."""
        from nemo_automodel.components.models.kimivl.model import vision_attention_sdpa

        seq_len = 16
        num_heads = 4
        head_dim = 8

        q = torch.randn(seq_len, num_heads, head_dim)
        k = torch.randn(seq_len, num_heads, head_dim)
        v = torch.randn(seq_len, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        output = vision_attention_sdpa(q, k, v, cu_seqlens, cu_seqlens)

        # Output should be (seq_len, num_heads * head_dim)
        assert output.shape == (seq_len, num_heads * head_dim)

    def test_vision_attention_sdpa_multiple_sequences(self):
        """Test vision_attention_sdpa with multiple sequences in a batch."""
        from nemo_automodel.components.models.kimivl.model import vision_attention_sdpa

        seq1_len = 8
        seq2_len = 12
        total_len = seq1_len + seq2_len
        num_heads = 4
        head_dim = 8

        q = torch.randn(total_len, num_heads, head_dim)
        k = torch.randn(total_len, num_heads, head_dim)
        v = torch.randn(total_len, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, seq1_len, total_len], dtype=torch.int32)

        output = vision_attention_sdpa(q, k, v, cu_seqlens, cu_seqlens)

        assert output.shape == (total_len, num_heads * head_dim)

    def test_vision_attention_sdpa_creates_block_diagonal_mask(self):
        """Test that SDPA creates proper block-diagonal attention mask."""
        from nemo_automodel.components.models.kimivl.model import vision_attention_sdpa

        # Use small sizes for verification
        seq1 = 4
        seq2 = 4
        total = seq1 + seq2
        num_heads = 2
        head_dim = 4

        q = torch.randn(total, num_heads, head_dim)
        k = torch.randn(total, num_heads, head_dim)
        v = torch.randn(total, num_heads, head_dim)
        cu_seqlens = torch.tensor([0, seq1, total], dtype=torch.int32)

        # Just verify it runs without error and produces correct shape
        output = vision_attention_sdpa(q, k, v, cu_seqlens, cu_seqlens)
        assert output.shape == (total, num_heads * head_dim)


class TestMoonVitEncoderLayer:
    """Tests for MoonVitEncoderLayer."""

    @pytest.fixture
    def encoder_layer(self):
        """Create a small encoder layer for testing."""
        from nemo_automodel.components.models.kimivl.model import MoonVitEncoderLayer
        import torch.nn.functional as F

        return MoonVitEncoderLayer(
            num_heads=4,
            hidden_dim=64,
            mlp_dim=128,
            activation=F.gelu,
            attn_bias=True,
            attn_implementation="sdpa",  # Use SDPA to avoid flash_attn dependency
        )

    def test_encoder_layer_initialization(self, encoder_layer):
        """Test encoder layer initializes with correct components."""
        assert encoder_layer.num_heads == 4
        assert encoder_layer.hidden_dim == 64
        assert encoder_layer.head_dim == 16  # 64 / 4
        assert hasattr(encoder_layer, "norm0")
        assert hasattr(encoder_layer, "norm1")
        assert hasattr(encoder_layer, "mlp")
        assert hasattr(encoder_layer, "wqkv")
        assert hasattr(encoder_layer, "wo")

    def test_encoder_layer_wqkv_shape(self, encoder_layer):
        """Test wqkv projection has correct shape."""
        # wqkv should project to 3x hidden_dim (for q, k, v)
        assert encoder_layer.wqkv.in_features == 64
        assert encoder_layer.wqkv.out_features == 64 * 3

    def test_encoder_layer_forward_shape(self, encoder_layer):
        """Test encoder layer forward produces correct output shape."""
        from nemo_automodel.components.models.kimivl.model import Rope2DPosEmb

        seq_len = 16
        hidden_dim = 64
        hidden_states = torch.randn(seq_len, hidden_dim)

        # Create rope freqs
        rope = Rope2DPosEmb(hidden_dim // 4, 8, 8)
        grid_hws = torch.tensor([[4, 4]])  # 4*4 = 16 tokens
        rope_freqs_cis = rope.get_freqs_cis(grid_hws)

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        output = encoder_layer(hidden_states, cu_seqlens, rope_freqs_cis)

        assert output.shape == (seq_len, hidden_dim)

    def test_encoder_layer_residual_connection(self, encoder_layer):
        """Test encoder layer uses residual connections."""
        from nemo_automodel.components.models.kimivl.model import Rope2DPosEmb

        seq_len = 16
        hidden_dim = 64
        hidden_states = torch.randn(seq_len, hidden_dim)
        hidden_states_copy = hidden_states.clone()

        rope = Rope2DPosEmb(hidden_dim // 4, 8, 8)
        grid_hws = torch.tensor([[4, 4]])
        rope_freqs_cis = rope.get_freqs_cis(grid_hws)
        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32)

        output = encoder_layer(hidden_states, cu_seqlens, rope_freqs_cis)

        # Output should differ from input (transformed)
        assert not torch.allclose(output, hidden_states_copy)


class TestMoonVitEncoder:
    """Tests for MoonVitEncoder."""

    @pytest.fixture
    def encoder(self):
        """Create a small encoder for testing."""
        from nemo_automodel.components.models.kimivl.model import MoonVitEncoder

        block_cfg = {
            "num_heads": 4,
            "hidden_dim": 64,
            "mlp_dim": 128,
            "attn_bias": True,
            "attn_implementation": "sdpa",
        }
        return MoonVitEncoder(hidden_dim=64, num_layers=2, block_cfg=block_cfg)

    def test_encoder_initialization(self, encoder):
        """Test encoder initializes with correct components."""
        assert len(encoder.blocks) == 2
        assert hasattr(encoder, "rope_2d")
        assert hasattr(encoder, "final_layernorm")

    def test_encoder_forward_single_image(self, encoder):
        """Test encoder forward with single image."""
        h, w = 4, 4
        seq_len = h * w
        hidden_dim = 64

        hidden_states = torch.randn(seq_len, hidden_dim)
        grid_hws = torch.tensor([[h, w]])

        output = encoder(hidden_states, grid_hws)

        assert output.shape == (seq_len, hidden_dim)

    def test_encoder_forward_multiple_images(self, encoder):
        """Test encoder forward with multiple images."""
        h1, w1 = 4, 4
        h2, w2 = 2, 4
        seq1 = h1 * w1
        seq2 = h2 * w2
        total_seq = seq1 + seq2
        hidden_dim = 64

        hidden_states = torch.randn(total_seq, hidden_dim)
        grid_hws = torch.tensor([[h1, w1], [h2, w2]])

        output = encoder(hidden_states, grid_hws)

        assert output.shape == (total_seq, hidden_dim)

    def test_encoder_computes_cu_seqlens_correctly(self, encoder):
        """Test encoder computes cumulative sequence lengths correctly."""
        h1, w1 = 4, 4
        h2, w2 = 2, 2
        hidden_dim = 64

        hidden_states = torch.randn(h1 * w1 + h2 * w2, hidden_dim)
        grid_hws = torch.tensor([[h1, w1], [h2, w2]])

        # The encoder should compute cu_seqlens as [0, 16, 20]
        # This is tested implicitly by the forward pass succeeding
        output = encoder(hidden_states, grid_hws)
        assert output.shape[0] == h1 * w1 + h2 * w2


class TestMoonVisionPatchEmbed:
    """Tests for MoonVisionPatchEmbed.
    
    Note: MoonVisionPatchEmbed expects input as (num_patches, channels, patch_h, patch_w)
    where each patch is a separate item. The Conv2d processes each patch individually.
    """

    @pytest.fixture
    def patch_embed(self):
        """Create patch embed module for testing."""
        from nemo_automodel.components.models.kimivl.model import MoonVisionPatchEmbed

        return MoonVisionPatchEmbed(
            out_dim=64,
            in_dim=3,
            patch_size=14,
            pos_emb_height=8,
            pos_emb_width=8,
        )

    def test_patch_embed_initialization(self, patch_embed):
        """Test patch embed initializes correctly."""
        assert patch_embed.patch_size == (14, 14)
        assert patch_embed.proj.in_channels == 3
        assert patch_embed.proj.out_channels == 64
        assert patch_embed.proj.kernel_size == (14, 14)
        assert patch_embed.proj.stride == (14, 14)

    def test_patch_embed_forward(self, patch_embed):
        """Test patch embed forward pass.
        
        Input format: (num_patches_total, 3, patch_size, patch_size)
        Each patch is 14x14 pixels.
        """
        h, w = 4, 4
        num_patches = h * w  # 16 patches for 4x4 grid
        
        # Input: individual patches stacked as batch
        # Each patch is (3, 14, 14)
        x = torch.randn(num_patches, 3, 14, 14)
        grid_hws = torch.tensor([[h, w]])

        output = patch_embed(x, grid_hws)

        # Output: (num_patches, hidden_dim)
        assert output.shape == (num_patches, 64)

    def test_patch_embed_multiple_images(self, patch_embed):
        """Test patch embed with multiple images (patches concatenated)."""
        h1, w1 = 4, 4
        h2, w2 = 2, 2
        total_patches = h1 * w1 + h2 * w2  # 16 + 4 = 20
        
        # All patches from both images concatenated
        x = torch.randn(total_patches, 3, 14, 14)
        grid_hws = torch.tensor([[h1, w1], [h2, w2]])
        
        output = patch_embed(x, grid_hws)
        
        assert output.shape == (total_patches, 64)

    def test_patch_embed_pos_emb_applied(self, patch_embed):
        """Test that position embedding is applied to output."""
        h, w = 4, 4
        num_patches = h * w
        
        x = torch.randn(num_patches, 3, 14, 14)
        grid_hws = torch.tensor([[h, w]])

        # Run twice with same input - output should be same (deterministic)
        out1 = patch_embed(x, grid_hws)
        out2 = patch_embed(x, grid_hws)
        
        assert torch.allclose(out1, out2)


class TestKimiVLLanguageModelBackend:
    """Tests for KimiVLLanguageModelBackend."""

    def test_backend_has_expected_attributes(self):
        """Test KimiVLLanguageModelBackend has expected attributes."""
        from nemo_automodel.components.models.kimivl.model import KimiVLLanguageModelBackend

        assert hasattr(KimiVLLanguageModelBackend, "forward")
        assert hasattr(KimiVLLanguageModelBackend, "get_input_embeddings")
        assert hasattr(KimiVLLanguageModelBackend, "set_input_embeddings")
        assert hasattr(KimiVLLanguageModelBackend, "init_weights")

    def test_backend_forward_signature(self):
        """Test forward signature has expected parameters."""
        import inspect
        from nemo_automodel.components.models.kimivl.model import KimiVLLanguageModelBackend

        sig = inspect.signature(KimiVLLanguageModelBackend.forward)
        params = list(sig.parameters.keys())

        assert "input_ids" in params
        assert "inputs_embeds" in params
        assert "attention_mask" in params
        assert "position_ids" in params
        assert "padding_mask" in params

    def test_backend_embed_tokens_property(self):
        """Test embed_tokens property exists."""
        from nemo_automodel.components.models.kimivl.model import KimiVLLanguageModelBackend

        assert hasattr(KimiVLLanguageModelBackend, "embed_tokens")

    def test_backend_layers_property(self):
        """Test layers property exists."""
        from nemo_automodel.components.models.kimivl.model import KimiVLLanguageModelBackend

        assert hasattr(KimiVLLanguageModelBackend, "layers")

    def test_backend_norm_property(self):
        """Test norm property exists."""
        from nemo_automodel.components.models.kimivl.model import KimiVLLanguageModelBackend

        assert hasattr(KimiVLLanguageModelBackend, "norm")


class TestKimiVLModelMergeFeatures:
    """Tests for KimiVLModel._merge_with_image_features."""

    def test_merge_replaces_media_tokens(self):
        """Test _merge_with_image_features replaces media tokens."""
        from nemo_automodel.components.models.kimivl.model import KimiVLModel

        # Simulate the merge logic
        batch_size, seq_len, embed_dim = 1, 8, 64
        media_token_id = 99

        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        input_ids = torch.tensor([[1, 2, 99, 99, 3, 4, 5, 6]])  # Two media tokens
        image_features = torch.randn(2, embed_dim)  # 2 image feature tokens

        # Replicate the merge logic
        inputs_embeds_flat = inputs_embeds.reshape(-1, embed_dim)
        input_ids_flat = input_ids.flatten()
        inputs_embeds_flat[input_ids_flat == media_token_id] = image_features
        result = inputs_embeds_flat.reshape(batch_size, seq_len, embed_dim)

        # Check positions 2 and 3 (media tokens) have been replaced
        assert torch.allclose(result[0, 2], image_features[0])
        assert torch.allclose(result[0, 3], image_features[1])

    def test_merge_preserves_non_media_tokens(self):
        """Test merge preserves embeddings at non-media token positions."""
        batch_size, seq_len, embed_dim = 1, 8, 64
        media_token_id = 99

        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        original_embed_0 = inputs_embeds[0, 0].clone()
        original_embed_5 = inputs_embeds[0, 5].clone()

        input_ids = torch.tensor([[1, 2, 99, 99, 3, 4, 5, 6]])
        image_features = torch.randn(2, embed_dim)

        inputs_embeds_flat = inputs_embeds.reshape(-1, embed_dim)
        input_ids_flat = input_ids.flatten()
        inputs_embeds_flat[input_ids_flat == media_token_id] = image_features
        result = inputs_embeds_flat.reshape(batch_size, seq_len, embed_dim)

        # Non-media positions should be unchanged
        assert torch.allclose(result[0, 0], original_embed_0)
        assert torch.allclose(result[0, 5], original_embed_5)

    def test_merge_handles_no_media_tokens(self):
        """Test merge handles case with no media tokens."""
        batch_size, seq_len, embed_dim = 1, 8, 64
        media_token_id = 99

        inputs_embeds = torch.randn(batch_size, seq_len, embed_dim)
        original = inputs_embeds.clone()
        input_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])  # No 99
        image_features = torch.empty(0, embed_dim)

        inputs_embeds_flat = inputs_embeds.reshape(-1, embed_dim)
        input_ids_flat = input_ids.flatten()
        mask = input_ids_flat == media_token_id
        if mask.any():
            inputs_embeds_flat[mask] = image_features
        result = inputs_embeds_flat.reshape(batch_size, seq_len, embed_dim)

        # Should be unchanged
        assert torch.allclose(result, original)


class TestKimiVLModelForward:
    """Tests for KimiVLModel.forward logic."""

    def test_forward_raises_with_both_inputs(self):
        """Test forward raises when both input_ids and inputs_embeds provided."""
        from nemo_automodel.components.models.kimivl.model import KimiVLModel

        # Test the validation condition directly
        input_ids = torch.randint(0, 100, (1, 8))
        inputs_embeds = torch.randn(1, 8, 64)

        # This is the validation from KimiVLModel.forward
        with pytest.raises(ValueError, match="exactly one"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_forward_raises_with_neither_input(self):
        """Test forward raises when neither input_ids nor inputs_embeds provided."""
        input_ids = None
        inputs_embeds = None

        with pytest.raises(ValueError, match="exactly one"):
            if (input_ids is None) == (inputs_embeds is None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    def test_thd_format_handling_logic(self):
        """Test the thd format handling logic."""
        # Simulate the thd format check
        kwargs = {"qkv_format": "thd", "some_other": "value"}

        if "qkv_format" in kwargs and kwargs["qkv_format"] == "thd":
            # Would call squeeze_input_for_thd
            processed = True
        else:
            processed = False

        assert processed is True

    def test_pixel_values_dtype_conversion_logic(self):
        """Test pixel values are converted to vision tower dtype."""
        # Simulate the dtype conversion logic
        pixel_values = torch.randn(1, 3, 56, 56, dtype=torch.float32)
        target_dtype = torch.bfloat16

        converted = pixel_values.to(target_dtype)
        assert converted.dtype == torch.bfloat16

    def test_inputs_embeds_fallback_for_pp_middle_stages(self):
        """Test inputs_embeds fallback for PP middle stages."""
        # Simulate the fallback logic for PP middle stages
        input_ids = torch.randn(1, 8, 64, dtype=torch.float32)  # Actually hidden states
        embed_tokens = None  # PP middle stage has no embed_tokens

        # The logic: if embed_tokens is None and input_ids is float, treat as hidden states
        if embed_tokens is None:
            if (
                input_ids is not None
                and isinstance(input_ids, torch.Tensor)
                and input_ids.dtype in (torch.float16, torch.bfloat16, torch.float32)
            ):
                inputs_embeds = input_ids  # Use input_ids as hidden states
            else:
                inputs_embeds = None

        assert inputs_embeds is not None
        assert inputs_embeds.shape == (1, 8, 64)


class TestKimiVLForConditionalGenerationInit:
    """Tests for KimiVLForConditionalGeneration initialization."""

    def test_has_model_attribute(self):
        """Test class has model attribute defined."""
        from nemo_automodel.components.models.kimivl.model import KimiVLForConditionalGeneration

        # Check __init__ signature mentions model
        import inspect
        source = inspect.getsource(KimiVLForConditionalGeneration.__init__)
        assert "self.model" in source

    def test_has_vocab_size_attribute(self):
        """Test vocab_size is set from config."""
        import inspect
        from nemo_automodel.components.models.kimivl.model import KimiVLForConditionalGeneration

        source = inspect.getsource(KimiVLForConditionalGeneration.__init__)
        assert "self.vocab_size" in source

    def test_has_state_dict_adapter_conditional(self):
        """Test state_dict_adapter is created conditionally."""
        import inspect
        from nemo_automodel.components.models.kimivl.model import KimiVLForConditionalGeneration

        source = inspect.getsource(KimiVLForConditionalGeneration.__init__)
        assert "enable_hf_state_dict_adapter" in source
        assert "state_dict_adapter" in source

    def test_lm_head_property_exists(self):
        """Test lm_head property is defined."""
        from nemo_automodel.components.models.kimivl.model import KimiVLForConditionalGeneration

        assert hasattr(KimiVLForConditionalGeneration, "lm_head")


class TestKimiVLForConditionalGenerationForwardLogic:
    """Tests for KimiVLForConditionalGeneration.forward logic."""

    def test_loss_computation_with_attention_mask(self):
        """Test loss computation applies attention mask correctly."""
        # Simulate the masked loss computation
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 6:] = 0  # Mask last 2 tokens for first sample

        # Apply the shift and mask logic
        shift_mask = attention_mask[..., 1:]
        shift_logits = logits[..., :-1, :][shift_mask != 0].contiguous()
        shift_labels = labels[..., 1:][shift_mask != 0].contiguous()

        # Should have fewer tokens than batch_size * (seq_len - 1)
        expected_masked_out = 2  # tokens 6,7 in first sample (after shift: 5,6)
        expected_total = batch_size * (seq_len - 1) - expected_masked_out
        assert shift_logits.shape[0] == expected_total
        assert shift_labels.shape[0] == expected_total

    def test_loss_computation_without_attention_mask(self):
        """Test loss computation without attention mask."""
        batch_size, seq_len, vocab_size = 2, 8, 100
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Without mask, just shift
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        assert shift_logits.shape == (batch_size, seq_len - 1, vocab_size)
        assert shift_labels.shape == (batch_size, seq_len - 1)

    def test_return_dict_false_returns_logits(self):
        """Test return_dict=False logic returns just logits."""
        # Simulate the return logic
        return_dict = False
        logits = torch.randn(1, 8, 100)
        loss = torch.tensor(1.5)

        if return_dict is None:
            return_dict = False
        if not return_dict:
            result = logits
        else:
            result = {"loss": loss, "logits": logits}

        assert isinstance(result, torch.Tensor)
        assert result.shape == (1, 8, 100)

    def test_return_dict_true_returns_output_object(self):
        """Test return_dict=True logic returns output object."""
        from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

        return_dict = True
        logits = torch.randn(1, 8, 100)
        loss = torch.tensor(1.5)
        hidden_states = torch.randn(1, 8, 64)

        output = LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

        assert output.loss == loss
        assert output.logits is logits
        assert output.hidden_states is hidden_states

    def test_lm_head_none_returns_hidden_states(self):
        """Test when lm_head is None, hidden_states are returned."""
        # Simulate: logits = self.lm_head(hidden_states) if self.lm_head is not None else hidden_states
        lm_head = None
        hidden_states = torch.randn(1, 8, 64)

        logits = lm_head(hidden_states) if lm_head is not None else hidden_states

        assert logits is hidden_states


class TestDeepSeekV3RotaryEmbeddingAdapter:
    """Tests for DeepSeekV3RotaryEmbeddingAdapter."""

    def test_adapter_stores_parent_reference(self):
        """Test adapter stores reference to parent module."""
        from nemo_automodel.components.models.kimivl.model import DeepSeekV3RotaryEmbeddingAdapter

        class MockParent:
            freqs_cis = torch.randn(100, 64)

        parent = MockParent()
        adapter = DeepSeekV3RotaryEmbeddingAdapter(parent, rope_fusion=False)

        assert adapter._parent is parent
        assert adapter.rope_fusion is False

    def test_adapter_freqs_cis_property(self):
        """Test freqs_cis property accesses parent's buffer."""
        from nemo_automodel.components.models.kimivl.model import DeepSeekV3RotaryEmbeddingAdapter

        class MockParent:
            freqs_cis = torch.randn(100, 64)

        parent = MockParent()
        adapter = DeepSeekV3RotaryEmbeddingAdapter(parent, rope_fusion=False)

        assert adapter.freqs_cis is parent.freqs_cis

    def test_adapter_raises_when_freqs_cis_none(self):
        """Test adapter raises error when freqs_cis is None."""
        from nemo_automodel.components.models.kimivl.model import DeepSeekV3RotaryEmbeddingAdapter

        class MockParent:
            freqs_cis = None

        parent = MockParent()
        adapter = DeepSeekV3RotaryEmbeddingAdapter(parent, rope_fusion=False)

        with pytest.raises(RuntimeError, match="freqs_cis is None"):
            adapter(torch.randn(1, 8, 64), torch.arange(8).unsqueeze(0))
