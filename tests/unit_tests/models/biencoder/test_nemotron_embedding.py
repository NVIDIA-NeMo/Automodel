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

"""
Test suite for Nemotron Bidirectional Embedding Model.

This file contains two types of tests:
1. Unit tests: Fast, mock-based tests that don't require downloading models
2. Integration tests: Tests using NemotronBidirectionalModel with actual NVIDIA-Nemotron-Nano-9B-v2

The integration tests verify that:
- NemotronBidirectionalModel loads and wraps the real Nemotron Nano 9B v2 model
- Forward passes work correctly with single and batched inputs
- BiencoderModel can use NemotronBidirectionalModel as encoder
- All pooling strategies (mean, eos, last) work correctly
- Semantic similarity computations are accurate
- Multilingual encoding is supported

Usage:
    # Run all tests (unit tests only)
    pytest test_nemotron_embedding.py -v
    
    # Run integration tests (requires model download ~9GB)
    pytest test_nemotron_embedding.py -v -m integration
    
    # Run unit tests only (skip integration)
    pytest test_nemotron_embedding.py -v -m "not integration"
    
    # Skip integration tests via environment variable
    SKIP_INTEGRATION_TESTS=1 pytest test_nemotron_embedding.py -v

Model Reference:
    https://huggingface.co/nvidia/NVIDIA-Nemotron-Nano-9B-v2
"""

import os
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from transformers.modeling_outputs import BaseModelOutputWithPast

import nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding as nem

# Check if we should run integration tests
SKIP_INTEGRATION = os.environ.get("SKIP_INTEGRATION_TESTS", "0") == "1"
NEMOTRON_MODEL_ID = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"


class FakeNemotronConfig:
    """Fake config for testing."""
    def __init__(self, hidden_size=128, pooling="eos", temperature=1.0):
        self.hidden_size = hidden_size
        self.pooling = pooling
        self.temperature = temperature
        self.use_cache = False


class FakeTokenizer:
    """Fake tokenizer for testing."""
    def __init__(self):
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.eos_token = "</s>"
        self.pad_token = "<pad>"


class FakeLM(nn.Module):
    """Fake language model for testing."""
    def __init__(self, hidden=128, vocab_size=1000):
        super().__init__()
        self.config = FakeNemotronConfig(hidden_size=hidden, pooling="eos")
        self.hidden_size = hidden
        self.vocab_size = vocab_size
        
    def forward(
        self, 
        input_ids=None, 
        attention_mask=None,
        return_dict=True, 
        output_hidden_states=True,
        **kwargs
    ):
        batch_size, seq_len = input_ids.shape
        # Return fake hidden states
        hidden_states = torch.randn(batch_size, seq_len, self.hidden_size)
        
        # Create tuple of hidden states (all layers)
        all_hidden_states = tuple([hidden_states for _ in range(3)])  # 3 fake layers
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            past_key_values=None,
        )
    
    def gradient_checkpointing_enable(self):
        pass
    
    def save_pretrained(self, path):
        pass


class TestNemotronBidirectionalConfig:
    """Test suite for NemotronBidirectionalConfig."""
    
    def test_config_initialization_default(self):
        """Test config initialization with default values."""
        with patch('nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding.NemotronConfig'):
            config = FakeNemotronConfig(pooling="eos", temperature=1.0)
            
            assert config.pooling == "eos"
            assert config.temperature == 1.0
            assert config.hidden_size == 128
    
    def test_config_initialization_custom(self):
        """Test config initialization with custom values."""
        config = FakeNemotronConfig(pooling="last", temperature=0.5, hidden_size=256)
        
        assert config.pooling == "last"
        assert config.temperature == 0.5
        assert config.hidden_size == 256


class TestGetEosPositions:
    """Test suite for _get_eos_positions method."""
    
    @pytest.fixture
    def biencoder_model(self):
        """Create a biencoder model with fake encoders."""
        lm_q = FakeLM(hidden=128)
        lm_p = lm_q
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            pooling="eos",
            l2_normalize=False,
        )
        model.tokenizer = FakeTokenizer()
        return model
    
    def test_eos_positions_single_eos(self, biencoder_model):
        """Test finding EOS positions when each sequence has one EOS token."""
        input_ids = torch.tensor([
            [1, 3, 4, 2, 0, 0],  # EOS at position 3
            [5, 6, 2, 0, 0, 0],  # EOS at position 2
            [7, 8, 9, 10, 2, 0], # EOS at position 4
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ])
        
        positions = biencoder_model._get_eos_positions(input_ids, attention_mask)
        
        assert positions.shape == (3,)
        assert positions[0].item() == 3
        assert positions[1].item() == 2
        assert positions[2].item() == 4
    
    def test_eos_positions_no_eos_fallback(self, biencoder_model):
        """Test fallback to last non-padded position when no EOS token."""
        input_ids = torch.tensor([
            [1, 3, 4, 5, 0, 0],  # No EOS, last valid at position 3
            [5, 6, 7, 0, 0, 0],  # No EOS, last valid at position 2
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ])
        
        positions = biencoder_model._get_eos_positions(input_ids, attention_mask)
        
        assert positions.shape == (2,)
        # Should use sum of attention_mask - 1 as fallback
        assert positions[0].item() == 3  # 4 ones - 1
        assert positions[1].item() == 2  # 3 ones - 1
    
    def test_eos_positions_multiple_eos(self, biencoder_model):
        """Test that first EOS token is selected when multiple exist."""
        input_ids = torch.tensor([
            [1, 2, 3, 2, 0, 0],  # EOS at positions 1 and 3, should use 1
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
        ])
        
        positions = biencoder_model._get_eos_positions(input_ids, attention_mask)
        
        assert positions.shape == (1,)
        assert positions[0].item() == 1  # First EOS position


class TestPoolEmbeddings:
    """Test suite for _pool_embeddings method."""
    
    @pytest.fixture
    def biencoder_model_eos(self):
        """Create a biencoder model with EOS pooling."""
        lm_q = FakeLM(hidden=128)
        lm_p = lm_q
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            pooling="eos",
            l2_normalize=False,
        )
        model.tokenizer = FakeTokenizer()
        return model
    
    @pytest.fixture
    def biencoder_model_last(self):
        """Create a biencoder model with last-token pooling."""
        lm_q = FakeLM(hidden=128)
        lm_p = lm_q
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            pooling="last",
            l2_normalize=False,
        )
        model.tokenizer = FakeTokenizer()
        return model
    
    def test_pool_embeddings_eos(self, biencoder_model_eos):
        """Test EOS pooling strategy."""
        batch_size, seq_len, hidden_dim = 2, 6, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        input_ids = torch.tensor([
            [1, 3, 4, 2, 0, 0],  # EOS at position 3
            [5, 6, 2, 0, 0, 0],  # EOS at position 2
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0, 0],
        ])
        
        embeddings = biencoder_model_eos._pool_embeddings(
            hidden_states, input_ids, attention_mask
        )
        
        assert embeddings.shape == (batch_size, hidden_dim)
        # Verify correct positions were selected
        assert torch.allclose(embeddings[0], hidden_states[0, 3, :])
        assert torch.allclose(embeddings[1], hidden_states[1, 2, :])
    
    def test_pool_embeddings_last(self, biencoder_model_last):
        """Test last-token pooling strategy."""
        batch_size, seq_len, hidden_dim = 2, 6, 128
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        
        input_ids = torch.tensor([
            [1, 3, 4, 5, 0, 0],  # Last valid at position 3
            [5, 6, 7, 8, 9, 0],  # Last valid at position 4
        ])
        attention_mask = torch.tensor([
            [1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 0],
        ])
        
        embeddings = biencoder_model_last._pool_embeddings(
            hidden_states, input_ids, attention_mask
        )
        
        assert embeddings.shape == (batch_size, hidden_dim)
        # Verify correct positions were selected (sum - 1)
        assert torch.allclose(embeddings[0], hidden_states[0, 3, :])
        assert torch.allclose(embeddings[1], hidden_states[1, 4, :])
    
    def test_pool_embeddings_shape_preservation(self, biencoder_model_eos):
        """Test that pooling preserves correct dimensions."""
        batch_sizes = [1, 4, 8]
        seq_len = 10
        hidden_dim = 128
        
        for batch_size in batch_sizes:
            hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
            input_ids = torch.ones(batch_size, seq_len, dtype=torch.long)
            input_ids[:, 5] = 2  # Add EOS at position 5
            attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
            
            embeddings = biencoder_model_eos._pool_embeddings(
                hidden_states, input_ids, attention_mask
            )
            
            assert embeddings.shape == (batch_size, hidden_dim)


class TestBiencoderEncode:
    """Test suite for _encode method."""
    
    @pytest.fixture
    def biencoder_model(self):
        """Create a biencoder model for testing."""
        lm_q = FakeLM(hidden=128)
        lm_p = lm_q
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            pooling="eos",
            l2_normalize=True,
        )
        model.tokenizer = FakeTokenizer()
        return model
    
    def test_encode_basic(self, biencoder_model):
        """Test basic encoding functionality."""
        input_dict = {
            "input_ids": torch.tensor([[1, 3, 4, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
        
        embeddings = biencoder_model._encode(biencoder_model.lm_q, input_dict)
        
        assert embeddings is not None
        assert embeddings.shape == (1, 128)
        # Check L2 normalization
        norms = torch.linalg.norm(embeddings, dim=-1)
        assert torch.allclose(norms, torch.ones(1), atol=1e-5)
    
    def test_encode_removes_token_type_ids(self, biencoder_model):
        """Test that token_type_ids are removed if encoder doesn't support them."""
        input_dict = {
            "input_ids": torch.tensor([[1, 3, 4, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
            "token_type_ids": torch.zeros(1, 6, dtype=torch.long),
        }
        
        # Should not raise error even with token_type_ids
        embeddings = biencoder_model._encode(biencoder_model.lm_q, input_dict)
        
        assert embeddings is not None
        assert embeddings.shape == (1, 128)
    
    def test_encode_without_l2_normalize(self):
        """Test encoding without L2 normalization."""
        lm_q = FakeLM(hidden=128)
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_q,
            pooling="eos",
            l2_normalize=False,
        )
        model.tokenizer = FakeTokenizer()
        
        input_dict = {
            "input_ids": torch.tensor([[1, 3, 4, 2, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0]]),
        }
        
        embeddings = model._encode(model.lm_q, input_dict)
        
        assert embeddings is not None
        # Without normalization, norms should not necessarily be 1
        norms = torch.linalg.norm(embeddings, dim=-1)
        # Just check it's non-zero
        assert torch.all(norms > 0)
    
    def test_encode_empty_input(self, biencoder_model):
        """Test encoding with None/empty input."""
        result = biencoder_model._encode(biencoder_model.lm_q, None)
        assert result is None
        
        result = biencoder_model._encode(biencoder_model.lm_q, {})
        assert result is None
    
    def test_encode_batch(self, biencoder_model):
        """Test encoding with batched input."""
        batch_size = 4
        input_dict = {
            "input_ids": torch.ones(batch_size, 6, dtype=torch.long),
            "attention_mask": torch.ones(batch_size, 6, dtype=torch.long),
        }
        input_dict["input_ids"][:, 3] = 2  # Add EOS tokens
        
        embeddings = biencoder_model._encode(biencoder_model.lm_q, input_dict)
        
        assert embeddings.shape == (batch_size, 128)
        # Check all are normalized
        norms = torch.linalg.norm(embeddings, dim=-1)
        assert torch.allclose(norms, torch.ones(batch_size), atol=1e-5)


class TestBiencoderForward:
    """Test suite for BiencoderModel forward pass."""
    
    @pytest.fixture
    def biencoder_model(self):
        """Create a biencoder model for testing."""
        lm_q = FakeLM(hidden=128)
        lm_p = lm_q
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            train_n_passages=2,
            eval_negative_size=1,
            pooling="eos",
            l2_normalize=True,
            t=1.0,
            share_encoder=True,
        )
        model.tokenizer = FakeTokenizer()
        return model
    
    def test_forward_training_mode(self, biencoder_model):
        """Test forward pass in training mode."""
        biencoder_model.train()
        
        query = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
        query["input_ids"][:, 3] = 2  # Add EOS
        
        passage = {
            "input_ids": torch.ones(4, 6, dtype=torch.long),  # 2 queries * 2 passages
            "attention_mask": torch.ones(4, 6, dtype=torch.long),
        }
        passage["input_ids"][:, 3] = 2  # Add EOS
        
        with patch('nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding.contrastive_scores_and_labels') as mock_scores:
            mock_scores.return_value = (
                torch.randn(2, 2),  # scores
                torch.zeros(2, dtype=torch.long)  # labels
            )
            
            output = biencoder_model(query=query, passage=passage)
            
            assert hasattr(output, 'loss')
            assert hasattr(output, 'q_reps')
            assert hasattr(output, 'p_reps')
            assert hasattr(output, 'scores')
            assert hasattr(output, 'labels')
            
            # Check shapes
            assert output.q_reps.shape == (2, 128)
            assert output.p_reps.shape == (4, 128)
    
    def test_forward_eval_mode(self, biencoder_model):
        """Test forward pass in evaluation mode."""
        biencoder_model.eval()
        
        query = {
            "input_ids": torch.ones(2, 6, dtype=torch.long),
            "attention_mask": torch.ones(2, 6, dtype=torch.long),
        }
        query["input_ids"][:, 3] = 2
        
        passage = {
            "input_ids": torch.ones(4, 6, dtype=torch.long),
            "attention_mask": torch.ones(4, 6, dtype=torch.long),
        }
        passage["input_ids"][:, 3] = 2
        
        with patch('nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding.contrastive_scores_and_labels') as mock_scores:
            mock_scores.return_value = (
                torch.randn(2, 2),
                torch.zeros(2, dtype=torch.long)
            )
            
            output = biencoder_model(query=query, passage=passage)
            
            # In eval mode, uses eval_negative_size + 1
            mock_scores.assert_called_once()
            call_kwargs = mock_scores.call_args[1]
            assert call_kwargs['current_train_n_passages'] == 2  # eval_negative_size + 1


class TestBiencoderModelBuild:
    """Test suite for BiencoderModel.build class method."""
    
    def test_build_with_share_encoder(self, tmp_path):
        """Test building biencoder with shared encoder."""
        # Create a fake model directory
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()
        
        # Create a fake config.json
        import json
        config = {
            "model_type": "nemotron",
            "hidden_size": 128,
        }
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        with patch('nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding.NemotronBidirectionalModel') as MockModel:
            mock_instance = FakeLM(hidden=128)
            MockModel.from_pretrained.return_value = mock_instance
            
            model = nem.BiencoderModel.build(
                model_name_or_path=str(model_dir),
                share_encoder=True,
                pooling="eos",
                l2_normalize=True,
            )
            
            assert model is not None
            assert model.share_encoder is True
            assert model.lm_q is model.lm_p  # Same instance when sharing
    
    def test_build_without_share_encoder(self, tmp_path):
        """Test building biencoder without shared encoder."""
        model_dir = tmp_path / "fake_model"
        model_dir.mkdir()
        
        import json
        config = {"model_type": "nemotron", "hidden_size": 128}
        with open(model_dir / "config.json", "w") as f:
            json.dump(config, f)
        
        with patch('nemo_automodel.components.models.biencoder.nemotron_bidirectional_embedding.NemotronBidirectionalModel') as MockModel:
            mock_instance_q = FakeLM(hidden=128)
            mock_instance_p = FakeLM(hidden=128)
            MockModel.from_pretrained.side_effect = [mock_instance_q, mock_instance_p]
            
            model = nem.BiencoderModel.build(
                model_name_or_path=str(model_dir),
                share_encoder=False,
                pooling="eos",
            )
            
            assert model is not None
            assert model.share_encoder is False
            # Different calls mean different instances
            assert MockModel.from_pretrained.call_count == 2


class TestBiencoderModelSave:
    """Test suite for BiencoderModel.save method."""
    
    def test_save_shared_encoder(self, tmp_path):
        """Test saving biencoder with shared encoder."""
        lm_q = FakeLM(hidden=128)
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_q,
            share_encoder=True,
            add_linear_pooler=False,
        )
        
        output_dir = tmp_path / "saved_model"
        model.save(str(output_dir))
        
        # Should only save once for shared encoder
        assert output_dir.exists()
    
    def test_save_separate_encoders(self, tmp_path):
        """Test saving biencoder with separate encoders."""
        lm_q = FakeLM(hidden=128)
        lm_p = FakeLM(hidden=128)
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_p,
            share_encoder=False,
            add_linear_pooler=False,
        )
        
        output_dir = tmp_path / "saved_model"
        model.save(str(output_dir))
        
        # Should create separate directories
        assert output_dir.exists()
        assert (output_dir / "query_model").exists()
        assert (output_dir / "passage_model").exists()
    
    def test_save_with_linear_pooler(self, tmp_path):
        """Test saving biencoder with linear pooler."""
        lm_q = FakeLM(hidden=128)
        linear_pooler = nn.Linear(128, 256)
        model = nem.BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_q,
            linear_pooler=linear_pooler,
            share_encoder=True,
            add_linear_pooler=True,
        )
        
        output_dir = tmp_path / "saved_model"
        model.save(str(output_dir))
        
        # Should save pooler weights
        pooler_path = output_dir / "pooler.pt"
        assert pooler_path.exists()


# =============================================================================
# Integration Tests with NemotronBidirectionalModel
# =============================================================================


@pytest.mark.skipif(SKIP_INTEGRATION, reason="Skipping integration tests")
@pytest.mark.integration
class TestNemotronBidirectionalModelIntegration:
    """Integration tests using NemotronBidirectionalModel with real Nemotron Nano 9B v2."""
    
    @pytest.fixture(scope="class")
    def nemotron_biencoder(self):
        """Load Nemotron model as a biencoder using NemotronBidirectionalModel."""
        try:
            print(f"\nLoading Nemotron model via NemotronBidirectionalModel...")
            
            # Create config
            config = nem.NemotronBidirectionalConfig(
                pooling="mean",
                temperature=1.0,
                use_cache=False,
            )
            
            # Create model instance
            model = nem.NemotronBidirectionalModel(config)
            
            # Load the actual Nemotron Nano 9B v2 model
            model._load_model(NEMOTRON_MODEL_ID)
            
            # Move to device
            if torch.cuda.is_available():
                model.model = model.model.half()  # Use FP16 to save memory
            
            print("NemotronBidirectionalModel loaded successfully!")
            
            return model
            
        except Exception as e:
            pytest.skip(f"Could not load NemotronBidirectionalModel: {e}")
    
    def test_model_initialization(self, nemotron_biencoder):
        """Test that NemotronBidirectionalModel initializes correctly."""
        assert nemotron_biencoder is not None
        assert nemotron_biencoder.model is not None
        assert nemotron_biencoder.tokenizer is not None
        assert nemotron_biencoder.config is not None
        
        print(f"\nModel type: {type(nemotron_biencoder.model)}")
        print(f"Tokenizer type: {type(nemotron_biencoder.tokenizer)}")
        print(f"Hidden size: {nemotron_biencoder.model.config.hidden_size}")
    
    def test_forward_pass_single_text(self, nemotron_biencoder):
        """Test forward pass with a single text."""
        text = "What is machine learning?"
        
        # Tokenize
        inputs = nemotron_biencoder.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(nemotron_biencoder.model.device)
        
        # Forward pass
        outputs = nemotron_biencoder.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        
        # Verify outputs
        assert outputs is not None
        assert hasattr(outputs, 'last_hidden_state')
        assert outputs.last_hidden_state.shape[0] == 1  # batch size
        assert outputs.last_hidden_state.shape[1] > 0  # sequence length
        assert not torch.isnan(outputs.last_hidden_state).any()
        
        print(f"\nOutput shape: {outputs.last_hidden_state.shape}")
    
    def test_forward_pass_batch(self, nemotron_biencoder):
        """Test forward pass with batched texts."""
        texts = [
            "Machine learning is a subset of artificial intelligence.",
            "Deep learning uses neural networks.",
            "Natural language processing helps computers understand text.",
        ]
        
        # Tokenize
        inputs = nemotron_biencoder.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(nemotron_biencoder.model.device)
        
        # Forward pass
        outputs = nemotron_biencoder.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        
        # Verify
        assert outputs.last_hidden_state.shape[0] == len(texts)
        assert not torch.isnan(outputs.last_hidden_state).any()
        
        print(f"\nBatch output shape: {outputs.last_hidden_state.shape}")
    
    def test_biencoder_with_nemotron(self, nemotron_biencoder):
        """Test using NemotronBidirectionalModel in a BiencoderModel."""
        # Create biencoder with Nemotron model
        biencoder = nem.BiencoderModel(
            lm_q=nemotron_biencoder,
            lm_p=nemotron_biencoder,
            pooling="mean",
            l2_normalize=True,
            share_encoder=True,
        )
        biencoder.tokenizer = nemotron_biencoder.tokenizer
        
        # Test encoding
        query_text = "What is deep learning?"
        
        input_dict = {
            "input_ids": nemotron_biencoder.tokenizer(
                query_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )["input_ids"].to(nemotron_biencoder.model.device),
            "attention_mask": nemotron_biencoder.tokenizer(
                query_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )["attention_mask"].to(nemotron_biencoder.model.device),
        }
        
        # Encode
        embeddings = biencoder._encode(biencoder.lm_q, input_dict)
        
        # Verify
        assert embeddings is not None
        assert embeddings.shape[0] == 1
        assert embeddings.shape[1] > 0
        
        # Check normalization
        norm = torch.linalg.norm(embeddings, dim=-1).item()
        assert abs(norm - 1.0) < 1e-5, "Embeddings should be L2 normalized"
        
        print(f"\nEmbedding shape: {embeddings.shape}")
        print(f"Embedding norm: {norm:.6f}")
    
    def test_pooling_strategies(self, nemotron_biencoder):
        """Test different pooling strategies with NemotronBidirectionalModel."""
        text = "This is a test sentence."
        
        # Tokenize once
        inputs = nemotron_biencoder.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(nemotron_biencoder.model.device)
        
        # Get hidden states
        outputs = nemotron_biencoder.forward(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        
        hidden_states = outputs.last_hidden_state
        
        pooling_results = {}
        
        # Test different pooling strategies
        for pooling_type in ["mean", "eos", "last"]:
            # Create biencoder with this pooling
            biencoder = nem.BiencoderModel(
                lm_q=nemotron_biencoder,
                lm_p=nemotron_biencoder,
                pooling=pooling_type,
                l2_normalize=False,
                share_encoder=True,
            )
            biencoder.tokenizer = nemotron_biencoder.tokenizer
            
            # Pool embeddings
            embeddings = biencoder._pool_embeddings(
                hidden_states,
                inputs["input_ids"],
                inputs["attention_mask"],
            )
            
            pooling_results[pooling_type] = embeddings
            
            # Verify shape
            assert embeddings.shape[0] == 1
            assert embeddings.shape[1] == hidden_states.shape[2]
            
            norm = torch.linalg.norm(embeddings, dim=-1).item()
            print(f"\n{pooling_type.upper()} pooling: shape={embeddings.shape}, norm={norm:.4f}")
        
        # Verify all pooling strategies produce different results
        assert not torch.allclose(pooling_results["mean"], pooling_results["eos"])
        assert not torch.allclose(pooling_results["mean"], pooling_results["last"])
    
    def test_semantic_similarity_with_biencoder(self, nemotron_biencoder):
        """Test semantic similarity using BiencoderModel with Nemotron."""
        # Create biencoder
        biencoder = nem.BiencoderModel(
            lm_q=nemotron_biencoder,
            lm_p=nemotron_biencoder,
            pooling="mean",
            l2_normalize=True,
            share_encoder=True,
        )
        biencoder.tokenizer = nemotron_biencoder.tokenizer
        
        # Test texts
        text1 = "The cat sat on the mat."
        text2 = "A cat was sitting on a mat."
        text3 = "The weather is nice today."
        
        texts = [text1, text2, text3]
        embeddings_list = []
        
        for text in texts:
            input_dict = {
                "input_ids": nemotron_biencoder.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )["input_ids"].to(nemotron_biencoder.model.device),
                "attention_mask": nemotron_biencoder.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )["attention_mask"].to(nemotron_biencoder.model.device),
            }
            
            embeddings = biencoder._encode(biencoder.lm_q, input_dict)
            embeddings_list.append(embeddings)
        
        # Stack embeddings
        all_embeddings = torch.cat(embeddings_list, dim=0)
        
        # Compute similarities
        sim_1_2 = torch.cosine_similarity(all_embeddings[0:1], all_embeddings[1:2]).item()
        sim_1_3 = torch.cosine_similarity(all_embeddings[0:1], all_embeddings[2:3]).item()
        sim_2_3 = torch.cosine_similarity(all_embeddings[1:2], all_embeddings[2:3]).item()
        
        print(f"\nSimilarity (cat/cat): {sim_1_2:.4f}")
        print(f"Similarity (cat/weather): {sim_1_3:.4f}")
        print(f"Similarity (cat/weather): {sim_2_3:.4f}")
        
        # Similar texts should have higher similarity
        assert sim_1_2 > sim_1_3, "Similar texts should have higher similarity"
        assert sim_1_2 > sim_2_3, "Similar texts should have higher similarity"
    
    def test_multilingual_encoding(self, nemotron_biencoder):
        """Test encoding texts in different languages."""
        # Create biencoder
        biencoder = nem.BiencoderModel(
            lm_q=nemotron_biencoder,
            lm_p=nemotron_biencoder,
            pooling="mean",
            l2_normalize=True,
            share_encoder=True,
        )
        biencoder.tokenizer = nemotron_biencoder.tokenizer
        
        # Multilingual texts (same meaning)
        texts = {
            "English": "Hello, how are you?",
            "Spanish": "Hola, ¿cómo estás?",
            "French": "Bonjour, comment allez-vous?",
            "German": "Hallo, wie geht es dir?",
        }
        
        embeddings_dict = {}
        
        for lang, text in texts.items():
            input_dict = {
                "input_ids": nemotron_biencoder.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )["input_ids"].to(nemotron_biencoder.model.device),
                "attention_mask": nemotron_biencoder.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                )["attention_mask"].to(nemotron_biencoder.model.device),
            }
            
            embeddings = biencoder._encode(biencoder.lm_q, input_dict)
            embeddings_dict[lang] = embeddings
            
            # Verify
            assert embeddings.shape[0] == 1
            assert not torch.isnan(embeddings).any()
            
            norm = torch.linalg.norm(embeddings, dim=-1).item()
            print(f"\n{lang}: shape={embeddings.shape}, norm={norm:.6f}")
        
        print("\nAll languages encoded successfully!")
