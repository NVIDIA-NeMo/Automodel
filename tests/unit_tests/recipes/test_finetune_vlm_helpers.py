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
import pytest
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from nemo_automodel.recipes.vlm.finetune import _freeze_model, build_model_and_optimizer
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig



def _count_trainable(parameters):
    return sum(p.numel() for p in parameters if getattr(p, "requires_grad", False))

@pytest.fixture(autouse=True)
def _mock_missing_cuda(monkeypatch):
    """Some helper functions unconditionally access torch.cuda APIs. When running on a
    CPU-only build they raise `RuntimeError: Torch not compiled with CUDA`.
    Patch the relevant CUDA APIs with no-op stubs when CUDA is unavailable."""
    if torch.cuda.is_available():
        yield  # nothing to do
        return

    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [], raising=False)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda _: None, raising=False)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0, raising=False)
    yield


class DummyModel(nn.Module):
    """Simple model containing an embedding and a linear layer ("language_model")."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10, 4)
        # expose as attribute so apply_parameter_freezing can find it
        self.language_model = nn.Linear(4, 4)

    def forward(self, x):  # pragma: no cover – not needed for these unit tests
        return self.language_model(self.embedding(x))


class DummyOptConfig:
    """Mimics an optimizer config object with an *instantiate* method."""

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.foreach = None 
    def instantiate(self, params):
        # Always return an SGD optimizer for the given params
        return torch.optim.SGD(params, lr=self.lr)

    def get(self, key, default):
        return getattr(self, key, default)

class DummyModelConfig:
    """Mimics the Hydra/OmegaConf model config with an *instantiate* method."""

    def instantiate(self):
        return DummyModel()

    def get(self, key, default=None):
        return getattr(self, key, default)


# -----------------------------------------------------------------------------
# _freeze_model
# -----------------------------------------------------------------------------

def test_freeze_model_embedding_only():
    """When cfg_freeze is *None*, embeddings should be frozen by default."""
    model = DummyModel()
    assert model.embedding.weight.requires_grad  # sanity check – initially True

    _freeze_model(model, cfg_freeze=None, freeze_embeddings=True)

    # embedding weights must be frozen; linear layer should still train
    assert not model.embedding.weight.requires_grad
    assert model.language_model.weight.requires_grad


def test_freeze_model_with_config(monkeypatch):
    """Custom freeze config should be respected (freeze_language_model only)."""
    model = DummyModel()

    freeze_cfg = {
        "freeze_embeddings": False,  # keep embeddings trainable
        "freeze_language_model": True,
    }

    _freeze_model(model, cfg_freeze=freeze_cfg, freeze_embeddings=False)

    # embedding remains trainable, language_model is frozen
    assert model.embedding.weight.requires_grad
    assert not model.language_model.weight.requires_grad


# -----------------------------------------------------------------------------
# build_model_and_optimizer
# -----------------------------------------------------------------------------

def test_build_model_and_optimizer_basic():
    cfg_model = DummyModelConfig()
    cfg_opt = DummyOptConfig(lr=0.01)

    device = torch.device("cpu")
    # Minimal checkpointer to satisfy build_model_and_optimizer signature
    ckpt_cfg = CheckpointingConfig(
        enabled=False,
        checkpoint_dir="checkpoints/",
        model_save_format="safetensors",
        model_cache_dir=".",
        model_repo_id="dummy/model",
        save_consolidated=False,
        is_peft=False,
    )
    checkpointer = Checkpointer(config=ckpt_cfg, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
    model, _, optim = build_model_and_optimizer(
        device=device,
        cfg_model=cfg_model,
        cfg_opt=cfg_opt,
        cfg_freeze=None,
        cfg_peft=None,
        model_wrapper=None,
        seed=123,
        checkpointer=checkpointer,
        tp_size=1,
        freeze_embeddings=True,
    )

    # Check returned objects and their properties
    assert isinstance(model, DummyModel)
    assert isinstance(optim, torch.optim.Optimizer)

    # Model parameters should reside on the requested device
    assert next(model.parameters()).device == device

    # Embedding weights should be frozen by default
    assert not model.embedding.weight.requires_grad

    # Optimizer should hold only trainable parameters
    trainable_param_count = _count_trainable(model.parameters())
    optim_param_count = sum(p.numel() for group in optim.param_groups for p in group["params"])
    assert trainable_param_count == optim_param_count


# -----------------------------------------------------------------------------
# AutoProcessor exception handling test
# -----------------------------------------------------------------------------

def test_autoprocessor_success():
    """Test successful AutoProcessor creation."""
    
    with patch('transformers.AutoProcessor') as mock_auto_processor:
        mock_processor = MagicMock()
        mock_auto_processor.from_pretrained.return_value = mock_processor

        model_id = "test/model"

        processor = mock_auto_processor.from_pretrained(model_id)
        
        assert processor is mock_processor
        mock_auto_processor.from_pretrained.assert_called_once_with("test/model")


def test_autoprocessor_exception_handling(caplog):
    """Test AutoProcessor exception handling and logging in build_dataloader."""
    import logging
    from nemo_automodel.recipes.vlm.finetune import build_dataloader
    
    with patch('transformers.AutoProcessor.from_pretrained') as mock_from_pretrained, \
         patch('nemo_automodel.components.training.rng.StatefulRNG'), \
         patch('torch.utils.data.distributed.DistributedSampler'), \
         patch('nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS', {'NoneType': MagicMock()}):
        
        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")
        
        # Mock configurations - minimal setup
        cfg_ds = MagicMock()
        cfg_ds.instantiate.return_value = []
        cfg_ds.path_or_dataset = "test/dataset"
        
        cfg_dl = MagicMock()
        cfg_dl.get.return_value = None  # No custom settings
        cfg_dl.instantiate.return_value = MagicMock()
        
        cfg_processor = None  # This triggers the exception path
        
        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)
        
        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model")
        


def test_autoprocessor_with_processor_kwargs(caplog):
    """Test AutoProcessor exception handling when cfg_processor has no instantiate method."""
    import logging
    from nemo_automodel.recipes.vlm.finetune import build_dataloader
    
    # Simple processor config class without instantiate method
    class ProcessorConfig:
        def to_dict(self):
            return {"trust_remote_code": True, "some_param": "value"}
    
    with patch('transformers.AutoProcessor.from_pretrained') as mock_from_pretrained, \
         patch('nemo_automodel.components.training.rng.StatefulRNG'), \
         patch('torch.utils.data.distributed.DistributedSampler'), \
         patch('nemo_automodel.components.datasets.vlm.collate_fns.COLLATE_FNS', {'NoneType': MagicMock()}):
        
        # Set up the exception
        mock_from_pretrained.side_effect = Exception("Model does not have AutoProcessor")
        
        # Mock configurations - minimal setup
        cfg_ds = MagicMock()
        cfg_ds.instantiate.return_value = []
        cfg_ds.path_or_dataset = "test/dataset"
        
        cfg_dl = MagicMock()
        cfg_dl.get.return_value = None  # No custom settings
        cfg_dl.instantiate.return_value = MagicMock()
        
        cfg_processor = ProcessorConfig()  # This has to_dict but no instantiate
        
        with caplog.at_level(logging.WARNING):
            dataloader, processor = build_dataloader(cfg_ds, cfg_dl, "test/model", cfg_processor, None, 123, 1)
        
        # Verify the results
        assert processor is None
        mock_from_pretrained.assert_called_once_with("test/model", trust_remote_code=True, some_param="value")
        