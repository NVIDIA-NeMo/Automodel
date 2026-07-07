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

"""CPU source-load parity tests for force-HF remote-code models."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM

from nemo_automodel._transformers.auto_model import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.model_init import _init_model
from nemo_automodel._transformers.utils import apply_cache_compatibility_patches

REMOTE_MODELING_CODE = """
import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast


class TinyRemoteConfig(PretrainedConfig):
    model_type = "tiny_remote_lm"

    def __init__(
        self,
        vocab_size=16,
        hidden_size=8,
        tie_word_embeddings=True,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class TinyRemoteForCausalLM(PreTrainedModel):
    config_class = TinyRemoteConfig
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids=None, **kwargs):
        hidden_states = self.model.embed_tokens(input_ids)
        logits = self.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits)
"""


def _load_local_module(module_path: Path):
    module_name = f"_tiny_remote_{id(module_path)}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _is_aliased(model: torch.nn.Module) -> bool:
    return model.lm_head.weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()


def _write_tiny_remote_checkpoint(model_dir: Path, *, tie_word_embeddings: bool) -> Path:
    model_dir.mkdir()
    module_path = model_dir / "modeling_tiny_remote.py"
    module_path.write_text(REMOTE_MODELING_CODE)
    module = _load_local_module(module_path)

    config = module.TinyRemoteConfig(
        vocab_size=17,
        hidden_size=6,
        tie_word_embeddings=tie_word_embeddings,
    )
    config.architectures = ["TinyRemoteForCausalLM"]
    config.auto_map = {
        "AutoConfig": "modeling_tiny_remote.TinyRemoteConfig",
        "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM",
    }

    model = module.TinyRemoteForCausalLM(config)
    with torch.no_grad():
        embed_weight = torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).reshape(
            config.vocab_size, config.hidden_size
        )
        model.model.embed_tokens.weight.copy_(embed_weight / 100)
        if tie_word_embeddings:
            model.tie_weights()
            assert _is_aliased(model)
        else:
            head_weight = torch.arange(config.vocab_size * config.hidden_size, dtype=torch.float32).reshape(
                config.vocab_size, config.hidden_size
            )
            model.lm_head.weight.copy_(head_weight.flip(0) / 50)
            assert not _is_aliased(model)

    model.save_pretrained(model_dir, safe_serialization=False)
    return model_dir


@pytest.mark.parametrize("tie_word_embeddings", [False, True])
def test_force_hf_remote_code_source_load_matches_raw_hf(tmp_path: Path, tie_word_embeddings: bool):
    """NeMo force-HF source load should match the raw HF model for list-form tied-key remote code."""
    apply_cache_compatibility_patches()
    model_dir = _write_tiny_remote_checkpoint(tmp_path / "tiny_remote", tie_word_embeddings=tie_word_embeddings)

    hf_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.float32).eval()
    is_custom_model, nemo_model = _init_model(
        NeMoAutoModelForCausalLM,
        str(model_dir),
        attn_implementation="eager",
        torch_dtype=torch.float32,
        quantization_config=None,
        force_hf=True,
        trust_remote_code=True,
    )
    nemo_model.eval()

    assert is_custom_model is False
    assert getattr(nemo_model, "_nemo_tied_weights_keys") == {"lm_head.weight": "model.embed_tokens.weight"}
    assert _is_aliased(nemo_model) is tie_word_embeddings
    assert _is_aliased(hf_model) is tie_word_embeddings
    torch.testing.assert_close(nemo_model.get_input_embeddings().weight, hf_model.get_input_embeddings().weight)
    torch.testing.assert_close(nemo_model.lm_head.weight, hf_model.lm_head.weight)

    input_ids = torch.tensor([[0, 1, 2, 3, 4, 5, 6]], dtype=torch.long)
    with torch.no_grad():
        hf_logits = hf_model(input_ids=input_ids).logits
        nemo_logits = nemo_model(input_ids=input_ids).logits

    torch.testing.assert_close(nemo_logits, hf_logits, rtol=0, atol=0)
