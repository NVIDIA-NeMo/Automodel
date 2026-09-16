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

import json
import os
import subprocess
import sys
import textwrap

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from nemo_automodel.components.models.llama_nemotron_vl.model import LlamaNemotronVLConfig, LlamaNemotronVLModel


@pytest.fixture
def local_vl_checkpoint(tmp_path):
    """Save real tiny text/vision weights and tokenizer assets, without Hub code."""
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1}, unk_token="[UNK]")),
        unk_token="[UNK]",
        pad_token="[PAD]",
    )
    tokenizer.save_pretrained(tmp_path)
    config = LlamaNemotronVLConfig(
        name_or_path=str(tmp_path),
        vision_config={
            "model_type": "siglip_vision_model",
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "image_size": 4,
            "patch_size": 2,
        },
        llm_config={
            "model_type": "llama",
            "architectures": ["LlamaBidirectionalModel"],
            "vocab_size": 16,
            "hidden_size": 8,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
        },
        # Match Hub metadata but omit the remote source: Automodel owns the config contract.
        auto_map={"AutoConfig": "configuration_llama_nemotron_vl.LlamaNemotronVLConfig"},
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        LlamaNemotronVLModel(config).save_pretrained(tmp_path)
    return tmp_path


@pytest.mark.parametrize("trust_remote_code", [False, True])
@pytest.mark.parametrize("is_causal", [None, False, True])
# A fresh interpreter re-imports PyTorch; keep its lifetime bounded beyond the 5s unit-test default.
@pytest.mark.timeout(60)
def test_public_bi_encoder_loads_vl_checkpoint_in_fresh_process(local_vl_checkpoint, trust_remote_code, is_causal):
    script = textwrap.dedent(
        """
        import json
        import sys

        import torch

        assert "nemo_automodel.components.models.llama_nemotron_vl.model" not in sys.modules
        from nemo_automodel._transformers.retrieval import BiEncoderModel

        checkpoint, trust_remote_code, requested_policy = sys.argv[1:]
        requested_policy = json.loads(requested_policy)
        encoder = BiEncoderModel.build(
            checkpoint,
            trust_remote_code=json.loads(trust_remote_code),
            is_causal=requested_policy,
            attn_implementation="eager",
            local_files_only=True,
            pooling="cls",
        ).eval()
        expected_policy = requested_policy is True
        assert encoder.is_causal is expected_policy
        assert encoder.config.get_text_config(decoder=True) is encoder.model.language_model.config
        assert encoder.config.llm_config.is_causal is expected_policy
        assert "is_causal" not in vars(encoder.config)
        assert "is_causal" not in vars(encoder.config.vision_config)

        inputs = {"input_ids": torch.tensor([[2, 3, 4, 5]]), "attention_mask": torch.ones(1, 4, dtype=torch.long)}
        changed_inputs = {**inputs, "input_ids": torch.tensor([[2, 3, 4, 6]])}
        with torch.no_grad():
            original = encoder(inputs)
            changed = encoder(changed_inputs)
        assert torch.isfinite(original).all()
        if expected_policy:
            torch.testing.assert_close(original, changed)
        else:
            assert not torch.allclose(original, changed, atol=1e-6)
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, str(local_vl_checkpoint), json.dumps(trust_remote_code), json.dumps(is_causal)],
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
