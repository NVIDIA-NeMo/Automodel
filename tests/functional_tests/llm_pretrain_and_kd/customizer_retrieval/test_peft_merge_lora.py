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

"""Functional tests: PEFT + biencoder + merge_lora end-to-end.

Verifies that ``tools/merge_lora.py`` correctly handles non-CausalLM models
(embedding / feature-extraction models) by:

1. Building a LlamaBidirectionalModel (the backbone used by the retrieval
   customizer biencoder).
2. Applying HF PEFT LoRA with ``task_type=FEATURE_EXTRACTION``.
3. Merging the adapter via ``merge_lora`` (which must auto-detect the task
   type and use ``AutoModel`` instead of ``AutoModelForCausalLM``).
4. Comparing the merged model's weights and embeddings against a reference
   merge produced by HF PEFT's ``PeftModel.merge_and_unload``.

Two test classes are provided:

* ``TestMergeLoraEmbeddingModel`` – self-contained tiny model, no external
  data needed, validates the core merge_lora logic for FEATURE_EXTRACTION.
* ``TestMergeLoraRealBiencoder`` – uses the real
  ``llama-nemotron-embed-1b-v2`` checkpoint when available in CI.
"""

import json
import os
import shutil
from pathlib import Path

import pytest
import torch

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/TestData/automodel/llama-nemotron-embed-1b-v2",
)


# ---------------------------------------------------------------------------
# Self-contained tests with a tiny model (no external data needed)
# ---------------------------------------------------------------------------


def _make_tiny_llama_config():
    from transformers import LlamaConfig

    return LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=256,
        max_position_embeddings=64,
    )


LORA_R = 8
LORA_ALPHA = 32
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


class TestMergeLoraEmbeddingModel:
    """merge_lora with FEATURE_EXTRACTION task type (embedding / biencoder models).

    Uses a tiny randomly-initialised model so the test is fast and needs no
    external data or GPU.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = str(tmp_path)

    def _save_tiny_base_model(self):
        """Create a tiny LlamaModel (AutoModel-compatible), save, return path."""
        from transformers import AutoModel

        config = _make_tiny_llama_config()
        model = AutoModel.from_config(config)
        base_dir = os.path.join(self.tmp_path, "base_model")
        model.save_pretrained(base_dir)
        return base_dir

    def _apply_hf_peft_and_save(self, base_dir):
        """Apply HF PEFT LoRA (FEATURE_EXTRACTION) and save adapter."""
        from peft import LoraConfig, get_peft_model

        from transformers import AutoModel

        model = AutoModel.from_pretrained(base_dir)
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(model, lora_config)

        torch.manual_seed(42)
        for _, param in peft_model.named_parameters():
            if param.requires_grad:
                param.data = torch.randn_like(param.data) * 0.01

        adapter_dir = os.path.join(self.tmp_path, "adapter")
        peft_model.save_pretrained(adapter_dir)
        return adapter_dir

    def _reference_merge(self, base_dir, adapter_dir):
        """Merge using HF PEFT directly and return merged state dict."""
        from peft import PeftModel

        from transformers import AutoModel

        model = AutoModel.from_pretrained(base_dir)
        model = PeftModel.from_pretrained(model, adapter_dir)
        merged = model.merge_and_unload()
        return {n: p.data.clone() for n, p in merged.named_parameters()}

    def test_merge_produces_correct_weights(self):
        """Merged weights must match HF PEFT's merge_and_unload reference."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)
        ref_sd = self._reference_merge(base_dir, adapter_dir)

        merged_dir = os.path.join(self.tmp_path, "merged")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        merged_model = AutoModel.from_pretrained(merged_dir)
        tool_sd = {n: p.data for n, p in merged_model.named_parameters()}

        assert set(ref_sd.keys()) == set(tool_sd.keys()), "Key sets differ"
        for key in ref_sd:
            torch.testing.assert_close(
                tool_sd[key],
                ref_sd[key],
                rtol=1e-5,
                atol=1e-5,
                msg=f"Mismatch at {key}",
            )

    def test_auto_detects_feature_extraction_task_type(self):
        """merge_lora must auto-detect FEATURE_EXTRACTION and use AutoModel."""
        from tools.merge_lora import _resolve_auto_cls
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)

        cls = _resolve_auto_cls(adapter_dir)
        assert cls is AutoModel, f"Expected AutoModel for FEATURE_EXTRACTION, got {cls.__name__}"

    def test_merged_weights_differ_from_base(self):
        """Merged model must have different weights from the base model."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)

        base_sd = {n: p.data.clone() for n, p in AutoModel.from_pretrained(base_dir).named_parameters()}

        merged_dir = os.path.join(self.tmp_path, "merged_diff")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        merged_model = AutoModel.from_pretrained(merged_dir)
        changed = sum(
            1
            for name, param in merged_model.named_parameters()
            if name in base_sd and not torch.equal(param.data, base_sd[name])
        )
        assert changed > 0, "No weights changed after merge"

    def test_no_lora_params_remain(self):
        """After merge, no LoRA parameters should be present."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)

        merged_dir = os.path.join(self.tmp_path, "merged_clean")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )

        merged_model = AutoModel.from_pretrained(merged_dir)
        for name, _ in merged_model.named_parameters():
            assert "lora_" not in name, f"LoRA param {name} should be absent"

    def test_merged_embeddings_match_reference(self):
        """Hidden states of the merged model must match the adapter-applied reference."""
        from peft import PeftModel

        from tools.merge_lora import merge_lora
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)

        merged_dir = os.path.join(self.tmp_path, "merged_emb")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
        )
        merged_model = AutoModel.from_pretrained(merged_dir).eval()

        ref_base = AutoModel.from_pretrained(base_dir)
        ref_model = PeftModel.from_pretrained(ref_base, adapter_dir).eval()

        torch.manual_seed(123)
        input_ids = torch.randint(0, 256, (2, 32))
        attention_mask = torch.ones_like(input_ids)

        with torch.no_grad():
            merged_out = merged_model(input_ids, attention_mask=attention_mask)
            ref_out = ref_model(input_ids, attention_mask=attention_mask)

        merged_hidden = merged_out.last_hidden_state
        ref_hidden = ref_out.last_hidden_state

        assert merged_hidden.shape == ref_hidden.shape
        assert torch.isfinite(merged_hidden).all()
        torch.testing.assert_close(
            merged_hidden,
            ref_hidden,
            rtol=1e-5,
            atol=1e-5,
            msg="Embedding mismatch between merged model and reference",
        )

    def test_explicit_model_class_override(self):
        """merge_lora with explicit model_class='AutoModel' works correctly."""
        from tools.merge_lora import merge_lora
        from transformers import AutoModel

        base_dir = self._save_tiny_base_model()
        adapter_dir = self._apply_hf_peft_and_save(base_dir)

        merged_dir = os.path.join(self.tmp_path, "merged_explicit")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
            model_class="AutoModel",
        )

        merged_model = AutoModel.from_pretrained(merged_dir)
        assert merged_model is not None

        input_ids = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            out = merged_model(input_ids)
        assert torch.isfinite(out.last_hidden_state).all()


# ---------------------------------------------------------------------------
# Real biencoder tests (require CI model data + GPU)
# ---------------------------------------------------------------------------

_HAS_REAL_MODEL = os.path.isdir(BASE_MODEL_PATH)
_HAS_CUDA = torch.cuda.is_available()
_SKIP_REASON = "Requires real model at BASE_MODEL_PATH and CUDA" if not (_HAS_REAL_MODEL and _HAS_CUDA) else ""


@pytest.mark.skipif(not (_HAS_REAL_MODEL and _HAS_CUDA), reason=_SKIP_REASON)
class TestMergeLoraRealBiencoder:
    """End-to-end merge_lora with the real biencoder model.

    Loads ``llama-nemotron-embed-1b-v2``, applies HF PEFT LoRA, merges with
    ``merge_lora``, and verifies that the merged model produces valid
    embeddings via the BiencoderModel wrapper.
    """

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.tmp_path = str(tmp_path)

    def test_merge_lora_real_biencoder(self):
        from peft import LoraConfig, PeftModel, get_peft_model

        from nemo_automodel.components.models.biencoder import BiencoderModel
        from nemo_automodel.components.models.llama_bidirectional import LlamaBidirectionalModel
        from tools.merge_lora import merge_lora

        # 1. Load base model
        base_model = LlamaBidirectionalModel.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.float32,
        )

        # Save to a temp dir so merge_lora can reload it.
        # Set auto_map so AutoModel can find the custom class.
        base_dir = os.path.join(self.tmp_path, "base_model")
        base_model.save_pretrained(base_dir)

        # Copy custom model code so trust_remote_code can discover it.
        import nemo_automodel.components.models.llama_bidirectional.model as _mod

        model_py_src = Path(_mod.__file__)
        shutil.copy2(str(model_py_src), os.path.join(base_dir, "model.py"))

        # Patch config.json with auto_map for AutoModel loading.
        config_path = os.path.join(base_dir, "config.json")
        with open(config_path) as f:
            cfg = json.load(f)
        cfg["auto_map"] = {
            "AutoModel": "model.LlamaBidirectionalModel",
            "AutoConfig": "model.LlamaBidirectionalConfig",
        }
        cfg["model_type"] = "LlamaBidirectionalModel"
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        # 2. Apply HF PEFT LoRA
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=LORA_TARGET_MODULES,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        peft_model = get_peft_model(base_model, lora_config)

        torch.manual_seed(42)
        for _, param in peft_model.named_parameters():
            if param.requires_grad:
                param.data = torch.randn_like(param.data) * 0.01

        adapter_dir = os.path.join(self.tmp_path, "adapter")
        peft_model.save_pretrained(adapter_dir)

        # 3. Reference merge
        ref_base = LlamaBidirectionalModel.from_pretrained(
            base_dir,
            torch_dtype=torch.float32,
        )
        ref_peft = PeftModel.from_pretrained(ref_base, adapter_dir)
        ref_merged = ref_peft.merge_and_unload().eval()

        # 4. merge_lora (uses AutoModel via FEATURE_EXTRACTION detection)
        merged_dir = os.path.join(self.tmp_path, "merged")
        merge_lora(
            base_model=base_dir,
            adapter_path=adapter_dir,
            output_dir=merged_dir,
            dtype="float32",
            device="cpu",
            save_tokenizer=False,
            trust_remote_code=True,
        )

        # 5. Load merged model
        merged_model = LlamaBidirectionalModel.from_pretrained(
            merged_dir,
            torch_dtype=torch.float32,
        ).eval()

        # 6. Compare weights
        ref_sd = dict(ref_merged.named_parameters())
        for name, param in merged_model.named_parameters():
            assert name in ref_sd, f"Missing key {name} in reference"
            torch.testing.assert_close(
                param.data,
                ref_sd[name].data,
                rtol=1e-4,
                atol=1e-4,
                msg=f"Weight mismatch at {name}",
            )

        # 7. Verify embeddings via biencoder wrapper
        biencoder_ref = BiencoderModel(
            lm_q=ref_merged,
            lm_p=ref_merged,
            share_encoder=True,
            pooling="avg",
            l2_normalize=True,
        ).eval()
        biencoder_merged = BiencoderModel(
            lm_q=merged_model,
            lm_p=merged_model,
            share_encoder=True,
            pooling="avg",
            l2_normalize=True,
        ).eval()

        torch.manual_seed(99)
        input_ids = torch.randint(0, 1000, (4, 64))
        attention_mask = torch.ones_like(input_ids)
        input_dict = {"input_ids": input_ids, "attention_mask": attention_mask}

        with torch.no_grad():
            emb_ref = biencoder_ref.encode(input_dict, encoder="query")
            emb_merged = biencoder_merged.encode(input_dict, encoder="query")

        assert emb_ref is not None and emb_merged is not None
        assert torch.isfinite(emb_merged).all(), "Merged embeddings contain non-finite values"

        # Embeddings should be L2-normalized.
        norms = emb_merged.norm(dim=-1)
        torch.testing.assert_close(
            norms,
            torch.ones_like(norms),
            atol=1e-4,
            rtol=1e-4,
            msg="Merged embeddings are not L2-normalized",
        )

        # Merged and reference embeddings must match.
        torch.testing.assert_close(
            emb_merged,
            emb_ref,
            rtol=1e-4,
            atol=1e-4,
            msg="Embedding mismatch between merge_lora output and reference",
        )

        print("\n[PASS] merge_lora real biencoder: embeddings match reference")
        print(f"  Embedding shape: {tuple(emb_merged.shape)}")
        print(f"  Cosine(0,1): {float(emb_merged[0] @ emb_merged[1]):.4f}")
