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
Functional tests for biencoder checkpoint restoration.

  1. Full-model checkpoint: train a biencoder with NeMo Automodel, load the
     trained model back in NeMo, save, then restore using the transformers
     library (``LlamaBidirectionalModel.from_pretrained``) and verify the
     state dicts match exactly.
  2. PEFT checkpoint: same flow but with LoRA applied.  Verifies that base
     weights are correctly loaded by transformers and LoRA adapter weights
     round-trip through safetensors.

Paths below are defaults that match the CI / functional-test environment.
Override them with environment variables when running locally:

    BASE_MODEL_PATH     - pretrained HF checkpoint directory
    CHECKPOINT_DIR      - where full-model training writes checkpoints
    PEFT_CHECKPOINT_DIR - where PEFT training writes checkpoints
    RECIPE_YAML         - training recipe for full-model restoration test
    PEFT_RECIPE_YAML    - training recipe for PEFT restoration test
"""

import glob as _glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Default paths (overridable via env vars)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/TestData/automodel/llama-nemotron-embed-1b-v2",
)
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    "/workspace/output/biencoder_ckpt_restore/checkpoints",
)
PEFT_CHECKPOINT_DIR = os.environ.get(
    "PEFT_CHECKPOINT_DIR",
    "/workspace/output/biencoder_ckpt_restore_peft/checkpoints",
)
RECIPE_YAML = os.environ.get(
    "RECIPE_YAML",
    str(_THIS_DIR / "recipe_ckpt_restore.yaml"),
)
PEFT_RECIPE_YAML = os.environ.get(
    "PEFT_RECIPE_YAML",
    str(_THIS_DIR / "recipe_peft.yaml"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_training(recipe_yaml: str, checkpoint_dir: str) -> Path:
    """Launch biencoder training as a subprocess and return the checkpoint dir."""
    cmd = [
        sys.executable,
        "-m", "coverage", "run",
        "--data-file=/workspace/.coverage",
        "--source=/workspace/",
        "--parallel-mode",
        "-m", "nemo_automodel.recipes.biencoder.train_biencoder",
        "--config",
        recipe_yaml,
    ]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"

    ckpt_root = Path(checkpoint_dir)
    matches = sorted(ckpt_root.glob("epoch_*_step_*"))
    assert matches, f"No epoch_*_step_* checkpoints found under {ckpt_root}"
    return matches[-1].resolve()


def _load_safetensors_from_dir(directory: str) -> dict[str, torch.Tensor]:
    """Load all safetensors files from *directory* into a single state dict."""
    from safetensors.torch import load_file

    sf_files = sorted(_glob.glob(os.path.join(directory, "*.safetensors")))
    assert sf_files, f"No .safetensors files found in {directory}"
    state_dict: dict[str, torch.Tensor] = {}
    for path in sf_files:
        state_dict.update(load_file(path, device="cpu"))
    return state_dict


def _compare_state_dicts(
    sd_a: dict[str, torch.Tensor],
    sd_b: dict[str, torch.Tensor],
    *,
    atol: float = 1e-5,
    prefix_a: str = "A",
    prefix_b: str = "B",
    key_filter=None,
):
    """Assert that two state dicts have the same keys and numerically equal values.

    Args:
        sd_a, sd_b: state dicts to compare.
        atol: absolute tolerance for float comparison.
        prefix_a, prefix_b: labels for error messages.
        key_filter: optional ``(key) -> bool`` predicate to restrict comparison.
    """
    keys_a = set(sd_a.keys())
    keys_b = set(sd_b.keys())

    if key_filter is not None:
        keys_a = {k for k in keys_a if key_filter(k)}
        keys_b = {k for k in keys_b if key_filter(k)}

    assert keys_a == keys_b, (
        f"Key mismatch: {prefix_a} extra = {keys_a - keys_b}, "
        f"{prefix_b} extra = {keys_b - keys_a}"
    )
    for key in sorted(keys_a):
        ta = sd_a[key].float().cpu()
        tb = sd_b[key].float().cpu()
        assert ta.shape == tb.shape, (
            f"Shape mismatch at {key}: {ta.shape} vs {tb.shape}"
        )
        assert torch.allclose(ta, tb, atol=atol), (
            f"Value mismatch at {key}: max diff = {(ta - tb).abs().max().item():.6e}"
        )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBiencoderCheckpointRestoration:
    """Verify that biencoder checkpoints produced by NeMo Automodel training
    can be restored by the transformers / safetensors libraries."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove training checkpoints after each test."""
        yield
        for ckpt_dir in (CHECKPOINT_DIR, PEFT_CHECKPOINT_DIR):
            p = Path(ckpt_dir)
            if p.exists():
                shutil.rmtree(p, ignore_errors=True)

    # ------------------------------------------------------------------ #
    # Test 1: full-model checkpoint restoration                           #
    # ------------------------------------------------------------------ #

    def test_full_model_checkpoint_restoration(self):
        """Train biencoder -> load trained model back in NeMo -> save ->
        restore with transformers -> verify state dicts match."""

        from nemo_automodel._transformers.biencoder import BiencoderModel
        from nemo_automodel.components.models.llama_bidirectional import LlamaBidirectionalModel

        # ---- Step 1: Train ------------------------------------------------
        checkpoint_dir = _run_training(RECIPE_YAML, CHECKPOINT_DIR)

        # With ``save_consolidated: true`` in the recipe, the checkpointer
        # writes a fully HF-compatible directory at ``model/consolidated/``
        # containing config.json, consolidated safetensors, and the index.
        consolidated_dir = checkpoint_dir / "model" / "consolidated"
        assert os.path.isfile(consolidated_dir / "config.json"), (
            "config.json not found in consolidated checkpoint "
            f"(looked in {consolidated_dir})"
        )

        # ---- Step 2: Load trained model back in NeMo ----------------------
        # The consolidated checkpoint is a standard HF checkpoint that
        # ``from_pretrained`` can load directly.
        lm_q = LlamaBidirectionalModel.from_pretrained(
            str(consolidated_dir), torch_dtype=torch.bfloat16
        )
        nemo_model = BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_q,
            share_encoder=True,
            pooling="avg",
            l2_normalize=True,
        )

        # ---- Step 3: Save from NeMo (HF format) --------------------------
        with tempfile.TemporaryDirectory() as save_dir:
            nemo_model.save_pretrained(save_dir)

            assert os.path.isfile(os.path.join(save_dir, "config.json")), (
                "config.json not found after save_pretrained"
            )
            sf_files = _glob.glob(os.path.join(save_dir, "*.safetensors"))
            assert sf_files, "No .safetensors after save_pretrained"

            # ---- Step 4: Restore with transformers ------------------------
            hf_model = LlamaBidirectionalModel.from_pretrained(
                save_dir, torch_dtype=torch.bfloat16
            )

            # ---- Step 5: Compare state dicts ------------------------------
            _compare_state_dicts(
                nemo_model.lm_q.state_dict(),
                hf_model.state_dict(),
                prefix_a="NeMo",
                prefix_b="HF-transformers",
            )

        print("\n[PASS] Full-model checkpoint round-trip: NeMo -> save -> HF transformers")

    # ------------------------------------------------------------------ #
    # Test 2: PEFT (LoRA) checkpoint restoration                          #
    # ------------------------------------------------------------------ #

    def test_peft_checkpoint_restoration(self):
        """Train biencoder with LoRA -> load in NeMo -> save -> verify base
        weights restored by transformers + LoRA weights by safetensors."""

        from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
        from nemo_automodel._transformers.biencoder import BiencoderModel
        from nemo_automodel.components.models.llama_bidirectional import LlamaBidirectionalModel

        # ---- Step 1: Train with PEFT -------------------------------------
        checkpoint_dir = _run_training(PEFT_RECIPE_YAML, PEFT_CHECKPOINT_DIR)

        # The PEFT checkpointer saves adapter_model.safetensors and
        # adapter_config.json under the ``model/`` subdirectory.
        model_dir = checkpoint_dir / "model"
        assert os.path.isfile(model_dir / "adapter_config.json"), (
            "adapter_config.json not found in training checkpoint "
            f"(looked in {model_dir})"
        )

        # ---- Step 2: Load trained PEFT model in NeMo ----------------------
        # The PEFT training checkpoint contains only adapter (LoRA) weights
        # saved by the checkpointer.  We rebuild the model exactly as the
        # recipe does: base model + apply LoRA + load adapter weights.
        lm_q = LlamaBidirectionalModel.from_pretrained(
            BASE_MODEL_PATH, torch_dtype=torch.bfloat16
        )
        nemo_model = BiencoderModel(
            lm_q=lm_q,
            lm_p=lm_q,
            share_encoder=True,
            pooling="avg",
            l2_normalize=True,
        )

        peft_config = PeftConfig(match_all_linear=True, dim=8, alpha=32, use_triton=False)
        n_adapted = apply_lora_to_linear_modules(nemo_model, peft_config, quantization_config=None)
        assert n_adapted > 0, "No modules were adapted with LoRA"
        print(f"\nApplied LoRA to {n_adapted} modules")

        # Load adapter weights from the checkpoint.  The checkpointer stores
        # PEFT weights with HF-style prefixes produced by the state-dict
        # adapter: ``base_model.model.model.<param>`` for the query encoder.
        # Strip the ``base_model.model.model.`` prefix so keys match lm_q.
        raw_ckpt_sd = _load_safetensors_from_dir(str(model_dir))
        _PREFIX_TO_STRIP = "base_model.model.model."
        ckpt_sd = {
            k[len(_PREFIX_TO_STRIP):] if k.startswith(_PREFIX_TO_STRIP) else k: v
            for k, v in raw_ckpt_sd.items()
        }
        missing, unexpected = nemo_model.lm_q.load_state_dict(ckpt_sd, strict=False)
        # lora_dropout has no parameters so it won't appear in the safetensors;
        # missing keys should only be non-persistent buffers (e.g. rotary_emb)
        # and base model weights (which are already loaded from BASE_MODEL_PATH).
        print(f"load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")

        # ---- Step 3: Save from NeMo (HF format) --------------------------
        with tempfile.TemporaryDirectory() as save_dir:
            nemo_model.save_pretrained(save_dir)

            assert os.path.isfile(os.path.join(save_dir, "config.json")), (
                "config.json not found after save_pretrained"
            )
            sf_files = _glob.glob(os.path.join(save_dir, "*.safetensors"))
            assert sf_files, "No .safetensors after save_pretrained"

            # ---- Step 4: Load safetensors from the saved checkpoint -------
            saved_sd = _load_safetensors_from_dir(save_dir)

            # ---- Step 5: Verify LoRA weights are present ------------------
            lora_keys = sorted(k for k in saved_sd if "lora_" in k)
            assert len(lora_keys) > 0, (
                "No LoRA adapter weights found in saved checkpoint"
            )
            print(f"Found {len(lora_keys)} LoRA weight tensors in checkpoint")

            # ---- Step 6: Restore with transformers + LoRA ------------------
            # The saved checkpoint contains the full lm_q state dict
            # (base weights + LoRA weights).  Build a fresh model from
            # the base checkpoint (no LoRA keys → no UNEXPECTED noise),
            # apply LoRA, then load the complete saved state dict so
            # every weight (base + adapter) is populated in one shot.
            hf_model = LlamaBidirectionalModel.from_pretrained(
                BASE_MODEL_PATH, torch_dtype=torch.bfloat16
            )
            hf_peft_config = PeftConfig(match_all_linear=True, dim=8, alpha=32, use_triton=False)
            apply_lora_to_linear_modules(hf_model, hf_peft_config, quantization_config=None)
            hf_full_sd = _load_safetensors_from_dir(save_dir)
            hf_model.load_state_dict(hf_full_sd, strict=False)

            # ---- Step 7: Compare ALL weights (base + LoRA) ----------------
            nemo_sd = nemo_model.lm_q.state_dict()
            hf_sd = hf_model.state_dict()

            def _is_comparable_key(k):
                # Exclude dropout layers (no parameters) and
                # non-persistent buffers (e.g. rotary_emb).
                return "lora_dropout" not in k

            _compare_state_dicts(
                nemo_sd,
                hf_sd,
                prefix_a="NeMo",
                prefix_b="HF-transformers",
                key_filter=_is_comparable_key,
            )

        print("\n[PASS] PEFT checkpoint round-trip: NeMo -> save -> HF transformers + safetensors")
