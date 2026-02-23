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
Functional test: train a biencoder with the customizer-aligned recipe, then
verify that the fine-tuned model does not degrade vs the baseline on held-out
data (paired t-test + Cohen's D check).

Paths below are defaults that match the CI / functional-test environment.
Override them with environment variables when running locally:

    BASE_MODEL_PATH  – pretrained HF checkpoint directory
    CHECKPOINT_DIR   – where training writes checkpoints
    TEST_DATA_JSONL  – evaluation JSONL file
    RECIPE_YAML      – training recipe (defaults to the one next to this file)
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Default paths (overridable via env vars)
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]  # Automodel6/

BASE_MODEL_PATH = os.environ.get(
    "BASE_MODEL_PATH",
    "/home/TestData/automodel/llama-nemotron-embed-1b-v2",
)
CHECKPOINT_DIR = os.environ.get(
    "CHECKPOINT_DIR",
    "/workspace/output/biencoder_inline/checkpoints",
)
TEST_DATA_JSONL = os.environ.get(
    "TEST_DATA_JSONL",
    "/home/TestData/automodel/embedding_testdata/testing.jsonl",
)
RECIPE_YAML = os.environ.get(
    "RECIPE_YAML",
    str(_THIS_DIR / "recipe.yaml"),
)

# Evaluation hyper-parameters (aligned with customizer defaults).
EVAL_MAX_LENGTH = 128
EVAL_BATCH_SIZE = 8
EVAL_TEMPERATURE = 0.02  # native inference temperature for the model


# ---------------------------------------------------------------------------
# Helpers (thin wrappers around compare_biencoder_models logic)
# ---------------------------------------------------------------------------

def _run_training() -> Path:
    """Launch the biencoder training recipe as a subprocess and return the
    checkpoint directory produced by the run."""
    cmd = [
        sys.executable, "-m", "nemo_automodel.recipes.biencoder.train_biencoder",
        "--config", RECIPE_YAML,
    ]
    result = subprocess.run(cmd, cwd=str(_REPO_ROOT), check=True)
    assert result.returncode == 0, f"Training failed with return code {result.returncode}"

    # Resolve the latest checkpoint under CHECKPOINT_DIR.
    ckpt_root = Path(CHECKPOINT_DIR)
    matches = sorted(ckpt_root.glob("epoch_*_step_*"))
    assert matches, f"No epoch_*_step_* checkpoints found under {ckpt_root}"
    return matches[-1].resolve()


def _build_eval_model(device: torch.device):
    """Build a NeMoAutoModelBiencoder for evaluation."""
    from nemo_automodel._transformers.auto_model import NeMoAutoModelBiencoder

    return NeMoAutoModelBiencoder.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        share_encoder=True,
        add_linear_pooler=False,
        out_dimension=768,
        do_gradient_checkpointing=False,
        pooling="avg",
        l2_normalize=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        torch_dtype=torch.bfloat16,
    ).to(device).eval()


def _build_eval_dataset():
    """Load the evaluation dataset."""
    from nemo_automodel.components.datasets.llm import retrieval_dataset_inline as rdi

    return rdi.make_retrieval_dataset(
        data_dir_list=TEST_DATA_JSONL,
        data_type="eval",
        train_n_passages=2,
        eval_negative_size=1,
        do_shuffle=False,
    )


def _build_collator():
    """Build tokenizer and collator for evaluation."""
    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
    from nemo_automodel.components.datasets.llm import RetrievalBiencoderCollator

    tokenizer = NeMoAutoTokenizer.from_pretrained(BASE_MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    collator = RetrievalBiencoderCollator(
        tokenizer=tokenizer,
        q_max_len=EVAL_MAX_LENGTH,
        p_max_len=EVAL_MAX_LENGTH,
        query_prefix="",
        passage_prefix="",
        padding="longest",
        pad_to_multiple_of=1,
    )
    return collator


def _iter_batches(ds, batch_size: int, max_samples: int):
    n = min(len(ds), max_samples)
    batch = []
    for i in range(n):
        batch.append(ds[i])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


@torch.no_grad()
def _compute_pos_neg_diffs(model, collator, ds, device, batch_size, max_samples):
    """Compute per-sample (pos_score - neg_score) diffs."""
    from nemo_automodel.recipes.biencoder.train_biencoder import contrastive_scores_and_labels

    model.eval()
    diffs: list[np.ndarray] = []

    for batch_examples in _iter_batches(ds, batch_size=batch_size, max_samples=max_samples):
        batch = collator(batch_examples)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        query = {k[2:]: v for k, v in batch.items() if k.startswith("q_")}
        passage = {k[2:]: v for k, v in batch.items() if k.startswith("d_")}

        q_reps = model.encode(query, encoder="query")
        p_reps = model.encode(passage, encoder="passage")

        # 2 passages per query: 1 positive + 1 negative
        n_passages = 2
        scores, _ = contrastive_scores_and_labels(q_reps, p_reps, n_passages)

        # [batch, n_passages]
        assert scores is not None and scores.shape[-1] >= 2, (
            f"Unexpected scores shape: {None if scores is None else tuple(scores.shape)}"
        )
        diff = (scores[:, 0] - scores[:, 1]).float().detach().cpu().numpy()
        diffs.append(diff)

    result = np.concatenate(diffs, axis=0) if diffs else np.array([], dtype=np.float32)
    assert result.size > 0, "No diffs computed (empty dataset?)"
    assert np.isfinite(result).all(), "Non-finite diffs found"
    return result


def _load_finetuned_weights(model, checkpoint_dir: Path):
    """Load fine-tuned weights into an existing model instance.

    The training recipe saves via ``Checkpointer.save_model`` which writes
    safetensors into a ``model/`` sub-directory under the checkpoint step
    folder.  The ``BiencoderStateDictAdapter`` on the model handles the
    key translations (lm_q.* <-> model.*) so no explicit ``key_mapping``
    is needed here.
    """
    import glob as _glob

    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

    model_dir = checkpoint_dir / "model"
    st = _glob.glob(str(checkpoint_dir / "**" / "*.safetensors"), recursive=True)
    assert st, f"No .safetensors found under {checkpoint_dir}"

    ckpt_cfg = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(checkpoint_dir),
        model_save_format="safetensors",
        model_cache_dir="/tmp",
        model_repo_id="__local__",
        save_consolidated=False,
        is_peft=False,
    )
    checkpointer = Checkpointer(config=ckpt_cfg, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
    checkpointer.load_model(model, model_path=str(model_dir))
    checkpointer.close()
    return model


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestCustomizerRetrieval:
    """End-to-end: train biencoder with customizer-aligned recipe, then assert
    the fine-tuned model is not degraded vs baseline."""

    @pytest.fixture(autouse=True)
    def _cleanup(self):
        """Remove training checkpoints after each test."""
        yield
        ckpt_dir = Path(CHECKPOINT_DIR)
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir, ignore_errors=True)

    def test_biencoder_finetuning_not_degraded(self):
        # ---- Step 1: Train ------------------------------------------------
        checkpoint_dir = _run_training()

        # ---- Step 2: Initialize torch.distributed for eval ----------------
        from nemo_automodel.components.distributed.init_utils import initialize_distributed

        dist = initialize_distributed(backend="nccl", timeout_minutes=5)
        device = dist.device if dist.device is not None else torch.device("cpu")

        # ---- Step 3: Build eval infrastructure ----------------------------
        model = _build_eval_model(device)
        ds = _build_eval_dataset()
        collator = _build_collator()
        max_samples = len(ds)

        # ---- Step 4: Compute baseline diffs -------------------------------
        base_diffs = _compute_pos_neg_diffs(
            model=model, collator=collator, ds=ds,
            device=device, batch_size=EVAL_BATCH_SIZE, max_samples=max_samples,
        )

        # ---- Step 5: Load fine-tuned weights & recompute ------------------
        model = _load_finetuned_weights(model, checkpoint_dir)
        ft_diffs = _compute_pos_neg_diffs(
            model=model, collator=collator, ds=ds,
            device=device, batch_size=EVAL_BATCH_SIZE, max_samples=max_samples,
        )

        # ---- Step 6: Statistical comparison -------------------------------
        import scipy.stats

        t_stat, p_value = scipy.stats.ttest_rel(base_diffs, ft_diffs)
        if not np.isfinite(p_value):
            p_value = 1.0

        delta = ft_diffs - base_diffs
        denom = float(np.std(delta, ddof=1))
        cohen_d = float(np.mean(delta) / denom) if denom > 0 else 0.0

        print(f"\nBaseline mean(diff): {base_diffs.mean():.6f}")
        print(f"Fine-tuned mean(diff): {ft_diffs.mean():.6f}")
        print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}, CohenD={cohen_d:.4f}")

        # Pass when: not statistically significant (p > 0.05), OR
        # significant AND the fine-tuned model is *better* (cohen_d > 0).
        model_not_degraded = (p_value > 0.05) or (p_value < 0.05 and cohen_d > 0)
        assert model_not_degraded, (
            f"Fine-tuned model appears degraded vs baseline "
            f"(t={t_stat:.4f}, p={p_value:.6f}, CohenD={cohen_d:.4f})"
        )
