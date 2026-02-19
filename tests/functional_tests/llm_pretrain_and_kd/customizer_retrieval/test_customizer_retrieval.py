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

ONNX_OUTPUT_DIR = os.environ.get(
    "ONNX_OUTPUT_DIR",
    "/workspace/output/biencoder_inline/onnx",
)


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
    """Build a NeMoAutoModelBiencoder for evaluation with the native inference
    temperature (t=0.02)."""
    from nemo_automodel.components.models.biencoder import NeMoAutoModelBiencoder

    return NeMoAutoModelBiencoder.from_pretrained(
        pretrained_model_name_or_path=BASE_MODEL_PATH,
        share_encoder=True,
        add_linear_pooler=False,
        out_dimension=768,
        do_gradient_checkpointing=False,
        train_n_passages=2,
        eval_negative_size=1,
        pooling="avg",
        l2_normalize=True,
        t=EVAL_TEMPERATURE,
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
    model.eval()
    diffs: list[np.ndarray] = []

    for batch_examples in _iter_batches(ds, batch_size=batch_size, max_samples=max_samples):
        batch = collator(batch_examples)
        batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

        query = {k[2:]: v for k, v in batch.items() if k.startswith("q_")}
        passage = {k[2:]: v for k, v in batch.items() if k.startswith("d_")}

        out = model(query=query, passage=passage)
        scores = out.scores  # [batch, n_passages]
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
    """Load fine-tuned weights into an existing model instance."""
    import glob as _glob

    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig

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
    key_mapping = {r"^(?!(model\.|linear_pooler\.))": "model."}
    checkpointer.load_model(model, model_path=str(checkpoint_dir), key_mapping=key_mapping)
    checkpointer.close()
    return model


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestCustomizerRetrieval:
    """End-to-end: train biencoder with customizer-aligned recipe, then assert
    the fine-tuned model is not degraded vs baseline, and the ONNX export
    produces valid embeddings."""

    # -- Fixtures -----------------------------------------------------------

    @pytest.fixture(scope="class")
    def checkpoint_dir(self):
        """Run training once for all tests in this class and return the
        checkpoint path.  Clean up after all tests complete."""
        ckpt = _run_training()
        yield ckpt
        # Cleanup after all tests in the class.
        ckpt_root = Path(CHECKPOINT_DIR)
        if ckpt_root.exists():
            shutil.rmtree(ckpt_root, ignore_errors=True)
        onnx_dir = Path(ONNX_OUTPUT_DIR)
        if onnx_dir.exists():
            shutil.rmtree(onnx_dir, ignore_errors=True)

    @pytest.fixture(scope="class")
    def dist_device(self):
        """Initialize torch.distributed once for the class and return device."""
        from nemo_automodel.components.distributed.init_utils import initialize_distributed

        dist = initialize_distributed(backend="nccl", timeout_minutes=5)
        return dist.device if dist.device is not None else torch.device("cpu")

    # -- Test: finetuned model not degraded ---------------------------------

    def test_biencoder_finetuning_not_degraded(self, checkpoint_dir, dist_device):
        device = dist_device

        # Build eval infrastructure.
        model = _build_eval_model(device)
        ds = _build_eval_dataset()
        collator = _build_collator()
        max_samples = len(ds)

        # Compute baseline diffs.
        base_diffs = _compute_pos_neg_diffs(
            model=model, collator=collator, ds=ds,
            device=device, batch_size=EVAL_BATCH_SIZE, max_samples=max_samples,
        )

        # Load fine-tuned weights and recompute.
        model = _load_finetuned_weights(model, checkpoint_dir)
        ft_diffs = _compute_pos_neg_diffs(
            model=model, collator=collator, ds=ds,
            device=device, batch_size=EVAL_BATCH_SIZE, max_samples=max_samples,
        )

        # Statistical comparison.
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

    # -- Test: ONNX export + verification -----------------------------------

    def test_onnx_export_and_verify(self, checkpoint_dir):
        """Export the fine-tuned checkpoint to ONNX and verify the graph
        produces valid, finite, correctly-shaped embeddings."""
        import onnxruntime
        from transformers import AutoTokenizer

        from nemo_automodel.components.models.biencoder.export_onnx import export_to_onnx

        onnx_path = export_to_onnx(
            model_path=str(checkpoint_dir),
            output_dir=ONNX_OUTPUT_DIR,
            tokenizer_path=BASE_MODEL_PATH,
            pooling="avg",
            normalize=True,
            opset=17,
            export_dtype="fp32",
            verify=False,  # we do our own checks below
        )

        # 1. File exists.
        assert Path(onnx_path).exists(), f"ONNX model not found at {onnx_path}"

        # 2. Tokenizer was saved alongside.
        tokenizer_dir = Path(ONNX_OUTPUT_DIR) / "tokenizer"
        assert tokenizer_dir.exists(), f"Tokenizer dir not found at {tokenizer_dir}"

        # 3. Load the ONNX model in onnxruntime.
        session = onnxruntime.InferenceSession(onnx_path)
        input_names = [inp.name for inp in session.get_inputs()]
        output_names = [out.name for out in session.get_outputs()]

        assert "input_ids" in input_names, f"Expected 'input_ids' in inputs, got {input_names}"
        assert "attention_mask" in input_names, f"Expected 'attention_mask' in inputs, got {input_names}"
        assert "embeddings" in output_names, f"Expected 'embeddings' in outputs, got {output_names}"

        # 4. Run inference on sample sentences.
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), trust_remote_code=True)
        sentences = [
            "What is deep learning?",
            "Neural networks are a type of machine learning model.",
            "The cat sat on the mat.",
        ]
        tokenized = tokenizer(sentences, return_tensors="np", padding=True, truncation=True)
        feed = {name: tokenized[name] for name in input_names if name in tokenized}

        outputs = session.run(output_names, feed)
        embeddings = outputs[0]

        print(f"\nONNX output shape: {embeddings.shape}")
        print(f"ONNX embedding[0][:8]: {embeddings[0][:8]}")

        # 5. Shape: [batch_size, hidden_dim].
        assert embeddings.ndim == 2, f"Expected 2-D output, got shape {embeddings.shape}"
        assert embeddings.shape[0] == len(sentences), (
            f"Batch mismatch: expected {len(sentences)}, got {embeddings.shape[0]}"
        )

        # 6. All values are finite.
        assert np.isfinite(embeddings).all(), "ONNX output contains non-finite values"

        # 7. Embeddings are L2-normalised (norm ≈ 1.0 per row).
        norms = np.linalg.norm(embeddings, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-4, err_msg="Embeddings are not L2-normalised")

        # 8. Different sentences should produce different embeddings.
        cos_01 = float(np.dot(embeddings[0], embeddings[1]))
        cos_02 = float(np.dot(embeddings[0], embeddings[2]))
        print(f"Cosine(0,1)={cos_01:.4f}  Cosine(0,2)={cos_02:.4f}")
        assert not np.allclose(embeddings[0], embeddings[1], atol=1e-3), (
            "Different sentences produced identical embeddings"
        )
