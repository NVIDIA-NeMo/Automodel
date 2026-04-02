# Copyright (c) 2026, NVIDIA CORPORATION.
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

"""Train biencoder -> checkpoint -> reload from consolidated, verify embeddings match via cosine similarity.

Biencoder models (e.g. nvidia/llama-nemotron-embed-1b-v2) output embeddings rather than
next-token logits, so we compare checkpoint fidelity using cosine similarity instead of
KL divergence.

Launch: torchrun --nproc-per-node=<N> -m pytest <this_file> -c <config.yaml>
    [--cosine_threshold <float>] [--check_resume]
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe

# Default test sentence for embedding extraction
_DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog"


def _extract_custom_args(argv):
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {
        "--cosine_threshold",
        "--resume_loss_threshold",
    }
    boolean_keys = {"--check_resume"}
    custom = {}
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] in custom_keys:
            custom[argv[i].lstrip("-")] = argv[i + 1]
            i += 2
        elif argv[i] in boolean_keys:
            custom[argv[i].lstrip("-")] = True
            i += 1
        else:
            remaining.append(argv[i])
            i += 1
    return custom, remaining


def _rss_gb() -> float:
    """Current RSS in GB from /proc/self/statm."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * page_size / 1024**3


def _get_embeddings(model, tokenizer, prompt: str, device) -> torch.Tensor:
    """Forward pass returning float32 embeddings on CPU.

    Tokenizes the prompt as a query and runs the biencoder query encoder
    to produce a single embedding vector.
    """
    model.eval()
    # Use underlying HF tokenizer to avoid NeMoAutoTokenizer's _add_token issue with return_tensors="pt"
    hf_tokenizer = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    encoded = hf_tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_dict = {k: v.to(device) for k, v in encoded.items()}
    with torch.no_grad():
        embeddings = model(input_dict, encoder="query")
    return embeddings.float().cpu()


def _cosine_similarity(ref: torch.Tensor, cand: torch.Tensor) -> float:
    """Compute cosine similarity between two embedding tensors."""
    return F.cosine_similarity(ref.flatten().unsqueeze(0), cand.flatten().unsqueeze(0)).item()


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def test_checkpoint_robustness_biencoder():
    """Train biencoder -> checkpoint -> reload from consolidated, compare embeddings."""
    custom_args, config_argv = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + config_argv
    cosine_threshold = float(custom_args.get("cosine_threshold", "0.999"))
    check_resume = bool(custom_args.get("check_resume", False))
    resume_loss_threshold = float(custom_args.get("resume_loss_threshold", "5e-3"))

    # ------------------------------------------------------------------
    # Phase 1: Train biencoder and checkpoint
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    cfg = parse_args_and_load_config()
    trainer = TrainBiEncoderRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_cpu_gb = _rss_gb()
    if _rank0():
        print(f"\n[Memory] Peak VRAM: {peak_vram_gb:.2f} GB, Peak CPU RSS: {peak_cpu_gb:.2f} GB")

    # ------------------------------------------------------------------
    # Phase 2: Capture reference embeddings before teardown
    # ------------------------------------------------------------------
    device = next(trainer.model_parts[0].parameters()).device
    tokenizer = trainer.tokenizer
    reference_embeddings = _get_embeddings(trainer.model_parts[0], tokenizer, _DEFAULT_PROMPT, device)
    if _rank0():
        print(f"\n[Phase 2] Reference embedding shape: {reference_embeddings.shape}")
        print(f"[Phase 2] Reference embedding norm: {reference_embeddings.norm().item():.6f}")

    # ------------------------------------------------------------------
    # Phase 3: Reload from consolidated checkpoint, compare embeddings
    # ------------------------------------------------------------------
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    ckpt_step_dirs = sorted(checkpoint_dir.glob("epoch_*_step_*"))
    assert len(ckpt_step_dirs) > 0, f"No checkpoint subdirectories found under {checkpoint_dir}"
    ckpt_step_dir = ckpt_step_dirs[-1]
    consolidated_dir = ckpt_step_dir / "model" / "consolidated"

    del trainer
    torch.cuda.empty_cache()

    cfg = parse_args_and_load_config()
    cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
    cfg.checkpoint.enabled = False
    restored_trainer = TrainBiEncoderRecipe(cfg)
    restored_trainer.setup()

    restored_embeddings = _get_embeddings(
        restored_trainer.model_parts[0], restored_trainer.tokenizer, _DEFAULT_PROMPT, device
    )

    cosine_sim = _cosine_similarity(reference_embeddings, restored_embeddings)
    if _rank0():
        print(
            f"\n[Phase 3] Cosine similarity (original vs consolidated): "
            f"{cosine_sim:.6f} (threshold: {cosine_threshold})"
        )
    assert cosine_sim >= cosine_threshold, (
        f"Cosine similarity between original and consolidated embeddings too low: "
        f"{cosine_sim:.6f} < threshold {cosine_threshold}"
    )

    del restored_trainer
    torch.cuda.empty_cache()
    _barrier()

    # ------------------------------------------------------------------
    # Phase 4 (optional): Training resumption -- verify loss continuity
    # ------------------------------------------------------------------
    if check_resume:
        # Baseline: fresh continuous run for max_steps+3, saving losses
        baseline_dir = tempfile.mkdtemp(prefix="resume_baseline_biencoder_")
        cfg = parse_args_and_load_config()
        original_max_steps = cfg.step_scheduler.max_steps
        resume_max_steps = original_max_steps + 3
        cfg.step_scheduler.max_steps = resume_max_steps
        cfg.checkpoint.checkpoint_dir = baseline_dir
        cfg.checkpoint.enabled = False
        baseline_trainer = TrainBiEncoderRecipe(cfg)
        baseline_trainer.setup()
        baseline_trainer.run_train_validation_loop()

        baseline_losses = {}
        baseline_jsonl = Path(baseline_dir) / "training.jsonl"
        if _rank0() and baseline_jsonl.exists():
            with open(baseline_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["step"] >= original_max_steps:
                        baseline_losses[entry["step"]] = entry["loss"]

        del baseline_trainer
        torch.cuda.empty_cache()
        shutil.rmtree(baseline_dir, ignore_errors=True)

        # Resume: reload from Phase 1 checkpoint and train to resume_max_steps
        cfg = parse_args_and_load_config()
        cfg.checkpoint.restore_from = str(ckpt_step_dir)
        cfg.step_scheduler.max_steps = resume_max_steps
        resume_trainer = TrainBiEncoderRecipe(cfg)
        resume_trainer.setup()
        resume_trainer.run_train_validation_loop()

        # Compare losses at the overlapping steps
        resume_jsonl = checkpoint_dir / "training.jsonl"
        if _rank0():
            assert baseline_losses, "Phase 4: baseline_losses is empty -- no steps to compare"
            assert resume_jsonl.exists(), f"Phase 4: {resume_jsonl} not found"

            resume_losses = {}
            with open(resume_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["step"] in baseline_losses:
                        resume_losses[entry["step"]] = entry["loss"]

            matched_steps = 0
            for step in sorted(baseline_losses):
                if step in resume_losses:
                    matched_steps += 1
                    bl = baseline_losses[step]
                    rl = resume_losses[step]
                    diff = abs(bl - rl)
                    print(
                        f"[Phase 4] Step {step}: baseline_loss={bl:.6f}, "
                        f"resume_loss={rl:.6f}, diff={diff:.6e}"
                    )
                    assert diff < resume_loss_threshold, (
                        f"Contrastive loss mismatch after resume at step {step}: "
                        f"baseline={bl:.6f}, resume={rl:.6f}, diff={diff:.6e}"
                    )

            assert matched_steps > 0, (
                f"Phase 4: no overlapping steps found between baseline ({sorted(baseline_losses.keys())}) "
                f"and resume ({sorted(resume_losses.keys())})"
            )
            print(f"[Phase 4] Training resumption verified ({matched_steps} steps compared)")

        del resume_trainer
        torch.cuda.empty_cache()
        _barrier()
