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

Launch: torchrun --nproc-per-node=<N> -m <this_module> --config <config.yaml>
    [--cosine_threshold <float>] [--hf_cosine_threshold <float>]
    [--check_hf_reload] [--check_resume]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe
from tests.functional_tests.checkpoint_robustness.test_checkpoint_robustness_llm import (
    _finish_hf_reload_sync,
    _prepare_hf_reload_sync,
    _raise_distributed_failure,
)
from tests.functional_tests.checkpoint_robustness.resume_trajectory import (
    _TrajectoryRecorder,
    _checkpoint_for_completed_steps,
    _checkpoint_state_snapshot,
    _configure_resumed_run,
    _configure_uninterrupted_run,
    _gather_rank_failures,
    _load_reference_trajectory,
    _persist_reference_trajectory,
    _restored_state_mismatch,
    _resume_plan_from_config,
    _trajectory_mismatch,
)

# Default test sentence for embedding extraction
_DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog"


def _extract_custom_args(argv: list[str]) -> tuple[dict[str, str | bool], list[str]]:
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {
        "--cosine_threshold",
        "--hf_cosine_threshold",
        "--resume_first_loss_threshold",
        "--resume_loss_threshold",
    }
    boolean_keys = {"--check_hf_reload", "--check_resume"}
    custom: dict[str, str | bool] = {}
    remaining: list[str] = []
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

    config_path = None
    for index, arg in enumerate(remaining):
        if arg == "--config" and index + 1 < len(remaining):
            config_path = remaining[index + 1]
            break
    if config_path:
        import yaml

        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f)
        ci_robustness = raw_cfg.get("ci", {}).get("checkpoint_robustness") or {}
        for cli_key in custom_keys | boolean_keys:
            key = cli_key.lstrip("-")
            if key in custom or key not in ci_robustness:
                continue
            value = ci_robustness[key]
            if isinstance(value, bool):
                if value:
                    custom[key] = True
            else:
                custom[key] = str(value)
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


def _get_hf_style_embeddings(model, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return query and image-document embeddings from a raw HF-compatible backbone."""
    model = getattr(model, "module", model)
    backbone = getattr(model, "model", model)
    backbone.eval()
    image = Image.new("RGB", (64, 64), color=(32, 96, 160))
    with torch.no_grad():
        query_embeddings = backbone.encode_queries([prompt])
        document_embeddings = backbone.encode_documents(images=[image])
    return query_embeddings.float().cpu(), document_embeddings.float().cpu()


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
    hf_cosine_threshold = float(custom_args.get("hf_cosine_threshold", "0.999"))
    check_hf_reload = bool(custom_args.get("check_hf_reload", False))
    check_resume = bool(custom_args.get("check_resume", False))
    resume_first_loss_threshold = float(custom_args.get("resume_first_loss_threshold", "1e-6"))
    resume_loss_threshold = float(custom_args.get("resume_loss_threshold", "5e-3"))

    # ------------------------------------------------------------------
    # Phase 1: Train biencoder and checkpoint
    # ------------------------------------------------------------------
    torch.cuda.reset_peak_memory_stats()
    cfg = parse_args_and_load_config()
    resume_plan = _resume_plan_from_config(cfg) if check_resume else None
    if resume_plan is not None:
        _configure_uninterrupted_run(cfg, resume_plan)
    trainer = TrainBiEncoderRecipe(cfg)
    trainer.setup()
    resume_recorder = None
    if resume_plan is not None:
        resume_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=True)
        resume_recorder.attach(trainer)
    trainer.run_train_validation_loop()
    if resume_recorder is not None:
        _persist_reference_trajectory(resume_recorder)
        _barrier()
        if _rank0():
            print(
                "[Resume correctness] Retrieval Phase 1 persisted the exact uninterrupted continuation"
            )
            print(
                "[Training reproducibility] Not evaluated in the retrieval resume phase; independent runs are "
                "not a checkpoint-restore oracle"
            )

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
    hf_reference_query = None
    hf_reference_document = None
    if check_hf_reload:
        hf_reference_query, hf_reference_document = _get_hf_style_embeddings(trainer.model_parts[0], _DEFAULT_PROMPT)
    if _rank0():
        print(f"\n[Phase 2] Reference embedding shape: {reference_embeddings.shape}")
        print(f"[Phase 2] Reference embedding norm: {reference_embeddings.norm().item():.6f}")

    # ------------------------------------------------------------------
    # Phase 3: Reload from consolidated checkpoint, compare embeddings
    # ------------------------------------------------------------------
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    if resume_plan is not None:
        ckpt_step_dir = _checkpoint_for_completed_steps(resume_plan, resume_plan.final_max_steps)
    else:
        ckpt_step_dirs = list(checkpoint_dir.glob("epoch_*_step_*"))
        assert ckpt_step_dirs, f"No checkpoint subdirectories found under {checkpoint_dir}"
        ckpt_step_dir = max(
            ckpt_step_dirs,
            key=lambda path: tuple(int(part) for part in (path.name.split("_")[1], path.name.split("_")[3])),
        )
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
    # Phase 4 (optional): Reload with vanilla Hugging Face AutoModel
    # ------------------------------------------------------------------
    if check_hf_reload:
        hf_reload_sync_paths = _prepare_hf_reload_sync(cfg)
        hf_reload_error = None
        if _rank0():
            try:
                from transformers import AutoModel

                assert hf_reference_query is not None
                assert hf_reference_document is not None
                hf_model = AutoModel.from_pretrained(
                    str(consolidated_dir),
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                ).to(device)
                hf_query, hf_document = _get_hf_style_embeddings(hf_model, _DEFAULT_PROMPT)
                query_cosine_sim = _cosine_similarity(hf_reference_query, hf_query)
                document_cosine_sim = _cosine_similarity(hf_reference_document, hf_document)
                print(
                    f"\n[Phase 4] HF reload query cosine similarity: {query_cosine_sim:.6f}; "
                    f"image-document cosine similarity: {document_cosine_sim:.6f} "
                    f"(threshold: {hf_cosine_threshold})"
                )
                if query_cosine_sim < hf_cosine_threshold:
                    hf_reload_error = (
                        f"HF-reloaded query embedding cosine similarity too low: "
                        f"{query_cosine_sim:.6f} < threshold {hf_cosine_threshold}"
                    )
                if document_cosine_sim < hf_cosine_threshold:
                    document_error = (
                        f"HF-reloaded image-document embedding cosine similarity too low: "
                        f"{document_cosine_sim:.6f} < threshold {hf_cosine_threshold}"
                    )
                    hf_reload_error = "\n".join(filter(None, (hf_reload_error, document_error)))
                del hf_model
                torch.cuda.empty_cache()
            except Exception as exc:
                hf_reload_error = f"Vanilla HF reload failed: {type(exc).__name__}: {exc}"
        hf_reload_error = _finish_hf_reload_sync(hf_reload_sync_paths, hf_reload_error)
        assert hf_reload_error is None, hf_reload_error

    # ------------------------------------------------------------------
    # Phase 5 (optional): restore the exact Phase 1 boundary and replay its continuation.
    # ------------------------------------------------------------------
    if check_resume:
        assert resume_plan is not None
        reference_trajectory = _load_reference_trajectory(resume_plan)
        checkpoint_path = _checkpoint_for_completed_steps(resume_plan, resume_plan.boundary_step)
        cfg = parse_args_and_load_config()
        _configure_resumed_run(cfg, resume_plan, checkpoint_path)
        resume_trainer = TrainBiEncoderRecipe(cfg)
        resume_trainer.setup()
        restored_state = _checkpoint_state_snapshot(resume_trainer, state_is_being_saved=False)
        local_failure = _restored_state_mismatch(reference_trajectory["boundary_state"], restored_state)
        failure_message = _gather_rank_failures(local_failure, check="restored_state")
        _raise_distributed_failure(failure_message)

        resumed_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=False)
        resumed_recorder.attach(resume_trainer)
        resume_trainer.run_train_validation_loop()
        resumed_trajectory = resumed_recorder.to_dict()
        local_failure = _trajectory_mismatch(
            reference_trajectory,
            resumed_trajectory,
            first_loss_threshold=resume_first_loss_threshold,
            later_loss_threshold=resume_loss_threshold,
        )
        failure_message = _gather_rank_failures(local_failure, check="shared_trajectory")
        _raise_distributed_failure(failure_message)
        if _rank0():
            print(
                f"[Resume correctness] Retrieval shared trajectory verified for "
                f"{resume_plan.continuation_steps} steps; first-step threshold={resume_first_loss_threshold:.3e}, "
                f"later-step threshold={resume_loss_threshold:.3e}"
            )

        del resume_trainer
        torch.cuda.empty_cache()
        _barrier()


if __name__ == "__main__":
    test_checkpoint_robustness_biencoder()
