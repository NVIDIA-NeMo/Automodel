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

"""Train -> checkpoint -> reload via automodel & vanilla HF from consolidated, verify logits match via KL divergence.

Launch: torchrun --nproc-per-node=<N> -m pytest <this_file> -c <config.yaml>
    [--kl_threshold <float>] [--hf_kl_threshold <float>]
    [--cross_tp_size <int>] [--cross_tp_kl_threshold <float>]
"""

from __future__ import annotations

import sys
from pathlib import Path

import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

datasets.disable_caching()

# Llama token IDs for "The quick brown fox jumps over the lazy dog"
INPUT_IDS = [791, 4996, 14198, 39935, 35308, 927, 279, 16053, 5679]


def _extract_custom_args(argv):
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {"--kl_threshold", "--hf_kl_threshold", "--cross_tp_size", "--cross_tp_kl_threshold"}
    custom = {}
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] in custom_keys:
            custom[argv[i].lstrip("-")] = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1
    return custom, remaining


def _kl_divergence_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """Per-token KL(reference || candidate) for full [B, T, V] logits."""
    assert reference_logits.shape == candidate_logits.shape
    vocab_size = reference_logits.shape[-1]
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_log_probs = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    return F.kl_div(cand_log_probs, ref_log_probs, reduction="none", log_target=True).sum(-1)


def _get_logits(model, input_ids, device) -> torch.Tensor:
    """Forward pass returning float32 logits on CPU."""
    model.eval()
    ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return logits.float().cpu()


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def test_checkpoint_robustness():
    """Train -> checkpoint -> reload automodel from consolidated -> reload vanilla HF, compare logits."""
    custom_args, config_argv = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + config_argv
    kl_threshold = float(custom_args.get("kl_threshold", "0"))
    hf_kl_threshold = float(custom_args.get("hf_kl_threshold", "5e-3"))
    cross_tp_size = int(custom_args.get("cross_tp_size", "0"))
    cross_tp_kl_threshold = float(custom_args.get("cross_tp_kl_threshold", "5e-3"))

    # Phase 1: Train and checkpoint
    cfg = parse_args_and_load_config()
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # Phase 2: Capture reference logits before teardown
    device = next(trainer.model_parts[0].parameters()).device
    reference_logits = _get_logits(trainer.model_parts[0], INPUT_IDS, device)

    # Phase 3: Reload automodel from consolidated checkpoint
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    ckpt_step_dirs = sorted(checkpoint_dir.glob("epoch_*_step_*"))
    assert len(ckpt_step_dirs) > 0, f"No checkpoint subdirectories found under {checkpoint_dir}"
    ckpt_step_dir = ckpt_step_dirs[-1]
    consolidated_dir = ckpt_step_dir / "model" / "consolidated"

    is_peft = hasattr(cfg, "peft")
    original_pretrained_path = cfg.model.pretrained_model_name_or_path

    del trainer
    torch.cuda.empty_cache()

    cfg = parse_args_and_load_config()
    if not is_peft:
        cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
        cfg.checkpoint.enabled = False
    restored_trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    restored_trainer.setup()

    restored_logits = _get_logits(restored_trainer.model_parts[0], INPUT_IDS, device)

    kl_restored = _kl_divergence_from_logits(reference_logits, restored_logits)
    max_kl_restored = kl_restored.max().item()
    if _rank0():
        print(f"\n[Phase 3] Automodel-from-consolidated max KL: {max_kl_restored:.6e} (threshold: {kl_threshold:.6e})")
    assert max_kl_restored <= kl_threshold, (
        f"KL divergence between original and automodel-from-consolidated too large: "
        f"max per-token KL = {max_kl_restored:.6e} > threshold {kl_threshold:.6e}"
    )

    # Phase 4: Load into vanilla HF (rank 0 only)
    del restored_trainer
    torch.cuda.empty_cache()

    if _rank0():
        from transformers import AutoModelForCausalLM

        if is_peft:
            from peft import PeftModel

            base_model = AutoModelForCausalLM.from_pretrained(
                original_pretrained_path, torch_dtype=torch.bfloat16
            ).to(device)
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt_step_dir / "model"))
            hf_logits = _get_logits(peft_model, INPUT_IDS, device)
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(str(consolidated_dir), torch_dtype=torch.bfloat16).to(
                device
            )
            hf_logits = _get_logits(hf_model, INPUT_IDS, device)

        kl_hf = _kl_divergence_from_logits(reference_logits, hf_logits)
        max_kl_hf = kl_hf.max().item()
        print(f"[Phase 4] HF-loaded max KL: {max_kl_hf:.6e} (threshold: {hf_kl_threshold:.6e})")
        assert max_kl_hf <= hf_kl_threshold, (
            f"KL divergence between original and HF-loaded model too large: "
            f"max per-token KL = {max_kl_hf:.6e} > threshold {hf_kl_threshold:.6e}"
        )

    _barrier()

    # Phase 5 (optional): Cross-TP — reload consolidated with a different TP size
    if cross_tp_size > 0 and not is_peft:
        cfg = parse_args_and_load_config()
        cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
        cfg.checkpoint.enabled = False
        cfg.distributed.tp_size = cross_tp_size
        cfg.distributed.dp_size = None
        cross_tp_trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
        cross_tp_trainer.setup()

        cross_tp_logits = _get_logits(cross_tp_trainer.model_parts[0], INPUT_IDS, device)

        kl_cross_tp = _kl_divergence_from_logits(reference_logits, cross_tp_logits)
        max_kl_cross_tp = kl_cross_tp.max().item()
        if _rank0():
            print(
                f"[Phase 5] Cross-TP (tp_size={cross_tp_size}) max KL: "
                f"{max_kl_cross_tp:.6e} (threshold: {cross_tp_kl_threshold:.6e})"
            )
        assert max_kl_cross_tp <= cross_tp_kl_threshold, (
            f"KL divergence between original and cross-TP model too large: "
            f"max per-token KL = {max_kl_cross_tp:.6e} > threshold {cross_tp_kl_threshold:.6e}"
        )

        del cross_tp_trainer
        torch.cuda.empty_cache()
        _barrier()
