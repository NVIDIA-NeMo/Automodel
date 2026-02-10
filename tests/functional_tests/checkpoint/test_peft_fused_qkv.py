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

"""Functional test: LoRA + fused QKV checkpoint save / resume / HF-PEFT restore.

This test exercises the full train_ft loop with a *tiny* Llama model (2 layers,
random weights) that uses combined ``qkv_proj`` and ``gate_up_proj`` projections.
It verifies:

1. Train for 2 steps, save a PEFT checkpoint.
2. Resume from that checkpoint and confirm the LoRA weights match exactly.
3. The saved ``adapter_model.safetensors`` contains only split HF-compatible
   projection names (``q_proj``, ``k_proj``, ``v_proj``, ``gate_proj``,
   ``up_proj``) — no combined names (``qkv_proj``, ``gate_up_proj``).
4. HuggingFace PEFT (``PeftModel.from_pretrained``) can load the adapter
   without errors and produces a working model.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import PeftModel
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, LlamaConfig

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    calculate_loss,
)

import datasets

datasets.disable_caching()


# ---------------------------------------------------------------------------
# Tiny Llama config (random weights, no download needed)
# ---------------------------------------------------------------------------
TINY_LLAMA_CONFIG = dict(
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    hidden_size=64,
    intermediate_size=128,
    vocab_size=256,
    max_position_embeddings=128,
)


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_validation_loss(
    model: nn.Module,
    val_batch: dict[str, torch.Tensor],
    loss_fn: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
    model.eval()
    labels = val_batch.pop("labels")
    with torch.no_grad():
        out = model(**val_batch)
        return calculate_loss(loss_fn, logits=out.logits, labels=labels)


def _collect_lora_params(model_parts) -> dict[str, torch.Tensor]:
    """Collect LoRA trainable parameters (on CPU) from model parts."""
    sd = ModelState(model_parts, is_peft=True).state_dict()
    return {k: v.cpu().clone() for k, v in sd.items()}


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------


def test_peft_fused_qkv_checkpoint():
    """End-to-end: train 2 steps ➜ save ckpt ➜ resume ckpt ➜ verify HF PEFT load."""

    script_dir = Path(__file__).parent.resolve()
    # Re-use the existing squad PEFT config as a base; we override everything we
    # need via CLI-style arguments baked into the config object.
    base_cfg_path = (
        Path(__file__).parents[3]
        / "examples"
        / "llm_finetune"
        / "llama3_2"
        / "llama3_2_1b_squad_peft.yaml"
    )
    cfg = parse_args_and_load_config(base_cfg_path)

    # ----- Override config for a tiny, random Llama -------------------------
    # Use from_config (random init) with our tiny LlamaConfig so we don't
    # need to download any pretrained weights.
    cfg.model._target_ = "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    # Remove pretrained_model_name_or_path if present
    if hasattr(cfg.model, "pretrained_model_name_or_path"):
        del cfg.model.pretrained_model_name_or_path
    # Inject the tiny config
    cfg.model.config = LlamaConfig(**TINY_LLAMA_CONFIG)

    # PEFT: LoRA on all linear layers (includes fused qkv_proj & gate_up_proj)
    cfg.peft.match_all_linear = True
    cfg.peft.dim = 4
    cfg.peft.alpha = 16
    cfg.peft.use_triton = False

    # Step scheduler: 2 training steps, checkpoint at step 2
    cfg.step_scheduler.max_steps = 2
    cfg.step_scheduler.global_batch_size = 2
    cfg.step_scheduler.local_batch_size = 2
    cfg.step_scheduler.ckpt_every_steps = 2
    cfg.step_scheduler.num_epochs = 1
    # Disable validation to keep things fast
    cfg.step_scheduler.val_every_steps = 999999

    # Checkpointing
    ckpt_dir = "checkpoints_peft_fused_qkv_test/"
    cfg.checkpoint = cfg.get("checkpoint", {})
    cfg.checkpoint.enabled = True
    cfg.checkpoint.checkpoint_dir = ckpt_dir

    # Distributed: single GPU, FSDP2
    cfg.distributed.dp_size = None
    cfg.distributed.tp_size = 1
    cfg.distributed.cp_size = 1

    # No LR scheduler for simplicity
    if hasattr(cfg, "lr_scheduler"):
        del cfg.lr_scheduler

    # ------------------------------------------------------------------
    # Phase 1: Train for 2 steps and save a checkpoint
    # ------------------------------------------------------------------
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # Collect LoRA params after training
    lora_params_after_train = _collect_lora_params(trainer.model_parts)
    assert len(lora_params_after_train) > 0, "Expected LoRA parameters to be present"

    # Verify checkpoint was saved
    ckpt_step_dir = Path(ckpt_dir) / "epoch_0_step_1"
    assert ckpt_step_dir.exists(), f"Checkpoint directory {ckpt_step_dir} does not exist"

    model_dir = ckpt_step_dir / "model"
    assert (model_dir / "adapter_model.safetensors").exists(), "adapter_model.safetensors not found"
    assert (model_dir / "adapter_config.json").exists(), "adapter_config.json not found"

    # ------------------------------------------------------------------
    # Phase 2: Verify saved adapter has NO combined-projection keys
    # ------------------------------------------------------------------
    saved_adapter_sd = load_file(str(model_dir / "adapter_model.safetensors"))

    combined_keys = [k for k in saved_adapter_sd if "qkv_proj" in k or "gate_up_proj" in k]
    assert combined_keys == [], (
        f"Saved adapter should NOT contain combined-projection keys, found: {combined_keys}"
    )

    # Verify split names ARE present
    has_q_proj = any("q_proj" in k for k in saved_adapter_sd)
    has_k_proj = any("k_proj" in k for k in saved_adapter_sd)
    has_v_proj = any("v_proj" in k for k in saved_adapter_sd)
    assert has_q_proj and has_k_proj and has_v_proj, (
        f"Expected split q/k/v projection keys in saved adapter. Keys: {list(saved_adapter_sd.keys())}"
    )

    # Verify adapter_config.json target_modules are split
    with open(model_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)
    for mod in adapter_config.get("target_modules", []):
        assert "qkv_proj" not in mod, f"Combined qkv_proj found in target_modules: {mod}"
        assert "gate_up_proj" not in mod, f"Combined gate_up_proj found in target_modules: {mod}"

    # ------------------------------------------------------------------
    # Phase 3: Resume from checkpoint, verify LoRA weights match
    # ------------------------------------------------------------------
    resume_cfg = parse_args_and_load_config(base_cfg_path)
    # Apply same overrides
    resume_cfg.model._target_ = "nemo_automodel.NeMoAutoModelForCausalLM.from_config"
    if hasattr(resume_cfg.model, "pretrained_model_name_or_path"):
        del resume_cfg.model.pretrained_model_name_or_path
    resume_cfg.model.config = LlamaConfig(**TINY_LLAMA_CONFIG)
    resume_cfg.peft.match_all_linear = True
    resume_cfg.peft.dim = 4
    resume_cfg.peft.alpha = 16
    resume_cfg.peft.use_triton = False
    resume_cfg.step_scheduler.max_steps = 2
    resume_cfg.step_scheduler.global_batch_size = 2
    resume_cfg.step_scheduler.local_batch_size = 2
    resume_cfg.step_scheduler.ckpt_every_steps = 2
    resume_cfg.step_scheduler.num_epochs = 1
    resume_cfg.step_scheduler.val_every_steps = 999999
    resume_cfg.checkpoint = resume_cfg.get("checkpoint", {})
    resume_cfg.checkpoint.enabled = True
    resume_cfg.checkpoint.checkpoint_dir = ckpt_dir
    resume_cfg.checkpoint.restore_from = str(ckpt_step_dir)
    resume_cfg.distributed.dp_size = None
    resume_cfg.distributed.tp_size = 1
    resume_cfg.distributed.cp_size = 1
    if hasattr(resume_cfg, "lr_scheduler"):
        del resume_cfg.lr_scheduler

    resumed_trainer = TrainFinetuneRecipeForNextTokenPrediction(resume_cfg)
    resumed_trainer.setup()

    # Collect LoRA params after resume (before running any more steps)
    lora_params_after_resume = _collect_lora_params(resumed_trainer.model_parts)

    # Verify the LoRA weights from training match the resumed weights exactly
    assert set(lora_params_after_train.keys()) == set(lora_params_after_resume.keys()), (
        "LoRA parameter key sets differ between trained and resumed models.\n"
        f"Only in trained: {set(lora_params_after_train.keys()) - set(lora_params_after_resume.keys())}\n"
        f"Only in resumed: {set(lora_params_after_resume.keys()) - set(lora_params_after_train.keys())}"
    )

    for key in lora_params_after_train:
        trained_val = lora_params_after_train[key]
        resumed_val = lora_params_after_resume[key]
        assert torch.allclose(trained_val, resumed_val, atol=1e-6), (
            f"LoRA parameter mismatch after resume for key: {key}\n"
            f"Max diff: {(trained_val - resumed_val).abs().max().item()}"
        )

    # Also verify the models produce the same validation loss
    val_batch = next(iter(trainer.val_dataloaders["default"]))
    loss_orig = _get_validation_loss(
        trainer.model_parts[0], val_batch, trainer.loss_fn, trainer.dist_env.device
    )
    loss_resumed = _get_validation_loss(
        resumed_trainer.model_parts[0], val_batch, resumed_trainer.loss_fn, resumed_trainer.dist_env.device
    )
    assert torch.allclose(loss_orig, loss_resumed, atol=1e-5), (
        f"Validation loss mismatch: orig={loss_orig.item():.6f} vs resumed={loss_resumed.item():.6f}"
    )

    # ------------------------------------------------------------------
    # Phase 4: Verify HF PEFT can load the adapter without errors
    # ------------------------------------------------------------------
    if _rank0():
        # Build a base HF Llama model (same tiny config, random weights).
        # We need to create it from the same config so architecture matches.
        hf_config = LlamaConfig(**TINY_LLAMA_CONFIG)
        base_model = AutoModelForCausalLM.from_config(hf_config)
        base_model = base_model.to(dtype=trainer.model_parts[0].dtype)

        # Load the PEFT adapter
        peft_model = PeftModel.from_pretrained(base_model, str(model_dir))

        # Verify the PEFT model has LoRA layers
        lora_modules = [
            name
            for name, mod in peft_model.named_modules()
            if "lora" in name.lower() and hasattr(mod, "weight")
        ]
        assert len(lora_modules) > 0, "Expected LoRA modules in PEFT model"

        # Verify forward pass works
        peft_model.eval()
        test_input = torch.randint(0, hf_config.vocab_size, (1, 16))
        with torch.no_grad():
            output = peft_model(test_input)
            assert output.logits is not None, "PEFT model forward pass failed"
            assert output.logits.shape == (1, 16, hf_config.vocab_size), (
                f"Unexpected logits shape: {output.logits.shape}"
            )

        # Verify that the LoRA adapter weights in the PEFT model match what we saved.
        # The saved state dict uses "base_model.model." prefix; PEFT model uses its own naming.
        for saved_key, saved_param in saved_adapter_sd.items():
            saved_param = saved_param.to(dtype=trainer.model_parts[0].dtype)
            matched = False
            for peft_key, peft_param in peft_model.named_parameters():
                if "lora" in peft_key and saved_key.rsplit(".", 1)[0] in peft_key:
                    assert torch.allclose(saved_param, peft_param.data.cpu(), atol=1e-6), (
                        f"PEFT adapter weight mismatch for {saved_key} <-> {peft_key}"
                    )
                    matched = True
                    break
            # Not all keys need to match (some may be non-LoRA), but LoRA keys should
            if "lora" in saved_key.lower():
                assert matched, f"No matching PEFT param found for saved key: {saved_key}"

    _barrier()

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    if _rank0():
        if Path(ckpt_dir).exists():
            shutil.rmtree(ckpt_dir)
    _barrier()
