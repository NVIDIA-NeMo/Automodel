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
   ``up_proj``) -- no combined names (``qkv_proj``, ``gate_up_proj``).
4. HuggingFace PEFT (``PeftModel.from_pretrained``) can load the adapter
   without errors and produces a working model.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_file
from transformers import LlamaConfig

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    calculate_loss,
)

import datasets

datasets.disable_caching()

# ---------------------------------------------------------------------------
# Tiny Llama config (must match the YAML)
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

CKPT_DIR = "checkpoints_peft_fused_qkv_test/"
CFG_PATH = Path(__file__).parent / "peft_fused_qkv_config.yaml"


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _collect_lora_params(model_parts) -> dict[str, torch.Tensor]:
    """Collect LoRA trainable parameters (on CPU) from model parts."""
    sd = ModelState(model_parts, is_peft=True).state_dict()
    return {k: v.cpu().clone() for k, v in sd.items()}


def _get_validation_loss(model, val_batch, loss_fn, device):
    val_batch = {k: v.to(device, non_blocking=True) for k, v in val_batch.items()}
    model.eval()
    labels = val_batch.pop("labels")
    with torch.no_grad():
        out = model(**val_batch)
        return calculate_loss(loss_fn, logits=out.logits, labels=labels)


def test_peft_fused_qkv_checkpoint():
    """End-to-end: train 2 steps -> save ckpt -> resume ckpt -> verify HF PEFT load."""

    # ==================================================================
    # Phase 1: Train for 2 steps and save a checkpoint
    # ==================================================================
    cfg = parse_args_and_load_config(CFG_PATH)
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # Collect LoRA params after training
    lora_params_after_train = _collect_lora_params(trainer.model_parts)
    assert len(lora_params_after_train) > 0, "Expected LoRA parameters to be present"

    # Verify checkpoint was saved
    ckpt_step_dir = Path(CKPT_DIR) / "epoch_0_step_1"
    assert ckpt_step_dir.exists(), f"Checkpoint directory {ckpt_step_dir} does not exist"

    model_dir = ckpt_step_dir / "model"
    assert (model_dir / "adapter_model.safetensors").exists(), "adapter_model.safetensors not found"
    assert (model_dir / "adapter_config.json").exists(), "adapter_config.json not found"

    # ==================================================================
    # Phase 2: Verify saved adapter has NO combined-projection keys
    # ==================================================================
    saved_adapter_sd = load_file(str(model_dir / "adapter_model.safetensors"))

    combined_keys = [k for k in saved_adapter_sd if "qkv_proj" in k or "gate_up_proj" in k]
    assert combined_keys == [], (
        f"Saved adapter should NOT contain combined-projection keys, found: {combined_keys}"
    )

    # Verify split projection names ARE present
    has_q_proj = any("q_proj" in k for k in saved_adapter_sd)
    has_k_proj = any("k_proj" in k for k in saved_adapter_sd)
    has_v_proj = any("v_proj" in k for k in saved_adapter_sd)
    assert has_q_proj and has_k_proj and has_v_proj, (
        f"Expected split q/k/v projection keys in saved adapter. Keys: {list(saved_adapter_sd.keys())}"
    )

    # Verify adapter_config.json target_modules are split (no combined names)
    with open(model_dir / "adapter_config.json") as f:
        adapter_config = json.load(f)
    for mod in adapter_config.get("target_modules", []):
        assert "qkv_proj" not in mod, f"Combined qkv_proj found in target_modules: {mod}"
        assert "gate_up_proj" not in mod, f"Combined gate_up_proj found in target_modules: {mod}"

    # ==================================================================
    # Phase 3: Resume from checkpoint, verify LoRA weights match
    # ==================================================================
    resume_cfg = parse_args_and_load_config(CFG_PATH)
    resume_cfg.checkpoint.restore_from = str(ckpt_step_dir)

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

    # ==================================================================
    # Phase 4: Verify HF PEFT can load the adapter without errors
    # ==================================================================
    _peft_verified = False
    if _rank0():
        try:
            from peft import PeftModel
            from transformers import AutoModelForCausalLM
        except ImportError:
            print("[WARN] peft or transformers.AutoModelForCausalLM not importable, skipping HF PEFT verification")
            PeftModel = None

        if PeftModel is not None:
            # Build a base HF Llama model (same tiny config, random weights).
            # Use the same device as the trainer so Liger/Triton kernels (e.g. RMS norm)
            # get GPU tensors; they cannot run on CPU.
            device = next(trainer.model_parts[0].parameters()).device
            hf_config = LlamaConfig(**TINY_LLAMA_CONFIG)
            base_model = AutoModelForCausalLM.from_config(hf_config)
            base_model = base_model.to(device=device, dtype=trainer.model_parts[0].dtype)

            # Load the PEFT adapter
            peft_model = PeftModel.from_pretrained(base_model, str(model_dir))

            # Verify the PEFT model has LoRA layers
            lora_modules = [
                name
                for name, mod in peft_model.named_modules()
                if "lora" in name.lower() and hasattr(mod, "weight")
            ]
            assert len(lora_modules) > 0, "Expected LoRA modules in PEFT model"

            # Verify forward pass works (input must be on same device as model for Triton)
            peft_model.eval()
            test_input = torch.randint(0, hf_config.vocab_size, (1, 16), device=device)
            with torch.no_grad():
                output = peft_model(test_input)
                assert output.logits is not None, "PEFT model forward pass failed"
                assert output.logits.shape == (1, 16, hf_config.vocab_size), (
                    f"Unexpected logits shape: {output.logits.shape}"
                )

            # Verify that the saved LoRA adapter weights match what HF PEFT loaded
            for saved_key, saved_param in saved_adapter_sd.items():
                if "lora" not in saved_key.lower():
                    continue
                matched = False
                for peft_key, peft_param in peft_model.named_parameters():
                    if "lora" in peft_key and saved_key.rsplit(".", 1)[0] in peft_key:
                        # Compare in float32 to avoid dtype mismatch (safetensors
                        # may store bf16 while HF PEFT loads as fp32).
                        assert torch.allclose(
                            saved_param.float(), peft_param.data.cpu().float(), atol=1e-6
                        ), f"PEFT adapter weight mismatch for {saved_key} <-> {peft_key}"
                        matched = True
                        break
                assert matched, f"No matching PEFT param found for saved key: {saved_key}"
            _peft_verified = True

    _barrier()

    # ==================================================================
    # Cleanup
    # ==================================================================
    if _rank0():
        if Path(CKPT_DIR).exists():
            shutil.rmtree(CKPT_DIR)
    _barrier()
