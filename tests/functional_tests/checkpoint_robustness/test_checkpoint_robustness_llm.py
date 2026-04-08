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
    [--tokenizer_name <str>]
    [--check_fused_qkv_keys] [--check_phantom_keys] [--check_resume]
    [--max_vram_gb <float>] [--max_cpu_gb <float>]
"""

from __future__ import annotations

import os
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
_DEFAULT_INPUT_IDS = [791, 4996, 14198, 39935, 35308, 927, 279, 16053, 5679]
_DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog"


def _extract_custom_args(argv):
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {
        "--kl_threshold",
        "--hf_kl_threshold",
        "--cross_tp_size",
        "--cross_tp_kl_threshold",
        "--experts_implementation",
        "--tokenizer_name",
        "--max_vram_gb",
        "--max_cpu_gb",
        "--resume_loss_threshold",
    }
    boolean_keys = {"--trust_remote_code", "--check_fused_qkv_keys", "--check_phantom_keys", "--check_resume", "--hf_device_map_auto", "--no_check_hf"}
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

    # Read ci.checkpoint_robustness from the YAML config as defaults.
    # CLI args take precedence over YAML values.
    config_path = None
    for j, arg in enumerate(remaining):
        if arg == "--config" and j + 1 < len(remaining):
            config_path = remaining[j + 1]
            break
    if config_path:
        import yaml

        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f)
        ci_robustness = raw_cfg.get("ci", {}).get("checkpoint_robustness") or {}
        no_check_resume = ci_robustness.pop("no_check_resume", False)
        for k, v in ci_robustness.items():
            if k not in custom:
                if "." in k:
                    # Dotted keys are config overrides (e.g. distributed.tp_size),
                    # route them to the config parser instead of the custom dict.
                    remaining.extend([f"--{k}", str(v)])
                elif isinstance(v, bool) and v:
                    custom[k] = True
                elif not isinstance(v, bool):
                    custom[k] = str(v)
        # Enable check_resume by default unless no_check_resume is set
        if not no_check_resume and "check_resume" not in custom:
            custom["check_resume"] = True
    
    if _rank0():
        print(f"Custom args: {custom}, remaining: {remaining}")

    return custom, remaining


def _get_input_ids(tokenizer_name: str | None) -> list[int]:
    """Return input IDs for the test prompt, using dynamic tokenization if tokenizer_name is set."""
    if tokenizer_name is None:
        return _DEFAULT_INPUT_IDS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    return tokenizer.encode(_DEFAULT_PROMPT, add_special_tokens=False)


def _rss_gb() -> float:
    """Current RSS in GB from /proc/self/statm."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * page_size / 1024**3


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
        logits = out.logits if hasattr(out, "logits") else out
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return logits.float().cpu()


def _fix_meta_rotary_embeddings(model):
    """Re-materialize RotaryEmbedding tensors stuck on meta device.

    The HF remote Baichuan code creates inv_freq/cos_cached/sin_cached as
    plain tensor attributes (not registered buffers), so HF's meta-device
    init never materializes them.
    """
    for _name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and mod.inv_freq.device.type == "meta":
            dim = mod.inv_freq.shape[0] * 2
            mod.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            max_pos = mod.max_seq_len_cached
            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, mod.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            mod.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
            mod.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)
    return model


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
    trust_remote_code = bool(custom_args.get("trust_remote_code", False))
    experts_implementation = custom_args.get("experts_implementation", None)
    tokenizer_name = custom_args.get("tokenizer_name", None)
    max_vram_gb = float(custom_args.get("max_vram_gb", "0"))
    max_cpu_gb = float(custom_args.get("max_cpu_gb", "0"))
    check_fused_qkv_keys = bool(custom_args.get("check_fused_qkv_keys", False))
    check_phantom_keys = bool(custom_args.get("check_phantom_keys", False))
    check_resume = bool(custom_args.get("check_resume", False))
    resume_loss_threshold = float(custom_args.get("resume_loss_threshold", "5e-3"))
    hf_device_map_auto = bool(custom_args.get("hf_device_map_auto", False))
    no_check_hf = bool(custom_args.get("no_check_hf", False))

    input_ids = _get_input_ids(tokenizer_name)

    # Phase 1: Train and checkpoint
    torch.cuda.reset_peak_memory_stats()
    cfg = parse_args_and_load_config()
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()

    # Memory tracking after training
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_cpu_gb = _rss_gb()
    if _rank0():
        print(f"\n[Memory] Peak VRAM: {peak_vram_gb:.2f} GB, Peak CPU RSS: {peak_cpu_gb:.2f} GB")
    if max_vram_gb > 0:
        assert peak_vram_gb <= max_vram_gb, (
            f"Peak VRAM {peak_vram_gb:.2f} GB exceeds threshold {max_vram_gb:.2f} GB"
        )
    if max_cpu_gb > 0:
        assert peak_cpu_gb <= max_cpu_gb, (
            f"Peak CPU RSS {peak_cpu_gb:.2f} GB exceeds threshold {max_cpu_gb:.2f} GB"
        )

    # Phase 2: Capture reference logits before teardown
    device = next(trainer.model_parts[0].parameters()).device
    reference_logits = _get_logits(trainer.model_parts[0], input_ids, device)

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

    # Phantom key check: scan consolidated safetensors for leaked quantization keys
    if check_phantom_keys and _rank0():
        from safetensors import safe_open

        assert consolidated_dir.exists(), f"Phantom key check: {consolidated_dir} does not exist"
        sf_files = sorted(consolidated_dir.glob("*.safetensors"))
        assert len(sf_files) > 0, f"Phantom key check: no .safetensors files in {consolidated_dir}"
        for sf_path in sf_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    assert "_blocks" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"
                    assert "_scales" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"
        print(f"[Phantom keys] Scanned {len(sf_files)} files, no _blocks/_scales keys ✓")

    cfg = parse_args_and_load_config()
    if not is_peft:
        cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
        cfg.checkpoint.enabled = False
    restored_trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    restored_trainer.setup()

    restored_logits = _get_logits(restored_trainer.model_parts[0], input_ids, device)

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
    _barrier()  # ensure all ranks free memory before rank 0 loads HF model

    if _rank0():
        from transformers import AutoModelForCausalLM

        hf_kwargs = dict(torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code)
        if experts_implementation:
            hf_kwargs["experts_implementation"] = experts_implementation
            hf_kwargs["trust_remote_code"] = False
        if hf_device_map_auto:
            hf_kwargs["device_map"] = "auto"

        if is_peft:
            from peft import PeftModel

            if hf_device_map_auto:
                base_model = AutoModelForCausalLM.from_pretrained(original_pretrained_path, **hf_kwargs)
            else:
                base_model = AutoModelForCausalLM.from_pretrained(original_pretrained_path, **hf_kwargs).to(device)
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt_step_dir / "model"))
            hf_logits = _get_logits(peft_model, input_ids, device)

            # PEFT fused QKV key verification
            if check_fused_qkv_keys:
                from safetensors import safe_open

                adapter_path = ckpt_step_dir / "model" / "adapter_model.safetensors"
                assert adapter_path.exists(), f"adapter_model.safetensors not found at {adapter_path}"
                with safe_open(str(adapter_path), framework="pt") as f:
                    adapter_keys = list(f.keys())
                combined_keys = [k for k in adapter_keys if "qkv_proj" in k or "gate_up_proj" in k]
                assert len(combined_keys) == 0, (
                    f"Fused QKV check failed: adapter_model.safetensors contains combined projection keys: "
                    f"{combined_keys}"
                )
                print(f"[Fused QKV] No combined projection keys in adapter ({len(adapter_keys)} keys checked) ✓")

            del peft_model, base_model
        else:
            if hf_device_map_auto:
                hf_model = AutoModelForCausalLM.from_pretrained(str(consolidated_dir), **hf_kwargs)
            else:
                hf_model = _fix_meta_rotary_embeddings(
                    AutoModelForCausalLM.from_pretrained(str(consolidated_dir), **hf_kwargs)
                ).to(device)
            hf_logits = _get_logits(hf_model, input_ids, device)
            del hf_model

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

        cross_tp_logits = _get_logits(cross_tp_trainer.model_parts[0], input_ids, device)

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

    # Phase 6 (optional): Training resumption — verify loss continuity
    # Phase 1 trained for max_steps (e.g. 5) and checkpointed. We now train a fresh baseline
    # for max_steps+3 (no checkpoint save), then resume from the checkpoint and train to
    # max_steps+3. For SFT, losses should match to ~4 decimal places.
    if check_resume:
        import json
        import shutil
        import tempfile

        # Baseline: fresh continuous run for max_steps+3, saving losses to a temp dir
        baseline_dir = tempfile.mkdtemp(prefix="resume_baseline_")
        cfg = parse_args_and_load_config()
        original_max_steps = cfg.step_scheduler.max_steps
        resume_max_steps = original_max_steps + 3
        cfg.step_scheduler.max_steps = resume_max_steps
        cfg.checkpoint.checkpoint_dir = baseline_dir
        cfg.checkpoint.enabled = False
        baseline_trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
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
        resume_trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
        resume_trainer.setup()
        resume_trainer.run_train_validation_loop()

        # Compare losses at the overlapping steps
        resume_jsonl = checkpoint_dir / "training.jsonl"
        if _rank0():
            assert baseline_losses, "Phase 6: baseline_losses is empty — no steps to compare"
            assert resume_jsonl.exists(), f"Phase 6: {resume_jsonl} not found"

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
                        f"[Phase 6] Step {step}: baseline_loss={bl:.6f}, resume_loss={rl:.6f}, diff={diff:.6e}"
                    )
                    if not is_peft:
                        assert diff < resume_loss_threshold, (
                            f"SFT loss mismatch after resume at step {step}: "
                            f"baseline={bl:.6f}, resume={rl:.6f}, diff={diff:.6e}"
                        )

            assert matched_steps > 0, (
                f"Phase 6: no overlapping steps found between baseline ({sorted(baseline_losses.keys())}) "
                f"and resume ({sorted(resume_losses.keys())})"
            )
            print(f"[Phase 6] Training resumption verified ({matched_steps} steps compared) ✓")

        del resume_trainer
        torch.cuda.empty_cache()
        _barrier()


if __name__ == "__main__":
    test_checkpoint_robustness()
