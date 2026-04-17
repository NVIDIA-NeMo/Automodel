"""Standalone TPS benchmark for the HuggingFace Transformers v5 path.

Loads a model via AutoModelForCausalLM, wraps with FSDP2 per-layer,
runs forward+backward on dummy data, and reports TPS excluding the
optimizer step.

Usage:
    torchrun --nproc-per-node=8 blog_experiments/bench_v5.py --model gptoss_20b
    torchrun --nproc-per-node=8 blog_experiments/bench_v5.py --model qwen3_30b
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


MODEL_CONFIGS = {
    "gptoss_20b": {
        "pretrained_model_name_or_path": "openai/gpt-oss-20b",
        "attn_implementation": "eager",
        "experts_implementation": "grouped_mm",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
        "local_batch_size": 1,  # eager uses more memory, reduce from 4
    },
    "qwen3_30b": {
        "pretrained_model_name_or_path": "Qwen/Qwen3-30B-A3B",
        "attn_implementation": "flash_attention_2",
        "experts_implementation": "grouped_mm",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "local_batch_size": 1,
    },
    "nemotron_nano": {
        "pretrained_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "attn_implementation": "flash_attention_2",
        "experts_implementation": "grouped_mm",
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,  # native v5 + patched lazy_load_kernel for mamba CUDA kernels
        "local_batch_size": 1,
    },
}

SEQ_LEN = 4096
WARMUP_STEPS = 10
TIMED_STEPS = 30


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--lbs", type=int, default=None, help="Override local batch size")
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--steps", type=int, default=TIMED_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    parser.add_argument("--v4", action="store_true", help="Transformers v4 mode: skip v5-only kwargs, force trust_remote_code=True")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    lbs = args.lbs if args.lbs is not None else cfg["local_batch_size"]
    seq_len = args.seq_len

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    variant = "v4" if args.v4 else "v5"
    log(rank, f"=== {variant.upper()} Benchmark: {args.model} ===")
    log(rank, f"World size: {world_size}, LBS: {lbs}, Seq len: {seq_len}")
    log(rank, f"Warmup: {args.warmup}, Timed steps: {args.steps}")

    # --- Patch lazy_load_kernel to try regular pip import first ---
    # HF v5's lazy_load_kernel downloads pre-built binaries from HF hub (kernels-community/*).
    # When no binary matches the CUDA/torch version, or the hub is rate-limited, it fails.
    # This patch tries regular `import mamba_ssm` first, falling back to the original.
    try:
        import importlib
        from types import ModuleType
        from transformers.integrations import hub_kernels
        _original_lazy_load = hub_kernels.lazy_load_kernel

        def _patched_lazy_load_kernel(kernel_name, mapping=hub_kernels._KERNEL_MODULE_MAPPING):
            if kernel_name in mapping and isinstance(mapping[kernel_name], ModuleType):
                return mapping[kernel_name]
            # Try regular pip import first (avoids HF hub API calls entirely)
            try:
                mod = importlib.import_module(kernel_name.replace("-", "_"))
                mapping[kernel_name] = mod
                return mod
            except ImportError:
                pass
            # Fall back to original HF kernel loading
            try:
                return _original_lazy_load(kernel_name, mapping)
            except Exception:
                mapping[kernel_name] = None
                return None

        hub_kernels.lazy_load_kernel = _patched_lazy_load_kernel
        import transformers.integrations
        transformers.integrations.lazy_load_kernel = _patched_lazy_load_kernel
    except Exception:
        pass  # v4 doesn't have this module

    # --- Load model ---
    from transformers import AutoModelForCausalLM, AutoConfig

    t_load_start = time.perf_counter()
    trust_remote_code = True if args.v4 else cfg["trust_remote_code"]
    hf_config = AutoConfig.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        trust_remote_code=trust_remote_code,
    )

    load_kwargs = {
        "torch_dtype": cfg["torch_dtype"],
        "trust_remote_code": trust_remote_code,
        "device_map": {"": rank},  # load directly to local GPU
    }
    if not args.v4:
        # v5-only kwargs
        load_kwargs["attn_implementation"] = cfg["attn_implementation"]
        if "experts_implementation" in cfg:
            load_kwargs["experts_implementation"] = cfg["experts_implementation"]
    model = AutoModelForCausalLM.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        **load_kwargs,
    )
    t_load_end = time.perf_counter()
    log(rank, f"Model loaded in {t_load_end - t_load_start:.1f}s")

    # --- FSDP2 wrap per-layer ---
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    # Find the layers module list
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "backbone") and hasattr(model.backbone, "layers"):
        layers = model.backbone.layers
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = model.transformer.h
    else:
        raise RuntimeError("Cannot find transformer layers for FSDP wrapping")

    # Ensure uniform dtype for FSDP (some models have mixed fp32/bf16 params).
    # Only cast individual non-bf16 params to avoid a slow full-model .to() on CPU.
    for p in model.parameters():
        if p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)

    for layer in layers:
        fully_shard(layer, mp_policy=mp)
    fully_shard(model, mp_policy=mp)

    model.train()
    log(rank, f"FSDP wrapped {len(layers)} layers")

    # --- Optimizer (needed for realistic grad accumulation behavior) ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

    # --- Dummy data ---
    vocab_size = hf_config.vocab_size
    input_ids = torch.randint(0, vocab_size, (lbs, seq_len), device=device)
    labels = input_ids.clone()

    total_tokens_per_step = lbs * seq_len * world_size

    # --- Warmup ---
    log(rank, f"\n--- Warmup ({args.warmup} steps) ---")
    for i in range(args.warmup):
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0 and i % 5 == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            log(rank, f"  warmup {i}: loss={out.loss.item():.4f} peak_mem={mem:.1f} GiB")
    torch.cuda.reset_peak_memory_stats()

    # --- Timed steps ---
    log(rank, f"\n--- Timed ({args.steps} steps) ---")
    step_results = []

    for step in range(args.steps):
        # Sync before timing
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward (includes loss computation)
        out = model(input_ids=input_ids, labels=labels)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        # Backward
        out.loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        # TPS timing ends here (fwd + bwd only)
        fwd_bwd_time = t_bwd - t0
        tps_total = total_tokens_per_step / fwd_bwd_time
        tps_per_gpu = tps_total / world_size

        # Optimizer step (excluded from TPS)
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        t_opt = time.perf_counter()

        result = {
            "step": step,
            "fwd_ms": 1000 * (t_fwd - t0),
            "bwd_ms": 1000 * (t_bwd - t_fwd),
            "opt_ms": 1000 * (t_opt - t_bwd),
            "fwd_bwd_ms": 1000 * fwd_bwd_time,
            "tps_total": tps_total,
            "tps_per_gpu": tps_per_gpu,
            "loss": out.loss.item(),
        }
        step_results.append(result)

        if rank == 0:
            print(
                f"  step {step:3d}: fwd={result['fwd_ms']:.0f}ms bwd={result['bwd_ms']:.0f}ms "
                f"opt={result['opt_ms']:.0f}ms | fwd+bwd={result['fwd_bwd_ms']:.0f}ms "
                f"tps={result['tps_total']:.0f} ({result['tps_per_gpu']:.0f}/gpu) "
                f"loss={result['loss']:.4f}",
                flush=True,
            )

    # --- Summary ---
    if rank == 0:
        tps_values = [r["tps_per_gpu"] for r in step_results]
        fwd_values = [r["fwd_ms"] for r in step_results]
        bwd_values = [r["bwd_ms"] for r in step_results]
        peak_mem = torch.cuda.max_memory_allocated() / 1024**3

        avg_tps = sum(tps_values) / len(tps_values)
        min_tps = min(tps_values)
        max_tps = max(tps_values)
        std_tps = (sum((x - avg_tps) ** 2 for x in tps_values) / len(tps_values)) ** 0.5

        summary = {
            "benchmark": variant,
            "model": args.model,
            "world_size": world_size,
            "local_batch_size": lbs,
            "seq_len": seq_len,
            "tokens_per_step": total_tokens_per_step,
            "warmup_steps": args.warmup,
            "timed_steps": args.steps,
            "tps_per_gpu_avg": round(avg_tps, 1),
            "tps_per_gpu_std": round(std_tps, 1),
            "tps_per_gpu_min": round(min_tps, 1),
            "tps_per_gpu_max": round(max_tps, 1),
            "tps_total_avg": round(avg_tps * world_size, 1),
            "avg_fwd_ms": round(sum(fwd_values) / len(fwd_values), 1),
            "avg_bwd_ms": round(sum(bwd_values) / len(bwd_values), 1),
            "peak_mem_gib": round(peak_mem, 2),
        }

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(f"RESULT: {variant} {args.model} — {summary['tps_per_gpu_avg']:.0f} tps/gpu "
              f"(+/- {summary['tps_per_gpu_std']:.0f}), peak mem {summary['peak_mem_gib']:.1f} GiB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
