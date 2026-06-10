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
    "nemotron_ultra": {
        # 550B (55B active) hybrid Mamba2 + attention + LatentMoE. Same native nemotron_h arch as
        # nemotron_nano (model_type=nemotron_h, no auto_map) -> trust_remote_code=False. HF v5 runs
        # the 108-layer backbone + lm_head ONLY; it silently drops the MTP head
        # (_keys_to_ignore_on_load_unexpected=[r"mtp.*"]) -> compare vs the AutoModel MTP-OFF number.
        # Cannot use device_map={"":rank} (~1.1 TB bf16 weights >> 80 GB), so meta_init_load=True selects
        # the meta-init -> per-layer FSDP2 -> RANDOM-weight path below (throughput bench on dummy data;
        # the real checkpoint is per-expert + "backbone."-prefixed and needs a custom HF->grouped sharded
        # converter that DCP raw load cannot do -- weight values do not affect timing, only routing balance,
        # so random init == the balanced-gate regime; compare to AutoModel fake_balanced_gate=True).
        "pretrained_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "attn_implementation": "flash_attention_2",
        "experts_implementation": "grouped_mm",  # REQUIRED: per-expert for-loop over 512 experts is unusably slow
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": False,
        "local_batch_size": 1,
        "activation_checkpointing": True,  # mandatory for memory at 16 nodes
        "meta_init_load": True,
        "optimizer": "sgd",  # opt.step is excluded from TPS; stateless SGD frees ~17 GiB Adam states so the no-EP 20 GiB expert all-gather fits 16 nodes at seq=4096
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
    parser.add_argument("--opt", choices=["adam", "sgd"], default=None, help="Override optimizer (debug: Adam memory pressure)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    lbs = args.lbs if args.lbs is not None else cfg["local_batch_size"]
    seq_len = args.seq_len

    # Device placement MUST use LOCAL_RANK (0..gpus_per_node-1), NOT the global rank: on multi-node
    # every rank>gpus_per_node would otherwise crash with "CUDA error: invalid device ordinal".
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")

    variant = "v4" if args.v4 else "v5"
    import transformers as _tfm
    log(rank, f"=== {variant.upper()} Benchmark: {args.model} ===")
    log(rank, f"USING transformers {_tfm.__version__} @ {_tfm.__file__}")
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
    from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)

    def _find_layers(m):
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        elif hasattr(m, "backbone") and hasattr(m.backbone, "layers"):
            return m.backbone.layers
        elif hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise RuntimeError("Cannot find transformer layers for FSDP wrapping")

    def _enable_activation_checkpointing(layer_list):
        # NemotronHForCausalLM gates model.gradient_checkpointing_enable() behind
        # supports_gradient_checkpointing=False, but each block IS a GradientCheckpointingLayer
        # whose __call__ checkpoints when `self.gradient_checkpointing and self.training`. Set the
        # flag + func directly on each block: FQN-safe (no module wrapper -> DCP load keys stay clean)
        # and bypasses the model-level gate.
        import functools

        from torch.utils.checkpoint import checkpoint as _ckpt

        gc_func = functools.partial(_ckpt, use_reentrant=False)
        n = 0
        for layer in layer_list:
            if hasattr(layer, "gradient_checkpointing"):
                layer.gradient_checkpointing = True
                layer._gradient_checkpointing_func = gc_func
                n += 1
        return n

    t_load_start = time.perf_counter()
    trust_remote_code = True if args.v4 else cfg["trust_remote_code"]
    hf_config = AutoConfig.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        trust_remote_code=trust_remote_code,
    )
    use_ac = bool(cfg.get("activation_checkpointing", False))
    if use_ac:
        hf_config.use_cache = False  # required with gradient checkpointing

    if cfg.get("meta_init_load", False):
        # --- Large-model path (e.g. Ultra-550B): the full model (~1.1 TB bf16) cannot fit one GPU.
        # Init on meta -> per-layer FSDP2 shard -> to_empty -> initialize RANDOM weights (no checkpoint
        # load). This is a THROUGHPUT bench on dummy data: weight VALUES do not change fwd/bwd FLOPs or
        # FSDP comm, only MoE routing balance. Random init gives approximately uniform routing -> the
        # BALANCED-gate regime, directly comparable to the AutoModel fake_balanced_gate=True numbers.
        # (Loading the real checkpoint is intentionally skipped: it is per-expert + "backbone."-prefixed
        # and would need HF's load-time key-rename + per-expert -> stacked-3D expert conversion, which
        # DCP raw load cannot do; that custom sharded converter is a follow-up if a real-weight
        # realistic-routing HF number is needed.)
        from torch.distributed.tensor import DTensor

        if not args.v4 and "experts_implementation" in cfg:
            # @use_experts_implementation reads config._experts_implementation at module-build time.
            hf_config._experts_implementation = cfg["experts_implementation"]
        from_config_kwargs = {"dtype": cfg["torch_dtype"]}
        if not args.v4:
            from_config_kwargs["attn_implementation"] = cfg["attn_implementation"]
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(hf_config, **from_config_kwargs)

        layers = _find_layers(model)
        if use_ac:
            nac = _enable_activation_checkpointing(layers)
            log(rank, f"Activation checkpointing enabled on {nac}/{len(layers)} blocks")
        for layer in layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)

        # Allocate the (sharded) real tensors on-device, then fill RANDOM weights (per-rank local shard).
        model.to_empty(device=device)
        torch.manual_seed(1234 + rank)
        with torch.no_grad():
            for p in model.parameters():
                t = p.to_local() if isinstance(p, DTensor) else p
                if t.numel() > 0 and t.is_floating_point():
                    t.normal_(mean=0.0, std=0.02)
            for b in model.buffers():
                t = b.to_local() if isinstance(b, DTensor) else b
                if t.numel() > 0 and t.is_floating_point():
                    t.zero_()
        t_load_end = time.perf_counter()
        log(rank, f"Model meta-init + FSDP shard ({len(layers)} layers) + RANDOM weights "
                  f"(perf bench; checkpoint load skipped) in {t_load_end - t_load_start:.1f}s")
    else:
        # --- Small-model path: load full model to the local GPU, then per-layer FSDP shard. ---
        load_kwargs = {
            "torch_dtype": cfg["torch_dtype"],
            "trust_remote_code": trust_remote_code,
            "device_map": {"": local_rank},  # load directly to local GPU (LOCAL_RANK, not global)
        }
        if not args.v4:
            load_kwargs["attn_implementation"] = cfg["attn_implementation"]
            if "experts_implementation" in cfg:
                load_kwargs["experts_implementation"] = cfg["experts_implementation"]
        model = AutoModelForCausalLM.from_pretrained(cfg["pretrained_model_name_or_path"], **load_kwargs)
        t_load_end = time.perf_counter()
        log(rank, f"Model loaded in {t_load_end - t_load_start:.1f}s")

        layers = _find_layers(model)
        # Ensure uniform dtype for FSDP (some models have mixed fp32/bf16 params).
        for p in model.parameters():
            if p.dtype != torch.bfloat16:
                p.data = p.data.to(torch.bfloat16)
        if use_ac:
            _enable_activation_checkpointing(layers)
        for layer in layers:
            fully_shard(layer, mp_policy=mp)
        fully_shard(model, mp_policy=mp)

    model.train()
    log(rank, f"FSDP wrapped {len(layers)} layers")

    # --- Optimizer ---
    # optimizer.step() is EXCLUDED from the TPS timing, so the optimizer's state memory is pure
    # overhead that doesn't affect the measured fwd+bwd number. For the no-EP large path, Adam's
    # m+v states (~17 GiB/rank) on top of the full 20 GiB per-layer expert all-gather push the
    # forward over 80 GiB; a stateless SGD frees that headroom so seq=4096 fits, with identical
    # fwd+bwd TPS. Small models keep Adam.
    if (args.opt or cfg.get("optimizer", "adam")) == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.0, foreach=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, foreach=False)

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
        print(f"RESULT: {variant} {args.model} — avg over {args.steps} post-warmup steps: "
              f"{summary['tps_per_gpu_avg']:.0f} tps/gpu (+/- {summary['tps_per_gpu_std']:.0f}) | "
              f"{summary['tps_total_avg']:.0f} tps total ({world_size} GPUs), peak mem {summary['peak_mem_gib']:.1f} GiB")
        _s = torch.cuda.memory_stats()
        print(f"ALLOC: num_alloc_retries={_s.get('num_alloc_retries', 0)} num_ooms={_s.get('num_ooms', 0)} "
              f"reserved_gib={_s.get('reserved_bytes.all.peak', 0) / 1024**3:.1f}", flush=True)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
