"""Standalone TPS benchmark for the NeMo Automodel path.

Uses NeMoAutoModelForCausalLM.from_pretrained with BackendConfig + EP=8.
Custom models return raw logits, so loss is computed manually.
TPS = tokens / (fwd + loss + bwd), optimizer step excluded.

Usage:
    torchrun --nproc-per-node=8 blog_experiments/bench_automodel.py --model gptoss_20b
    torchrun --nproc-per-node=8 blog_experiments/bench_automodel.py --model qwen3_30b

    # Nemotron-3-Ultra-550B (full fine-tune): full FT needs ~4.4 TB of resident state
    # (8 B/param), so it requires multi-node. Verified config: 16 H100 nodes (128 GPUs),
    # EP=64, lbs=2, balanced gate -> ~876 tps/gpu @ 58.5 GiB. Activation checkpointing +
    # fused linear CE + MTP are enabled by the config (all defaults below).
    #   srun --nodes=16 --ntasks-per-node=1 bash -c 'torchrun --nnodes=16 --nproc-per-node=8 \
    #     --rdzv-backend=c10d --rdzv-endpoint="$MASTER_ADDR":29500 --node-rank=$SLURM_NODEID \
    #     blog_experiments/bench_automodel.py --model nemotron_ultra'
"""

import argparse
import json
import os
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Per-model backend configs matching the YAML configs
MODEL_CONFIGS = {
    "gptoss_20b": {
        "pretrained_model_name_or_path": "openai/gpt-oss-20b",
        "backend": {
            "attn": "te",
            "linear": "te",
            "rms_norm": "te",
            "experts": "gmm",
            "dispatcher": "deepep",
            "fake_balanced_gate": False,
            "enable_hf_state_dict_adapter": True,
        },
        "extra_kwargs": {"moe_overrides": {"aux_loss_coeff": 0.0}},
        "trust_remote_code": False,
        "local_batch_size": 2,
        "returns_logits_only": True,
    },
    "qwen3_30b": {
        "pretrained_model_name_or_path": "Qwen/Qwen3-30B-A3B",
        "backend": {
            "attn": "te",
            "linear": "te",
            "rms_norm": "torch_fp32",
            "experts": "torch_mm",
            "gate_precision": "bf16",
            "dispatcher": "deepep",
            "fake_balanced_gate": True,
            "enable_hf_state_dict_adapter": True,
            "enable_fsdp_optimizations": True,
        },
        "extra_kwargs": {},
        "trust_remote_code": True,
        "local_batch_size": 1,
        "returns_logits_only": True,
    },
    "nemotron_nano": {
        "pretrained_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "backend": {
            "attn": "te",
            "linear": "te",
            "rms_norm": "te",
            "experts": "torch_mm",
            "dispatcher": "deepep",
            "fake_balanced_gate": True,
            "enable_hf_state_dict_adapter": True,
            "enable_fsdp_optimizations": True,
        },
        "extra_kwargs": {},
        "trust_remote_code": True,
        "local_batch_size": 1,
        "returns_logits_only": False,
    },
    "nemotron_ultra": {
        "pretrained_model_name_or_path": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-BF16",
        "backend": {
            "attn": "te",
            "linear": "te",
            "rms_norm": "te",
            "experts": "torch_mm",
            "dispatcher": "deepep",
            "fake_balanced_gate": True,
            "enable_fsdp_optimizations": False,
        },
        "extra_kwargs": {
            "num_nextn_predict_layers": 2,
            "mtp_use_repeated_layer": True,
            "mtp_loss_scaling_factor": 0.1,
            "output_hidden_states": True,
        },
        "trust_remote_code": True,
        "local_batch_size": 2,
        "returns_logits_only": False,
        "ep_size": 64,
        "activation_checkpointing": True,
        "moe": {
            "reshard_after_forward": True,
            "wrap_outer_model": False,
        },
        "use_fused_ce": True,  # fused LM-head projection + CE; no [lbs, seq, vocab] logits
        "use_mtp": True,
    },
}

SEQ_LEN = 4096
WARMUP_STEPS = 10
TIMED_STEPS = 30


def log(rank, msg):
    if rank == 0:
        print(msg, flush=True)


def compute_causal_lm_loss(logits, labels):
    """Standard causal LM loss: shift logits/labels, cross-entropy in bfloat16."""
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--seq-len", type=int, default=SEQ_LEN, help="Sequence length")
    parser.add_argument("--steps", type=int, default=TIMED_STEPS)
    parser.add_argument("--warmup", type=int, default=WARMUP_STEPS)
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    seq_len = args.seq_len
    lbs = cfg["local_batch_size"]

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    ep_size = cfg.get("ep_size", 8)

    log(rank, f"=== Automodel Benchmark: {args.model} ===")
    log(rank, f"World size: {world_size}, LBS: {lbs}, Seq len: {seq_len}, EP: {ep_size}")
    log(rank, f"Warmup: {args.warmup}, Timed steps: {args.steps}")
    log(rank, f"Backend: {cfg['backend']}")
    log(rank, f"MoE: {cfg.get('moe')}")

    # --- Set up distributed mesh with EP ---
    from nemo_automodel import NeMoAutoModelForCausalLM
    from nemo_automodel.components.models.common.utils import BackendConfig
    from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

    dist_setup = create_distributed_setup_from_config(
        {
            "strategy": "fsdp2",
            "dp_size": None,
            "dp_replicate_size": None,
            "tp_size": 1,
            "pp_size": 1,
            "cp_size": 1,
            "ep_size": ep_size,
            "activation_checkpointing": cfg.get("activation_checkpointing", False),
            **({"moe": cfg["moe"]} if "moe" in cfg else {}),
        },
        world_size=world_size,
    )

    # --- Load model via NeMoAutoModelForCausalLM ---
    t_load_start = time.perf_counter()

    backend_config = BackendConfig(**cfg["backend"])

    model = NeMoAutoModelForCausalLM.from_pretrained(
        cfg["pretrained_model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=cfg["trust_remote_code"],
        distributed_setup=dist_setup,
        backend=backend_config,
        **cfg["extra_kwargs"],
    )

    t_load_end = time.perf_counter()
    log(rank, f"Model loaded + FSDP wrapped in {t_load_end - t_load_start:.1f}s")

    model.train()

    # --- Optimizer ---
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8, foreach=False)

    # --- Dummy data ---
    # Get vocab size from model config
    vocab_size = model.config.vocab_size
    device = torch.device(f"cuda:{local_rank}")
    input_ids = torch.randint(0, vocab_size, (lbs, seq_len), device=device)
    labels = input_ids.clone()

    total_tokens_per_step = lbs * seq_len * world_size
    returns_logits_only = cfg.get("returns_logits_only", True)
    use_fused_ce = cfg.get("use_fused_ce", False)
    use_mtp = cfg.get("use_mtp", False)

    if use_fused_ce:
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
        from nemo_automodel.components.loss.utils import calculate_loss
        from nemo_automodel.components.training.model_output_utils import get_final_hidden_states

        if use_mtp:
            from nemo_automodel.components.loss.mtp import calculate_mtp_loss

        fused_loss_fn = FusedLinearCrossEntropy()
        num_label_tokens = int((labels != -100).sum().item())

    def forward_and_loss():
        """Returns loss. Handles raw-logits, CausalLMOutput, and fused-CE (+MTP) models."""
        if use_fused_ce:
            out = model(input_ids=input_ids, logits_to_keep=1)
            loss = calculate_loss(
                fused_loss_fn,
                labels=labels,
                model=model,
                hidden_states=get_final_hidden_states(out),
                num_label_tokens=num_label_tokens,
            )
            if use_mtp:
                # MTP only fires in model.train() (set above); mtp_per_depth_h is None otherwise.
                mtp_per_depth_h = getattr(out, "mtp_per_depth_h", None)
                if mtp_per_depth_h is not None:
                    loss = loss + calculate_mtp_loss(
                        fused_loss_fn,
                        mtp_per_depth_h=mtp_per_depth_h,
                        labels=labels,
                        model=model,
                        scaling_factor=out.mtp_loss_scaling_factor,
                        num_label_tokens=num_label_tokens,
                    )
            return loss
        elif returns_logits_only:
            out = model(input_ids=input_ids)
            # Merged main returns a CausalLMOutput; older builds returned a raw logits tensor.
            logits = out.logits if hasattr(out, "logits") else out
            return compute_causal_lm_loss(logits, labels)
        else:
            out = model(input_ids=input_ids, labels=labels)
            return out.loss

    # --- Warmup ---
    log(rank, f"\n--- Warmup ({args.warmup} steps) ---")
    for i in range(args.warmup):
        loss = forward_and_loss()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if rank == 0 and i % 5 == 0:
            mem = torch.cuda.max_memory_allocated() / 1024**3
            log(rank, f"  warmup {i}: loss={loss.item():.4f} peak_mem={mem:.1f} GiB")
    torch.cuda.reset_peak_memory_stats()

    # --- Timed steps ---
    log(rank, f"\n--- Timed ({args.steps} steps) ---")
    step_results = []

    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Forward + Loss
        loss = forward_and_loss()
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        # Backward
        loss.backward()
        torch.cuda.synchronize()
        t_bwd = time.perf_counter()

        # TPS timing: fwd + loss + bwd
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
            "loss": loss.item(),
        }
        step_results.append(result)

        if rank == 0:
            print(
                f"  step {step:3d}: fwd+loss={result['fwd_ms']:.0f}ms "
                f"bwd={result['bwd_ms']:.0f}ms opt={result['opt_ms']:.0f}ms | "
                f"fwd+bwd={result['fwd_bwd_ms']:.0f}ms "
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
            "benchmark": "automodel",
            "model": args.model,
            "world_size": world_size,
            "local_batch_size": lbs,
            "seq_len": seq_len,
            "ep_size": ep_size,
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
            "backend": cfg["backend"],
        }

        print("\n=== SUMMARY ===")
        print(json.dumps(summary, indent=2))
        print(
            f"RESULT: automodel {args.model} EP={ep_size} — avg over {args.steps} post-warmup steps: "
            f"{summary['tps_per_gpu_avg']:.0f} tps/gpu (+/- {summary['tps_per_gpu_std']:.0f}) | "
            f"{summary['tps_total_avg']:.0f} tps total ({world_size} GPUs), peak mem {summary['peak_mem_gib']:.1f} GiB"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
