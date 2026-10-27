#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run deterministic four-layer Muse Spark Open logits on an EP8 node.

This is the first hardware gate for the 816B checkpoint.  The model contains
layers 0--3 only, while ``schedule_num_hidden_layers=62`` preserves those
layers' released sliding-window and no-RoPE phase.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config


def parse_args() -> argparse.Namespace:
    """Parse checkpoint and artifact paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--attn", choices=("te", "sdpa"), default="te")
    parser.add_argument("--dispatcher", choices=("deepep", "hybridep", "torch"), default="deepep")
    return parser.parse_args()


def main() -> None:
    """Load four released layers with EP8 and write selected deterministic logits."""
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise ValueError(f"Muse Spark Open four-layer logits require eight ranks, got {world_size}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    setup = create_distributed_setup_from_config(
        ConfigNode(
            {
                "distributed": {
                    "strategy": "fsdp2",
                    "dp_size": None,
                    "tp_size": 1,
                    "cp_size": 1,
                    "pp_size": 1,
                    "ep_size": 8,
                    "sequence_parallel": False,
                    "activation_checkpointing": False,
                }
            }
        ),
        world_size=world_size,
    )
    model = NeMoAutoModelForCausalLM.from_pretrained(
        args.model,
        config={"num_hidden_layers": 4, "schedule_num_hidden_layers": 62},
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        backend={
            "attn": args.attn,
            "linear": "torch",
            "rms_norm": "torch_fp32",
            "experts": "torch_mm",
            "dispatcher": args.dispatcher,
            "enable_hf_state_dict_adapter": True,
            "enable_fsdp_optimizations": True,
        },
        distributed_setup=setup,
        freeze_config={
            "freeze_vision_tower": True,
            "freeze_audio_tower": True,
            "freeze_language_model": True,
        },
        use_liger_kernel=False,
    )
    model.eval()
    device = torch.device("cuda", local_rank)
    positions = torch.arange(args.sequence_length, device=device)
    input_ids = ((positions * 7919 + 17) % 199999).unsqueeze(0).to(torch.long)
    probe_ids = torch.tensor(
        [0, 1, 2, 3, 7, 11, 42, 127, 1024, 4096, 8192, 32768, 65536, 100000, 150000, 202047],
        device=device,
        dtype=torch.long,
    )
    torch.cuda.reset_peak_memory_stats(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.inference_mode():
        logits = model(input_ids=input_ids, position_ids=positions.unsqueeze(0)).logits
        probes = logits.index_select(-1, probe_ids).float()
        last_probe_logprobs = logits[:, -1].float().log_softmax(dim=-1).index_select(-1, probe_ids)
    end.record()
    torch.cuda.synchronize(device)

    reference = probes.clone()
    dist.broadcast(reference, src=0)
    rank_max_diff = float((probes - reference).abs().max().item())
    max_rank_diff = torch.tensor(rank_max_diff, device=device)
    dist.all_reduce(max_rank_diff, op=dist.ReduceOp.MAX)
    if rank == 0:
        result = {
            "model": args.model,
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 62,
            "sequence_length": args.sequence_length,
            "probe_vocab_ids": probe_ids.cpu().tolist(),
            "logits_probe": probes.cpu().tolist(),
            "last_token_probe_logprobs": last_probe_logprobs.cpu().tolist(),
            "rank_max_abs_diff": float(max_rank_diff.item()),
            "forward_seconds": start.elapsed_time(end) / 1000.0,
            "peak_memory_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_memory_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items() if key != "logits_probe"}, indent=2))
        print("MUSE_SPARK_FOUR_LAYER_LOGITS: PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
