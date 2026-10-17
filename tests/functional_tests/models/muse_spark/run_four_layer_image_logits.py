#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate four-layer Muse Spark Open image logits against the SGLang BF16 recipe."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.muse_spark.processing import build_muse_spark_processor
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config

_REFERENCE_VOCAB_IDS = [
    63398,
    179566,
    127705,
    12291,
    152049,
    194682,
    163093,
    27547,
    9142,
    12211,
    186805,
    49734,
    13524,
    702,
    59859,
    21480,
    9979,
    127123,
    160103,
    21715,
]
_REFERENCE_LOGPROBS = [
    -4.7811737060546875,
    -5.267911911010742,
    -5.77264404296875,
    -5.819413185119629,
    -5.866329193115234,
    -5.866329193115234,
    -5.866329193115234,
    -5.960597991943359,
    -5.960597991943359,
    -6.007952690124512,
    -6.103096008300781,
    -6.103096008300781,
    -6.103096008300781,
    -6.103096008300781,
    -6.150886535644531,
    -6.3434906005859375,
    -6.3434906005859375,
    -6.3434906005859375,
    -6.392000198364258,
    -6.4406538009643555,
]


def parse_args() -> argparse.Namespace:
    """Parse checkpoint and artifact paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--attn", choices=("te", "sdpa"), default="te")
    parser.add_argument("--dispatcher", choices=("deepep", "hybridep", "torch"), default="deepep")
    return parser.parse_args()


def main() -> None:
    """Run the released image tower plus four language layers with EP8."""
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise ValueError(f"Muse Spark Open four-layer image parity requires eight ranks, got {world_size}")
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

    processor = build_muse_spark_processor(args.model, fix_mistral_regex=False)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "How does the patient history play a role?"},
                {"type": "image", "image": "<image>"},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        current_date="2026-09-02",
    )
    batch = processor(images=Image.open(args.image), text=prompt, return_tensors="pt")
    input_ids = batch["input_ids"].cuda(local_rank)
    pixel_values = batch["pixel_values"].cuda(local_rank)
    image_grid_thw = batch["image_grid_thw"].cuda(local_rank)
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    probe_ids = torch.tensor(_REFERENCE_VOCAB_IDS, device=input_ids.device, dtype=torch.long)
    reference_logprobs = torch.tensor(_REFERENCE_LOGPROBS, device=input_ids.device)

    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            position_ids=positions,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
        ).logits
        last_logprobs = logits[:, -1].float().log_softmax(dim=-1)
        probe_logprobs = last_logprobs.index_select(-1, probe_ids).squeeze(0)
        top_id = int(last_logprobs.argmax(dim=-1).item())
    rank_reference = probe_logprobs.clone()
    dist.broadcast(rank_reference, src=0)
    rank_max_diff = torch.tensor(
        float((probe_logprobs - rank_reference).abs().max().item()),
        device=input_ids.device,
    )
    dist.all_reduce(rank_max_diff, op=dist.ReduceOp.MAX)
    absolute_error = (probe_logprobs - reference_logprobs).abs()
    cosine = torch.nn.functional.cosine_similarity(probe_logprobs, reference_logprobs, dim=0)
    passed = (
        input_ids.shape[1] == 280
        and int((input_ids == processor.image_token_id).sum().item()) == 132
        and top_id == _REFERENCE_VOCAB_IDS[0]
        and float(absolute_error.max().item()) < 0.4
        and float(cosine.item()) > 0.999
        and float(rank_max_diff.item()) < 0.5
    )

    if rank == 0:
        result = {
            "model": args.model,
            "num_hidden_layers": 4,
            "schedule_num_hidden_layers": 62,
            "prompt_tokens": input_ids.shape[1],
            "image_tokens": int((input_ids == processor.image_token_id).sum().item()),
            "image_grid_thw": image_grid_thw.cpu().tolist(),
            "reference_vocab_ids": _REFERENCE_VOCAB_IDS,
            "reference_logprobs": _REFERENCE_LOGPROBS,
            "automodel_logprobs": probe_logprobs.cpu().tolist(),
            "top_token_id": top_id,
            "max_abs_error": float(absolute_error.max().item()),
            "mean_abs_error": float(absolute_error.mean().item()),
            "cosine_similarity": float(cosine.item()),
            "rank_max_abs_diff": float(rank_max_diff.item()),
            "passed": passed,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        print(f"MUSE_SPARK_FOUR_LAYER_IMAGE_PARITY: {'PASS' if passed else 'FAIL'}", flush=True)
    dist.barrier()
    dist.destroy_process_group()
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
