# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Dump full-sequence logits from the released DSV4 Vision implementation."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from safetensors.torch import load_model


def _torch_hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Apply the released normalized Walsh-Hadamard transform in PyTorch."""
    width = x.shape[-1]
    stages = int(math.log2(width))
    if width != 1 << stages:
        raise ValueError(f"Hadamard width must be a power of two, got {width}")
    output = x.unsqueeze(-1)
    for _ in range(stages):
        even = output[..., ::2, :]
        odd = output[..., 1::2, :]
        output = torch.cat((even + odd, even - odd), dim=-1)
    return output.squeeze(-2) * scale


def _initialize_distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if world_size > 1:
        dist.init_process_group("nccl")
    return rank, world_size, local_rank


def main() -> None:
    """Load the released TP checkpoint and dump each full logits tensor."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--converted-checkpoint", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    inference_dir = args.reference_dir / "inference"
    sys.path.insert(0, str(inference_dir))
    # The imports intentionally happen after adding the released implementation
    # to sys.path. They are not AutoModel implementations.
    import image_processor as reference_image_processor  # noqa: PLC0415
    import model as reference_model  # noqa: PLC0415

    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError:
        # The extension is a performance dependency, not a distinct model
        # operation. Keep the released rotation exactly defined when the
        # validation image omits that optional package.
        hadamard_transform = _torch_hadamard_transform
    reference_model.rotate_activation = lambda x: hadamard_transform(x, scale=x.size(-1) ** -0.5)

    rank, world_size, _ = _initialize_distributed()
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)

    with (inference_dir / "config.json").open(encoding="utf-8") as handle:
        model_args = reference_model.ModelArgs(**json.load(handle))
    model_args.max_batch_size = 1
    model_args.max_seq_len = 4096
    model_args.temperature = 0.0
    with torch.device("cuda"):
        model = reference_model.Transformer(model_args)
    missing, unexpected = load_model(
        model,
        args.converted_checkpoint / f"model{rank}-mp{world_size}.safetensors",
        # The released runtime deliberately keeps norms and the LM head in
        # fp32 although their checkpoint tensors are BF16. safetensors<0.7
        # rejects those safe PyTorch casts during its strict preflight, so let
        # load_state_dict perform the cast and enforce key coverage below.
        strict=False,
    )
    if missing or unexpected:
        raise RuntimeError(f"Reference load mismatch: missing={missing}, unexpected={unexpected}")
    # Match inference/generate.py: several released cache/rotary helpers create
    # tensors without an explicit device after checkpoint loading.
    torch.set_default_device("cuda")

    # Transformer.forward calls the head with its next-token default. Force the
    # released head's supported full_logits mode, and avoid allocating Gumbel
    # sampling probabilities for the resulting [B,S,V] tensor.
    original_head_forward = model.head.forward
    model.head.forward = lambda hidden, full_logits=False: original_head_forward(hidden, full_logits=True)
    reference_model.sample = lambda logits, temperature=1.0: logits.argmax(dim=-1)
    model.eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((args.inputs_dir / "manifest.json").read_text(encoding="utf-8"))
    for record in manifest:
        artifact = torch.load(args.inputs_dir / record["path"], map_location="cpu", weights_only=True)
        input_ids = artifact["input_ids"].cuda(non_blocking=True)
        token_types = artifact["vision_token_types"][0]
        visual_positions = (token_types >= 0).nonzero(as_tuple=False).flatten()
        if visual_positions.numel() == 0:
            raise ValueError(f"Parity case {record['case']} has no visual pseudo tokens")
        start = int(visual_positions[0])
        end = int(visual_positions[-1]) + 1
        n_vit_h, n_vit_w = map(int, artifact["image_grid_hws"][0].tolist())
        n_llm_h = (n_vit_h + model_args.vision_downsample_ratio - 1) // model_args.vision_downsample_ratio
        n_llm_w = (n_vit_w + model_args.vision_downsample_ratio - 1) // model_args.vision_downsample_ratio
        reference_types, permutation = reference_image_processor.build_image_block(n_llm_h, n_llm_w, start)
        if not torch.equal(reference_types.cpu(), token_types[start:end]):
            raise ValueError(f"Reference N-layout differs for parity case {record['case']}")
        image = reference_image_processor.ImageInput(
            start=start,
            patches=artifact["pixel_values"],
            n_vit_h=n_vit_h,
            n_vit_w=n_vit_w,
            types=reference_types,
            perm=permutation,
        )

        with torch.inference_mode():
            _, logits, _ = model(input_ids, start_pos=0, images=[[image]])
        if logits.shape != (1, 4096, model_args.vocab_size):
            raise RuntimeError(f"Unexpected reference logits shape: {tuple(logits.shape)}")
        if rank == 0:
            output = args.output_dir / record["path"].replace(".pt", "_reference_logits.pt")
            torch.save(logits.cpu().to(torch.bfloat16), output)
            print(
                json.dumps(
                    {
                        "case": record["case"],
                        "source_size": record["source_size"],
                        "logits": str(output),
                        "shape": list(logits.shape),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        del artifact, input_ids, logits
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
