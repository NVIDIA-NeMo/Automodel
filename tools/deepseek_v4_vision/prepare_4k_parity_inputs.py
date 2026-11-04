# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Create deterministic 4k DeepSeek-V4 Vision parity inputs.

The cases deliberately cover square, ragged landscape/portrait, and aspect
ratios beyond the model's 8:1 clamp.  Artifacts contain the exact pseudo-token
layout and reference-normalized patches, so both the released implementation
and AutoModel consume identical tensors.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nemo_automodel.components.models.deepseek_v4.processing import (
    IMAGE,
    DeepseekV4VisionProcessor,
)

IMAGE_SIZES = (
    (384, 384),
    (511, 377),
    (377, 511),
    (1000, 60),
    (60, 1000),
)


def _make_image(width: int, height: int, seed: int) -> Image.Image:
    """Return deterministic non-constant RGB pixels for one size."""
    y, x = np.indices((height, width), dtype=np.uint32)
    rgb = np.stack(
        [
            (x * 17 + y * 3 + seed * 29) % 256,
            (x * 5 + y * 19 + seed * 11) % 256,
            (x * 13 + y * 7 + seed * 23) % 256,
        ],
        axis=-1,
    ).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _pad_to_sequence_length(batch: dict[str, torch.Tensor], tokenizer, sequence_length: int) -> None:
    """Append deterministic ordinary text IDs until the model length is exact."""
    current = int(batch["input_ids"].shape[1])
    if current > sequence_length:
        raise ValueError(f"Expanded visual prompt has {current} tokens, above requested {sequence_length}")
    needed = sequence_length - current
    if needed == 0:
        return
    filler = tokenizer.encode(" Medical image parity validation.", add_special_tokens=False)
    if not filler:
        raise ValueError("Tokenizer produced no filler tokens")
    repeated = (filler * ((needed + len(filler) - 1) // len(filler)))[:needed]
    filler_ids = torch.tensor([repeated], dtype=torch.long)
    batch["input_ids"] = torch.cat([batch["input_ids"], filler_ids], dim=1)
    batch["attention_mask"] = torch.cat([batch["attention_mask"], torch.ones_like(filler_ids)], dim=1)
    batch["vision_token_types"] = torch.cat([batch["vision_token_types"], torch.full_like(filler_ids, -1)], dim=1)


def main() -> None:
    """Build every configured parity case and write its manifest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()

    processor = DeepseekV4VisionProcessor.from_pretrained(str(args.checkpoint), local_files_only=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for case_id, (width, height) in enumerate(IMAGE_SIZES):
        image = _make_image(width, height, case_id + 1)
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Describe this {width} by {height} test image."},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Deterministic parity case."}]},
        ]
        batch = processor.apply_chat_template(
            conversation,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        _pad_to_sequence_length(batch, processor.tokenizer, args.sequence_length)
        batch["n_images_per_sample"] = torch.tensor([1], dtype=torch.long)

        types = batch["vision_token_types"][0]
        artifact = {key: value.cpu() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        artifact["width"] = width
        artifact["height"] = height
        output = args.output_dir / f"case_{case_id:02d}_{width}x{height}.pt"
        torch.save(artifact, output)
        record = {
            "case": case_id,
            "path": output.name,
            "source_size": [width, height],
            "vit_grid_hw": artifact["image_grid_hws"][0].tolist(),
            "sequence_length": int(artifact["input_ids"].shape[1]),
            "visual_block_tokens": int((types >= 0).sum()),
            "aligned_image_tokens": int((types == IMAGE).sum()),
            "patches": int(artifact["pixel_values"].shape[0]),
        }
        manifest.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
