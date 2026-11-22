#!/usr/bin/env python
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the native Muse Spark Open vision tower with SGLang BF16 features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from safetensors import safe_open

from nemo_automodel.components.models.muse_spark.config import MuseSpark1xOmniConfig
from nemo_automodel.components.models.muse_spark.processing import build_muse_spark_processor
from nemo_automodel.components.models.muse_spark.vision import MuseSparkVisionModel

_PERCEPTION_FILE = "model-perception-00001-of-00001.safetensors"
_INDEX_FILE = "model.safetensors.index.json"
_PROJECTION_KEY = "modalities.vision_perception.encoder.vision_projection.weight"


def parse_args() -> argparse.Namespace:
    """Parse checkpoint, image, reference, and artifact paths."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--reference-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _load_projection(model_path: Path, projection: torch.nn.Linear) -> None:
    index = json.loads((model_path / _INDEX_FILE).read_text())
    shard = model_path / index["weight_map"][_PROJECTION_KEY]
    with safe_open(shard, framework="pt", device="cpu") as checkpoint:
        projection.weight.data.copy_(checkpoint.get_tensor(_PROJECTION_KEY))


def main() -> None:
    """Run the released vision tower on one GPU and compare its BF16 output."""
    args = parse_args()
    config = MuseSpark1xOmniConfig.from_pretrained(args.model)
    vision_model = MuseSparkVisionModel(config)
    with safe_open(args.model / _PERCEPTION_FILE, framework="pt", device="cpu") as checkpoint:
        state = {
            key.removeprefix("vision_model."): checkpoint.get_tensor(key)
            for key in checkpoint.keys()
            if key.startswith("vision_model.")
        }
    vision_model.load_state_dict(state, strict=True)
    del state
    projection = torch.nn.Linear(
        vision_model.output_size,
        config.hidden_size,
        bias=False,
        dtype=torch.bfloat16,
    )
    _load_projection(args.model, projection)
    vision_model = vision_model.cuda().eval()
    projection = projection.cuda().eval()

    processor = build_muse_spark_processor(args.model, fix_mistral_regex=False)
    batch = processor(images=Image.open(args.image), text="<image>", return_tensors="pt")
    pixel_values = batch["pixel_values"].cuda()
    image_grid_thw = batch["image_grid_thw"].cuda()
    with torch.inference_mode():
        features = vision_model(pixel_values, image_grid_thw).last_hidden_state
        features = projection(features)
        features = torch.nn.functional.rms_norm(
            features,
            (features.shape[-1],),
            weight=None,
            eps=config.rms_norm_eps,
        ).float()

    reference = torch.load(args.reference_features, map_location="cuda", weights_only=True).float()
    if reference.shape != features.shape:
        raise ValueError(f"Reference shape {reference.shape} does not match output {features.shape}")
    difference = (features - reference).abs()
    cosine = torch.nn.functional.cosine_similarity(features.flatten(), reference.flatten(), dim=0)
    passed = (
        float(difference.max().item()) < 3.0 and float(difference.mean().item()) < 0.05 and float(cosine.item()) > 0.998
    )
    result = {
        "model": str(args.model),
        "image": str(args.image),
        "shape": list(features.shape),
        "max_abs_error": float(difference.max().item()),
        "mean_abs_error": float(difference.mean().item()),
        "cosine_similarity": float(cosine.item()),
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"MUSE_SPARK_VISION_FEATURE_PARITY: {'PASS' if passed else 'FAIL'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
