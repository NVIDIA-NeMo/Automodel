# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare the released and AutoModel DSV4 ViT/aligner on real image tensors."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from safetensors import safe_open

from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.processing import build_image_block
from nemo_automodel.components.models.deepseek_v4.vision import (
    DeepseekV4VisionAligner,
    DeepseekV4VisionTransformer,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_vision_state(checkpoint: Path) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))["weight_map"]
    filenames = {filename for name, filename in index.items() if name.startswith(("vision.", "aligner."))}
    if len(filenames) != 1:
        raise ValueError(f"Expected vision weights in one shard, got {sorted(filenames)}")
    vision_state = {}
    aligner_state = {}
    with safe_open(checkpoint / filenames.pop(), framework="pt", device="cpu") as handle:
        for name in handle.keys():
            if name.startswith("vision."):
                vision_state[name.removeprefix("vision.")] = handle.get_tensor(name)
            elif name.startswith("aligner."):
                aligner_state[name.removeprefix("aligner.")] = handle.get_tensor(name)
    return vision_state, aligner_state


def _parity(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference_float = reference.float()
    candidate_float = candidate.float()
    delta = (candidate_float - reference_float).abs()
    return {
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "cosine": float(F.cosine_similarity(reference_float.flatten(), candidate_float.flatten(), dim=0).item()),
    }


def main() -> None:
    """Load real weights once, then compare all prepared image-size cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Real 32-layer vision parity requires a CUDA device")
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")

    reference_vision = _load_module("deepseek_v4_released_vision", args.checkpoint / "inference" / "vision.py")
    reference_processor = _load_module(
        "deepseek_v4_released_image_processor", args.checkpoint / "inference" / "image_processor.py"
    )
    config = DeepseekV4Config.from_pretrained(str(args.checkpoint), local_files_only=True)
    reference_config = config.to_dict()
    reference_config["dim"] = int(config.hidden_size)
    reference_args = SimpleNamespace(**reference_config)

    reference_vit = reference_vision.ViT(reference_args).cuda().eval()
    reference_aligner = reference_vision.Aligner(reference_args).cuda().eval()
    candidate_vit = DeepseekV4VisionTransformer(config).cuda().eval()
    candidate_aligner = DeepseekV4VisionAligner(config).cuda().eval()
    vision_state, aligner_state = _load_vision_state(args.checkpoint)
    for module, state in (
        (reference_vit, vision_state),
        (candidate_vit, vision_state),
        (reference_aligner, aligner_state),
        (candidate_aligner, aligner_state),
    ):
        missing, unexpected = module.load_state_dict(state, strict=True)
        if missing or unexpected:
            raise RuntimeError(f"Vision state mismatch: missing={missing}, unexpected={unexpected}")
    del vision_state, aligner_state

    manifest = json.loads((args.inputs_dir / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for record in manifest:
        artifact = torch.load(args.inputs_dir / record["path"], map_location="cpu", weights_only=True)
        patches = artifact["pixel_values"].cuda()
        n_vit_h, n_vit_w = map(int, artifact["image_grid_hws"][0].tolist())
        with torch.inference_mode():
            reference_encoded = reference_vit(patches, n_vit_h, n_vit_w)
            candidate_encoded = candidate_vit(patches, n_vit_h, n_vit_w)
            reference_aligned = reference_aligner(reference_encoded, n_vit_h, n_vit_w)
            candidate_aligned = candidate_aligner(candidate_encoded, n_vit_h, n_vit_w)

        n_llm_h = (n_vit_h + int(config.vision_downsample_ratio) - 1) // int(config.vision_downsample_ratio)
        n_llm_w = (n_vit_w + int(config.vision_downsample_ratio) - 1) // int(config.vision_downsample_ratio)
        candidate_types, candidate_perm = build_image_block(n_llm_h, n_llm_w, start_pos=2)
        reference_types, reference_perm = reference_processor.build_image_block(n_llm_h, n_llm_w, start_pos=2)
        result = {
            "case": record["case"],
            "source_size": record["source_size"],
            "vit_grid_hw": record["vit_grid_hw"],
            "vision": _parity(reference_encoded, candidate_encoded),
            "aligner": _parity(reference_aligned, candidate_aligned),
            "types_exact": bool(torch.equal(reference_types.cpu(), candidate_types.cpu())),
            "permutation_exact": bool(torch.equal(reference_perm.cpu(), candidate_perm.cpu())),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    passed = all(
        result["vision"]["max_abs"] == 0.0
        and result["aligner"]["max_abs"] == 0.0
        and result["types_exact"]
        and result["permutation_exact"]
        for result in results
    )
    report = {"passed": passed, "cases": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("DeepSeek-V4 real-weight vision embedding parity failed")


if __name__ == "__main__":
    main()
