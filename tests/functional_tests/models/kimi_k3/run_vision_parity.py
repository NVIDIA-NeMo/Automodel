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

"""Compare K3 vision components against a pinned, separately imported HF source."""

import argparse
import copy
import importlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nemo_automodel.components.models.kimi_k3.config import KimiK3Config
from nemo_automodel.components.models.kimi_k3.vision import (
    MoonViT3dPretrainedModel,
    PatchMergerMLPV2,
    Rope2DPosEmbRepeated,
    apply_rope,
)


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compare same-shaped tensors with arbitrary leading axes.

    Args:
        actual: Tensor of shape [..., hidden].
        expected: Reference tensor of shape [..., hidden].
    Returns:
        Scalar maximum error, relative RMS error, and cosine.
    """
    a, b = actual.detach().float().flatten(), expected.detach().float().flatten()
    return {
        "max_abs": (a - b).abs().max().item(),
        "relative_rms": ((a - b).square().mean().sqrt() / b.square().mean().sqrt().clamp_min(1e-12)).item(),
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--real-checkpoint", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(24)
    torch.set_num_threads(8)
    cls = get_class_from_dynamic_module("modeling_kimi_k3.KimiK3ForConditionalGeneration", str(args.reference))
    hf = importlib.import_module(cls.__module__)
    raw = json.loads((args.reference / "config.json").read_text())
    if not args.real_checkpoint:
        raw["vision_config"].update(
            vt_hidden_size=32,
            vt_intermediate_size=64,
            vt_num_hidden_layers=2,
            vt_num_attention_heads=2,
            qkv_hidden_size=32,
            mm_hidden_size=32,
            text_hidden_size=64,
            patch_size=2,
            init_pos_emb_height=4,
            init_pos_emb_width=4,
        )
        raw["text_config"]["hidden_size"] = 64
    results = {"real_checkpoint": args.real_checkpoint, "reference": str(args.reference), "cases": []}
    cases = (
        [("cpu", torch.float32, "eager")]
        if not args.real_checkpoint
        else [("cuda", torch.bfloat16, "eager"), ("cuda", torch.bfloat16, "flash_attention_2")]
    )
    for device, dtype, backend in cases:
        hf_cfg = hf.KimiK3Config(**copy.deepcopy(raw))
        hf_cfg.vision_config._attn_implementation = backend
        ref_tower = hf.MoonViT3dPretrainedModel(hf.VisionTowerConfig(hf_cfg.vision_config)).to(
            device=device, dtype=dtype
        )
        ref_projector = hf.PatchMergerMLPV2(hf.ProjectorConfig(hf_cfg.vision_config)).to(device=device, dtype=dtype)
        cfg = KimiK3Config(**copy.deepcopy(raw))
        vc = copy.deepcopy(cfg.vision_config)
        vc.hidden_size = vc.vt_hidden_size
        vc.num_hidden_layers = vc.vt_num_hidden_layers
        vc.num_attention_heads = vc.vt_num_attention_heads
        vc.intermediate_size = vc.vt_intermediate_size
        vc.kimi_attn_implementation = backend
        vc._attn_implementation = backend
        tower = MoonViT3dPretrainedModel(vc).to(device=device, dtype=dtype)
        projector = PatchMergerMLPV2(hf.ProjectorConfig(hf_cfg.vision_config)).to(device=device, dtype=dtype)
        if args.real_checkpoint:
            index = json.loads((args.reference / "model.safetensors.index.json").read_text())["weight_map"]
            for prefix, module in [("vision_tower.", ref_tower), ("mm_projector.", ref_projector)]:
                state = {}
                for shard in sorted({v for k, v in index.items() if k.startswith(prefix)}):
                    with safe_open(args.reference / shard, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith(prefix):
                                state[key.removeprefix(prefix)] = f.get_tensor(key)
                module.load_state_dict(state, strict=True)
                del state
        tower.load_state_dict(ref_tower.state_dict(), strict=True)
        projector.load_state_dict(ref_projector.state_dict(), strict=True)
        # Different aspect ratios exercise interpolation and preserve image boundaries.
        grid = torch.tensor([[1, 4, 6], [1, 8, 4], [1, 2, 2]], device=device)
        patch = vc.patch_size
        pixels = torch.randn(60, 3, patch, patch, device=device, dtype=dtype)
        ref_input = pixels.clone().requires_grad_()
        actual_input = pixels.clone().requires_grad_()
        ref_encoded = ref_tower(ref_input, grid)
        actual_encoded = tower(actual_input, grid)
        ref_features = ref_projector(ref_encoded)
        actual_features = projector(actual_encoded)
        case = {
            "device": device,
            "dtype": str(dtype),
            "backend": backend,
            "grid": grid.tolist(),
            "encoder": [],
            "projector": [],
            "grads": {},
        }
        tolerance = 1e-6 if dtype == torch.float32 else 0.0
        for a, b in zip(actual_encoded, ref_encoded):
            case["encoder"].append(metrics(a, b))
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        for a, b in zip(actual_features, ref_features):
            case["projector"].append(metrics(a, b))
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        # A fixed, non-symmetric cotangent exercises every projected feature.
        cotangents = [torch.randn_like(x) for x in ref_features]
        sum((x.float() * g.float()).mean() for x, g in zip(ref_features, cotangents)).backward()
        sum((x.float() * g.float()).mean() for x, g in zip(actual_features, cotangents)).backward()
        torch.testing.assert_close(actual_input.grad, ref_input.grad, atol=tolerance, rtol=tolerance)
        case["input_gradient"] = metrics(actual_input.grad, ref_input.grad)
        for prefix, actual, reference in [("vision", tower, ref_tower), ("projector", projector, ref_projector)]:
            ref_params = dict(reference.named_parameters())
            for name, p in actual.named_parameters():
                q = ref_params[name]
                assert p.grad is not None and q.grad is not None, name
                assert torch.isfinite(p.grad).all(), name
                torch.testing.assert_close(p.grad, q.grad, atol=tolerance, rtol=tolerance)
                case["grads"][prefix + "." + name] = metrics(p.grad, q.grad)
        results["cases"].append(case)
        print(json.dumps({k: v for k, v in case.items() if k != "grads"}), flush=True)
        del tower, projector, ref_tower, ref_projector
        if device == "cuda":
            torch.cuda.empty_cache()
    # Independent trigonometric oracle: adjacent complex pairs alternate x and y.
    dim = 16
    grid = torch.tensor([[1, 4, 6]])
    rope = Rope2DPosEmbRepeated(dim, 8, 8)
    freqs = rope.get_freqs_cis(grid, torch.device("cpu"))
    x = torch.randn(24, 2, dim)
    actual, _ = apply_rope(x, x, freqs)
    coords = torch.tensor([(w, h) for h in range(4) for w in range(6)], dtype=torch.float32)
    angles = torch.stack([coords[:, i % 2] * 10000 ** (-4 * (i // 2) / dim) for i in range(dim // 2)], dim=-1)
    even, odd = x[..., 0::2], x[..., 1::2]
    oracle = torch.stack(
        (
            even * angles.cos()[:, None] - odd * angles.sin()[:, None],
            even * angles.sin()[:, None] + odd * angles.cos()[:, None],
        ),
        dim=-1,
    ).flatten(-2)
    torch.testing.assert_close(actual, oracle, atol=5e-7, rtol=1e-6)
    wrong, _ = apply_rope(x, x, freqs.roll(1, dims=-1))
    assert (wrong - oracle).abs().max() > 0.1
    results["rope_oracle"] = metrics(actual, oracle)
    results["negative_control_max_abs"] = (wrong - oracle).abs().max().item()
    args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
