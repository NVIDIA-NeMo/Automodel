# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compare full DSV4 Vision AutoModel logits with the released reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config


def _mean_reference_kl(
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    *,
    chunk_tokens: int,
    vision_token_types: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute parity metrics without materializing full fp32 logits copies."""
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(f"Logit shapes differ: {reference_logits.shape} vs {candidate_logits.shape}")
    kl_sum = torch.zeros((), device=candidate_logits.device, dtype=torch.float64)
    abs_sum = torch.zeros_like(kl_sum)
    max_abs = torch.zeros((), device=candidate_logits.device, dtype=torch.float32)
    cosine_dot = torch.zeros_like(kl_sum)
    cosine_ref_sq = torch.zeros_like(kl_sum)
    cosine_candidate_sq = torch.zeros_like(kl_sum)
    top1_matches = torch.zeros((), device=candidate_logits.device, dtype=torch.long)
    token_kls = []
    token_count = reference_logits.shape[0] * reference_logits.shape[1]
    element_count = reference_logits.numel()
    for start in range(0, reference_logits.shape[1], chunk_tokens):
        end = min(start + chunk_tokens, reference_logits.shape[1])
        ref = reference_logits[:, start:end].to(device=candidate_logits.device, dtype=torch.float32)
        candidate = candidate_logits[:, start:end].float()
        ref_log_probs = F.log_softmax(ref, dim=-1)
        candidate_log_probs = F.log_softmax(candidate, dim=-1)
        token_kl = F.kl_div(candidate_log_probs, ref_log_probs, reduction="none", log_target=True).sum(dim=-1)
        kl_sum += token_kl.sum().double()
        token_kls.append(token_kl.detach().cpu())
        delta = (candidate - ref).abs()
        abs_sum += delta.sum().double()
        max_abs = torch.maximum(max_abs, delta.max())
        cosine_dot += (candidate * ref).sum().double()
        cosine_ref_sq += ref.square().sum().double()
        cosine_candidate_sq += candidate.square().sum().double()
        top1_matches += (candidate.argmax(dim=-1) == ref.argmax(dim=-1)).sum()
    token_kl = torch.cat(token_kls, dim=1).flatten()
    metrics = {
        "mean_kl_reference_to_automodel": float((kl_sum / token_count).item()),
        "mean_abs_logit_delta": float((abs_sum / element_count).item()),
        "max_abs_logit_delta": float(max_abs.item()),
        "logits_cosine_similarity": float((cosine_dot / torch.sqrt(cosine_ref_sq * cosine_candidate_sq)).item()),
        "top1_token_agreement": float((top1_matches.float() / token_count).item()),
        "max_token_kl": float(token_kl.max().item()),
        "max_token_kl_position": int(token_kl.argmax().item()),
    }
    if vision_token_types is not None:
        visual_mask = vision_token_types.flatten().cpu() >= 0
        if visual_mask.numel() != token_kl.numel():
            raise ValueError("vision_token_types does not match the logits sequence")
        text_mask = ~visual_mask
        metrics["mean_kl_visual_tokens"] = float(token_kl[visual_mask].mean().item())
        metrics["mean_kl_text_tokens"] = float(token_kl[text_mask].mean().item())
        visual_positions = visual_mask.nonzero(as_tuple=False).flatten()
        if visual_positions.numel() > 0 and int(visual_positions[-1]) + 1 < token_kl.numel():
            post_image = token_kl[int(visual_positions[-1]) + 1 :]
            metrics["mean_kl_post_image"] = float(post_image.mean().item())
            metrics["mean_kl_first_512_post_image"] = float(post_image[:512].mean().item())
    return metrics


def main() -> None:
    """Load the production EP model and validate all reference logits cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--inputs-dir", type=Path, required=True)
    parser.add_argument("--reference-logits-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mean-kl-threshold",
        type=float,
        default=0.03,
        help="Maximum mean KL over every token in every 4k image case.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=8)
    args = parser.parse_args()

    dist_info = initialize_distributed("nccl", timeout_minutes=120)
    cfg = ConfigNode(
        {
            "distributed": {
                "strategy": "fsdp2",
                "tp_size": 1,
                "cp_size": 1,
                "pp_size": 1,
                "ep_size": dist_info.world_size,
                "activation_checkpointing": False,
                "moe": {"reshard_after_forward": False, "wrap_outer_model": True},
            }
        }
    )
    distributed_setup = create_distributed_setup_from_config(cfg, world_size=dist_info.world_size)
    config = DeepseekV4Config.from_pretrained(
        str(args.checkpoint),
        name_or_path=str(args.checkpoint),
        num_nextn_predict_layers=0,
        local_files_only=True,
    )
    backend = BackendConfig(
        attn="tilelang",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        dispatcher="hybridep",
        experts="torch_mm",
        gate_precision="float32",
        enable_hf_state_dict_adapter=True,
        enable_fsdp_optimizations=True,
    )
    model = NeMoAutoModelForImageTextToText.from_config(
        config=config,
        backend=backend,
        distributed_setup=distributed_setup,
        load_base_model=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_liger_kernel=False,
        use_sdpa_patching=False,
    )
    model.requires_grad_(False)
    model.eval()

    manifest = json.loads((args.inputs_dir / "manifest.json").read_text(encoding="utf-8"))
    results = []
    for record in manifest:
        artifact = torch.load(args.inputs_dir / record["path"], map_location="cpu", weights_only=True)
        model_inputs = {
            key: artifact[key].to(dist_info.device, non_blocking=True)
            for key in (
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_hws",
                "n_images_per_sample",
                "vision_token_types",
            )
        }
        with torch.inference_mode():
            output = model(**model_inputs)
        logits = output.logits if hasattr(output, "logits") else output[0]
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        metrics = None
        if dist_info.is_main:
            reference_path = args.reference_logits_dir / record["path"].replace(".pt", "_reference_logits.pt")
            reference_logits = torch.load(reference_path, map_location="cpu", weights_only=True)
            metrics = _mean_reference_kl(
                reference_logits,
                logits,
                chunk_tokens=args.chunk_tokens,
                vision_token_types=artifact["vision_token_types"],
            )
            metrics.update(
                {
                    "case": record["case"],
                    "source_size": record["source_size"],
                    "vit_grid_hw": record["vit_grid_hw"],
                    "sequence_length": record["sequence_length"],
                    "visual_block_tokens": record["visual_block_tokens"],
                }
            )
            del reference_logits
        metrics_object = [metrics]
        dist.broadcast_object_list(metrics_object, src=0)
        metrics = metrics_object[0]
        if metrics is None:
            raise RuntimeError("Rank 0 did not publish DeepSeek-V4 Vision parity metrics")
        results.append(metrics)
        if dist_info.is_main:
            print(json.dumps(metrics, sort_keys=True), flush=True)
        del artifact, model_inputs, output, logits
        torch.cuda.empty_cache()
        dist.barrier()

    aggregate = {
        "cases": results,
        "case_count": len(results),
        "sequence_length": 4096,
        "mean_kl_reference_to_automodel": sum(result["mean_kl_reference_to_automodel"] for result in results)
        / len(results),
        "max_case_mean_kl_reference_to_automodel": max(result["mean_kl_reference_to_automodel"] for result in results),
        "mean_kl_threshold": args.mean_kl_threshold,
    }
    # The requested parity metric is the mean over the complete 4k logits,
    # including every visual pseudo-token position and every image shape in the
    # manifest. Keep the worst individual case as a diagnostic, but do not
    # silently turn a mean-KL contract into a max-over-cases contract.
    aggregate["passed"] = aggregate["mean_kl_reference_to_automodel"] <= args.mean_kl_threshold
    if dist_info.is_main:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"aggregate": aggregate}, sort_keys=True), flush=True)
    dist.barrier()
    if not aggregate["passed"]:
        raise RuntimeError(
            "DeepSeek-V4 Vision logits parity failed: "
            f"aggregate mean KL {aggregate['mean_kl_reference_to_automodel']:.6g} "
            f"> {args.mean_kl_threshold:.6g}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
