# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Opt-in routed-MoE diagnostics for checkpoint cross-framework parity."""

from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _layer_index(module_name: str) -> int:
    parts = module_name.split(".")
    for part_index, part in enumerate(parts[:-1]):
        if part == "layers":
            return int(parts[part_index + 1])
    raise ValueError(f"Could not determine transformer layer index from module name {module_name!r}")


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        return tensor.to_local()
    return tensor


def _persist_capture(path: Path, framework: str, captures: dict[int, dict[str, object]]) -> None:
    if not captures:
        raise RuntimeError(f"No {framework} GLM router calls were captured")
    payload = {
        "schema_version": 1,
        "framework": framework,
        "model_family": "glm4_moe_lite",
        "layers": {
            layer_index: {
                "router_logits": _local_tensor(layer["router_logits"]).to(device="cpu", dtype=torch.float32),
                "correction_bias": _local_tensor(layer["correction_bias"]).to(device="cpu", dtype=torch.float32),
                "indices": _local_tensor(layer["indices"]).to(device="cpu", dtype=torch.int64),
                "score_func": layer["score_func"],
                "n_groups": layer["n_groups"],
            }
            for layer_index, layer in captures.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


@contextmanager
def capture_glm_hf_routers(model: torch.nn.Module, output_path: Path) -> Iterator[None]:
    """Capture GLM router inputs and actual selections during one vanilla-HF forward.

    Args:
        model: Loaded vanilla-HF GLM model.
        output_path: Cross-process capture path owned by the robustness artifact directory.

    Yields:
        Control to exactly one model forward.
    """
    if not _rank0():
        yield
        return

    captures: dict[int, dict[str, object]] = {}
    patched_modules: list[tuple[torch.nn.Module, bool, object | None]] = []
    for module_name, module in model.named_modules():
        if module.__class__.__name__ != "Glm4MoeLiteMoE":
            continue
        layer_index = _layer_index(module_name)
        original_route = module.route_tokens_to_experts
        had_instance_override = "route_tokens_to_experts" in module.__dict__
        original_instance_value = module.__dict__.get("route_tokens_to_experts")

        def capture_route(self, router_logits, *args, _original=original_route, _layer=layer_index, **kwargs):
            result = _original(router_logits, *args, **kwargs)
            indices, _weights = result
            correction_bias = self.gate.e_score_correction_bias
            if correction_bias is None:
                correction_bias = torch.zeros(router_logits.shape[-1], device=router_logits.device)
            captures[_layer] = {
                "router_logits": router_logits.detach(),
                "correction_bias": correction_bias.detach(),
                "indices": indices.detach(),
                "score_func": "sigmoid",
                "n_groups": int(getattr(self.config, "n_group", 1)),
            }
            return result

        module.route_tokens_to_experts = MethodType(capture_route, module)
        patched_modules.append((module, had_instance_override, original_instance_value))

    if not patched_modules:
        raise ValueError("capture_router_diagnostics currently supports only vanilla-HF Glm4MoeLiteMoE models")

    completed = False
    try:
        yield
        completed = True
    finally:
        for module, had_instance_override, original_instance_value in patched_modules:
            if had_instance_override:
                module.route_tokens_to_experts = original_instance_value
            else:
                delattr(module, "route_tokens_to_experts")
        if completed:
            _persist_capture(output_path, "hf", captures)


@contextmanager
def capture_glm_automodel_routers(model: torch.nn.Module, output_path: Path) -> Iterator[None]:
    """Capture GLM router inputs and actual selections during one AutoModel forward.

    Args:
        model: Constructed GLM AutoModel module.
        output_path: Cross-process capture path owned by the robustness artifact directory.

    Yields:
        Control to exactly one model forward.
    """
    if not _rank0():
        yield
        return

    from nemo_automodel.components.moe.layers import Gate

    captures: dict[int, dict[str, object]] = {}
    patched_gates: list[tuple[Gate, bool, object | None]] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, Gate):
            continue
        layer_index = _layer_index(module_name)
        original_route = module._route_scores
        had_instance_override = "_route_scores" in module.__dict__
        original_instance_value = module.__dict__.get("_route_scores")

        def capture_route(self, router_logits, _original=original_route, _layer=layer_index):
            result = _original(router_logits)
            _weights, indices, _original_scores = result
            correction_bias = self._local_score_correction_bias()
            if correction_bias is None:
                correction_bias = torch.zeros(router_logits.shape[-1], device=router_logits.device)
            captures[_layer] = {
                "router_logits": router_logits.detach(),
                "correction_bias": correction_bias.detach(),
                "indices": indices.detach(),
                "score_func": self.score_func,
                "n_groups": self.n_groups,
            }
            return result

        module._route_scores = MethodType(capture_route, module)
        patched_gates.append((module, had_instance_override, original_instance_value))

    if not patched_gates:
        raise ValueError("capture_router_diagnostics requires an AutoModel with learned MoE Gate modules")

    completed = False
    try:
        yield
        completed = True
    finally:
        for module, had_instance_override, original_instance_value in patched_gates:
            if had_instance_override:
                module._route_scores = original_instance_value
            else:
                delattr(module, "_route_scores")
        if completed:
            _persist_capture(output_path, "automodel", captures)


def _quantiles(values: torch.Tensor) -> dict[str, float] | None:
    if values.numel() == 0:
        return None
    values = values.float()
    quantiles = torch.quantile(values, torch.tensor([0.0, 0.05, 0.5, 0.95, 1.0]))
    return {
        "min": quantiles[0].item(),
        "p05": quantiles[1].item(),
        "median": quantiles[2].item(),
        "p95": quantiles[3].item(),
        "max": quantiles[4].item(),
    }


def _token_kl(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(
            f"Final-logit shape mismatch: {tuple(reference_logits.shape)} != {tuple(candidate_logits.shape)}"
        )
    vocab_size = reference_logits.shape[-1]
    reference_tokens = reference_logits.detach().reshape(-1, vocab_size)
    candidate_tokens = candidate_logits.detach().reshape(-1, vocab_size)
    token_kl_chunks = []
    for start in range(0, reference_tokens.shape[0], 16):
        reference_log_probs = F.log_softmax(reference_tokens[start : start + 16].float(), dim=-1)
        candidate_log_probs = F.log_softmax(candidate_tokens[start : start + 16].float(), dim=-1)
        token_kl_chunks.append(
            (reference_log_probs.exp() * (reference_log_probs - candidate_log_probs)).sum(dim=-1).cpu()
        )
    return torch.cat(token_kl_chunks)


def _kl_summary(values: torch.Tensor) -> dict[str, int | float] | None:
    if values.numel() == 0:
        return None
    return {
        "token_count": values.numel(),
        "mean_kl": values.mean().item(),
        "p95_kl": torch.quantile(values, 0.95).item(),
        "max_kl": values.max().item(),
    }


def compare_glm_router_captures(
    hf_path: Path,
    automodel_path: Path,
    report_path: Path,
    *,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
) -> dict[str, object]:
    """Compare paired GLM captures and persist evidence for near-tie route flips.

    Args:
        hf_path: Router capture from the vanilla-HF source forward.
        automodel_path: Router capture from the AutoModel source forward.
        report_path: JSON report path.
        reference_logits: Vanilla-HF final logits from the captured forward.
        candidate_logits: AutoModel final logits from the captured forward.

    Returns:
        Machine-readable aggregate and per-layer router diagnostics.
    """
    hf_capture = torch.load(hf_path, map_location="cpu", weights_only=True)
    automodel_capture = torch.load(automodel_path, map_location="cpu", weights_only=True)
    hf_layers = hf_capture["layers"]
    automodel_layers = automodel_capture["layers"]
    if set(hf_layers) != set(automodel_layers):
        raise ValueError(
            f"HF and AutoModel router layer sets differ: {sorted(hf_layers)} != {sorted(automodel_layers)}"
        )

    layer_metrics: dict[int, dict[str, object]] = {}
    all_hf_margins = []
    all_flip_margins = []
    all_flip_score_deltas = []
    total_layer_tokens = 0
    total_route_flips = 0
    total_explained_flips = 0
    total_router_values = 0
    logit_abs_sum = 0.0
    logit_abs_max = 0.0
    score_abs_sum = 0.0
    score_abs_max = 0.0
    logit_dot = 0.0
    hf_logit_square_sum = 0.0
    automodel_logit_square_sum = 0.0
    bias_abs_max = 0.0
    examples: list[dict[str, object]] = []
    flipped_layer_counts: torch.Tensor | None = None

    for layer_index in sorted(hf_layers):
        hf_layer = hf_layers[layer_index]
        automodel_layer = automodel_layers[layer_index]
        if hf_layer["score_func"] != "sigmoid" or automodel_layer["score_func"] != "sigmoid":
            raise ValueError("GLM router diagnostics currently require sigmoid routing")
        if hf_layer["n_groups"] != 1 or automodel_layer["n_groups"] != 1:
            raise ValueError("GLM router boundary diagnostics currently require n_groups=1")

        hf_logits = hf_layer["router_logits"].reshape(-1, hf_layer["router_logits"].shape[-1]).float()
        automodel_logits = automodel_layer["router_logits"].reshape_as(hf_logits).float()
        hf_indices = hf_layer["indices"].reshape(-1, hf_layer["indices"].shape[-1]).long()
        automodel_indices = automodel_layer["indices"].reshape_as(hf_indices).long()
        if hf_logits.shape != automodel_logits.shape or hf_indices.shape != automodel_indices.shape:
            raise ValueError(f"Router capture shape mismatch in layer {layer_index}")

        hf_bias = hf_layer["correction_bias"].float()
        automodel_bias = automodel_layer["correction_bias"].float()
        hf_scores = hf_logits.sigmoid() + hf_bias
        automodel_scores = automodel_logits.sigmoid() + automodel_bias
        topk = hf_indices.shape[-1]
        if hf_scores.shape[-1] <= topk:
            raise ValueError(f"Layer {layer_index} needs at least topk+1 router scores for boundary diagnostics")

        route_flip_mask = (hf_indices.sort(dim=-1).values != automodel_indices.sort(dim=-1).values).any(dim=-1)
        if flipped_layer_counts is None:
            flipped_layer_counts = torch.zeros_like(route_flip_mask, dtype=torch.int64)
        elif flipped_layer_counts.shape != route_flip_mask.shape:
            raise ValueError("Router layers contain different token counts")
        flipped_layer_counts += route_flip_mask
        score_delta_per_token = (hf_scores - automodel_scores).abs().amax(dim=-1)
        top_values = hf_scores.topk(topk + 1, dim=-1).values
        hf_boundary_margin = top_values[:, topk - 1] - top_values[:, topk]
        explained_flip_mask = route_flip_mask & (hf_boundary_margin <= 2 * score_delta_per_token)

        logit_delta = (hf_logits - automodel_logits).abs()
        score_delta = (hf_scores - automodel_scores).abs()
        layer_logit_dot = torch.sum(hf_logits.double() * automodel_logits.double()).item()
        layer_hf_square_sum = torch.sum(hf_logits.double().square()).item()
        layer_automodel_square_sum = torch.sum(automodel_logits.double().square()).item()
        layer_cosine_denominator = (layer_hf_square_sum * layer_automodel_square_sum) ** 0.5
        layer_flip_count = int(route_flip_mask.sum().item())
        layer_explained_count = int(explained_flip_mask.sum().item())

        layer_metrics[layer_index] = {
            "token_count": hf_logits.shape[0],
            "route_flip_token_count": layer_flip_count,
            "route_flip_token_fraction": layer_flip_count / hf_logits.shape[0],
            "router_logit_mean_abs_diff": logit_delta.mean().item(),
            "router_logit_max_abs_diff": logit_delta.max().item(),
            "router_logit_cosine": layer_logit_dot / layer_cosine_denominator,
            "routing_score_mean_abs_diff": score_delta.mean().item(),
            "routing_score_max_abs_diff": score_delta.max().item(),
            "correction_bias_max_abs_diff": (hf_bias - automodel_bias).abs().max().item(),
            "hf_topk_boundary_margin": _quantiles(hf_boundary_margin),
            "flipped_hf_topk_boundary_margin": _quantiles(hf_boundary_margin[route_flip_mask]),
            "flipped_max_routing_score_diff": _quantiles(score_delta_per_token[route_flip_mask]),
            "flips_within_score_perturbation_bound_fraction": (
                layer_explained_count / layer_flip_count if layer_flip_count else None
            ),
        }

        if len(examples) < 12:
            for token_index in route_flip_mask.nonzero(as_tuple=False).flatten().tolist():
                examples.append(
                    {
                        "layer": layer_index,
                        "flattened_token_index": token_index,
                        "hf_experts": hf_indices[token_index].tolist(),
                        "automodel_experts": automodel_indices[token_index].tolist(),
                        "hf_topk_boundary_margin": hf_boundary_margin[token_index].item(),
                        "max_routing_score_diff": score_delta_per_token[token_index].item(),
                    }
                )
                if len(examples) == 12:
                    break

        all_hf_margins.append(hf_boundary_margin)
        all_flip_margins.append(hf_boundary_margin[route_flip_mask])
        all_flip_score_deltas.append(score_delta_per_token[route_flip_mask])
        total_layer_tokens += hf_logits.shape[0]
        total_route_flips += layer_flip_count
        total_explained_flips += layer_explained_count
        total_router_values += hf_logits.numel()
        logit_abs_sum += logit_delta.sum().item()
        logit_abs_max = max(logit_abs_max, logit_delta.max().item())
        score_abs_sum += score_delta.sum().item()
        score_abs_max = max(score_abs_max, score_delta.max().item())
        logit_dot += layer_logit_dot
        hf_logit_square_sum += layer_hf_square_sum
        automodel_logit_square_sum += layer_automodel_square_sum
        bias_abs_max = max(bias_abs_max, (hf_bias - automodel_bias).abs().max().item())

    cosine_denominator = (hf_logit_square_sum * automodel_logit_square_sum) ** 0.5
    assert flipped_layer_counts is not None
    per_token_kl = _token_kl(reference_logits, candidate_logits)
    if per_token_kl.shape != flipped_layer_counts.shape:
        raise ValueError(
            f"Router capture has {flipped_layer_counts.numel()} tokens, but final logits have {per_token_kl.numel()}"
        )
    natural_agreement_mask = flipped_layer_counts == 0
    routed_tail_mask = ~natural_agreement_mask
    token_kl_by_flipped_layer_count = {
        int(flip_count): _kl_summary(per_token_kl[flipped_layer_counts == flip_count])
        for flip_count in torch.unique(flipped_layer_counts).tolist()
    }
    report: dict[str, object] = {
        "schema_version": 1,
        "model_family": "glm4_moe_lite",
        "comparison": "hf_source_vs_automodel_source",
        "layer_count": len(layer_metrics),
        "layer_token_count": total_layer_tokens,
        "route_flip_token_count": total_route_flips,
        "route_flip_token_fraction": total_route_flips / total_layer_tokens,
        "router_logit_mean_abs_diff": logit_abs_sum / total_router_values,
        "router_logit_max_abs_diff": logit_abs_max,
        "router_logit_cosine": logit_dot / cosine_denominator,
        "routing_score_mean_abs_diff": score_abs_sum / total_router_values,
        "routing_score_max_abs_diff": score_abs_max,
        "correction_bias_max_abs_diff": bias_abs_max,
        "hf_topk_boundary_margin": _quantiles(torch.cat(all_hf_margins)),
        "flipped_hf_topk_boundary_margin": _quantiles(torch.cat(all_flip_margins)),
        "flipped_max_routing_score_diff": _quantiles(torch.cat(all_flip_score_deltas)),
        "flips_within_score_perturbation_bound_fraction": (
            total_explained_flips / total_route_flips if total_route_flips else None
        ),
        "final_token_kl": {
            "all_tokens": _kl_summary(per_token_kl),
            "natural_agreement_floor_no_flipped_layers": _kl_summary(per_token_kl[natural_agreement_mask]),
            "routed_tail_one_or_more_flipped_layers": _kl_summary(per_token_kl[routed_tail_mask]),
            "by_flipped_layer_count": token_kl_by_flipped_layer_count,
        },
        "layer_metrics": layer_metrics,
        "examples": examples,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_report_path = report_path.with_suffix(".tmp")
    temporary_report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    temporary_report_path.replace(report_path)
    print(f"CHECKPOINT_ROUTER_DIAGNOSTICS {json.dumps(report, sort_keys=True)}")
    return report
