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

"""Compare Ministral parameters loaded from FP8 and BF16 checkpoints."""

import argparse
import logging
import math
from collections import Counter
from dataclasses import dataclass

import torch
import transformers
from torch import nn
from transformers import FineGrainedFP8Config, Mistral3ForConditionalGeneration

FP8_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512"
BF16_MODEL_ID = "mistralai/Ministral-3-3B-Instruct-2512-BF16"


@dataclass(frozen=True)
class _ComparisonConfig:
    """Numerical comparison settings."""

    chunk_elements: int
    atol: float
    rtol: float
    top_k: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-model", default=FP8_MODEL_ID)
    parser.add_argument("--bf16-model", default=BF16_MODEL_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--chunk-elements", type=int, default=4_194_304)
    parser.add_argument("--atol", type=float, default=1e-3)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def _describe_model(label: str, model: nn.Module) -> None:
    """Log parameter storage details for one loaded model."""
    parameters = list(model.named_parameters())
    buffers = list(model.named_buffers())
    parameter_dtypes = Counter(str(parameter.dtype) for _, parameter in parameters)
    buffer_dtypes = Counter(str(buffer.dtype) for _, buffer in buffers)
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for _, parameter in parameters)

    logging.info("\n%s", label)
    logging.info("  class: %s.%s", type(model).__module__, type(model).__name__)
    logging.info("  parameter dtypes: %s", dict(sorted(parameter_dtypes.items())))
    logging.info("  buffer dtypes: %s", dict(sorted(buffer_dtypes.items())))
    logging.info("  unique parameter storage: %.3f GiB", parameter_bytes / 2**30)
    logging.info("  device map: %s", model.hf_device_map)


def _compare_parameters(
    left_label: str,
    left_model: nn.Module,
    right_label: str,
    right_model: nn.Module,
    config: _ComparisonConfig,
) -> None:
    """Compare matching model parameters after casting chunks to BF16.

    Args:
        left_label: Display label for the left model.
        left_model: Module whose named parameter tensors may have arbitrary shapes and devices.
        right_label: Display label for the right model.
        right_model: Module whose named parameter tensors may have arbitrary shapes and devices.
        config: Chunk size, closeness tolerances, and reporting limit.
    """
    left_parameters = dict(left_model.named_parameters())
    right_parameters = dict(right_model.named_parameters())
    left_names = set(left_parameters)
    right_names = set(right_parameters)
    left_only = sorted(left_names - right_names)
    right_only = sorted(right_names - left_names)
    shared_names = sorted(left_names & right_names)

    logging.info("\n=== %s vs %s ===", left_label, right_label)
    logging.info("shared parameter names: %d", len(shared_names))
    logging.info("only in %s (%d): %s", left_label, len(left_only), left_only[:20])
    logging.info("only in %s (%d): %s", right_label, len(right_only), right_only[:20])

    shape_mismatches: list[str] = []
    dtype_pairs: Counter[str] = Counter()
    top_differences: list[tuple[float, float, str]] = []
    compared_tensors = 0
    compared_elements = 0
    exact_elements = 0
    close_elements = 0
    nonfinite_elements = 0
    absolute_error_sum = 0.0
    squared_error_sum = 0.0
    dot_product_sum = 0.0
    left_squared_sum = 0.0
    right_squared_sum = 0.0
    global_max_error = 0.0

    with torch.no_grad():
        for parameter_index, name in enumerate(shared_names, start=1):
            left_parameter = left_parameters[name].detach()
            right_parameter = right_parameters[name].detach()
            if left_parameter.shape != right_parameter.shape:
                shape_mismatches.append(f"{name}: {tuple(left_parameter.shape)} != {tuple(right_parameter.shape)}")
                continue
            if left_parameter.device.type == "meta" or right_parameter.device.type == "meta":
                raise RuntimeError(f"Cannot compare meta parameter {name!r}")

            dtype_pairs[f"{left_parameter.dtype} vs {right_parameter.dtype}"] += 1
            left_flat = left_parameter.view(-1)
            right_flat = right_parameter.view(-1)
            tensor_max_error = 0.0
            tensor_absolute_error_sum = 0.0

            for start in range(0, left_flat.numel(), config.chunk_elements):
                end = min(start + config.chunk_elements, left_flat.numel())
                left_bf16 = left_flat[start:end].to(dtype=torch.bfloat16)
                right_bf16 = right_flat[start:end].to(device=left_bf16.device, dtype=torch.bfloat16)
                left_float = left_bf16.float()
                right_float = right_bf16.float()
                difference = (left_float - right_float).abs()

                chunk_elements = end - start
                chunk_max_error = difference.max().item() if chunk_elements else 0.0
                chunk_absolute_error = difference.sum(dtype=torch.float64).item()
                tensor_max_error = max(tensor_max_error, chunk_max_error)
                tensor_absolute_error_sum += chunk_absolute_error
                global_max_error = max(global_max_error, chunk_max_error)
                absolute_error_sum += chunk_absolute_error
                squared_error_sum += difference.square().sum(dtype=torch.float64).item()
                dot_product_sum += (left_float * right_float).sum(dtype=torch.float64).item()
                left_squared_sum += left_float.square().sum(dtype=torch.float64).item()
                right_squared_sum += right_float.square().sum(dtype=torch.float64).item()
                exact_elements += (left_bf16 == right_bf16).sum().item()
                close_elements += (
                    torch.isclose(
                        left_float,
                        right_float,
                        atol=config.atol,
                        rtol=config.rtol,
                        equal_nan=True,
                    )
                    .sum()
                    .item()
                )
                nonfinite_elements += (~torch.isfinite(left_float) | ~torch.isfinite(right_float)).sum().item()
                compared_elements += chunk_elements

            compared_tensors += 1
            tensor_mean_error = tensor_absolute_error_sum / max(left_flat.numel(), 1)
            top_differences.append((tensor_max_error, tensor_mean_error, name))
            if parameter_index % 50 == 0 or parameter_index == len(shared_names):
                logging.info("compared %d/%d shared parameter names", parameter_index, len(shared_names))

    if compared_elements == 0:
        raise RuntimeError(f"No same-shaped parameters were available for {left_label} vs {right_label}")
    norm_product = left_squared_sum * right_squared_sum
    cosine_similarity = dot_product_sum / math.sqrt(norm_product) if norm_product > 0.0 else float("nan")
    mean_absolute_error = absolute_error_sum / max(compared_elements, 1)
    root_mean_squared_error = math.sqrt(squared_error_sum / max(compared_elements, 1))
    exact_fraction = exact_elements / max(compared_elements, 1)
    close_fraction = close_elements / max(compared_elements, 1)

    logging.info("shape mismatches (%d): %s", len(shape_mismatches), shape_mismatches[:20])
    logging.info("dtype pairs by parameter name: %s", dict(sorted(dtype_pairs.items())))
    logging.info("compared tensors/elements: %d / %d", compared_tensors, compared_elements)
    logging.info("nonfinite elements: %d", nonfinite_elements)
    logging.info("exact BF16 elements: %.8f%%", 100.0 * exact_fraction)
    logging.info(
        "close BF16 elements (atol=%g, rtol=%g): %.8f%%",
        config.atol,
        config.rtol,
        100.0 * close_fraction,
    )
    logging.info("max absolute error: %.9e", global_max_error)
    logging.info("mean absolute error: %.9e", mean_absolute_error)
    logging.info("RMSE: %.9e", root_mean_squared_error)
    logging.info("global cosine similarity: %.12f", cosine_similarity)
    logging.info("top %d parameters by max error:", config.top_k)
    for max_error, mean_error, name in sorted(top_differences, reverse=True)[: config.top_k]:
        logging.info("  max=%9.3e mean=%9.3e %s", max_error, mean_error, name)


def main() -> int:
    """Load three model variants and compare their parameter values."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required to load and compare these models")
    if args.chunk_elements <= 0:
        raise ValueError(f"--chunk-elements must be positive, got {args.chunk_elements}")
    if args.top_k <= 0:
        raise ValueError(f"--top-k must be positive, got {args.top_k}")
    if args.atol < 0.0 or args.rtol < 0.0:
        raise ValueError(f"--atol and --rtol must be nonnegative, got atol={args.atol}, rtol={args.rtol}")

    logging.info("torch=%s transformers=%s", torch.__version__, transformers.__version__)
    logging.info("Loading A: FP8 checkpoint with dtype=torch.bfloat16")
    model_a = Mistral3ForConditionalGeneration.from_pretrained(
        args.fp8_model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    ).eval()
    logging.info("Loading B: FP8 checkpoint with FineGrainedFP8Config(dequantize=True)")
    model_b = Mistral3ForConditionalGeneration.from_pretrained(
        args.fp8_model,
        revision=args.revision,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
        quantization_config=FineGrainedFP8Config(dequantize=True),
    ).eval()
    logging.info("Loading C: BF16 checkpoint with dtype=torch.bfloat16")
    model_c = Mistral3ForConditionalGeneration.from_pretrained(
        args.bf16_model,
        revision=args.revision,
        dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    ).eval()

    _describe_model("A: FP8 checkpoint + dtype=BF16", model_a)
    _describe_model("B: FP8 checkpoint + dequantize=True", model_b)
    _describe_model("C: BF16 checkpoint + dtype=BF16", model_c)

    raw_fp8_dtypes = {parameter.dtype for parameter in model_a.parameters()}
    if any(str(dtype).startswith("torch.float8") for dtype in raw_fp8_dtypes):
        logging.warning(
            "A still contains FP8 parameters. A-vs-B/C metrics cast raw FP8 storage values to BF16 but do not apply "
            "the separate block scales; use B-vs-C for the meaningful dequantized-weight comparison."
        )

    comparison_config = _ComparisonConfig(
        chunk_elements=args.chunk_elements,
        atol=args.atol,
        rtol=args.rtol,
        top_k=args.top_k,
    )
    _compare_parameters("B: FP8 dequantized", model_b, "C: BF16 checkpoint", model_c, comparison_config)
    _compare_parameters("A: FP8 checkpoint + dtype=BF16", model_a, "B: FP8 dequantized", model_b, comparison_config)
    _compare_parameters("A: FP8 checkpoint + dtype=BF16", model_a, "C: BF16 checkpoint", model_c, comparison_config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
