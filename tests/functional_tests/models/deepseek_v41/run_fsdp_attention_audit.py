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

"""Audit real FSDP attention parameters after unshard and before each module runs."""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from run_reference_parity import _CheckpointReader, _load_native, _Options, _reference_dequantize, _tensor_metrics
from safetensors.torch import load_file, save_file
from torch import nn


def main() -> None:
    """Capture actual compute dtypes, exact loaded weights, and attention stages."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)
    logging.basicConfig(level=logging.INFO)
    source = load_file(str(args.artifact), device="cpu")
    options = _Options(
        args.checkpoint,
        args.checkpoint / "inference",
        args.output,
        source["input_ids"].shape[1],
        1,
        32,
        None,
        42,
        "tilelang",
    )
    torch.set_float32_matmul_precision("highest")
    model, load_audit = _load_native(options, device)
    reader = _CheckpointReader(args.checkpoint)
    parameters, captures, handles = {}, {}, []
    layer = model.model.layers["0"]
    before_forward = {}
    for name, parameter in layer.named_parameters():
        if "_hc." not in name and name != "attn.sinks_param.weight":
            continue
        original = name.replace("attn.sinks_param.weight", "attn.attn_sink")
        original = original.replace("attn_hc.", "hc_attn_").replace("ffn_hc.", "hc_ffn_")
        expected = reader.tensor(f"layers.0.{original}").to(device)
        full = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
        local = parameter.to_local() if hasattr(parameter, "to_local") else parameter
        before_forward[name] = {
            "dtype": str(parameter.dtype),
            "local_dtype": str(local.dtype),
            "equal": torch.equal(expected, full),
            "metrics": asdict(_tensor_metrics(expected, full)),
        }

    def before(name: str):
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            """Inspect unsharded direct parameters and tensor inputs without mutation."""
            for parameter_name, parameter in module.named_parameters(recurse=False):
                key = f"{name}.{parameter_name}"
                original = key.replace("attn.sinks_param.weight", "attn.attn_sink")
                original = original.replace("attn_hc.", "hc_attn_").replace("ffn_hc.", "hc_ffn_")
                original = f"layers.0.{original}"
                weight = reader.tensor(original)
                scale_key = original.removesuffix("weight") + "scale"
                scale = (
                    reader.tensor(scale_key) if original.endswith("weight") and scale_key in reader.weight_map else None
                )
                expected = (
                    weight.to(parameter.device)
                    if scale is None
                    else _reference_dequantize(weight, scale, destination=parameter)
                )
                parameters[key] = {
                    "actual_dtype": str(parameter.dtype),
                    "expected_source_dtype": str(weight.dtype),
                    "shape": list(parameter.shape),
                    "equal": torch.equal(expected, parameter),
                    "metrics": asdict(_tensor_metrics(expected, parameter)),
                }
            if inputs and isinstance(inputs[0], torch.Tensor):
                captures[f"{name}.input"] = inputs[0].detach().cpu()

        return capture

    def after(name: str):
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            """Observe tensor outputs or dataclass attention/mHC values."""
            if isinstance(output, torch.Tensor):
                captures[f"{name}.output"] = output.detach().cpu()
            elif hasattr(output, "hidden_states"):
                captures[f"{name}.output"] = output.hidden_states.detach().cpu()
            elif hasattr(output, "pre"):
                for field in ("pre", "post", "comb"):
                    captures[f"{name}.{field}"] = getattr(output, field).detach().cpu()

        return capture

    for name, module in layer.named_modules():
        if name and (name.startswith(("attn", "ffn_hc")) or name == "ffn_norm"):
            handles.append(module.register_forward_pre_hook(before(name)))
            handles.append(module.register_forward_hook(after(name)))
    try:
        with torch.inference_mode():
            model(source["input_ids"].to(device))
        for handle in handles:
            handle.remove()
        comparison = {
            name: asdict(_tensor_metrics(source[f"layer.0.{name}"], captures[f"{name}.output"]))
            for name in ("attn_norm", "attn", "ffn_norm")
        }
        report = {
            "state_audit": asdict(load_audit),
            "before_forward": before_forward,
            "parameters": parameters,
            "comparisons": comparison,
            "math_flags": {
                "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            },
            "captured_shapes": {
                name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype)} for name, tensor in captures.items()
            },
        }
        output = args.output.with_name(f"{args.output.stem}.rank{rank}.json")
        output.write_text(json.dumps(report, indent=2) + "\n")
        if rank == 0:
            save_file(
                {name: value.contiguous() for name, value in captures.items()},
                str(args.output.with_suffix(".safetensors")),
            )
        logging.info("Attention comparison: %s", comparison)
        logging.info(
            "Unequal loaded parameters: %s", [name for name, value in parameters.items() if not value["equal"]]
        )
        if not all(value["equal"] for value in parameters.values()) or not all(
            value["equal"] and value["dtype"] == "torch.float32" and value["local_dtype"] == "torch.float32"
            for value in before_forward.values()
        ):
            raise RuntimeError("Original FP32 parameter storage or post-unshard values failed exact checkpoint audit")
    finally:
        reader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
