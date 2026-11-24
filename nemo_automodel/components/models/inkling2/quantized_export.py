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

"""Write a trained Inkling2 checkpoint in the layout of the released checkpoint it was trained from.

Training saves Inkling2 with floating-point experts (for example ``<step>/model/consolidated``). Serving
runtimes that read the released ``hf_quant_config.json`` load NVFP4 or MXFP8 experts only. This module rewrites
such a floating export with the released tensor names, dtypes, shapes and shards, packed NVFP4/MXFP8 experts,
and the released ``config.json``, ``hf_quant_config.json`` and other top-level files::

    python -m nemo_automodel.components.models.inkling2.quantized_export \\
        --released /path/to/Inkling-2-Mini \\
        --trained /path/to/checkpoints/epoch_0_step_99/model/consolidated \\
        --out /path/to/Inkling-2-Mini-finetuned

A tensor whose trained value equals its released value decoded to the trained dtype keeps its released bytes.
For experts this is the only way to stay byte-identical: the released FP32 ``amax`` cannot be recovered from
BF16 values. Changed experts are encoded with ``quantize_nvfp4_experts`` or ``quantize_mxfp8_experts``, and
other changed tensors are cast to their released dtype. ``model.visual.*`` tensors that a text-only export
lacks are copied from the release. The result is for serving; resume training from the floating checkpoint.
"""

import argparse
import json
import logging
import re
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from nemo_automodel.components.models.inkling2.config import Inkling2TextConfig
from nemo_automodel.components.models.inkling2.quantization import quantize_mxfp8_experts, quantize_nvfp4_experts
from nemo_automodel.components.models.inkling2.state_dict_adapter import Inkling2StateDictAdapter

logger = logging.getLogger(__name__)

_INDEX = "model.safetensors.index.json"
_EXPERT = re.compile(r"model\.llm\.layers\.\d+\.mlp\.experts\.w(?:13|2)_weight")
_NVFP4_METADATA = ("_scale", "_amax", "_outer_scale_denominator")
_FLOATING = (torch.float32, torch.bfloat16, torch.float16)
# Released experts are decoded about this many elements at a time to compare them with the trained values.
_DECODE_ELEMENTS = 1 << 26


@dataclass(frozen=True)
class ExportSummary:
    """What ``export_quantized`` wrote.

    Attributes:
        kept_experts: Experts, counted per projection, whose released bytes were kept.
        encoded_experts: Experts, counted per projection, encoded from their trained values.
        kept_tensors: Floating tensors whose released bytes were kept.
        converted_tensors: Floating tensors written from their trained values in the released dtype.
        released_only_tensors: ``model.visual.*`` tensors absent from the trained export, copied from the release.
    """

    kept_experts: int
    encoded_experts: int
    kept_tensors: int
    converted_tensors: int
    released_only_tensors: int


def _bytes(tensor: torch.Tensor) -> torch.Tensor:
    """Raw bytes of a tensor, one row per index of its leading axis.

    Args:
        tensor: Tensor of shape [rows, ...] or a scalar, any dtype.

    Returns:
        uint8 tensor of shape [rows, bytes_per_row] (one row for a scalar). Comparing bytes treats -0.0 and
        0.0 as different values, as the serialized checkpoint does.
    """
    return tensor.reshape(tensor.shape[0] if tensor.ndim else 1, -1).contiguous().view(torch.uint8)


def _weight_map(folder: Path) -> dict[str, str]:
    """Tensor name to shard file name, from a checkpoint directory's SafeTensors index.

    Args:
        folder: Checkpoint directory holding ``model.safetensors.index.json``.

    Returns:
        The index's ``weight_map``.
    """
    index = folder / _INDEX
    if not index.is_file():
        raise FileNotFoundError(f"{index} not found; expected a sharded SafeTensors checkpoint directory")
    return json.loads(index.read_text())["weight_map"]


def _export_experts(
    adapter: Inkling2StateDictAdapter, released: dict[str, torch.Tensor], name: str, trained: torch.Tensor
) -> tuple[dict[str, torch.Tensor], int]:
    """Encode one grouped expert projection, keeping the released bytes of unchanged experts.

    Args:
        adapter: Decodes released experts exactly as checkpoint loading does.
        released: The projection's released tensors: ``name`` (NVFP4 uint8 codes [experts, output, input/2] or
            MXFP8 float8_e4m3fn values [experts, output, input]), ``name + "_scale"`` and, for NVFP4,
            ``name + "_amax"`` [experts] and ``name + "_outer_scale_denominator"`` [1].
        name: Released weight name, ``model.llm.layers.N.mlp.experts.{w13,w2}_weight``.
        trained: Floating tensor of shape [experts, output, input], the released axis order (gate/up rows of
            ``w13`` interleaved).

    Returns:
        Newly allocated tensors with the released names, dtypes and shapes, and the number of encoded experts.
        The inputs are not modified.
    """
    weight = released[name]
    nvfp4 = weight.dtype == torch.uint8
    shape = (*weight.shape[:-1], weight.shape[-1] * (2 if nvfp4 else 1))
    if trained.dtype not in _FLOATING or tuple(trained.shape) != shape:
        raise ValueError(
            f"{name}: expected a floating tensor of shape {list(shape)}, got {trained.dtype} {list(trained.shape)}"
        )
    output = {key: value.clone() for key, value in released.items()}
    step = max(1, _DECODE_ELEMENTS // (shape[1] * shape[2]))
    encoded = 0
    for start in range(0, shape[0], step):
        stop = min(shape[0], start + step)
        chunk = {
            key: value if key.endswith("_outer_scale_denominator") else value[start:stop]
            for key, value in released.items()
        }
        (decoded,) = adapter.from_hf(chunk).values()
        decoded = decoded.transpose(-1, -2).to(trained.dtype)
        changed = (_bytes(decoded) != _bytes(trained[start:stop])).any(dim=1)
        if not changed.any():
            continue
        encoded += int(changed.sum())
        values = trained[start:stop][changed]
        # Float8 storage is written through uint8 views, which every supported PyTorch can index.
        if nvfp4:
            codes, scales, amax = quantize_nvfp4_experts(values, released[name + "_outer_scale_denominator"])
            output[name][start:stop][changed] = codes
            output[name + "_scale"].view(torch.uint8)[start:stop][changed] = scales.view(torch.uint8)
            output[name + "_amax"][start:stop][changed] = amax
        else:
            values, scales = quantize_mxfp8_experts(values)
            output[name].view(torch.uint8)[start:stop][changed] = values.view(torch.uint8)
            output[name + "_scale"][start:stop][changed] = scales
    return output, encoded


def export_quantized(*, released: str | Path, trained: str | Path, out: str | Path) -> ExportSummary:
    """Write ``trained`` in the layout of ``released`` to the new directory ``out``.

    Args:
        released: Local directory of the released checkpoint the run started from, with its sharded
            SafeTensors index, ``config.json`` and ``hf_quant_config.json``.
        trained: Floating Inkling2 export with released tensor names and a SafeTensors index, such as a
            consolidated training save. Experts are [experts, output, input] in F32, BF16 or F16.
        out: Output directory; it must not exist. The export is written next to it and renamed into place
            once complete, so a failed export leaves no ``out``.

    Returns:
        Counts of kept and re-encoded tensors.
    """
    released, trained, out = Path(released), Path(trained), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists")
    released_map, trained_map = _weight_map(released), _weight_map(trained)
    metadata = {
        weight + suffix
        for weight in released_map
        if _EXPERT.fullmatch(weight)
        for suffix in _NVFP4_METADATA
        if weight + suffix in released_map
    }
    expected = set(released_map) - metadata
    unknown = sorted(set(trained_map) - expected)
    if unknown:
        raise ValueError(f"Trained tensors absent from the released checkpoint: {unknown[:8]}")
    missing = sorted(key for key in expected - set(trained_map) if not key.startswith("model.visual."))
    if missing:
        raise ValueError(f"Released tensors missing from the trained export: {missing[:8]}")
    for key in metadata:
        if released_map[key] != released_map[key.rsplit("_weight", 1)[0] + "_weight"]:
            raise ValueError(f"{key} is stored apart from its expert weight")
    adapter = Inkling2StateDictAdapter(Inkling2TextConfig.from_pretrained(released))
    shards = sorted(set(released_map.values()))
    kept_experts = encoded_experts = kept_tensors = converted_tensors = released_only = 0
    partial = out.with_name(f".{out.name}.partial")
    partial.mkdir(parents=True)
    try:
        with ExitStack() as stack:
            sources = {
                shard: stack.enter_context(safe_open(str(trained / shard), framework="pt"))
                for shard in sorted(set(trained_map.values()))
            }
            for shard in shards:
                tensors: dict[str, torch.Tensor] = {}
                with safe_open(str(released / shard), framework="pt") as source:
                    for key in sorted(k for k in expected if released_map[k] == shard):
                        value = source.get_tensor(key)
                        if key not in trained_map:
                            tensors[key] = value
                            released_only += 1
                            continue
                        update = sources[trained_map[key]].get_tensor(key)
                        if _EXPERT.fullmatch(key) and value.dtype in (torch.uint8, torch.float8_e4m3fn):
                            names = [key, *(key + suffix for suffix in _NVFP4_METADATA if key + suffix in released_map)]
                            written, encoded = _export_experts(
                                adapter, {name: source.get_tensor(name) for name in names}, key, update
                            )
                            tensors.update(written)
                            encoded_experts += encoded
                            kept_experts += value.shape[0] - encoded
                        elif update.dtype not in _FLOATING or update.shape != value.shape:
                            raise ValueError(
                                f"{key}: expected a floating tensor of shape {list(value.shape)}, "
                                f"got {update.dtype} {list(update.shape)}"
                            )
                        elif torch.equal(_bytes(value.to(update.dtype)), _bytes(update)):
                            tensors[key] = value
                            kept_tensors += 1
                        else:
                            tensors[key] = update.to(value.dtype)
                            converted_tensors += 1
                    shard_metadata = source.metadata()
                save_file(tensors, str(partial / shard), metadata=shard_metadata)
                logger.info("Wrote %s (%d tensors)", shard, len(tensors))
        # Names, dtypes and shapes match the release, so its index and metadata files describe the export as is.
        for path in sorted(released.iterdir()):
            if path.is_file() and path.name not in shards and not path.name.startswith("."):
                shutil.copyfile(path, partial / path.name)
        partial.rename(out)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise
    return ExportSummary(
        kept_experts=kept_experts,
        encoded_experts=encoded_experts,
        kept_tensors=kept_tensors,
        converted_tensors=converted_tensors,
        released_only_tensors=released_only,
    )


def main(argv: list[str] | None = None) -> int:
    """Command line entry point; see the module docstring.

    Args:
        argv: Arguments without the program name; ``None`` reads ``sys.argv``.

    Returns:
        Process exit status, 0 on success.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--released", required=True, type=Path, help="released checkpoint directory")
    parser.add_argument("--trained", required=True, type=Path, help="floating Inkling2 export, e.g. model/consolidated")
    parser.add_argument("--out", required=True, type=Path, help="new output directory")
    args = parser.parse_args(argv)
    summary = export_quantized(released=args.released, trained=args.trained, out=args.out)
    print(f"wrote {args.out}: {summary}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
