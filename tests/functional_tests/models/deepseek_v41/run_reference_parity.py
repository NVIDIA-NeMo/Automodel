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

"""Compare pretrained native V4.1 against its unmodified released inference code.

The reference is DeepSeek's inference implementation from the pinned Hugging Face
snapshot, not a Transformers implementation. Both sides use dequantized BF16
weights; official KV/indexer activation quantization remains enabled. A single
GPU covers the first layer. A torchrun launch uses official TP/expert sharding
and native EP, with real owner-sharded Engram tables on both sides.

Example:
    python tests/functional_tests/models/deepseek_v41/run_reference_parity.py \
        --checkpoint /path/to/snapshots/df42c109f1defefcbfcedbe7d905718a12266e40 \
        --sequence-length 4096 --output logs/deepseek_v41/layer0_parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.nn import functional as F
from transformers import AutoTokenizer

LOGGER = logging.getLogger(__name__)
REVISION = "df42c109f1defefcbfcedbe7d905718a12266e40"
REFERENCE_HASHES = {
    "model.py": "4e9ae23620edc8028ccc5d5fef552ab7fdc7dcd6f79608754fe9f67644056f65",
    "kernel.py": "1236c3507019ed176f5dba5e04bcea58867cf654818c6cf138ed4845398c2455",
}
# Fixed before the first four-layer result; maximum KL is reported separately.
PARITY_THRESHOLDS = {"mean_kl_max": 1e-3, "p95_kl_max": 2e-3, "cosine_min": 0.999, "top1_min": 0.98}
DEFAULT_TEXT = """A research team compares two implementations of the same language model.
Both implementations receive identical token sequences, checkpoint weights, and positions.
The comparison measures every vocabulary logit at every position, including the beginning
of the sequence and positions beyond the sliding attention window. A causal model must
never use information from a future token. Numerical precision can change rounding, so
the team records distribution divergence as well as the largest individual difference.

模型验证必须使用相同的输入和权重，并逐层比较中间结果。先检查数值一致性，再进行分布式训练。
For a matrix A and vector x, the expression y = A @ x defines a linear transformation.
The gradient follows the chain rule. Repeated experiments should retain their exact settings.
"""


class _TensorSlice(Protocol):
    def get_shape(self) -> list[int]:
        """Return the stored tensor dimensions without materializing data."""
        ...

    def __getitem__(self, indices: slice | tuple[slice, slice]) -> torch.Tensor:
        """Return a CPU tensor sliced along the stored row and column axes.

        Args:
            indices: Row slice, or row/column slices of [rows, columns] storage.

        Returns:
            Tensor with the selected rows and optional selected columns.
        """
        ...


class _TensorFile(Protocol):
    def get_tensor(self, name: str) -> torch.Tensor:
        """Read the checkpoint tensor, preserving its stored logical layout."""
        ...

    def get_slice(self, name: str) -> _TensorSlice:
        """Return a lazy view of a named checkpoint tensor."""
        ...


@dataclass(frozen=True)
class _Options:
    checkpoint: Path
    reference_dir: Path
    output: Path
    sequence_length: int
    num_layers: int
    metric_chunk_size: int
    input_text: Path | None
    seed: int
    attention_backend: str
    expert_backend: str | None = None


@dataclass(frozen=True)
class _StateAudit:
    expected: int
    loaded: int
    missing: tuple[str, ...]
    unexpected: tuple[str, ...]
    shape_mismatches: tuple[str, ...]
    checkpoint_source_tensors: int
    checkpoint_source_bytes: int
    elapsed_seconds: float
    loaded_bytes: int = 0
    max_chunk_source_bytes: int = 0
    max_chunk_output_bytes: int = 0


@dataclass(frozen=True)
class _TensorMetrics:
    shape: tuple[int, ...]
    mean_absolute_difference: float
    max_absolute_difference: float
    relative_rmse: float
    cosine_similarity: float


class _CheckpointReader:
    """Open each required SafeTensors shard once and retain no converted tensors."""

    def __init__(self, checkpoint: Path) -> None:
        self.checkpoint = checkpoint
        self.weight_map: dict[str, str] = json.loads((checkpoint / "model.safetensors.index.json").read_text())[
            "weight_map"
        ]
        self.files: dict[str, _TensorFile] = {}
        self.stack = ExitStack()
        self.read_names: set[str] = set()
        self.read_bytes = 0

    def tensor(self, name: str) -> torch.Tensor:
        """Read one tensor from its source shard.

        Args:
            name: Exact checkpoint tensor key.

        Returns:
            CPU tensor with the checkpoint's shape and storage dtype. Packed FP4
            tensors retain [output, input / 2] byte storage until dequantization.
        """
        tensor = self._file(name).get_tensor(name)
        self.read_bytes += tensor.numel() * tensor.element_size()
        self.read_names.add(name)
        return tensor

    def _file(self, name: str) -> _TensorFile:
        filename = self.weight_map[name]
        if filename not in self.files:
            self.files[filename] = self.stack.enter_context(
                safe_open(str(self.checkpoint / filename), framework="pt", device="cpu")
            )
        return self.files[filename]

    def shape(self, name: str) -> tuple[int, ...]:
        """Read tensor dimensions without mapping or copying its complete data."""
        return tuple(self._file(name).get_slice(name).get_shape())

    def slice(self, name: str, *, rows: slice, columns: slice | None = None) -> torch.Tensor:
        """Read only the requested checkpoint rows and columns.

        Args:
            name: Exact checkpoint key.
            rows: Slice along axis zero.
            columns: Optional slice along axis one of [rows, columns] storage.

        Returns:
            CPU tensor in its stored dtype, with only the requested local shape.
        """
        view = self._file(name).get_slice(name)
        tensor = view[rows] if columns is None else view[rows, columns]
        self.read_bytes += tensor.numel() * tensor.element_size()
        self.read_names.add(name)
        return tensor

    def close(self) -> None:
        """Release mapped shard handles after all destination copies complete."""
        self.stack.close()


def _import_reference(reference_dir: Path) -> ModuleType:
    """Import the pinned source without replacing any reference operation."""
    for filename, expected in REFERENCE_HASHES.items():
        actual = hashlib.sha256((reference_dir / filename).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Reference {filename} differs from revision {REVISION}: {actual}")
    sys.path.insert(0, str(reference_dir))
    spec = importlib.util.spec_from_file_location("deepseek_v41_official_reference", reference_dir / "model.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import official model from {reference_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _native_source_manifest() -> dict[str, str]:
    """Hash model code and shared numerical/loading paths before native imports."""
    root = Path(__file__).resolve().parents[4]
    paths = set((root / "nemo_automodel/components/models/deepseek_v41").glob("*.py"))
    paths.update((root / "nemo_automodel/components/models/deepseek_v4").rglob("*.py"))
    paths.update((root / "nemo_automodel/components/models/qwen3_8_flash_next").rglob("*.py"))
    paths.update((root / "nemo_automodel/components/moe").rglob("*.py"))
    for relative in (
        "nemo_automodel/components/models/common/utils.py",
        "nemo_automodel/components/distributed/parallelizer.py",
        "nemo_automodel/components/distributed/parallelizer_utils.py",
        "nemo_automodel/_transformers/model_init.py",
        "nemo_automodel/shared/utils.py",
    ):
        paths.add(root / relative)
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}


class _OfficialPrefix(nn.Module):
    """Official submodules and the literal text-only Transformer.forward sequence.

    Only the prefix and shared text head are instantiated. The vision-enabled
    ModelArgs remain intact, so every official router retains its VL bias without
    allocating an unused vision tower. No reference math or kernels are replaced.
    """

    def __init__(self, reference: ModuleType, options: _Options, device: torch.device) -> None:
        super().__init__()
        values = json.loads((options.reference_dir / "config.json").read_text())
        values.update(
            n_layers=options.num_layers,
            n_mtp_layers=0,
            dspark_block_size=0,
            dtype="bf16",
            expert_dtype=None,
            max_batch_size=1,
            max_seq_len=options.sequence_length,
        )
        args = reference.ModelArgs(**values)
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        active_engram = any(index < options.num_layers for index in args.engram_layer_ids)
        if active_engram and world_size < 4:
            raise ValueError("A released Engram prefix requires at least four owner ranks on the development GPUs")
        # These are exactly the runtime globals set by official Transformer.__init__.
        reference.world_size = world_size
        reference.rank = rank
        reference.default_dtype = torch.bfloat16
        self.streams = args.hc_mult
        self.reference = reference
        tokenizer = AutoTokenizer.from_pretrained(options.checkpoint, local_files_only=True) if active_engram else None
        layout = reference.EngramLayout.from_args(args) if active_engram else None
        with torch.device(device), reference.set_dtype(torch.bfloat16):
            self.engram_hash = reference.NgramHashState(args, layout, tokenizer) if layout is not None else None
            self.embed = reference.ParallelEmbedding(args.vocab_size, args.dim)
            self.layers = nn.ModuleDict(
                {str(index): reference.Block(index, args, layout) for index in range(args.n_layers)}
            )
            self.norm = reference.RMSNorm(args.dim, args.norm_eps)
            self.head = reference.ParallelHead(args.vocab_size, args.dim, args.norm_eps, args.hc_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Execute official text blocks and request full-position logits.

        Args:
            input_ids: Integer tensor of shape [1, sequence], with no padding.

        Returns:
            FP32 logits of shape [1, sequence, vocab] for every vocabulary entry.
        """
        with torch.device(input_ids.device):
            hashes = self.engram_hash(input_ids, 0) if self.engram_hash is not None else None
            hidden = self.embed(input_ids)
            hidden = hidden.unsqueeze(2).repeat(1, 1, self.streams, 1)
            pre_mix = self.reference.make_identity_pre_mix(hidden, self.streams)
            for layer in self.layers.values():
                if layer.engram is not None:
                    hidden = layer.engram(hidden, hashes[:, :, layer.engram.layer_hash_index, :], None)
                hidden, pre_mix = layer(hidden, 0, pre_mix, None)
            hidden = layer.hc_pre(hidden, pre_mix)
            return self.head(self.norm(hidden), full_logits=True)


def _reference_dequantize(
    weight: torch.Tensor, scale: torch.Tensor | None, *, destination: torch.Tensor
) -> torch.Tensor:
    """Independently decode the published weight formats for official BF16 GEMMs.

    Args:
        weight: CPU tensor of shape [output, input] for FP8/dense weights, or
            [output, input / 2] INT8 storage for packed low-nibble-first E2M1.
            Non-matrix dense parameters preserve their original shape.
        scale: Optional E8M0 scale tensor [ceil(output / 32), ceil(input / 32)]
            for FP8, or [output, input / 32] for packed FP4 experts.
        destination: Final parameter whose shape, device, and dtype define output.

    Returns:
        Tensor with destination's logical shape, device, and dtype. A temporary
        conversion contains at most one projection matrix, never the whole model.
    """
    weight = weight.to(destination.device)
    if scale is None:
        return weight.to(destination.dtype)
    scale = scale.to(device=destination.device, dtype=torch.float32)
    if weight.dtype == torch.int8:
        # Same value table and low/high nibble ordering as official convert.py.
        table = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            dtype=torch.float32,
            device=destination.device,
        )
        packed = weight.view(torch.uint8)
        values = torch.stack((table[(packed & 15).long()], table[(packed >> 4).long()]), dim=-1).flatten(-2)
        values = values.unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)
        return values.flatten(-2).to(destination.dtype)
    if weight.dtype != torch.float8_e4m3fn:
        raise ValueError(f"Unsupported scaled official checkpoint dtype {weight.dtype}")
    # Blockwise multiplication is independent of the native adapter's converter.
    expanded = scale.repeat_interleave(32, 0).repeat_interleave(32, 1)
    return (weight.float() * expanded[: weight.shape[0], : weight.shape[1]]).to(destination.dtype)


@torch.no_grad()
def _load_official(model: _OfficialPrefix, checkpoint: Path) -> _StateAudit:
    """Load every official prefix parameter, including unused text/VL router biases.

    Args:
        model: Official prefix with standard named parameters: dense projections
            [output, input], expert projections [output, input], and norms [hidden].
        checkpoint: Directory containing the pinned SafeTensors shards.

    Returns:
        Exact expected/loaded/source counts; missing or mismatched tensors raise.
    """
    reader = _CheckpointReader(checkpoint)
    started = time.monotonic()
    expected = dict(model.named_parameters())
    missing = set(expected) - set(reader.weight_map)
    if missing:
        raise ValueError(f"Official checkpoint missing expected parameters: {sorted(missing)}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    try:
        for index, (name, destination) in enumerate(expected.items()):
            if ".engram.embed." in name:
                # Official Engram keeps raw FP8 weights plus E8M0 scales. Stream
                # directly into its real row owner, never dequantize the table.
                source_rows = reader.shape(name)[0]
                start = rank * destination.shape[0]
                valid_rows = max(0, min(destination.shape[0], source_rows - start))
                for offset in range(0, valid_rows, 262144):
                    end = min(valid_rows, offset + 262144)
                    chunk = reader.slice(name, rows=slice(start + offset, start + end))
                    destination[offset:end].copy_(chunk)
                    if offset % (262144 * 64) == 0:
                        LOGGER.info("Official %s rows %d/%d", name, end, valid_rows)
                if valid_rows < destination.shape[0]:
                    destination[valid_rows:].fill_(1 if name.endswith(".scale") else 0)
                continue
            axis = None
            if name in ("embed.weight", "head.weight") or name.endswith(
                (".wq_b.weight", ".wo_a.weight", ".weights_proj.weight", ".attn_sink")
            ):
                axis = 0
            elif name.endswith(".wo_b.weight"):
                axis = 1
            if world_size > 1 and axis is not None:
                start, end = rank * destination.shape[axis], (rank + 1) * destination.shape[axis]
                weight = reader.slice(
                    name,
                    rows=slice(start, end) if axis == 0 else slice(None),
                    columns=slice(start, end) if axis == 1 else None,
                )
            else:
                weight = reader.tensor(name)
            scale_name = name.removesuffix("weight") + "scale"
            scale = None
            if name.endswith(".weight") and scale_name in reader.weight_map:
                if world_size > 1 and axis is not None:
                    if start % 32 or end % 32:
                        raise ValueError(
                            f"Official TP slice is not aligned with 32-element quantization blocks: {name}"
                        )
                    scale = reader.slice(
                        scale_name,
                        rows=slice(start // 32, end // 32) if axis == 0 else slice(None),
                        columns=slice(start // 32, end // 32) if axis == 1 else None,
                    )
                else:
                    scale = reader.tensor(scale_name)
            converted = _reference_dequantize(weight, scale, destination=destination)
            if converted.shape != destination.shape:
                raise ValueError(f"Official shape mismatch for {name}: {converted.shape} != {destination.shape}")
            destination.copy_(converted)
            if index % 200 == 0:
                LOGGER.info("Official load %d/%d: %s", index + 1, len(expected), name)
        torch.cuda.synchronize()
        return _StateAudit(
            len(expected),
            len(expected),
            (),
            (),
            (),
            len(reader.read_names),
            reader.read_bytes,
            time.monotonic() - started,
        )
    finally:
        reader.close()


def _load_native(options: _Options, device: torch.device) -> tuple[nn.Module, _StateAudit]:
    """Load the same checkpoint scope through the model-owned native adapter."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter

    config = DeepseekV41Config.from_pretrained(options.checkpoint)
    config.text_config.num_hidden_layers = options.num_layers
    # Text-prefix parity preserves both router biases but never executes the
    # vision tower. Its separate multimodal checkpoint gate loads those weights.
    config.vision_config.num_hidden_layers = 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    if world_size < 4 and any(index < options.num_layers for index in config.text_config.engram_layer_ids):
        raise ValueError("The released native Engram prefix requires at least four owner ranks on the development GPUs")
    if world_size > 1:
        from nemo_automodel import NeMoAutoModelForCausalLM
        from nemo_automodel.components.distributed import DistributedSetup, FSDP2Config, ParallelismSizes
        from nemo_automodel.components.distributed.config import MoEParallelizerConfig

        backend = BackendConfig(
            attn=options.attention_backend,
            linear="torch",
            rms_norm="torch_fp32",
            experts=options.expert_backend or "torch_mm",
            dispatcher="hybridep",
            gate_precision="float32",
        )
        setup = DistributedSetup.build(
            strategy=FSDP2Config(),
            parallelism_sizes=ParallelismSizes(ep_size=world_size),
            moe_parallel_config=MoEParallelizerConfig(
                reshard_after_forward=True,
                lm_head_precision="float32",
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    output_dtype=None,
                    cast_forward_inputs=False,
                ),
            ),
            activation_checkpointing=False,
            world_size=world_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(options.checkpoint, local_files_only=True)
        model = NeMoAutoModelForCausalLM.from_config(
            config,
            backend=backend,
            tokenizer=tokenizer,
            distributed_setup=setup,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            torch_dtype=torch.bfloat16,
        )
        started = time.monotonic()
        loaded = model.state_dict_adapter.load_from_checkpoint(
            model, options.checkpoint, device_mesh=setup.mesh_context.moe_mesh["ep"]
        )
        torch.cuda.synchronize()
        expected_count = len(model.state_dict())
        audit = _StateAudit(
            expected_count,
            expected_count,
            (),
            (),
            (),
            len(loaded.loaded_keys),
            loaded.source_bytes,
            time.monotonic() - started,
            loaded.loaded_bytes,
            loaded.max_chunk_source_bytes,
            loaded.max_chunk_output_bytes,
        )
        return model.eval().requires_grad_(False), audit
    backend = BackendConfig(
        attn=options.attention_backend,
        linear="torch",
        rms_norm="torch_fp32",
        experts=options.expert_backend or "torch",
        dispatcher="torch",
    )
    with torch.device(device):
        model = DeepseekV41ForCausalLM(config, backend=backend).eval().requires_grad_(False)
    reader = _CheckpointReader(options.checkpoint)
    prefixes = tuple(f"layers.{index}." for index in range(options.num_layers))
    names = [
        name
        for name in reader.weight_map
        if name.startswith(prefixes) or name in ("embed.weight", "head.weight", "norm.weight")
    ]
    started = time.monotonic()
    try:
        state = {name: reader.tensor(name) for name in names}
        adapter = DeepseekV41StateDictAdapter(config, model.model.moe_config, backend, dtype=torch.bfloat16)
        converted = adapter.from_hf(state)
        expected = model.state_dict()
        missing = tuple(sorted(set(expected) - set(converted)))
        unexpected = tuple(sorted(set(converted) - set(expected)))
        mismatched = tuple(
            name for name in expected.keys() & converted.keys() if expected[name].shape != converted[name].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f"Native state audit failed: missing={missing}, unexpected={unexpected}, shapes={mismatched}"
            )
        model.load_state_dict(converted, strict=True)
        torch.cuda.synchronize()
        audit = _StateAudit(
            len(expected),
            len(converted),
            missing,
            unexpected,
            mismatched,
            len(reader.read_names),
            reader.read_bytes,
            time.monotonic() - started,
        )
    finally:
        reader.close()
    return model, audit


def _capture_outputs(
    model: nn.Module, *, native: bool
) -> tuple[dict[str, torch.Tensor], list[torch.utils.hooks.RemovableHandle]]:
    """Attach observer hooks at corresponding boundaries without changing outputs.

    Args:
        model: Prefix module whose activations have shapes [batch, sequence, hidden]
            or [batch, sequence, streams, hidden] at block boundaries.
        native: Whether model uses the AutoModel module hierarchy.

    Returns:
        CPU activation mapping with each tensor's original layout and hook handles.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook_for(name: str) -> Callable:
        def capture(module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor | tuple) -> None:
            """Copy an observed activation without changing its producer.

            Args:
                module: Producing module.
                inputs: Positional tensor inputs with their producer's documented layouts.
                output: Tensor [batch, sequence, hidden] or block result whose first
                    tensor is [batch, sequence, streams, hidden]. Native attention
                    instead returns a dataclass with hidden_states [batch, sequence, hidden].
            """
            if hasattr(output, "hidden_states"):
                tensor = output.hidden_states
            else:
                tensor = output[0] if isinstance(output, tuple) else output
            captured[name] = tensor.detach().to("cpu", copy=True)

        return capture

    backbone = model.model if native else model
    handles.append(backbone.norm.register_forward_hook(hook_for("final_norm")))
    if backbone.engram_hash is not None:
        handles.append(backbone.engram_hash.register_forward_hook(hook_for("engram_hash")))
    for index, layer in backbone.layers.items():
        for name in ("attn_norm", "attn", "ffn_norm", "ffn"):
            handles.append(getattr(layer, name).register_forward_hook(hook_for(f"layer.{index}.{name}")))
        if layer.engram is not None:
            handles.append(layer.engram.register_forward_hook(hook_for(f"layer.{index}.engram")))
        handles.append(layer.register_forward_hook(hook_for(f"layer.{index}.output")))
    return captured, handles


def _tensor_metrics(reference: torch.Tensor, native: torch.Tensor) -> _TensorMetrics:
    """Measure paired activations in bounded flattened chunks.

    Args:
        reference: Tensor of arbitrary shape, with the reference activation layout.
        native: Tensor with the identical semantic shape and axis order.

    Returns:
        Difference, global cosine, and reference-normalized RMSE statistics.
    """
    if reference.shape != native.shape:
        raise ValueError(f"Activation shapes differ: {reference.shape} != {native.shape}")
    sums = torch.zeros(5, dtype=torch.float64, device="cpu")
    maximum = 0.0
    for start in range(0, reference.numel(), 1 << 20):
        ref = reference.reshape(-1)[start : start + (1 << 20)].float()
        actual = native.reshape(-1)[start : start + (1 << 20)].to(ref.device).float()
        difference = actual - ref
        chunk = torch.stack(
            (
                difference.abs().sum(),
                difference.square().sum(),
                ref.square().sum(),
                actual.square().sum(),
                (ref * actual).sum(),
            )
        )
        sums += chunk.cpu().double()
        maximum = max(maximum, difference.abs().max().item())
    cosine = (sums[4] / (sums[2] * sums[3]).sqrt().clamp_min(1e-30)).clamp(-1, 1)
    return _TensorMetrics(
        tuple(reference.shape),
        (sums[0] / reference.numel()).item(),
        maximum,
        (sums[1] / sums[2].clamp_min(1e-30)).sqrt().item(),
        cosine.item(),
    )


def _logit_metrics(
    reference: torch.Tensor, native: torch.Tensor, *, chunk_size: int
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Compare every logit and probability distribution without a full FP64 copy.

    Args:
        reference: FP32 logits of shape [1, sequence, vocab].
        native: FP32 logits of shape [1, sequence, vocab], with identical tokens.
        chunk_size: Number of complete-vocabulary distributions per chunk.

    Returns:
        Scalar metrics and CPU vectors kl/jsd/top1 [sequence], preserving every
        token position. KL direction is reference to native.
    """
    if reference.shape != native.shape or reference.shape[0] != 1:
        raise ValueError("Logits must have equal [1, sequence, vocab] shapes")
    kl_values, jsd_values, agreements = [], [], []
    for start in range(0, reference.shape[1], chunk_size):
        ref = reference[0, start : start + chunk_size].to(native.device).float()
        actual = native[0, start : start + chunk_size].float()
        if not torch.isfinite(ref).all() or not torch.isfinite(actual).all():
            raise ValueError(f"Nonfinite logits beginning at position {start}")
        log_reference = F.log_softmax(ref, dim=-1)
        log_native = F.log_softmax(actual, dim=-1)
        reference_probability = log_reference.exp()
        native_probability = log_native.exp()
        log_middle = torch.logaddexp(log_reference, log_native) - math.log(2)
        kl_values.append((reference_probability * (log_reference - log_native)).sum(-1).cpu())
        jsd_values.append(
            (
                0.5
                * (
                    reference_probability * (log_reference - log_middle)
                    + native_probability * (log_native - log_middle)
                ).sum(-1)
            ).cpu()
        )
        agreements.append((ref.argmax(-1) == actual.argmax(-1)).cpu())
    per_position = {"kl": torch.cat(kl_values), "jsd": torch.cat(jsd_values), "top1_agreement": torch.cat(agreements)}
    common = _tensor_metrics(reference, native)
    return {
        "mean_kl_reference_to_native": per_position["kl"].mean().item(),
        "p95_kl_reference_to_native": per_position["kl"].quantile(0.95).item(),
        "max_kl_reference_to_native": per_position["kl"].max().item(),
        "mean_jsd": per_position["jsd"].mean().item(),
        "logits_cosine_similarity": common.cosine_similarity,
        "top1_agreement_fraction": per_position["top1_agreement"].float().mean().item(),
        "mean_absolute_logit_difference": common.mean_absolute_difference,
        "max_absolute_logit_difference": common.max_absolute_difference,
        "relative_logit_rmse": common.relative_rmse,
    }, per_position


def _evaluate_gates(metrics: dict[str, float], audits: dict[str, dict]) -> dict:
    """Evaluate predetermined numerical gates and exact checkpoint coverage."""
    checks = {
        "mean_kl": metrics["mean_kl_reference_to_native"] <= PARITY_THRESHOLDS["mean_kl_max"],
        "p95_kl": metrics["p95_kl_reference_to_native"] <= PARITY_THRESHOLDS["p95_kl_max"],
        "cosine": metrics["logits_cosine_similarity"] >= PARITY_THRESHOLDS["cosine_min"],
        "top1": metrics["top1_agreement_fraction"] >= PARITY_THRESHOLDS["top1_min"],
        "all_positions_finite": all(math.isfinite(value) for value in metrics.values()),
        "state_coverage": all(
            audit["expected"] == audit["loaded"]
            and not any(audit[key] for key in ("missing", "unexpected", "shape_mismatches"))
            for audit in audits.values()
        ),
    }
    return {"passed": all(checks.values()), "checks": checks, "thresholds": PARITY_THRESHOLDS, "max_kl": "report_only"}


def _tokenize(options: _Options) -> torch.Tensor:
    """Create deterministic, unpadded text tokens of shape [1, sequence] on CPU."""
    tokenizer = AutoTokenizer.from_pretrained(options.checkpoint, local_files_only=True)
    text = options.input_text.read_text() if options.input_text else DEFAULT_TEXT
    if not text.strip():
        raise ValueError("The parity text must not be empty")
    token_count = len(tokenizer.encode(text, add_special_tokens=False))
    repeated = (text + "\n\n") * (math.ceil(options.sequence_length / max(token_count, 1)) + 1)
    input_ids = tokenizer(
        repeated, add_special_tokens=True, truncation=True, max_length=options.sequence_length, return_tensors="pt"
    )["input_ids"]
    if input_ids.shape != (1, options.sequence_length):
        raise ValueError(f"Tokenization did not produce the requested sequence: {input_ids.shape}")
    return input_ids


def main() -> int:
    """Run the pretrained prefix diagnostic and write all numerical evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--metric-chunk-size", type=int, default=32)
    parser.add_argument("--input-text", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-backend", choices=("eager", "sdpa", "tilelang"), default="sdpa")
    parser.add_argument("--expert-backend", choices=("torch", "torch_mm", "torch_linear"))
    parsed = parser.parse_args()
    options = _Options(**{**vars(parsed), "reference_dir": parsed.reference_dir or parsed.checkpoint / "inference"})
    if options.sequence_length <= 0 or options.metric_chunk_size <= 0 or options.num_layers <= 0:
        raise ValueError("Sequence length, metric chunk size, and layer count must be positive")
    if not torch.cuda.is_available():
        raise ValueError("Official inference parity requires CUDA GPUs")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size == 1 and torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one GPU, or use torchrun for the distributed prefix gate")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if world_size > 1:
        dist.init_process_group("nccl", device_id=device)
    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO, format=f"%(asctime)s rank={rank} %(levelname)s %(message)s", stream=sys.stdout
    )
    options.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(options.seed)
    torch.set_float32_matmul_precision("highest")
    reference_module = _import_reference(options.reference_dir)
    LOGGER.info("Unmodified official reference imported; loading pretrained prefix")
    official = _OfficialPrefix(reference_module, options, device).eval().requires_grad_(False)
    official_audit = _load_official(official, options.checkpoint)
    LOGGER.info("Official state audit: %s", asdict(official_audit))
    native_source_hashes = _native_source_manifest()
    native, native_audit = _load_native(options, device)
    LOGGER.info("Native state audit: %s", asdict(native_audit))
    input_ids = _tokenize(options)
    input_file = options.output.with_suffix(".inputs.safetensors")
    if rank == 0:
        save_file({"input_ids": input_ids}, str(input_file))
    input_digest = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()
    input_ids = input_ids.to(device)
    reference_activations, reference_hooks = _capture_outputs(official, native=False)
    native_activations, native_hooks = _capture_outputs(native, native=True)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        started = time.monotonic()
        LOGGER.info("Official forward begins, sequence=%d, layers=%d", options.sequence_length, options.num_layers)
        reference_logits = official(input_ids)
        torch.cuda.synchronize()
        reference_seconds = time.monotonic() - started
        LOGGER.info("Official forward completed in %.2fs; native forward begins", reference_seconds)
        started = time.monotonic()
        native_logits = native(input_ids).logits
        torch.cuda.synchronize()
        native_seconds = time.monotonic() - started
        LOGGER.info("Native forward completed in %.2fs; comparing every position and vocabulary entry", native_seconds)
        metrics, per_position = _logit_metrics(reference_logits, native_logits, chunk_size=options.metric_chunk_size)
    for handle in reference_hooks + native_hooks:
        handle.remove()
    if reference_activations.keys() != native_activations.keys():
        raise ValueError("Reference and native activation observers covered different boundaries")
    if "engram_hash" in reference_activations and not torch.equal(
        reference_activations["engram_hash"], native_activations["engram_hash"]
    ):
        raise ValueError("Native Engram hashes differ from the official tokenizer/hash implementation")
    activation_metrics = {
        name: asdict(_tensor_metrics(values, native_activations[name]))
        for name, values in reference_activations.items()
    }
    rank_output = (
        options.output if world_size == 1 else options.output.with_name(f"{options.output.stem}.rank{rank}.json")
    )
    save_file(per_position, str(rank_output.with_suffix(".positions.safetensors")))
    report = {
        "reference_kind": "unmodified official DeepSeek inference code from Hugging Face snapshot",
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "reference_source_files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(options.reference_dir.glob("*.py"))
        },
        "checkpoint": str(options.checkpoint),
        "rank": rank,
        "automodel_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "automodel_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        "native_source_hashes_before_import": native_source_hashes,
        "configuration": {
            "layers": options.num_layers,
            "sequence_length": options.sequence_length,
            "vocabulary_size": reference_logits.shape[-1],
            "all_positions_and_logits": True,
            "weight_compute_dtype": "bfloat16",
            "head_compute_dtype": "float32",
            "official_dtype_argument": "bf16",
            "official_expert_dtype_argument": None,
            "official_activation_quantization": {
                "window_kv": "FP8, block32 E8M0 scales",
                "compressed_kv": "FP4, block16 E4M3 scales",
                "indexer_qk": "FP4, block32 E8M0 scales",
            },
            "native_attention_backend": options.attention_backend,
            "native_expert_backend": options.expert_backend or ("torch_mm" if world_size > 1 else "torch"),
            "text_only": True,
            "active_engram_layers": [
                int(index) for index, layer in official.layers.items() if layer.engram is not None
            ],
            "reference_parallelism": {"tp": world_size, "ep": world_size, "engram_owners": world_size},
            "native_parallelism": {"tp": 1, "ep": world_size, "engram_owners": world_size, "fsdp": world_size},
            "native_dispatcher": "hybridep" if world_size > 1 else "torch",
            "engram_hashes_exact": "engram_hash" in reference_activations,
            "seed": options.seed,
        },
        "input_tokens": {"path": str(input_file), "sha256": input_digest},
        "state_audit": {"official": asdict(official_audit), "native": asdict(native_audit)},
        "logits": metrics,
        "activations": activation_metrics,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
            "reference_forward_seconds": reference_seconds,
            "native_forward_seconds": native_seconds,
        },
        "scope": "Pretrained prefix forward diagnostic with the complete vocabulary and real checkpoint tables. Later layers, training gradients, and checkpoint resume require their separate gates.",
    }
    report["gates"] = _evaluate_gates(metrics, report["state_audit"])
    rank_output.write_text(json.dumps(report, indent=2) + "\n")
    LOGGER.info("Full-vocabulary metrics: %s", json.dumps(metrics, sort_keys=True))
    for name, values in activation_metrics.items():
        LOGGER.info("Activation %s: %s", name, json.dumps(values, sort_keys=True))
    LOGGER.info("Evidence written to %s", rank_output)
    if world_size > 1:
        reports = [None] * world_size if rank == 0 else None
        dist.gather_object(report, reports, dst=0)
        if rank == 0:
            if len({item["input_tokens"]["sha256"] for item in reports}) != 1:
                raise ValueError("Distributed parity ranks used different token sequences")
            report["all_rank_logits"] = {str(item["rank"]): item["logits"] for item in reports}
            report["all_rank_state_audits"] = {str(item["rank"]): item["state_audit"] for item in reports}
            report["all_rank_gates"] = {str(item["rank"]): item["gates"] for item in reports}
            options.output.write_text(json.dumps(report, indent=2) + "\n")
        dist.barrier()
        dist.destroy_process_group()
    return 0 if report["gates"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
