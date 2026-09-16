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

"""LoRA merged-weight QAT composition and packed reference checkpoints.

Packed checkpoints decode into ordinary floating-point models. They are not a
native FP4/FP8 GPU serving format. Prepare after PEFT injection, before sharding
and optimizer construction; export currently requires a complete local model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fnmatch import fnmatchcase
from itertools import chain
from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.distributed.tensor import DTensor

from nemo_automodel.components._peft.lora import LinearLoRA
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import QuantizedWeight, WeightFakeQuantizer
from nemo_automodel.shared.import_utils import safe_import

_HAS_GROUPED_GEMM, _ = safe_import("grouped_gemm.ops")
_LORA_TYPES = (LinearLoRA, GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA)


@dataclass(frozen=True)
class _Projection:
    """Borrowed projection storage; no tensors are copied or materialized.

    Args:
        name: Direct base parameter name.
        base: Frozen [out, in] dense or [experts, in, out] grouped weight.
        a: Dense [rank, in] or grouped [experts, in, rank] adapter.
        b: Dense [out, rank] or grouped [experts, rank, out] adapter.
        layout: ``out_in`` for dense; ``experts_in_out`` for grouped storage.
            All tensors must be local strided floats of matching dtype/device.
            Meta tensors are allowed during preparation only.
    """

    name: str
    base: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    layout: Literal["out_in", "experts_in_out"]

    @property
    def shape(self) -> tuple[int, ...]:
        """Canonical [out, in] or [experts, out, in] dimensions, without a tensor allocation."""
        shape = tuple(self.base.shape)
        return shape if self.layout == "out_in" else (*shape[:-2], shape[-1], shape[-2])


def _projections(module: LinearLoRA | GroupedExpertsLoRA | GroupedExpertsDeepEPLoRA) -> tuple[_Projection, ...]:
    """Borrow base and adapter tensors with the layouts documented on _Projection."""
    if isinstance(module, LinearLoRA):
        return (_Projection("weight", module.weight, module.lora_A.weight, module.lora_B.weight, "out_in"),)
    return (
        _Projection(
            "gate_and_up_projs",
            module.gate_and_up_projs,
            module.lora_gate_and_up_A,
            module.lora_gate_and_up_B,
            "experts_in_out",
        ),
        _Projection("down_projs", module.down_projs, module.lora_down_A, module.lora_down_B, "experts_in_out"),
    )


def _reject_distributed(model: nn.Module) -> None:
    """Reject already-sharded parameters and buffers without inspecting their contents."""
    for name, tensor in chain(model.named_parameters(), model.named_buffers()):
        if isinstance(tensor, DTensor):
            raise TypeError(f"{name}: LoRA QAT preparation/export/load requires local tensors, not DTensor")


def _reject_tied_lora_bases(model: nn.Module, names: list[str]) -> None:
    """Reject base parameter aliases, including multiple paths to the same module.

    An independent merged LoRA delta cannot be loaded into a fresh tied
    architecture without changing the other aliases. Ordinary tied parameters
    that are not LoRA projection bases remain supported.
    """
    aliases: dict[int, list[str]] = {}
    for path, parameter in model.named_parameters(remove_duplicate=False):
        aliases.setdefault(id(parameter), []).append(path)
    for name in names:
        for proj in _projections(model.get_submodule(name)):
            paths = aliases[id(proj.base)]
            if len(paths) > 1:
                raise ValueError(f"{name}: tied base parameter is unsupported for LoRA QAT: {paths}")


def _validate_lora(
    name: str,
    module: LinearLoRA | GroupedExpertsLoRA | GroupedExpertsDeepEPLoRA,
    quantizer: WeightFakeQuantizer | None,
) -> None:
    """Validate LoRA storage metadata only, including meta-device initialization."""
    if isinstance(module, LinearLoRA):
        if module.use_dora or (module.dropout_p != 0 and (quantizer is not None or module.training)):
            raise ValueError(f"{name}: LoRA QAT does not support DoRA or nonzero dropout")
        if getattr(module, "super_fwd", None) is not None or getattr(module, "quant_state", None) is not None:
            raise ValueError(f"{name}: delegated super_fwd or quantized base weights are unsupported")
        if module.lora_A.bias is not None or module.lora_B.bias is not None:
            raise ValueError(f"{name}: LoRA adapters must not have biases")
        if module.bias is not None and (
            module.bias.shape != (module.out_features,)
            or module.bias.dtype != module.weight.dtype
            or module.bias.device != module.weight.device
        ):
            raise ValueError(f"{name}: bias must match base weight output shape, dtype and device")
    elif module.use_mxfp8:
        raise ValueError(f"{name}: MXFP8 expert backend is unsupported for LoRA QAT")
    elif isinstance(module, GroupedExpertsDeepEPLoRA) and not module.use_torch_mm and not _HAS_GROUPED_GEMM:
        raise ValueError(f"{name}: DeepEP QAT requires grouped_gemm or use_torch_mm=True")
    if not math.isfinite(module.scale):
        raise ValueError(f"{name}: LoRA scale must be finite")
    projections = _projections(module)
    for proj in projections:
        label = f"{name}.{proj.name}"
        for tensor in (proj.base, proj.a, proj.b):
            if isinstance(tensor, DTensor):
                raise TypeError(f"{label}: DTensor is unsupported during preparation")
            if (
                tensor.layout != torch.strided
                or tensor.is_quantized
                or tensor.dtype not in (torch.float32, torch.bfloat16, torch.float16)
            ):
                raise TypeError(f"{label}: weights must be unquantized strided float32/bfloat16/float16 tensors")
            if tensor.device.type not in ("cpu", "cuda", "meta"):
                raise ValueError(f"{label}: unsupported device {tensor.device}")
        if proj.base.requires_grad:
            raise ValueError(f"{label}: base weight must be frozen")
        if len({tensor.dtype for tensor in (proj.base, proj.a, proj.b)}) != 1:
            raise TypeError(f"{label}: base/A/B dtype must match")
        if len({tensor.device for tensor in (proj.base, proj.a, proj.b)}) != 1:
            raise ValueError(f"{label}: base/A/B device must match")
        if proj.layout == "out_in":
            valid = (
                proj.base.ndim == proj.a.ndim == proj.b.ndim == 2
                and proj.a.shape == (module.dim, module.in_features)
                and proj.b.shape == (module.out_features, module.dim)
                and proj.base.shape == (module.out_features, module.in_features)
            )
        else:
            valid = (
                proj.base.ndim == proj.a.ndim == proj.b.ndim == 3
                and proj.a.shape == (*proj.base.shape[:2], module.lora_dim)
                and proj.b.shape == (proj.base.shape[0], module.lora_dim, proj.base.shape[2])
            )
        if not valid or any(size <= 0 for tensor in (proj.base, proj.a, proj.b) for size in tensor.shape):
            raise ValueError(f"{label}: incompatible base/A/B shapes")
        if quantizer is not None:
            block_size = quantizer.config.block_size
            assert block_size is not None  # WeightFakeQuantizer resolves the default at construction.
            rows, cols = block_size
            if proj.shape[-2] % rows or proj.shape[-1] % cols:
                raise ValueError(f"{label}: canonical shape {proj.shape} is not divisible by block_size {(rows, cols)}")
    if len(projections) == 2:
        gate, down = projections
        if gate.base.dtype != down.base.dtype or gate.base.device != down.base.device:
            raise TypeError(f"{name}: grouped gate/up and down dtype/device must match")
        expected_up = module.config.moe_inter_dim * (2 if module.is_gated else 1)
        if gate.shape != (module.n_routed_experts, expected_up, module.config.expert_dim) or down.shape != (
            module.n_routed_experts,
            module.config.expert_dim,
            module.config.moe_inter_dim,
        ):
            raise ValueError(f"{name}: incompatible grouped projection shapes")


class _LoRAQAT:
    """Own rule selection, stateless LoRA QAT attachment, and merged-weight extraction.

    Args:
        rules: Resolved numerical rules constructed by QATConfig.build().
    """

    def __init__(self, rules: tuple[QATRule, ...]) -> None:
        self._rules = rules

    def _targets(self, model: nn.Module) -> dict[str, WeightFakeQuantizer]:
        """Resolve every rule and validate all targets before attaching anything."""
        _reject_distributed(model)
        modules = dict(model.named_modules(remove_duplicate=False))
        targets = {}
        owners = {}
        for index, rule in enumerate(self._rules):
            matches = [name for name in modules if any(fnmatchcase(name, p) for p in rule.target_modules)]
            if not matches:
                raise ValueError(f"rule {index}: no modules match {rule.target_modules}")
            for name in matches:
                module = modules[name]
                if not isinstance(module, _LORA_TYPES):
                    raise TypeError(f"{name}: QAT target is not a supported LoRA module")
                if id(module) in owners and owners[id(module)] != index:
                    raise ValueError(f"{name}: overlapping LoRA QAT rules")
                owners[id(module)] = index
                quantizer = rule.weight.build()
                _validate_lora(name, module, quantizer)
                targets[name] = quantizer
        _reject_tied_lora_bases(model, list(targets))
        return targets

    def prepare(self, model: nn.Module) -> nn.Module:
        """Attach quantizers only after all rules pass metadata-only validation.

        Args:
            model: PEFT-patched local model, possibly on meta. Existing parameter
                names, identities, trainability, and tensor storage are preserved.
                Selected LoRA bases must have exactly one parameter path; tied
                parameters and duplicate paths to the same LoRA module are unsupported.

        Returns:
            The same model with stateless weight_fake_quantizer children. No
            hooks, forward wrappers, persistent buffers, or merged weights are made.
        """
        targets = self._targets(model)
        for name, quantizer in targets.items():
            module = model.get_submodule(name)
            quantizer.train(module.training)
            module.weight_fake_quantizer = quantizer
        return model

    @torch.no_grad()
    def export(self, model: nn.Module, directory: str | Path) -> None:
        """Write complete local eval state plus packed merged-weight bytes.

        Args:
            model: Prepared model with every child in eval mode. All LoRA weights
                are merged, including untargeted adapters. Grouped operands are
                transposed from [experts, in, out] to [experts, out, in] for packing.
                Export clones detached tensors and never changes model state.
                Packed payloads and scales are snapshotted to CPU per projection,
                before processing the next projection.
                All LoRA bases must have exactly one parameter path, including
                untargeted adapters; unrelated ordinary tied parameters are supported.
            directory: Destination for manifest.json, model.safetensors and
                quantized.safetensors. Both tensor files exist even when empty.
        """
        # Keep checkpoint parsing and its optional dependency off the training import path.
        from nemo_automodel._transformers.qat_checkpoint import HAS_SAFETENSORS, QATCheckpoint, QuantizedProjection

        if not HAS_SAFETENSORS:
            raise ImportError("LoRA QAT export requires safetensors")
        if any(module.training for module in model.modules()):
            raise ValueError("LoRA QAT export requires model.eval() for every module")
        targets = self._targets(model)
        selected = {id(model.get_submodule(name)): quantizer for name, quantizer in targets.items()}
        _reject_tied_lora_bases(
            model,
            [name for name, module in model.named_modules(remove_duplicate=False) if isinstance(module, _LORA_TYPES)],
        )
        state = QATCheckpoint.model_state(model)
        quantized: list[QuantizedProjection] = []
        for name, module in model.named_modules(remove_duplicate=False):
            if not isinstance(module, _LORA_TYPES):
                continue
            quantizer = selected.get(id(module))
            attached = module.weight_fake_quantizer
            if quantizer is not None:
                if not isinstance(attached, WeightFakeQuantizer) or attached.config != quantizer.config:
                    raise ValueError(f"{name}: prepare this model with the export controller before exporting")
            elif attached is not None:
                raise ValueError(f"{name}: attached quantizer is outside this controller's rules")
            _validate_lora(name, module, quantizer)
            prefix = name + "." if name else ""
            for proj in _projections(module):
                key = prefix + proj.name
                effective = (
                    module.materialize_effective_weight()
                    if isinstance(module, LinearLoRA)
                    else (proj.base + module.scale * torch.bmm(proj.a, proj.b)).transpose(-2, -1)
                )
                if quantizer is None:
                    state[key] = effective if proj.layout == "out_in" else effective.transpose(-2, -1)
                    continue
                encoded = quantizer.quantize(effective)
                # Retain only CPU snapshots, not every projection's GPU packed storage.
                encoded = QuantizedWeight(
                    payload=encoded.payload.detach().cpu().clone(),
                    scales=encoded.scales.detach().cpu().clone(),
                    config=encoded.config,
                    shape=encoded.shape,
                )
                quantized.append(QuantizedProjection(key, effective.dtype, proj.layout, encoded))
                del state[key]
            # Only these actual LoRA owners define adapter keys. Unrelated names
            # containing "lora_A" elsewhere in the model are ordinary state.
            if isinstance(module, LinearLoRA):
                adapter_keys = [
                    prefix + field + "." + key
                    for field in ("lora_A", "lora_B")
                    for key in getattr(module, field).state_dict()
                ]
            else:
                adapter_keys = [
                    prefix + field
                    for field in ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
                ]
            for key in adapter_keys:
                del state[key]
        QATCheckpoint.save(state, quantized, directory)


class QAT:
    """Compose a numerical QAT plan with model preparation and persistence.

    Args:
        config: Canonical QATConfig. Weight rules currently require LoRA modules;
            legacy TorchAO preparation remains available without PEFT.
    """

    def __init__(self, config: QATConfig) -> None:
        if not isinstance(config, QATConfig):
            raise TypeError("config must be a QATConfig")
        plan = config.build()
        self._controller = _LoRAQAT(plan) if isinstance(plan, tuple) else plan

    def prepare(self, model: nn.Module) -> nn.Module:
        """Prepare before sharding and optimizer construction.

        Weight rules preserve the metadata-only preparation contract documented
        on _LoRAQAT.prepare, including meta tensors and parameter identity.
        Legacy TorchAO configs use their quantizer's prepare contract.
        """
        if not isinstance(self._controller, _LoRAQAT) and any(isinstance(m, _LORA_TYPES) for m in model.modules()):
            raise ValueError("TorchAO INT4 QAT with PEFT is not currently supported")
        return self._controller.prepare(model)

    def export(self, model: nn.Module, directory: str | Path) -> None:
        """Export a complete local eval model using _LoRAQAT.export's contract.

        Packed reference export is supported only for weight-rule plans, not
        legacy TorchAO quantizers. Untargeted adapters export as merged floats.
        """
        if not isinstance(self._controller, _LoRAQAT):
            raise NotImplementedError("Packed reference export requires QAT weight/rules, not TorchAO INT4")
        self._controller.export(model, directory)

    @staticmethod
    @torch.no_grad()
    def load_quantized_checkpoint(model: nn.Module, directory: str | Path) -> None:
        """Decode a packed checkpoint into the original, non-LoRA architecture.

        Args:
            model: Fresh materialized local model without LoRA adapters. Keys,
                shapes and dtypes must match the exported original architecture.
                Canonical [out, in] weights load directly; [experts, out, in]
                weights transpose back to [experts, in, out]. Ordinary state
                retains its original layout. Loading uses strict state-dict semantics.
            directory: Directory containing the three fixed checkpoint filenames.

        This is dequantized reference inference, not native packed GPU execution.
        All metadata, files, keys and tensor contracts are checked before loading.
        """
        from nemo_automodel._transformers.qat_checkpoint import HAS_SAFETENSORS, QATCheckpoint

        if not HAS_SAFETENSORS:
            raise ImportError("LoRA QAT loading requires safetensors")
        if any(isinstance(module, _LORA_TYPES) for module in model.modules()):
            raise TypeError("load_quantized_checkpoint requires a fresh NON-LoRA model")
        QATCheckpoint.load(model, directory)
