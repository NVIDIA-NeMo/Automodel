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

"""Inkling2 text checkpoint mapping and mixed-precision import through DCP load targets."""

import re
from dataclasses import dataclass
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate
from torch.distributed.tensor import empty as dtensor_empty

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.moe.state_dict_utils import is_dtensor

from .config import Inkling2Config, Inkling2TextConfig


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Return the tensor's local storage without copying.

    Args:
        tensor: Tensor of shape [...], with arbitrary rank and axis order. A DTensor
            describes the global shape and its mesh placements.

    Returns:
        Tensor with the same axis order, dtype and device. For a DTensor, each dimension
        has its per-rank local shard extent; otherwise the input is returned. The result
        aliases the input storage.
    """
    # DTensor is an external tensor wrapper; its public local-storage API is required for checkpoint writes.
    return cast(DTensor, tensor).to_local() if is_dtensor(tensor) else tensor


def _quantized_storage(template: torch.Tensor, divisor: int, dtype: torch.dtype) -> torch.Tensor:
    """Allocate source-layout storage for [..., input/divisor], preserving shard placements.

    Args:
        template: Floating destination [..., input]; a DTensor supplies the global shape and
            placements, and both its global and local trailing extents must be multiples
            of ``divisor``.
        divisor: Source elements per storage element along the trailing axis: 2 for packed
            FP4 pairs, the block size for block scales.
        dtype: Storage dtype of the source tensor.

    Returns:
        Newly allocated, uninitialized [..., input/divisor] storage on the template's device.
        A DTensor template yields a DTensor on the same mesh and placements whose local
        shard has the template's local extents. Nothing aliases the template.
    """
    local = _local(template)
    if template.shape[-1] % divisor or local.shape[-1] % divisor:
        raise ValueError("Quantized Inkling2 input shards must align with the quantization block size")
    shape = (*template.shape[:-1], template.shape[-1] // divisor)
    storage = torch.empty((*local.shape[:-1], local.shape[-1] // divisor), device=local.device, dtype=dtype)
    if not is_dtensor(template):
        return storage
    distributed = cast(DTensor, template)
    if any(p.is_partial() for p in distributed.placements):
        raise ValueError("Checkpoint destinations cannot have Partial placements")
    return DTensor.from_local(
        storage,
        distributed.device_mesh,
        distributed.placements,
        shape=torch.Size(shape),
        stride=torch.empty(shape, device="meta").stride(),
    )


def _expert_amax_storage(template: torch.Tensor) -> torch.Tensor:
    """Allocate [experts] outer-scale metadata, replicated across inner-axis shards.

    Args:
        template: Expert tensor [experts, ...]; a DTensor supplies the mesh and placements,
            of which only a leading-axis shard is kept.

    Returns:
        Newly allocated, uninitialized FP32 [experts] storage on the template's device. A
        DTensor template yields a DTensor sharded only where the template shards its
        leading axis and replicated elsewhere. Nothing aliases the template.
    """
    local = _local(template)
    storage = torch.empty(local.shape[0], dtype=torch.float32, device=local.device)
    if not is_dtensor(template):
        return storage
    distributed = cast(DTensor, template)
    placements = tuple(p if p.is_shard(0) else Replicate() for p in distributed.placements)
    return DTensor.from_local(
        storage,
        distributed.device_mesh,
        placements,
        shape=torch.Size((template.shape[0],)),
        stride=(1,),
    )


@dataclass(frozen=True)
class _QuantizedProjection:
    """Source [experts, output, input] tensors and a matching floating destination view.

    The destination aliases model storage, potentially a noncontiguous transpose.
    Packed FP4 has input/2 columns, scales have input/16 or input/32 columns.
    Only one expert and at most 128 output rows are expanded at a time.
    """

    destination: torch.Tensor
    weight: torch.Tensor
    scale: torch.Tensor
    amax: torch.Tensor | None = None
    denominator: torch.Tensor | None = None

    @torch.no_grad()
    def finish(self) -> None:
        """Decode source tiles in FP32, cast once and copy into the existing destination."""
        destination, weight, scale = (_local(t) for t in (self.destination, self.weight, self.scale))
        fp4 = self.amax is not None
        block = 16 if fp4 else 32
        if destination.ndim != 3 or destination.shape[-1] % block:
            raise ValueError("Expert destinations must be [experts, output, block-aligned input]")
        expected_weight = (*destination.shape[:-1], destination.shape[-1] // (2 if fp4 else 1))
        if weight.shape != expected_weight or scale.shape != (*destination.shape[:-1], destination.shape[-1] // block):
            raise ValueError("Packed expert weights/scales do not match destination shape")
        if weight.dtype != (torch.uint8 if fp4 else torch.float8_e4m3fn):
            raise ValueError("Unexpected packed expert weight dtype")
        if scale.dtype != (torch.float8_e4m3fn if fp4 else torch.uint8):
            raise ValueError("Unexpected expert block-scale dtype")
        outer = None
        if fp4:
            assert self.amax is not None and self.denominator is not None
            amax, denominator = _local(self.amax), _local(self.denominator)
            if amax.shape != (destination.shape[0],) or denominator.shape != (1,):
                raise ValueError("NVFP4 requires per-expert amax and one outer-scale denominator")
            if amax.dtype != torch.float32 or denominator.dtype != torch.float32:
                raise ValueError("NVFP4 outer-scale metadata must be FP32")
            if (
                not torch.isfinite(amax).all()
                or torch.any(amax < 0)
                or not torch.isfinite(denominator).all()
                or torch.any(denominator <= 0)
            ):
                raise ValueError("NVFP4 outer scales must be finite, with nonnegative amax and positive denominator")
            outer = amax / denominator
        magnitudes = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=weight.device, dtype=torch.float32)
        for expert in range(destination.shape[0]):
            for start in range(0, destination.shape[1], 128):
                packed = weight[expert, start : start + 128]
                scales = scale[expert, start : start + 128]
                if fp4:
                    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
                    decoded = magnitudes[codes & 7] * torch.where(codes < 8, 1, -1)
                    multiplier = scales.float()
                    if not torch.isfinite(multiplier).all() or torch.any(multiplier < 0):
                        raise ValueError("NVFP4 block scales must be finite and nonnegative")
                else:
                    if torch.any(scales == 255):
                        raise ValueError("MXFP8 scale byte 255 encodes NaN")
                    decoded = packed.float()
                    if not torch.isfinite(decoded).all():
                        raise ValueError("MXFP8 weights must be finite")
                    multiplier = torch.exp2(scales.float() - 127)
                decoded = (decoded.reshape(*decoded.shape[:-1], -1, block) * multiplier.unsqueeze(-1)).flatten(-2)
                if outer is not None:
                    decoded = decoded * outer[expert]
                converted = decoded.to(destination.dtype)
                if not torch.isfinite(converted).all():
                    raise ValueError("Decoded expert weights overflow the destination dtype")
                destination[expert, start : start + 128].copy_(converted)


def _needs_layout_buffer(name: str, tensor: torch.Tensor) -> bool:
    """Identify source-layout operations that intersect a native row shard.

    Args:
        name: Native parameter name.
        tensor: Projection [rows, hidden] or residual query [queries, hidden],
            optionally DTensor with Shard(0) on one or more mesh axes.

    Returns:
        True for a sharded fused row split or removal of a sharded singleton axis.
    """
    return (
        isinstance(tensor, DTensor)
        and any(placement.is_shard(0) for placement in tensor.placements)
        and (".attn.qkvr." in name or (name.endswith("query.weight") and tensor.shape[0] == 1))
    )


@dataclass(frozen=True)
class _FloatingLayout:
    """One native-shaped temporary and its original sharded destination."""

    destination: DTensor
    buffer: DTensor

    @torch.no_grad()
    def finish(self) -> None:
        """Copy [rows, hidden] or [queries, hidden] into the original DP/EP shard storage."""
        self.destination.copy_(self.buffer.redistribute(placements=self.destination.placements))


_PEFT_KEY = re.compile(
    r"base_model\.model\.layers\.(\d+)\.(?:"
    r"attn\.(?:qkvr|wo_ud)\.lora_[AB]\.weight|"
    r"mlp\.experts\.lora_(?:gate_and_up|down)_[AB])"
)
_EXPERT_KEY = re.compile(r"layers\.(\d+)\.mlp\.experts\.(?:gate_and_up_projs|down_projs)")


class Inkling2StateDictAdapter(StateDictAdapter):
    """Map the delivered text weights; preserve floating checkpoint layouts on export.

    Multimodal configs additionally map visual parameters without changing their layout.
    Load destinations normally alias model storage. Quantized experts, row-sharded fused
    projections and singleton residual queries instead get staging tensors from
    ``to_hf(for_checkpoint_load=True)``; ``from_hf`` decodes or copies them into the
    original storage and reports those keys through ``view_loaded_native_keys``.
    """

    # Direct aliases plus per-tensor staging avoid a full model copy.
    _supports_low_memory_dcp_load = True

    def __init__(self, config: Inkling2TextConfig | Inkling2Config) -> None:
        self.multimodal = isinstance(config, Inkling2Config)
        config = config.text_config if isinstance(config, Inkling2Config) else config
        self.config = config
        self.source_quantized = config.quantization_config is not None
        self.nvfp4_layer_count = (
            0 if config.quantization_config is None else config.quantization_config["nvfp4_layer_count"]
        )
        # Staging allocated by to_hf(for_checkpoint_load=True) and completed by the following from_hf.
        self._pending: list[tuple[frozenset[str], _QuantizedProjection | _FloatingLayout]] = []
        self._view_loaded_native_keys: set[str] = set()

    def _split_sizes(self, layer: int) -> tuple[int, ...]:
        cfg = self.config
        local = layer in cfg.local_layer_ids
        q = cfg.swa_num_attention_heads if local else cfg.num_attention_heads
        kv = cfg.swa_num_key_value_heads if local else cfg.num_key_value_heads
        d = cfg.swa_head_dim if local else cfg.head_dim
        return q * d, kv * d, kv * d, q * cfg.d_rel, q * d, q

    def get_hf_state_dict_keys(self, state_dict: dict[str, torch.Tensor]) -> list[str]:
        """List released base keys without treating newly injected LoRA rows as base weights.

        Args:
            state_dict: Native model tensors, possibly with unprefixed LoRA A/B
                matrices [rank, input]/[output, rank] or grouped [experts, ..., rank].

        Returns:
            Released base tensor names only. Adapter saves use the shared PEFT
            prefix and retain their native layout through to_hf instead.
        """
        base = {
            key: value for key, value in state_dict.items() if _PEFT_KEY.fullmatch("base_model.model." + key) is None
        }
        return super().get_hf_state_dict_keys(base)

    def forced_hf_dtype_mapping(self, state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
        """Export actual floating dtypes instead of reusing packed source-checkpoint dtype metadata.

        Args:
            state_dict: Source-named floating export views, with arbitrary parameter shapes.

        Returns:
            SafeTensors dtype names for every exported tensor, including intrinsic FP32 groups.
        """
        names = {torch.float32: "F32", torch.bfloat16: "BF16", torch.float16: "F16"}
        if any(tensor.dtype not in names for tensor in state_dict.values()):
            raise ValueError("Inkling2 training export requires floating weights")
        return {key: names[tensor.dtype] for key, tensor in state_dict.items()}

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        *,
        for_checkpoint_load: bool = False,
        **kwargs: Any,
    ) -> list[tuple[str, torch.Tensor]]:
        """Return source-layout tensors, gathering row shards only when the layout requires it.

        Args:
            fqn: Native text parameter or persistent-buffer name.
            tensor: Native parameter layout; grouped experts are [experts, input, output].
            for_checkpoint_load: Zero omitted fused-projection padding in destination storage before loading.
                Floating export otherwise leaves the input unchanged.

        Returns:
            Source names and tensors retaining dtype/device. Fused row splits and
            singleton query squeezes replicate Shard(0) axes for export; those
            outputs do not alias the original shards. Direct load views must alias.
        """
        # StateDictAdapter imposes an open keyword boundary; unknown modes are rejected.
        if kwargs:
            raise TypeError(f"Unsupported Inkling2 adapter options: {sorted(kwargs)}")
        # Adapter checkpoints retain the native fused-projection/grouped-expert
        # layout. Automodel's shared PEFT loader restores these names and owns
        # EP slicing; no serving-format split or weight transpose applies.
        if match := _PEFT_KEY.fullmatch(fqn):
            if int(match[1]) >= self.config.num_hidden_layers:
                raise ValueError(f"Inkling2 adapter layer is out of range: {fqn}")
            return [(fqn, tensor)]
        if _needs_layout_buffer(fqn, tensor):
            if for_checkpoint_load:
                raise ValueError("Row-sharded Inkling2 layout destinations are allocated by to_hf, not per tensor")
            distributed = cast(DTensor, tensor)
            tensor = distributed.redistribute(
                placements=tuple(Replicate() if p.is_shard(0) else p for p in distributed.placements)
            )
        if self.multimodal and fqn.startswith("visual."):
            return [("model." + fqn, tensor)]
        roots = {
            "embed_tokens.weight": "embed.weight",
            "embed_norm.weight": "embed_norm.weight",
            "norm.weight": "norm.weight",
            "lm_head.weight": "unembed.weight",
        }
        if fqn in roots:
            return [("model.llm." + roots[fqn], tensor)]
        if fqn == "output_query.weight":
            return [("model.llm.attn_res.output_query", tensor.squeeze(0))]
        match = re.fullmatch(r"layers\.(\d+)\.(.+)", fqn)
        if match is None:
            raise ValueError(f"Unknown Inkling2 native tensor: {fqn}")
        layer, suffix = int(match[1]), match[2]
        prefix = f"model.llm.layers.{layer}."
        if suffix in ("attn_query.weight", "mlp_query.weight"):
            kind = "attn" if suffix.startswith("attn") else "mlp"
            index = layer - 1 if kind == "attn" else layer
            return [(f"model.llm.attn_res.{kind}_queries.{index}", tensor.squeeze(0))]
        if suffix.startswith("attn.qkvr."):
            leaf = suffix.rsplit(".", 1)[1]
            sizes = self._split_sizes(layer)
            if for_checkpoint_load and tensor.shape[0] > sum(sizes):
                # These native-only rows have no source tensor. Materialized meta models may contain garbage here.
                with torch.no_grad():
                    tensor[sum(sizes) :].zero_()
            parts = tensor[: sum(sizes)].split(sizes, dim=0)
            names = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
            return [(prefix + f"attn.{name}.{leaf}", value) for name, value in zip(names, parts)]
        match = re.fullmatch(r"attn_norms\.(\d+)\.weight", suffix)
        if match:
            names = (
                ("attn_norm",)
                if layer in self.config.local_layer_ids
                else ("q_input_norm", "k_input_norm", "v_input_norm")
            )
            return [(prefix + names[int(match[1])] + ".weight", tensor)]
        if suffix in ("mlp.experts.gate_and_up_projs", "mlp.experts.down_projs"):
            name = "w13" if suffix.endswith("gate_and_up_projs") else "w2"
            return [(prefix + f"mlp.experts.{name}_weight", tensor.transpose(-1, -2))]
        suffix = suffix.replace("attn.relative_table", "attn.rel_logits_proj.proj")
        suffix = suffix.replace(".weight_proj.weight", ".weight_proj")
        suffix = suffix.replace("mlp.gate.e_score_correction_bias", "mlp.gate.bias")
        allowed = re.fullmatch(
            r"(attn\.(wo_ud\.(weight|bias)|q_l2_clip\.scale|rel_l2_clip\.scale|rel_logits_proj\.proj|[kv]_sconv\.(weight|static_w|weight_proj))|mlp\.gate\.(weight|bias)|mlp_norm\.weight|(?:attn|mlp)_sconv\.weight)",
            suffix,
        )
        if allowed is None:
            raise ValueError(f"Unknown Inkling2 native tensor: {fqn}")
        return [(prefix + suffix, tensor)]

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        for_checkpoint_load: bool = False,
        device_mesh: DeviceMesh | None = None,
        v4_compatible: bool = False,
        legacy_paramwrapper_layout: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Export floating source-layout views, or allocate DCP load destinations.

        Args:
            state_dict: Native tensors in their original floating layouts and placements.
            exclude_key_regex: Optional shared-loader exclusion pattern.
            quantization: Allocate packed expert staging matching the released NVFP4/MXFP8 layout.
                Requires for_checkpoint_load; trained weights are exported without quantization.
            for_checkpoint_load: Whether shared checkpointing requested DCP destinations. Row-sharded
                fused projections and singleton residual queries then get one replicated temporary
                each, which from_hf redistributes into the original shards.
            device_mesh: Shared-loader context; placements are retained on each tensor view.
            v4_compatible: Legacy config export is not qualified for this text entry point.
            legacy_paramwrapper_layout: Legacy expert LoRA export is unsupported.

        Returns:
            Source-named tensors. Export returns floating views and leaves inputs unchanged. Load
            destinations alias model storage except for the staging above; loading canonicalizes
            omitted projection padding to zero.
        """
        if kwargs:
            raise TypeError(f"Unsupported Inkling2 adapter options: {sorted(kwargs)}")
        if v4_compatible or legacy_paramwrapper_layout:
            raise ValueError("Inkling2 legacy config and ParamWrapper export are not qualified")
        if quantization and not for_checkpoint_load:
            raise ValueError(
                "Quantization targets are for checkpoint loading only; export trained weights without quantization"
            )
        if for_checkpoint_load:
            self._pending, self._view_loaded_native_keys = [], set()
        output: dict[str, torch.Tensor] = {}
        for key, tensor in state_dict.items():
            if exclude_key_regex is not None and re.search(exclude_key_regex, key) is not None:
                continue
            if for_checkpoint_load and _needs_layout_buffer(key, tensor):
                output.update(self._layout_load_targets(key, cast(DTensor, tensor)))
            elif quantization and _EXPERT_KEY.fullmatch(key):
                output.update(self._quantized_load_targets(key, tensor))
            else:
                output.update(self.convert_single_tensor_to_hf(key, tensor, for_checkpoint_load=for_checkpoint_load))
        return output

    def _layout_load_targets(self, name: str, tensor: DTensor) -> dict[str, torch.Tensor]:
        """Allocate one replicated native-shaped temporary for a row-sharded layout conversion.

        Args:
            name: Native fused projection or singleton residual query name.
            tensor: Final destination with Shard(0) on one or more mesh axes.

        Returns:
            Source-named views of the temporary for DCP to fill.
        """
        placements = tuple(Replicate() if p.is_shard(0) else p for p in tensor.placements)
        buffer = dtensor_empty(tensor.shape, device_mesh=tensor.device_mesh, placements=placements, dtype=tensor.dtype)
        targets = dict(self.convert_single_tensor_to_hf(name, buffer, for_checkpoint_load=True))
        self._pending.append((frozenset(targets), _FloatingLayout(tensor, buffer)))
        self._view_loaded_native_keys.add(name)
        return targets

    def _quantized_load_targets(self, name: str, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        """Allocate packed source-layout staging for one grouped expert projection.

        Args:
            name: Native grouped expert key, ``layers.N.mlp.experts.{gate_and_up_projs,down_projs}``.
            tensor: Final floating [experts, input, output] storage; DTensor shards are preserved and
                Partial placements are rejected.

        Returns:
            Uninitialized packed weight, block scales and, for NVFP4 layers, per-expert amax and the
            outer-scale denominator, for DCP to fill.
        """
        [(key, destination)] = self.convert_single_tensor_to_hf(name, tensor, for_checkpoint_load=True)
        fp4 = int(name.split(".")[1]) < self.nvfp4_layer_count
        packed = _quantized_storage(destination, 2 if fp4 else 1, torch.uint8 if fp4 else torch.float8_e4m3fn)
        scale = _quantized_storage(destination, 16 if fp4 else 32, torch.float8_e4m3fn if fp4 else torch.uint8)
        targets = {key: packed, key + "_scale": scale}
        amax, denominator = None, None
        if fp4:
            amax = _expert_amax_storage(destination)
            denominator = torch.empty(1, dtype=torch.float32, device=_local(destination).device)
            targets.update({key + "_amax": amax, key + "_outer_scale_denominator": denominator})
        self._pending.append((frozenset(targets), _QuantizedProjection(destination, packed, scale, amax, denominator)))
        self._view_loaded_native_keys.add(name)
        return targets

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Native keys completed in model storage by from_hf after the most recent DCP load.

        The shared loader counts them as loaded although from_hf omits them from its result.
        """
        return set(self._view_loaded_native_keys)

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Convert an already-read text checkpoint, preserving interleaving and supplied floating dtypes.

        Args:
            hf_state_dict: Delivered source names and floating or packed tensors. After a DCP load into
                ``to_hf(for_checkpoint_load=True)`` destinations, the staged quantized experts and replicated
                layout temporaries are decoded or copied into model storage here and omitted from the result.
                model.visual tensors are omitted intentionally by the text-only entry point. Quantized experts
                read into a full CPU state dict use [E,output,packed_input] storage.
            device_mesh: Shared-loader context; DTensor values keep their own placements.

        Returns:
            Native tensor mapping. Direct weights are views. Keys completed in model storage are reported
            through ``view_loaded_native_keys`` instead; full-dict expert decoding and projection
            concatenation allocate.
        """
        if kwargs:
            raise TypeError(f"Unsupported Inkling2 adapter options: {sorted(kwargs)}")
        if hf_state_dict and all(_PEFT_KEY.fullmatch(key) for key in hf_state_dict):
            for key in hf_state_dict:
                match = _PEFT_KEY.fullmatch(key)
                assert match is not None
                if int(match[1]) >= self.config.num_hidden_layers:
                    raise ValueError(f"Inkling2 adapter layer is out of range: {key}")
            return dict(hf_state_dict)
        staged: set[str] = set()
        for keys, conversion in self._pending:
            # Only staging that DCP filled in this load is completed; leftovers from an aborted load are dropped.
            if keys <= hf_state_dict.keys():
                conversion.finish()
                staged |= keys
        self._pending = []
        output, fused, consumed = {}, {}, set(staged)
        roots = {
            "embed.weight": "embed_tokens.weight",
            "embed_norm.weight": "embed_norm.weight",
            "norm.weight": "norm.weight",
            "unembed.weight": "lm_head.weight",
        }
        projections = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")
        for key, value in hf_state_dict.items():
            if key in staged:
                continue
            if key.startswith("model.visual."):
                if self.multimodal:
                    output[key.removeprefix("model.")] = value
                consumed.add(key)
                continue
            if not key.startswith("model.llm."):
                raise ValueError(f"Unknown Inkling2 checkpoint tensor: {key}")
            suffix = key.removeprefix("model.llm.")
            if suffix in roots:
                output[roots[suffix]] = value
            elif suffix == "attn_res.output_query":
                output["output_query.weight"] = value.unsqueeze(0)
            elif match := re.fullmatch(r"attn_res\.(attn|mlp)_queries\.(\d+)", suffix):
                layer = int(match[2]) + (1 if match[1] == "attn" else 0)
                output[f"layers.{layer}.{match[1]}_query.weight"] = value.unsqueeze(0) if value.ndim == 1 else value
            elif match := re.fullmatch(r"layers\.(\d+)\.(.+)", suffix):
                layer, leaf = int(match[1]), match[2]
                prefix = f"layers.{layer}."
                if sub := re.fullmatch(
                    r"attn\.(wq_du|wk_dv|wv_dv|wr_du|gate_attn|attn_sink_proj)\.(weight|bias)", leaf
                ):
                    fused.setdefault(prefix + "attn.qkvr." + sub[2], {})[projections.index(sub[1])] = value
                elif sub := re.fullmatch(r"(attn_norm|[qkv]_input_norm)\.weight", leaf):
                    index = (
                        0 if sub[1] == "attn_norm" else ("q_input_norm", "k_input_norm", "v_input_norm").index(sub[1])
                    )
                    output[prefix + f"attn_norms.{index}.weight"] = value
                elif sub := re.fullmatch(r"mlp\.experts\.(w13|w2)_weight", leaf):
                    native = prefix + "mlp.experts." + ("gate_and_up_projs" if sub[1] == "w13" else "down_projs")
                    if value.dtype in (torch.uint8, torch.float8_e4m3fn):
                        fp4 = value.dtype == torch.uint8
                        shape = (*value.shape[:-1], value.shape[-1] * (2 if fp4 else 1))
                        destination = torch.empty(shape, device=value.device, dtype=torch.float32)
                        extra = ["_scale", "_amax", "_outer_scale_denominator"] if fp4 else ["_scale"]
                        missing = [key + s for s in extra if key + s not in hf_state_dict]
                        if missing:
                            raise ValueError(f"Missing Inkling2 quantization metadata: {missing}")
                        _QuantizedProjection(
                            destination,
                            value,
                            hf_state_dict[key + "_scale"],
                            hf_state_dict[key + "_amax"] if fp4 else None,
                            hf_state_dict[key + "_outer_scale_denominator"] if fp4 else None,
                        ).finish()
                        consumed.update(key + s for s in extra)
                        value = destination
                    elif value.dtype not in (torch.float32, torch.bfloat16, torch.float16):
                        raise ValueError("Unsupported Inkling2 expert weight dtype")
                    output[native] = value.transpose(-1, -2)
                elif re.fullmatch(r"mlp\.experts\.w(?:13|2)_weight_(scale|amax|outer_scale_denominator)", leaf):
                    continue
                else:
                    native = leaf.replace("attn.rel_logits_proj.proj", "attn.relative_table")
                    native = native.replace(".weight_proj", ".weight_proj.weight")
                    native = native.replace("mlp.gate.bias", "mlp.gate.e_score_correction_bias")
                    # Validate the inverse mapping, rather than silently accepting a misspelled field.
                    if self.convert_single_tensor_to_hf(prefix + native, value)[0][0] != key:
                        raise ValueError(f"Unsupported Inkling2 checkpoint tensor: {key}")
                    output[prefix + native] = value
            else:
                raise ValueError(f"Unknown Inkling2 checkpoint tensor: {key}")
            consumed.add(key)
        for key, parts in fused.items():
            if set(parts) != set(range(6)):
                raise ValueError(f"Incomplete fused attention projection: {key}")
            if any(is_dtensor(piece) for piece in parts.values()):
                # DTensor pieces are to_hf load views that alias the fused destination; DCP already filled it.
                if not all(is_dtensor(piece) for piece in parts.values()):
                    raise ValueError(f"Mixed local and distributed fused projection pieces: {key}")
                self._view_loaded_native_keys.add(key)
                continue
            value = torch.cat([parts[i] for i in range(6)], dim=0)
            pad = -value.shape[0] % 8
            output[key] = torch.cat((value, value.new_zeros((pad, *value.shape[1:]))), dim=0) if pad else value
        unused = set(hf_state_dict) - consumed
        if unused:
            raise ValueError(f"Orphaned quantization metadata: {sorted(unused)}")
        return output
