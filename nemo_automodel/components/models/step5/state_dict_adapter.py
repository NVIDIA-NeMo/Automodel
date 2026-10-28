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

"""Step-5 FP8 loading, interleaved QKV mapping and native expert conversion."""

import re
from collections.abc import Iterator, Mapping
from typing import Any, cast

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import CheckpointLoadPart
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import dequantize_from_fp8
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.models.step5.configuration import ConfigValue, Step5Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
)

_DECODER_LAYER = re.compile(r"^model\.layers\.(\d+)\.")

_INDEXER_NAMES = {
    "sparse_indexer_q": "indexer.q_proj",
    "sparse_indexer_k": "indexer.k_proj",
    "sparse_indexer_z": "indexer.z_proj",
    "sparse_indexer_w": "indexer.w_proj",
    "sparse_indexer_q_norm": "indexer.q_norm",
    "sparse_indexer_k_norm": "indexer.k_norm",
    "ssmax_s": "indexer.ssmax_s",
}


class Step5StateDictAdapter(Step3p5StateDictAdapter):
    """Preserve Step-family expert conversion and add Step-5 attention mappings.

    Loading needs bounded per-projection conversion storage, so the adapter does
    not advertise the DCP direct-destination fast path.
    """

    _supports_low_memory_dcp_load = False

    def __init__(self, config: Step5Config, moe_config: MoEConfig, backend: BackendConfig, dtype: torch.dtype) -> None:
        super().__init__(config, moe_config, backend, dtype)
        quantization = config.quantization_config
        self._load_fp8_experts = quantization is not None
        excluded: tuple[str, ...] = ()
        if quantization is not None:
            if quantization.get("quant_method") != "fp8" or quantization.get("weight_block_size") != [128, 128]:
                raise ValueError("Step5 checkpoint loading requires FP8 blocks of shape [128, 128]")
            configured_exclusions = quantization.get("modules_to_not_convert", [])
            if not isinstance(configured_exclusions, list) or any(
                not isinstance(name, str) for name in configured_exclusions
            ):
                raise ValueError("Step5 modules_to_not_convert must be a list of module or parameter names")
            excluded = tuple(cast(list[str], configured_exclusions))
        self._unquantized_expert_modules = frozenset(excluded)

    def iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        checkpoint_metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[CheckpointLoadPart] | None:
        """Load in bounded parts unless this rank holds only part of the decoder.

        Every rank issues one load collective per part. Pipeline stages own
        different decoder layers, so their part counts differ; such ranks use
        the ordinary checkpoint load instead.

        Args:
            model_state_dict: Final native storage; see ``_iter_checkpoint_load_parts``.
            device_mesh: Optional expert mesh used for native/HF conversion.
            checkpoint_metadata: Unused; part destinations derive from model shapes.

        Returns:
            Checkpoint load parts, or ``None`` on a distributed rank whose state
            dict lacks some of the decoder layers.
        """
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            present = {int(match.group(1)) for name in model_state_dict if (match := _DECODER_LAYER.match(name))}
            if not set(range(self.config.num_hidden_layers)) <= present:
                return None
        return self._iter_checkpoint_load_parts(model_state_dict, device_mesh)

    def _iter_checkpoint_load_parts(
        self,
        model_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
    ) -> Iterator[CheckpointLoadPart]:
        """Load mixed FP8/BF16 expert projections into final BF16 storage.

        Args:
            model_state_dict: Final native storage: fused gate/up tensors
                [experts, hidden, 2 * intermediate], down tensors
                [experts, intermediate, hidden], and other native model tensors.
                DTensors shard experts over ep and input dimensions over ep_shard.
            device_mesh: Optional expert mesh used for native/HF conversion.

        Yields:
            Checkpoint load parts with expert projections [experts, out, in].
            Quantized projections use temporary FP8 tensors and FP32 block scales
            [experts, ceil(out / 128), ceil(in / 128)]; finish callbacks decode local
            experts into final storage. Excluded or unquantized projections use
            BF16 views aliasing final parameter storage, without temporary copies.
            Backend _extra_state entries retain their initialized runtime values;
            they are accounted for by the first weight part without checkpoint I/O.
        """
        extra_state_keys = frozenset(name for name in model_state_dict if name.endswith("_extra_state"))
        for name, destination in model_state_dict.items():
            if name.endswith("_extra_state"):
                continue
            model_keys = frozenset((name,)) | extra_state_keys
            extra_state_keys = frozenset()
            if name.endswith(".moe.experts.gate_and_up_projs"):
                gate, up = destination.chunk(2, dim=-1)
                prefix = name.removesuffix("experts.gate_and_up_projs")
                packed_name = prefix + "experts.w13_weight"
                views = {
                    prefix + "gate_proj.weight": gate.transpose(1, 2),
                    prefix + "up_proj.weight": up.transpose(1, 2),
                }
            elif name.endswith(".moe.experts.down_projs"):
                prefix = name.removesuffix("experts.down_projs")
                packed_name = prefix + "experts.w2_weight"
                views = {prefix + "down_proj.weight": destination.transpose(1, 2)}
            else:
                tensors = dict(self.convert_single_tensor_to_hf(name, destination))

                def finish_plain(
                    tensors: dict[str, torch.Tensor] = tensors,
                    name: str = name,
                    destination: torch.Tensor = destination,
                ) -> None:
                    """Install one converted non-expert checkpoint part.

                    Args:
                        tensors: Source-named tensors in the layouts documented by from_hf.
                        name: Native destination key.
                        destination: Final native tensor matching the converted tensor's
                            global shape and DTensor placements; mutated in place.
                    """
                    converted = self.from_hf(dict(tensors), device_mesh=device_mesh)
                    with torch.no_grad():
                        destination.copy_(converted[name])

                yield CheckpointLoadPart(tensors, model_keys, frozenset(tensors), finish_plain)
                continue

            tensors = {}
            temporary_keys = set()
            decode_parts = []
            for source_name, view in views.items():
                # Source quantization metadata may name the packed original
                # expert parameter or the exported gate/up/down projection.
                aliases = (source_name, source_name.removesuffix(".weight"), packed_name)
                excluded = any(
                    alias == module or alias.startswith(module + ".")
                    for alias in aliases
                    for module in self._unquantized_expert_modules
                )
                if not self._load_fp8_experts or excluded:
                    tensors[source_name] = view
                    continue
                local_view = view.to_local() if isinstance(view, DTensor) else view
                packed = torch.empty_like(view, dtype=torch.float8_e4m3fn, memory_format=torch.contiguous_format)
                local_packed = packed.to_local() if isinstance(packed, DTensor) else packed
                scale_shape = (*view.shape[:-2], (view.shape[-2] + 127) // 128, (view.shape[-1] + 127) // 128)
                local_scale_shape = (
                    *local_view.shape[:-2],
                    (local_view.shape[-2] + 127) // 128,
                    (local_view.shape[-1] + 127) // 128,
                )
                local_scales = torch.empty(local_scale_shape, dtype=torch.float32, device=local_view.device)
                if isinstance(view, DTensor):
                    for axis, placement in enumerate(view.placements):
                        if isinstance(placement, Shard) and placement.dim in (1, 2):
                            size = view.shape[placement.dim]
                            if size % (128 * view.device_mesh.size(axis)):
                                raise ValueError("FP8 expert load requires block-aligned inner-axis sharding")
                    scales = DTensor.from_local(
                        local_scales,
                        view.device_mesh,
                        view.placements,
                        shape=torch.Size(scale_shape),
                        stride=torch.empty(scale_shape, device="meta").stride(),
                    )
                else:
                    scales = local_scales
                tensors[source_name] = packed
                tensors[source_name + "_scale_inv"] = scales
                temporary_keys.update((source_name, source_name + "_scale_inv"))
                decode_parts.append((source_name, local_packed, local_scales, local_view))

            def finish_experts(
                parts: tuple[tuple[str, torch.Tensor, torch.Tensor, torch.Tensor], ...] = tuple(decode_parts),
            ) -> None:
                """Decode local projections directly into final weight views.

                Args:
                    parts: Source name, FP8 weights [local_experts, out, in],
                        FP32 block scales [local_experts, ceil(out / 128), ceil(in / 128)],
                        and destination view [local_experts, out, in] for each projection.
                        Destination views alias final model parameters and are mutated.
                """
                with torch.no_grad():
                    for source_name, packed, scales, target in parts:
                        for expert in range(target.shape[0]):
                            decoded = dequantize_from_fp8(
                                packed[expert], scales[expert], dtype=target.dtype, name=source_name
                            )
                            target[expert].copy_(decoded)

            yield CheckpointLoadPart(tensors, model_keys, frozenset(temporary_keys), finish_experts)

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: ConfigValue,
    ) -> dict[str, torch.Tensor]:
        """Load source tensors into trainable, interleaved native storage.

        Args:
            hf_state_dict: Mapping to source tensors. Experts have layout
                [experts, out, in]; Q/K/V have layout [heads * head_dim, hidden].
            device_mesh: Optional expert mesh handled by the Step-family adapter.
            **kwargs: Checkpointer options forwarded to the base adapter.

        Returns:
            Mapping to native tensors. QKV has layout
            [kv_groups * (query_heads_per_group + 2) * head_dim, hidden]; experts
            use [experts, hidden, 2 * intermediate] and [experts, intermediate, hidden].
        """
        converted = {}
        for name, tensor in hf_state_dict.items():
            layer = re.search(r"(?:^|\.)layers\.(\d+)\.", name)
            if layer and int(layer.group(1)) >= self.config.num_hidden_layers:
                continue
            if name.endswith("_scale_inv"):
                continue
            if tensor.dtype == torch.float8_e4m3fn:
                scale_name = name + "_scale_inv"
                if scale_name not in hf_state_dict:
                    raise ValueError(f"Missing FP8 scale for {name}")
                scales = hf_state_dict[scale_name]
                if tensor.ndim == 3:
                    # Select this rank's experts before dequantization. A DTensor
                    # wrapper prevents the base adapter from slicing them twice.
                    reference = tensor if isinstance(tensor, DTensor) else None
                    if reference is not None:
                        tensor = reference.to_local()
                        scales = scales.to_local() if isinstance(scales, DTensor) else scales
                    elif device_mesh is not None:
                        start, stop = get_expert_range_for_rank_from_mesh(device_mesh, self.moe_config.n_routed_experts)
                        tensor, scales = tensor[start:stop], scales[start:stop]
                    decoded = torch.empty(tensor.shape, dtype=self.dtype, device=tensor.device)
                    for expert in range(tensor.shape[0]):
                        decoded[expert] = dequantize_from_fp8(
                            tensor[expert], scales[expert], dtype=self.dtype, name=name
                        )
                    if reference is not None:
                        tensor = DTensor.from_local(
                            decoded,
                            reference.device_mesh,
                            reference.placements,
                            shape=reference.shape,
                            stride=reference.stride(),
                        )
                    elif device_mesh is not None:
                        tensor = create_dtensor_from_local(decoded, device_mesh)
                    else:
                        tensor = decoded
                else:
                    tensor = dequantize_from_fp8(tensor, scales, dtype=self.dtype, name=name)
            for source, target in sorted(_INDEXER_NAMES.items(), key=lambda item: len(item[0]), reverse=True):
                name = name.replace(".self_attn." + source, ".self_attn." + target)
            converted[name] = tensor
        for name in list(converted):
            if not name.endswith(".self_attn.q_proj.weight"):
                continue
            prefix = name.removesuffix("q_proj.weight")
            q, k, v = (converted.pop(prefix + part + "_proj.weight") for part in ("q", "k", "v"))
            groups = self.config.num_attention_groups
            heads = self.config.num_attention_heads // groups
            dim, hidden = self.config.head_dim, self.config.hidden_size
            converted[prefix + "qkv_proj.weight"] = torch.cat(
                (
                    q.reshape(groups, heads, dim, hidden),
                    k.reshape(groups, 1, dim, hidden),
                    v.reshape(groups, 1, dim, hidden),
                ),
                dim=1,
            ).reshape(-1, hidden)
        return super().from_hf(converted, device_mesh=device_mesh, **kwargs)

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: torch.Tensor,
        *,
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: ConfigValue,
    ) -> list[tuple[str, torch.Tensor]]:
        """Export native attention and expert tensors to checkpoint names.

        Args:
            fqn: Native tensor name.
            tensor: Tensor with the native layout documented by from_hf.
            exclude_key_regex: Optional exported-name exclusion regex.
            quantization: Whether quantized export is requested (unsupported).
            **kwargs: Checkpointer options forwarded to the base adapter.

        Returns:
            Named source tensors with [out, in] projections or [experts, out, in] experts.
        """
        if quantization:
            raise ValueError("Step5 export writes training weights; FP8 requantization is unsupported")
        if fqn.endswith(".self_attn.qkv_proj.weight"):
            groups = self.config.num_attention_groups
            heads = self.config.num_attention_heads // groups
            dim, hidden = self.config.head_dim, self.config.hidden_size
            if isinstance(tensor, DTensor) and any(
                isinstance(placement, Shard) and placement.dim == 0 and groups % tensor.device_mesh.size(mesh_dim) != 0
                for mesh_dim, placement in enumerate(tensor.placements)
            ):
                # QKV head groups may cross an FSDP shard boundary. Export the
                # small attention projection with replicated group ownership.
                tensor = tensor.redistribute(
                    placements=tuple(
                        Replicate() if isinstance(p, Shard) and p.dim == 0 else p for p in tensor.placements
                    )
                )
            grouped = tensor.reshape(groups, heads + 2, dim, hidden)
            prefix = fqn.removesuffix("qkv_proj.weight")
            result = [
                (prefix + "q_proj.weight", grouped[:, :heads].reshape(-1, hidden)),
                (prefix + "k_proj.weight", grouped[:, heads].reshape(-1, hidden)),
                (prefix + "v_proj.weight", grouped[:, heads + 1].reshape(-1, hidden)),
            ]
        else:
            for source, target in sorted(_INDEXER_NAMES.items(), key=lambda item: len(item[0]), reverse=True):
                fqn = fqn.replace(".self_attn." + target, ".self_attn." + source)
            result = super().convert_single_tensor_to_hf(fqn, tensor, **kwargs)
        return [
            (name, value)
            for name, value in result
            if exclude_key_regex is None or not re.match(exclude_key_regex, name)
        ]
