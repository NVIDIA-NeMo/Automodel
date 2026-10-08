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

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.state_dict_adapter import dequantize_from_fp8
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.models.step5.configuration import ConfigValue, Step5Config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.state_dict_utils import (
    create_dtensor_from_local,
    get_expert_range_for_rank_from_mesh,
)

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
