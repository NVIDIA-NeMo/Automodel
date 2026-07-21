# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

import re
from typing import Any

import torch
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step3p5.state_dict_adapter import Step3p5StateDictAdapter
from nemo_automodel.components.moe.config import MoEConfig


def _mtp_layer_range(config: Any) -> tuple[int, int]:
    text_config = getattr(config, "text_config", config)
    start = int(getattr(text_config, "mtp_base_layer_idx", getattr(text_config, "num_hidden_layers", 0)) or 0)
    depth = int(getattr(text_config, "num_nextn_predict_layers", 0) or 0)
    return start, start + depth


class Step3p7StateDictAdapter(StateDictAdapter):
    """Adapter for Step3.7 VLM checkpoints.

    The released checkpoint stores the Step3.5 language backbone at top-level
    keys such as ``model.layers.*`` and stores vision keys as
    ``vision_model.*`` / ``vit_large_projector.*``.  The native AutoModel VLM
    keeps the language backbone under ``model.language_model`` so PP can split
    it as a nested text module, and reuses the Step3p5 expert-weight adapter for
    EP sharding.
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.config = config
        self.text_adapter = Step3p5StateDictAdapter(config.text_config, moe_config, backend, dtype=dtype)

    @staticmethod
    def _is_text_key(key: str) -> bool:
        text_prefixes = (
            "model.embed_tokens.",
            "model.layers.",
            "model.norm.",
            "model.language_model.",
            "language_model.",
        )
        return key.startswith(text_prefixes)

    @staticmethod
    def _to_text_hf_key(key: str) -> str:
        if key.startswith("model.language_model."):
            return "model." + key[len("model.language_model.") :]
        if key.startswith("language_model."):
            return "model." + key[len("language_model.") :]
        return key

    @staticmethod
    def _to_native_text_key(key: str) -> str:
        if key.startswith("mtp."):
            return key
        if key.startswith("model."):
            return "model.language_model." + key[len("model.") :]
        if key.startswith("language_model."):
            return "model." + key
        return "model.language_model." + key

    @staticmethod
    def _map_non_text_from_hf(key: str) -> str | None:
        if key.endswith(".weight_scale_inv"):
            # FP8 scale tensors are only meaningful with the FP8 checkpoint path.
            # The full-training recipe uses BF16 weights, so ignore these extras
            # instead of leaving unexpected tensors in the native state dict.
            return None
        if key.startswith("vision_model."):
            return "model." + key
        if key.startswith("vit_large_projector."):
            return "model." + key
        return key

    @staticmethod
    def _map_non_text_to_hf(key: str) -> str:
        if key.startswith("model.vision_model."):
            return key[len("model.") :]
        if key.startswith("model.vit_large_projector."):
            return key[len("model.") :]
        return key

    def _map_mtp_from_hf(self, key: str) -> str | None:
        match = re.match(r"(?:(?:model\.)?(?:language_model\.)?)layers\.(?P<layer>\d+)\.(?P<suffix>.+)", key)
        if match is None:
            return None

        mtp_start, mtp_end = _mtp_layer_range(self.config)
        layer_idx = int(match.group("layer"))
        if mtp_start <= layer_idx < mtp_end:
            depth = layer_idx - mtp_start
            return f"mtp.layers.{depth}.{match.group('suffix')}"
        return None

    def _map_mtp_to_hf(self, key: str) -> str | None:
        match = re.match(r"mtp\.layers\.(?P<depth>\d+)\.(?P<suffix>.+)", key)
        if match is None:
            return None

        mtp_start, _ = _mtp_layer_range(self.config)
        layer_idx = mtp_start + int(match.group("depth"))
        return f"model.layers.{layer_idx}.{match.group('suffix')}"

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: DeviceMesh | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        text_hf: dict[str, Any] = {}
        native: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            mtp_key = self._map_mtp_from_hf(key)
            if mtp_key is not None:
                native[mtp_key] = value
                continue

            if self._is_text_key(key):
                text_hf[self._to_text_hf_key(key)] = value
                continue

            mapped_key = self._map_non_text_from_hf(key)
            if mapped_key is not None:
                native[mapped_key] = value

        text_native = self.text_adapter.from_hf(text_hf, device_mesh=device_mesh, **kwargs)
        for key, value in text_native.items():
            native[self._to_native_text_key(key)] = value

        return native

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: str | None = None,
        quantization: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        hf_state_dict: dict[str, Any] = {}
        for key, tensor in state_dict.items():
            for hf_key, hf_tensor in self.convert_single_tensor_to_hf(
                key,
                tensor,
                exclude_key_regex=exclude_key_regex,
                quantization=quantization,
                **kwargs,
            ):
                hf_state_dict[hf_key] = hf_tensor
        return hf_state_dict

    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: Any,
        **kwargs: Any,
    ) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex")
        mtp_key = self._map_mtp_to_hf(fqn)
        if mtp_key is not None:
            if exclude_key_regex and re.match(exclude_key_regex, mtp_key):
                return []
            return [(mtp_key, tensor)]
        if fqn.startswith("model.language_model."):
            text_key = "model." + fqn[len("model.language_model.") :]
            return self.text_adapter.convert_single_tensor_to_hf(text_key, tensor, **kwargs)

        hf_key = self._map_non_text_to_hf(fqn)
        if exclude_key_regex and re.match(exclude_key_regex, hf_key):
            return []
        return [(hf_key, tensor)]
