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

"""Whole-state, unsharded checkpoint layout for one-depth ERNIE 5.5 MTP.

The decoder conversion is delegated to its existing adapter. Shared MTP body
keys are export aliases only; native storage has no duplicate body parameters.
Low-memory/in-place DCP loading and streaming consolidation are not established
by this whole-state adapter. Native state_dict save/restore does not require this adapter.
"""

from __future__ import annotations

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config, mtp_decoder_config
from nemo_automodel.components.models.ernie5_5.state_dict_adapter import Ernie5_5StateDictAdapter

_OWNED = {
    "enorm.weight": "enorm.weight",
    "hnorm.weight": "hnorm.weight",
    "e_proj.weight": "e_proj.weight",
    "h_proj.weight": "h_proj.weight",
    "hyper_head.hc_fn": "hc_head_fn",
    "hyper_head.hc_base": "hc_head_base",
    "hyper_head.hc_scale": "hc_head_scale",
    "final_norm.weight": "shared_head.norm.weight",
    "router_state.e_score_correction_bias": "block_sparse_moe.e_score_correction_bias",
}


def _mtp_body_suffix(decoder_suffix: str) -> str:
    """Use the released extra nesting for the ten shared hyper-connection tensors."""
    if decoder_suffix.startswith(("mlp_hyper_connection.", "self_attention_hyper_connection.")):
        return "transformer_layer." + decoder_suffix
    return decoder_suffix


class Ernie5_5MTPStateDictAdapter(StateDictAdapter):
    """Map eight owned parameters, one bias and shared export aliases exactly."""

    _supports_low_memory_dcp_load = False

    def __init__(self, config: Ernie5_5Config) -> None:
        self.decoder = Ernie5_5StateDictAdapter(mtp_decoder_config(config))
        self.mtp_prefix = f"model.layers.{config.num_hidden_layers}."
        self.body_prefix = f"model.layers.{config.num_hidden_layers - 1}."
        self.bias_suffix = "block_sparse_moe.e_score_correction_bias"

    def to_hf(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        quantization: bool = False,
        for_checkpoint_load: bool = False,
        v4_compatible: bool = False,
        **kwargs: str | bool | DeviceMesh | None,
    ) -> dict[str, torch.Tensor]:
        """Export the whole native state without copying shared body storage.

        Args:
            state_dict: Unsharded native tensors in their registered layouts;
                decoder tensors have model.* names and MTP-owned tensors mtp.*.
                mHC head projection is [streams, streams * hidden] natively.
            quantization: Must be False.
            for_checkpoint_load: Must be False; in-place load sessions are not supported.
            v4_compatible: Must be False.
            **kwargs: Checkpointer compatibility arguments passed to the decoder adapter.

        Returns:
            HF tensor layouts, including [streams * hidden, streams] MTP head
            projection. Shared body values alias the decoder export; other views
            follow its adapter's contract. No second body is constructed.

        Raises:
            NotImplementedError: If load destinations, quantization or sharded tensors are requested.
            ValueError: If the eight owned parameters and persistent router bias are not present exactly once.
        """
        if quantization or for_checkpoint_load or v4_compatible:
            raise NotImplementedError("Whole-state MTP export does not support load destinations or quantization")
        if any(isinstance(value, DTensor) for value in state_dict.values()):
            raise NotImplementedError("Whole-state MTP HF export requires consolidated native tensors")
        owned = {name[4:]: value for name, value in state_dict.items() if name.startswith("mtp.")}
        if owned.keys() != _OWNED.keys():
            raise ValueError("MTP native state must contain exactly eight owned parameters and one bias")
        result = self.decoder.to_hf(
            {name: value for name, value in state_dict.items() if not name.startswith("mtp.")}, **kwargs
        )
        for key, value in list(result.items()):
            if key.startswith(self.body_prefix) and key != self.body_prefix + self.bias_suffix:
                result[self.mtp_prefix + _mtp_body_suffix(key[len(self.body_prefix) :])] = value
        for name, suffix in _OWNED.items():
            value = owned[name]
            result[self.mtp_prefix + suffix] = value.T if name == "hyper_head.hc_fn" else value
        return result

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        device_mesh: DeviceMesh | None = None,
        **kwargs: str | bool | None,
    ) -> dict[str, torch.Tensor]:
        """Convert a complete HF mapping, refusing conflicting shared copies.

        Args:
            hf_state_dict: Whole unsharded HF tensors; decoder keys and MTP body
                keys have matching shapes/dtypes/values. The owned head matrix
                has HF shape [streams * hidden, streams]. Meta inputs are unsupported.
            device_mesh: Must be None for this whole-state conversion.
            **kwargs: Compatibility options forwarded to the decoder adapter.

        Returns:
            Unique native tensors in registered layouts; the owned head matrix
            is transposed to [streams, streams * hidden]. load_state_dict with
            assign=False preserves the existing registered owners.

        Raises:
            NotImplementedError: If the HF mapping contains sharded tensors.
            ValueError: If required keys are missing, layouts are invalid or shared copies differ from their owners.
        """
        if device_mesh is not None or any(isinstance(value, DTensor) for value in hf_state_dict.values()):
            raise NotImplementedError("Whole-state MTP HF import requires unsharded tensors")
        owned_keys = {self.mtp_prefix + suffix for suffix in _OWNED.values()}
        decoder = {key: value for key, value in hf_state_dict.items() if not key.startswith(self.mtp_prefix)}
        shared = {
            self.mtp_prefix + _mtp_body_suffix(key[len(self.body_prefix) :]): key
            for key in decoder
            if key.startswith(self.body_prefix) and key != self.body_prefix + self.bias_suffix
        }
        actual = {key for key in hf_state_dict if key.startswith(self.mtp_prefix)}
        if actual != owned_keys | shared.keys():
            raise ValueError("MTP HF keys do not exactly cover the shared body and nine owned tensors")
        for target, source in shared.items():
            left, right = hf_state_dict[target], decoder[source]
            if left.shape != right.shape or left.dtype != right.dtype or not torch.equal(left, right):
                raise ValueError(f"MTP shared tensor differs from its decoder owner: {target}")
        result = self.decoder.from_hf(decoder, **kwargs)
        for name, suffix in _OWNED.items():
            value = hf_state_dict[self.mtp_prefix + suffix]
            result["mtp." + name] = value.T.contiguous() if name == "hyper_head.hc_fn" else value
        return result

    def convert_single_tensor_to_hf(
        self, fqn: str, tensor: torch.Tensor, **kwargs: str | bool | None
    ) -> list[tuple[str, torch.Tensor]]:
        """Reject streaming export until its full alias/decoder contract is implemented.

        Args:
            fqn: Native tensor name.
            tensor: Tensor in the native parameter's semantic layout.
            **kwargs: Framework compatibility options.

        Returns:
            No result; this adapter supports whole-state conversion only.

        Raises:
            NotImplementedError: Single-tensor conversion cannot construct the complete shared-body HF layout.
        """
        raise NotImplementedError("Use whole-state to_hf for the unsharded MTP adapter")
