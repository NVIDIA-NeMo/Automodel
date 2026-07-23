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

"""State-dict adapter for ``diffusion_gemma``.

The HF checkpoint stores all transformer weights under ``model.decoder.*`` (the
"decoder" of the encoder-decoder framing). The encoder's text weights and the
``lm_head`` are tied (and therefore absent from the checkpoint) and are
reconstructed at load time; only ``model.encoder.language_model.layers.{L}.layer_scalar``
buffers are emitted (duplicates of the decoder ``layer_scalar``) and are dropped
here. The native model uses a **single shared stack** at ``model.*`` (no
encoder/decoder split), so the mapping is essentially ``model.decoder.X -> model.X``
plus the Gemma4 MoE expert/router transforms shared with ``gemma4_moe``:

* ``decoder.layers.{L}.experts.gate_up_proj``  [E, 2*inter, hidden]
      -> ``model.layers.{L}.moe.experts.gate_and_up_projs``  [E, hidden, 2*inter]
* ``decoder.layers.{L}.experts.down_proj``     [E, hidden, inter]
      -> ``model.layers.{L}.moe.experts.down_projs``  [E, inter, hidden]
      (with ``router.per_expert_scale`` folded in)
* ``decoder.layers.{L}.router.{proj.weight,scale}`` -> ``model.layers.{L}.moe.gate.{proj.weight,scale}``
* ``decoder.embed_tokens.weight`` -> ``model.embed_tokens.weight`` (also tied to ``lm_head.weight``)
* ``decoder.norm.weight`` -> ``model.norm.weight``
* ``decoder.self_conditioning.*`` -> ``model.self_conditioning.*``

Full-attention layers ({5, 11, 17, 23, 29}) have no ``v_proj`` in the checkpoint;
those keys are simply absent and the model has no ``v_proj`` parameter there, so
no special handling is needed (the pass-through preserves whatever is present).
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.layers import MoEConfig

# HF checkpoint prefix for the shared transformer stack ("decoder" framing).
_DECODER_PREFIX = "model.decoder."
# Native prefix for the single shared stack.
_NATIVE_PREFIX = "model."


class DiffusionGemmaStateDictAdapter(StateDictAdapter):
    """Converts between HF ``diffusion_gemma`` checkpoints and the NeMo layout.

    Handles:
      1. ``model.decoder.*`` -> ``model.*`` re-prefixing (single shared stack).
      2. Expert weight transpose/concat (gate_up_proj + down_proj) and
         ``per_expert_scale`` absorption into ``down_projs``.
      3. Router key remapping (``router.* -> moe.gate.*``).
      4. ``lm_head.weight`` reconstructed from the tied embedding.
      5. Dropping the duplicate encoder ``layer_scalar`` buffers.
      6. Expert-parallel sharding when a device mesh is provided.
    """

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype

    # ------------------------------------------------------------------
    # HF -> NeMo
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        n_experts = self.moe_config.n_routed_experts
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
        else:
            start_expert, end_expert = 0, n_experts
            rank = None

        expert_buffers: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        state_dict: dict[str, Any] = {}
        embed_weight: torch.Tensor | None = None

        for key, value in hf_state_dict.items():
            # Drop duplicate encoder layer_scalar buffers (tied to the decoder).
            if key.startswith("model.encoder."):
                continue

            # Everything else lives under model.decoder.* -> model.*
            if key.startswith(_DECODER_PREFIX):
                native_key = _NATIVE_PREFIX + key[len(_DECODER_PREFIX) :]
            else:
                native_key = key

            # --- Router keys: router.{proj.weight,scale,per_expert_scale} ---
            router_match = re.search(r"(layers\.\d+)\.router\.(proj\.weight|scale|per_expert_scale)$", native_key)
            if router_match:
                layer_path = router_match.group(1)
                router_attr = router_match.group(2)
                if router_attr == "per_expert_scale":
                    expert_buffers[layer_path]["per_expert_scale"] = value
                else:
                    new_key = native_key.replace(
                        f"{layer_path}.router.{router_attr}", f"{layer_path}.moe.gate.{router_attr}"
                    )
                    state_dict[new_key] = value
                continue

            # --- Expert weight keys ---
            expert_match = re.search(r"(layers\.\d+)\.experts\.(gate_up_proj|down_proj)$", native_key)
            if expert_match:
                layer_path = expert_match.group(1)
                expert_buffers[layer_path][expert_match.group(2)] = value
                continue

            if native_key == "model.embed_tokens.weight":
                embed_weight = value

            # --- Pass-through (attn, mlp, norms, layer_scalar, embed, self_cond) ---
            state_dict[native_key] = value

        # A PEFT/LoRA checkpoint carries only adapter tensors (lora_A/lora_B); the
        # frozen base — including the tied embedding/lm_head — is loaded separately
        # from the pretrained model, so an adapter-only state dict has no embedding
        # and nothing to reconstruct. The tied-lm_head rebuild below applies only to
        # a full base checkpoint; skip it (and its guard) for adapter-only loads.
        is_adapter_only = any(".lora_" in k for k in hf_state_dict)

        # Reconstruct the tied lm_head from the embedding. Guard the tied path: if a
        # full checkpoint has no explicit lm_head AND the embedding was not captured
        # at the expected native key, fail loudly rather than silently returning a
        # state_dict missing lm_head/embed (e.g. if a future released checkpoint
        # relocates the embedding key).
        if "lm_head.weight" not in state_dict and not is_adapter_only:
            if embed_weight is None:
                raise RuntimeError(
                    "Cannot reconstruct tied lm_head.weight: the checkpoint has no "
                    "explicit lm_head and the tied embedding was not found at "
                    "'model.embed_tokens.weight'. If the released checkpoint moved the "
                    "embedding, update the captured key in this adapter."
                )
            state_dict["lm_head.weight"] = embed_weight

        # Process collected expert weights per layer.
        _REQUIRED = {"gate_up_proj", "down_proj", "per_expert_scale"}
        for layer_path, tensors in expert_buffers.items():
            missing = _REQUIRED - tensors.keys()
            if missing:
                raise RuntimeError(
                    f"Incomplete expert weights for {layer_path}: missing {missing}. "
                    f"Available keys: {list(tensors.keys())}"
                )

            gate_up_proj = tensors["gate_up_proj"]  # [E, 2*inter, hidden]
            down_proj = tensors["down_proj"]  # [E, hidden, inter]
            per_expert_scale = tensors["per_expert_scale"]  # [E]

            gate_and_up = gate_up_proj.transpose(-2, -1)  # [E, hidden, 2*inter]
            down = self._fold_per_expert_scale(down_proj.transpose(-2, -1), per_expert_scale)  # [E, inter, hidden]

            # Slice to this rank's expert range only under EP (device_mesh set).
            # Under pure FSDP (device_mesh is None) the tensors span all experts —
            # plain [E,...] for a single-tensor load, or a Shard(0) DTensor on the
            # DCP path; in both cases the full-range slice is a no-op we skip to
            # avoid DTensor indexing on the sharded dim.
            if device_mesh is not None:
                gate_and_up = gate_and_up[start_expert:end_expert]
                down = down[start_expert:end_expert]
            gate_and_up_local = gate_and_up.to(self.dtype)
            down_local = down.to(self.dtype)

            if device_mesh is not None and "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_mesh = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                ep_shard_size = ep_shard_mesh.size()
                if ep_shard_size > 1:
                    ep_shard_rank = ep_shard_mesh.get_local_rank()
                    gate_shard_size = gate_and_up_local.shape[1] // ep_shard_size
                    gate_start = ep_shard_rank * gate_shard_size
                    gate_and_up_local = gate_and_up_local[:, gate_start : gate_start + gate_shard_size, :]

                    down_shard_size = down_local.shape[1] // ep_shard_size
                    down_start = ep_shard_rank * down_shard_size
                    down_local = down_local[:, down_start : down_start + down_shard_size, :]

            prefix = f"{_NATIVE_PREFIX}{layer_path}"
            state_dict[f"{prefix}.moe.experts.gate_and_up_projs"] = state_dict_utils.create_dtensor_from_local(
                gate_and_up_local, device_mesh, rank
            )
            state_dict[f"{prefix}.moe.experts.down_projs"] = state_dict_utils.create_dtensor_from_local(
                down_local, device_mesh, rank
            )

        return state_dict

    @staticmethod
    def _fold_per_expert_scale(down: torch.Tensor, per_expert_scale: torch.Tensor) -> torch.Tensor:
        """Fold ``per_expert_scale[E]`` into ``down[E, inter, hidden]`` (scale by row).

        Under pure FSDP the standard DCP load path leaves ``down`` as a ``Shard(0)``
        DTensor (each rank holds ``[E/world, ...]``) while ``per_expert_scale`` is
        loaded replicated as a plain ``[E]`` tensor. A plain ``down * scale[:, None,
        None]`` then broadcasts a global-``[E]`` factor against a local-``[E/world]``
        shard and fails. Wrap the scale as a DTensor replicated on ``down``'s mesh so
        the multiply runs DTensor-vs-DTensor and the scale is sliced per shard.
        """
        if state_dict_utils.is_dtensor(down) and not state_dict_utils.is_dtensor(per_expert_scale):
            from torch.distributed.tensor import DTensor, Replicate

            # Under FSDP CPU-offload the local shard lives on CPU while
            # ``down.device`` reports the mesh (CUDA) device; place ``scale`` on the
            # LOCAL shard's device so the per-shard multiply below stays on one
            # device. Without offload this is the same CUDA device as before.
            local_device = down.to_local().device
            scale = per_expert_scale.to(device=local_device, dtype=down.dtype)
            scale = DTensor.from_local(
                scale,
                down.device_mesh,
                [Replicate()] * down.device_mesh.ndim,
            )
            return down * scale[:, None, None]
        return down * per_expert_scale[:, None, None]

    # ------------------------------------------------------------------
    # NeMo -> HF
    # ------------------------------------------------------------------
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        device_mesh: Optional[DeviceMesh] = kwargs.get("device_mesh")
        n_experts = self.moe_config.n_routed_experts
        hf_state_dict: dict[str, Any] = {}

        for fqn, tensor in state_dict.items():
            # lm_head is tied to the embedding; the checkpoint omits it.
            if fqn == "lm_head.weight":
                continue

            # --- Router keys: moe.gate.{proj.weight,scale} -> router.{...} ---
            gate_match = re.search(r"(layers\.\d+)\.moe\.gate\.(proj\.weight|scale)$", fqn)
            if gate_match:
                layer_path = gate_match.group(1)
                gate_attr = gate_match.group(2)
                hf_key = _DECODER_PREFIX + fqn[len(_NATIVE_PREFIX) :].replace(
                    f"{layer_path}.moe.gate.{gate_attr}", f"{layer_path}.router.{gate_attr}"
                )
                hf_state_dict[hf_key] = tensor
                continue

            # --- Expert: gate_and_up_projs -> experts.gate_up_proj ---
            if ".moe.experts.gate_and_up_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                hf_state_dict[f"{_DECODER_PREFIX}layers.{layer_num}.experts.gate_up_proj"] = global_tensor.transpose(
                    -2, -1
                ).contiguous()
                continue

            # --- Expert: down_projs -> experts.down_proj + router.per_expert_scale ---
            if ".moe.experts.down_projs" in fqn:
                layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
                global_tensor = self._gather_expert_tensor(tensor, device_mesh, n_experts)
                hf_state_dict[f"{_DECODER_PREFIX}layers.{layer_num}.experts.down_proj"] = global_tensor.transpose(
                    -2, -1
                ).contiguous()
                hf_state_dict[f"{_DECODER_PREFIX}layers.{layer_num}.router.per_expert_scale"] = torch.ones(
                    n_experts, dtype=self.dtype
                )
                continue

            # --- Pass-through: model.X -> model.decoder.X ---
            if fqn.startswith(_NATIVE_PREFIX):
                hf_state_dict[_DECODER_PREFIX + fqn[len(_NATIVE_PREFIX) :]] = tensor
            else:
                hf_state_dict[fqn] = tensor

        if exclude_key_regex:
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not re.match(exclude_key_regex, k)}

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor back to HF format (weight-streaming refit).

        Mirrors :meth:`to_hf` per-tensor: router rename, expert transpose +
        ``per_expert_scale`` emission, ``model.X -> model.decoder.X`` re-prefix,
        and dropping the tied ``lm_head.weight``.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if exclude_key_regex and re.match(exclude_key_regex, fqn):
            return []
        if fqn == "lm_head.weight":
            return []

        gate_match = re.search(r"(layers\.\d+)\.moe\.gate\.(proj\.weight|scale)$", fqn)
        if gate_match:
            layer_path = gate_match.group(1)
            gate_attr = gate_match.group(2)
            hf_key = _DECODER_PREFIX + fqn[len(_NATIVE_PREFIX) :].replace(
                f"{layer_path}.moe.gate.{gate_attr}", f"{layer_path}.router.{gate_attr}"
            )
            return [(hf_key, tensor)]

        if ".moe.experts.gate_and_up_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            hf_key = f"{_DECODER_PREFIX}layers.{layer_num}.experts.gate_up_proj"
            return [(hf_key, tensor.transpose(-2, -1).contiguous())]

        if ".moe.experts.down_projs" in fqn:
            layer_num = re.search(r"layers\.(\d+)", fqn).group(1)
            hf_key = f"{_DECODER_PREFIX}layers.{layer_num}.experts.down_proj"
            scale_key = f"{_DECODER_PREFIX}layers.{layer_num}.router.per_expert_scale"
            n_experts = tensor.shape[0]
            return [
                (hf_key, tensor.transpose(-2, -1).contiguous()),
                (scale_key, torch.ones(n_experts, dtype=tensor.dtype)),
            ]

        if fqn.startswith(_NATIVE_PREFIX):
            return [(_DECODER_PREFIX + fqn[len(_NATIVE_PREFIX) :], tensor)]
        return [(fqn, tensor)]

    def _gather_expert_tensor(
        self,
        tensor: torch.Tensor,
        device_mesh: Optional[DeviceMesh],
        n_experts: int,
    ) -> torch.Tensor:
        """Map a stacked expert tensor to its HF (global ``[n_experts, ...]``) form.

        ``device_mesh`` is the MoE/EP mesh and is ``None`` under pure FSDP
        (``ep_size=1``). In that case the tensor is either a plain ``[E, ...]``
        tensor or an FSDP ``Shard(0)`` DTensor whose **global** first dimension
        is already ``n_experts``; both are returned unchanged so DCP sees the
        checkpoint's global expert shape and reads/writes each rank's shard
        itself. Calling ``to_local()`` here would expose the per-rank ``[E/world]``
        shard as the apparent global shape and break the DCP size match
        (``saved [E, ...] vs current [E/world, ...]``). With an EP mesh the
        tensor is gathered across EP ranks into the full ``[n_experts, ...]``.
        """
        if device_mesh is None:
            return tensor

        if state_dict_utils.is_dtensor(tensor):
            split_weights, expert_ids = state_dict_utils.split_experts_weights_dtensor_aware(tensor, n_experts)
            local_weights = [
                (weight.full_tensor() if state_dict_utils.is_dtensor(weight) else weight).to(self.dtype).cpu()
                for weight in split_weights
            ]
        else:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            expert_ids = list(range(start_expert, end_expert))
            local_weights = [tensor[i].to(self.dtype).cpu() for i in range(tensor.shape[0])]

        global_tensor = torch.zeros(
            (n_experts, local_weights[0].shape[0], local_weights[0].shape[1]),
            dtype=self.dtype,
            device="cpu",
        )

        if dist.is_initialized() and "ep" in device_mesh.mesh_dim_names:
            try:
                ep_dim = device_mesh.mesh_dim_names.index("ep")
                ep_group = device_mesh.get_group(ep_dim)
            except Exception:
                ep_group = None

            if ep_group is not None:
                payload = (expert_ids, local_weights)
                gathered: list[tuple[list[int], list[torch.Tensor]]] = [None] * dist.get_world_size(ep_group)
                dist.all_gather_object(gathered, payload, group=ep_group)
                for ids, weights in gathered:
                    for eid, w in zip(ids, weights):
                        global_tensor[eid].copy_(w.to(self.dtype).cpu())
            else:
                for weight, expert_id in zip(local_weights, expert_ids):
                    global_tensor[expert_id].copy_(weight)
        else:
            for weight, expert_id in zip(local_weights, expert_ids):
                global_tensor[expert_id].copy_(weight)

        del local_weights, expert_ids
        return global_tensor
