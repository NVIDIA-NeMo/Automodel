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

"""State-dict adapter between the raw Thinking-Machines Inkling checkpoint and the
native grouped-experts model.

The published ``thinkingmachines/Inkling`` checkpoint stores weights in the raw
SGLang layout (``model.llm.*``, ``attn.wq_du``, ``mlp.experts.w13_weight`` with
*interleaved* gate/up, ``unembed``, ...). This adapter loads that raw checkpoint
**directly**, with no offline pre-conversion step. ``from_hf`` applies the exact
transformers ``inkling_mm_model`` conversion (key renames + de-interleave / chunk
of the fused gate/up, mirroring ``tools/convert_inkling_checkpoint.py``) followed
by the native transpose of the routed experts; ``to_hf`` is its exact inverse and
emits the raw layout again (used both to build DCP load destinations and to write
HF-interop checkpoints).

Three families of tensors need real work; every other key is a pure rename that
passes through unchanged (its tensor is loaded in place):

* **Routed experts** -- raw ``mlp.experts.w13_weight`` ``[E, 2I, H]`` (interleaved
  gate/up) de-interleaves and transposes to native ``gate_and_up_projs``
  ``[E, H, 2I]``; raw ``w2_weight`` ``[E, H, I]`` transposes to ``down_projs``
  ``[E, I, H]``. Sharded on dim 0 over the expert-parallel (EP) mesh. The
  interleave forces a copy, so ``w13`` is NOT loaded through a strided view (only
  the transpose-only ``w2`` uses the writable-view fast path).
* **Shared experts** -- raw ``shared_w13_weight`` ``[S, 2I, H]`` de-interleaves and
  splits (dim 1) into native ``gate_proj`` / ``up_proj`` ``[S, I, H]``. FSDP shards
  dim 0, so the interleave/split on dim 1 is local-safe.
* **Dense MLP** -- raw ``w13_dn.weight`` ``[2I, H]`` de-interleaves and splits
  (dim 0) into native ``gate_proj.weight`` / ``up_proj.weight`` ``[I, H]``. The
  interleave/split is on dim 0, which FSDP may shard, so it is done on the full
  (gathered) tensor and re-sharded by ``set_model_state_dict``.

``from_hf`` also accepts the intermediate HF-*module* layout
(``model.language_model.*.mlp.experts.gate_up_proj``) for backward compatibility
with the unit tests; the input format is auto-detected from the keys.
"""

import re
from typing import Any, Optional

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Replicate, Shard

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe import state_dict_utils
from nemo_automodel.components.moe.config import MoEConfig

# ---------------------------------------------------------------------------
# Raw <-> native key renames. These mirror the transformers ``inkling_mm_model``
# conversion (see ``tools/convert_inkling_checkpoint.py``); the native model reuses
# the HF-module names for everything except the routed experts (which are
# transposed). ``_RAW_TO_NATIVE`` is raw -> native (from_hf); ``_NATIVE_TO_RAW`` is
# its exact inverse (native -> raw, to_hf). Both are ordered sequential ``re.sub``.
# ---------------------------------------------------------------------------
_RAW_TO_NATIVE: list[tuple[str, str]] = [
    (r"model\.llm\.layers", "model.language_model.layers"),
    (r"model\.llm\.embed_norm\.weight", "model.language_model.embed_norm.weight"),
    (r"model\.llm\.embed\.weight", "model.language_model.embed_tokens.weight"),
    (r"model\.llm\.norm\.weight", "model.language_model.norm.weight"),
    (r"model\.llm\.unembed\.weight", "lm_head.weight"),
    (r"model\.audio\.", "model.audio_tower."),
    (r"model\.visual", "model.vision_tower"),
    (r"vision_tower\.layers\.linear_(\d+)", r"vision_tower.encoder_layers.\1.projection"),
    (r"vision_tower\.layers\.norm_(\d+)", r"vision_tower.encoder_layers.\1.layer_norm"),
    (r"audio_tower\.encoder\.weight", "audio_tower.embed_audio_tokens.embed_audio_tokens.weight"),
    (r"audio_tower\.final_norm\.weight", "audio_tower.norm.weight"),
    (r"mlp\.experts\.w2_weight", "mlp.experts.down_proj"),
    (r"shared_w2_weight", "down_proj"),
    (r"mlp\.w2_md\.weight", "mlp.down_proj.weight"),
    (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias"),
    (r"attn\.wq_du", "self_attn.q_proj"),
    (r"attn\.wk_dv", "self_attn.k_proj"),
    (r"attn\.wv_dv", "self_attn.v_proj"),
    (r"attn\.wr_du", "self_attn.r_proj"),
    (r"attn\.wo_ud", "self_attn.o_proj"),
    (r"\.attn\.q_norm", ".self_attn.q_norm"),
    (r"\.attn\.k_norm", ".self_attn.k_norm"),
    (r"\.attn\.k_sconv", ".self_attn.k_sconv.conv1d"),
    (r"\.attn\.v_sconv", ".self_attn.v_sconv.conv1d"),
    (r"\.attn\.rel_logits_proj", ".self_attn.rel_logits_proj"),
    (r"attn_sconv\.weight$", "attn_sconv.conv1d.weight"),
    (r"mlp_sconv\.weight$", "mlp_sconv.conv1d.weight"),
    (r"mlp_norm", "post_attention_layernorm"),
    (r"attn_norm", "input_layernorm"),
]

# Inverse of ``_RAW_TO_NATIVE`` (native -> raw), applied in reverse order.
_NATIVE_TO_RAW: list[tuple[str, str]] = [
    (r"post_attention_layernorm", "mlp_norm"),
    (r"input_layernorm", "attn_norm"),
    (r"mlp_sconv\.conv1d\.weight$", "mlp_sconv.weight"),
    (r"attn_sconv\.conv1d\.weight$", "attn_sconv.weight"),
    (r"\.self_attn\.rel_logits_proj", ".attn.rel_logits_proj"),
    (r"\.self_attn\.v_sconv\.conv1d", ".attn.v_sconv"),
    (r"\.self_attn\.k_sconv\.conv1d", ".attn.k_sconv"),
    (r"\.self_attn\.k_norm", ".attn.k_norm"),
    (r"\.self_attn\.q_norm", ".attn.q_norm"),
    (r"self_attn\.o_proj", "attn.wo_ud"),
    (r"self_attn\.r_proj", "attn.wr_du"),
    (r"self_attn\.v_proj", "attn.wv_dv"),
    (r"self_attn\.k_proj", "attn.wk_dv"),
    (r"self_attn\.q_proj", "attn.wq_du"),
    (r"mlp\.gate\.e_score_correction_bias", "mlp.gate.bias"),
    (r"\.mlp\.down_proj\.weight", ".mlp.w2_md.weight"),
    (r"\.mlp\.shared_experts\.down_proj$", ".mlp.shared_experts.shared_w2_weight"),
    (r"audio_tower\.embed_audio_tokens\.embed_audio_tokens\.weight", "audio_tower.encoder.weight"),
    (r"audio_tower\.norm\.weight", "audio_tower.final_norm.weight"),
    (r"vision_tower\.encoder_layers\.(\d+)\.projection", r"vision_tower.layers.linear_\1"),
    (r"vision_tower\.encoder_layers\.(\d+)\.layer_norm", r"vision_tower.layers.norm_\1"),
    (r"model\.vision_tower", "model.visual"),
    (r"model\.audio_tower\.", "model.audio."),
    (r"model\.language_model\.layers", "model.llm.layers"),
    (r"model\.language_model\.embed_norm\.weight", "model.llm.embed_norm.weight"),
    (r"model\.language_model\.embed_tokens\.weight", "model.llm.embed.weight"),
    (r"model\.language_model\.norm\.weight", "model.llm.norm.weight"),
    (r"lm_head\.weight", "model.llm.unembed.weight"),
]

# HF-module routed-expert keys (backward-compat ``from_hf`` input).
_HF_EXPERT_RE = re.compile(r"(?:model\.)?language_model\.layers\.(\d+)\.mlp\.experts\.(gate_up_proj|down_proj)$")


def _apply_renames(key: str, rules: list[tuple[str, str]]) -> str:
    """Apply an ordered list of ``re.sub`` renames to ``key``."""
    for pat, repl in rules:
        key = re.sub(pat, repl, key)
    return key


def _deinterleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Raw interleaved gate/up ``[g0, u0, g1, u1, ...]`` -> contiguous ``[g.., u..]``.

    This is the transformers ``Interleave`` op (see ``tools/convert_inkling_checkpoint.py``).
    """
    shape = list(t.shape)
    shape[dim : dim + 1] = [shape[dim] // 2, 2]
    return t.reshape(shape).transpose(dim, dim + 1).reshape(t.shape).contiguous()


def _interleave(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Inverse of :func:`_deinterleave`: contiguous ``[g.., u..]`` -> raw ``[g0, u0, ...]``."""
    shape = list(t.shape)
    shape[dim : dim + 1] = [2, shape[dim] // 2]
    return t.reshape(shape).transpose(dim, dim + 1).reshape(t.shape).contiguous()


def _map_placements(placements, swap: tuple[int, int]):
    """Copy ``placements``, swapping the two ``Shard`` dims named in ``swap``."""
    a, b = swap
    out = []
    for p in placements:
        if isinstance(p, Shard) and p.dim == a:
            out.append(Shard(b))
        elif isinstance(p, Shard) and p.dim == b:
            out.append(Shard(a))
        else:
            out.append(p)
    return tuple(out)


class InklingStateDictAdapter(StateDictAdapter):
    """Convert Inkling weights between the raw checkpoint layout and the native model."""

    def __init__(
        self,
        config: Any,
        moe_config: MoEConfig,
        backend: BackendConfig,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.moe_config = moe_config
        self.backend = backend
        self.dtype = dtype
        self._uses_model_prefix = True
        # Native expert keys loaded in-place via strided views during the most recent
        # to_hf/DCP load (only the transpose-only routed ``down_projs``). Read by the
        # checkpoint loader (see checkpointing.load_model) so they are not counted as
        # "missing".
        self._view_loaded_native_keys: set[str] = set()

    @property
    def view_loaded_native_keys(self) -> set[str]:
        """Native keys loaded in-place through transposed views during the last to_hf."""
        return self._view_loaded_native_keys

    # ------------------------------------------------------------------
    # native -> raw (to_hf)
    # ------------------------------------------------------------------
    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a native state dict to the raw Inkling checkpoint layout.

        Routed experts, shared experts and the dense MLP are re-interleaved (and
        combined, for the split gate/up params) into their fused raw tensors; every
        other key is a pure rename. Used both to build DCP load destinations (which
        DCP overwrites with on-disk data) and to write raw HF-interop checkpoints.
        """
        self._view_loaded_native_keys = set()
        hf_state_dict: dict[str, Any] = {}
        # Buffer the split gate/up params so the two native tensors can be combined
        # into one fused raw tensor once both are present.
        pending_gate_up: dict[str, dict[str, Any]] = {}

        for fqn, tensor in state_dict.items():
            if fqn.endswith(".mlp.experts.gate_and_up_projs"):
                raw_key = fqn[: -len(".mlp.experts.gate_and_up_projs")] + ".mlp.experts.w13_weight"
                raw_key = _apply_renames(raw_key, _NATIVE_TO_RAW)
                hf_state_dict[raw_key] = self._routed_gateup_to_raw(tensor)
                continue
            if fqn.endswith(".mlp.experts.down_projs"):
                raw_key = fqn[: -len(".mlp.experts.down_projs")] + ".mlp.experts.w2_weight"
                raw_key = _apply_renames(raw_key, _NATIVE_TO_RAW)
                hf_state_dict[raw_key] = self._transpose_for_hf(fqn, tensor)
                continue
            if fqn.endswith(".mlp.shared_experts.gate_proj"):
                base = fqn[: -len(".gate_proj")]
                pending_gate_up.setdefault(base, {"kind": "shared"})["gate"] = tensor
                continue
            if fqn.endswith(".mlp.shared_experts.up_proj"):
                base = fqn[: -len(".up_proj")]
                pending_gate_up.setdefault(base, {"kind": "shared"})["up"] = tensor
                continue
            if fqn.endswith(".mlp.gate_proj.weight"):
                base = fqn[: -len(".gate_proj.weight")]
                pending_gate_up.setdefault(base, {"kind": "dense"})["gate"] = tensor
                continue
            if fqn.endswith(".mlp.up_proj.weight"):
                base = fqn[: -len(".up_proj.weight")]
                pending_gate_up.setdefault(base, {"kind": "dense"})["up"] = tensor
                continue

            raw_key = _apply_renames(fqn, _NATIVE_TO_RAW)
            if exclude_key_regex and re.match(exclude_key_regex, raw_key):
                continue
            hf_state_dict[raw_key] = tensor

        for base, parts in pending_gate_up.items():
            if parts["kind"] == "shared":
                raw_key = _apply_renames(base + ".shared_w13_weight", _NATIVE_TO_RAW)
                hf_state_dict[raw_key] = self._combine_gate_up_to_raw(parts["gate"], parts["up"], fused_dim=1)
            else:
                raw_key = _apply_renames(base + ".w13_dn.weight", _NATIVE_TO_RAW)
                hf_state_dict[raw_key] = self._combine_gate_up_to_raw(parts["gate"], parts["up"], fused_dim=0)

        return hf_state_dict

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        """Convert a single native tensor to raw layout (1:1 keys only).

        Handles the routed experts and every pure-rename key. The fused shared/dense
        gate/up params are combine-only (two native tensors -> one raw tensor) and
        cannot be produced per-tensor, so they must go through :meth:`to_hf`; this
        method raises for them.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        if fqn.endswith(".mlp.experts.gate_and_up_projs"):
            raw_key = _apply_renames(
                fqn[: -len(".mlp.experts.gate_and_up_projs")] + ".mlp.experts.w13_weight", _NATIVE_TO_RAW
            )
            return [(raw_key, self._routed_gateup_to_raw(tensor))]
        if fqn.endswith(".mlp.experts.down_projs"):
            raw_key = _apply_renames(fqn[: -len(".mlp.experts.down_projs")] + ".mlp.experts.w2_weight", _NATIVE_TO_RAW)
            return [(raw_key, self._transpose_for_hf(fqn, tensor))]
        if fqn.endswith((".mlp.shared_experts.gate_proj", ".mlp.shared_experts.up_proj")) or (
            fqn.endswith((".mlp.gate_proj.weight", ".mlp.up_proj.weight"))
        ):
            raise NotImplementedError(
                f"{fqn} is a combine-only fused gate/up param; use to_hf() with the full state dict."
            )
        raw_key = _apply_renames(fqn, _NATIVE_TO_RAW)
        if exclude_key_regex and re.match(exclude_key_regex, raw_key):
            return []
        return [(raw_key, tensor)]

    def _routed_gateup_to_raw(self, tensor: Any) -> Any:
        """Native routed ``gate_and_up_projs`` ``[E, H, 2I]`` -> raw ``w13_weight`` ``[E, 2I, H]``.

        Transpose to the module gate/up layout, then re-interleave gate/up on dim 1
        (a copy). The interleave dim is never the sharded dim, so DTensors are
        transformed on their local shard and re-wrapped.
        """
        if state_dict_utils.is_dtensor(tensor):
            local = tensor.to_local()  # [E_loc, H_loc, 2I]
            local = _interleave(local.transpose(-1, -2).contiguous(), dim=1)  # [E_loc, 2I, H_loc]
            raw_placements = _map_placements(tensor.placements, swap=(1, 2))
            return DTensor.from_local(local, tensor.device_mesh, raw_placements)
        return _interleave(tensor.transpose(-1, -2).contiguous(), dim=1)

    def _combine_gate_up_to_raw(self, gate: Any, up: Any, fused_dim: int) -> Any:
        """Combine native ``gate_proj`` / ``up_proj`` into one fused raw tensor.

        ``fused_dim`` is the axis the gate/up interleave lives on (1 for shared
        experts, 0 for the dense MLP). The shared/dense gate/up are tiny (n_shared=2,
        two dense layers) and FSDP-sharded on dim 0, which is *degenerate* when dim 0
        (size 2) is spread over many ranks -- a local-shard + ``from_local`` rebuild
        then infers a wrong global shape (e.g. ``[mesh_size, ...]`` instead of the
        true ``[2, ...]``). The raw checkpoint stores these unsharded, so gather the
        true full tensor and return it **replicated** on the mesh: DCP then loads the
        full checkpoint tensor onto every rank correctly, regardless of native
        sharding.
        """
        if not state_dict_utils.is_dtensor(gate):
            return _interleave(torch.cat([gate, up], dim=fused_dim), dim=fused_dim)

        gate_f, up_f = gate.full_tensor(), up.full_tensor()
        combined = _interleave(torch.cat([gate_f, up_f], dim=fused_dim), dim=fused_dim)
        return DTensor.from_local(combined, gate.device_mesh, tuple(Replicate() for _ in gate.placements))

    def _transpose_for_hf(self, native_fqn: str, tensor: Any) -> Any:
        """Transpose-only routed ``down_projs`` -> raw ``w2_weight`` (writable strided view).

        Returns a non-contiguous view aliasing model storage for DTensor / on-device
        tensors so DCP writes through it in place; otherwise a materialized copy.
        """
        is_dtensor = state_dict_utils.is_dtensor(tensor)
        inplace = is_dtensor or (isinstance(tensor, torch.Tensor) and tensor.is_cuda and not tensor.is_meta)
        if inplace:
            self._view_loaded_native_keys.add(native_fqn)
            return tensor.transpose(-1, -2)
        return tensor.transpose(-1, -2).contiguous()

    # ------------------------------------------------------------------
    # raw -> native (from_hf)
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[DeviceMesh] = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Convert a raw (or HF-module) checkpoint state dict to native layout.

        The input format is auto-detected: raw keys (``model.llm.*``, ``w13_weight``,
        ``wq_du``) take the full conversion path; HF-module keys
        (``model.language_model.*``, ``gate_up_proj``) take the legacy transpose-only
        path (used by the unit tests).
        """
        is_raw = any(key.startswith("model.llm.") or "w13_weight" in key or ".wq_du" in key for key in hf_state_dict)
        if not is_raw:
            return self._from_hf_module(hf_state_dict, device_mesh)
        return self._from_hf_raw(hf_state_dict, device_mesh)

    def _expert_ranges(self, device_mesh: Optional[DeviceMesh]):
        """Return ``(start_expert, end_expert, rank, ep_shard_rank, ep_shard_size)``."""
        n_experts = self.moe_config.n_routed_experts
        start_expert, end_expert, rank = 0, n_experts, None
        ep_shard_rank, ep_shard_size = 0, 1
        if device_mesh is not None:
            start_expert, end_expert = state_dict_utils.get_expert_range_for_rank_from_mesh(device_mesh, n_experts)
            rank = (
                state_dict_utils.get_submesh(device_mesh, ("ep",)).get_rank()
                if "ep" in device_mesh.mesh_dim_names
                else device_mesh.get_rank()
            )
            if "ep_shard" in device_mesh.mesh_dim_names:
                ep_shard_sub = state_dict_utils.get_submesh(device_mesh, ("ep_shard",))
                if ep_shard_sub.size() > 1:
                    ep_shard_rank = ep_shard_sub.get_local_rank()
                    ep_shard_size = ep_shard_sub.size()
        return start_expert, end_expert, rank, ep_shard_rank, ep_shard_size

    def _from_hf_raw(self, hf_state_dict: dict[str, Any], device_mesh: Optional[DeviceMesh]) -> dict[str, Any]:
        start_expert, end_expert, rank, ep_shard_rank, ep_shard_size = self._expert_ranges(device_mesh)

        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key.startswith("model.mtp."):
                continue  # multi-token-prediction head is not instantiated
            if key.endswith("_scale_inv"):
                continue

            native_key = _apply_renames(key, _RAW_TO_NATIVE)

            if key.endswith(".mlp.experts.w13_weight"):
                nk = native_key.replace(".mlp.experts.w13_weight", ".mlp.experts.gate_and_up_projs")
                state_dict[nk] = self._raw_w13_to_routed(
                    value, start_expert, end_expert, ep_shard_rank, ep_shard_size, rank, device_mesh
                )
                continue
            if key.endswith(".mlp.experts.w2_weight"):
                nk = native_key.replace(".mlp.experts.down_proj", ".mlp.experts.down_projs")
                if state_dict_utils.is_dtensor(value):
                    # Loaded in place through the to_hf transposed view; nothing to emit.
                    self._view_loaded_native_keys.add(nk)
                    continue
                state_dict[nk] = self._raw_w2_to_routed(
                    value, start_expert, end_expert, ep_shard_rank, ep_shard_size, rank, device_mesh
                )
                continue
            if key.endswith(".mlp.shared_experts.shared_w13_weight"):
                base = native_key.replace(".mlp.shared_experts.shared_w13_weight", ".mlp.shared_experts.")
                gate, up = self._raw_w13_split(value, fused_dim=1)
                state_dict[base + "gate_proj"] = gate
                state_dict[base + "up_proj"] = up
                continue
            if key.endswith(".mlp.w13_dn.weight"):
                base = native_key.replace(".mlp.w13_dn.weight", ".mlp.")
                gate, up = self._raw_w13_split(value, fused_dim=0)
                state_dict[base + "gate_proj.weight"] = gate
                state_dict[base + "up_proj.weight"] = up
                continue

            state_dict[native_key] = value

        return state_dict

    def _raw_w13_to_routed(self, value, start_expert, end_expert, ep_shard_rank, ep_shard_size, rank, device_mesh):
        """Raw ``w13_weight`` ``[E, 2I, H]`` -> native ``gate_and_up_projs`` ``[E, H, 2I]``."""
        if state_dict_utils.is_dtensor(value):
            local = value.to_local()  # [E_loc, 2I, H_loc]
            native_local = _deinterleave(local, dim=1).transpose(-1, -2).contiguous()  # [E_loc, H_loc, 2I]
            native_placements = _map_placements(value.placements, swap=(1, 2))
            return DTensor.from_local(native_local, value.device_mesh, native_placements)

        native = _deinterleave(value, dim=1).transpose(-1, -2).contiguous().to(self.dtype)
        native = native[start_expert:end_expert].contiguous()
        if ep_shard_size > 1:
            assert native.shape[1] % ep_shard_size == 0
            chunk = native.shape[1] // ep_shard_size
            native = native[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :].contiguous()
        if device_mesh is not None:
            native = state_dict_utils.create_dtensor_from_local(native, device_mesh, rank)
        return native

    def _raw_w2_to_routed(self, value, start_expert, end_expert, ep_shard_rank, ep_shard_size, rank, device_mesh):
        """Raw ``w2_weight`` ``[E, H, I]`` -> native ``down_projs`` ``[E, I, H]`` (plain path)."""
        native = value.transpose(-1, -2).contiguous().to(self.dtype)
        native = native[start_expert:end_expert].contiguous()
        if ep_shard_size > 1:
            assert native.shape[1] % ep_shard_size == 0
            chunk = native.shape[1] // ep_shard_size
            native = native[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :].contiguous()
        if device_mesh is not None:
            native = state_dict_utils.create_dtensor_from_local(native, device_mesh, rank)
        return native

    def _raw_w13_split(self, value, fused_dim: int):
        """De-interleave a fused shared/dense ``w13`` tensor and split into gate/up.

        Symmetric with :meth:`_combine_gate_up_to_raw`: the ``to_hf`` destination for
        these tiny tensors is replicated, so gather the full tensor, split, and return
        the halves **replicated**. ``set_model_state_dict`` then re-shards each to the
        native (FSDP) placement. Gathering avoids any degenerate local-shard rebuild
        when dim 0 (size 2) is spread over many ranks.
        """
        if not state_dict_utils.is_dtensor(value):
            gate, up = _deinterleave(value, dim=fused_dim).chunk(2, dim=fused_dim)
            return gate.contiguous(), up.contiguous()

        full = value.full_tensor()
        gate, up = _deinterleave(full, dim=fused_dim).chunk(2, dim=fused_dim)
        repl = tuple(Replicate() for _ in value.placements)
        return (
            DTensor.from_local(gate.contiguous(), value.device_mesh, repl),
            DTensor.from_local(up.contiguous(), value.device_mesh, repl),
        )

    def _from_hf_module(self, hf_state_dict: dict[str, Any], device_mesh: Optional[DeviceMesh]) -> dict[str, Any]:
        """Legacy path: HF-module keys -> native (transpose routed experts only).

        Retained for the unit tests, which build ``hf.state_dict()`` (module layout).
        """
        self._uses_model_prefix = any(key.startswith("model.") for key in hf_state_dict)
        model_prefix = "model." if self._uses_model_prefix else ""
        start_expert, end_expert, rank, ep_shard_rank, ep_shard_size = self._expert_ranges(device_mesh)

        state_dict: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            match = _HF_EXPERT_RE.match(key)
            if match:
                layer_num, which = match.group(1), match.group(2)
                native_key = f"{model_prefix}language_model.layers.{layer_num}.mlp.experts."
                native_key += "gate_and_up_projs" if which == "gate_up_proj" else "down_projs"

                if state_dict_utils.is_dtensor(value):
                    self._view_loaded_native_keys.add(native_key)
                    continue

                local_tensor = value.transpose(-1, -2).contiguous().to(self.dtype)
                local_tensor = local_tensor[start_expert:end_expert].contiguous()
                if ep_shard_size > 1:
                    assert local_tensor.shape[1] % ep_shard_size == 0
                    chunk = local_tensor.shape[1] // ep_shard_size
                    local_tensor = local_tensor[:, ep_shard_rank * chunk : (ep_shard_rank + 1) * chunk, :]
                if device_mesh is not None:
                    local_tensor = state_dict_utils.create_dtensor_from_local(local_tensor, device_mesh, rank)
                state_dict[native_key] = local_tensor
                continue

            if key.endswith("_scale_inv"):
                continue

            if key.startswith("model.") or key.startswith("lm_head."):
                state_dict[key] = value
            else:
                state_dict[f"{model_prefix}{key}"] = value

        return state_dict
