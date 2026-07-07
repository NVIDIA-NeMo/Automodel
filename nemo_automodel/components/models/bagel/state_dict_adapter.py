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

"""State-dict adapter for BAGEL HF checkpoints.

BAGEL-7B-MoT ships two on-disk files with complementary key namespaces:

* ``ema.safetensors`` — everything under the top-level module tree, both the
  **UND** (understanding) path and the **GEN** (``*_moe_gen``) Mixture-of-
  Transformers siblings plus the flow-matching scaffolding
  (``time_embedder``, ``vae2llm``, ``llm2vae``, ``latent_pos_embed``).
* ``ae.safetensors`` — the VAE encoder/decoder weights. These live in a
  *separate* file and are loaded by the Stage 2 recipe via upstream
  ``load_ae``; they are not parameters of ``BagelForUnifiedMultimodal``.

Special cases:

* ``vit_pos_embed.pos_embed`` *is* present in the checkpoint. Upstream BAGEL
  declares it as a frozen ``nn.Parameter(requires_grad=False)`` (sinusoidal
  2D position embedding) — so it serializes like any other parameter. It is
  classified as UND since it feeds the understanding-side connector.
* The released BAGEL-7B-MoT checkpoint stores
  ``vit_model.vision_model.embeddings.patch_embedding.weight`` in the
  post-conversion linear layout ``(out_channels, in_channels * P * P)``. AM
  swaps the fresh ``Conv2d`` module to ``Linear`` before loading that
  checkpoint so the tensor shape matches directly.
* ``embed_tokens`` / ``lm_head`` / the final ``norm`` are **UND-side tensors
  that are logically read by the GEN path** (GEN tokens use text embeddings
  and the shared LM head). No physical tensor sharing is required in the
  checkpoint — the module tree uses references, not copies.

Stage 1 loads the UND subset only (``UND_PATTERNS``). Stage 2 additionally
loads the GEN subset (``GEN_PATTERNS``). VAE keys are recognized only so that
accidentally merged checkpoints can be reported cleanly instead of being
treated as unknown model weights.
"""

from __future__ import annotations

import logging
import pathlib
import re
from typing import TYPE_CHECKING, Any, Optional

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    import torch
    from torch.distributed.device_mesh import DeviceMesh

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Regex patterns over upstream BAGEL's state-dict keys.
# ---------------------------------------------------------------------------

UND_PATTERNS = [
    r"^language_model\.model\.embed_tokens\.weight$",
    r"^language_model\.model\.norm\.weight$",
    r"^language_model\.lm_head\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|qkv)_proj\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.o_proj\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_norm\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_norm\.weight$",
    r"^language_model\.model\.layers\.\d+\.mlp\.(gate|up|down|gate_up)_proj\.weight$",
    r"^language_model\.model\.layers\.\d+\.input_layernorm\.weight$",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm\.weight$",
    r"^vit_model\.",
    r"^connector\.(fc1|fc2)\.(weight|bias)$",
    r"^vit_pos_embed\.pos_embed$",
]

GEN_PATTERNS = [
    r"^language_model\.model\.layers\.\d+\.self_attn\.(q|k|v|qkv)_proj_moe_gen\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.o_proj_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_norm_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_norm_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.mlp_moe_gen\.(gate|up|down|gate_up)_proj\.weight$",
    r"^language_model\.model\.layers\.\d+\.input_layernorm_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm_moe_gen\.weight$",
    r"^language_model\.model\.norm_moe_gen\.weight$",
    r"^time_embedder\.mlp\.(0|2)\.(weight|bias)$",
    r"^vae2llm\.(weight|bias)$",
    r"^llm2vae\.(weight|bias)$",
    r"^latent_pos_embed\.pos_embed$",
]

VAE_PATTERNS = [
    r"^encoder\.",
    r"^decoder\.",
]

EXTRA_STATE_PATTERNS = [
    r"\._extra_state$",
]

# No physical tensor sharing: embed_tokens / lm_head / final norm are UND-side
# but are logically READ by the gen path (gen tokens use text embeddings).
SHARED_PATTERNS: list[str] = []


def _compile(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p) for p in patterns]


_UND_RES = _compile(UND_PATTERNS)
_GEN_RES = _compile(GEN_PATTERNS)
_VAE_RES = _compile(VAE_PATTERNS)
_EXTRA_STATE_RES = _compile(EXTRA_STATE_PATTERNS)

_QWEN_LAYER_PREFIX = r"(?:(?:language_model\.)?model)\.layers\.\d+"
_SPLIT_QKV_RE = re.compile(rf"^({_QWEN_LAYER_PREFIX}\.self_attn)\.(q_proj|k_proj|v_proj)(_moe_gen)?\.(weight|bias)$")
_SPLIT_GATE_UP_RE = re.compile(rf"^({_QWEN_LAYER_PREFIX}\.mlp(?:_moe_gen)?)\.(gate_proj|up_proj)\.weight$")
_FUSED_QKV_RE = re.compile(rf"^({_QWEN_LAYER_PREFIX}\.self_attn)\.qkv_proj(_moe_gen)?\.(weight|bias)$")
_FUSED_GATE_UP_RE = re.compile(rf"^({_QWEN_LAYER_PREFIX}\.mlp(?:_moe_gen)?)\.gate_up_proj\.weight$")


def _matches_any(key: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(key) is not None for p in patterns)


def fuse_qwen_split_projections(
    weights: dict[str, Any],
    *,
    fuse_qkv: bool = True,
    fuse_gate_up: bool = True,
) -> dict[str, Any]:
    """Replace selected complete Q/K/V and gate/up groups with fused tensors."""
    import torch

    qkv_groups: dict[tuple[str, str, str], dict[str, tuple[str, Any]]] = {}
    gate_up_groups: dict[str, dict[str, tuple[str, Any]]] = {}

    for key, tensor in weights.items():
        qkv_match = _SPLIT_QKV_RE.match(key) if fuse_qkv else None
        if qkv_match:
            prefix, projection, gen_suffix, weight_or_bias = qkv_match.groups()
            group = (prefix, gen_suffix or "", weight_or_bias)
            qkv_groups.setdefault(group, {})[projection] = (key, tensor)
            continue
        gate_up_match = _SPLIT_GATE_UP_RE.match(key) if fuse_gate_up else None
        if gate_up_match:
            prefix, projection = gate_up_match.groups()
            gate_up_groups.setdefault(prefix, {})[projection] = (key, tensor)

    for (prefix, gen_suffix, weight_or_bias), parts in qkv_groups.items():
        missing = {"q_proj", "k_proj", "v_proj"} - parts.keys()
        if missing:
            raise KeyError(
                f"Cannot fuse {prefix}.qkv_proj{gen_suffix}.{weight_or_bias}: missing split tensors {sorted(missing)}."
            )
        fused_key = f"{prefix}.qkv_proj{gen_suffix}.{weight_or_bias}"
        if fused_key in weights:
            raise KeyError(f"Checkpoint contains both {fused_key} and its split Q/K/V tensors.")

    for prefix, parts in gate_up_groups.items():
        missing = {"gate_proj", "up_proj"} - parts.keys()
        if missing:
            raise KeyError(f"Cannot fuse {prefix}.gate_up_proj.weight: missing split tensors {sorted(missing)}.")
        fused_key = f"{prefix}.gate_up_proj.weight"
        if fused_key in weights:
            raise KeyError(f"Checkpoint contains both {fused_key} and its split gate/up tensors.")

    # Mutate only after every group has been validated so a malformed state
    # dict is not left half-converted.
    for (prefix, gen_suffix, weight_or_bias), parts in qkv_groups.items():
        fused_key = f"{prefix}.qkv_proj{gen_suffix}.{weight_or_bias}"
        weights[fused_key] = torch.cat([parts[name][1] for name in ("q_proj", "k_proj", "v_proj")], dim=0)
        for source_key, _ in parts.values():
            del weights[source_key]

    for prefix, parts in gate_up_groups.items():
        fused_key = f"{prefix}.gate_up_proj.weight"
        weights[fused_key] = torch.cat([parts[name][1] for name in ("gate_proj", "up_proj")], dim=0)
        for source_key, _ in parts.values():
            del weights[source_key]

    return weights


def _guard_row_sharded_split(key: str, tensor: Any) -> None:
    """Reject a split that would make DTensor replicate a fused parameter."""
    placements = getattr(tensor, "placements", ())
    if any(getattr(placement, "dim", None) == 0 for placement in placements):
        raise RuntimeError(
            f"Cannot split row-sharded DTensor {key!r} into HF projections without redistributing the full tensor. "
            "Use BAGEL's native fused DCP layout for checkpoint/resume. Initialize from an HF BAGEL checkpoint "
            "through model.init_mode=auto, which fuses full CPU tensors before distribution, and perform HF export "
            "from an unsharded/full state dict."
        )


def _split_fused_projections(
    weights: dict[str, Any],
    qkv_split_sizes: Optional[tuple[int, int, int]] = None,
    *,
    split_qkv: bool = True,
    split_gate_up: bool = True,
) -> dict[str, Any]:
    """Replace selected fused QKV and gate/up tensors with split tensors."""
    conversions: list[tuple[str, list[tuple[str, Any]]]] = []

    for key, tensor in weights.items():
        qkv_match = _FUSED_QKV_RE.match(key) if split_qkv else None
        if qkv_match:
            if qkv_split_sizes is None:
                raise RuntimeError("QKV split sizes are required to split a fused qkv_proj tensor.")
            prefix, gen_suffix, weight_or_bias = qkv_match.groups()
            gen_suffix = gen_suffix or ""
            target_keys = [
                f"{prefix}.{projection}{gen_suffix}.{weight_or_bias}" for projection in ("q_proj", "k_proj", "v_proj")
            ]
            conflicts = [target_key for target_key in target_keys if target_key in weights]
            if conflicts:
                raise KeyError(f"Checkpoint contains both {key} and split projection tensor(s) {conflicts}.")
            _guard_row_sharded_split(key, tensor)
            parts = tensor.split(qkv_split_sizes, dim=0)
            conversions.append((key, list(zip(target_keys, parts))))
            continue

        gate_up_match = _FUSED_GATE_UP_RE.match(key) if split_gate_up else None
        if gate_up_match:
            (prefix,) = gate_up_match.groups()
            target_keys = [f"{prefix}.gate_proj.weight", f"{prefix}.up_proj.weight"]
            conflicts = [target_key for target_key in target_keys if target_key in weights]
            if conflicts:
                raise KeyError(f"Checkpoint contains both {key} and split projection tensor(s) {conflicts}.")
            if tensor.shape[0] % 2:
                raise ValueError(f"Cannot evenly split {key} with shape {tuple(tensor.shape)} into gate/up tensors.")
            _guard_row_sharded_split(key, tensor)
            conversions.append((key, list(zip(target_keys, tensor.chunk(2, dim=0)))))

    for source_key, converted_parts in conversions:
        del weights[source_key]
        weights.update(converted_parts)
    return weights


def _normalize_stage(stage: Any) -> str:
    """Normalize ``stage`` to one of ``"stage1"`` or ``"stage2"``.

    Accepts either strings (``"stage1"`` / ``"stage2"``) or integers
    (``1`` / ``2``) for backward compatibility with earlier callers.
    """
    if isinstance(stage, int):
        stage = f"stage{stage}"
    stage = str(stage).lower()
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unknown stage {stage!r}; expected 'stage1' or 'stage2'.")
    return stage


def _partition(state_dict: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Partition a flat checkpoint dict into UND / GEN / VAE / extra / unknown buckets."""
    buckets: dict[str, dict[str, Any]] = {"und": {}, "gen": {}, "vae": {}, "extra": {}, "unknown": {}}
    for key, tensor in state_dict.items():
        if _matches_any(key, _EXTRA_STATE_RES):
            buckets["extra"][key] = tensor
        elif _matches_any(key, _UND_RES):
            buckets["und"][key] = tensor
        elif _matches_any(key, _GEN_RES):
            buckets["gen"][key] = tensor
        elif _matches_any(key, _VAE_RES):
            buckets["vae"][key] = tensor
        else:
            buckets["unknown"][key] = tensor
    return buckets


class BagelStateDictAdapter(StateDictAdapter):
    """HF <-> NeMo state-dict converter for BAGEL.

    Stage 1 returns only the UND subset from ``from_hf``. Stage 2 additionally
    returns the GEN subset. The VAE remains outside this module and is loaded
    separately by the training recipe.

    Because ``BagelForUnifiedMultimodal`` wraps the upstream BAGEL module
    tree under ``self.model``, HF checkpoint keys are unrooted
    (``language_model...``) while native AM keys are rooted
    (``model.language_model...``). The adapter filters upstream UND/GEN keys
    and handles this root mapping.

    Args:
        config: ``BagelConfig``. Its text config selects split versus fused
            projection layout and supplies the Q/K/V split sizes for export.
        stage: Default stage used when ``from_hf`` is called without an
            explicit ``stage`` kwarg. Accepts ``"stage1"`` / ``"stage2"`` or
            ``1`` / ``2``.
    """

    def __init__(self, config: Any = None, *, stage: Any = "stage1") -> None:
        self.config = config
        self.stage = _normalize_stage(stage)

    # ------------------------------------------------------------------
    # Keyspace helpers.
    # ------------------------------------------------------------------
    def _hf_to_nemo_key(self, hf_key: str) -> str:
        return hf_key if hf_key.startswith("model.") else f"model.{hf_key}"

    def _nemo_to_hf_key(self, nemo_key: str) -> str:
        return nemo_key.removeprefix("model.")

    def _strip_nemo_root(self, key: str) -> str:
        return key.removeprefix("model.")

    def _text_config(self) -> Any:
        if self.config is None:
            return None
        return getattr(self.config, "text_config", None) or getattr(self.config, "llm_config", None)

    def _uses_fused_qkv_projections(self) -> bool:
        text_config = self._text_config()
        if text_config is None:
            return False
        fused_qkv = getattr(text_config, "fused_qkv_projections", None)
        return bool(getattr(text_config, "fused_projections", False) if fused_qkv is None else fused_qkv)

    def _uses_fused_gate_up_projections(self) -> bool:
        text_config = self._text_config()
        if text_config is None:
            return False
        fused_gate_up = getattr(text_config, "fused_gate_up_projections", None)
        return bool(getattr(text_config, "fused_projections", False) if fused_gate_up is None else fused_gate_up)

    def _uses_any_fused_projections(self) -> bool:
        return self._uses_fused_qkv_projections() or self._uses_fused_gate_up_projections()

    def uses_native_checkpoint_layout(self) -> bool:
        """Return whether DCP should preserve the model's fused parameter layout."""
        return self._uses_any_fused_projections()

    def requires_full_state_dict_init(self) -> bool:
        """Fuse HF projections before distributing them into sharded parameters."""
        return self._uses_any_fused_projections()

    def _qkv_split_sizes(self) -> tuple[int, int, int]:
        text_config = self._text_config()
        if text_config is None:
            raise RuntimeError(
                "Splitting a fused BAGEL qkv_proj requires a config with num_attention_heads, "
                "num_key_value_heads, and hidden_size."
            )
        hidden_size = int(text_config.hidden_size)
        num_attention_heads = int(text_config.num_attention_heads)
        if hidden_size % num_attention_heads:
            raise ValueError(
                f"BAGEL hidden_size={hidden_size} must be divisible by num_attention_heads={num_attention_heads}."
            )
        head_dim = hidden_size // num_attention_heads
        configured_head_dim = getattr(text_config, "head_dim", None)
        if configured_head_dim is not None and int(configured_head_dim) != head_dim:
            raise ValueError(
                f"BAGEL config head_dim={configured_head_dim} disagrees with the model-derived head_dim={head_dim}."
            )
        q_size = text_config.num_attention_heads * head_dim
        kv_size = text_config.num_key_value_heads * head_dim
        return q_size, kv_size, kv_size

    def _split_fused_projection(self, key: str, tensor: "torch.Tensor") -> list[tuple[str, "torch.Tensor"]]:
        qkv_match = _FUSED_QKV_RE.match(key)
        if qkv_match:
            prefix, gen_suffix, weight_or_bias = qkv_match.groups()
            q_size, k_size, v_size = self._qkv_split_sizes()
            _guard_row_sharded_split(key, tensor)
            q, k, v = tensor.split([q_size, k_size, v_size], dim=0)
            gen_suffix = gen_suffix or ""
            return [
                (f"{prefix}.q_proj{gen_suffix}.{weight_or_bias}", q),
                (f"{prefix}.k_proj{gen_suffix}.{weight_or_bias}", k),
                (f"{prefix}.v_proj{gen_suffix}.{weight_or_bias}", v),
            ]

        gate_up_match = _FUSED_GATE_UP_RE.match(key)
        if gate_up_match:
            (prefix,) = gate_up_match.groups()
            _guard_row_sharded_split(key, tensor)
            gate, up = tensor.chunk(2, dim=0)
            return [
                (f"{prefix}.gate_proj.weight", gate),
                (f"{prefix}.up_proj.weight", up),
            ]
        return [(key, tensor)]

    # ------------------------------------------------------------------
    # from_hf: upstream BAGEL checkpoint -> NeMo module tree.
    # ------------------------------------------------------------------
    def from_hf(
        self,
        hf_state_dict: dict[str, "torch.Tensor"],
        device_mesh: Optional["DeviceMesh"] = None,
        *,
        stage: Optional[Any] = None,
        strict: bool = True,
        **kwargs: Any,
    ) -> dict[str, "torch.Tensor"]:
        """Convert an HF-layout BAGEL state dict to the NeMo module-tree layout.

        Args:
            hf_state_dict: Flat dict loaded from ``ema.safetensors``. If a
                caller accidentally passes merged VAE keys, they are recognized
                and excluded from the model state dict.
            device_mesh: Unused for BAGEL (no expert parallelism yet); kept
                for base-class signature compatibility.
            stage: ``"stage1"`` keeps only UND keys. ``"stage2"`` keeps UND
                and GEN keys. Defaults to ``self.stage``.
            strict: If ``True`` (default), raise ``KeyError`` on any key that
                matches no known pattern. If ``False``, log and drop.

        Returns:
            Filtered state dict in NeMo layout, ready to feed into
            ``BagelForUnifiedMultimodal.load_state_dict(...)``.

        Raises:
            KeyError: When ``strict=True`` and one or more input keys match
                no UND/GEN/VAE pattern.
        """
        if kwargs.get("native_checkpoint", False):
            if not self._uses_any_fused_projections():
                raise RuntimeError("A native fused BAGEL checkpoint requires at least one fused projection group.")
            return dict(hf_state_dict)

        stage = _normalize_stage(stage if stage is not None else self.stage)
        buckets = _partition({self._strip_nemo_root(k): v for k, v in hf_state_dict.items()})

        fuse_qkv = self._uses_fused_qkv_projections()
        fuse_gate_up = self._uses_fused_gate_up_projections()
        has_fused_qkv_to_split = not fuse_qkv and any(
            _FUSED_QKV_RE.match(key) for bucket_name in ("und", "gen") for key in buckets[bucket_name]
        )
        split_sizes = self._qkv_split_sizes() if has_fused_qkv_to_split else None
        for bucket_name in ("und", "gen"):
            fuse_qwen_split_projections(
                buckets[bucket_name],
                fuse_qkv=fuse_qkv,
                fuse_gate_up=fuse_gate_up,
            )
            _split_fused_projections(
                buckets[bucket_name],
                split_sizes,
                split_qkv=not fuse_qkv,
                split_gate_up=not fuse_gate_up,
            )

        if buckets["extra"]:
            logger.info(
                "BagelStateDictAdapter.from_hf: dropping %d Transformer-Engine _extra_state key(s).",
                len(buckets["extra"]),
            )

        if buckets["unknown"]:
            sample = list(buckets["unknown"])[:5]
            if strict:
                raise KeyError(
                    f"{len(buckets['unknown'])} key(s) in HF state dict did not match "
                    f"any BAGEL UND/GEN/VAE pattern (examples: {sample}). "
                    "Either widen the pattern set or pass strict=False."
                )
            logger.warning(
                "BagelStateDictAdapter.from_hf: dropping %d unmatched key(s); examples=%s",
                len(buckets["unknown"]),
                sample,
            )

        if stage == "stage1":
            selected = dict(buckets["und"])
            logger.info(
                "BagelStateDictAdapter.from_hf stage1: kept %d UND keys (dropped %d GEN, %d VAE).",
                len(buckets["und"]),
                len(buckets["gen"]),
                len(buckets["vae"]),
            )
        else:  # stage2
            selected = {**buckets["und"], **buckets["gen"]}
            logger.info(
                "BagelStateDictAdapter.from_hf stage2: kept %d UND + %d GEN = %d model keys "
                "(excluded %d VAE keys; VAE loads separately).",
                len(buckets["und"]),
                len(buckets["gen"]),
                len(selected),
                len(buckets["vae"]),
            )

        # Identity keyspace for now.
        return {self._hf_to_nemo_key(k): v for k, v in selected.items()}

    # ------------------------------------------------------------------
    # to_hf: NeMo module tree -> upstream BAGEL checkpoint layout.
    # ------------------------------------------------------------------
    def to_hf(self, state_dict: dict[str, "torch.Tensor"], **kwargs: Any) -> dict[str, "torch.Tensor"]:
        """Convert a NeMo-layout state dict back to the HF BAGEL layout.

        The VAE is not part of this module tree and should be saved/loaded
        separately. Ordinary fused DCP checkpoints pass
        ``native_checkpoint=True`` so row-sharded fused parameters remain
        write-through DTensors. Explicit HF export requires an unsharded/full
        state dict.
        """
        native_checkpoint = bool(kwargs.get("native_checkpoint", False))
        if native_checkpoint and not self._uses_any_fused_projections():
            raise RuntimeError("A native fused BAGEL checkpoint requires at least one fused projection group.")
        exclude_key_regex = kwargs.get("exclude_key_regex")
        exclude_pattern = re.compile(exclude_key_regex) if exclude_key_regex else None
        converted: dict[str, "torch.Tensor"] = {}
        for nemo_key, tensor in state_dict.items():
            if native_checkpoint:
                converted[nemo_key] = tensor
                continue
            if "._extra_state" in nemo_key:
                continue
            if exclude_pattern is not None and exclude_pattern.match(nemo_key):
                continue
            hf_key = self._nemo_to_hf_key(nemo_key)
            converted.update(self._split_fused_projection(hf_key, tensor))
        return converted

    # ------------------------------------------------------------------
    # Per-tensor conversion for explicit HF export tools.
    # ------------------------------------------------------------------
    def convert_single_tensor_to_hf(
        self,
        fqn: str,
        tensor: "torch.Tensor",
        **kwargs: Any,
    ) -> list[tuple[str, "torch.Tensor"]]:
        """Return ``[(hf_fqn, tensor)]`` for a single NeMo tensor.

        Fused projections are split into the standard HF checkpoint layout.
        """
        if "._extra_state" in fqn:
            return []
        return self._split_fused_projection(self._nemo_to_hf_key(fqn), tensor)


# ---------------------------------------------------------------------------
# Convenience loader.
# ---------------------------------------------------------------------------


def load_bagel_checkpoint_state_dict(
    checkpoint_dir: str | pathlib.Path,
    *,
    stage: Any = "stage1",
    strict: bool = True,
    config: Any = None,
) -> dict[str, "torch.Tensor"]:
    """Load a BAGEL HF checkpoint directory into a NeMo-layout state dict.

    Reads ``ema.safetensors`` from ``checkpoint_dir`` and passes the result
    through :class:`BagelStateDictAdapter`. Stage 2 VAE weights live in
    ``ae.safetensors`` but are loaded separately by the recipe because the VAE
    is not an ``nn.Module`` child of ``BagelForUnifiedMultimodal``.

    Args:
        checkpoint_dir: Path to a directory containing ``ema.safetensors``.
        stage: ``"stage1"`` or ``"stage2"``.
        strict: Forwarded to ``from_hf``; raise on unmatched keys.
        config: Optional ``BagelConfig`` for the adapter's log context.

    Returns:
        A flat ``{key: Tensor}`` dict in NeMo layout, ready for
        ``BagelForUnifiedMultimodal.load_state_dict(...)``.
    """
    from safetensors.torch import load_file  # local import: only needed on the load path

    checkpoint_dir = pathlib.Path(checkpoint_dir)
    stage_norm = _normalize_stage(stage)

    ema_path = checkpoint_dir / "ema.safetensors"
    if not ema_path.exists():
        raise FileNotFoundError(f"Expected BAGEL UND/GEN weights at {ema_path}")
    merged: dict[str, Any] = load_file(str(ema_path))
    logger.info("Loaded %d keys from %s", len(merged), ema_path)

    adapter = BagelStateDictAdapter(config=config, stage=stage_norm)
    return adapter.from_hf(merged, stage=stage_norm, strict=strict)
