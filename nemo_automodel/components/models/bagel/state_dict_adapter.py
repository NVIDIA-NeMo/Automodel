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
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_proj\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_proj\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.v_proj\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.o_proj\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_norm\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_norm\.weight$",
    r"^language_model\.model\.layers\.\d+\.mlp\.(gate|up|down)_proj\.weight$",
    r"^language_model\.model\.layers\.\d+\.input_layernorm\.weight$",
    r"^language_model\.model\.layers\.\d+\.post_attention_layernorm\.weight$",
    r"^vit_model\.",
    r"^connector\.(fc1|fc2)\.(weight|bias)$",
    r"^vit_pos_embed\.pos_embed$",
]

GEN_PATTERNS = [
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_proj_moe_gen\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_proj_moe_gen\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.v_proj_moe_gen\.(weight|bias)$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.o_proj_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.q_norm_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.self_attn\.k_norm_moe_gen\.weight$",
    r"^language_model\.model\.layers\.\d+\.mlp_moe_gen\.(gate|up|down)_proj\.weight$",
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

def _matches_any(key: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(key) is not None for p in patterns)


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
        config: ``BagelConfig`` (or ``None``; currently only used for log
            context -- no shape sanity checks yet).
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
        stage = _normalize_stage(stage if stage is not None else self.stage)
        buckets = _partition({self._strip_nemo_root(k): v for k, v in hf_state_dict.items()})

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
        separately.
        """
        exclude_key_regex = kwargs.get("exclude_key_regex")
        exclude_pattern = re.compile(exclude_key_regex) if exclude_key_regex else None
        converted: dict[str, "torch.Tensor"] = {}
        for nemo_key, tensor in state_dict.items():
            if "._extra_state" in nemo_key:
                continue
            if exclude_pattern is not None and exclude_pattern.match(nemo_key):
                continue
            converted[self._nemo_to_hf_key(nemo_key)] = tensor
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
        """Return ``[(hf_fqn, tensor)]`` for a single NeMo tensor."""
        if "._extra_state" in fqn:
            return []
        return [(self._nemo_to_hf_key(fqn), tensor)]


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
