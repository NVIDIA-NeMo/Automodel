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

"""Model-specific strategies for diffusion LLM (dLLM) training.

Each strategy encapsulates the variation points that differ across dLLM
model families:

1. **Loss function creation** — which loss module to use.
2. **Pre-step processing** — corruption (MDLM) or target-model forwards (DFlash).
3. **Forward-backward** — the per-microbatch forward + loss + backward.
4. **Normalization mode** — loss denominator: supervised tokens or noise tokens.
5. **Extra setup** — loading auxiliary models (e.g. frozen target for DFlash).

To add a new dLLM variant, implement a :class:`DLLMStrategy` subclass and
register it in :data:`DLLM_STRATEGIES`.  No changes to the recipe are required.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.dllm.corruption import corrupt_uniform
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.dllm_loss import MDLMCrossEntropyLoss

logger = logging.getLogger(__name__)


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly-spaced target hidden-layer indices for DFlash feature extraction."""
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start, end = 1, int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


class DLLMStrategy(ABC):
    """Abstract base for dLLM model strategies."""

    @property
    def normalization_mode(self) -> str:
        """Token count used as the loss denominator: ``"supervised"`` or ``"noise"``.

        * ``"supervised"`` — total ``loss_mask == 1`` positions (default).
        * ``"noise"`` — actually-corrupted positions (``noise_mask == True``).
        """
        return "supervised"

    @property
    def loss_log_key(self) -> str:
        """Metric key used for dLLM loss in MetricsSample and console log lines."""
        return "Loss/Train_DLLM"

    @abstractmethod
    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        """Return the loss module for this model type."""

    def setup_extra(self, recipe) -> None:
        """Hook called at the end of :meth:`DiffusionLMSFTRecipe.setup`.

        Strategies that need auxiliary models (e.g. a frozen target LM) or
        that resolve ``recipe.mask_token_id`` should do so here.
        """

    def pre_step(self, recipe, batches) -> tuple[int, int]:
        """Pre-process all microbatches before the forward-backward loop.

        Called once per training step (and once per val batch) with the full
        list of microbatch dicts.  May mutate batch dicts in-place to stash
        pre-computed tensors for :meth:`forward_backward`.

        Returns:
            ``(num_noise_tokens, num_supervised_tokens)`` — raw (un-allreduced)
            token counts used for loss normalisation and metrics.
        """
        num_noise = 0
        num_supervised = 0
        for batch in batches:
            noisy_input_ids, noise_mask, p_mask = recipe._apply_corruption(
                batch["input_ids"], batch["loss_mask"]
            )
            batch["_noisy_input_ids"] = noisy_input_ids
            batch["_noise_mask"] = noise_mask
            batch["_p_mask"] = p_mask
            batch["_clean_input_ids"] = batch["input_ids"].clone()
            num_noise += int(noise_mask.sum().item())
            num_supervised += int(batch["loss_mask"].sum().item())
        return num_noise, num_supervised

    def forward_backward(
        self,
        recipe,
        idx: int,
        batch: dict,
        *,
        loss_buffer: list,
        num_diffusion_tokens: int,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """Run one microbatch forward + loss + (optionally) backward.

        Default implementation delegates to the recipe's existing MDLM
        ``_forward_backward_step`` so that the MDLM code path is unchanged.
        """
        recipe._forward_backward_step(
            idx,
            batch,
            loss_buffer=loss_buffer,
            num_diffusion_tokens=num_diffusion_tokens,
            num_batches=num_batches,
            is_train=is_train,
        )

    @abstractmethod
    def apply_corruption(
        self,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        mask_token_id: int,
        *,
        eps: float,
        block_size: Optional[int],
        half_life_ratio: Optional[float],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(noisy_input_ids, noise_mask, p_mask)``."""

    @abstractmethod
    def prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        noisy_input_ids: torch.Tensor,
        noise_mask: torch.Tensor,
        clean_input_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Mutate *batch* in-place for the model's forward pass and return it."""


class MDLMStrategy(DLLMStrategy):
    """Strategy for MDLM / LLaDA-style models.

    - Loss: :class:`MDLMCrossEntropyLoss`
    - Corruption: uniform masking (``corrupt_uniform``)
    - Batch: model receives noisy (corrupted) tokens as ``input_ids``
    """

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        return MDLMCrossEntropyLoss()

    def apply_corruption(self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio):
        return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        batch["input_ids"] = noisy_input_ids
        batch.pop("attention_mask", None)  # MDLM models are bidirectional
        return batch


class DFlashStrategy(DLLMStrategy):
    """Strategy for DFlash dual-model draft training.

    DFlash training differs from MDLM in three ways:

    1. A frozen causal target LM provides hidden-state context.
    2. One clean anchor token starts each block; the rest are mask-filled.
    3. Loss is decay-weighted by position within the block (Eq. 4).

    All DFlash-specific logic lives here so :class:`DiffusionLMSFTRecipe`
    requires no subclassing for DFlash.

    YAML configuration (under the ``dflash:`` key):

    - ``target_model_id`` (**required**) — frozen causal LM hub ID.
    - ``target_torch_dtype`` (default ``"bfloat16"``) — target dtype.
    - ``block_size`` (default 0) — draft block size; 0 reads from draft config.
    - ``loss_decay_gamma`` (default 0.0) — γ for Eq. 4; 0 uses paper defaults.
    """

    def __init__(self):
        self.target_model = None
        self.target_embed = None
        self.target_head = None
        self.block_size: int = 0
        self.layer_ids: list = []
        self.dflash_loss_fn = None

    @property
    def loss_log_key(self) -> str:
        return "Loss/Train_DFlash"

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        return MDLMCrossEntropyLoss()  # placeholder; real loss is self.dflash_loss_fn

    # ------------------------------------------------------------------
    # apply_corruption / prepare_batch — not used by DFlash but required
    # by the abstract interface; forward_backward overrides both paths.
    # ------------------------------------------------------------------

    def apply_corruption(self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio):
        return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        return batch

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_extra(self, recipe) -> None:
        """Load and freeze the target LM; resolve block_size, layer_ids, decay loss."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from nemo_automodel.components.loss.dflash_loss import DFlashDecayLoss

        dflash_cfg = recipe.cfg.get("dflash", None) or {}

        # Resolve mask_token_id when the tokenizer (e.g. Qwen3) has none.
        if recipe.mask_token_id is None:
            tok_id = dflash_cfg.get("target_model_id") or recipe.cfg.get(
                "model.pretrained_model_name_or_path"
            )
            if tok_id:
                tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
                if tok.mask_token_id is None:
                    tok.add_special_tokens({"mask_token": "<|MASK|>"})
                recipe.mask_token_id = int(tok.mask_token_id)
                logger.info("DFlash: resolved mask_token_id=%d from %s", recipe.mask_token_id, tok_id)

        # --- Frozen target model ---
        target_model_id = dflash_cfg.get("target_model_id")
        if not target_model_id:
            raise ValueError("dflash.target_model_id must be set in config.")

        target_dtype_str = dflash_cfg.get("target_torch_dtype", "bfloat16")
        target_dtype = getattr(torch, target_dtype_str, torch.bfloat16)

        logger.info("DFlash: loading frozen target model %s (%s)", target_model_id, target_dtype_str)
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id, torch_dtype=target_dtype
        )
        self.target_model.eval()
        self.target_model.requires_grad_(False)
        self.target_model = self.target_model.to(recipe.dist_env.device)

        self.target_embed = self.target_model.get_input_embeddings()
        self.target_head = self.target_model.get_output_embeddings()
        if self.target_embed is None:
            self.target_embed = getattr(
                getattr(self.target_model, "model", None), "embed_tokens", None
            )
        if self.target_head is None:
            self.target_head = getattr(self.target_model, "lm_head", None)
        if self.target_embed is None or self.target_head is None:
            raise ValueError("Target model must expose input embeddings and lm_head.")

        # --- Block size ---
        draft = recipe.model_parts[0]
        block_size = int(dflash_cfg.get("block_size", 0))
        if block_size <= 0:
            draft_cfg = getattr(draft, "config", None)
            block_size = getattr(draft, "block_size", None) or getattr(
                draft_cfg, "block_size", None
            )
        if not block_size:
            raise ValueError(
                "Cannot infer block_size from draft config. Set dflash.block_size in the YAML."
            )
        self.block_size = int(block_size)
        if self.block_size < 2:
            raise ValueError("dflash.block_size must be at least 2.")

        # --- Layer IDs for hidden-state extraction ---
        draft_cfg = getattr(draft, "config", None)
        layer_ids = getattr(draft, "target_layer_ids", None)
        if layer_ids is None and draft_cfg is not None:
            num_tgt = getattr(draft_cfg, "num_target_layers", None)
            num_hid = getattr(draft_cfg, "num_hidden_layers", None)
            if num_tgt is not None and num_hid is not None:
                layer_ids = _build_target_layer_ids(int(num_tgt), int(num_hid))
        if layer_ids is None:
            mid = self.target_model.config.num_hidden_layers // 2
            layer_ids = [mid]
            logger.warning(
                "DFlash: cannot determine target_layer_ids from draft config; "
                "falling back to single mid-layer %d.",
                mid,
            )
        self.layer_ids = list(layer_ids)

        # --- Decay loss (paper Eq. 4) ---
        gamma_cfg = float(dflash_cfg.get("loss_decay_gamma", 0.0))
        loss_gamma = (
            gamma_cfg
            if gamma_cfg > 0.0
            else {16: 7.0, 10: 5.0, 8: 4.0}.get(
                self.block_size, max(2.0, self.block_size / 2.0)
            )
        )
        self.dflash_loss_fn = DFlashDecayLoss(loss_gamma=loss_gamma)

        logger.info(
            "DFlash setup: target=%s, block_size=%d, layer_ids=%s, loss_gamma=%.1f",
            target_model_id,
            self.block_size,
            self.layer_ids,
            loss_gamma,
        )

    # ------------------------------------------------------------------
    # Pre-step: anchor-block sampling + target forwards
    # ------------------------------------------------------------------

    def _sample_anchor_block(
        self, recipe, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        device = input_ids.device
        valid_len = int(attention_mask.sum(dim=1).min().item())
        max_start = max(1, valid_len - self.block_size)
        start = int(torch.randint(1, max_start + 1, (1,), device=device).item())

        block_output_ids = input_ids.new_full((B, self.block_size), recipe.mask_token_id)
        block_output_ids[:, 0] = input_ids[:, start]
        block_targets = input_ids[:, start + 1 : start + self.block_size]
        block_mask = attention_mask[:, start + 1 : start + self.block_size].float()
        return start, block_output_ids, block_targets, block_mask

    @torch.no_grad()
    def _run_target_forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, start: int
    ) -> torch.Tensor:
        out = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        offset = 1  # skip embedding layer (index 0)
        return torch.cat(
            [out.hidden_states[lid + offset] for lid in self.layer_ids], dim=-1
        )[:, :start, :]

    def pre_step(self, recipe, batches) -> tuple[int, int]:
        """Sample anchor blocks and run frozen target forwards for all microbatches."""
        device = recipe.dist_env.device
        num_predicted = 0
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            start, block_output_ids, block_targets, block_mask = self._sample_anchor_block(
                recipe, input_ids, attn
            )
            target_hidden = self._run_target_forward(input_ids, attn, start)
            # Offload to CPU so draft backward has the full VRAM budget.
            batch["_dflash_start"] = start
            batch["_dflash_target_hidden"] = target_hidden.cpu()
            batch["_dflash_block_output_ids"] = block_output_ids
            batch["_dflash_block_targets"] = block_targets
            batch["_dflash_block_mask"] = block_mask
            num_predicted += int(block_mask.sum().item())
        return num_predicted, num_predicted

    # ------------------------------------------------------------------
    # Forward-backward
    # ------------------------------------------------------------------

    def forward_backward(
        self,
        recipe,
        idx: int,
        batch: dict,
        *,
        loss_buffer: list,
        num_diffusion_tokens: int,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """DFlash microbatch: draft forward + decay loss + (optional) backward."""
        device = recipe.dist_env.device
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        if "_dflash_start" in batch:
            start = batch.pop("_dflash_start")
            target_hidden = batch.pop("_dflash_target_hidden").to(device)
            block_output_ids = batch.pop("_dflash_block_output_ids")
            block_targets = batch.pop("_dflash_block_targets")
            block_mask = batch.pop("_dflash_block_mask")
        else:
            # Fallback: compute on the fly (e.g. when called outside pre_step).
            input_ids = batch["input_ids"]
            attn = batch.get("attention_mask", torch.ones_like(input_ids))
            start, block_output_ids, block_targets, block_mask = self._sample_anchor_block(
                recipe, input_ids, attn
            )
            target_hidden = self._run_target_forward(input_ids, attn, start)

        B = block_output_ids.size(0)
        noise_embedding = self.target_embed(block_output_ids)
        position_ids = (
            torch.arange(start + self.block_size, device=device).unsqueeze(0).expand(B, -1)
        )

        draft = recipe.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                draft,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(
                    recipe.distributed_config, "defer_fsdp_grad_sync", True
                ),
            )
            if is_train
            else nullcontext()
        )
        autocast_dtype = getattr(recipe.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        fp8_ctx = (
            recipe.te_fp8.maybe_te_autocast() if recipe.te_fp8 is not None else nullcontext()
        )
        train_ctx, _ = make_cp_batch_and_ctx(recipe.device_mesh, {})

        with train_ctx(), sync_ctx, fp8_ctx, autocast_ctx:
            draft_hidden = draft(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
            if not torch.is_tensor(draft_hidden):
                draft_hidden = getattr(draft_hidden, "last_hidden_state", draft_hidden[0])

            logits = self.target_head(draft_hidden[:, -self.block_size + 1 :, :])

            loss_result = self.dflash_loss_fn(
                logits=logits,
                target_ids=block_targets,
                block_mask=block_mask,
                num_tokens=num_diffusion_tokens,
            )
            microbatch_loss = loss_result.total_loss
            loss_buffer.append(microbatch_loss.detach().clone())
            recipe._dllm_loss_buffer.append(loss_result.dllm_loss)

            if is_train:
                (microbatch_loss * recipe._get_dp_group_size(include_cp=True)).backward()


DLLM_STRATEGIES: Dict[str, type] = {
    "mdlm": MDLMStrategy,
    "dflash": DFlashStrategy,
}


def get_dllm_strategy(mode: str) -> DLLMStrategy:
    """Look up and instantiate a dLLM strategy by mode name.

    Raises:
        ValueError: If *mode* is not registered in :data:`DLLM_STRATEGIES`.
    """
    cls = DLLM_STRATEGIES.get(mode)
    if cls is None:
        raise ValueError(
            f"Unknown dllm.mode: {mode!r}. Available: {sorted(DLLM_STRATEGIES)}"
        )
    return cls()
