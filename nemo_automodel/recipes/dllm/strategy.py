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

from nemo_automodel.components.attention.idlm_mask import (
    create_idlm_block_mask,
    create_idlm_sdpa_mask,
)
from nemo_automodel.components.datasets.dllm.corruption import (
    corrupt_all_masked,
    corrupt_blockwise,
    corrupt_uniform,
    corrupt_uniform_random,
)
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.dllm_loss import (
    BlockDiffusionCrossEntropyLoss,
    HybridDiffusionLLMLoss,
    IDLMLoss,
    MDLMCrossEntropyLoss,
)

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
        return "dllm_loss"

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
            noisy_input_ids, noise_mask, p_mask = recipe._apply_corruption(batch["input_ids"], batch["loss_mask"])
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
        num_ar_tokens: Optional[int] = None,
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
            num_ar_tokens=num_ar_tokens,
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
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(noisy_input_ids, noise_mask, p_mask)``.

        ``generator`` (optional): a step-seeded ``torch.Generator`` for the
        corruption draws so noise reproduces on resume. Strategies that draw from
        the global RNG may ignore it; ``block_diffusion`` threads it through.
        """

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

    def apply_corruption(
        self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio, generator=None
    ):
        return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        batch["input_ids"] = noisy_input_ids
        batch.pop("attention_mask", None)  # MDLM models are bidirectional
        return batch


class HybridStrategy(DLLMStrategy):
    """Strategy for hybrid diffusion + AR models (e.g., Nemotron-Labs-Diffusion).

    - Loss: :class:`HybridDiffusionLLMLoss` with configurable ``ar_loss_alpha``.
    - Corruption: uniform when ``block_size`` is ``None``, blockwise otherwise.
    - Batch: model receives clean tokens + ``masked_indices`` sidecar; the
      model applies masking internally during its forward pass.
    - Normalization: hybrid models normalize diffusion loss by the corrupted
      (noise) token count, not the full supervised count.
    """

    @property
    def normalization_mode(self) -> str:
        return "noise"

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        return HybridDiffusionLLMLoss(alpha=float(dllm_cfg.get("ar_loss_alpha", 1.0)))

    def apply_corruption(
        self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio, generator=None
    ):
        if block_size is None:
            return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)
        return corrupt_blockwise(
            input_ids,
            loss_mask,
            mask_token_id,
            block_size=block_size,
            eps=eps,
            half_life_ratio=half_life_ratio if half_life_ratio is not None else 0.25,
        )

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        batch["input_ids"] = clean_input_ids
        batch["masked_indices"] = noise_mask
        batch.pop("attention_mask", None)
        batch.pop("use_cache", None)
        batch["labels"] = clean_input_ids
        batch["skip_loss"] = True
        return batch


class IDLMStrategy(DLLMStrategy):
    """Strategy for Introspective Diffusion LM (I-DLM) all-masked finetuning.

    Converts an AR causal LM into a diffusion LM (Yu et al., 2026):

    - Corruption: deterministic all-masked over the supervised region
      (:func:`corrupt_all_masked`).
    - Forward: the noisy and clean copies are concatenated into a length-``2L``
      ``[x_t | x_0]`` sequence and run under the block-diffusion attention mask
      (:func:`create_idlm_sdpa_mask` / :func:`create_idlm_block_mask`), so decode
      tokens attend the clean ground-truth prefix and the clean copy stays
      strict-causal.
    - Loss: :class:`IDLMLoss` (Dream-shifted ``CE_noisy + alpha*CE_clean``,
      both supervised on the response).

    The mask is built for ``sdpa``/``eager`` (dense additive) or
    ``flex_attention`` (sparse ``BlockMask``, preferred at scale); FlashAttention-2
    is unsupported (it ignores arbitrary masks), as is context parallelism.
    """

    def __init__(self):
        self.block_size = 1

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        self.block_size = int(dllm_cfg.get("block_length", 1))
        return IDLMLoss(
            clean_loss_weight=float(dllm_cfg.get("clean_loss_weight", 0.2)),
            auto_balance=bool(dllm_cfg.get("auto_balance_clean_loss", False)),
        )

    def setup_extra(self, recipe) -> None:
        if getattr(recipe.distributed_config, "cp_size", 1) > 1:
            raise ValueError("I-DLM does not support context parallelism (cp_size must be 1).")
        model_config = getattr(recipe.model_parts[0], "config", None)
        attn_impl = getattr(model_config, "_attn_implementation", None)
        if attn_impl == "flash_attention_2":
            raise ValueError(
                "I-DLM needs the block-diffusion attention mask; set the model's "
                "attn_implementation to 'sdpa' (FlashAttention-2 ignores 4D masks)."
            )
        # A wrong id silently masks with a real token and trains garbage.
        vocab_size = getattr(model_config, "vocab_size", None)
        if recipe.mask_token_id is None:
            raise ValueError("I-DLM requires dllm.mask_token_id to be set explicitly.")
        if vocab_size is not None and not 0 <= int(recipe.mask_token_id) < int(vocab_size):
            raise ValueError(
                f"dllm.mask_token_id={recipe.mask_token_id} is outside the model vocab (size {vocab_size})."
            )

    def apply_corruption(
        self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio, generator=None
    ):
        # `generator` is accepted for interface compatibility but unused: all-masked
        # corruption is deterministic.
        return corrupt_all_masked(input_ids, loss_mask, mask_token_id)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        """Required by the abstract interface but unused on the I-DLM path.

        :meth:`forward_backward` is overridden and builds the ``[x_t | x_0]``
        concat itself from the ``pre_step`` sidecars, so the recipe never routes
        an I-DLM batch through here. Kept (and kept correct) only to satisfy
        :class:`DLLMStrategy`; assigning the noisy ids matches what the base
        MDLM path would do if a future edit re-enabled that route.
        """
        batch["input_ids"] = noisy_input_ids
        return batch

    def forward_backward(
        self,
        recipe,
        idx: int,
        batch: dict,
        *,
        loss_buffer: list,
        num_diffusion_tokens: int,
        num_ar_tokens: Optional[int] = None,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """I-DLM microbatch: single ``[x_t | x_0]`` forward + two-CE loss."""
        device = recipe.dist_env.device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        clean_input_ids = batch.pop("_clean_input_ids")
        noisy_input_ids = batch.pop("_noisy_input_ids")
        noise_mask = batch.pop("_noise_mask")
        batch.pop("_p_mask", None)
        batch.pop("loss_mask", None)
        attn = batch.get("attention_mask")
        if attn is None:
            attn = torch.ones_like(noisy_input_ids)

        model = recipe.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(recipe.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        autocast_dtype = getattr(recipe.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        )
        fp8_ctx = recipe.te_fp8.maybe_te_autocast() if recipe.te_fp8 is not None else nullcontext()
        train_ctx, _ = make_cp_batch_and_ctx(recipe.device_mesh, {})

        L = noisy_input_ids.size(1)
        concat_input_ids = torch.cat([noisy_input_ids, clean_input_ids], dim=1)
        pos = torch.arange(L, device=device)
        concat_position_ids = torch.cat([pos, pos]).unsqueeze(0).expand(concat_input_ids.size(0), -1)
        if getattr(getattr(model, "config", None), "_attn_implementation", None) == "flex_attention":
            block_mask = create_idlm_block_mask(L, self.block_size, attn, device=device)
        else:
            mask_dtype = autocast_dtype if autocast_dtype is not None else torch.float32
            block_mask = create_idlm_sdpa_mask(L, self.block_size, attn, device=device, dtype=mask_dtype)

        with train_ctx(), sync_ctx, fp8_ctx, autocast_ctx:
            out = model(
                input_ids=concat_input_ids,
                attention_mask=block_mask,
                position_ids=concat_position_ids,
                use_cache=False,
            )
            logits = out.logits if not torch.is_tensor(out) else out
            loss_result = recipe.dllm_loss_fn(
                logits, clean_input_ids, noise_mask, attn, seq_len=L, num_diffusion_tokens=num_diffusion_tokens
            )
            microbatch_loss = loss_result.total_loss
            loss_buffer.append(microbatch_loss.detach().clone())
            recipe._dllm_loss_buffer.append(loss_result.dllm_loss)

            if is_train:
                (microbatch_loss * recipe._get_dp_group_size(include_cp=True)).backward()


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
    - ``target_torch_dtype`` (default ``"bfloat16"``) — target dtype string.
    - ``block_size`` (default 0) — draft block size; 0 reads from draft config.
    - ``loss_decay_gamma`` (default 0.0) — γ for Eq. 4; 0 uses paper defaults.
    - ``num_blocks_per_sample`` (default 1) — N anchor blocks per sequence per
      step, enabling the multi-block sparse-attention pass from §4.2. Paper
      default is 512 (Appendix A.1); requires ``attention_backend=flex_attention``.
    - ``attention_backend`` (default ``"sdpa"``) — ``"sdpa"`` materialises a
      dense ``[B, 1, N·bs, S+N·bs]`` mask (OOMs at high N); ``"flex_attention"``
      uses a sparse :class:`BlockMask` and matches the paper's setup.
    - ``overlap_anchors`` (default ``True``) — when ``True``, anchors are
      sampled independently (paper behaviour); when ``False``, anchors are
      forced non-overlapping (stars-and-bars, caps at ``seq_len // block_size``).
    """

    def __init__(self):
        self.target_model = None
        self.target_embed = None
        self.target_head = None
        self.block_size: int = 0
        self.num_blocks_per_sample: int = 1
        self.layer_ids: list = []
        self.dflash_loss_fn = None
        self.attention_backend: str = "sdpa"
        self.overlap_anchors: bool = True
        self.use_fused_linear_ce: bool = True
        self.fixed_ctx_len: int = 0

    @property
    def loss_log_key(self) -> str:
        return "dllm_loss"

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        return MDLMCrossEntropyLoss()  # placeholder; real loss is self.dflash_loss_fn

    # ------------------------------------------------------------------
    # apply_corruption / prepare_batch — not used by DFlash but required
    # by the abstract interface; forward_backward overrides both paths.
    # ------------------------------------------------------------------

    def apply_corruption(
        self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio, generator=None
    ):
        # generator is accepted for signature parity with the base/other strategies
        # (the recipe passes it unconditionally); DFlash draws from the global RNG.
        return corrupt_uniform(input_ids, loss_mask, mask_token_id, eps=eps)

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        return batch

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup_extra(self, recipe) -> None:
        """Load and freeze the target LM; resolve block_size, layer_ids, decay loss."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from nemo_automodel.components.loss.dllm_loss import DFlashDecayLoss

        dflash_cfg = recipe.cfg.get("dflash", None) or {}

        # Resolve mask_token_id when the tokenizer (e.g. Qwen3) has none.
        if recipe.mask_token_id is None:
            tok_id = dflash_cfg.get("target_model_id") or recipe.cfg.get("model.pretrained_model_name_or_path")
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
            target_model_id, dtype=target_dtype, trust_remote_code=True
        )
        self.target_model.eval()
        self.target_model.requires_grad_(False)
        self.target_model = self.target_model.to(recipe.dist_env.device)

        self.target_embed = self.target_model.get_input_embeddings()
        self.target_head = self.target_model.get_output_embeddings()
        if self.target_embed is None:
            self.target_embed = getattr(getattr(self.target_model, "model", None), "embed_tokens", None)
        if self.target_head is None:
            self.target_head = getattr(self.target_model, "lm_head", None)
        if self.target_embed is None or self.target_head is None:
            raise ValueError("Target model must expose input embeddings and lm_head.")

        # --- Block size ---
        draft = recipe.model_parts[0]
        block_size = int(dflash_cfg.get("block_size", 0))
        if block_size <= 0:
            draft_cfg = getattr(draft, "config", None)
            block_size = getattr(draft, "block_size", None) or getattr(draft_cfg, "block_size", None)
        if not block_size:
            raise ValueError("Cannot infer block_size from draft config. Set dflash.block_size in the YAML.")
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
                "DFlash: cannot determine target_layer_ids from draft config; falling back to single mid-layer %d.",
                mid,
            )
        self.layer_ids = list(layer_ids)

        # --- Decay loss (paper Eq. 4) ---
        gamma_cfg = float(dflash_cfg.get("loss_decay_gamma", 0.0))
        loss_gamma = (
            gamma_cfg
            if gamma_cfg > 0.0
            else {16: 7.0, 10: 5.0, 8: 4.0}.get(self.block_size, max(2.0, self.block_size / 2.0))
        )
        # Chunked linear cross-entropy: projects the LM head + CE in
        # torch.utils.checkpoint position chunks so the [B, N*(block_size-1), vocab]
        # logits tensor is never materialised — required to fit paper-default
        # num_blocks_per_sample=512 on a full-vocab target. Plain autograd, so it
        # trains correctly under FSDP2. Default on; set
        # dflash.use_fused_linear_ce: false to fall back to dense logits + CE.
        # ce_chunk_size trades peak memory (smaller = lower) against recompute.
        self.use_fused_linear_ce = bool(dflash_cfg.get("use_fused_linear_ce", True))
        ce_chunk_size = int(dflash_cfg.get("ce_chunk_size", 1024))
        self.dflash_loss_fn = DFlashDecayLoss(
            loss_gamma=loss_gamma,
            use_fused_linear_ce=self.use_fused_linear_ce,
            chunk_size=ce_chunk_size,
        )

        # --- Multi-block ---
        self.num_blocks_per_sample = int(dflash_cfg.get("num_blocks_per_sample", 1))
        self.overlap_anchors = bool(dflash_cfg.get("overlap_anchors", True))

        # Fixed context length for static FlexAttention shapes. The collator pads
        # each batch to its own block-aligned max, so KV_LEN would still vary
        # batch-to-batch and force recompiles. Padding the target context up to a
        # single fixed length (dataset.seq_length) makes Q_LEN/KV_LEN constant
        # across every step → the kernel compiles once. The block-diagonal mask
        # (kv_idx < anchor) never reads the padded tail, so it is loss-neutral.
        ds_cfg = recipe.cfg.get("dataset", None)
        self.fixed_ctx_len = int(ds_cfg.get("seq_length", 0)) if ds_cfg is not None else 0

        # --- Attention backend (sdpa | flex_attention) ---
        backend = str(dflash_cfg.get("attention_backend", "sdpa")).lower()
        if backend not in ("sdpa", "flex_attention"):
            raise ValueError(f"dflash.attention_backend must be 'sdpa' or 'flex_attention', got {backend!r}")
        if backend == "flex_attention":
            # Route the draft model's per-layer attention through transformers'
            # flex_attention dispatcher. The draft model reads
            # ``self.config._attn_implementation`` at runtime via ALL_ATTENTION_FUNCTIONS.
            if draft_cfg is not None:
                draft_cfg._attn_implementation = "flex_attention"
        self.attention_backend = backend

        logger.info(
            "DFlash setup: target=%s, block_size=%d, num_blocks=%d, layer_ids=%s, "
            "loss_gamma=%.1f, attention_backend=%s, overlap_anchors=%s",
            target_model_id,
            self.block_size,
            self.num_blocks_per_sample,
            self.layer_ids,
            loss_gamma,
            self.attention_backend,
            self.overlap_anchors,
        )

    # ------------------------------------------------------------------
    # Pre-step: anchor-block sampling + target forwards
    # ------------------------------------------------------------------

    def _sample_anchor_block(
        self,
        recipe,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = input_ids.size(0)
        device = input_ids.device
        valid_len = int(attention_mask.sum(dim=1).min().item())
        max_start = max(1, valid_len - self.block_size)
        start = int(torch.randint(1, max_start + 1, (1,), device=device).item())

        block_output_ids = input_ids.new_full((B, self.block_size), recipe.mask_token_id)
        block_output_ids[:, 0] = input_ids[:, start]
        block_targets = input_ids[:, start + 1 : start + self.block_size]
        effective_mask = attention_mask if loss_mask is None else attention_mask * loss_mask
        block_mask = effective_mask[:, start + 1 : start + self.block_size].float()
        return start, block_output_ids, block_targets, block_mask

    @torch.no_grad()
    def _run_target_forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, start: int) -> torch.Tensor:
        out = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        offset = 1  # skip embedding layer (index 0)
        return torch.cat([out.hidden_states[lid + offset] for lid in self.layer_ids], dim=-1)[:, :start, :]

    def _sample_anchor_blocks(
        self,
        recipe,
        input_ids: torch.Tensor,
        attn: torch.Tensor,
        num_blocks: int,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample ``num_blocks`` anchors **per sample** and gather block tensors.

        Each sequence in the batch independently draws ``N = num_blocks`` anchor
        positions from its own ``[1, valid_len_b - block_size]`` range (paper
        §4.2 "randomly sample anchor tokens"). Per-sample sampling gives more
        position diversity per step than sharing one anchor set across the batch.

        Samples with fewer than ``2 * block_size`` supervised tokens are dropped
        via ``block_keep_mask`` so degenerate short sequences contribute no loss.

        Returns:
            anchor_positions: ``[B, N]`` long — per-sample anchor positions.
            block_keep_mask:  ``[B, N]`` bool — False for dropped/short samples.
            block_output_ids: ``[B, N*block_size]`` — anchor token at each block
                start, ``mask_token_id`` elsewhere.
            block_targets:    ``[B, N*(block_size-1)]`` — gathered target tokens.
            block_mask:       ``[B, N*(block_size-1)]`` — float mask: supervised
                AND in-bounds AND kept.
        """
        B, L = input_ids.shape
        device = input_ids.device
        bs = self.block_size
        N = max(1, num_blocks)

        effective = (attn if loss_mask is None else attn * loss_mask).float()  # [B, L]
        valid_lens = attn.sum(dim=1)  # [B] attended length per sample
        supervised_lens = effective.sum(dim=1)  # [B] supervised tokens per sample
        max_anchor = (valid_lens - bs).clamp(min=1)  # [B] latest valid anchor (>=1 for safe sampling)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)  # [B, N]

        n_valid = N  # number of real (non-padding) blocks; only < N in non-overlap mode
        if self.overlap_anchors:
            # Per-sample independent anchors in [1, valid_len_b - block_size].
            rand = torch.rand(B, N, device=device)
            anchor_positions = (1 + (rand * (max_anchor - 1).unsqueeze(1).float()).round().long()).clamp(min=1)
            anchor_positions = torch.minimum(anchor_positions, max_anchor.unsqueeze(1))
        else:
            # Legacy batch-shared non-overlapping (stars-and-bars), broadcast to
            # [B, N]; padding blocks (when fewer than N fit) get keep=False below.
            vmin = int(valid_lens.min().item())
            n_valid = min(N, max(1, (vmin - 1) // bs))
            avail = vmin - n_valid * bs
            if avail < 1:
                starts = torch.arange(1, n_valid + 1, device=device)
            else:
                perm = torch.randperm(avail, device=device)[:n_valid].sort().values
                starts = perm + torch.arange(n_valid, device=device) * bs + 1
            if n_valid < N:  # pad to fixed N (keep=False masks the padding)
                starts = torch.cat([starts, starts.new_full((N - n_valid,), int(starts[-1]))])
            anchor_positions = starts.unsqueeze(0).expand(B, N).contiguous()

        # Min-loss-token filter: a sample must hold a block and have at least
        # 2*block_size supervised tokens, else all its blocks are dropped.
        # Padding blocks beyond n_valid (non-overlap mode) are also dropped.
        sample_ok = (supervised_lens >= 2 * bs) & (valid_lens > bs)  # [B]
        block_keep_mask = sample_ok.unsqueeze(1) & (torch.arange(N, device=device).unsqueeze(0) < n_valid)
        block_keep_mask = block_keep_mask.contiguous()  # [B, N]

        # block_output_ids: anchor token at each block start, mask elsewhere.
        block_output_ids = input_ids.new_full((B, N * bs), recipe.mask_token_id)
        anchor_tokens = input_ids[batch_idx, anchor_positions.clamp(max=L - 1)]  # [B, N]
        block_starts = (torch.arange(N, device=device) * bs).unsqueeze(0).expand(B, N)  # [B, N]
        block_output_ids[batch_idx, block_starts] = torch.where(
            block_keep_mask, anchor_tokens, anchor_tokens.new_full(anchor_tokens.shape, recipe.mask_token_id)
        )

        # Targets + mask for predicted positions anchor+1 .. anchor+block_size-1.
        tgt_off = torch.arange(1, bs, device=device).view(1, 1, -1)  # [1, 1, bs-1]
        tgt_idx = anchor_positions.unsqueeze(-1) + tgt_off  # [B, N, bs-1]
        in_bounds = tgt_idx < valid_lens.view(B, 1, 1)  # within attended region
        safe_idx = tgt_idx.clamp(max=L - 1)  # [B, N, bs-1]
        block_targets = torch.gather(input_ids.unsqueeze(1).expand(B, N, L), 2, safe_idx).reshape(B, N * (bs - 1))
        bm = (
            torch.gather(effective.unsqueeze(1).expand(B, N, L), 2, safe_idx)
            * in_bounds.float()
            * block_keep_mask.unsqueeze(-1).float()
        )
        block_mask = bm.reshape(B, N * (bs - 1))

        return anchor_positions, block_keep_mask, block_output_ids, block_targets, block_mask

    def pre_step(self, recipe, batches) -> tuple[int, int]:
        """Sample anchor blocks and run frozen target forwards for all microbatches."""
        device = recipe.dist_env.device
        num_predicted = 0
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            loss_mask = batch.get("loss_mask")
            if loss_mask is not None:
                loss_mask = loss_mask.to(device)
            anchor_positions, block_keep_mask, block_output_ids, block_targets, block_mask = self._sample_anchor_blocks(
                recipe, input_ids, attn, self.num_blocks_per_sample, loss_mask
            )
            # Run the target over the FULL (constant) sequence length, not up to
            # the deepest anchor. A varying context length would make KV_LEN vary
            # and force FlexAttention to recompile each step; a fixed length lets
            # the kernel compile once. The block-diagonal mask (kv_idx < anchor)
            # still stops any block from reading past its own anchor.
            ctx_len = int(input_ids.shape[1])
            target_hidden = self._run_target_forward(input_ids, attn, ctx_len)
            # Offload to CPU so draft backward has the full VRAM budget.
            batch["_dflash_anchor_positions"] = anchor_positions
            batch["_dflash_block_keep"] = block_keep_mask
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
        num_ar_tokens: Optional[int] = None,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """DFlash microbatch: draft forward + decay loss + (optional) backward."""
        device = recipe.dist_env.device
        batch = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Retrieve pre-computed DFlash tensors (set by pre_step).
        if "_dflash_anchor_positions" in batch:
            anchor_positions = batch.pop("_dflash_anchor_positions")
            block_keep_mask = batch.pop("_dflash_block_keep")
        else:
            # Fallback: compute on the fly (e.g. when called outside pre_step).
            input_ids = batch["input_ids"]
            attn = batch.get("attention_mask", torch.ones_like(input_ids))
            anchor_positions, block_keep_mask, boi, bt, bm = self._sample_anchor_blocks(
                recipe, input_ids, attn, self.num_blocks_per_sample, batch.get("loss_mask")
            )
            target_hidden = self._run_target_forward(input_ids, attn, int(input_ids.shape[1]))
            batch["_dflash_target_hidden"] = target_hidden
            batch["_dflash_block_output_ids"] = boi
            batch["_dflash_block_targets"] = bt
            batch["_dflash_block_mask"] = bm

        target_hidden = batch.pop("_dflash_target_hidden").to(device)
        anchor_positions = anchor_positions.to(device)
        block_keep_mask = block_keep_mask.to(device)
        block_output_ids = batch.pop("_dflash_block_output_ids")
        block_targets = batch.pop("_dflash_block_targets")
        block_mask = batch.pop("_dflash_block_mask")

        B = block_output_ids.size(0)
        N = anchor_positions.shape[1]
        # Pad the target context up to the fixed sequence length so Q_LEN/KV_LEN
        # are constant across every step (the collator pads each batch to its own
        # max, which would otherwise keep changing KV_LEN and recompile the
        # FlexAttention kernel each step). The block-diagonal mask only attends to
        # kv_idx < anchor < valid_len, so the zero-padded tail is never read —
        # loss-neutral, purely a shape stabiliser.
        if self.fixed_ctx_len and target_hidden.shape[1] < self.fixed_ctx_len:
            pad_n = self.fixed_ctx_len - target_hidden.shape[1]
            pad = target_hidden.new_zeros(target_hidden.shape[0], pad_n, target_hidden.shape[2])
            target_hidden = torch.cat([target_hidden, pad], dim=1)
        # ctx_len is now the fixed sequence length → FlexAttention compiles once.
        ctx_len = target_hidden.shape[1]
        noise_embedding = self.target_embed(block_output_ids)  # [B, N*block_size, dim]

        # Per-sample position IDs: shared context positions then each block's own
        # anchor range (RoPE correctness). anchor_positions is [B, N], so block
        # positions differ per sample — no broadcast.
        ctx_pos = torch.arange(ctx_len, device=device).unsqueeze(0).expand(B, -1)  # [B, ctx_len]
        blk_off = torch.arange(self.block_size, device=device).view(1, 1, -1)  # [1, 1, block_size]
        block_pos = (anchor_positions.unsqueeze(-1) + blk_off).reshape(B, N * self.block_size)  # [B, N*block_size]
        position_ids = torch.cat([ctx_pos, block_pos], dim=1)  # [B, ctx_len + N*block_size]

        # Sparse block-diagonal attention mask. For N=1 with the SDPA backend
        # we can skip the mask — the context-prefix slicing in
        # _run_target_forward already prevents post-anchor context leakage. For
        # FlexAttention we always build a BlockMask so the dispatcher gets the
        # expected type. anchor_positions/block_keep_mask are already per-sample
        # [B, N], so the mask is per-sample with no broadcast.
        attn_mask = None
        if N > 1 or self.attention_backend == "flex_attention":
            from nemo_automodel.components.attention.dflash_mask import (
                create_dflash_block_mask,
                create_dflash_sdpa_mask,
            )

            if self.attention_backend == "flex_attention":
                attn_mask = create_dflash_block_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    ctx_len=ctx_len,
                    block_size=self.block_size,
                    device=device,
                )
            else:
                attn_mask = create_dflash_sdpa_mask(
                    anchor_positions=anchor_positions,
                    block_keep_mask=block_keep_mask,
                    ctx_len=ctx_len,
                    block_size=self.block_size,
                    device=device,
                    dtype=noise_embedding.dtype,
                )

        draft = recipe.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                draft,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(recipe.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        autocast_dtype = getattr(recipe.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        )
        fp8_ctx = recipe.te_fp8.maybe_te_autocast() if recipe.te_fp8 is not None else nullcontext()
        train_ctx, _ = make_cp_batch_and_ctx(recipe.device_mesh, {})

        with train_ctx(), sync_ctx, fp8_ctx, autocast_ctx:
            draft_kwargs = dict(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids,
                use_cache=False,
                is_causal=False,
            )
            if attn_mask is not None:
                draft_kwargs["attention_mask"] = attn_mask

            draft_hidden = draft(**draft_kwargs)
            if not torch.is_tensor(draft_hidden):
                draft_hidden = getattr(draft_hidden, "last_hidden_state", draft_hidden[0])

            # Extract predicted positions (skip the anchor token at index 0 of
            # each block). draft_hidden: [B, N*block_size, dim] → [B, N*(block_size-1), dim].
            pred = draft_hidden.view(B, N, self.block_size, -1)[:, :, 1:, :].reshape(B, N * (self.block_size - 1), -1)
            if self.use_fused_linear_ce:
                # Fuse the LM-head projection into the CE — avoids materialising
                # the [B, N*(block_size-1), vocab] logits tensor (the main OOM
                # source at large N on a full-vocab target).
                loss_result = self.dflash_loss_fn.forward_fused(
                    hidden=pred,
                    lm_head_weight=self.target_head.weight,
                    target_ids=block_targets,
                    block_mask=block_mask,
                    num_tokens=num_diffusion_tokens,
                    block_size=self.block_size if N > 1 else None,
                    lm_head_bias=getattr(self.target_head, "bias", None),
                )
            else:
                logits = self.target_head(pred)
                loss_result = self.dflash_loss_fn(
                    logits=logits,
                    target_ids=block_targets,
                    block_mask=block_mask,
                    num_tokens=num_diffusion_tokens,
                    block_size=self.block_size if N > 1 else None,
                )
            microbatch_loss = loss_result.total_loss
            loss_buffer.append(microbatch_loss.detach().clone())
            recipe._dllm_loss_buffer.append(loss_result.dllm_loss)
            if loss_result.draft_correct_per_pos is not None:
                recipe._dflash_correct_per_pos_buffer.append(loss_result.draft_correct_per_pos.detach())
                recipe._dflash_count_per_pos_buffer.append(loss_result.draft_count_per_pos.detach())

            if is_train:
                (microbatch_loss * recipe._get_dp_group_size(include_cp=True)).backward()


class BlockDiffusionStrategy(DLLMStrategy):
    """Strategy for ``diffusion_gemma`` block-diffusion SFT (single-turn v1).

    - Loss: :class:`BlockDiffusionCrossEntropyLoss` (flat CE, no ``1/p``, no AR).
    - Corruption: :func:`corrupt_uniform_random` — per-block ``t~U(eps,1)``,
      supervised positions replaced with uniform random vocab tokens (no
      ``[MASK]``). Requires ``vocab_size`` (see :meth:`create_loss_fn`).
    - Normalization: ``"supervised"`` (denominator = all supervised canvas-token
      count, matching Google's all-canvas loss support — NOT corrupted-only).
    - Batch: the encoder sees the **clean full sequence** (prompt + response);
      the decoder canvas is the **noised response region only**, sliced and the
      block-causal mask built by ``DiffusionGemmaSFTRecipe`` (the recipe owns
      the response-window construction because it also needs the sliced loss
      tensors). :meth:`prepare_batch` only assigns the encoder input and the
      full noised sequence; :meth:`split_prompt_response` gives the per-example
      prompt boundary the recipe slices on.

    v1 is **single-turn only**: multi-turn ``ChatDataset`` ``loss_mask`` is
    ``0..0 1..1 0..0 1..1`` (not a contiguous suffix), so the prompt|response
    split is ill-defined. Interleaved multi-turn masking is deferred.
    """

    def __init__(self) -> None:
        # vocab_size is needed by corrupt_uniform_random but is not part of the
        # apply_corruption ABC signature (which passes mask_token_id, unused
        # here). It is captured from the dllm config in create_loss_fn, which the
        # recipe always calls during setup before any corruption runs.
        self._vocab_size: int | None = None

    @property
    def normalization_mode(self) -> str:
        # "supervised": denominator = ALL supervised canvas tokens (corrupted +
        # uncorrupted), matching Google's all-canvas loss support. Was "noise"
        # (corrupted-only) — the loss-support bug.
        return "supervised"

    def create_loss_fn(self, dllm_cfg: dict) -> nn.Module:
        vocab_size = dllm_cfg.get("vocab_size", None)
        if vocab_size is not None:
            self._vocab_size = int(vocab_size)
        return BlockDiffusionCrossEntropyLoss()

    def apply_corruption(
        self, input_ids, loss_mask, mask_token_id, *, eps, block_size, half_life_ratio, generator=None
    ):
        if self._vocab_size is None:
            raise ValueError(
                "BlockDiffusionStrategy requires dllm.vocab_size to be set in the config "
                "(uniform random-token corruption samples replacements over [0, vocab_size); "
                "there is no mask_token_id)."
            )
        # One corruption level t per example (block_size=None), matching Google's
        # scheme: a single canvas is scored per step, so per-block t is moot.
        return corrupt_uniform_random(
            input_ids,
            loss_mask,
            self._vocab_size,
            block_size=None,
            eps=eps,
            generator=generator,
        )

    def pre_step(self, recipe, batches) -> tuple[int, int]:
        """Per-microbatch corruption, threading ``microbatch_idx`` into the seed.

        Same sidecar stashing as the base ``pre_step``, but passes ``microbatch_idx``
        to ``recipe._apply_corruption`` so each grad-accum microbatch draws distinct,
        resume-reproducible block-diffusion noise.
        """
        num_noise = 0
        num_supervised = 0
        for microbatch_idx, batch in enumerate(batches):
            noisy_input_ids, noise_mask, p_mask = recipe._apply_corruption(
                batch["input_ids"], batch["loss_mask"], microbatch_idx=microbatch_idx
            )
            batch["_noisy_input_ids"] = noisy_input_ids
            batch["_noise_mask"] = noise_mask
            batch["_p_mask"] = p_mask
            batch["_clean_input_ids"] = batch["input_ids"].clone()
            num_noise += int(noise_mask.sum().item())
            num_supervised += int(batch["loss_mask"].sum().item())
        return num_noise, num_supervised

    @staticmethod
    def split_prompt_response(
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single-turn boundary: prefix length(s) and a per-position response mask.

        Returns ``(prefix_lengths, response_mask)`` where ``prefix_lengths[b]`` is
        the index of the first supervised (``loss_mask == 1``) position in row
        ``b`` (the start of the response) and ``response_mask`` is the boolean
        mask of response positions (``position >= prefix_length``). Used by the
        ``_forward_backward_step`` override to build ``canvas_ids`` and the
        block-causal mask's ``prefix_lengths``.

        Single-turn assumption: ``loss_mask`` is a single contiguous suffix.
        """
        lm = loss_mask.bool()
        has_sup = lm.any(dim=1)
        first_sup = torch.argmax(lm.int(), dim=1)  # first True index; 0 if none
        # Rows with no supervised token: treat the whole row as prompt.
        prefix_lengths = torch.where(has_sup, first_sup, torch.full_like(first_sup, input_ids.shape[1]))
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        response_mask = positions >= prefix_lengths[:, None]
        return prefix_lengths, response_mask

    def prepare_batch(self, batch, noisy_input_ids, noise_mask, clean_input_ids):
        # Encoder sees the CLEAN full sequence (prompt + response); the decoder
        # canvas is the noised sequence. Bidirectional model -> drop attention_mask.
        # ``DiffusionGemmaSFTRecipe._forward_backward_step`` then slices
        # ``canvas_ids`` (and the matching loss tensors) to the response region
        # and builds the block-causal ``decoder_attention_mask`` — that slicing
        # lives in the recipe because it must also reshape the loss tensors,
        # which are not visible here.
        batch["input_ids"] = clean_input_ids
        batch["canvas_ids"] = noisy_input_ids
        batch.pop("attention_mask", None)
        batch.pop("use_cache", None)
        return batch


DLLM_STRATEGIES: Dict[str, type] = {
    "mdlm": MDLMStrategy,
    "hybrid": HybridStrategy,
    "idlm": IDLMStrategy,
    "dflash": DFlashStrategy,
    "block_diffusion": BlockDiffusionStrategy,
}


def get_dllm_strategy(mode: str) -> DLLMStrategy:
    """Look up and instantiate a dLLM strategy by mode name.

    Raises:
        ValueError: If *mode* is not registered in :data:`DLLM_STRATEGIES`.
    """
    cls = DLLM_STRATEGIES.get(mode)
    if cls is None:
        raise ValueError(f"Unknown dllm.mode: {mode!r}. Available: {sorted(DLLM_STRATEGIES)}")
    return cls()
