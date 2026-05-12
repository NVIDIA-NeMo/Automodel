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

"""DFlash draft-model SFT recipe for Automodel.

DFlash differs from plain MDLM in three ways that require a dedicated recipe:

1. **Dual-model forward**: a trainable draft model and a frozen target causal LM.
2. **Anchor-block masking**: one clean anchor token at position 0 of a block;
   the remaining ``block_size - 1`` positions are masked.
3. **Decay-weighted loss**: position k is weighted by ``exp(-(k-1)/γ)``
   (Eq. 4 of the DFlash paper) instead of the MDLM ``1/p`` schedule.

The recipe extends :class:`DiffusionLMSFTRecipe` to inherit distributed setup,
optimizer, dataloader, tokenizer, and checkpoint infrastructure, then overrides
the forward-backward step with the DFlash-specific logic.

The **draft model** is configured under the ``model:`` YAML key and receives
FSDP2 sharding.  The **target model** is loaded from ``dflash.target_model_id``
as a per-GPU frozen copy (no FSDP2 — weights are read-only).

Usage::

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/dllm/train_dflash.py \\
        -c examples/dllm_sft/dflash_sft.yaml
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Optional

import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp
from transformers import AutoModelForCausalLM

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.dflash_loss import DFlashDecayLoss
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Target hidden-state helpers
# ---------------------------------------------------------------------------


def _build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> list[int]:
    """Evenly-spaced target hidden-layer indices for feature extraction."""
    if num_draft_layers == 1:
        return [int(num_target_layers // 2)]
    start, end = 1, int(num_target_layers) - 3
    span = end - start
    return [int(round(start + (i * span) / (num_draft_layers - 1))) for i in range(num_draft_layers)]


def _extract_context_features(hidden_states: tuple, layer_ids: list[int]) -> torch.Tensor:
    """Concatenate selected target hidden states along the feature dim."""
    offset = 1  # skip the embedding layer (index 0)
    return torch.cat([hidden_states[lid + offset] for lid in layer_ids], dim=-1)


def _get_target_embeddings(model: torch.nn.Module):
    """Return ``(input_embed, lm_head)`` from a causal LM."""
    embed = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if embed is None:
        embed = getattr(getattr(model, "model", None), "embed_tokens", None)
    if head is None:
        head = getattr(model, "lm_head", None)
    if embed is None:
        raise ValueError("Target model must expose input embeddings.")
    if head is None:
        raise ValueError("Target model must expose output embeddings (lm_head).")
    return embed, head


# ---------------------------------------------------------------------------
# Recipe
# ---------------------------------------------------------------------------


class DFlashSFTRecipe(DiffusionLMSFTRecipe):
    """SFT recipe for fine-tuning DFlash draft models.

    YAML configuration keys under the ``dflash`` section:

    - ``target_model_id`` (str, **required**) — HF hub ID of the frozen target LM.
    - ``target_torch_dtype`` (str, default ``"bfloat16"``) — target model dtype.
    - ``block_size`` (int, default 0) — draft block size; 0 reads from draft config.
    - ``loss_decay_gamma`` (float, default 0.0) — γ for Eq. 4; 0 uses paper defaults
      (7 for block 16, 5 for block 10, 4 for block 8).

    The ``model`` section in the YAML configures the **draft** model only.
    The draft receives FSDP2 sharding; the target is a full frozen copy per GPU.
    """

    def setup(self):
        """Build all components, then load and freeze the target model."""
        # Ensure parent's dllm setup can resolve mask_token_id even when neither
        # the YAML nor the base tokenizer (Qwen3) define one. DFlash uses
        # <|MASK|> as a virtual mask token; we add it to the tokenizer here so
        # the parent sees a valid integer and won't raise.
        if self.cfg.get("dllm.mask_token_id") is None:
            from transformers import AutoTokenizer
            dflash_cfg = self.cfg.get("dflash", None)
            tok_id = (
                dflash_cfg.get("target_model_id") if dflash_cfg else None
            ) or self.cfg.get("model.pretrained_model_name_or_path")
            if tok_id is not None:
                try:
                    _tok = AutoTokenizer.from_pretrained(tok_id, trust_remote_code=True)
                    if _tok.mask_token_id is None:
                        _tok.add_special_tokens({"mask_token": "<|MASK|>"})
                    self.cfg.set_by_dotted("dllm.mask_token_id", int(_tok.mask_token_id))
                    logger.info("DFlash: resolved mask_token_id=%d from %s", _tok.mask_token_id, tok_id)
                except Exception as e:
                    logger.warning("Could not resolve mask_token_id from %s: %s", tok_id, e)

        # Parent sets up: dist_env, draft model (self.model_parts), optimizer,
        # dataloader, tokenizer, step_scheduler, checkpointer, and dLLM fields.
        super().setup()

        dflash_cfg = self.cfg.get("dflash", {})

        # --- Frozen target model ---
        target_model_id = dflash_cfg.get("target_model_id", None)
        if target_model_id is None:
            raise ValueError("dflash.target_model_id must be set in config.")

        target_dtype_str = dflash_cfg.get("target_torch_dtype", "bfloat16")
        target_dtype = getattr(torch, target_dtype_str, torch.bfloat16)

        logger.info("Loading frozen target model: %s (dtype=%s)", target_model_id, target_dtype_str)
        target_model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=target_dtype,
        )
        target_model.eval()
        target_model.requires_grad_(False)
        target_model = target_model.to(self.dist_env.device)
        self.target_model = target_model

        self.target_embed, self.target_head = _get_target_embeddings(self.target_model)

        # --- Block size ---
        draft = self.model_parts[0]
        block_size = int(dflash_cfg.get("block_size", 0))
        if block_size <= 0:
            draft_cfg = getattr(draft, "config", None)
            block_size = getattr(draft, "block_size", None) or getattr(draft_cfg, "block_size", None)
        if block_size is None:
            raise ValueError(
                "Cannot infer block_size from draft config. Set dflash.block_size in the YAML."
            )
        self.dflash_block_size = int(block_size)
        if self.dflash_block_size < 2:
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
                "Cannot determine target_layer_ids from draft config; falling back to single mid-layer %d.",
                mid,
            )
        self.dflash_layer_ids = list(layer_ids)

        # --- Decay loss (Eq. 4) ---
        gamma_cfg = float(dflash_cfg.get("loss_decay_gamma", 0.0))
        loss_gamma = (
            gamma_cfg
            if gamma_cfg > 0.0
            else {16: 7.0, 10: 5.0, 8: 4.0}.get(self.dflash_block_size, max(2.0, self.dflash_block_size / 2.0))
        )
        self.dflash_loss_fn = DFlashDecayLoss(loss_gamma=loss_gamma)

        logger.info(
            "DFlash setup complete: target=%s, block_size=%d, layer_ids=%s, loss_gamma=%.1f",
            target_model_id,
            self.dflash_block_size,
            self.dflash_layer_ids,
            loss_gamma,
        )

    # ------------------------------------------------------------------
    # DFlash-specific helpers
    # ------------------------------------------------------------------

    def _sample_anchor_block(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random anchor and build the draft block tensors.

        Returns:
            start: Anchor position index (scalar int, shared across batch).
            block_output_ids: shape ``[B, block_size]`` — anchor at [:, 0],
                mask token elsewhere.
            block_targets: shape ``[B, block_size-1]`` — ground-truth labels.
            block_mask: shape ``[B, block_size-1]`` — float valid-position mask.
        """
        B = input_ids.size(0)
        block_size = self.dflash_block_size
        device = input_ids.device

        valid_len = int(attention_mask.sum(dim=1).min().item())
        max_start = max(1, valid_len - block_size)
        start = int(torch.randint(1, max_start + 1, (1,), device=device).item())

        block_output_ids = input_ids.new_full((B, block_size), self.mask_token_id)
        block_output_ids[:, 0] = input_ids[:, start]

        block_targets = input_ids[:, start + 1 : start + block_size]
        block_mask = attention_mask[:, start + 1 : start + block_size].float()

        return start, block_output_ids, block_targets, block_mask

    @torch.no_grad()
    def _run_target_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        start: int,
    ) -> torch.Tensor:
        """Run the frozen target and return context features for tokens ``[:start]``.

        Returns:
            Tensor of shape ``[B, start, hidden_dim * num_layers]``.
        """
        out = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        return _extract_context_features(out.hidden_states, self.dflash_layer_ids)[:, :start, :]

    # ------------------------------------------------------------------
    # Forward-backward override
    # ------------------------------------------------------------------

    def _dflash_forward_backward_step(
        self,
        idx: int,
        batch: dict,
        *,
        loss_buffer: list,
        num_batches: int,
        is_train: bool = True,
    ) -> None:
        """Single DFlash forward-backward microstep."""
        device = self.dist_env.device
        batch = {
            k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Use pre-computed inputs when available (training path)
        if "_dflash_start" in batch:
            start = batch.pop("_dflash_start")
            target_hidden = batch.pop("_dflash_target_hidden").to(device)
            block_output_ids = batch.pop("_dflash_block_output_ids")
            block_targets = batch.pop("_dflash_block_targets")
            block_mask = batch.pop("_dflash_block_mask")
        else:
            # Validation path: compute on the fly
            input_ids = batch["input_ids"]
            attn = batch.get("attention_mask", torch.ones_like(input_ids))
            start, block_output_ids, block_targets, block_mask = self._sample_anchor_block(input_ids, attn)
            target_hidden = self._run_target_forward(input_ids, attn, start)

        B = block_output_ids.size(0)
        noise_embedding = self.target_embed(block_output_ids)
        position_ids = (
            torch.arange(start + self.dflash_block_size, device=device).unsqueeze(0).expand(B, -1)
        )

        draft = self.model_parts[0]
        sync_ctx = (
            get_sync_ctx(
                draft,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        autocast_dtype = getattr(self.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if autocast_dtype is not None
            else nullcontext()
        )
        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()
        train_ctx, _ = make_cp_batch_and_ctx(self.device_mesh, {})

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

            # Project through frozen target lm_head (block_size-1 predicted positions)
            logits = self.target_head(draft_hidden[:, -self.dflash_block_size + 1 :, :])

            loss_result = self.dflash_loss_fn(
                logits=logits,
                target_ids=block_targets,
                block_mask=block_mask,
            )
            microbatch_loss = loss_result.total_loss
            loss_buffer.append(microbatch_loss.detach().clone())
            self._dllm_loss_buffer.append(loss_result.dllm_loss)

            if is_train:
                (microbatch_loss * self._get_dp_group_size(include_cp=True)).backward()

    # ------------------------------------------------------------------
    # Training step override
    # ------------------------------------------------------------------

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute one DFlash training step across all microbatches.

        Target forwards are pre-computed for all microbatches (no-grad),
        then offloaded to CPU so GPU memory is free for draft backwards.
        """
        device = self.dist_env.device

        # --- Pre-compute target hidden states (no grad) ---
        num_predicted_tokens = 0
        num_tokens_in_batch = 0
        for batch in batches:
            input_ids = batch["input_ids"].to(device)
            attn = batch.get("attention_mask", torch.ones_like(input_ids)).to(device)
            start, block_output_ids, block_targets, block_mask = self._sample_anchor_block(input_ids, attn)
            target_hidden = self._run_target_forward(input_ids, attn, start)

            # Offload target hidden to CPU to free VRAM for draft backward
            batch["_dflash_start"] = start
            batch["_dflash_target_hidden"] = target_hidden.cpu()
            batch["_dflash_block_output_ids"] = block_output_ids
            batch["_dflash_block_targets"] = block_targets
            batch["_dflash_block_mask"] = block_mask
            num_predicted_tokens += int(block_mask.sum().item())
            num_tokens_in_batch += input_ids.numel()

        num_predicted_tokens = self._dp_allreduce(
            torch.tensor(num_predicted_tokens, dtype=torch.long)
        ).item()
        num_tokens_in_batch = self._dp_allreduce(
            torch.tensor(num_tokens_in_batch, dtype=torch.long)
        ).item()

        # --- Draft forward-backward ---
        loss_buffer = []
        num_batches = len(batches)
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)
            self._dflash_forward_backward_step(
                i, batch, loss_buffer=loss_buffer, num_batches=num_batches
            )
            if i == 0:
                prepare_after_first_microbatch()

        # --- Gradient clipping + optimizer step ---
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=max(num_predicted_tokens, 1),
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for sched in self.lr_scheduler:
                sched.step(1)

        fp8_cfg = self.cfg.get("fp8", None)
        if (
            fp8_cfg is not None
            and fp8_cfg.get("enabled", False)
            and fp8_cfg.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and not self.pp_enabled
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        # --- Metrics ---
        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        total_loss = self._dp_allreduce(
            torch.sum(torch.stack(loss_buffer)), include_cp=True
        ).cpu().item()
        dflash_loss = self._dp_allreduce(
            torch.stack(self._dllm_loss_buffer).sum(), include_cp=True
        ).item()
        self._dllm_loss_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "loss": total_loss,
                "Loss/Train_Total": total_loss,
                "Loss/Train_DFlash": dflash_loss,
                "grad_norm": grad_norm,
                "Train/grad_norm": grad_norm,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "Train/lr": self.optimizer[0].param_groups[0]["lr"],
                "Train/mem": torch.cuda.max_memory_allocated() / 1024**3,
                "Train/tps": tps,
                "Train/tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
                "Train/mfu": None,
                "Train/tokens_per_step": num_tokens_in_batch,
                "Train/supervised_tokens": num_predicted_tokens,
                "Train/block_size": self.dflash_block_size,
                "Train/mode": self.dllm_mode,
            },
        )

    # ------------------------------------------------------------------
    # Validation override
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one DFlash validation pass."""
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_weighted_loss = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
            total_tokens = 0

            for batch in val_dataloader:
                input_ids = batch["input_ids"].to(self.dist_env.device)
                attn = batch.get("attention_mask", torch.ones_like(input_ids)).to(self.dist_env.device)
                start, block_output_ids, block_targets, block_mask = self._sample_anchor_block(input_ids, attn)
                target_hidden = self._run_target_forward(input_ids, attn, start)

                batch["_dflash_start"] = start
                batch["_dflash_target_hidden"] = target_hidden
                batch["_dflash_block_output_ids"] = block_output_ids
                batch["_dflash_block_targets"] = block_targets
                batch["_dflash_block_mask"] = block_mask

                n_tokens = int(block_mask.sum().item())
                n_tokens_global = self._dp_allreduce(
                    torch.tensor(n_tokens, dtype=torch.long)
                ).item()

                loss_buffer = []
                self._dflash_forward_backward_step(
                    0, batch, loss_buffer=loss_buffer, num_batches=1, is_train=False
                )

                batch_loss = self._dp_allreduce(
                    torch.sum(torch.stack(loss_buffer)).to(self.dist_env.device),
                    include_cp=True,
                ).item()
                total_weighted_loss += batch_loss * n_tokens_global
                total_tokens += n_tokens_global

        val_loss = float(total_weighted_loss / max(total_tokens, 1e-8))
        self._dllm_loss_buffer.clear()

        for mp in self.model_parts:
            mp.train()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    # ------------------------------------------------------------------
    # Logging override
    # ------------------------------------------------------------------

    def log_train_metrics(self, log_data):
        """Log DFlash-specific training metrics."""
        if not self.dist_env.is_main:
            return

        if self.step_scheduler.is_remote_logging_step:
            remote_metrics = {
                k: v for k, v in log_data.to_dict().items() if k not in ("step", "epoch", "timestamp")
            }
            if wandb.run is not None:
                wandb.log(remote_metrics, step=self.step_scheduler.step)
            if self.mlflow_logger is not None:
                self.mlflow_logger.log_metrics(remote_metrics, step=log_data.step)
            if self.comet_logger is not None:
                self.comet_logger.log_metrics(remote_metrics, step=log_data.step)

        self.metric_logger_train.log(log_data)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | dflash_loss {:.4f} | grad_norm {:.4f} | "
            "lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu) | mode {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["Loss/Train_Total"],
                log_data.metrics["Loss/Train_DFlash"],
                log_data.metrics["Train/grad_norm"],
                log_data.metrics["Train/lr"],
                log_data.metrics["Train/mem"],
                log_data.metrics["Train/tps"],
                log_data.metrics["Train/tps_per_gpu"],
                log_data.metrics["Train/mode"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(config_path=None):
    """Main entry point for DFlash SFT recipe."""
    if config_path is None:
        config_path = "examples/dllm_sft/dflash_sft.yaml"
    cfg = parse_args_and_load_config(config_path)
    trainer = DFlashSFTRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
