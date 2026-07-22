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

"""Diffusion LLM (dLLM) SFT recipe for Automodel.

Extends ``TrainFinetuneRecipeForNextTokenPrediction`` to support diffusion LLM
training. Instead of next-token prediction, the model is trained as a denoiser:
tokens are randomly corrupted and the model predicts the clean token at each
position.  Loss is weighted by the inverse corruption probability.

Model-specific behaviour (loss function, corruption strategy, batch preparation)
is encapsulated in :mod:`~nemo_automodel.recipes.dllm.strategy` so that new
dLLM variants can be added without modifying this recipe.  Current modes:

- **mdlm**: Pure masked denoising.  Uses ``MDLMCrossEntropyLoss``.

Usage::

    python -m torch.distributed.run --nproc-per-node=8 \\
        nemo_automodel/recipes/dllm/train_ft.py \\
        -c examples/dllm_sft/mdlm_sft.yaml
"""

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from typing import Optional

import mlflow
import torch
import wandb
from torchao.float8 import precompute_float8_dynamic_scale_for_fsdp

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.dllm.collate import DLLMCollator
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loggers.mlflow_utils import to_float_metrics
from nemo_automodel.components.loss.dllm_loss import encoder_ar_loss
from nemo_automodel.components.models.diffusion_gemma.attention_mask import build_block_diffusion_training_mask
from nemo_automodel.components.training.rng import ScopedRNG
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.flops_utils import calculate_mfu
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs
from nemo_automodel.recipes.dllm.strategy import BlockDiffusionStrategy, get_dllm_strategy
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

logger = logging.getLogger(__name__)


class DiffusionLMSFTRecipe(TrainFinetuneRecipeForNextTokenPrediction):
    """Recipe for dLLM (diffusion LLM) supervised fine-tuning.

    Extends the standard fine-tuning recipe by:

    1. Wrapping the dataloader collate function to produce unshifted batches
    2. Applying token corruption before each forward pass
    3. Using dLLM-specific loss functions via a pluggable strategy
    """

    # Whether the collator uses DiffusionGemma response-window EOS-fill (response-relative,
    # attended) + the single-turn guard. False here = pre-response-window behavior
    # (absolute, non-attended fill) for full-sequence dLLMs (llada / nemotron); the
    # DiffusionGemma subclass overrides it to True.
    _response_window_collation: bool = False

    def setup(self):
        """Build all training components, then apply dLLM-specific overrides."""
        # Diffusion-LM training expects the user-specified ``torch_dtype`` to
        # be honored as the master-weight dtype. AM's default loading path
        # restores the on-disk dtype after load, which would silently downcast
        # an fp32 load back to the checkpoint's bf16 and break the standard
        # mixed-precision recipe (fp32 master + bf16 compute). Disable that
        # restoration here only — other recipes are unaffected.
        # ``self.cfg.model`` is a ``ConfigNode``; use attribute access (no
        # ``__setitem__``) and check ``__dict__`` for explicit user overrides.
        # Only set _restore_loaded_dtype for NeMo model loading paths — it is
        # an internal NeMo flag unknown to vanilla transformers.AutoModel, and
        # passing it to trust_remote_code models (e.g. DFlashDraftModel) raises
        # a TypeError.
        model_cfg = self.cfg.get("model", None)
        if model_cfg is not None and "_restore_loaded_dtype" not in model_cfg.__dict__:
            target = str(model_cfg.get("_target_", ""))
            if "nemo_automodel" in target:
                model_cfg._restore_loaded_dtype = False

        # Let parent build model, optimizer, dataloader, scheduler, etc.
        super().setup()

        # --- dLLM config ---
        dllm_cfg = self.cfg.get("dllm", None)
        if dllm_cfg is None:
            raise ValueError("Config must contain a 'dllm' section for DiffusionLMSFTRecipe")

        self.dllm_mode = dllm_cfg.get("mode", "mdlm")
        self.dllm_strategy = get_dllm_strategy(self.dllm_mode)
        if self.dllm_strategy.normalization_mode not in ("supervised", "noise"):
            raise ValueError(
                f"Invalid normalization_mode {self.dllm_strategy.normalization_mode!r} "
                f"from strategy {type(self.dllm_strategy).__name__}. "
                f"Must be 'supervised' or 'noise'."
            )

        self.dllm_eps = float(dllm_cfg.get("eps", 1e-3))
        self.dllm_block_size = dllm_cfg.get("block_size", None)
        if self.dllm_block_size is not None:
            self.dllm_block_size = int(self.dllm_block_size)
        hlr = dllm_cfg.get("half_life_ratio", 0.25)
        self.dllm_half_life_ratio = float(hlr) if hlr is not None else None

        # Padding config (two-stage block-aligned padding)
        pbs = dllm_cfg.get("pad_block_size", None)
        self.dllm_pad_block_size = int(pbs) if pbs is not None else None
        psld = dllm_cfg.get("pad_seq_len_divisible", None)
        self.dllm_pad_seq_len_divisible = int(psld) if psld is not None else None

        # Resolve mask_token_id — may stay None if the strategy's setup_extra() will set it.
        self.mask_token_id = dllm_cfg.get("mask_token_id", None)
        if self.mask_token_id is None:
            if (
                self.tokenizer is not None
                and hasattr(self.tokenizer, "mask_token_id")
                and self.tokenizer.mask_token_id is not None
            ):
                self.mask_token_id = self.tokenizer.mask_token_id
        if self.mask_token_id is not None:
            self.mask_token_id = int(self.mask_token_id)

        # --- Build dLLM loss function via strategy ---
        self.dllm_loss_fn = self.dllm_strategy.create_loss_fn(dllm_cfg)

        logger.info(
            f"dLLM SFT setup: mode={self.dllm_mode}, mask_token_id={self.mask_token_id}, "
            f"eps={self.dllm_eps}, block_size={self.dllm_block_size}, "
            f"half_life_ratio={self.dllm_half_life_ratio}, "
            f"normalization_mode={self.dllm_strategy.normalization_mode}"
        )

        # --- Wrap dataloader collate to produce unshifted format ---
        self._wrap_dataloader_collate()

        # Buffers for dLLM-specific metrics
        self._dllm_loss_buffer = []
        # Per-rank raw (correct, count) sums per block offset for DFlash draft
        # accuracy — SUM-allreduced across DP/CP, then divided to give global
        # per-position acceptance-length proxy plus the overall mean.
        self._dflash_correct_per_pos_buffer = []
        self._dflash_count_per_pos_buffer = []

        # --- Strategy post-setup hook (e.g. loads frozen target for DFlash) ---
        self.dllm_strategy.setup_extra(self)
        if self.mask_token_id is None:
            raise ValueError(
                "dllm.mask_token_id must be set in config, resolved by the tokenizer, or set by strategy.setup_extra()."
            )
        self.mask_token_id = int(self.mask_token_id)

    def _wrap_dataloader_collate(self):
        """Replace dataloader collate functions with the dLLM single-pass collater.

        Uses :class:`DLLMCollator` which goes directly from
        variable-length sample lists to block-aligned tensors in one pass.

        Requires datasets to produce unshifted format (``input_ids`` +
        ``loss_mask``, via ``_package_tokenized_example(unshifted=True)``).
        """
        pad_token_id = 0
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "pad_token_id")
            and self.tokenizer.pad_token_id is not None
        ):
            pad_token_id = self.tokenizer.pad_token_id

        eos_token_id = None
        if (
            self.tokenizer is not None
            and hasattr(self.tokenizer, "eos_token_id")
            and self.tokenizer.eos_token_id is not None
        ):
            eos_token_id = self.tokenizer.eos_token_id

        max_seq_len = self.cfg.get("dataset.seq_length", None)
        if max_seq_len is not None:
            max_seq_len = int(max_seq_len)

        dllm_cfg = self.cfg.get("dllm", {})
        supervise_padding = bool(dllm_cfg.get("supervise_padding", False))

        collator = DLLMCollator(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            block_size=self.dllm_pad_block_size,
            pad_seq_len_divisible=self.dllm_pad_seq_len_divisible,
            max_seq_len=max_seq_len,
            supervise_padding=supervise_padding,
            response_window=self._response_window_collation,
        )

        self.dataloader.collate_fn = collator
        for _name, val_dl in self.val_dataloaders.items():
            val_dl.collate_fn = collator

    def _apply_corruption(self, input_ids, loss_mask, microbatch_idx: int = 0):
        """Apply token corruption via the configured strategy.

        Args:
            input_ids: Clean token IDs, shape [B, L].
            loss_mask: Supervised positions mask, shape [B, L].
            microbatch_idx: Index of this microbatch within the step; folded into
                the corruption seed so distinct microbatches get distinct noise.

        Returns:
            Tuple of (noisy_input_ids, noise_mask, p_mask).
        """
        # Step/rank/microbatch-seeded generator so the D3PM corruption noise is a
        # deterministic function of (step, microbatch, rank) and reproduces exactly
        # on checkpoint resume. Corruption previously drew from the GLOBAL RNG, whose
        # state is not faithfully reinstated at the first post-resume draw, so a
        # resumed run trained on a different noise realization (the resume loss/grad
        # spike). Mirrors the already-resume-safe block-selection (+1<<42) and
        # self-conditioning (+0) step-seeded generators; the distinct (+2<<42) offset
        # decorrelates the three streams, and the rank term preserves per-DP-rank
        # noise diversity (each rank corrupts its own microbatch independently).
        step = int(self.step_scheduler.step)
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        seed = (
            int(getattr(self, "_self_cond_base_seed", 42))
            + 7919 * step
            + int(microbatch_idx)
            + 104729 * rank
            + (2 << 42)
        )
        gen = torch.Generator(device=input_ids.device).manual_seed(seed)
        return self.dllm_strategy.apply_corruption(
            input_ids,
            loss_mask,
            self.mask_token_id,
            eps=self.dllm_eps,
            block_size=self.dllm_block_size,
            half_life_ratio=self.dllm_half_life_ratio,
            generator=gen,
        )

    def _augment_batch_for_model(self, batch, *, clean_input_ids, loss_mask):
        """Add any model-specific forward inputs derived from the batch.

        Default is a no-op; ``block_diffusion`` overrides this to attach the
        block-causal attention masks and canvas position ids.
        """
        return batch

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_diffusion_tokens,
        num_ar_tokens=None,
        num_batches,
        is_train: bool = True,
    ):
        """Override: apply dLLM corruption and compute dLLM loss."""
        # Move batch to device
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) for dk, dv in v.items() if dv is not None}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }

        # Use pre-computed corruption if available (from _run_train_optim_step),
        # otherwise compute on the fly (validation path).
        if "_noise_mask" in batch:
            noisy_input_ids = batch.pop("_noisy_input_ids")
            noise_mask = batch.pop("_noise_mask")
            p_mask = batch.pop("_p_mask")
            clean_input_ids = batch.pop("_clean_input_ids")
            loss_mask = batch.pop("loss_mask")
        else:
            loss_mask = batch.pop("loss_mask")
            clean_input_ids = batch["input_ids"].clone()
            noisy_input_ids, noise_mask, p_mask = self._apply_corruption(clean_input_ids, loss_mask, microbatch_idx=idx)

        batch = self.dllm_strategy.prepare_batch(batch, noisy_input_ids, noise_mask, clean_input_ids)

        model = self.model_parts[0]

        # Context parallel setup (no labels to pass for dLLM)
        prepared_cp = self.cp_sharder.shard(None, batch)
        train_ctx, batch = prepared_cp.context, prepared_cp.batch
        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )

        autocast_dtype = getattr(self.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        )

        with train_ctx, sync_ctx, fp8_ctx, autocast_ctx:
            # Hook for model families that need extra forward inputs derived from
            # the (clean) batch — e.g. block_diffusion builds the block-causal
            # attention masks + canvas position ids here. No-op by default.
            batch = self._augment_batch_for_model(batch, clean_input_ids=clean_input_ids, loss_mask=loss_mask)
            batch = filter_forward_kwargs(model, batch)
            out = model(**batch)
            logits = getattr(out, "logits", out)
            # Hybrid models (e.g. Nemotron-Labs-Diffusion in block_diff mode)
            # also return causal_logits for the AR branch of the loss.  When
            # absent (e.g. pure-MDLM models like LLaDA), the AR branch is
            # silently skipped by HybridDiffusionLLMLoss / MDLMCrossEntropyLoss.
            causal_logits = getattr(out, "causal_logits", None)
            del out

            # Compute dLLM loss (unified interface via DLLMLossOutput)
            has_causal = causal_logits is not None
            loss_result = self.dllm_loss_fn(
                logits=logits,
                target_ids=clean_input_ids,
                noise_mask=noise_mask,
                p_mask=p_mask,
                loss_mask=loss_mask,
                loss_mask_ar=loss_mask if has_causal else None,
                num_diffusion_tokens=num_diffusion_tokens,
                num_ar_tokens=num_ar_tokens if has_causal else None,
                causal_logits=causal_logits,
            )
            microbatch_loss = loss_result.total_loss
            dllm_loss = loss_result.dllm_loss.detach().clone()

            loss_buffer.append(microbatch_loss.clone().detach())
            self._dllm_loss_buffer.append(dllm_loss)

            if is_train:
                (microbatch_loss * self._get_dp_group_size(include_cp=True)).backward()

    def _compute_loss_denominators(self, batches, num_noise_tokens, num_supervised_tokens):
        """Return ``(num_diffusion_tokens, num_ar_tokens)`` — the GLOBAL token
        denominators for this step's diffusion + AR losses.

        Base: the pre-mask counts (diffusion per the strategy's ``normalization_mode``,
        AR = supervised). Subclasses whose *final* loss masks differ from these raw
        counts (e.g. block-diffusion, which restricts the diffusion loss to one
        selected canvas block and drops padding) override this to count the final
        masks — otherwise the losses are divided by tokens that cannot contribute and
        are silently under-scaled.
        """
        if self.dllm_strategy.normalization_mode == "noise":
            num_diffusion_tokens = num_noise_tokens
        else:
            num_diffusion_tokens = num_supervised_tokens
        return num_diffusion_tokens, num_supervised_tokens

    def _run_train_optim_step(self, batches, max_grad_norm: Optional[float] = None):
        """Execute a single training step with dLLM loss.

        Follows the parent pattern but uses loss_mask from the collate wrapper
        instead of labels != -100 for token counting.
        """
        # Pre-process all microbatches (corruption for MDLM, target forwards for DFlash).
        # The strategy's pre_step stashes _noisy_input_ids/_noise_mask/_p_mask/_clean_input_ids
        # on each batch and threads microbatch_idx into the corruption seed (resume-safe).
        num_noise_tokens_raw, num_supervised_tokens_raw = self.dllm_strategy.pre_step(self, batches)
        num_noise_tokens = self._dp_allreduce(torch.tensor(num_noise_tokens_raw, dtype=torch.long)).item()
        num_supervised_tokens = self._dp_allreduce(torch.tensor(num_supervised_tokens_raw, dtype=torch.long)).item()

        # Global token denominators for the diffusion + AR losses (overridable so
        # subclasses whose FINAL loss masks differ from these raw counts can recount).
        num_diffusion_tokens, num_ar_tokens = self._compute_loss_denominators(
            batches, num_noise_tokens, num_supervised_tokens
        )

        loss_buffer = []

        # Count total tokens excluding tail padding
        num_tokens_in_batch = torch.tensor(sum(batch["input_ids"].numel() for batch in batches), dtype=torch.long)
        num_tokens_in_batch = self._dp_allreduce(num_tokens_in_batch).item()

        num_batches = len(batches)
        prepare_for_grad_accumulation(self.model_parts, pp_enabled=self.pp_enabled)

        # Optional one-shot FLOPs measurement for the perf report.
        # FlopCounterMode captures the ACTUAL per-GPU model FLOPs of one post-warmup
        # step (encoder + both decode passes + active experts + both lm heads) — which
        # an analytical formula can't easily express for this encoder-decoder MoE. Gated
        # by `measure_flops` (default off → no effect on normal / llada / nemotron runs);
        # attempted once, after `measure_flops_after` steps; any failure degrades to "no
        # TFLOPs" rather than crashing the step.
        _fcm = None
        if (
            bool(self.cfg.get("measure_flops", False))
            and not getattr(self, "_flop_measure_attempted", False)
            and self.step_scheduler.step >= int(self.cfg.get("measure_flops_after", 3))
        ):
            self._flop_measure_attempted = True
            try:
                from torch.utils.flop_counter import FlopCounterMode

                _fcm = FlopCounterMode(display=False)
                _fcm.__enter__()
            except Exception as e:
                logging.warning("FLOPs measurement unavailable, skipping: %s", e)
                _fcm = None

        for i, batch in enumerate(batches):
            if i == num_batches - 1:
                prepare_for_final_backward(self.model_parts, pp_enabled=self.pp_enabled)

            self.dllm_strategy.forward_backward(
                self,
                i,
                batch,
                loss_buffer=loss_buffer,
                num_diffusion_tokens=num_diffusion_tokens,
                num_ar_tokens=num_ar_tokens,
                num_batches=num_batches,
            )

            if i == 0:
                prepare_after_first_microbatch()

        if _fcm is not None:
            try:
                _fcm.__exit__(None, None, None)
                # Per-GPU FLOPs for one full step (all microbatches' fwd+bwd).
                self._step_flops_per_gpu = float(_fcm.get_total_flops())
                logging.info(
                    "Measured per-GPU step FLOPs: %.3e (TFLOPs reported on later clean steps)",
                    self._step_flops_per_gpu,
                )
            except Exception as e:
                logging.warning("FLOPs measurement failed, skipping: %s", e)
                self._step_flops_per_gpu = None

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
            num_label_tokens=num_ar_tokens,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        self.checkpointer.maybe_wait_for_staging()
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        # Precompute FP8 scales
        fp8_config = self.cfg.get("fp8", None)
        if (
            fp8_config is not None
            and fp8_config.get("enabled", False)
            and fp8_config.get("precompute_float8_dynamic_scale_for_fsdp", False)
            and not self.pp_enabled
            and self.device_mesh is not None
            and self.device_mesh["dp_shard"].size() > 1
        ):
            precompute_float8_dynamic_scale_for_fsdp(self.model_parts[0])

        t = time.perf_counter()
        time_delta = t - self.timestamp
        self.timestamp = t
        tps = num_tokens_in_batch / time_delta

        mfu = None
        mfu_calculator = getattr(self, "mfu_calculator", None)
        if batches and mfu_calculator is not None:
            step_flops = 0.0
            flops_supported = True
            for batch in batches:
                input_ids = batch.get("input_ids")
                if input_ids is None:
                    flops_supported = False
                    break
                batch_flops = mfu_calculator.get_flops(input_ids)
                if batch_flops is None:
                    flops_supported = False
                    break
                step_flops += float(batch_flops)

            if flops_supported:
                step_flops = self._dp_allreduce(
                    torch.tensor(step_flops, dtype=torch.float64, device=self.dist_env.device), include_cp=True
                ).item()
                mfu = calculate_mfu(step_flops / 1e12, self.dist_env.world_size, time_delta)

        total_loss = torch.sum(torch.stack(loss_buffer))
        total_loss = self._dp_allreduce(total_loss, include_cp=True).cpu().item()

        dllm_loss = self._dp_allreduce(torch.stack(self._dllm_loss_buffer).sum(), include_cp=True).item()
        self._dllm_loss_buffer.clear()

        # DFlash draft top-1 accuracy. Per-rank raw (correct, count) per block
        # offset are summed over grad-accum microbatches, SUM-allreduced across
        # DP+CP (same primitive as dllm_loss), then divided post-reduction to
        # give per-position acceptance-length proxy and the overall mean.
        # Buffers stay empty for non-DFlash modes, so draft_acc(_k) stays None.
        draft_acc = None
        draft_acc_per_pos = None
        if self._dflash_correct_per_pos_buffer:
            correct_per_pos = self._dp_allreduce(
                torch.stack(self._dflash_correct_per_pos_buffer).sum(dim=0), include_cp=True
            )
            count_per_pos = self._dp_allreduce(
                torch.stack(self._dflash_count_per_pos_buffer).sum(dim=0), include_cp=True
            )
            total_correct = correct_per_pos.sum().item()
            total_count = count_per_pos.sum().item()
            if total_count > 0:
                draft_acc = total_correct / total_count
            count_safe = count_per_pos.clamp_min(1.0)
            draft_acc_per_pos = (correct_per_pos / count_safe).tolist()
        self._dflash_correct_per_pos_buffer.clear()
        self._dflash_count_per_pos_buffer.clear()

        metrics = {
            "loss": total_loss,
            "dllm_loss": dllm_loss,
            "grad_norm": grad_norm,
            "lr": self.optimizer[0].param_groups[0]["lr"],
            "mem": torch.cuda.max_memory_allocated() / 1024**3,
            "tps": tps,
            "tps_per_gpu": tps / self._get_cp_group_size() / max(self._get_dp_group_size(), 1),
            "mfu": mfu,
            "tokens_per_step": num_tokens_in_batch,
            "supervised_tokens": num_supervised_tokens,
            "draft_acc": draft_acc,
            "mode": self.dllm_mode,
        }
        if draft_acc_per_pos is not None:
            for k, v in enumerate(draft_acc_per_pos, start=1):
                metrics[f"draft_acc_k{k}"] = v
        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics=metrics,
        )

    @torch.no_grad()
    def _run_validation_epoch(self, val_dataloader):
        """Run one validation pass with dLLM corruption and loss.

        Computes per-batch loss with proper denominators, then accumulates
        weighted by noise token count to produce a per-noise-token average
        across the val set.
        """
        with ScopedRNG(seed=1, ranked=True):
            for mp in self.model_parts:
                mp.eval()

            total_weighted_loss = torch.tensor(0.0, dtype=torch.float32, device=self.dist_env.device)
            total_norm_tokens = 0

            for batch in val_dataloader:
                # Pre-process this val batch via the strategy (mirrors training pre_step):
                # corrupts and stashes _noisy_input_ids/_noise_mask/_p_mask/_clean_input_ids
                # on the batch and returns the raw token counts.
                num_noise_raw, num_supervised_raw = self.dllm_strategy.pre_step(self, [batch])
                num_noise = self._dp_allreduce(torch.tensor(num_noise_raw, dtype=torch.long)).item()
                num_supervised = self._dp_allreduce(torch.tensor(num_supervised_raw, dtype=torch.long)).item()
                # Delegate to the (overridable) denominator hook so a subclass whose FINAL
                # loss masks differ from the raw counts (DiffusionGemma response window: one
                # selected canvas block for diffusion + attention-scoped AR) divides val
                # losses by the SAME counts as training -> comparable val/train curves. For
                # the base recipes this returns (num_noise if the strategy normalizes on
                # noise else num_supervised, num_supervised), so the val path is unchanged.
                num_diffusion_tokens, num_ar_tokens = self._compute_loss_denominators(
                    [batch], num_noise, num_supervised
                )
                num_norm = num_diffusion_tokens

                loss_buffer = []
                self.dllm_strategy.forward_backward(
                    self,
                    0,
                    batch,
                    loss_buffer=loss_buffer,
                    num_diffusion_tokens=num_diffusion_tokens,
                    num_ar_tokens=num_ar_tokens,
                    num_batches=1,
                    is_train=False,
                )

                # Accumulate: per-token-avg loss * norm_count
                batch_loss = torch.sum(torch.stack(loss_buffer)).item()
                batch_loss = self._dp_allreduce(
                    torch.tensor(batch_loss, dtype=torch.float32, device=self.dist_env.device),
                    include_cp=True,
                ).item()
                total_weighted_loss += batch_loss * num_norm
                total_norm_tokens += num_norm

        val_loss = total_weighted_loss / max(total_norm_tokens, 1e-8)
        val_loss = val_loss.item() if isinstance(val_loss, torch.Tensor) else val_loss

        # Clear dLLM loss buffer from validation
        self._dllm_loss_buffer.clear()
        self._dflash_correct_per_pos_buffer.clear()
        self._dflash_count_per_pos_buffer.clear()

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics={
                "val_loss": val_loss,
                "lr": self.optimizer[0].param_groups[0]["lr"],
                "num_label_tokens": total_norm_tokens,
                "mem": torch.cuda.max_memory_allocated() / 1024**3,
            },
        )

    def log_train_metrics(self, log_data):
        """Log dLLM-specific training metrics."""
        if not self.dist_env.is_main:
            return

        # Buffer numeric metrics every step; on a remote-logging step emit the MEAN
        # over the window so the wandb curve is de-noised (the per-step diffusion
        # loss is dominated by the random corruption level t). Console + file
        # loggers below stay per-step.
        if not hasattr(self, "_remote_log_window"):
            self._remote_log_window = []
        self._remote_log_window.append(
            {k: float(v) for k, v in log_data.metrics.items() if isinstance(v, (int, float))}
        )

        if self.step_scheduler.is_remote_logging_step:
            # Window mean of each numeric metric (Loss/Train_Total, Loss/Train_DLLM,
            # grad_norm, lr, ...) since the last remote log.
            _win = self._remote_log_window
            remote_metrics = {
                k: sum(d[k] for d in _win) / len(_win) for k in _win[0] if k not in ("step", "epoch", "timestamp")
            }
            self._remote_log_window = []
            if wandb.run is not None:
                wandb.log(remote_metrics, step=self.step_scheduler.step)
            if mlflow.active_run() is not None:
                mlflow.log_metrics(to_float_metrics(remote_metrics), step=log_data.step)
            if self.comet_logger is not None:
                self.comet_logger.log_metrics(remote_metrics, step=log_data.step)

        self.metric_logger_train.log(log_data)
        draft_acc = log_data.metrics.get("draft_acc")
        acc_str = "" if draft_acc is None else " | draft_acc {:.4f}".format(draft_acc)
        logging.info(
            "step {} | epoch {} | loss {:.4f} | dllm_loss {:.4f} | grad_norm {:.4f} | "
            "lr {:.2e} | mem {:.2f} GiB | tps {:.2f}({:.2f}/gpu){} | mode {}".format(
                log_data.step,
                log_data.epoch,
                log_data.metrics["loss"],
                log_data.metrics["dllm_loss"],
                log_data.metrics["grad_norm"],
                log_data.metrics["lr"],
                log_data.metrics["mem"],
                log_data.metrics["tps"],
                log_data.metrics["tps_per_gpu"],
                acc_str,
                log_data.metrics["mode"],
            )
        )
        torch.cuda.reset_peak_memory_stats()


class DiffusionGemmaSFTRecipe(DiffusionLMSFTRecipe):
    """dLLM SFT recipe for the ``diffusion_gemma`` block-diffusion model.

    Extends :class:`DiffusionLMSFTRecipe` (``dllm.mode = block_diffusion``) with
    the model-specific **response-window canvas** wiring (single-turn v1):

    * The encoder sees the **clean full sequence** (prompt + response). The
      decoder canvas is the **noised response region only**: per example the
      contiguous supervised suffix ``[prefix_len, S)`` is left-aligned into a
      ``[B, R]`` canvas (``R = S - min(prefix_len)``; shorter responses are
      right-padded). Canvas position ``0`` is the first response token, so the
      diffusion block boundaries align to the response — block ``i`` is
      bidirectional within itself and attends (offset-block-causal, strict
      ``>``) the clean prompt + clean earlier response blocks in the encoder KV.
      This matches the reference inference contract (each block conditioned on
      the prompt + already-generated blocks).
    * The block-causal masks (full + sliding) are built by
      :func:`build_block_diffusion_training_mask` with per-example
      ``prefix_lengths = prompt length``, ``response_length = response length``,
      ``enc_len = full sequence length``; ``decoder_position_ids`` are the
      response tokens' absolute positions so their query RoPE aligns with the
      encoder key RoPE.
    * Because the model returns **canvas-only** logits (``[B, R, V]``), the loss
      tensors (``target_ids`` / ``noise_mask`` / ``loss_mask`` / ``p_mask``) are
      sliced to the same response window in :meth:`_forward_backward_step` so
      they align with the logits (the inherited path passes full-sequence
      targets against canvas-only logits → misalignment).
    * The two-pass self-conditioning is orchestrated inside ``model.forward``,
      so the recipe still calls ``model(**batch)`` once.

    **Single-turn assumption.** The response is taken to be the single
    contiguous supervised suffix (:meth:`BlockDiffusionStrategy.split_prompt_response`).
    Multi-turn ``ChatDataset`` ``loss_mask`` (``0..0 1..1 0..0 1..1``) is not a
    contiguous suffix, so the prompt|response split is ill-defined; interleaved
    multi-turn masking is deferred.

    ``corrupt_uniform_random`` does not use a ``mask_token_id`` (random-token
    corruption); this recipe injects a harmless default so the parent setup,
    which expects one, does not fail.
    """

    # Response-window collation: response-relative + attended EOS-fill, single-turn
    # guard (matches Google's ChunkResponseIntoCanvases). See DLLMCollator.
    _response_window_collation: bool = True

    def setup(self):
        """Inject a dummy mask_token_id (unused by block_diffusion) then build."""
        dllm_cfg = self.cfg.get("dllm", None)
        if dllm_cfg is not None and dllm_cfg.get("mask_token_id", None) is None:
            # Random-token corruption ignores this; satisfy the parent's check.
            dllm_cfg.mask_token_id = 0
        super().setup()

        model = self.model_parts[0]

        # Canvas/block geometry for the block-causal mask. Prefer the model's
        # own config; fall back to the dllm.block_size config / 256.
        self.canvas_length = int(getattr(model, "canvas_length", self.dllm_block_size or 256))
        text_config = getattr(model, "text_config", None)
        self.block_diffusion_sliding_window = int(getattr(text_config, "sliding_window", 1024)) if text_config else 1024

        # Per-step probability of running the two-pass self-conditioning (the
        # ~0.5 zero-feed branch of Analog-Bits). The base seed makes the per-step
        # decision identical across DP ranks (see _decide_self_conditioning).
        dllm_cfg = self.cfg.get("dllm", {})
        self.self_conditioning_p = float(dllm_cfg.get("self_conditioning_p", 0.5))
        self._self_cond_base_seed = int(self.cfg.get("seed", 42))

        # Co-trained autoregressive encoder loss (matches Google: total = diffusion
        # + encoder_loss_weight * AR). The pad id (default 0) masks AR loss at pads.
        self._encoder_loss_weight = float(dllm_cfg.get("encoder_loss_weight", 1.0))
        tok = getattr(self, "tokenizer", None)
        pad_id = getattr(tok, "pad_token_id", None) if tok is not None else None
        self._pad_token_id = int(pad_id) if pad_id is not None else 0

        # Freeze the router on the live model (the config flag also freezes it in
        # __init__, but freezing here is idempotent and covers from_config paths).
        if getattr(model, "freeze_router", False) and hasattr(model, "freeze_router_params"):
            model.freeze_router_params()

    def _compute_loss_denominators(self, batches, num_noise_tokens, num_supervised_tokens):
        """Count the GLOBAL denominators from the FINAL response-window masks.

        The raw pre-mask counts over-count: ``_build_response_window`` restricts the
        diffusion loss to ONE step-selected canvas block (one-canvas scheme) and drops
        padding, while the AR loss is scored on the padding-masked, next-token-shifted
        supervised mask. Dividing by ``num_noise_tokens`` / ``num_supervised_tokens``
        would include tokens that cannot contribute, under-scaling diffusion (~by the
        response-block count) and AR (by padding + the shift). We build each
        microbatch's window once here (caching it on the batch so the forward reuses
        the *same* step-seeded block selection) and sum the actually-scored tokens.
        """
        device = self.dist_env.device
        num_diffusion = 0
        num_ar = 0
        for microbatch_idx, batch in enumerate(batches):
            clean = batch["_clean_input_ids"].to(device)
            noisy = batch["_noisy_input_ids"].to(device)
            noise_mask = batch["_noise_mask"].to(device)
            p_mask = batch["_p_mask"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            attn = batch.get("attention_mask", None)
            attn = attn.to(device) if attn is not None else None
            window = self._build_response_window(
                clean, noisy, noise_mask, loss_mask, p_mask, attention_mask=attn, microbatch_idx=microbatch_idx
            )
            # Cache so _forward_backward_step reuses the same window (same one-canvas
            # selection) instead of rebuilding it.
            batch["_window"] = window
            # Diffusion: ALL supervised canvas tokens in the step-selected block (= the
            # loss numerator's support and Google's target_mask; NOT noise-gated). Using
            # noise_mask & loss_mask (corrupted-only) mismatched the all-supervised loss
            # numerator -> a random ~1/t inflation of loss and gradients.
            num_diffusion += int(window["loss_mask"].sum().item())
            # AR: next-token pairs over the FULL valid sequence (prompt + canvas +
            # EOS-fill), matching Google's encoder_target_mask = full_valid &
            # shifted_valid where full_valid = concat([prompt!=pad, canvas_mask]).
            # With the collator now attending the EOS block-fill, attention_mask IS
            # that full_valid (prompt+response+EOS, excluding only global pad), so
            # the AR denominator counts the same positions the AR loss scores.
            ar = attn.to(dtype=torch.bool) if attn is not None else loss_mask.bool()
            num_ar += int((ar[:, :-1] & ar[:, 1:]).sum().item())
        num_diffusion = self._dp_allreduce(torch.tensor(num_diffusion, dtype=torch.long)).item()
        num_ar = self._dp_allreduce(torch.tensor(num_ar, dtype=torch.long)).item()
        # Guard a degenerate all-pad step against divide-by-zero in the loss.
        return max(num_diffusion, 1), max(num_ar, 1)

    def _build_response_window(
        self,
        clean_input_ids,
        noisy_input_ids,
        noise_mask,
        loss_mask,
        p_mask,
        attention_mask=None,
        microbatch_idx: int = 0,
    ):
        """Slice the batch to the response window and build the block-causal masks.

        Returns a dict with the canvas-region tensors (all ``[B, R]``) and the
        forward inputs (canvas ids, masks, position ids). ``R`` is the longest
        response in the batch; shorter responses are right-padded and the pad
        positions are dropped from the (sliced) ``loss_mask`` / ``noise_mask`` so
        they never contribute to the loss.
        """
        batch_size, seq_len = clean_input_ids.shape
        device = clean_input_ids.device

        prefix_lengths, _ = BlockDiffusionStrategy.split_prompt_response(clean_input_ids, loss_mask)
        if attention_mask is not None:
            effective_lengths = attention_mask.to(device=device, dtype=torch.long).sum(dim=1)
        else:
            effective_lengths = torch.full((batch_size,), seq_len, device=device, dtype=torch.long)
        response_lengths = (effective_lengths - prefix_lengths).clamp(min=0)  # [B]

        # NOTE: the single-turn requirement (the supervised response is a single
        # contiguous run, not multiple assistant turns) is enforced on the RAW
        # per-sample loss_mask in DLLMCollator, BEFORE the EOS block-fill. Checking
        # it here (post-collation) is wrong: the EOS-fill legitimately appends a
        # second supervised run after any trailing unsupervised token (e.g. a single
        # EOS that some datasets leave loss=0 after the response), which is benign —
        # the canvas stays contiguous and the unscored interior token is harmless.
        canvas_len = int(response_lengths.max().item()) if batch_size else 0
        # Degenerate batch with no supervised tokens: keep a 1-wide canvas so the
        # forward/loss shapes stay valid; the empty loss_mask zeroes the loss.
        canvas_len = max(canvas_len, 1)

        # Gather index of each canvas position into the absolute sequence:
        # canvas pos j of example b <- absolute pos prefix_lengths[b] + j.
        offsets = torch.arange(canvas_len, device=device)[None, :]  # [1, R]
        abs_idx = prefix_lengths[:, None] + offsets  # [B, R]
        valid = (abs_idx < effective_lengths[:, None]) & (
            offsets < response_lengths[:, None]
        )  # [B, R] real response token
        gather_idx = abs_idx.clamp(max=seq_len - 1)  # clamp pad positions for gather

        def _gather(t):
            return torch.gather(t, 1, gather_idx)

        canvas_ids = _gather(noisy_input_ids)
        target_ids = _gather(clean_input_ids)
        # Pad canvas positions are unsupervised: AND with `valid`.
        canvas_noise = _gather(noise_mask) & valid
        canvas_loss = _gather(loss_mask.bool()) & valid
        canvas_p = _gather(p_mask)
        # Decoder positions are the response tokens' absolute positions (RoPE
        # alignment with the clean encoder keys). Pad rows keep the clamped value
        # — harmless, those rows are unsupervised and masked.
        decoder_position_ids = abs_idx.clamp(max=seq_len - 1)

        block_size = self.dllm_block_size or self.canvas_length

        # One-canvas-per-step (match Google): restrict the diffusion loss to a
        # single randomly-chosen response block per example. The forward still
        # denoises all blocks, but only the selected block's corrupted tokens are
        # scored, so the gradient matches Google's one-canvas scheme. Selection is
        # per-example (no cross-rank control flow), step-seeded for reproducibility.
        canvas_block_id = (offsets // block_size).expand(batch_size, -1)  # [B, R]
        num_valid_blocks = ((response_lengths - 1).clamp(min=0) // block_size + 1).clamp(min=1)  # [B]
        step = int(getattr(getattr(self, "step_scheduler", None), "step", 0))
        # Per-(step, microbatch) seed so each grad-accumulation microbatch selects an
        # INDEPENDENT block — a per-step-only seed reused the same uniform draw across
        # microbatches, biasing the selection. Offset by a large constant to
        # decorrelate this stream from the self-conditioning coin (which seeds with
        # base + 7919*step + microbatch_idx) so the two never collide.
        sel_seed = int(getattr(self, "_self_cond_base_seed", 42)) + 7919 * step + int(microbatch_idx) + (1 << 42)
        gen = torch.Generator().manual_seed(sel_seed)
        sel_block = torch.floor(torch.rand(batch_size, generator=gen) * num_valid_blocks.float().cpu()).long()
        sel_block = sel_block.to(device)  # [B]
        canvas_loss = canvas_loss & (canvas_block_id == sel_block[:, None])

        sliding_window = self.block_diffusion_sliding_window
        mask_dtype = getattr(self.distributed_config, "autocast_dtype", None) or torch.float32
        mask_full, mask_sliding = self._build_batched_block_mask(
            prefix_lengths=prefix_lengths,
            response_lengths=response_lengths,
            canvas_len=canvas_len,
            enc_len=seq_len,
            block_size=block_size,
            sliding_window=sliding_window,
            device=device,
            dtype=mask_dtype,
        )

        encoder_positions = torch.arange(seq_len, device=device)[None, :].expand(batch_size, -1)
        encoder_padding_mask = (
            attention_mask.to(device=device, dtype=torch.bool).logical_not() if attention_mask is not None else None
        )
        return {
            "canvas_ids": canvas_ids,
            "target_ids": target_ids,
            "noise_mask": canvas_noise,
            "loss_mask": canvas_loss,
            "p_mask": canvas_p,
            "decoder_attention_mask": {"full_attention": mask_full, "sliding_attention": mask_sliding},
            "encoder_position_ids": encoder_positions,
            "encoder_padding_mask": encoder_padding_mask,
            "decoder_position_ids": decoder_position_ids,
            "decoder_padding_mask": valid.logical_not(),
        }

    @staticmethod
    def _build_batched_block_mask(
        *, prefix_lengths, response_lengths, canvas_len, enc_len, block_size, sliding_window, device, dtype
    ):
        """Assemble the ``[B, 1, R, enc_len + R]`` block-causal mask from per-example builds.

        Each example satisfies ``build_block_diffusion_training_mask``'s
        ``prefix + response_length == enc_len`` contract exactly, so the builder
        is called once per example (batch is small) and the result is placed into
        a padded additive mask. Pad query rows (``j >= response_length``) are kept
        well-defined by leaving their canvas self-diagonal unmasked, so the shared
        transformer never sees an all-``-inf`` softmax row; those rows are
        unsupervised and discarded from the loss.
        """
        batch_size = int(prefix_lengths.shape[0])
        key_len = enc_len + canvas_len
        neg = torch.finfo(dtype).min
        mask_full = torch.full((batch_size, 1, canvas_len, key_len), neg, dtype=dtype, device=device)
        mask_sliding = torch.full((batch_size, 1, canvas_len, key_len), neg, dtype=dtype, device=device)

        for b in range(batch_size):
            resp = int(response_lengths[b].item())
            if resp > 0:
                full_b, sliding_b = build_block_diffusion_training_mask(
                    prefix_lengths=int(prefix_lengths[b].item()),
                    response_length=resp,
                    enc_len=enc_len,
                    block_size=block_size,
                    sliding_window=sliding_window,
                    batch_size=1,
                    device=device,
                    dtype=dtype,
                )
                # Encoder columns [0, enc_len) and this example's canvas columns
                # [enc_len, enc_len + resp) map directly; the rest stay masked.
                mask_full[b, 0, :resp, :enc_len] = full_b[0, 0, :, :enc_len]
                mask_full[b, 0, :resp, enc_len : enc_len + resp] = full_b[0, 0, :, enc_len:]
                mask_sliding[b, 0, :resp, :enc_len] = sliding_b[0, 0, :, :enc_len]
                mask_sliding[b, 0, :resp, enc_len : enc_len + resp] = sliding_b[0, 0, :, enc_len:]
            # Every canvas query row attends at least its own canvas position so
            # no softmax row is fully masked (real rows already keep this via M_BD).
            diag = enc_len + torch.arange(canvas_len, device=device)
            mask_full[b, 0, torch.arange(canvas_len, device=device), diag] = 0
            mask_sliding[b, 0, torch.arange(canvas_len, device=device), diag] = 0
        return mask_full, mask_sliding

    def _decide_self_conditioning(self, batch_size: int, microbatch_idx: int = 0) -> torch.Tensor:
        """Per-EXAMPLE two-pass self-conditioning coins -> ``[B]`` bool tensor.

        Google draws the coin PER EXAMPLE (``uniform(B) < p``) and mixes
        conditioned / zero-conditioned examples within a batch. Returns a ``[B]``
        mask so the model gates the self-cond branch per example, while pass-1 (the
        no_grad self-cond signal) ALWAYS runs in the forward. Always running pass-1
        makes the FSDP collectives identical every step regardless of the coins (no
        rank desync) and keeps it correct for ``local_batch_size > 1`` — a scalar
        per-microbatch coin only matched Google's per-example mix when ``B == 1``.
        Seeded by ``(step, microbatch_idx)`` for reproducibility; rank-correlation
        of the coins is harmless now that pass-1 no longer branches on them.
        """
        step = getattr(getattr(self, "step_scheduler", None), "step", 0)
        seed = self._self_cond_base_seed + 7919 * int(step) + int(microbatch_idx)
        gen = torch.Generator().manual_seed(seed)
        return torch.rand(batch_size, generator=gen) < self.self_conditioning_p

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_diffusion_tokens,
        num_ar_tokens=None,
        num_batches,
        is_train: bool = True,
    ):
        """Response-window forward/backward: canvas-only logits + canvas-sliced loss."""
        # Reuse the window built during denominator counting (same step-seeded
        # one-canvas selection). Pop before the to-device sweep, which can't handle
        # the nested decoder-mask dict.
        cached_window = batch.pop("_window", None)
        batch = {
            k: (
                {dk: dv.to(self.dist_env.device, non_blocking=True) for dk, dv in v.items() if dv is not None}
                if isinstance(v, dict)
                else (v.to(self.dist_env.device, non_blocking=True) if isinstance(v, torch.Tensor) else v)
            )
            for k, v in batch.items()
        }

        if "_noise_mask" in batch:
            noisy_input_ids = batch.pop("_noisy_input_ids")
            noise_mask = batch.pop("_noise_mask")
            p_mask = batch.pop("_p_mask")
            clean_input_ids = batch.pop("_clean_input_ids")
            loss_mask = batch.pop("loss_mask")
        else:
            loss_mask = batch.pop("loss_mask")
            clean_input_ids = batch["input_ids"].clone()
            noisy_input_ids, noise_mask, p_mask = self._apply_corruption(clean_input_ids, loss_mask, microbatch_idx=idx)

        attention_mask = batch.get("attention_mask", None)
        # Encoder input + drop attention_mask/use_cache; canvas slicing follows.
        batch = self.dllm_strategy.prepare_batch(batch, noisy_input_ids, noise_mask, clean_input_ids)
        # Valid mask for the AR encoder loss = the FULL non-pad sequence (prompt +
        # response + EOS-fill) = concat([prompt != pad, canvas_mask]). The encoder is a
        # standard causal LM here, so it is trained to predict the next token
        # across the prompt and the canvas (incl. the EOS-fill), not only the
        # response. attention_mask now spans exactly that span (EOS-fill attended);
        # num_ar_tokens counts the matching shifted pairs.
        if attention_mask is not None:
            ar_full_loss_mask = attention_mask.to(device=loss_mask.device, dtype=torch.bool)
        else:
            ar_full_loss_mask = loss_mask.bool()
        window = (
            cached_window
            if cached_window is not None
            else self._build_response_window(
                clean_input_ids,
                noisy_input_ids,
                noise_mask,
                loss_mask,
                p_mask,
                attention_mask=attention_mask,
                microbatch_idx=idx,
            )
        )

        batch["canvas_ids"] = window["canvas_ids"]
        batch["decoder_attention_mask"] = window["decoder_attention_mask"]
        batch["encoder_position_ids"] = window["encoder_position_ids"]
        if window["encoder_padding_mask"] is not None:
            batch["encoder_padding_mask"] = window["encoder_padding_mask"]
        batch["decoder_position_ids"] = window["decoder_position_ids"]
        batch["decoder_padding_mask"] = window["decoder_padding_mask"]
        batch["do_self_conditioning"] = self._decide_self_conditioning(clean_input_ids.shape[0], idx)
        # Canvas-region loss tensors aligned with the canvas-only logits.
        target_ids = window["target_ids"]
        noise_mask = window["noise_mask"]
        loss_mask = window["loss_mask"]
        p_mask = window["p_mask"]

        model = self.model_parts[0]
        prepared_cp = self.cp_sharder.shard(None, batch)
        train_ctx, batch = prepared_cp.context, prepared_cp.batch
        fp8_ctx = self.te_fp8.maybe_te_autocast() if self.te_fp8 is not None else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )
        autocast_dtype = getattr(self.distributed_config, "autocast_dtype", None)
        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype) if autocast_dtype is not None else nullcontext()
        )

        with train_ctx, sync_ctx, fp8_ctx, autocast_ctx:
            batch = filter_forward_kwargs(model, batch)
            out = model(**batch)
            logits = getattr(out, "logits", out)
            encoder_logits = getattr(out, "encoder_logits", None)
            del out

            loss_result = self.dllm_loss_fn(
                logits=logits,
                target_ids=target_ids,
                noise_mask=noise_mask,
                p_mask=p_mask,
                loss_mask=loss_mask,
                loss_mask_ar=None,
                num_diffusion_tokens=num_diffusion_tokens,
                num_ar_tokens=None,
                causal_logits=None,
            )
            microbatch_loss = loss_result.total_loss
            dllm_loss = loss_result.dllm_loss.detach().clone()

            # Co-trained autoregressive encoder loss: next-token CE over the clean
            # sequence, added at encoder_loss_weight, only when training (encoder_logits
            # is produced by the forward only then). The SCOPE matches Google (full_valid:
            # prompt + canvas + EOS-fill, via attention_mask); the REDUCTION does NOT —
            # we use token-normalization (Σ CE / Σ tokens over the global count) whereas
            # Google takes a per-example mean (Σ_b CE_b / N_b, then mean over B).
            # Token-norm weights long examples more; it is kept deliberately (#4 decision)
            # for consistency with the diffusion loss's global-token normalization and the
            # recipe's `* dp_group_size` backward scaling.
            if is_train and encoder_logits is not None and self._encoder_loss_weight > 0.0:
                # Normalize by the GLOBAL supervised-token count (num_ar_tokens),
                # exactly like the diffusion loss uses num_diffusion_tokens, so the
                # recipe's `* dp_group_size` backward scaling is correct for both
                # terms (a local mean here would be over-scaled by dp_group_size).
                ar_loss = encoder_ar_loss(
                    encoder_logits,
                    clean_input_ids,
                    valid_mask=ar_full_loss_mask,
                    num_tokens=num_ar_tokens,
                )
                microbatch_loss = microbatch_loss + self._encoder_loss_weight * ar_loss

            loss_buffer.append(microbatch_loss.clone().detach())
            self._dllm_loss_buffer.append(dllm_loss)

            if is_train:
                (microbatch_loss * self._get_dp_group_size(include_cp=True)).backward()


# Entry point
def main(config_path=None):
    """Main entry point for dLLM SFT recipe."""
    if config_path is None:
        config_path = "examples/dllm_sft/mdlm_sft.yaml"
    cfg = parse_args_and_load_config(config_path)
    recipe_name = cfg.get("recipe", "DiffusionLMSFTRecipe")
    recipe_cls = DiffusionGemmaSFTRecipe if recipe_name == "DiffusionGemmaSFTRecipe" else DiffusionLMSFTRecipe
    trainer = recipe_cls(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
