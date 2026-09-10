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

"""Shared training-loop utilities for the speculative-decoding recipes.

EAGLE-1/2, EAGLE-3, and DFlash all hand-roll the same gradient-accumulation
bookkeeping (ceil optimizer-steps-per-epoch and the DDP ``no_sync`` skip) and the
same warmup + cosine LR schedule. Centralizing them here keeps the recipes from
drifting apart when one is fixed and the others are missed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from nemo_automodel.components.quantization.fp8 import apply_fp8_to_model, build_fp8_config
from nemo_automodel.components.utils.compile_utils import build_compile_config, compile_module_inplace


def apply_draft_fp8(draft_model: nn.Module, cfg_fp8: Any) -> None:
    """Optionally convert the draft's ``nn.Linear`` layers to torchao ``Float8Linear``, in place.

    ``cfg_fp8`` is the recipe's top-level ``fp8:`` YAML block (dict-like, same
    surface as the SFT recipe's -- see ``FP8Config``). No-op when the block is
    absent or ``enabled`` is false. Modifies the draft in place (never
    reassign ``self.draft_model`` -- ``BaseRecipe.__setattr__`` rejects
    re-tracking an ``nn.Module`` attribute). Must be called before the draft is
    wrapped (DDP / ``fully_shard``) so the swapped modules are what gets
    replicated or sharded.
    """
    if cfg_fp8 is None:
        return
    apply_fp8_to_model(draft_model, config=build_fp8_config(cfg_fp8))


def apply_draft_compile(draft_model: nn.Module, cfg_compile: Any) -> None:
    """Optionally ``torch.compile`` the draft in place (top-level ``compile:`` block).

    Same YAML surface as the SFT recipes (``CompileConfig``); no-op when the
    block is absent or ``enabled`` is false. Uses ``nn.Module.compile()`` so
    the draft object and its state-dict keys are unchanged (the recipes track
    the module by reference and checkpoint it directly). Must run after
    ``apply_draft_fp8`` so inductor traces the swapped ``Float8Linear``
    modules: fp8's cast/scale ops only pay off once fused into the GEMM
    prologue, and in eager mode fp8 is typically slower than bf16.
    """
    if cfg_compile is None:
        return
    compile_module_inplace(draft_model, build_compile_config(cfg_compile))


def raise_if_peft_configured(cfg: Any, recipe_name: str) -> None:
    """Fail fast on EAGLE-3-only draft knobs set on a recipe that does not support them.

    ``peft:``: the DFlash-family and DSpark drafts register trainable non-LoRA
    modules on the draft itself (Domino's ``prefix_gru``/``embed_proj``, DSpark's
    Markov and confidence heads); LoRA's freeze-everything-but-adapters contract
    would silently freeze them, so reject the config instead of ignoring it.
    ``recipe_args.draft_weights_path``: only the EAGLE-3 recipe implements the
    warm-start load; a silently ignored knob would train from random init while
    the user believes the draft was warm-started.
    """
    if cfg.get("peft", None) is not None:
        raise ValueError(
            f"peft is not supported by {recipe_name}; LoRA draft training is only available in the EAGLE-3 recipe."
        )
    recipe_args = cfg.get("recipe_args", None)
    if recipe_args is not None and recipe_args.get("draft_weights_path", None):
        raise ValueError(
            f"recipe_args.draft_weights_path is not supported by {recipe_name}; the draft warm start is only "
            "available in the EAGLE-3 recipe."
        )


def optim_steps_per_epoch(num_batches_per_epoch: int, grad_accumulation_steps: int) -> int:
    """Return ceil(num_batches / accum), the actual number of optimizer steps per epoch.

    Floor division silently drops the trailing partial accumulation window
    (up to ``grad_accumulation_steps - 1`` micro-batches) from the LR
    scheduler's view of training, even though the trainer flushes those
    gradients with an explicit step. Ceil keeps the scheduler aligned with
    the actual number of ``optimizer.step()`` calls.
    """
    if num_batches_per_epoch <= 0 or grad_accumulation_steps <= 0:
        return 0
    return -(-num_batches_per_epoch // grad_accumulation_steps)


def should_sync_grads(
    *,
    pending_micro_batches: int,
    grad_accumulation_steps: int,
    batch_idx: int,
    batches_per_epoch: int | None,
    is_ddp: bool,
) -> bool:
    """Return True when this micro-batch's backward should all-reduce gradients.

    Under DDP with gradient accumulation only the micro-batch immediately
    followed by an ``optimizer.step()`` needs to synchronize: at that point the
    locally-accumulated ``.grad`` already holds the whole window's contribution,
    so a single all-reduce averages the complete window and the intervening
    micro-batches can run under ``no_sync()`` -- saving ``grad_accumulation_steps - 1``
    all-reduces per window. That step is either the window closer
    (``pending_micro_batches + 1 == grad_accumulation_steps``) or the epoch's
    final batch (which the trailing-flush step consumes). When the dataloader
    length is unknown we cannot identify the final batch, so we sync every step
    (correct, just no speedup). With a single process (no DDP) there is nothing
    to synchronize, so this is always True.
    """
    if not is_ddp or batches_per_epoch is None:
        return True
    closes_window = pending_micro_batches + 1 == grad_accumulation_steps
    is_last_batch = batch_idx == batches_per_epoch - 1
    return closes_window or is_last_batch


def make_warmup_cosine_schedule(
    warmup_steps: int, total_optim_steps: int, min_lr_ratio: float
) -> Callable[[int], float]:
    """Build the ``LambdaLR`` multiplier: linear warmup then cosine decay to ``min_lr_ratio``.

    Linear from 0 to 1 over the first ``warmup_steps`` optimizer steps, then a
    cosine from 1 down to ``min_lr_ratio`` over the remaining steps. Shared by the
    EAGLE and DFlash recipes, which train the draft from scratch and diverge under
    a flat LR after the first epoch.
    """

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return _lr_lambda


def raise_if_target_not_materialized(model: nn.Module, target_path: str) -> None:
    """Fail loudly when a frozen target's weights were never loaded from disk.

    ``NeMoAutoModel*.from_pretrained`` initializes on the meta device whenever
    ``world_size > 1`` and no sharding infrastructure was requested, on the
    assumption that a later FSDP2 / MegatronFSDP wrap materializes the weights.
    A draft recipe that keeps one full frozen replica per rank never performs
    that step, so ``.to(device)`` turns the meta tensors into uninitialized
    memory and the target silently becomes a randomly initialized teacher: the
    draft then trains against noise, loss stalls at ``ln(vocab)``, and nothing
    raises.

    This check is cheap (one norm per checked tensor) and runs once at setup.
    It compares against the theoretical norm of HuggingFace's default
    ``N(0, 0.02)`` initialization, which is what uninitialized-then-copied
    memory ends up resembling.

    Args:
        model: The frozen target model, already moved to its device.
        target_path: Model id or path, echoed in the error message.

    Raises:
        RuntimeError: If a parameter is still on the meta device, or if the
            input embedding matches an untrained initialization.
    """
    meta = [name for name, param in model.named_parameters() if param.is_meta]
    if meta:
        raise RuntimeError(
            f"The frozen target {target_path} still has {len(meta)} parameter(s) on the meta device "
            f"(e.g. {meta[0]}). It was initialized on meta and never materialized, so training would "
            "distill the draft against random weights. Load the target without meta-device init."
        )

    embedding = model.get_input_embeddings()
    weight = getattr(embedding, "weight", None)
    if weight is None or weight.numel() == 0:
        return
    observed = weight.detach().float().norm().item()
    # HF initializes embeddings from N(0, initializer_range); the norm of such a
    # matrix concentrates tightly around std * sqrt(numel).
    # A diagnostic must never be the thing that breaks a run, so every lookup
    # it makes degrades to the HuggingFace default rather than raising.
    config = getattr(model, "config", None)
    std = float(getattr(config, "initializer_range", 0.02) or 0.02)
    untrained = std * math.sqrt(weight.numel())
    if untrained > 0 and abs(observed - untrained) / untrained < 0.02:
        raise RuntimeError(
            f"The frozen target {target_path} has an input embedding whose norm ({observed:.2f}) matches an "
            f"untrained N(0, {std}) initialization ({untrained:.2f}), so its weights were never loaded from "
            "the checkpoint. This happens when the model is initialized on the meta device for a "
            "world_size > 1 run and nothing materializes it. Training would distill the draft against a "
            "random teacher without raising."
        )
