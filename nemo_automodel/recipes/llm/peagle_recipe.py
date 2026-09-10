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

"""P-EAGLE recipe-level logic, split out of the EAGLE-3 recipe.

``PeagleRecipeMixin`` holds the P-EAGLE-only step methods so ``TrainEagle3Recipe``
keeps only the shared EAGLE-3 training flow plus the ``parallel_drafting`` /
``_peagle_partitioned`` dispatch. The mixin relies on the recipe attributes it is
mixed into (``device``, ``target_wrapper``, ``trainer_module``,
``grad_accumulation_steps``, ``_module``).
"""

from contextlib import nullcontext
from types import SimpleNamespace

import torch
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.speculative.eagle import PEagleTrainerModule


class PeagleRecipeMixin:
    """P-EAGLE setup and sequence-partitioning step methods for ``TrainEagle3Recipe``."""

    # True only for P-EAGLE with ``sequence_partitions > 1`` (set in ``setup``);
    # the train loop then drives the per-segment forward/backward step.
    _peagle_partitioned = False

    def _configure_peagle_draft_config(self, recipe_cfg, draft_config, draft_base_config) -> int:
        """Validate P-EAGLE recipe args and populate ``draft_config``; return ``mask_token_id``.

        Mutates ``draft_config`` in place with the P-EAGLE keys (``mask_token_id``,
        ``num_depths``, COD ratios, draft ``num_hidden_layers``) that are serialized
        into the saved draft ``config.json`` so the checkpoint loads into vLLM's
        parallel-drafting runtime unchanged. Called only on the ``parallel_drafting``
        branch of ``setup``.
        """
        # ``mask_token_id`` must be chosen deliberately: it indexes the draft
        # ``embed_tokens`` table for the masked draft slots and (with frozen
        # embeddings) that row's vector is the only "mask token" signal. A
        # silent default (e.g. 0) usually maps to a meaningful token and
        # pollutes the masked-slot semantics, so require it explicitly and
        # range-check it. The value is serialized into the draft config so
        # vLLM reads the same id at serve time.
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "parallel_drafting=True requires recipe_args.mask_token_id to be set explicitly. "
                "Pick a reserved / rarely-used token id for the masked draft slots (e.g. one of "
                "Llama-3's reserved special tokens) so the masked-slot embedding does not collide "
                "with real content; the same id must be present in the draft config.json vLLM reads "
                "at serve time."
            )
        mask_token_id = int(mask_token_id)
        if not 0 <= mask_token_id < draft_base_config.vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of range for the target vocab "
                f"[0, {draft_base_config.vocab_size}); it indexes the draft embed_tokens table."
            )
        draft_config["mask_token_id"] = mask_token_id
        draft_config["num_depths"] = int(recipe_cfg.get("num_depths", 8))
        draft_config["down_sample_ratio"] = float(recipe_cfg.get("down_sample_ratio", 0.7))
        draft_config["down_sample_ratio_min"] = float(recipe_cfg.get("down_sample_ratio_min", 0.2))
        # P-EAGLE stacks ``num_draft_layers`` draft decoder layers (the fused
        # first layer + vanilla Llama layers). The speculators reference uses
        # 4. This overrides the draft's ``num_hidden_layers`` (which would
        # otherwise inherit the target's full depth); the EAGLE-3 TTT path
        # ignores it and always builds a single layer.
        draft_num_hidden_layers = int(recipe_cfg.get("num_draft_layers", 1))
        draft_config["num_hidden_layers"] = draft_num_hidden_layers
        if "layer_types" in draft_config:
            draft_config["layer_types"] = draft_config["layer_types"][:draft_num_hidden_layers]
        return mask_token_id

    def build_peagle_trainer(self, recipe_cfg, selected_token_ids, selected_token_mask, mask_token_id):
        """Construct the P-EAGLE trainer and record the sequence-partitioning flag.

        ``sequence_partitions`` (S) is a training-only memory knob: when > 1 the
        P-EAGLE trainer splits each sequence into S segments and runs a separate
        forward+backward per segment, accumulating gradients, so only one
        segment's activations are resident at once (P-EAGLE Algorithm 1,
        arXiv:2602.01469). It does not alter the loss or the saved checkpoint, so
        it is NOT serialized into the draft config. Default 1 == single flat
        forward. ``num_depths`` is P-EAGLE's K (number of parallel COD depths),
        default 8 to match speculators. Called only on the ``parallel_drafting``
        branch of ``setup``.
        """
        sequence_partitions = int(recipe_cfg.get("sequence_partitions", 1))
        self._peagle_partitioned = sequence_partitions > 1
        return PEagleTrainerModule(
            self.draft_model,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            num_depths=int(recipe_cfg.get("num_depths", 8)),
            mask_token_id=mask_token_id,
            down_sample_ratio=float(recipe_cfg.get("down_sample_ratio", 0.7)),
            down_sample_ratio_min=float(recipe_cfg.get("down_sample_ratio_min", 0.2)),
            sequence_partitions=sequence_partitions,
        ).to(self.device)

    def _peagle_supervision(self, batch):
        """Move a batch to device and run the live target to its draft supervision.

        Used by the partitioned step; P-EAGLE requires the live target (the
        offline cache is EAGLE-3 TTT-only).
        """
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
        )
        return target_batch.to_trainer_inputs()

    def _peagle_partitioned_step(self, batch):
        """Forward+backward one batch via P-EAGLE sequence partitioning.

        Builds the segment plan, then runs one ``DDP.forward`` per segment and
        back-propagates each here (the recipe owns ``backward()`` so DDP's
        gradient all-reduce fires). ``no_sync`` defers the all-reduce on every
        segment except the last, so there is exactly one all-reduce per
        micro-batch -- matching the single-pass path and keeping the per-rank
        collective count aligned. Gradients are divided by
        ``grad_accumulation_steps`` exactly like the single-pass backward; the
        returned (detached) metrics aggregate the whole batch for logging.
        """
        sup = self._peagle_supervision(batch)
        plan = self._module().build_peagle_plan(sup["loss_mask"])
        accum = float(self.grad_accumulation_steps)
        is_ddp = isinstance(self.trainer_module, DistributedDataParallel)
        num_units = len(plan.units)

        loss_sum = torch.zeros((), device=self.device)
        correct_sum = torch.zeros((), device=self.device)
        valid_sum = torch.zeros((), device=self.device)
        if num_units == 0:
            # Fully-unsupervised batch: still issue one synced backward so the
            # per-rank all-reduce count stays aligned (avoids a DDP hang).
            (sum(p.sum() for p in self.trainer_module.parameters()) * 0.0).backward()
            return SimpleNamespace(loss=loss_sum, accuracy=correct_sum, valid_tokens=valid_sum)

        for i in range(num_units):
            defer_sync = is_ddp and i < num_units - 1
            sync_ctx = self.trainer_module.no_sync() if defer_sync else nullcontext()
            with sync_ctx:
                seg = self.trainer_module(**sup, peagle_segment=(plan, i))
                (seg.loss / accum).backward()
            loss_sum = loss_sum + seg.loss.detach()
            valid_sum = valid_sum + seg.valid_tokens
            correct_sum = correct_sum + seg.accuracy.detach() * seg.valid_tokens
        return SimpleNamespace(loss=loss_sum, accuracy=correct_sum / valid_sum.clamp_min(1.0), valid_tokens=valid_sum)
