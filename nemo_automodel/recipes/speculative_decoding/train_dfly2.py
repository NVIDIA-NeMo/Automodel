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

"""DFly2 draft-model training recipe (DeepSeek V4.1 targets).

DFly2 (arXiv:2611.34776) is a block-parallel drafter for sparse-attention targets. Each
draft layer is paired with one target sparse-attention layer and reads that layer's own
keys at the positions it attends from the row before the anchor, so the served drafter
shares the target's KV cache and indexer instead of keeping its own. See
``nemo_automodel.components.speculative.dfly2``.

This recipe reuses every piece of the DSpark recipe -- the frozen EP/FSDP target, anchor
sampling, gradient accumulation with a trailing-window flush, metrics, and checkpointing
-- and swaps in the model-owned DFly2 draft, the target capture of the paired layers'
keys and key lists, and the D-PACE-weighted LK objective. ``target_layer_ids`` names the
paired target layers, one per draft layer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.speculative.dfly2 import DFly2TargetModel, DFly2TrainerModule, ModelOwnedDFly2Provider
from nemo_automodel.components.speculative.dspark.core import DSparkStepMetrics
from nemo_automodel.components.speculative.dspark.registry import ModelOwnedDSparkProvider
from nemo_automodel.components.speculative.dspark.target import _DSparkPackedForward
from nemo_automodel.recipes.speculative_decoding.train_dspark import TrainDSparkRecipe, _DraftArgs

logger = logging.getLogger(__name__)


def check_dfly2_checkpoint_settings(
    path: str, *, target_layer_ids: list[int], block_size: int, mask_token_id: int
) -> None:
    """Refuse warm-start weights whose recorded DFly2 settings differ from this run's.

    The paired target layers, the block size and the mask token own no tensors, so a draft
    trained with others loads cleanly and then reads the wrong keys. A released draft and an
    export of this recipe record them in ``config.json`` under ``dfly_config``.

    Args:
        path: The configured ``draft_weights_path``, a directory or a ``.safetensors`` file.
        target_layer_ids: Paired target layers of this run, in draft-layer order.
        block_size: Draft slots per block of this run.
        mask_token_id: Mask token of this run.

    Raises:
        ValueError: A recorded setting differs from this run's.
    """
    root = Path(path)
    directory = root if root.is_dir() else root.parent
    # A consolidated export holds config.json; a step checkpoint's model/ keeps it in .hf_metadata/.
    candidates = (directory / "config.json", directory / ".hf_metadata" / "config.json")
    config_path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if config_path is None:
        return
    recorded = json.loads(config_path.read_text()).get("dfly_config")
    if not recorded:
        return
    expected = {
        "target_kv_layers": [int(layer_id) for layer_id in target_layer_ids],
        "block_size": int(block_size),
        "mask_token_id": int(mask_token_id),
    }
    mismatched = {
        name: {"checkpoint": recorded[name], "this run": value}
        for name, value in expected.items()
        if name in recorded and recorded[name] != value
    }
    if mismatched:
        raise ValueError(f"draft_weights_path={path!r} was trained with other DFly2 settings: {mismatched}.")


def check_fsdp_expert_split(num_experts: int, world_size: int) -> None:
    """Refuse an FSDP2 world size under which the draft's routed experts cannot be checkpointed.

    FSDP2 shards the grouped expert tensors along the expert dimension like ``torch.chunk``
    (``ceil(experts / ranks)`` per rank, the last ranks short), while the checkpoint export
    expects the balanced split (the remainder spread over the first ranks). Where the two
    differ, the first save raises on some ranks and the others wait in a collective.

    Args:
        num_experts: Routed experts per draft layer.
        world_size: Ranks the draft is sharded over.

    Raises:
        ValueError: The two splits differ for ``world_size``.
    """

    def splits_agree(ranks: int) -> bool:
        chunked = [len(part) for part in torch.arange(num_experts).chunk(ranks)]
        chunked += [0] * (ranks - len(chunked))
        balanced = [num_experts // ranks + (rank < num_experts % ranks) for rank in range(ranks)]
        return chunked == balanced

    if world_size > 1 and not splits_agree(world_size):
        usable = [ranks for ranks in range(2, 2 * num_experts + 1) if splits_agree(ranks)]
        raise ValueError(
            f"An FSDP2 world size of {world_size} cannot checkpoint the draft's {num_experts} routed experts per "
            f"layer: FSDP2 splits them like torch.chunk and the checkpoint export expects an even split. "
            f"Use one of {usable}."
        )


class TrainDFly2Recipe(TrainDSparkRecipe):
    """Recipe for DFly2 draft-model training on a colocated sparse-attention target."""

    # The released draft has no confidence head, so a run that does not ask for one can warm-start from it.
    default_confidence_head_alpha = 0.0

    def setup(self):
        """Reject the DSpark options DFly2 does not support, then build everything via the DSpark recipe."""
        recipe_cfg = self.cfg.recipe_args
        unsupported = {
            "cached_target_path": recipe_cfg.get("cached_target_path", None) is not None,
            "target_model_backend=remote": str(recipe_cfg.get("target_model_backend", "colocated")) != "colocated",
            "packed_sequence_size": int(recipe_cfg.get("packed_sequence_size", 0) or 0) > 0,
            "multimodal": bool(recipe_cfg.get("multimodal", False)),
        }
        rejected = [name for name, requested in unsupported.items() if requested]
        if rejected:
            # The draft reads the target's attention state, which only a colocated
            # text-only forward on unpacked rows produces here.
            raise NotImplementedError(f"DFly2 trains against a colocated, unpacked text target; got {rejected}.")
        super().setup()

    def _build_target_wrapper(
        self, target_layer_ids: list[int], target_packed_forward: _DSparkPackedForward | None
    ) -> DFly2TargetModel:
        """Capture the paired layers' keys and key lists instead of hidden states."""
        del target_packed_forward  # packing is rejected in setup
        return DFly2TargetModel(self.target_model, target_layer_ids)

    def _build_model_owned_draft(self, target_config: ModelOwnedDSparkProvider, margs: _DraftArgs) -> torch.nn.Module:
        """Build the target package's DFly2 draft and start its layers from their paired target layers."""
        if not isinstance(target_config, ModelOwnedDFly2Provider):
            raise ValueError(f"{type(target_config).__name__} does not provide a DFly2 draft")
        recipe_cfg = self.cfg.recipe_args
        options = _DraftArgs(
            target_layer_ids=margs.target_layer_ids,
            block_size=margs.block_size,
            num_anchors=margs.num_anchors,
            mask_token_id=margs.mask_token_id,
            enable_confidence_head=float(margs.confidence_head_alpha) > 0.0,
            confidence_head_stop_gradient=bool(recipe_cfg.get("confidence_head_stop_gradient", True)),
            # Long rows need the sparse kernel ("tilelang" or "cudnn"); "sdpa" builds a dense mask.
            attn_backend=str(recipe_cfg.get("draft_attn_backend", "sdpa")),
        )
        draft_weights_path = recipe_cfg.get("draft_weights_path", None)
        if draft_weights_path:
            check_dfly2_checkpoint_settings(
                str(draft_weights_path),
                target_layer_ids=list(margs.target_layer_ids),
                block_size=int(margs.block_size),
                mask_token_id=int(margs.mask_token_id),
            )
        draft_model = target_config.build_dfly2_draft(options)
        dist_cfg = self.cfg.get("distributed", None)
        strategy = str(dist_cfg.get("strategy", "fsdp2")).lower() if dist_cfg is not None else "fsdp2"
        if strategy == "fsdp2":
            # Fail here rather than hang at the first checkpoint.
            check_fsdp_expert_split(draft_model.moe_config.n_routed_experts, self.dist_env.world_size)
        # Collective over the sharded target: every rank builds the draft and gathers the same tensors.
        draft_model.initialize_from_target(self.target_model)
        return draft_model

    def _build_trainer_module(self, recipe_cfg, confidence_head_alpha: float) -> torch.nn.Module:
        """Build the DFly2 trainer wrapper (D-PACE-weighted LK plus confidence)."""
        return DFly2TrainerModule(
            self.draft_model,
            dpace_alpha=float(recipe_cfg.get("dpace_alpha", 0.5)),
            lk_eta=float(recipe_cfg.get("lk_eta", 3.0)),
            confidence_head_alpha=confidence_head_alpha,
        )

    @staticmethod
    def _draft_fsdp_units(draft_model: torch.nn.Module) -> list[torch.nn.Module]:
        """Shard the heads by dtype too: the corrections hold FP32 sinks and biases, the confidence head is FP32."""
        heads = [draft_model.mtp_fusion, draft_model.hidden_correction]
        if draft_model.confidence_head is not None:
            heads.append(draft_model.confidence_head)
        return [*draft_model.layers, *heads]

    def _forward_batch(self, batch) -> DSparkStepMetrics:
        """Run the frozen target once and train the draft on its paired layers' keys."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], loss_mask=batch["loss_mask"]
        )
        features = target_batch.features
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            loss_mask=target_batch.loss_mask,
            context_keys=features.context_keys,
            context_indices=features.context_indices,
            target_last_hidden_states=features.last_hidden_states,
        )


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDFly2Recipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDFly2Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
