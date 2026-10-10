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

"""EAGLE-1 / EAGLE-2 training recipe for Llama-style dense LLMs (Llama, Phi-3, Qwen3) and MoE backbones (Qwen3-MoE)."""

from __future__ import annotations

import logging
from contextlib import nullcontext

import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.tp_replicas import broadcast_tp_replicas, synchronize_tp_replica_gradients
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle.core_v12 import EagleTrainerModule, FeatureNoiseConfig
from nemo_automodel.components.speculative.eagle.registry import resolve_eagle1_draft_spec
from nemo_automodel.components.speculative.eagle.target_v12 import HFEagleTargetModel
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.utils.model_utils import print_trainable_parameters
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes.llm._spec_recipe_base import SpecDecodeRecipeBase
from nemo_automodel.recipes.llm._spec_train_utils import (
    all_reduce_mean,
    apply_draft_compile,
    apply_draft_fp8,
    packing_kwargs,
    raise_if_peft_configured,
    should_sync_grads,
    submesh_or_none,
    validate_packing_gates,
)

logger = logging.getLogger(__name__)


def _build_feature_noise(half_width: float) -> FeatureNoiseConfig | None:
    """Build the EAGLE-1/2 augmentation from the recipe's fixed-width knob."""
    return FeatureNoiseConfig.fixed(half_width) if half_width > 0 else None


class TrainEagle1Recipe(SpecDecodeRecipeBase):
    """Recipe for EAGLE-1 training on Llama-style dense LLMs (Llama, Phi-3, Qwen3) and MoE backbones (Qwen3-MoE)."""

    # Existing EAGLE checkpoints key the RNG file on the global rank.
    rng_checkpoint_on_dp_ranks = False

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        """Build target model, draft model, data, optimizer, and trainer module."""
        self.dist_env = initialize_distributed(
            backend=self.cfg.get("dist_env", {}).get("backend", "nccl"),
            timeout_minutes=self.cfg.get("dist_env", {}).get("timeout_minutes", 30),
        )
        setup_logging()

        recipe_cfg = self.cfg.recipe_args
        self.device = self.dist_env.device or torch.device("cpu")
        raise_if_peft_configured(self.cfg, type(self).__name__)

        target_path = recipe_cfg.target_model_name_or_path
        target_config = AutoConfig.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        architectures = getattr(target_config, "architectures", []) or []
        # Dispatch via the eagle registry. New architectures are added by
        # appending to ``_DENSE_ARCHITECTURES`` (or registering a custom
        # ``DraftSpec``) in ``components/speculative/eagle/registry.py``;
        # no recipe change required.
        draft_spec = resolve_eagle1_draft_spec(architectures)

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # Optional ``distributed:`` YAML section. Required for targets that
        # do not fit on a single GPU (e.g. Qwen3-30B-A3B MoE). Absent =>
        # original single-GPU-per-rank behavior, preserved for 8B-class dense.
        # ``force_hf`` opt-in; see the train_eagle3 recipe for rationale.
        target_kwargs = dict(
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        )
        self.dist_setup = None
        self.distributed_config = None
        self.device_mesh = None
        self.moe_mesh = None
        self.dp_mesh = None
        if self.cfg.get("distributed", None) is not None:
            self.dist_setup = create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
            self.distributed_config = self.dist_setup.strategy_config
            self.device_mesh = self.dist_setup.mesh_context.device_mesh
            self.moe_mesh = self.dist_setup.mesh_context.moe_mesh
            # Tensor parallelism (distributed.tp_size>1) shards the target's
            # linears in place via ``from_pretrained`` below; the draft is small
            # and stays replicated. The flattened "dp" axis excludes "tp", so the
            # draft DDP group and the dataloader sampler key on it to replicate
            # across TP ranks (the target wrapper gathers the vocab-sharded
            # logits). EAGLE-1/2 has no context parallelism, so only "dp" matters.
            self.dp_mesh = submesh_or_none(self.device_mesh, "dp")
            target_kwargs.update(
                distributed_setup=self.dist_setup,
            )
        self.target_model = NeMoAutoModelForCausalLM.from_pretrained(target_path, **target_kwargs)
        if self.dist_setup is None:
            # ``nn.Module.to`` is in-place; reassigning ``self.target_model``
            # would re-trigger ``BaseRecipe.__setattr__`` state-tracking and
            # raise ``RuntimeError: State key 'target_model' is already tracked``.
            self.target_model.to(self.device)
        self.target_model.requires_grad_(False)
        self.target_wrapper = HFEagleTargetModel(self.target_model)

        # ``packed_sequence_size > 0`` enables sequence packing (greedy first-fit
        # of documents into fixed-width rows), removing the padding waste of the
        # default ``padding="max_length"`` path. The colocated HF target and the
        # single-step draft both consume the block-causal packing metadata
        # (position_ids / seq_lens / doc_remaining) the packed loader emits.
        packed_sequence_size = int(recipe_cfg.get("packed_sequence_size", 0) or 0)
        if packed_sequence_size > 0:
            # Fail fast at setup (before the multi-GPU load + dataloader build) on
            # packing configs EAGLE-1/2 cannot honor, rather than mid-run.
            validate_packing_gates(
                method="EAGLE-1/2",
                cp_size=int(self.cfg.get("distributed.cp_size", 1) or 1),
                target_attn_impl=getattr(self.target_model.config, "_attn_implementation", None) or "",
                micro_batch_size=int(recipe_cfg.micro_batch_size),
            )
        self.train_dataloader = build_eagle3_dataloader(
            data_path=recipe_cfg.train_data_path,
            tokenizer=self.tokenizer,
            seq_length=recipe_cfg.seq_length,
            batch_size=recipe_cfg.micro_batch_size,
            shuffle=True,
            num_workers=recipe_cfg.get("num_workers", 0),
            split=recipe_cfg.get("train_split", None),
            distributed=self.dist_env.world_size > 1,
            shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
            mask_reasoning_content=recipe_cfg.get("mask_reasoning_content", False),
            mask_generation_prompt=recipe_cfg.get("mask_generation_prompt", False),
            packed_sequence_size=packed_sequence_size,
            dp_mesh=self.dp_mesh,
        )
        self.val_dataloader = None
        if recipe_cfg.get("val_data_path", None):
            self.val_dataloader = build_eagle3_dataloader(
                data_path=recipe_cfg.val_data_path,
                tokenizer=self.tokenizer,
                seq_length=recipe_cfg.seq_length,
                batch_size=recipe_cfg.micro_batch_size,
                shuffle=False,
                num_workers=recipe_cfg.get("num_workers", 0),
                split=recipe_cfg.get("val_split", None),
                distributed=self.dist_env.world_size > 1,
                shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
                mask_reasoning_content=recipe_cfg.get("mask_reasoning_content", False),
                mask_generation_prompt=recipe_cfg.get("mask_generation_prompt", False),
                packed_sequence_size=packed_sequence_size,
                dp_mesh=self.dp_mesh,
            )

        draft_config = target_config.to_dict()
        draft_config["architectures"] = ["LlamaEagleDraftModel"]
        draft_config["draft_num_hidden_layers"] = int(recipe_cfg.get("draft_num_hidden_layers", 1))
        # Reuse the target's concrete config class (LlamaConfig / Phi3Config / ...)
        # so architecture-specific defaults like attention_bias and head_dim
        # flow into the draft.
        draft_config_obj = type(target_config).from_dict(draft_config)
        self.draft_model = draft_spec.draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)
        self.draft_model.copy_embeddings_from_target(self.target_wrapper.get_input_embeddings())
        if recipe_cfg.get("freeze_embeddings", True):
            self.draft_model.freeze_embeddings()
        # Optional FP8 draft compute, in place (see apply_draft_fp8); must precede the DDP wrap.
        apply_draft_fp8(self.draft_model, self.cfg.get("fp8", None))
        # Optional torch.compile of the draft, in place; after the fp8 swap.
        apply_draft_compile(self.draft_model, self.cfg.get("compile", None))
        # The target's "Model summary" is logged by apply_model_infrastructure when it
        # loads; the draft is built directly, so log its (trainable) summary here too.
        print_trainable_parameters(self.draft_model, name="Draft")

        trainer_module = EagleTrainerModule(
            self.draft_model,
            target_lm_head=self.target_wrapper.get_lm_head(),
            hidden_loss_weight=float(recipe_cfg.get("hidden_loss_weight", 1.0)),
            token_loss_weight=float(recipe_cfg.get("token_loss_weight", 0.1)),
            # EAGLE feature-noise augmentation U(-0.1, 0.1); paper default, set 0 to disable.
            # Unscaled: the paper states one width, unlike ViSpec's seq-len-scaled draw.
            feature_noise_config=_build_feature_noise(float(recipe_cfg.get("feature_noise", 0.1))),
        ).to(self.device)
        if self.dist_env.world_size > 1:
            # Restrict the draft's gradient all-reduce to the "dp" sub-axis. With
            # tensor parallelism the draft is replicated across tp ranks, so a
            # full-world all-reduce would average duplicate gradients; the dp
            # group (which excludes tp) reduces only across real data replicas.
            # Without a mesh (tp_size=1) dp_mesh is None -> full-world DDP,
            # unchanged.
            dp_process_group = (
                self.dp_mesh.get_group()
                if self.dp_mesh is not None and self.dp_mesh.size() < self.dist_env.world_size
                else None
            )
            trainer_module = DistributedDataParallel(
                trainer_module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
                process_group=dp_process_group,
            )
        self.trainer_module = trainer_module
        # DDP broadcasts only inside the DP subgroup. The draft is replicated
        # across TP, so align those independently initialized copies before the
        # optimizer captures them and TP replica gradients are averaged.
        broadcast_tp_replicas([self.trainer_module], self.device_mesh)

        self._finalize_setup(recipe_cfg=recipe_cfg, target_path=target_path, wandb_name_prefix="eagle1_")

    def _finalize_setup(self, *, recipe_cfg, target_path: str, wandb_name_prefix: str) -> None:
        """Build the optimizer, schedule, checkpointer, and logging around a ready trainer module.

        Runs once ``self.trainer_module`` / ``self.draft_model`` / the dataloaders
        exist, so recipes that assemble those differently (e.g. ViSpec's VLM
        target) share the rest of ``setup()`` instead of copying it.

        Args:
            recipe_cfg: The ``recipe_args`` config node.
            target_path: Target model id or path, used for checkpoint metadata
                and the default W&B run name.
            wandb_name_prefix: Prefix for the default W&B run name.
        """
        self._setup_training_state(recipe_cfg)

        self.rng = StatefulRNG(
            seed=int(recipe_cfg.get("shuffle_seed", 42)),
            ranked=self.dist_env.world_size > 1,
        )
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

        # Optional Weights & Biases logging (rank 0 only).
        self._init_wandb_run(wandb_name_prefix + str(target_path).rstrip("/").split("/")[-1])

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist EAGLE-recipe-specific scalars. Subclasses extend this."""
        self._save_meta(path, "eagle_meta.pt", epoch)

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore EAGLE-recipe-specific scalars. Subclasses extend this."""
        self._load_meta(ckpt_dir, "eagle_meta.pt", legacy_filename="eagle1_meta.pt")

    def _compute_metrics(self, batch: dict[str, torch.Tensor]):
        """Run the frozen target and the draft over one micro-batch.

        Args:
            batch: Dataloader batch whose tensors are already on ``self.device``.
                Carries ``input_ids`` / ``attention_mask`` / ``loss_mask``, each
                of shape [batch, sequence], plus any packing metadata.

        Returns:
            EagleStepMetrics for this micro-batch. Subclasses that feed the draft
            different supervision (e.g. ViSpec) override this hook.
        """
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            **packing_kwargs(batch),
        )
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            attention_mask=target_batch.attention_mask,
            loss_mask=target_batch.loss_mask,
            input_hidden_states=target_batch.input_hidden_states,
            target_hidden_states=target_batch.target_hidden_states,
            target_logits=target_batch.target_logits,
            position_ids=target_batch.position_ids,
            seq_lens=target_batch.seq_lens,
            doc_remaining=target_batch.doc_remaining,
        )

    def _loss_components(self, metrics) -> dict[str, float]:
        """Return the per-term losses this recipe logs alongside the total.

        Args:
            metrics: The step metrics returned by :meth:`_compute_metrics`.

        Returns:
            Mapping of log-suffix to scalar value, logged as ``train/<key>``.
        """
        components = {
            "hidden_loss": metrics.hidden_loss.detach().item(),
            "token_loss": metrics.token_loss.detach().item(),
        }
        # Only present when the optional ranking term is enabled, so EAGLE-1/2
        # runs do not gain an always-zero series.
        if getattr(metrics, "rank_loss", None) is not None:
            components["rank_loss"] = metrics.rank_loss.detach().item()
        return components

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_acc = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                metrics = self._compute_metrics(batch)
                total_loss += metrics.loss.detach()
                total_acc += metrics.accuracy.detach()
                total_batches += 1

        total_loss = all_reduce_mean(total_loss)
        total_acc = all_reduce_mean(total_acc)
        total_batches = all_reduce_mean(total_batches)
        self.trainer_module.train()
        return {
            "val_loss": (total_loss / total_batches.clamp_min(1)).item(),
            "val_accuracy": (total_acc / total_batches.clamp_min(1)).item(),
        }

    def run_train_validation_loop(self):
        """Run the training loop."""
        self.trainer_module.train()
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            return
        try:
            batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            batches_per_epoch = None
        is_ddp = isinstance(self.trainer_module, DistributedDataParallel)
        pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch_idx)

                running_loss = 0.0
                running_acc = 0.0
                running_components: dict[str, float] = {}
                running_micro_batches = 0
                epoch_loss = 0.0
                micro_step = 0
                pending_micro_batches = 0
                completed_steps = 0
                last_batch_idx = -1
                for batch_idx, batch in enumerate(self.train_dataloader):
                    last_batch_idx = batch_idx
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    # Skip DDP's per-micro-batch all-reduce on every micro-batch
                    # except the one an optimizer step immediately follows; that
                    # step's all-reduce covers the whole locally-accumulated window.
                    sync_grads = should_sync_grads(
                        pending_micro_batches=pending_micro_batches,
                        grad_accumulation_steps=self.grad_accumulation_steps,
                        batch_idx=batch_idx,
                        batches_per_epoch=batches_per_epoch,
                        is_ddp=is_ddp,
                    )
                    sync_ctx = nullcontext() if sync_grads else self.trainer_module.no_sync()
                    with sync_ctx:
                        metrics = self._compute_metrics(batch)
                        loss = metrics.loss / self.grad_accumulation_steps
                        loss.backward()

                    running_loss += metrics.loss.detach().item()
                    running_acc += metrics.accuracy.detach().item()
                    running_micro_batches += 1
                    for component_name, component_value in self._loss_components(metrics).items():
                        running_components[component_name] = (
                            running_components.get(component_name, 0.0) + component_value
                        )
                    epoch_loss += metrics.loss.detach().item()
                    micro_step += 1
                    pending_micro_batches += 1

                    if pending_micro_batches == self.grad_accumulation_steps:
                        synchronize_tp_replica_gradients(
                            [self.trainer_module],
                            getattr(self, "device_mesh", None),
                        )
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.lr_scheduler.step()
                        self.runtime.global_step += 1
                        if pbar is not None:
                            pbar.update(1)
                        completed_steps += 1
                        pending_micro_batches = 0
                        self._maybe_save_step_checkpoint(epoch_idx)

                        if self.dist_env.is_main and self.runtime.global_step % self.log_every_steps == 0:
                            n = max(1, running_micro_batches)
                            avg_loss = running_loss / n
                            avg_acc = running_acc / n
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                            if pbar is not None:
                                pbar.set_postfix(
                                    loss=f"{avg_loss:.4f}",
                                    acc=f"{avg_acc:.4f}",
                                    lr=f"{current_lr:.2e}",
                                )
                            # grad_norm is the pre-clip norm, so a value far above
                            # ``max_grad_norm`` means every step is being rescaled
                            # and the effective LR is ``lr / grad_norm``; the
                            # per-recipe components carry the loss breakdown that
                            # otherwise only reaches W&B.
                            component_text = " ".join(
                                f"{name}={value / n:.4g}" for name, value in sorted(running_components.items())
                            )
                            logger.info(
                                "epoch=%d step=%d loss=%.4f acc=%.4f lr=%.6g grad_norm=%.4g%s",
                                epoch_idx,
                                self.runtime.global_step,
                                avg_loss,
                                avg_acc,
                                current_lr,
                                float(grad_norm),
                                f" {component_text}" if component_text else "",
                            )
                            self._wandb_log(
                                {
                                    "train/loss": avg_loss,
                                    "train/accuracy": avg_acc,
                                    **{f"train/{name}": value / n for name, value in running_components.items()},
                                    "train/lr": current_lr,
                                    "train/grad_norm": float(grad_norm),
                                    "train/epoch": epoch_idx,
                                },
                                step=self.runtime.global_step,
                            )
                            running_loss = 0.0
                            running_acc = 0.0
                            running_components = {}
                            running_micro_batches = 0

                # Flush the trailing partial accumulation window. When
                # ``batches_per_epoch`` is not a multiple of ``grad_accumulation_steps``,
                # up to ``grad_accumulation_steps - 1`` micro-batches have run
                # ``backward()`` but never reached an ``optimizer.step()`` -- those
                # gradients would otherwise be wiped by the next epoch's
                # ``zero_grad`` and the samples wasted.
                #
                # Each micro-batch divided its loss by ``grad_accumulation_steps``
                # in anticipation of a full window. With only ``pending_micro_batches``
                # contributors, the accumulated gradient magnitude is
                # ``pending_micro_batches / grad_accumulation_steps`` of a normal
                # step; rescale by the inverse so the trailing step's gradient is on
                # the same scale as every other step.
                #
                # Assumption: every data-parallel rank reaches this flush with the
                # same ``pending_micro_batches``. That holds here because the loader
                # uses ``DistributedSampler`` with ``drop_last=False`` (and the
                # DataLoader likewise), which pads every rank to an equal sample
                # count -> equal batches per epoch -> equal trailing windows. If a
                # non-padding / variable-length sampler is ever introduced, revisit
                # this: a divergent per-rank ``scale`` would desync parameters, and
                # a rank that lands on ``pending_micro_batches == 0`` would skip the
                # flush (and the TP gradient synchronization inside it) while its
                # peers step, hanging on the mismatched collective.
                if pending_micro_batches > 0:
                    synchronize_tp_replica_gradients(
                        [self.trainer_module],
                        getattr(self, "device_mesh", None),
                    )
                    scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
                    for p in self.trainer_module.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.lr_scheduler.step()
                    self.runtime.global_step += 1
                    if pbar is not None:
                        pbar.update(1)
                    completed_steps += 1
                    pending_micro_batches = 0
                    self._maybe_save_step_checkpoint(epoch_idx)

                eval_metrics = self._run_eval()
                if self.dist_env.is_main:
                    msg = f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps}"
                    if eval_metrics is not None:
                        msg += (
                            f" val_loss={eval_metrics['val_loss']:.4f} val_accuracy={eval_metrics['val_accuracy']:.4f}"
                        )
                    logger.info(msg)
                if eval_metrics is not None:
                    self._wandb_log(
                        {"val/loss": eval_metrics["val_loss"], "val/accuracy": eval_metrics["val_accuracy"]},
                        step=self.runtime.global_step,
                    )

                if getattr(self, "save_checkpoint_every_epoch", False) and last_batch_idx >= 0:
                    avg_loss = epoch_loss / max(1, micro_step) if micro_step else None
                    self.save_checkpoint(
                        epoch=epoch_idx + 1,
                        step=self.runtime.global_step,
                        train_loss=avg_loss,
                        val_loss=eval_metrics,
                        best_metric_key="val_loss",
                        is_final_checkpoint=epoch_idx + 1 >= self.num_epochs,
                    )

            self._maybe_save_final_checkpoint(self.num_epochs)
            self._finalize_and_close_checkpointer()
        finally:
            if pbar is not None:
                pbar.close()

        if getattr(self, "wandb_run", None) is not None:
            self.wandb_run.finish()


def main(config_path: str | None = None):
    """Entrypoint for ``TrainEagle1Recipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle1Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
