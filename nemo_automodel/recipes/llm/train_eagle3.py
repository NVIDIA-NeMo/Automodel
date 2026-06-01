# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""EAGLE-3 training recipe for Llama-style dense LLMs (Llama, Phi-3, Qwen3) and MoE backbones (Qwen3-MoE)."""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import (
    build_eagle3_dataloader,
    build_eagle3_token_mapping,
)
from nemo_automodel.components.datasets.llm.eagle3_cache import (
    build_cached_eagle3_dataloader,
    read_manifest,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.eagle import (
    Eagle3TrainerModule,
    HFEagle3TargetModel,
)
from nemo_automodel.components.speculative.eagle.registry import resolve_eagle3_draft_spec
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes._dist_setup import setup_distributed
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _find_latest_checkpoint,
    _is_checkpoint_model_config_compatible,
    _resolve_restore_from_to_ckpt_dir,
)

logger = logging.getLogger(__name__)


def _optim_steps_per_epoch(num_batches_per_epoch: int, grad_accumulation_steps: int) -> int:
    """Return ceil(num_batches / accum), the actual number of optimizer steps per epoch.

    Floor division silently drops the trailing partial accumulation window
    (up to ``grad_accumulation_steps - 1`` micro-batches) from the LR
    scheduler's view of training, even though the trainer now flushes those
    gradients with an explicit step. Ceil keeps the scheduler aligned with
    the actual number of ``optimizer.step()`` calls.
    """
    if num_batches_per_epoch <= 0 or grad_accumulation_steps <= 0:
        return 0
    return -(-num_batches_per_epoch // grad_accumulation_steps)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


class TrainEagle3Recipe(BaseRecipe):
    """Recipe for EAGLE-3 training on Llama-style dense LLMs (Llama, Phi-3, Qwen3) and MoE backbones (Qwen3-MoE)."""

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

        target_path = recipe_cfg.target_model_name_or_path
        target_config = AutoConfig.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        architectures = getattr(target_config, "architectures", []) or []
        # Dispatch via the eagle registry. New architectures are added by
        # appending to ``_DENSE_ARCHITECTURES`` (or registering a custom
        # ``DraftSpec``) in ``components/speculative/eagle/registry.py``;
        # no recipe change required.
        draft_spec = resolve_eagle3_draft_spec(architectures)

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        # ``cached_target_path`` (optional) selects the SpecForge OFFLINE path:
        # the target's supervision was precomputed to disk by
        # ``precompute_eagle3``, so we skip loading and running the target
        # entirely and stream the cache instead. This is disk-heavy and largely
        # superseded by the online path -- see precompute_eagle3 for the warning.
        self.cached_target_path = recipe_cfg.get("cached_target_path", None)
        self.dist_setup = None
        self.distributed_config = None
        self.device_mesh = None
        self.moe_mesh = None
        if self.cached_target_path is None:
            selected_token_ids, selected_token_mask = self._setup_online_target(recipe_cfg, target_path, target_config)
        else:
            selected_token_ids, selected_token_mask = self._setup_cached_target(recipe_cfg, target_config)

        draft_config = target_config.to_dict()
        draft_config["draft_vocab_size"] = int(selected_token_ids.numel())
        draft_config["target_hidden_size"] = target_config.hidden_size
        draft_config["architectures"] = ["LlamaEagle3DraftModel"]
        # The draft owns an independent ``lm_head`` whose vocab can differ
        # from ``embed_tokens`` (vocab shrinking, ``draft_vocab_size <
        # target_vocab_size``). The target's ``tie_word_embeddings`` flag
        # does not apply here -- the two tables have different shapes by
        # design -- and would otherwise cause the checkpoint wrappers to
        # drop ``lm_head.weight`` on save and resurrect a shape-mismatched
        # tensor on load.
        draft_config["tie_word_embeddings"] = False
        # Draft attention backend. Defaults to ``eager`` to preserve the
        # pre-FA2 numerics. Set ``recipe_args.draft_attn_implementation:
        # flash_attention_2`` in YAML to opt into FlashAttention for the
        # T x T causal block (Eagle3LlamaAttention merges FA's softmax_lse
        # with the diagonal-extension columns in log space).
        draft_config["attn_implementation"] = recipe_cfg.get("draft_attn_implementation", "eager")
        # Cast to the target's compute dtype so every linear / embedding / norm
        # in the draft matches the bf16 (cuda) or fp32 (cpu) hidden states fed
        # in from the target. Without this, ``initialize_rms_norm_module`` defaults
        # to bf16 while ``nn.Linear`` defaults to fp32, and ``model.fc`` errors
        # with ``expected mat1 and mat2 to have the same dtype``.
        # Reuse the target's concrete config class (LlamaConfig / Phi3Config / ...)
        # so architecture-specific defaults like attention_bias and head_dim
        # flow into the draft.
        draft_config_obj = type(target_config).from_dict(draft_config)
        self.draft_model = draft_spec.draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)
        # Seed draft embeddings from the target: directly from the live target,
        # or from the embeddings stored alongside the offline cache.
        embed_source = (
            self.target_wrapper.get_input_embeddings() if self.target_wrapper is not None else self._cached_embed_source
        )
        self.draft_model.copy_embeddings_from_target(embed_source)
        if recipe_cfg.get("freeze_embeddings", True):
            self.draft_model.freeze_embeddings()

        trainer_module = Eagle3TrainerModule(
            self.draft_model,
            selected_token_ids=selected_token_ids,
            selected_token_mask=selected_token_mask,
            ttt_steps=recipe_cfg.ttt_steps,
        ).to(self.device)
        if self.dist_env.world_size > 1:
            trainer_module = DistributedDataParallel(
                trainer_module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
            )
        self.trainer_module = trainer_module

        opt_cfg = self.cfg.optimizer
        self.peak_lr = float(opt_cfg.lr)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.trainer_module.parameters() if p.requires_grad],
            lr=self.peak_lr,
            betas=tuple(opt_cfg.get("betas", (0.9, 0.95))),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
        )
        self.grad_accumulation_steps = recipe_cfg.get("grad_accumulation_steps", 1)
        self.max_grad_norm = recipe_cfg.get("max_grad_norm", 1.0)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Warmup + cosine LR schedule. EAGLE-3 from-scratch training with a
        # flat LR diverges after the first epoch (loss climbs back up); a
        # cosine schedule keeps the optimizer step size small enough once
        # AdamW's second-moment estimates have settled.
        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        # Use ceil division so a trailing partial accumulation window (i.e. when
        # ``num_batches_per_epoch`` is not a multiple of ``grad_accumulation_steps``)
        # is counted as a real optimizer step. The training loop flushes that
        # leftover window at the end of each epoch, so the LR scheduler must
        # cover those steps too -- otherwise ``progress`` saturates and the
        # final epoch trains at ``min_lr_ratio`` instead of the intended decay.
        total_optim_steps = max(
            1,
            self.num_epochs * _optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps),
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_optim_steps))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(warmup_steps)
            progress = (step - warmup_steps) / max(1, total_optim_steps - warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, _lr_lambda)
        self.total_optim_steps = total_optim_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0

        self.rng = StatefulRNG(
            seed=int(recipe_cfg.get("shuffle_seed", 42)),
            ranked=self.dist_env.world_size > 1,
        )
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

    def _setup_online_target(self, recipe_cfg, target_path, target_config):
        """Live path: load the target model and build the live dataloader.

        Sets ``self.target_model`` / ``self.target_wrapper`` /
        ``self.train_dataloader`` / ``self.val_dataloader`` and returns the
        ``(selected_token_ids, selected_token_mask)`` draft-vocab mapping built
        by scanning the training data.
        """
        # Optional ``distributed:`` YAML section. Required for targets that do
        # not fit on a single GPU (e.g. Qwen3-30B-A3B MoE). ``force_hf`` is
        # opt-in; default ``False`` so HF architectures with an AutoModel custom
        # impl take the MoE-capable custom path.
        target_kwargs = dict(
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        )
        if self.cfg.get("distributed", None) is not None:
            self.dist_setup = setup_distributed(self.cfg, world_size=self.dist_env.world_size)
            self.distributed_config = self.dist_setup.strategy_config
            self.device_mesh = self.dist_setup.device_mesh
            self.moe_mesh = self.dist_setup.moe_mesh
            target_kwargs.update(
                distributed_config=self.distributed_config,
                device_mesh=self.device_mesh,
                moe_mesh=self.moe_mesh,
                moe_config=self.dist_setup.moe_config,
                activation_checkpointing=self.dist_setup.activation_checkpointing,
            )
        self.target_model = NeMoAutoModelForCausalLM.from_pretrained(target_path, **target_kwargs)
        # ``nn.Module.to`` is in-place; reassigning ``self.target_model`` would
        # re-trigger ``BaseRecipe.__setattr__`` state-tracking and raise.
        if self.dist_setup is None:
            self.target_model.to(self.device)
        self.target_wrapper = HFEagle3TargetModel(
            self.target_model,
            aux_layer_ids=recipe_cfg.get("aux_layer_ids", None),
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
            )

        special_token_ids = [
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
        ]
        return build_eagle3_token_mapping(
            self.train_dataloader,
            target_vocab_size=target_config.vocab_size,
            draft_vocab_size=recipe_cfg.get("draft_vocab_size", None),
            special_token_ids=special_token_ids,
        )

    def _setup_cached_target(self, recipe_cfg, target_config):
        """Offline path: stream a precomputed cache; no target model is loaded.

        Reads the cache manifest for the draft-vocab mapping and loads the stored
        target embeddings (the one target tensor the draft still needs). Sets
        ``self.target_model`` / ``self.target_wrapper`` to ``None`` and builds the
        cache-backed dataloader.
        """
        from nemo_automodel.components.datasets.llm.eagle3_cache import read_target_embeddings

        self.target_model = None
        self.target_wrapper = None
        manifest = read_manifest(self.cached_target_path)
        if int(manifest["target_vocab_size"]) != int(target_config.vocab_size):
            raise ValueError(
                f"EAGLE-3 cache at {self.cached_target_path} was built for target_vocab_size="
                f"{manifest['target_vocab_size']}, but the configured target has {target_config.vocab_size}. "
                "The cache does not match this target."
            )
        # The draft's ``fc`` consumes ``target_hidden_size * 3`` aux features; a
        # cache from a different-width target would otherwise crash deep inside
        # ``fc`` with a confusing shape error.
        expected_aux_dim = int(target_config.hidden_size) * 3
        if int(manifest["aux_hidden_dim"]) != expected_aux_dim:
            raise ValueError(
                f"EAGLE-3 cache at {self.cached_target_path} has aux_hidden_dim={manifest['aux_hidden_dim']}, "
                f"but the configured target needs {expected_aux_dim} (hidden_size {target_config.hidden_size} x 3 "
                "aux layers). The cache was built for a different target."
            )
        selected_token_ids = torch.tensor(manifest["selected_token_ids"], dtype=torch.long)
        selected_token_mask = torch.zeros(int(target_config.vocab_size), dtype=torch.bool)
        selected_token_mask[selected_token_ids] = True
        self._cached_embed_source = SimpleNamespace(weight=read_target_embeddings(self.cached_target_path))

        self.train_dataloader = build_cached_eagle3_dataloader(
            cache_dir=self.cached_target_path,
            batch_size=recipe_cfg.micro_batch_size,
            shuffle=True,
            num_workers=recipe_cfg.get("num_workers", 0),
            distributed=self.dist_env.world_size > 1,
        )
        self.val_dataloader = None
        if self.dist_env.is_main:
            logger.info(
                "EAGLE-3 OFFLINE cache: streaming %d precomputed samples from %s (target model not loaded).",
                len(self.train_dataloader.dataset),
                self.cached_target_path,
            )
        return selected_token_ids, selected_token_mask

    def _forward_batch(self, batch):
        """Run the trainer module for one batch, from the live target or the cache."""
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        if self.target_wrapper is None:
            # Offline cache: the supervision is already in the batch.
            return self.trainer_module(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
                aux_hidden_states=batch["aux_hidden_states"],
                target_probs=batch["target_probs"],
                position_mask=batch["position_mask"],
            )
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
        )
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            attention_mask=target_batch.attention_mask,
            loss_mask=target_batch.loss_mask,
            aux_hidden_states=target_batch.aux_hidden_states,
            target_logits=target_batch.logits,
        )

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the standard recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
        # EAGLE recipes construct the draft model directly and bypass
        # `apply_model_infrastructure`, which is where `_pre_shard_hf_state_dict_keys`
        # would normally be attached. Capture the pre-shard keys here so the
        # consolidated-safetensors path in `_maybe_build_consolidated_index`
        # has something to diff against instead of `None`.
        draft_state_dict_keys = list(self.draft_model.state_dict().keys())
        ckpt_kwargs = dict(
            enabled=True,
            checkpoint_dir=default_dir,
            model_save_format="safetensors",
            model_repo_id=str(target_path),
            model_cache_dir=hf_constants.HF_HUB_CACHE,
            save_consolidated=True,
            is_peft=False,
            model_state_dict_keys=draft_state_dict_keys,
        )
        if ckpt_cfg is not None:
            user_cfg = ckpt_cfg.to_dict() if hasattr(ckpt_cfg, "to_dict") else dict(ckpt_cfg)
            user_cfg.pop("restore_from", None)
            ckpt_kwargs.update(user_cfg)
        if ckpt_kwargs.get("model_state_dict_keys") is None:
            ckpt_kwargs["model_state_dict_keys"] = draft_state_dict_keys

        self.checkpoint_config = CheckpointingConfig(**ckpt_kwargs)
        dp_rank = dist.get_rank() if dist.is_initialized() else 0
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config,
            dp_rank=dp_rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )

    def _module(self):
        return (
            self.trainer_module.module
            if isinstance(self.trainer_module, DistributedDataParallel)
            else self.trainer_module
        )

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float | None = None,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ) -> None:
        """Persist draft model, optimizer, scheduler, RNG, and EAGLE-3 meta.

        Overrides ``BaseRecipe.save_checkpoint`` because EAGLE recipes hold multiple
        ``nn.Module`` attributes (frozen target, target wrapper, trainer module wrapping
        the draft) — only ``draft_model`` should be persisted as the main model. The
        EAGLE-3 vocab mapping tensors ride along through ``_save_extra_state``.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        self.checkpointer.async_wait()

        prev_pending = getattr(self, "_last_pending_checkpoint_dir", None)
        prev_best_pending = getattr(self, "_last_pending_best_checkpoint_info", None)

        ckpt_root = self.checkpoint_config.checkpoint_dir
        path = os.path.join(str(ckpt_root), f"epoch_{epoch}_step_{step}")
        is_dist_initialized = dist.is_initialized()
        is_rank_0 = (not is_dist_initialized) or dist.get_rank() == 0
        best_val_metric = (
            val_loss.get(next(iter(val_loss.keys())) if len(val_loss) == 1 else best_metric_key) if val_loss else None
        )

        if prev_pending is not None:
            if is_rank_0:
                self._update_latest_symlink(prev_pending)
            setattr(self, "_last_pending_checkpoint_dir", None)
            if is_dist_initialized:
                dist.barrier()

        if prev_best_pending is not None:
            if is_rank_0 and prev_best_pending.get("val") is not None:
                self._update_best_symlink(prev_best_pending["path"], float(prev_best_pending["val"]))
            setattr(self, "_last_pending_best_checkpoint_info", None)
            if is_dist_initialized:
                dist.barrier()

        if is_rank_0:
            if os.path.exists(path):
                raise FileExistsError(f"Checkpoint directory {path} already exists")
            os.makedirs(path, exist_ok=True)
            loss_dict: dict[str, float] = {}
            if train_loss is not None:
                loss_dict["train_loss"] = float(train_loss)
            if val_loss:
                for k, v in val_loss.items():
                    loss_dict[k] = float(v)
            if loss_dict:
                with open(os.path.join(path, "losses.json"), "w") as f:
                    json.dump(loss_dict, f)
        if is_dist_initialized:
            dist.barrier()

        draft_model = self._module().draft_model
        self.checkpointer.save_model(draft_model, path, tokenizer=self.tokenizer)
        self.checkpointer.save_optimizer(self.optimizer, draft_model, path, self.lr_scheduler)
        self.checkpointer.save_on_dp_ranks(self.rng, "rng", path)

        if is_rank_0:
            self._save_extra_state(path, epoch=epoch)
            try:
                save_config(self.cfg.raw_config, path)
            except (AttributeError, OSError) as e:
                logger.warning("Failed to save config snapshot: %s", e)
        if is_dist_initialized:
            dist.barrier()

        if getattr(self.checkpointer.config, "is_async", False):
            setattr(self, "_last_pending_checkpoint_dir", path)
            if best_val_metric is not None:
                setattr(self, "_last_pending_best_checkpoint_info", {"path": path, "val": float(best_val_metric)})
        else:
            if is_rank_0:
                self._update_latest_symlink(path)
                if best_val_metric is not None:
                    self._update_best_symlink(path, float(best_val_metric))
            if is_dist_initialized:
                dist.barrier()

    def _save_extra_state(self, path: str, epoch: int) -> None:
        """Persist EAGLE-3 meta: global_step, epoch, and vocab mapping tensors."""
        torch.save(
            {
                "global_step": self.runtime.global_step,
                "epoch": int(epoch),
                "selected_token_ids": self._module().selected_token_ids.cpu(),
                "selected_token_mask": self._module().selected_token_mask.cpu(),
            },
            os.path.join(path, "eagle_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Resolve and restore a checkpoint produced by ``save_checkpoint``.

        Restores the draft model, optimizer, LR scheduler, RNG, ``global_step``, and the
        EAGLE-3 vocab mapping tensors. Target model weights are NOT restored — they are
        re-loaded from the HF hub on each run because the target is frozen.
        """
        checkpointer = getattr(self, "checkpointer", None)
        if checkpointer is None or not checkpointer.config.enabled:
            return
        is_rank_0 = (not dist.is_initialized()) or dist.get_rank() == 0
        ckpt_root = self.checkpoint_config.checkpoint_dir

        if restore_from:
            ckpt_dir = _resolve_restore_from_to_ckpt_dir(ckpt_root, restore_from)
            if ckpt_dir is None:
                if is_rank_0:
                    logger.warning("restore_from='LATEST' but no checkpoint found in %s", ckpt_root)
                return
            if not os.path.isdir(ckpt_dir):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {ckpt_dir}")
        else:
            auto = _find_latest_checkpoint(ckpt_root)
            if auto is None:
                return
            ckpt_dir = str(auto)

        ok, reason = _is_checkpoint_model_config_compatible(self.cfg, ckpt_dir)
        if not ok:
            if not restore_from:
                if is_rank_0:
                    logger.warning(
                        "Auto-detected checkpoint at %s is incompatible with current model configuration: %s. "
                        "Skipping restore.",
                        ckpt_dir,
                        reason,
                    )
                return
            if is_rank_0:
                logger.warning(
                    "Checkpoint at %s may be incompatible with current model configuration: %s. "
                    "Proceeding with restore anyway.",
                    ckpt_dir,
                    reason,
                )

        if is_rank_0:
            logger.info("Resuming from checkpoint: %s", ckpt_dir)

        draft_model = self._module().draft_model
        self.checkpointer.load_model(draft_model, os.path.join(ckpt_dir, "model"))
        self.checkpointer.load_optimizer(self.optimizer, draft_model, ckpt_dir, self.lr_scheduler)
        try:
            self.checkpointer.load_on_dp_ranks(self.rng, "rng", ckpt_dir)
        except FileNotFoundError:
            logger.warning("RNG state not found in %s; continuing without restoring RNG.", ckpt_dir)

        self._load_extra_state(ckpt_dir)

    def _load_extra_state(self, ckpt_dir: str) -> None:
        """Restore EAGLE-3 meta: global_step, epoch, and vocab mapping tensors."""
        meta_path = os.path.join(ckpt_dir, "eagle_meta.pt")
        if not os.path.exists(meta_path):
            legacy = os.path.join(ckpt_dir, "eagle3_meta.pt")
            meta_path = legacy if os.path.exists(legacy) else meta_path
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))
            ids = meta.get("selected_token_ids")
            mask = meta.get("selected_token_mask")
            if ids is not None and mask is not None:
                module = self._module()
                module.selected_token_ids.copy_(ids.to(module.selected_token_ids.device))
                module.selected_token_mask.copy_(mask.to(module.selected_token_mask.device))

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_acc = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        with torch.no_grad():
            for batch in self.val_dataloader:
                metrics = self._forward_batch(batch)
                total_loss = total_loss + metrics.loss.detach()
                total_acc = total_acc + metrics.accuracy.detach()
                total_batches = total_batches + 1
        total_loss = _all_reduce_mean(total_loss)
        total_acc = _all_reduce_mean(total_acc)
        total_batches = _all_reduce_mean(total_batches)
        self.trainer_module.train()
        return (total_loss / total_batches.clamp_min(1.0), total_acc / total_batches.clamp_min(1.0))

    def run_train_validation_loop(self):
        """Run the minimal EAGLE-3 train loop."""
        self.trainer_module.train()
        try:
            batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            batches_per_epoch = None
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            return
        if self.dist_env.is_main:
            logger.info(
                "Training start: start_epoch=%s num_epochs=%s batches_per_epoch=%s grad_accum=%s log_every=%s "
                "total_optim_steps=%s warmup_steps=%s peak_lr=%.3e min_lr_ratio=%s",
                start_epoch,
                self.num_epochs,
                batches_per_epoch,
                self.grad_accumulation_steps,
                self.log_every_steps,
                self.total_optim_steps,
                self.warmup_steps,
                self.peak_lr,
                self.min_lr_ratio,
            )

        for epoch in range(start_epoch, self.num_epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            running_loss = torch.zeros((), device=self.device)
            running_acc = torch.zeros((), device=self.device)
            running_steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            batches_processed = 0
            pending_micro_batches = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                metrics = self._forward_batch(batch)
                loss = metrics.loss / float(self.grad_accumulation_steps)
                loss.backward()

                running_loss = running_loss + metrics.loss.detach()
                running_acc = running_acc + metrics.accuracy.detach()
                running_steps += 1
                batches_processed = batch_idx + 1
                pending_micro_batches += 1

                if pending_micro_batches == self.grad_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.runtime.global_step += 1
                    pending_micro_batches = 0

                    if self.runtime.global_step % self.log_every_steps == 0:
                        mean_loss = _all_reduce_mean(running_loss / max(running_steps, 1))
                        mean_acc = _all_reduce_mean(running_acc / max(running_steps, 1))
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        if self.dist_env.is_main:
                            logger.info(
                                "epoch=%s step=%s train_loss=%.6f train_acc=%.6f lr=%.3e",
                                epoch,
                                self.runtime.global_step,
                                mean_loss.item(),
                                mean_acc.item(),
                                current_lr,
                            )
                        running_loss.zero_()
                        running_acc.zero_()
                        running_steps = 0

            # Flush the trailing partial accumulation window. When
            # ``batches_per_epoch`` is not a multiple of
            # ``grad_accumulation_steps``, up to ``grad_accumulation_steps - 1``
            # micro-batches have run ``backward()`` but never reached an
            # ``optimizer.step()`` -- those gradients would otherwise be wiped
            # by the next epoch's ``zero_grad`` and the samples wasted.
            #
            # Each micro-batch divided its loss by ``grad_accumulation_steps``
            # in anticipation of a full window. With only ``pending_micro_batches``
            # contributors, the accumulated gradient magnitude is
            # ``pending_micro_batches / grad_accumulation_steps`` of a normal
            # step; rescale by the inverse so the trailing step's gradient
            # is on the same scale as every other step.
            if pending_micro_batches > 0:
                scale = float(self.grad_accumulation_steps) / float(pending_micro_batches)
                for p in self.trainer_module.parameters():
                    if p.grad is not None:
                        p.grad.mul_(scale)
                torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.runtime.global_step += 1
                pending_micro_batches = 0

                if running_steps > 0:
                    mean_loss = _all_reduce_mean(running_loss / max(running_steps, 1))
                    mean_acc = _all_reduce_mean(running_acc / max(running_steps, 1))
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    if self.dist_env.is_main:
                        logger.info(
                            "epoch=%s step=%s train_loss=%.6f train_acc=%.6f lr=%.3e (trailing flush)",
                            epoch,
                            self.runtime.global_step,
                            mean_loss.item(),
                            mean_acc.item(),
                            current_lr,
                        )
                    running_loss.zero_()
                    running_acc.zero_()
                    running_steps = 0

            if self.dist_env.is_main:
                logger.info(
                    "Epoch %s done: total_batches_seen=%s global_step=%s",
                    epoch,
                    batches_processed,
                    self.runtime.global_step,
                )

            eval_metrics = self._run_eval()
            val_loss_dict: dict[str, float] | None = None
            if eval_metrics is not None:
                val_loss_dict = {
                    "val_loss": eval_metrics[0].item(),
                    "val_accuracy": eval_metrics[1].item(),
                }
                if self.dist_env.is_main:
                    logger.info(
                        "epoch=%s val_loss=%.6f val_acc=%.6f",
                        epoch,
                        val_loss_dict["val_loss"],
                        val_loss_dict["val_accuracy"],
                    )
            self.save_checkpoint(
                epoch=epoch + 1,
                step=self.runtime.global_step,
                train_loss=None,
                val_loss=val_loss_dict,
                best_metric_key="val_loss",
            )
            ckpt_cfg = getattr(self, "checkpoint_config", None)
            if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
                logger.info(
                    "Saved checkpoint to %s/epoch_%d_step_%d",
                    ckpt_cfg.checkpoint_dir,
                    epoch + 1,
                    self.runtime.global_step,
                )

        if self.dist_env.is_main:
            logger.info("Training complete: global_step=%s", self.runtime.global_step)


def main(config_path=None):
    """Main entry point for the EAGLE-3 recipe."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle3Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
