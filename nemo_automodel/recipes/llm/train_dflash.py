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

"""DFlash draft-model training recipe (Qwen3-style targets).

DFlash drafts a whole block of tokens in parallel via MASK-token denoising
conditioned on the frozen target's hidden states (see
``nemo_automodel.components.speculative.dflash``). This recipe mirrors the EAGLE
recipes' scaffolding -- online target hidden-state capture, gradient
accumulation with a trailing-window flush, and the same checkpointer plumbing --
but trains the DFlash draft with its block-wise cross-entropy objective.
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    Checkpointer,
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import build_eagle3_dataloader
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.speculative.dflash.core import DFlashTrainerModule, NoValidAnchorsError
from nemo_automodel.components.speculative.dflash.draft_qwen3 import build_target_layer_ids
from nemo_automodel.components.speculative.dflash.registry import resolve_dflash_draft_spec
from nemo_automodel.components.speculative.dflash.target import HFDFlashTargetModel
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config
from nemo_automodel.recipes.base_recipe import (
    BaseRecipe,
    _find_latest_checkpoint,
    _is_checkpoint_model_config_compatible,
    _resolve_restore_from_to_ckpt_dir,
)
from nemo_automodel.recipes.llm._spec_train_utils import (
    make_warmup_cosine_schedule,
    optim_steps_per_epoch,
    should_sync_grads,
)

logger = logging.getLogger(__name__)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def _all_ranks_have_valid(local_has_valid: int, is_ddp: bool, device) -> bool:
    """Min-reduce a per-rank "this micro-batch has valid anchors" flag.

    Under DDP a data-dependent ``NoValidAnchorsError`` skip is per-rank: if one
    rank skips its backward (and its gradient all-reduce) while another runs its,
    the collective mismatches (hang) and the accumulation windows desync. Taking
    the MIN across ranks makes the skip decision unanimous -- every rank skips the
    micro-batch unless all of them have something to learn from it. The reduce is
    a tiny independent collective, safe inside ``no_sync`` (which only gates the
    DDP backward all-reduce). Single-process runs return the local flag unchanged.
    """
    if not is_ddp or not (dist.is_available() and dist.is_initialized()):
        return bool(local_has_valid)
    flag = torch.tensor([local_has_valid], device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _submesh_or_none(device_mesh, name: str):
    """Return the named (flattened) submesh, or None if absent / no mesh.

    Uses ``get_flat_mesh`` so ``_flatten()``-created axes ("dp") resolve across
    torch versions. The "dp" axis excludes "tp", so keying the draft DDP group,
    the dataloader sampler, and the checkpointer dp_rank on it replicates the
    draft across tensor-parallel ranks (every TP rank in a draft replica sees the
    same batch).
    """
    if device_mesh is None:
        return None
    try:
        return get_flat_mesh(device_mesh, name)
    except KeyError:
        return None


class TrainDFlashRecipe(BaseRecipe):
    """Recipe for DFlash draft-model training on Qwen3-style dense / MoE targets."""

    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        """Build the target model, DFlash draft, data, optimizer, and trainer module."""
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
        draft_spec = resolve_dflash_draft_spec(architectures)

        self.tokenizer = NeMoAutoTokenizer.from_pretrained(
            target_path, trust_remote_code=recipe_cfg.get("trust_remote_code", False)
        )
        self.compute_dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.target_model = self._build_target_model(recipe_cfg, target_path)

        # Resolve the captured target layers once and share them between the
        # target wrapper (what to capture) and the draft config (the ``fc`` input
        # width) so the two never disagree.
        num_target_layers = int(target_config.num_hidden_layers)
        draft_num_hidden_layers = int(recipe_cfg.get("draft_num_hidden_layers", 5))
        target_layer_ids = list(
            recipe_cfg.get("target_layer_ids", None)
            or build_target_layer_ids(num_target_layers, draft_num_hidden_layers)
        )
        self.target_wrapper = self._build_target_wrapper(target_layer_ids)

        self.block_size = int(recipe_cfg.get("block_size", 16))
        self.mask_token_id = self._resolve_mask_token_id(recipe_cfg, target_config.vocab_size)

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
                dp_mesh=self.dp_mesh,
            )

        # DFlash draft config: a small non-causal Qwen3 stack that reuses the
        # target's architecture defaults (head_dim, rope_theta, rms_norm_eps, ...).
        draft_config = target_config.to_dict()
        draft_config["architectures"] = ["Qwen3DFlashDraftModel"]
        draft_config["num_hidden_layers"] = draft_num_hidden_layers
        # ``layer_types``/``max_window_layers`` are sized to the target's depth;
        # rebuild them for the (shallower) draft. The DFlash attention never uses
        # sliding windows, so every draft layer is full attention.
        draft_config["layer_types"] = ["full_attention"] * draft_num_hidden_layers
        draft_config["max_window_layers"] = draft_num_hidden_layers
        draft_config["num_target_layers"] = num_target_layers
        draft_config["block_size"] = self.block_size
        draft_config["dflash_config"] = self._build_dflash_config(recipe_cfg, target_layer_ids)
        # A single knob drives both the trainer's mask format and the draft's
        # attention function -- they must agree (a flex BlockMask only works with
        # the flex attention fn, a dense bool mask only with sdpa/eager).
        attention_backend = recipe_cfg.get("attention_backend", "flex_attention")
        draft_config_obj = Qwen3Config.from_dict(draft_config)
        draft_config_obj._attn_implementation = attention_backend
        self.draft_model = draft_spec.draft_cls(draft_config_obj).to(device=self.device, dtype=self.compute_dtype)

        trainer_module = self._build_trainer_module(attention_backend, recipe_cfg).to(self.device)
        if self.dist_env.world_size > 1:
            # The frozen target lm_head / embed_tokens are held as non-registered
            # references on the trainer, so DDP only sees the plain draft params
            # (no sharded DTensor params to broadcast). The draft's gradient
            # all-reduce is restricted to the "dp" sub-axis (see
            # ``_draft_ddp_process_group``).
            trainer_module = DistributedDataParallel(
                trainer_module,
                device_ids=[self.device.index] if self.device.type == "cuda" else None,
                output_device=self.device.index if self.device.type == "cuda" else None,
                broadcast_buffers=False,
                find_unused_parameters=False,
                process_group=self._draft_ddp_process_group(),
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
        # Checkpoint cadence (independent knobs; the fully-trained model is always
        # saved at the end). ``ckpt_every_steps`` saves every N optimizer steps;
        # ``save_checkpoint_every_epoch`` saves at each epoch boundary (off by default).
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
        self.output_dir = pathlib.Path(recipe_cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        try:
            num_batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            num_batches_per_epoch = 0
        total_optim_steps = max(
            1, self.num_epochs * optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps)
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_optim_steps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_warmup_cosine_schedule(warmup_steps, total_optim_steps, min_lr_ratio)
        )
        self.total_optim_steps = total_optim_steps
        self.runtime = SimpleNamespace(global_step=0)
        self._resume_epoch = 0
        self._skipped_micro_batches = 0

        # Seed by the dp coordinate, not the global rank: the draft is replicated
        # across cp (and tp) ranks and must sample the SAME anchor positions each
        # step, else the replicas diverge. _get_dp_rank() returns the global rank
        # when there is no mesh, so the plain data-parallel path is unchanged.
        self.rng = StatefulRNG(seed=int(recipe_cfg.get("shuffle_seed", 42)) + self._get_dp_rank(), ranked=False)
        self._build_checkpointer(target_path)
        self.load_checkpoint(self.cfg.get("checkpoint.restore_from", None))

    def _build_target_model(self, recipe_cfg, target_path: str) -> torch.nn.Module:
        """Load the frozen (optionally tensor-parallel) target model.

        With a ``distributed:`` section and ``tp_size>1`` the target is sharded
        in place by ``from_pretrained`` (its FSDP2 parallelize plan); the small
        draft stays replicated and runs DDP over the "dp" axis (which excludes
        "tp"), and the trainer module gathers the target's vocab-sharded lm_head
        / embed_tokens outputs. Absent, the original single-GPU-per-rank DP path
        is used. Sets ``self.dist_setup`` / ``self.device_mesh`` / ``self.dp_mesh``
        as a side effect and returns the (grad-disabled) target.
        """
        target_attn_implementation = recipe_cfg.get("target_attn_implementation", None)
        target_kwargs = dict(
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        )
        if target_attn_implementation is not None:
            target_kwargs["attn_implementation"] = target_attn_implementation
        self.dist_setup = None
        self.device_mesh = None
        self.dp_mesh = None
        # The target forward runs CP on "cp" (long-context memory relief); the draft,
        # dataloader sampler, and checkpointer key on "dp" (which excludes cp/tp, so
        # cp ranks in a dp group share data and draft weights). None without a mesh.
        self.cp_mesh = None
        if self.cfg.get("distributed", None) is not None:
            self.dist_setup = create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
            self.device_mesh = self.dist_setup.mesh_context.device_mesh
            self.dp_mesh = _submesh_or_none(self.device_mesh, "dp")
            self.cp_mesh = _submesh_or_none(self.device_mesh, "cp")
            target_kwargs["distributed_setup"] = self.dist_setup
            # CP gathers the target's captured layers back to the full sequence; that
            # gather does not yet handle a TP-sharded (DTensor) sequence, so the two
            # can't be combined. TP alone (draft replicated over "dp") is fine.
            cp_size = self.cp_mesh.size() if self.cp_mesh is not None else 1
            if cp_size > 1 and int(self.cfg.get("distributed.tp_size", 1) or 1) > 1:
                raise NotImplementedError(
                    "Context parallelism (cp_size>1) combined with tensor parallelism (tp_size>1) is not "
                    "yet supported for DFlash; the CP sequence gather does not handle a TP-sharded target "
                    "output. Set tp_size=1 or cp_size=1."
                )
            # The CP hook attends via the target's torch SDPA call; a custom-attention
            # (non-HF) target would silently skip it, so require the HF target path.
            if cp_size > 1 and not bool(recipe_cfg.get("target_force_hf", False)):
                raise NotImplementedError(
                    "Context parallelism (cp_size>1) requires recipe_args.target_force_hf=true so the "
                    "frozen target runs HuggingFace SDPA, which the CP K/V-gather hook intercepts."
                )
        target_model = NeMoAutoModelForCausalLM.from_pretrained(target_path, **target_kwargs)
        if self.dist_setup is None:
            # ``nn.Module.to`` is in-place; the sharded path is already placed by
            # ``from_pretrained``.
            target_model.to(self.device)
        target_model.requires_grad_(False)
        return target_model

    def _draft_ddp_process_group(self):
        """Process group for the draft's gradient all-reduce.

        With tensor parallelism the draft is replicated across tp ranks, so a
        full-world all-reduce would average duplicate gradients; restrict it to
        the "dp" sub-axis (which excludes tp) so it reduces only across real data
        replicas. Without a mesh (tp_size=1) ``dp_mesh`` is None -> return None ->
        the default full-world group, unchanged.
        """
        if self.dp_mesh is not None and self.dp_mesh.size() < self.dist_env.world_size:
            return self.dp_mesh.get_group()
        return None

    def _build_target_wrapper(self, target_layer_ids: list[int]) -> HFDFlashTargetModel:
        """Build the frozen-target hidden-state capture wrapper.

        Subclasses override to capture extra teacher signals (e.g. JetSpec also
        captures the target logits for its forward-KL distillation).
        """
        return HFDFlashTargetModel(
            self.target_model, target_layer_ids=target_layer_ids, cp_mesh=getattr(self, "cp_mesh", None)
        )

    def _build_dflash_config(self, recipe_cfg, target_layer_ids: list[int]) -> dict:
        """Build the draft ``dflash_config`` block. Subclasses extend it (e.g. Domino)."""
        return {
            "mask_token_id": self.mask_token_id,
            "target_layer_ids": target_layer_ids,
        }

    def _build_trainer_module(self, attention_backend: str, recipe_cfg):
        """Build the trainer wrapper. Subclasses override to swap the wrapper (e.g. Domino)."""
        return DFlashTrainerModule(
            draft_model=self.draft_model,
            target_lm_head=self.target_model.get_output_embeddings(),
            target_embed_tokens=self.target_model.get_input_embeddings(),
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
            attention_backend=attention_backend,
            num_anchors=int(recipe_cfg.get("num_anchors", 512)),
            # Paper default (Appendix A.1) for the shipped block_size=16 configs;
            # matches DFlashDecayLoss's own default. Set null explicitly in YAML
            # to disable the position decay (uniform weighting).
            loss_decay_gamma=recipe_cfg.get("loss_decay_gamma", 7.0),
        )

    def _run_trainer_step(self, target_batch):
        """Run one trainer-module forward. Subclasses override to inject extra inputs (e.g. lambda_base)."""
        return self.trainer_module(
            input_ids=target_batch.input_ids,
            hidden_states=target_batch.hidden_states,
            loss_mask=target_batch.loss_mask,
        )

    def _log_extra_train_metrics(self, epoch_idx: int) -> None:
        """Hook for subclasses to log extra per-step metrics at a log point (no-op here)."""

    @staticmethod
    def _resolve_mask_token_id(recipe_cfg, vocab_size: int) -> int:
        """Resolve and validate the MASK token id that fills non-anchor block positions.

        DFlash fills every non-anchor slot of a ``[anchor, MASK, MASK, ...]`` block
        with this id, and the draft's ``embed_tokens`` row at that id becomes the
        learned "predict here" signal. It must be chosen deliberately (a reserved /
        unused token), exactly like P-EAGLE's ``mask_token_id``: the previous silent
        fallback to ``tokenizer.pad_token_id`` was unsafe because ``pad`` is commonly
        aliased to ``eos`` (or another meaningful token), which conflates the mask
        signal with real content and quietly degrades acceptance without erroring.
        Require it explicitly and range-check it; the inference runtime must fill the
        block slots with the same id.
        """
        mask_token_id = recipe_cfg.get("mask_token_id", None)
        if mask_token_id is None:
            raise ValueError(
                "DFlash requires recipe_args.mask_token_id to be set explicitly (the token used for "
                "non-anchor block positions). Pick a reserved / rarely-used token id -- e.g. a model-specific "
                "reserved special token -- so the mask-slot embedding does not collide with real content, and "
                "use the same id in the inference runtime. (The previous fallback to tokenizer.pad_token_id was "
                "removed: pad is frequently aliased to eos, which silently degrades quality.)"
            )
        mask_token_id = int(mask_token_id)
        if not 0 <= mask_token_id < vocab_size:
            raise ValueError(
                f"mask_token_id={mask_token_id} is out of range for the vocab [0, {vocab_size}); "
                "it indexes the draft embed_tokens table."
            )
        return mask_token_id

    def _build_checkpointer(self, target_path: str) -> None:
        """Build the checkpointer using the same plumbing as the EAGLE recipes."""
        ckpt_cfg = self.cfg.get("checkpoint", None)
        default_dir = str(self.output_dir / "checkpoints")
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
        # The draft is replicated (never TP-sharded), so key the checkpoint shard
        # on the dp coordinate -- identical for every tp rank in a replica --
        # rather than the global rank. dp_mesh is None without a mesh (tp_size=1)
        # -> global rank, unchanged. tp_rank stays 0 (the draft is not sharded).
        dp_rank = (
            self.dp_mesh.get_local_rank()
            if getattr(self, "dp_mesh", None) is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        self.checkpointer = Checkpointer(
            config=self.checkpoint_config, dp_rank=dp_rank, tp_rank=0, pp_rank=0, moe_mesh=None
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
        is_final_checkpoint: bool = False,
    ) -> None:
        """Persist the DFlash draft model, optimizer, scheduler, RNG, and meta."""
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
        self.checkpointer.save_model(
            draft_model,
            path,
            tokenizer=self.tokenizer,
            is_final_checkpoint=is_final_checkpoint,
        )
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
        """Persist DFlash meta: global_step, epoch, block_size, and target layers."""
        torch.save(
            {
                "global_step": self.runtime.global_step,
                "epoch": int(epoch),
                "block_size": self.block_size,
                "mask_token_id": self.mask_token_id,
                "target_layer_ids": list(self.target_wrapper.target_layer_ids),
            },
            os.path.join(path, "dflash_meta.pt"),
        )

    def load_checkpoint(self, restore_from: str | None = None) -> None:
        """Restore the DFlash draft model, optimizer, scheduler, RNG, and global_step."""
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
        if not ok and not restore_from:
            if is_rank_0:
                logger.warning(
                    "Auto-detected checkpoint at %s is incompatible: %s. Skipping restore.", ckpt_dir, reason
                )
            return

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
        """Restore DFlash meta: global_step and epoch, and validate mask_token_id."""
        meta_path = os.path.join(ckpt_dir, "dflash_meta.pt")
        if os.path.exists(meta_path):
            meta = torch.load(meta_path, weights_only=False, map_location="cpu")
            self.runtime.global_step = int(meta.get("global_step", 0))
            self._resume_epoch = int(meta.get("epoch", 0))
            # ``mask_token_id`` comes only from the resume YAML (it is not
            # restored from the checkpoint); the draft's ``embed_tokens`` row at
            # that id is the learned "predict here" signal and the inference
            # runtime fills block slots with the same id. A resume YAML whose
            # ``mask_token_id`` disagrees with the trained one silently points the
            # mask slots at an untrained embedding row and degrades acceptance
            # with no error, so fail loudly on a mismatch. Legacy checkpoints
            # saved before this field existed (``None``) skip the check.
            saved_mask_token_id = meta.get("mask_token_id", None)
            if saved_mask_token_id is not None and int(saved_mask_token_id) != int(self.mask_token_id):
                raise ValueError(
                    f"mask_token_id mismatch on resume: the checkpoint at {ckpt_dir} was trained with "
                    f"mask_token_id={int(saved_mask_token_id)}, but recipe_args.mask_token_id="
                    f"{int(self.mask_token_id)}. The draft's mask-slot embedding was learned at the "
                    f"checkpoint's id; set recipe_args.mask_token_id={int(saved_mask_token_id)} to resume."
                )

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured."""
        every = getattr(self, "ckpt_every_steps", None)
        if every is None or every <= 0 or self.runtime.global_step % every != 0:
            return False
        total_optim_steps = getattr(self, "total_optim_steps", None)
        is_final_checkpoint = total_optim_steps is not None and self.runtime.global_step >= total_optim_steps
        self.save_checkpoint(
            epoch=epoch,
            step=self.runtime.global_step,
            best_metric_key="val_loss",
            is_final_checkpoint=is_final_checkpoint,
        )
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Always save the fully-trained model at the end, unless a cadence already saved the final step."""
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        every = getattr(self, "ckpt_every_steps", None)
        saved_by_step = bool(every and every > 0 and gs % every == 0)
        saved_by_epoch = bool(getattr(self, "save_checkpoint_every_epoch", False))
        if saved_by_step or saved_by_epoch:
            return False
        self.save_checkpoint(epoch=completed_epochs, step=gs, best_metric_key="val_loss", is_final_checkpoint=True)
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

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
                target_batch = self.target_wrapper.generate_batch(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    loss_mask=batch["loss_mask"],
                )
                try:
                    # Route through the same seam as training so subclass-specific
                    # inputs (Domino's lambda_base, JetSpec's target_logits) are wired.
                    metrics = self._run_trainer_step(target_batch)
                except NoValidAnchorsError:
                    # Every sample in this micro-batch is too short to form a block;
                    # skip it without counting, mirroring the training loop.
                    continue
                total_loss += metrics.loss.detach()
                total_acc += metrics.accuracy.detach()
                total_batches += 1
        total_loss = _all_reduce_mean(total_loss)
        total_acc = _all_reduce_mean(total_acc)
        total_batches = _all_reduce_mean(total_batches)
        self.trainer_module.train()
        return {
            "val_loss": (total_loss / total_batches.clamp_min(1)).item(),
            "val_accuracy": (total_acc / total_batches.clamp_min(1)).item(),
        }

    def run_train_validation_loop(self):
        """Run the DFlash training loop."""
        self.trainer_module.train()
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            return

        pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
        try:
            for epoch_idx in range(start_epoch, self.num_epochs):
                if hasattr(self.train_dataloader, "sampler") and hasattr(self.train_dataloader.sampler, "set_epoch"):
                    self.train_dataloader.sampler.set_epoch(epoch_idx)

                running_loss = 0.0
                running_acc = 0.0
                running_micro = 0
                epoch_loss = 0.0
                micro_step = 0
                pending_micro_batches = 0
                completed_steps = 0
                last_batch_idx = -1
                num_batches = len(self.train_dataloader)
                is_ddp = isinstance(self.trainer_module, DistributedDataParallel)
                for batch_idx, batch in enumerate(self.train_dataloader):
                    last_batch_idx = batch_idx
                    batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                    target_batch = self.target_wrapper.generate_batch(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        loss_mask=batch["loss_mask"],
                    )
                    sync_grads = should_sync_grads(
                        pending_micro_batches=pending_micro_batches,
                        grad_accumulation_steps=self.grad_accumulation_steps,
                        batch_idx=batch_idx,
                        batches_per_epoch=num_batches,
                        is_ddp=is_ddp,
                    )
                    sync_ctx = nullcontext() if sync_grads else self.trainer_module.no_sync()
                    with sync_ctx:
                        local_has_valid = 1
                        try:
                            metrics = self._run_trainer_step(target_batch)
                            loss = metrics.loss / self.grad_accumulation_steps
                        except NoValidAnchorsError:
                            # Every sample in this micro-batch is too short to form a
                            # block; nothing to learn from it on this rank.
                            local_has_valid = 0
                        # Decide skip-vs-backward in lockstep so a per-rank skip never
                        # leaves one rank issuing its gradient all-reduce alone (DDP
                        # hang) or desyncs the accumulation windows: if ANY rank has no
                        # valid anchors, ALL ranks skip this micro-batch together.
                        all_have_valid = _all_ranks_have_valid(local_has_valid, is_ddp, self.device)
                        if all_have_valid:
                            loss.backward()
                    if not all_have_valid:
                        self._skipped_micro_batches += 1
                        continue

                    running_loss += metrics.loss.detach().item()
                    running_acc += metrics.accuracy.detach().item()
                    running_micro += 1
                    epoch_loss += metrics.loss.detach().item()
                    micro_step += 1
                    pending_micro_batches += 1

                    if pending_micro_batches == self.grad_accumulation_steps:
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

                        if self.dist_env.is_main and self.runtime.global_step % self.log_every_steps == 0:
                            # Average over the micro-batches accumulated since the last
                            # log, not over optimizer steps: with grad_accumulation_steps>1
                            # (or skipped short micro-batches) the two differ, and dividing
                            # by log_every_steps would inflate the reported loss/acc.
                            avg_loss = running_loss / max(1, running_micro)
                            avg_acc = running_acc / max(1, running_micro)
                            current_lr = self.lr_scheduler.get_last_lr()[0]
                            if pbar is not None:
                                pbar.set_postfix(
                                    loss=f"{avg_loss:.4f}",
                                    acc=f"{avg_acc:.4f}",
                                    lr=f"{current_lr:.2e}",
                                )
                            logger.info(
                                "epoch=%d step=%d loss=%.4f acc=%.4f lr=%.6g",
                                epoch_idx,
                                self.runtime.global_step,
                                avg_loss,
                                avg_acc,
                                current_lr,
                            )
                            self._log_extra_train_metrics(epoch_idx)
                            running_loss = 0.0
                            running_acc = 0.0
                            running_micro = 0

                # Flush the trailing partial accumulation window (see EAGLE recipes
                # for the rescale rationale).
                if pending_micro_batches > 0:
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
                    msg = (
                        f"Finished epoch {epoch_idx + 1}/{self.num_epochs} completed_steps={completed_steps} "
                        f"skipped_short_micro_batches={self._skipped_micro_batches}"
                    )
                    if eval_metrics is not None:
                        msg += (
                            f" val_loss={eval_metrics['val_loss']:.4f} val_accuracy={eval_metrics['val_accuracy']:.4f}"
                        )
                    logger.info(msg)

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
                    self._log_saved_checkpoint("epoch", epoch_idx + 1, self.runtime.global_step)

            self._maybe_save_final_checkpoint(self.num_epochs)
            self._finalize_pending_checkpoint()
        finally:
            if pbar is not None:
                pbar.close()


def main(config_path: str | None = None):
    """Entrypoint for ``TrainDFlashRecipe``."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainDFlashRecipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
