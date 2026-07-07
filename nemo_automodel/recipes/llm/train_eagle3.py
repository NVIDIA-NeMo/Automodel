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
import os
import pathlib
from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace

import torch
import torch.distributed as dist
from huggingface_hub import constants as hf_constants
from torch.nn.parallel import DistributedDataParallel
from transformers import AutoConfig

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer
from nemo_automodel.components.checkpoint.checkpointing import (
    CheckpointingConfig,
    save_config,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.eagle3 import (
    build_eagle3_dataloader,
    load_or_build_eagle3_token_mapping,
)
from nemo_automodel.components.datasets.llm.eagle3_cache import (
    build_cached_eagle3_dataloader,
    read_manifest,
)
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.loggers.log_utils import setup_logging
from nemo_automodel.components.loggers.wandb_utils import init_wandb_run, suppress_wandb_log_messages
from nemo_automodel.components.speculative.eagle import (
    Eagle3TrainerModule,
    HFEagle3TargetModel,
    simulated_accept_length,
)
from nemo_automodel.components.speculative.eagle.registry import resolve_eagle3_draft_spec
from nemo_automodel.components.speculative.eagle.remote import RemoteEagle3TargetModel
from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.utils.model_utils import print_trainable_parameters
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
from nemo_automodel.recipes.llm.peagle_recipe import PeagleRecipeMixin

logger = logging.getLogger(__name__)


def _all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / dist.get_world_size()
    return value


def _all_reduce_sum(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return value


def _window_tau_sim(step_correct: torch.Tensor | None, step_valid: torch.Tensor | None) -> float | None:
    """Simulated accept length over a metrics window, reduced across ranks.

    ``step_correct`` / ``step_valid`` are the window's accumulated per-TTT-step
    hit / supervised counts (``None`` when the trainer does not report them,
    e.g. P-EAGLE). Counts are extensive, so they are sum-reduced across ranks
    before forming the per-step accuracies; every rank must therefore call
    this at the same point. Returns ``None`` when there is nothing to report.
    """
    if step_correct is None or step_valid is None:
        return None
    correct = _all_reduce_sum(step_correct.clone())
    valid = _all_reduce_sum(step_valid.clone())
    if valid.sum().item() <= 0:
        return None
    return simulated_accept_length(correct, valid).item()


def _submesh_or_none(device_mesh, name: str):
    """Return the named 1D submesh (e.g. "cp"/"dp") or None if absent.

    Uses ``get_flat_mesh`` so ``_flatten()``-created axes ("dp") resolve on
    PyTorch 2.9-2.11, where a plain ``device_mesh[name]`` is deprecated or the
    name is missing from ``mesh_dim_names``.
    """
    if device_mesh is None:
        return None
    try:
        return get_flat_mesh(device_mesh, name)
    except KeyError:
        return None


def _validate_cp_gates(cp_size: int, backend: str, packed_sequence_size: int) -> None:
    """Reject context-parallel combinations the EAGLE-3 target path cannot honor.

    CP shards the target forward along the sequence and forces ``is_causal`` (the
    self_attn hooks strip the attention_mask), so it is incompatible with sequence
    packing (which needs the 4D block-causal mask) and with the remote backend
    (whose target runs out-of-process).
    """
    if cp_size > 1 and backend == "remote":
        raise NotImplementedError(
            "Context parallelism (cp_size>1) is only supported with the colocated target "
            "backend; the remote backend runs the target out-of-process."
        )
    if cp_size > 1 and packed_sequence_size > 0:
        raise NotImplementedError(
            "Context parallelism (cp_size>1) is not yet supported with sequence packing; CP "
            "strips the 4D block-causal mask that packing relies on. Set cp_size=1 or "
            "packed_sequence_size=0."
        )


def _validate_tp_gates(tp_size: int, backend: str, cp_size: int) -> None:
    """Reject tensor-parallel combinations the EAGLE-3 target path cannot honor.

    TP shards only the colocated target (the FSDP2 parallelize plan column/row
    shards its linears and makes the lm_head logits a vocab-sharded ``DTensor``,
    which the target wrapper gathers). It is therefore meaningless with the remote
    backend (the target runs out-of-process), and combined TP+CP is not yet wired:
    the CP gather path does not handle a TP-sharded (DTensor) sequence.
    """
    if tp_size > 1 and backend != "colocated":
        raise NotImplementedError(
            "Tensor parallelism (tp_size>1) is only supported with the colocated target backend; "
            f"the {backend!r} backend runs the target out-of-process."
        )
    if tp_size > 1 and cp_size > 1:
        raise NotImplementedError(
            "Tensor parallelism (tp_size>1) combined with context parallelism (cp_size>1) is not "
            "yet supported; the CP sequence gather does not handle a TP-sharded target output. "
            "Set tp_size=1 or cp_size=1."
        )


def _best_effort(label: str, fn) -> None:
    """Run a teardown step, logging (never raising) on failure so one failed step
    does not abort the rest of cleanup."""
    try:
        fn()
    except Exception:
        logger.exception("error %s during cleanup", label)


def _validate_peagle_gates(backend: str, cached_target_path, packed_sequence_size: int) -> None:
    """Reject P-EAGLE (parallel_drafting) combinations its trainer cannot honor.

    ``PEagleTrainerModule.forward`` consumes the live colocated target's full-vocab
    ``target_logits`` only. The remote and offline-cache backends instead supply
    precomputed draft-vocab ``target_probs``/``position_mask`` (a parameter
    mismatch), and sequence packing feeds ``position_ids``/``seq_lens``/
    ``doc_remaining`` the P-EAGLE forward does not accept (and the partitioned
    path would run the target on a packed row without per-document masking,
    leaking across documents). P-EAGLE only safely supports a colocated live
    target on non-packed sequences.
    """
    if backend != "colocated":
        raise NotImplementedError(
            f"parallel_drafting (P-EAGLE) only supports target_model_backend='colocated', got "
            f"{backend!r}. The remote backend supplies precomputed draft-vocab supervision, which "
            f"the P-EAGLE trainer (full-vocab target_logits) does not accept."
        )
    if cached_target_path is not None:
        raise NotImplementedError(
            "parallel_drafting (P-EAGLE) does not support the offline cached target "
            "(cached_target_path); the cache stores precomputed draft-vocab supervision consumed "
            "only by the EAGLE-3 TTT trainer."
        )
    if packed_sequence_size > 0:
        raise NotImplementedError(
            "parallel_drafting (P-EAGLE) does not support sequence packing (packed_sequence_size>0); "
            "the P-EAGLE forward does not accept the per-document packing metadata "
            "(position_ids/seq_lens/doc_remaining)."
        )


class TrainEagle3Recipe(PeagleRecipeMixin, BaseRecipe):
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
        # P-EAGLE (parallel_drafting) only safely supports a colocated live target
        # on non-packed sequences; gate the unsupported combinations early, before
        # loading the target -- see _validate_peagle_gates.
        if bool(recipe_cfg.get("parallel_drafting", False)):
            _validate_peagle_gates(
                backend=recipe_cfg.get("target_model_backend", "colocated"),
                cached_target_path=self.cached_target_path,
                packed_sequence_size=int(recipe_cfg.get("packed_sequence_size", 0) or 0),
            )
        self.dist_setup = None
        self.distributed_config = None
        self.device_mesh = None
        self.moe_mesh = None
        # Context-parallel ("cp") and data-parallel ("dp") submeshes, populated
        # from device_mesh when a colocated target builds one (None otherwise).
        self.cp_mesh = None
        self.dp_mesh = None
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
        # EAGLE-3.1 drafter-side toggles. ``fc_norm`` adds one RMSNorm per aux
        # hidden-state chunk before ``model.fc``; ``norm_output`` routes the
        # post-``norm`` hidden state into ``compute_logits``. The draft reads
        # both from its config (default False), so they must be copied from
        # ``recipe_args`` into the draft config -- and thereby serialized into
        # the draft ``config.json`` -- for the flags to take effect at train
        # and serve time alike.
        draft_config["fc_norm"] = bool(recipe_cfg.get("fc_norm", False))
        draft_config["norm_output"] = bool(recipe_cfg.get("norm_output", False))
        # P-EAGLE (parallel drafting). When enabled, the draft registers a
        # learnable ``mask_hidden`` placeholder and the trainer predicts all
        # ``num_depths`` draft tokens in a single COD-subsampled parallel forward
        # (https://github.com/vllm-project/speculators/pull/480) instead of
        # EAGLE-3's autoregressive TTT unroll. ``mask_token_id`` is the reserved
        # token id placed at masked multi-token-prediction slots; ``num_depths``
        # and the COD ratios are serialized into the draft ``config.json`` so the
        # saved checkpoint loads into vLLM's parallel-drafting runtime unchanged.
        parallel_drafting = bool(recipe_cfg.get("parallel_drafting", False))
        draft_config["parallel_drafting"] = parallel_drafting
        mask_token_id = None
        if parallel_drafting:
            mask_token_id = self._configure_peagle_draft_config(recipe_cfg, draft_config, target_config)
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
        # Embed the draft->target vocab map (d2t/t2d) into the draft so a
        # compressed-vocab checkpoint carries the remap tables vLLM/SGLang need.
        # No-op when the vocab is not compressed. See set_vocab_mapping.
        self.draft_model.set_vocab_mapping(selected_token_ids)
        # EAGLE-3 TTT freezes the draft embeddings by default; P-EAGLE trains them
        # (speculators sets ``embed_requires_grad=True`` for parallel drafting).
        # Either default can still be overridden via ``recipe_args.freeze_embeddings``.
        freeze_embeddings_default = not parallel_drafting
        if recipe_cfg.get("freeze_embeddings", freeze_embeddings_default):
            self.draft_model.freeze_embeddings()
        # P-EAGLE memory knob: recompute the draft layers' activations in the
        # backward instead of storing them, lowering the activation peak of the
        # long flattened COD sequence (complements ``sequence_partitions``).
        # Off by default; only affects the parallel-drafting forward.
        if recipe_cfg.get("draft_gradient_checkpointing", False):
            self.draft_model.gradient_checkpointing_enable()
        # The target's "Model summary" is logged by apply_model_infrastructure when it
        # loads; the draft is built directly, so log its (trainable) summary here too.
        print_trainable_parameters(self.draft_model, name="Draft")

        if parallel_drafting:
            trainer_module = self.build_peagle_trainer(
                recipe_cfg, selected_token_ids, selected_token_mask, mask_token_id
            )
        else:
            trainer_module = Eagle3TrainerModule(
                self.draft_model,
                selected_token_ids=selected_token_ids,
                selected_token_mask=selected_token_mask,
                ttt_steps=recipe_cfg.ttt_steps,
            ).to(self.device)
        if self.dist_env.world_size > 1:
            # Under context parallelism the draft is replicated across cp ranks
            # (it runs on the full gathered sequence), so restrict the gradient
            # all-reduce to the dp sub-axis to avoid redundant cp all-reduces.
            # With cp_size==1 the dp group spans the whole world -> process_group
            # is None -> today's full-world DDP, unchanged.
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
        self.target_prefetch_depth = self._resolve_prefetch_depth(recipe_cfg)
        self.num_epochs = recipe_cfg.num_epochs
        self.log_every_steps = recipe_cfg.get("log_every_steps", 10)
        # Checkpoint cadence. The two knobs are independent:
        #   * ``ckpt_every_steps``            -- save every N optimizer steps (None/<=0 = off).
        #   * ``save_checkpoint_every_epoch`` -- save at each epoch boundary (off by default).
        # The fully-trained model is always saved once the run completes; these
        # only add intermediate checkpoints (and stack when both are set). With
        # both off, the end-of-run checkpoint is the only one written.
        # NOTE: field names mirror StepScheduler (components/training/step_scheduler.py),
        # which the SFT recipe uses for the same cadence; EAGLE hand-rolls its own
        # loop, so a future refactor could adopt StepScheduler here too.
        self.ckpt_every_steps = recipe_cfg.get("ckpt_every_steps", None)
        self.save_checkpoint_every_epoch = recipe_cfg.get("save_checkpoint_every_epoch", False)
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
            self.num_epochs * optim_steps_per_epoch(num_batches_per_epoch, self.grad_accumulation_steps),
        )
        warmup_ratio = float(opt_cfg.get("warmup_ratio", 0.05))
        min_lr_ratio = float(opt_cfg.get("min_lr_ratio", 0.1))
        warmup_steps = max(1, int(warmup_ratio * total_optim_steps))
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, make_warmup_cosine_schedule(warmup_steps, total_optim_steps, min_lr_ratio)
        )
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

        # Optional Weights & Biases logging (rank 0 only).
        self.wandb_run = None
        if self.dist_env.is_main and self.cfg.get("wandb", None) is not None:
            suppress_wandb_log_messages()
            self.wandb_run = init_wandb_run(
                self.cfg.wandb.to_dict(),
                self.cfg.to_dict(),
                default_name="eagle3_" + str(target_path).rstrip("/").split("/")[-1],
            )

    def _setup_online_target(self, recipe_cfg, target_path, target_config):
        """Live path: load the target model and build the live dataloader.

        Sets ``self.target_model`` / ``self.target_wrapper`` /
        ``self.train_dataloader`` / ``self.val_dataloader`` and returns the
        ``(selected_token_ids, selected_token_mask)`` draft-vocab mapping built
        by scanning the training data.
        """
        # ``target_model_backend`` selects where the frozen target runs:
        #   - ``colocated`` (default): load the target on this GPU and capture
        #     supervision in-process via the HuggingFace forward.
        #   - ``sglang``: like ``colocated`` but the in-process forward runs
        #     through SGLang's ModelRunner, which is substantially faster than
        #     the HF eager forward for mainstream architectures.
        #   - ``vllm``: like ``sglang`` but the in-process forward runs through
        #     vLLM (via its native ``extract_hidden_states`` path); supervision is
        #     numerically equivalent to the co-located HF backend.
        #   - ``remote``: the target runs as a standalone server (see
        #     ``serve_target``); this process only holds the draft and pulls
        #     precomputed supervision over HTTP + NCCL. No target weights are
        #     loaded here, which frees the training GPU's memory.
        backend = recipe_cfg.get("target_model_backend", "colocated")
        if backend not in ("colocated", "sglang", "vllm", "remote"):
            raise ValueError(
                f"Unknown target_model_backend={backend!r}; expected 'colocated', 'sglang', 'vllm', or 'remote'."
            )
        # Sequence packing is colocated-only: neither the remote server nor the
        # SGLang runner honors per-document masking (SGLang treats each row as one
        # full causal sequence), so a packed row would leak supervision across
        # document boundaries.
        packed_sequence_size = recipe_cfg.get("packed_sequence_size", 0)
        if packed_sequence_size > 0 and backend != "colocated":
            raise NotImplementedError(
                "packed_sequence_size > 0 is only supported with the colocated target backend; "
                f"the {backend!r} backend does not propagate per-document masking."
            )
        # Context- and tensor-parallel gates (read from config: the cp/tp
        # submeshes are only built later, inside the colocated path).
        cp_size = int(self.cfg.get("distributed.cp_size", 1) or 1)
        tp_size = int(self.cfg.get("distributed.tp_size", 1) or 1)
        _validate_cp_gates(cp_size, backend, packed_sequence_size)
        _validate_tp_gates(tp_size, backend, cp_size)
        if backend == "remote":
            self._setup_remote_target(recipe_cfg)
        elif backend == "sglang":
            self._setup_sglang_target(recipe_cfg, target_path)
        elif backend == "vllm":
            self._setup_vllm_target(recipe_cfg, target_path)
        else:  # colocated
            self._setup_colocated_target(recipe_cfg, target_path)

        self.train_dataloader = build_eagle3_dataloader(
            data_path=recipe_cfg.train_data_path,
            tokenizer=self.tokenizer,
            seq_length=recipe_cfg.seq_length,
            batch_size=recipe_cfg.micro_batch_size,
            shuffle=recipe_cfg.get("train_shuffle", True),
            num_workers=recipe_cfg.get("num_workers", 0),
            split=recipe_cfg.get("train_split", None),
            distributed=self.dist_env.world_size > 1,
            shuffle_seed=recipe_cfg.get("shuffle_seed", 42),
            mask_reasoning_content=recipe_cfg.get("mask_reasoning_content", False),
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
                packed_sequence_size=packed_sequence_size,
                dp_mesh=self.dp_mesh,
            )

        special_token_ids = [
            getattr(self.tokenizer, "bos_token_id", None),
            getattr(self.tokenizer, "eos_token_id", None),
            getattr(self.tokenizer, "pad_token_id", None),
            getattr(self.tokenizer, "unk_token_id", None),
        ]
        # ``selected_token_ids_path`` (optional) caches the draft-vocab selection
        # so reruns skip the full-dataset frequency scan. When unset, the mapping
        # is rebuilt every setup (original behavior). On resume this still runs,
        # but ``_load_extra_state`` then overrides it with the checkpoint's saved
        # mapping -- so the cache only matters for cold starts.
        selected_token_ids, selected_token_mask = load_or_build_eagle3_token_mapping(
            self.train_dataloader,
            target_vocab_size=target_config.vocab_size,
            draft_vocab_size=recipe_cfg.get("draft_vocab_size", None),
            special_token_ids=special_token_ids,
            cache_path=recipe_cfg.get("selected_token_ids_path", None),
        )
        # A remote target computes ``target_probs`` server-side, so it needs the
        # draft-vocab mapping. Co-located backends keep it on the trainer module
        # (the default ``set_vocab_mapping`` is a no-op).
        self.target_wrapper.set_vocab_mapping(selected_token_ids, selected_token_mask)
        return selected_token_ids, selected_token_mask

    def _setup_colocated_target(self, recipe_cfg, target_path):
        """Load the target on this GPU and capture supervision in-process."""
        # Optional ``distributed:`` YAML section. Required for targets that do
        # not fit on a single GPU (e.g. Qwen3-30B-A3B MoE). ``force_hf`` is
        # opt-in; default ``False`` so HF architectures with an AutoModel custom
        # impl take the MoE-capable custom path.
        target_kwargs = dict(
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            torch_dtype=self.compute_dtype,
            force_hf=bool(recipe_cfg.get("target_force_hf", False)),
        )
        # Optional target attention backend (default: HF auto-select). Pin to
        # ``sdpa`` to dodge the Qwen3 FA2 ``s_aux=None`` crash in transformers.
        target_attn_implementation = recipe_cfg.get("target_attn_implementation", None)
        if target_attn_implementation is not None:
            target_kwargs["attn_implementation"] = target_attn_implementation
        if self.cfg.get("distributed", None) is not None:
            self.dist_setup = create_distributed_setup_from_config(self.cfg, world_size=self.dist_env.world_size)
            self.distributed_config = self.dist_setup.strategy_config
            self.device_mesh = self.dist_setup.mesh_context.device_mesh
            self.moe_mesh = self.dist_setup.mesh_context.moe_mesh
            # Capture the cp/dp submeshes: the target forward runs CP on "cp",
            # while the draft DDP group, dataloader sampler, and checkpointer key
            # on "dp" (cp ranks within a dp group share data and draft weights).
            # Tensor parallelism (distributed.tp_size>1) needs no submesh here: the
            # target's linears are sharded in place by ``from_pretrained`` below
            # (its FSDP2 parallelize plan), the wrapper gathers the resulting
            # vocab-sharded logits, and the flattened "dp" axis already excludes
            # the "tp" axis so the draft and sampler replicate across TP ranks.
            self.cp_mesh = _submesh_or_none(self.device_mesh, "cp")
            self.dp_mesh = _submesh_or_none(self.device_mesh, "dp")
            target_kwargs.update(
                distributed_setup=self.dist_setup,
            )
        self.target_model = NeMoAutoModelForCausalLM.from_pretrained(target_path, **target_kwargs)
        # ``nn.Module.to`` is in-place; reassigning ``self.target_model`` would
        # re-trigger ``BaseRecipe.__setattr__`` state-tracking and raise.
        if self.dist_setup is None:
            self.target_model.to(self.device)
        # The target is frozen: it only supplies aux hidden states / logits as
        # supervision and is never optimized (the optimizer is built solely from
        # the draft trainer module). Mark the parameters explicitly so no future
        # code path accidentally trains the target -- matching EAGLE-1/2.
        self.target_model.requires_grad_(False)
        self.target_wrapper = HFEagle3TargetModel(
            self.target_model,
            aux_layer_ids=recipe_cfg.get("aux_layer_ids", None),
            cp_mesh=self.cp_mesh,
        )

    def _setup_sglang_target(self, recipe_cfg, target_path):
        """Co-located SGLang target: serve the frozen target through SGLang on this GPU.

        Same supervision contract as ``colocated`` (full-vocab logits shipped to
        the trainer, draft-vocab projection trainer-side), but the target forward
        runs through SGLang's ModelRunner. SGLang carves its weight + KV pool out
        of this GPU up front (``mem_fraction_static``), and the draft trains in
        the remainder.
        """
        if self.device.type != "cuda":
            raise ValueError("target_model_backend='sglang' requires CUDA; use 'colocated' for CPU runs.")
        if self.dist_env.world_size > 1:
            # SGLang's global parallel state requires world_size == tp_size * pp_size,
            # so per-rank tp=1 runners cannot share a multi-rank training process
            # group. Multi-GPU runs split target and draft onto separate processes
            # instead: ``serve_target --engine sglang`` + ``target_model_backend='remote'``.
            raise ValueError(
                "target_model_backend='sglang' supports single-process training only; "
                "for multi-GPU runs serve the target separately (serve_target --engine sglang) "
                "and set target_model_backend='remote'."
            )
        from nemo_automodel.components.speculative.eagle.sglang_target import SGLangEagle3TargetModel

        sglang_args = recipe_cfg.get("sglang_args", None) or {}
        sglang_kwargs = sglang_args.to_dict() if hasattr(sglang_args, "to_dict") else dict(sglang_args)
        # SGLang's ServerArgs default (~0.88 of GPU memory) would starve the
        # draft's optimizer states and activations; default to half the GPU and
        # let ``recipe_args.sglang_args.mem_fraction_static`` override.
        sglang_kwargs.setdefault("mem_fraction_static", 0.5)
        self.target_model = None
        self.target_wrapper = SGLangEagle3TargetModel.from_pretrained(
            target_path,
            aux_layer_ids=recipe_cfg.get("aux_layer_ids", None),
            dtype=self.compute_dtype,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            **sglang_kwargs,
        )

    def _setup_vllm_target(self, recipe_cfg, target_path):
        """Co-located vLLM target: serve the frozen target through vLLM on this GPU.

        Same supervision contract as ``colocated`` (full-vocab logits shipped to
        the trainer, draft-vocab projection trainer-side), but the target forward
        runs through vLLM's ``extract_hidden_states`` path. vLLM carves its weight
        + KV pool out of this GPU up front (``gpu_memory_utilization``), and the
        draft trains in the remainder.
        """
        if self.device.type != "cuda":
            raise ValueError("target_model_backend='vllm' requires CUDA; use 'colocated' for CPU runs.")
        if self.dist_env.world_size > 1:
            # vLLM owns its own engine process group, so per-rank tp=1 engines
            # cannot share a multi-rank training process group. Multi-GPU runs
            # split target and draft onto separate processes instead:
            # ``serve_target --engine vllm`` + ``target_model_backend='remote'``.
            raise ValueError(
                "target_model_backend='vllm' supports single-process training only; "
                "for multi-GPU runs serve the target separately (serve_target --engine vllm) "
                "and set target_model_backend='remote'."
            )
        from nemo_automodel.components.speculative.eagle.vllm_target import VLLMEagle3TargetModel

        vllm_args = recipe_cfg.get("vllm_args", None) or {}
        vllm_kwargs = vllm_args.to_dict() if hasattr(vllm_args, "to_dict") else dict(vllm_args)
        # vLLM's default (~0.9 of GPU memory) would starve the draft's optimizer
        # states and activations; default to half the GPU and let
        # ``recipe_args.vllm_args.gpu_memory_utilization`` override.
        vllm_kwargs.setdefault("gpu_memory_utilization", 0.5)
        self.target_model = None
        self.target_wrapper = VLLMEagle3TargetModel.from_pretrained(
            target_path,
            aux_layer_ids=recipe_cfg.get("aux_layer_ids", None),
            dtype=self.compute_dtype,
            trust_remote_code=recipe_cfg.get("trust_remote_code", False),
            **vllm_kwargs,
        )

    def _setup_remote_target(self, recipe_cfg):
        """Connect to one or more remote target servers (no target loaded here)."""
        urls = recipe_cfg.get("remote_urls", None)
        if not urls and recipe_cfg.get("remote_url", None):
            urls = [recipe_cfg.remote_url]
        if not urls:
            raise ValueError(
                "target_model_backend='remote' requires recipe_args.remote_urls "
                "(or remote_url) pointing at a running serve_target instance."
            )
        self.target_model = None
        self.target_wrapper = RemoteEagle3TargetModel.from_urls(
            list(urls),
            device=self.device,
            timeout=recipe_cfg.get("remote_timeout", 120),
            max_retries=recipe_cfg.get("remote_max_retries", 3),
        )

    def _resolve_prefetch_depth(self, recipe_cfg) -> int:
        """Validate and cap the prefetch depth for the configured backend."""
        depth = int(recipe_cfg.get("target_prefetch_depth", 0))
        if depth < 0:
            raise ValueError(f"target_prefetch_depth must be >= 0, got {depth}.")
        if depth == 0:
            return 0
        if self.target_wrapper is None or not self.target_wrapper.supports_async:
            raise ValueError(
                "target_prefetch_depth > 0 requires an async-capable backend (target_model_backend='remote')."
            )
        # One in-flight request per server keeps NCCL recv ordering unambiguous.
        num_servers = getattr(self.target_wrapper, "num_remote_servers", 1)
        capped = min(depth, num_servers)
        if capped != depth and self.dist_env.is_main:
            logger.info("Capping target_prefetch_depth %d -> %d (one in-flight request per server).", depth, capped)
        return capped

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

    def _forward_batch(self, batch, target_batch=None):
        """Run the trainer module for one batch, from the live target or the cache.

        ``target_batch`` may be supplied when the supervision was prefetched
        asynchronously (remote backend); it is already on the training device
        and self-contained, so the raw ``batch`` is not needed.
        """
        if target_batch is not None:
            return self.trainer_module(**target_batch.to_trainer_inputs())
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        # Sequence-packing metadata (present only when packed_sequence_size > 0).
        packing_kwargs = {}
        if "seq_lens" in batch:
            packing_kwargs = {
                "position_ids": batch["position_ids"],
                "seq_lens": batch["seq_lens"],
                "doc_remaining": batch["doc_remaining"],
            }
        if self.target_wrapper is None:
            # Offline cache: the supervision is already in the batch.
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            return self.trainer_module(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                loss_mask=batch["loss_mask"],
                aux_hidden_states=batch["aux_hidden_states"],
                target_probs=batch["target_probs"],
                position_mask=batch["position_mask"],
                **packing_kwargs,
            )
        target_batch = self.target_wrapper.generate_batch(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            loss_mask=batch["loss_mask"],
            **packing_kwargs,
        )
        return self.trainer_module(**target_batch.to_trainer_inputs())

    def _prefetched_batches(self, dataloader):
        """Yield ``(batch, target_batch)`` keeping up to ``target_prefetch_depth``
        remote target requests in flight, so target inference on the server(s)
        overlaps draft training on this GPU.

        Requests are dispatched round-robin across servers by the backend; the
        depth is capped to the server count (see ``setup``) so each server has
        at most one in-flight request -- required for NCCL recv ordering.
        """
        depth = self.target_prefetch_depth
        it = iter(dataloader)
        queue: deque = deque()
        exhausted = False

        def fill():
            nonlocal exhausted
            while not exhausted and len(queue) < depth:
                try:
                    batch = next(it)
                except StopIteration:
                    exhausted = True
                    return
                handle = self.target_wrapper.generate_batch_async(
                    batch["input_ids"], batch["attention_mask"], batch["loss_mask"]
                )
                queue.append((batch, handle))

        fill()
        while queue:
            batch, handle = queue.popleft()
            target_batch = handle.result()
            # Refill only after the popped request completed: round-robin makes
            # its server the next dispatch target, so refilling before the
            # result would put a second request in flight on a busy server and
            # break the one-in-flight-per-server invariant that NCCL recv
            # ordering and the server's hook-based aux capture rely on.
            fill()
            yield batch, target_batch

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
        # Under CP, several global ranks share one dp index (and identical draft
        # weights), so shard checkpoints by dp position, not global rank. With
        # cp_size==1 dp_mesh.get_local_rank() == global rank -> unchanged / resume
        # compatible with pre-CP checkpoints.
        dp_mesh = getattr(self, "dp_mesh", None)
        dp_rank = dp_mesh.get_local_rank() if dp_mesh is not None else (dist.get_rank() if dist.is_initialized() else 0)
        self.checkpointer = self.checkpoint_config.build(
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
        is_final_checkpoint: bool = False,
    ) -> None:
        """Persist draft model, optimizer, scheduler, RNG, and EAGLE-3 meta.

        Overrides ``BaseRecipe.save_checkpoint`` because EAGLE recipes hold multiple
        ``nn.Module`` attributes (frozen target, target wrapper, trainer module wrapping
        the draft) — only ``draft_model`` should be persisted as the main model. The
        EAGLE-3 vocab mapping tensors ride along through ``_save_extra_state``.

        ``is_final_checkpoint`` is computed by the caller (this hand-rolled loop
        has no ``step_scheduler`` for the checkpointer to infer it from);
        ``save_consolidated: final`` exports HF safetensors only when it is True.
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

    def _log_saved_checkpoint(self, kind: str, epoch: int, step: int) -> None:
        """Log a saved checkpoint on rank 0 when checkpointing is enabled."""
        ckpt_cfg = getattr(self, "checkpoint_config", None)
        if self.dist_env.is_main and ckpt_cfg is not None and ckpt_cfg.enabled:
            logger.info("Saved %s checkpoint to %s/epoch_%d_step_%d", kind, ckpt_cfg.checkpoint_dir, epoch, step)

    def _maybe_save_step_checkpoint(self, epoch: int) -> bool:
        """Save a checkpoint mid-epoch when ``ckpt_every_steps`` is configured.

        Called after every optimizer step. Saves whenever ``ckpt_every_steps`` is
        a positive integer and the current ``global_step`` is a multiple of it.
        Returns True if a checkpoint was written. The checkpoint directory is
        named ``epoch_{epoch}_step_{global_step}`` so it never collides with the
        end-of-epoch checkpoint (which uses ``epoch + 1``).
        """
        every = getattr(self, "ckpt_every_steps", None)
        if every is None or every <= 0 or self.runtime.global_step % every != 0:
            return False
        total_optim_steps = getattr(self, "total_optim_steps", None)
        is_final_checkpoint = total_optim_steps is not None and self.runtime.global_step >= total_optim_steps
        self.save_checkpoint(
            epoch=epoch,
            step=self.runtime.global_step,
            train_loss=None,
            val_loss=None,
            best_metric_key="val_loss",
            is_final_checkpoint=is_final_checkpoint,
        )
        self._log_saved_checkpoint("step", epoch, self.runtime.global_step)
        return True

    def _maybe_save_final_checkpoint(self, completed_epochs: int) -> bool:
        """Always save the fully-trained model at the end of a completed run,
        unless a periodic checkpoint already captured the final step.

        The end-of-run state is otherwise easy to lose: with no cadence nothing is
        saved at all, and with a pure step cadence the final step is skipped
        whenever the total step count is not a multiple of ``ckpt_every_steps``.
        This is a no-op only when a step or epoch checkpoint already landed on the
        final step, so it never duplicates or collides with one.
        """
        gs = self.runtime.global_step
        if gs <= 0:
            return False
        every = getattr(self, "ckpt_every_steps", None)
        saved_by_step = bool(every and every > 0 and gs % every == 0)
        saved_by_epoch = bool(getattr(self, "save_checkpoint_every_epoch", False))
        if saved_by_step or saved_by_epoch:
            return False
        self.save_checkpoint(
            epoch=completed_epochs,
            step=gs,
            train_loss=None,
            val_loss=None,
            best_metric_key="val_loss",
            is_final_checkpoint=True,
        )
        self._log_saved_checkpoint("final", completed_epochs, gs)
        return True

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
                # set_vocab_mapping was already called at setup() with the
                # freshly-scanned mapping, but resume can restore a different one
                # (the checkpoint's, e.g. when the data / split / cache changed).
                # Re-apply it so the draft's d2t/t2d tables and -- the actual bug
                # -- the remote target server (which projects to the draft vocab
                # itself) match the checkpoint, not the setup-time scan. Colocated
                # set_vocab_mapping is a no-op; the cached path has no target.
                self.draft_model.set_vocab_mapping(module.selected_token_ids)
                if getattr(self, "target_wrapper", None) is not None:
                    self.target_wrapper.set_vocab_mapping(module.selected_token_ids, module.selected_token_mask)

    def _run_eval(self):
        if self.val_dataloader is None:
            return None
        self.trainer_module.eval()
        total_loss = torch.zeros((), device=self.device)
        total_acc = torch.zeros((), device=self.device)
        total_batches = torch.zeros((), device=self.device)
        total_step_correct: torch.Tensor | None = None
        total_step_valid: torch.Tensor | None = None
        with torch.no_grad():
            for batch in self.val_dataloader:
                metrics = self._forward_batch(batch)
                total_loss = total_loss + metrics.loss.detach()
                total_acc = total_acc + metrics.accuracy.detach()
                total_batches = total_batches + 1
                step_correct = getattr(metrics, "step_correct", None)
                if step_correct is not None:
                    if total_step_correct is None:
                        total_step_correct = torch.zeros_like(step_correct, dtype=torch.float32)
                        total_step_valid = torch.zeros_like(total_step_correct)
                    total_step_correct += step_correct.float()
                    total_step_valid += metrics.step_valid.float()
        total_loss = _all_reduce_mean(total_loss)
        total_acc = _all_reduce_mean(total_acc)
        total_batches = _all_reduce_mean(total_batches)
        tau_sim = _window_tau_sim(total_step_correct, total_step_valid)
        self.trainer_module.train()
        return (
            total_loss / total_batches.clamp_min(1.0),
            total_acc / total_batches.clamp_min(1.0),
            tau_sim,
        )

    def _wandb_log(self, data: dict, step: int) -> None:
        """Log a metrics dict to W&B when a run is active (rank 0)."""
        run = getattr(self, "wandb_run", None)
        if run is not None:
            run.log(data, step=step)

    def run_train_validation_loop(self):
        """Run the minimal EAGLE-3 train loop."""
        self.trainer_module.train()
        try:
            batches_per_epoch = len(self.train_dataloader)
        except TypeError:
            batches_per_epoch = None
        is_ddp = isinstance(self.trainer_module, DistributedDataParallel)
        start_epoch = max(0, int(getattr(self, "_resume_epoch", 0)))
        if start_epoch >= self.num_epochs:
            if self.dist_env.is_main:
                logger.info("All %d epochs already completed; nothing to do.", self.num_epochs)
            self._finalize_training()
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

        pbar = self._make_progress_bar(total=self.total_optim_steps, initial=self.runtime.global_step)
        try:
            self._train_epochs(start_epoch, batches_per_epoch, is_ddp, pbar)
            self._maybe_save_final_checkpoint(self.num_epochs)
            self._finalize_pending_checkpoint()
            if self.dist_env.is_main:
                logger.info("Training complete: global_step=%s", self.runtime.global_step)
        finally:
            if pbar is not None:
                pbar.close()
            self._finalize_training()

    def _finalize_training(self) -> None:
        """Release training resources on any exit path (normal, early-return, or
        exception). Best-effort: each step is guarded so a failure in one does not
        block the others.

        The high-value step is disconnecting the remote target. Without it a
        mid-training crash leaves the long-lived target server with a stale
        client-idle state and a half-open NCCL transport, so the next run cannot
        connect. ``close()`` is a no-op for the co-located backend. The process
        group is intentionally left alone -- it is a framework-global resource
        that direct callers (tests, the interactive launcher) reuse after the
        loop returns, and ``initialize_distributed`` already destroys it at
        process exit.
        """
        if getattr(self, "target_wrapper", None) is not None:
            _best_effort("closing target backend", self.target_wrapper.close)
        if getattr(self, "wandb_run", None) is not None:
            _best_effort("finishing W&B run", self.wandb_run.finish)

    def _train_epochs(self, start_epoch, batches_per_epoch, is_ddp, pbar=None):
        """Run the epoch loop (extracted so :meth:`run_train_validation_loop` can
        wrap it in ``try/finally`` and guarantee teardown on any exit path)."""
        for epoch in range(start_epoch, self.num_epochs):
            if hasattr(self.train_dataloader.sampler, "set_epoch"):
                self.train_dataloader.sampler.set_epoch(epoch)

            running_loss = torch.zeros((), device=self.device)
            running_acc = torch.zeros((), device=self.device)
            # Per-TTT-step hit/valid counts for the simulated accept length;
            # allocated lazily on the first batch that reports them (P-EAGLE
            # metrics do not) so ttt_steps never has to be threaded here.
            running_step_correct: torch.Tensor | None = None
            running_step_valid: torch.Tensor | None = None
            running_steps = 0
            self.optimizer.zero_grad(set_to_none=True)

            batches_processed = 0
            pending_micro_batches = 0
            # With prefetch the supervision is fetched asynchronously and paired
            # with its batch; otherwise the target runs inline (target_batch=None).
            if self.target_prefetch_depth > 0:
                batch_source = enumerate(self._prefetched_batches(self.train_dataloader))
            else:
                batch_source = ((i, (b, None)) for i, b in enumerate(self.train_dataloader))
            for batch_idx, (batch, target_batch) in batch_source:
                if self._peagle_partitioned:
                    # P-EAGLE sequence partitioning owns its per-segment backward
                    # and its own no_sync (the all-reduce fires on the last
                    # segment), so it runs outside the grad-accum sync context.
                    metrics = self._peagle_partitioned_step(batch)
                else:
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
                        metrics = self._forward_batch(batch, target_batch)
                        loss = metrics.loss / float(self.grad_accumulation_steps)
                        loss.backward()

                running_loss = running_loss + metrics.loss.detach()
                running_acc = running_acc + metrics.accuracy.detach()
                step_correct = getattr(metrics, "step_correct", None)
                if step_correct is not None:
                    if running_step_correct is None:
                        running_step_correct = torch.zeros_like(step_correct, dtype=torch.float32)
                        running_step_valid = torch.zeros_like(running_step_correct)
                    running_step_correct += step_correct.float()
                    running_step_valid += metrics.step_valid.float()
                running_steps += 1
                batches_processed = batch_idx + 1
                pending_micro_batches += 1

                if pending_micro_batches == self.grad_accumulation_steps:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.runtime.global_step += 1
                    if pbar is not None:
                        pbar.update(1)
                    pending_micro_batches = 0
                    self._maybe_save_step_checkpoint(epoch)

                    if self.runtime.global_step % self.log_every_steps == 0:
                        mean_loss = _all_reduce_mean(running_loss / max(running_steps, 1))
                        mean_acc = _all_reduce_mean(running_acc / max(running_steps, 1))
                        tau_sim = _window_tau_sim(running_step_correct, running_step_valid)
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        if self.dist_env.is_main:
                            if pbar is not None:
                                postfix = {
                                    "loss": f"{mean_loss.item():.4f}",
                                    "acc": f"{mean_acc.item():.4f}",
                                    "lr": f"{current_lr:.2e}",
                                }
                                if tau_sim is not None:
                                    postfix["tau"] = f"{tau_sim:.2f}"
                                pbar.set_postfix(**postfix)
                            logger.info(
                                "epoch=%s step=%s train_loss=%.6f train_acc=%.6f%s lr=%.3e",
                                epoch,
                                self.runtime.global_step,
                                mean_loss.item(),
                                mean_acc.item(),
                                "" if tau_sim is None else f" train_tau_sim={tau_sim:.4f}",
                                current_lr,
                            )
                            wandb_data = {
                                "train/loss": mean_loss.item(),
                                "train/accuracy": mean_acc.item(),
                                "train/lr": current_lr,
                                "train/grad_norm": float(grad_norm),
                                "train/epoch": epoch,
                            }
                            if tau_sim is not None:
                                wandb_data["train/tau_sim"] = tau_sim
                            self._wandb_log(wandb_data, step=self.runtime.global_step)
                        running_loss.zero_()
                        running_acc.zero_()
                        if running_step_correct is not None:
                            running_step_correct.zero_()
                            running_step_valid.zero_()
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
                grad_norm = torch.nn.utils.clip_grad_norm_(self.trainer_module.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.runtime.global_step += 1
                if pbar is not None:
                    pbar.update(1)
                pending_micro_batches = 0
                self._maybe_save_step_checkpoint(epoch)

                if running_steps > 0:
                    mean_loss = _all_reduce_mean(running_loss / max(running_steps, 1))
                    mean_acc = _all_reduce_mean(running_acc / max(running_steps, 1))
                    tau_sim = _window_tau_sim(running_step_correct, running_step_valid)
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    if self.dist_env.is_main:
                        logger.info(
                            "epoch=%s step=%s train_loss=%.6f train_acc=%.6f%s lr=%.3e (trailing flush)",
                            epoch,
                            self.runtime.global_step,
                            mean_loss.item(),
                            mean_acc.item(),
                            "" if tau_sim is None else f" train_tau_sim={tau_sim:.4f}",
                            current_lr,
                        )
                        wandb_data = {
                            "train/loss": mean_loss.item(),
                            "train/accuracy": mean_acc.item(),
                            "train/lr": current_lr,
                            "train/grad_norm": float(grad_norm),
                            "train/epoch": epoch,
                        }
                        if tau_sim is not None:
                            wandb_data["train/tau_sim"] = tau_sim
                        self._wandb_log(wandb_data, step=self.runtime.global_step)
                    running_loss.zero_()
                    running_acc.zero_()
                    if running_step_correct is not None:
                        running_step_correct.zero_()
                        running_step_valid.zero_()
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
                val_loss, val_acc, val_tau_sim = eval_metrics
                val_loss_dict = {
                    "val_loss": val_loss.item(),
                    "val_accuracy": val_acc.item(),
                }
                if self.dist_env.is_main:
                    logger.info(
                        "epoch=%s val_loss=%.6f val_acc=%.6f%s",
                        epoch,
                        val_loss_dict["val_loss"],
                        val_loss_dict["val_accuracy"],
                        "" if val_tau_sim is None else f" val_tau_sim={val_tau_sim:.4f}",
                    )
                    wandb_data = {"val/loss": val_loss_dict["val_loss"], "val/accuracy": val_loss_dict["val_accuracy"]}
                    if val_tau_sim is not None:
                        wandb_data["val/tau_sim"] = val_tau_sim
                    self._wandb_log(wandb_data, step=self.runtime.global_step)
            if getattr(self, "save_checkpoint_every_epoch", False):
                self.save_checkpoint(
                    epoch=epoch + 1,
                    step=self.runtime.global_step,
                    train_loss=None,
                    val_loss=val_loss_dict,
                    best_metric_key="val_loss",
                    is_final_checkpoint=epoch + 1 >= self.num_epochs,
                )
                self._log_saved_checkpoint("epoch", epoch + 1, self.runtime.global_step)


def main(config_path=None):
    """Main entry point for the EAGLE-3 recipe."""
    cfg = parse_args_and_load_config(config_path)
    trainer = TrainEagle3Recipe(cfg)
    trainer.setup()
    trainer.run_train_validation_loop()


if __name__ == "__main__":
    main()
