# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""``Engine`` — the Automodel training surface.

One concrete class, one file. State lives as direct attributes
(``self.model``, ``self.optimizer``, ``self.mesh``); methods are short and
expose the orchestration they perform. There is no ABC, no registry, no
``ModelHandle`` wrapper.

Subclass to swap a method. Fork the file to swap the whole thing.

Tier conventions (informal — duck-typed):
  - SFT/PEFT-ready backends implement: ``build``, ``forward_backward``,
    ``zero_grad``, ``optimizer_step``, ``lr_scheduler_step``, ``train_mode``,
    ``eval_mode``, ``save_checkpoint``, ``load_checkpoint``.
  - RL-ready backends additionally implement ``export_weights``.
  - RL-best backends additionally implement ``to`` for train↔rollout offload.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, nullcontext as _nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from nemo_automodel.components.checkpoint.api import (
    export_weights as _export_weights,
    load_checkpoint as _load_checkpoint,
    save_checkpoint as _save_checkpoint,
)
from nemo_automodel.components.checkpoint.checkpointing import CheckpointingConfig
from nemo_automodel.components.distributed.build import init_distributed_and_build_mesh
from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.device import offload as _offload, onload as _onload
from nemo_automodel.components.distributed.mesh import MeshAxisName, MeshContext
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
from nemo_automodel.components.training.batch_split import split_into_microbatches
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.training.utils import (
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    """Read a key from a dict, ConfigNode, or dataclass."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _cfg_to_dict(cfg: Any) -> dict:
    """Best-effort dict coercion."""
    import dataclasses

    if cfg is None:
        return {}
    if isinstance(cfg, Mapping):
        return dict(cfg)
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if dataclasses.is_dataclass(cfg):
        return dataclasses.asdict(cfg)
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def _callable_and_kwargs(cfg: Any) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve ``_target_``-style configs into ``(factory, kwargs)``.

    Mirrors ``recipes._component_builders._callable_and_kwargs`` but kept
    inline so engine.py doesn't import from recipes/.
    """
    if hasattr(cfg, "to_dict") or isinstance(cfg, Mapping):
        d = _cfg_to_dict(cfg)
        target = d.pop("_target_", None)
        if target is not None:
            return target, d
    target = getattr(cfg, "_target_", None)
    if target is not None:
        return target, {}
    if callable(cfg):
        return cfg, {}
    if hasattr(cfg, "instantiate"):
        return cfg.instantiate, {}
    raise AttributeError("Config must provide _target_, be callable, or provide instantiate()")


class Engine:
    """Training engine. One model, one optimizer, one lifecycle.

    Usage::

        engine = Engine(Engine.Config(
            model=cfg.model,
            distributed=cfg.distributed,       # dict OR a pre-built MeshContext
            optimizer=cfg.optimizer,
            lr_scheduler=cfg.lr_scheduler,
        ))
        engine.build()

        with engine.train_mode():
            engine.zero_grad()
            out = engine.forward_backward(batch, num_microbatches=8)
            ok, grad_norm = engine.optimizer_step()
            lr = engine.lr_scheduler_step()
    """

    # ── Nested Config (TorchTitan / bumblebee pattern) ───────────────

    @dataclass
    class Config:
        """All Engine knobs in one dataclass.

        Field types are intentionally loose (``Any``) so callers can pass
        their existing ConfigNode / dataclass / dict sub-configs without
        forcing a schema change.
        """

        # ── Build inputs (sub-configs) ────────────────────────────────
        model: Any = None                       # arbitrary model config object
        distributed: Any = None                 # MeshContext OR distributed dict
        optimizer: Any = None                   # optimizer config (carries _target_)
        lr_scheduler: Any = None
        dist_env: Any = None                    # {backend, timeout_minutes}
        peft: Any = None
        quantization: Any = None
        fp8: Any = None
        pipeline: Any = None
        moe: Any = None

        # ── Behavior toggles ─────────────────────────────────────────
        activation_checkpointing: bool = False
        max_grad_norm: float = 1.0
        defer_fsdp_grad_sync: bool = True
        seed: int = 0

        # ── CP / THD batch shaping (passed to make_cp_batch_and_ctx) ──
        cp_use_te: bool = False
        cp_padding_token_id: int = 0
        cp_num_chunks: int = 1

        # ── Optional callable hooks ──────────────────────────────────
        # Context-manager factory applied around the forward (e.g. TE FP8 autocast).
        fp8_autocast: Callable[[], Any] | None = None
        # Extra loss term added to the main loss.
        # Signature: extra_loss_fn(out, model, labels, num_label_tokens) -> Tensor | None.
        # LLM training uses this for MTP (Multi-Token Prediction) auxiliary loss.
        extra_loss_fn: Callable[..., torch.Tensor | None] | None = None

    # ── Construction ─────────────────────────────────────────────────

    def __init__(self, config: "Engine.Config"):
        self.config = config

        # Runtime state, populated by build() or injected by a recipe that
        # already built the model/optimizer/mesh itself.
        self.model: nn.Module | Any = None          # nn.Module or AutoPipeline
        self.optimizer: torch.optim.Optimizer | None = None
        self.lr_scheduler: OptimizerParamScheduler | None = None
        self.mesh: MeshContext | None = None

    # ── Convenience accessors for config fields ──────────────────────
    # These wrappers keep call sites readable (``self.max_grad_norm`` vs
    # ``self.config.max_grad_norm``) and let recipes mutate the live values
    # (e.g. updating max_grad_norm per call) without touching the dataclass.

    @property
    def model_cfg(self): return self.config.model
    @property
    def distributed_cfg(self): return self.config.distributed
    @property
    def optimizer_cfg(self): return self.config.optimizer
    @property
    def lr_scheduler_cfg(self): return self.config.lr_scheduler
    @property
    def dist_env_cfg(self): return self.config.dist_env
    @property
    def peft_cfg(self): return self.config.peft
    @property
    def quantization_cfg(self): return self.config.quantization
    @property
    def fp8_cfg(self): return self.config.fp8
    @property
    def pipeline_cfg(self): return self.config.pipeline
    @property
    def moe_cfg(self): return self.config.moe
    @property
    def activation_checkpointing(self): return self.config.activation_checkpointing
    @property
    def defer_fsdp_grad_sync(self): return self.config.defer_fsdp_grad_sync
    @property
    def seed(self): return self.config.seed
    @property
    def max_grad_norm(self): return self.config.max_grad_norm
    @max_grad_norm.setter
    def max_grad_norm(self, value): self.config.max_grad_norm = value

    # CP / THD shaping (proxied to config so recipes can read/write either)
    @property
    def cp_use_te(self): return self.config.cp_use_te
    @cp_use_te.setter
    def cp_use_te(self, v): self.config.cp_use_te = v
    @property
    def cp_padding_token_id(self): return self.config.cp_padding_token_id
    @cp_padding_token_id.setter
    def cp_padding_token_id(self, v): self.config.cp_padding_token_id = v
    @property
    def cp_num_chunks(self): return self.config.cp_num_chunks
    @cp_num_chunks.setter
    def cp_num_chunks(self, v): self.config.cp_num_chunks = v

    # Optional callable hooks
    @property
    def fp8_autocast(self): return self.config.fp8_autocast
    @fp8_autocast.setter
    def fp8_autocast(self, v): self.config.fp8_autocast = v
    @property
    def extra_loss_fn(self): return self.config.extra_loss_fn
    @extra_loss_fn.setter
    def extra_loss_fn(self, v): self.config.extra_loss_fn = v

    # ── Build ────────────────────────────────────────────────────────

    def build(self) -> None:
        """Construct mesh, model, optimizer, scheduler. Read back via attributes."""
        # PR 2190 keeps build_model in the recipe (it's complex; not part of the
        # config-targets-out-of-components refactor). Imported lazily so the
        # heavy recipe chain isn't pulled at module load.
        from nemo_automodel.recipes.llm.train_ft import build_model
        from nemo_automodel.components.optim.build import build_optimizer as _build_optimizer_typed

        # 1. Resolve mesh — accept either a pre-built MeshContext or a dict/cfg.
        if isinstance(self.distributed_cfg, MeshContext):
            self.mesh = self.distributed_cfg
        else:
            _dist_info, self.mesh = init_distributed_and_build_mesh(
                self.distributed_cfg, dist_env_cfg=self.dist_env_cfg,
            )

        # 2. Build model. build_model returns (model_or_pp, list[optimizer]).
        #    The optimizer list is non-empty only when megatron_fsdp's
        #    fully_shard_optimizer path runs; otherwise it's empty.
        result = build_model(
            self.model_cfg,
            self.peft_cfg,
            self.seed,
            cfg_fp8=self.fp8_cfg,
            cfg_quantization=self.quantization_cfg,
            device_mesh=self.mesh.device_mesh,
            moe_mesh=self.mesh.moe_mesh,
            distributed_config=self.mesh.strategy_config,
            pipeline_config=self.mesh.pipeline_config,
            cfg_moe=self.mesh.moe_config or self.moe_cfg,
            activation_checkpointing=self.mesh.activation_checkpointing or self.activation_checkpointing,
        )
        if isinstance(result, tuple) and len(result) == 2:
            self.model, opts_from_build = result
        else:
            self.model, opts_from_build = result, []

        # 3. Optimizer. Prefer the one returned by build_model (megatron_fsdp
        #    path); otherwise build via the typed component builder. We do the
        #    _target_ resolution here so Engine never imports from recipes/_*.
        if opts_from_build:
            self.optimizer = opts_from_build[0] if isinstance(opts_from_build, list) else opts_from_build
        else:
            optimizer_factory, optimizer_kwargs = _callable_and_kwargs(self.optimizer_cfg)
            optimizers = _build_optimizer_typed(
                model=self.model,
                optimizer_factory=optimizer_factory,
                optimizer_kwargs=optimizer_kwargs,
                distributed_config=self.mesh.strategy_config,
                device_mesh=self.mesh.device_mesh,
            )
            self.optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers

        # 4. LR scheduler — construct OptimizerParamScheduler directly, bypassing
        #    the recipe's build_lr_scheduler (which is tied to a StepScheduler).
        self.lr_scheduler = self._build_lr_scheduler(self.optimizer)

    def _build_lr_scheduler(self, optimizer: torch.optim.Optimizer) -> OptimizerParamScheduler | None:
        """Construct an :class:`OptimizerParamScheduler` from ``self.lr_scheduler_cfg``.

        Required keys (read via ``_cfg_get``):
            ``total_steps``  (or ``lr_decay_steps``) — total optimizer steps.
        Optional:
            ``lr_warmup_steps`` (default ``0``)
            ``lr_warmup_steps_ratio`` (default ``0.0``; only used if ``lr_warmup_steps == 0``)
            ``lr_decay_style`` (default ``"cosine"``)
            ``init_lr_ratio`` (default ``0.1``; init_lr = ``base_lr * init_lr_ratio``)
            ``min_lr_ratio`` (default ``0.01``)
            ``wd_incr_style`` (default ``"constant"``)
        """
        cfg = self.lr_scheduler_cfg
        if cfg is None:
            return None

        total_steps = _cfg_get(cfg, "total_steps", _cfg_get(cfg, "lr_decay_steps"))
        if total_steps is None:
            raise ValueError(
                "lr_scheduler_cfg must include 'total_steps' (or 'lr_decay_steps')."
            )

        warmup_steps = _cfg_get(cfg, "lr_warmup_steps", 0) or 0
        if warmup_steps == 0:
            warmup_ratio = _cfg_get(cfg, "lr_warmup_steps_ratio", 0.0) or 0.0
            warmup_steps = int(warmup_ratio * total_steps)

        base_lr = optimizer.param_groups[0]["lr"]
        init_lr_ratio = _cfg_get(cfg, "init_lr_ratio", 0.1)
        min_lr_ratio = _cfg_get(cfg, "min_lr_ratio", 0.01)
        base_wd = optimizer.param_groups[0].get("weight_decay", 0.0)

        return OptimizerParamScheduler(
            optimizer=optimizer,
            init_lr=base_lr * init_lr_ratio,
            max_lr=base_lr,
            min_lr=base_lr * min_lr_ratio,
            lr_warmup_steps=int(warmup_steps),
            lr_decay_steps=int(total_steps),
            lr_decay_style=_cfg_get(cfg, "lr_decay_style", "cosine"),
            start_wd=base_wd,
            end_wd=base_wd,
            wd_incr_steps=int(total_steps),
            wd_incr_style=_cfg_get(cfg, "wd_incr_style", "constant"),
        )

    # ── Introspection (no wrapper class needed) ──────────────────────

    @property
    def parts(self) -> list[nn.Module]:
        """PP-unified parts. Always a list."""
        if self.model is None:
            return []
        if hasattr(self.model, "parts"):
            return list(self.model.parts)
        return [self.model]

    @property
    def pp_enabled(self) -> bool:
        return self.model is not None and hasattr(self.model, "parts")

    @property
    def device(self) -> torch.device:
        for p in self.parts:
            for param in p.parameters():
                return param.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def dp_rank(self) -> int:
        if self.mesh is None or self.mesh.device_mesh is None:
            return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        dm = self.mesh.device_mesh
        if MeshAxisName.CP in dm.mesh_dim_names and dm[MeshAxisName.CP].size() > 1:
            return get_flat_mesh(dm, "dp_cp").get_local_rank()
        return get_flat_mesh(dm, "dp").get_local_rank()

    @property
    def dp_size(self) -> int:
        if self.mesh is None or self.mesh.device_mesh is None:
            return torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        dm = self.mesh.device_mesh
        if MeshAxisName.CP in dm.mesh_dim_names and dm[MeshAxisName.CP].size() > 1:
            return get_flat_mesh(dm, "dp_cp").size()
        return get_flat_mesh(dm, "dp").size()

    @property
    def dp_group(self):
        if self.mesh is None or self.mesh.device_mesh is None:
            return None
        dm = self.mesh.device_mesh
        if MeshAxisName.CP in dm.mesh_dim_names and dm[MeshAxisName.CP].size() > 1:
            return get_flat_mesh(dm, "dp_cp").get_group()
        return get_flat_mesh(dm, "dp").get_group()

    # ── Forward-backward (everything inline — no helper file) ────────

    def forward_backward(
        self,
        batch,
        loss_fn: Callable | None = None,
        *,
        num_microbatches: int = 1,
        forward_only: bool = False,
        num_label_tokens: int | None = None,
    ) -> dict:
        """Run forward + (optional) backward over microbatches.

        ``batch`` may be a single dict (split internally into ``num_microbatches``
        slices along dim 0) or a list of dicts (used directly as pre-split
        microbatches — ``num_microbatches`` is ignored).

        Everything is inline here. The orchestration is visible top-to-bottom:
        gradient-accumulation prep, the MoE aux-loss scaler mutation, the
        microbatch loop, the per-microbatch CP/PP/loss/backward, and the
        final-backward hook. Fork this method to swap behavior.

        Returns ``{"loss": tensor, "metrics": {"loss": float}}``.
        """
        is_train = not forward_only
        device = self.device
        device_mesh = self.mesh.device_mesh if self.mesh is not None else None
        pp = self.model if self.pp_enabled else None
        is_fused_ce = isinstance(loss_fn, FusedLinearCrossEntropy)

        # Accept either a dict (split internally) or a pre-split list of dicts.
        if isinstance(batch, list):
            microbatches = batch
            num_microbatches = len(microbatches)
        else:
            microbatches = split_into_microbatches(batch, num_microbatches)

        # ── Pre-loop orchestration ──────────────────────────────────
        if is_train:
            prepare_for_grad_accumulation(self.parts, pp_enabled=self.pp_enabled)
            # MoE aux loss is injected via MoEAuxLossAutoScaler during backward.
            # Set the scale to cancel FSDP/PP grad scaling.
            if self.mesh is not None and self.mesh.moe_config is not None:
                if self.pp_enabled and num_label_tokens is not None:
                    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(num_label_tokens))
                else:
                    MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(float(self.dp_size))

        losses: list[torch.Tensor] = []

        # ── Microbatch loop ────────────────────────────────────────
        for i, mb in enumerate(microbatches):
            is_last = i == num_microbatches - 1
            if is_train and i == 1:
                prepare_after_first_microbatch()
            if is_train and is_last:
                prepare_for_final_backward(self.parts, pp_enabled=self.pp_enabled)

            # Move batch to device (tensors and dict-of-tensors values both).
            mb = {
                k: (
                    v.to(device, non_blocking=True) if isinstance(v, torch.Tensor)
                    else (
                        {dk: (dv.to(device, non_blocking=True) if isinstance(dv, torch.Tensor) else dv)
                         for dk, dv in v.items() if dv is not None}
                        if isinstance(v, dict) else v
                    )
                )
                for k, v in mb.items()
            }

            # Subclass hook: pre-CP preparation (e.g. VLM multimodal pre-embed).
            # Default base implementation is a no-op.
            mb = self._pre_cp_hook(mb)

            # CP / THD reshaping. `train_ctx()` is the CP attention context.
            train_ctx, mb = make_cp_batch_and_ctx(
                device_mesh, mb,
                use_te=self.cp_use_te,
                padding_token_id=self.cp_padding_token_id,
                num_chunks=self.cp_num_chunks,
            )
            # If loss_fn is provided we pop labels (loss computed externally).
            # Otherwise leave labels in mb so the model's forward computes loss.
            labels = mb.pop("labels") if loss_fn is not None else mb.get("labels")

            # ── Pipeline-parallel branch ────────────────────────────
            if pp is not None:
                fp8_ctx = self.fp8_autocast() if self.fp8_autocast is not None else _nullcontext()
                with train_ctx(), fp8_ctx:
                    pp_losses = [] if pp.info.has_last_stage else None
                    targets = labels.clone() if pp.info.has_last_stage else None
                    input_ids = mb.pop("input_ids")
                    pp.update_seq_len(input_ids.shape[1])
                    # Subclass hook: pre-PP-schedule preparation (e.g. VLM
                    # media-tensor chunking). Default base implementation is
                    # a no-op.
                    mb = self._pre_pp_schedule_hook(mb, pp=pp, input_ids=input_ids)
                    pp_batch = {
                        k: v for k, v in mb.items()
                        if v is not None and not (isinstance(v, dict) and len(v) == 0)
                    }
                    schedule_fn = pp.info.schedule.step if is_train else pp.info.schedule.eval
                    if pp.info.has_first_stage:
                        schedule_fn(input_ids, target=targets, losses=pp_losses, **pp_batch)
                    else:
                        schedule_fn(target=targets, losses=pp_losses, **pp_batch)
                if pp.info.has_last_stage:
                    losses.append(torch.sum(torch.stack(pp_losses)).detach())
                else:
                    losses.append(torch.tensor(0.0, device=device))
                continue

            # ── Non-PP branch ───────────────────────────────────────
            model = self.parts[0]
            sync_ctx = (
                get_sync_ctx(model, is_last, defer_fsdp_grad_sync=self.defer_fsdp_grad_sync)
                if is_train else _nullcontext()
            )
            fp8_ctx = self.fp8_autocast() if self.fp8_autocast is not None else _nullcontext()
            with train_ctx(), sync_ctx, fp8_ctx:
                fwd_kwargs = filter_forward_kwargs(model, mb)
                if is_fused_ce:
                    out = model(logits_to_keep=1, **fwd_kwargs)
                    hidden_states = get_final_hidden_states(out)
                    if hidden_states is None:
                        raise ValueError(
                            "FusedLinearCrossEntropy requires hidden states. "
                            "Set `model.output_hidden_states=True` in the config."
                        )
                else:
                    out = model(**fwd_kwargs)
                    hidden_states = get_final_hidden_states(out)

                if loss_fn is None:
                    # Standard SFT path — model.forward computed loss from labels.
                    loss = getattr(out, "loss", None)
                    if loss is None:
                        raise ValueError(
                            "loss_fn is None but the model output has no .loss attribute. "
                            "Either pass a loss_fn or have the model compute loss internally."
                        )
                elif is_fused_ce:
                    lm_weight = self._resolve_lm_head_weight(model)
                    loss = loss_fn(
                        hidden_states=hidden_states,
                        labels=labels,
                        lm_weight=lm_weight,
                        num_label_tokens=num_label_tokens,
                    )
                else:
                    loss = loss_fn(
                        logits=getattr(out, "logits", out),
                        labels=labels,
                        num_label_tokens=num_label_tokens,
                    )

                # Optional extra loss term (e.g. MTP for DeepSeek-V3 models).
                if self.extra_loss_fn is not None:
                    extra = self.extra_loss_fn(
                        out=out, model=model, labels=labels, num_label_tokens=num_label_tokens,
                    )
                    if extra is not None:
                        loss = loss + extra

                if is_train:
                    # Multiply by dp_group_size to cancel FSDP's grad averaging;
                    # we want a sum-of-grads across DP, not a mean.
                    (loss * self.dp_size).backward()

                losses.append(loss.detach())

        loss_tensor = torch.stack(losses).mean() if losses else torch.tensor(0.0, device=device)
        return {
            "loss": loss_tensor,
            "losses": losses,                       # per-microbatch detached losses
            "metrics": {"loss": float(loss_tensor)},
        }

    # ── Subclass hooks (no-op by default) ────────────────────────────

    def _pre_cp_hook(self, mb: dict) -> dict:
        """Per-microbatch hook invoked AFTER device move, BEFORE CP shaping.

        Default: identity. Subclasses (e.g. VLMEngine) override to insert
        modality-specific preparation such as multimodal pre-embedding.
        """
        return mb

    def _pre_pp_schedule_hook(self, mb: dict, *, pp: Any, input_ids: torch.Tensor) -> dict:
        """Per-microbatch hook invoked inside the PP branch, BEFORE pp.schedule.step.

        Default: identity. Subclasses (e.g. VLMEngine) override to pre-chunk
        non-standard tensors (pixel_values, image_grid) along the PP dimension
        and stash them on the stage0 model.
        """
        return mb

    @staticmethod
    def _resolve_lm_head_weight(model: nn.Module) -> torch.Tensor:
        """Find the LM-head weight tensor — required by FusedLinearCrossEntropy."""
        if hasattr(model, "get_output_embeddings"):
            emb = model.get_output_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                w = emb.weight
                return w.full_tensor() if hasattr(w, "full_tensor") else w
        for name, param in model.named_parameters(remove_duplicate=False):
            if "lm_head" in name and name.endswith(".weight"):
                return param.full_tensor() if hasattr(param, "full_tensor") else param
        raise ValueError("lm_head.weight not found in model")

    # ── Gradient step (clip visible inline) ──────────────────────────

    def zero_grad(self) -> None:
        if self.optimizer is not None:
            self.optimizer.zero_grad()

    def optimizer_step(self, num_label_tokens: int | None = None) -> tuple[bool, float]:
        """Scale + clip grads and step. Returns ``(update_succeeded, grad_norm)``.

        ``num_label_tokens`` is required only when PP is enabled — used to scale
        gradients by num_label_tokens / dp_group_size before clipping. Non-PP
        callers can leave it at ``None``.
        """
        if self.optimizer is None:
            raise RuntimeError("optimizer is None; did you call build()?")

        moe_mesh = self.mesh.moe_mesh if self.mesh is not None else None
        device_mesh = self.mesh.device_mesh if self.mesh is not None else None

        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=self.max_grad_norm,
            model_parts=self.parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=device_mesh,
            moe_mesh=moe_mesh,
            ep_axis_name="ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            num_label_tokens=num_label_tokens,
            dp_group_size=self.dp_size,
        )

        grad_norm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        ok = bool(torch.isfinite(torch.tensor(grad_norm_val)))
        if ok:
            self.optimizer.step()
        else:
            self.optimizer.zero_grad()
        return ok, grad_norm_val

    def lr_scheduler_step(self) -> float | list[float]:
        """Advance LR scheduler. Returns the new learning rate(s)."""
        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step(increment=1)
            except TypeError:
                self.lr_scheduler.step()
        lrs = [g["lr"] for g in self.optimizer.param_groups] if self.optimizer is not None else []
        return lrs[0] if len(lrs) == 1 else lrs

    # ── Mode switching ───────────────────────────────────────────────

    @contextmanager
    def train_mode(self):
        prev = [p.training for p in self.parts]
        for p in self.parts:
            p.train()
        try:
            yield
        finally:
            for p, was in zip(self.parts, prev):
                p.train(was)

    @contextmanager
    def eval_mode(self):
        prev = [p.training for p in self.parts]
        for p in self.parts:
            p.eval()
        try:
            yield
        finally:
            for p, was in zip(self.parts, prev):
                p.train(was)

    # ── Checkpoint ───────────────────────────────────────────────────

    def save_checkpoint(self, path: str, *, ckpt_cfg: CheckpointingConfig, **kw) -> None:
        _save_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            mesh=self.mesh, path=path, config=ckpt_cfg, **kw,
        )

    def load_checkpoint(self, path: str, *, ckpt_cfg: CheckpointingConfig) -> None:
        _load_checkpoint(
            self.model, self.optimizer, self.lr_scheduler,
            mesh=self.mesh, path=path, config=ckpt_cfg,
        )

    # ── Weight export (RL-ready tier) ────────────────────────────────

    def export_weights(self, *, to_hf: bool = True) -> Iterator[tuple[str, torch.Tensor]]:
        """Iterate ``(name, tensor)`` over model parameters for refit / eval."""
        yield from _export_weights(self.model, to_hf=to_hf, mesh=self.mesh)

    # ── Device move (RL-best tier) ───────────────────────────────────

    def to(
        self,
        device: str,
        *,
        model: bool = True,
        optimizer: bool = True,
        grad: bool = True,
    ) -> None:
        """Move model / optimizer / grad to ``device`` (``"cpu"`` or ``"cuda"``)."""
        if device == "cpu":
            _offload(
                self.model, self.optimizer,
                model_to_cpu=model, optimizer_to_cpu=optimizer, drop_grad=grad,
            )
        else:
            _onload(
                self.model, self.optimizer, device,
                model_to_device=model, optimizer_to_device=optimizer,
            )


__all__ = ["Engine"]
