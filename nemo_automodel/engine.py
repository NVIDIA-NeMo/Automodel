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

"""``Engine``: one model, one optimizer, one training lifecycle.

A small, readable orchestration surface over Automodel's existing primitives.
It is the in-process home for the tinker-like training API — ``forward`` /
``forward_backward`` over a :class:`Datum` contract returning a
:class:`ModelOutput`, plus ``optimizer_step`` / ``optim_step`` / device movement
/ mode switching — without the caller hand-wiring the microbatch lifecycle,
MoE aux-loss scaling, CP/THD batch shaping, and gradient clipping.

Design choices (see ``automodel_tinker_api_plan.md``):

* **Top-level module, not under ``components/``.** ``Engine`` orchestrates many
  components and the recipe builders, so it intentionally sits outside the
  "components must not import each other" independence contract.
* **Reuse, don't reimplement.** Construction calls the same builders the LLM
  recipe uses; the run loop calls the same ``prepare_*`` / ``make_cp_batch_and_ctx``
  / ``calculate_loss`` / ``scale_grads_and_clip_grad_norm`` helpers.
* **Lists, like the recipe.** ``model_parts`` / ``optimizers`` / ``lr_schedulers``
  are lists, matching what the builders return (PP parts, per-part optimizers).
* **No LR scheduler required.** The tinker UX drives LR per step via
  ``optim_step(lr=...)``; a scheduler is optional and injected by recipes that
  want scheduler-driven LR.
* **Two construction paths.** Build from an :class:`Engine.Config` (the standard
  FSDP2/DDP path), or inject already-built ``model_parts`` / ``optimizers`` (used
  by tests and by recipes/RL frameworks that build components themselves).

Pipeline parallelism is not yet wired into ``forward`` / ``forward_backward``
(the PP schedule path is a documented follow-up); ``pp_enabled`` engines raise a
clear error rather than silently running the wrong thing.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from nemo_automodel.components.datasets.datum import Datum, collate_datums
from nemo_automodel.components.training.model_output import (
    ModelOutput,
    compute_entropy,
    selected_token_logprobs,
)
from nemo_automodel.loss_fns import BUILTIN_LOSSES, LossFn

__all__ = ["Engine"]


class Engine:
    """Training engine. One model, one optimizer, one lifecycle.

    Subclass to swap a method; or build your own class with the same method
    names — duck-typing is the contract, there is no ABC to satisfy.
    """

    # ── Config ───────────────────────────────────────────────────────

    @dataclass
    class Config:
        """Engine construction inputs.

        Sub-config fields are typed loose (``Any``) so recipes can pass their
        existing ConfigNode sub-configs straight through. Leave ``model`` as
        ``None`` to construct an Engine shell and inject ``model_parts`` /
        ``optimizers`` manually (tests, custom build paths).
        """

        model: Any = None  # ConfigNode with _target_ (or None to inject)
        distributed: Any = None  # DistributedSetup, a distributed cfg, or None (single device)
        optimizer: Any = None  # ConfigNode with .build(model, device_mesh=, is_peft=)
        peft: Any = None  # already-instantiated PEFT config
        seed: int = 42
        max_grad_norm: float = 1.0
        defer_fsdp_grad_sync: bool = True
        has_packed_sequence: bool = False
        fp8: Any = None
        compile: Any = None
        quantization: Any = None
        qat: Any = None
        sdpa_method: Any = None

        # CP / THD batch shaping (forwarded to make_cp_batch_and_ctx).
        cp_use_te: bool = False
        cp_padding_token_id: int = 0
        cp_num_chunks: int = 1

        # Default loss for the SFT path when forward_backward is called without
        # an explicit loss_fn (e.g. MaskedCrossEntropy / FusedLinearCrossEntropy).
        loss_fn: Any = None

    # ── Construction ─────────────────────────────────────────────────

    def __init__(
        self,
        config: "Engine.Config | None" = None,
        *,
        model_parts: list[nn.Module] | None = None,
        optimizers: list[torch.optim.Optimizer] | None = None,
        lr_schedulers: list[Any] | None = None,
        distributed_setup: Any = None,
    ):
        self.config = config or Engine.Config()

        # State — either constructed from config.model or injected.
        self.model_parts: list[nn.Module] = list(model_parts) if model_parts is not None else []
        self.optimizers: list[torch.optim.Optimizer] = list(optimizers) if optimizers is not None else []
        self.lr_schedulers: list[Any] = list(lr_schedulers) if lr_schedulers is not None else []
        self.pp: Any = None  # AutoPipeline when PP is enabled, else None

        # Distributed handles.
        self.distributed_setup = distributed_setup
        self.mesh_context = getattr(distributed_setup, "mesh_context", None)
        self.device_mesh = getattr(self.mesh_context, "device_mesh", None)
        self.moe_mesh = getattr(self.mesh_context, "moe_mesh", None)
        self.distributed_config = getattr(distributed_setup, "strategy_config", None)

        if self.config.model is not None:
            self._construct()

    def _construct(self) -> None:
        """Build distributed setup + model + optimizer from ``self.config``.

        Reuses the LLM recipe's ``build_model`` and the optimizer config's
        ``.build()`` so the Engine and the recipe construct identical objects.
        LR schedulers are not built here — inject them or drive LR via
        ``optim_step(lr=...)``.
        """
        from nemo_automodel.components.distributed.config import DistributedSetup
        from nemo_automodel.recipes._dist_utils import (
            create_distributed_setup_from_config,
            shard_optimizers_for_megatron_fsdp,
        )
        from nemo_automodel.recipes.llm.train_ft import build_model

        # 1. Distributed setup.
        if self.distributed_setup is None:
            if isinstance(self.config.distributed, DistributedSetup):
                self.distributed_setup = self.config.distributed
            else:
                self.distributed_setup = create_distributed_setup_from_config(self.config.distributed)
        self.mesh_context = getattr(self.distributed_setup, "mesh_context", None)
        self.device_mesh = getattr(self.mesh_context, "device_mesh", None)
        self.moe_mesh = getattr(self.mesh_context, "moe_mesh", None)
        self.distributed_config = getattr(self.distributed_setup, "strategy_config", None)

        # 2. Model.
        model = build_model(
            self.config.model,
            self.config.peft,
            has_packed_sequence=self.config.has_packed_sequence,
            seed=self.config.seed,
            cfg_fp8=self.config.fp8,
            cfg_compile=self.config.compile,
            cfg_quantization=self.config.quantization,
            distributed_setup=self.distributed_setup,
            cfg_qat=self.config.qat,
            sdpa_method=self.config.sdpa_method,
        )

        # AutoPipeline carries .parts; a plain module is a single part.
        if hasattr(model, "parts"):
            self.model_parts = list(model.parts)
            self.pp = model
        else:
            self.model_parts = [model]
            self.pp = None

        # 3. Optimizer (a list — one per part — possibly megatron-fsdp sharded).
        if self.config.optimizer is not None:
            optimizer = self.config.optimizer.build(
                model, device_mesh=self.device_mesh, is_peft=self.config.peft is not None
            )
            allow = getattr(self.config.optimizer, "supports_megatron_fsdp_sharding", True)
            self.optimizers = shard_optimizers_for_megatron_fsdp(model, optimizer, self.distributed_config, allow=allow)

    # ── Introspection ────────────────────────────────────────────────

    @property
    def parts(self) -> list[nn.Module]:
        """PP-unified model parts. Always a list."""
        return self.model_parts

    @property
    def model(self) -> nn.Module:
        """The single model part (raises under PP — use ``parts``)."""
        if len(self.model_parts) != 1:
            raise RuntimeError(f"`model` is ambiguous with {len(self.model_parts)} parts; use `parts`.")
        return self.model_parts[0]

    @property
    def pp_enabled(self) -> bool:
        return self.pp is not None

    @property
    def device(self) -> torch.device:
        if self.model_parts:
            return next(self.model_parts[0].parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _dp_group(self, *, include_cp: bool = False):
        if not self.device_mesh:
            return None
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        if include_cp and self.device_mesh["cp"].size() > 1:
            return get_flat_mesh(self.device_mesh, "dp_cp").get_group()
        return get_flat_mesh(self.device_mesh, "dp").get_group()

    @property
    def dp_group(self):
        return self._dp_group()

    @property
    def dp_size(self) -> int:
        group = self._dp_group(include_cp=True)
        if group is None:
            return dist.get_world_size() if dist.is_initialized() else 1
        return group.size()

    @property
    def dp_rank(self) -> int:
        if not self.device_mesh:
            return dist.get_rank() if dist.is_initialized() else 0
        from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh

        name = "dp_cp" if self.device_mesh["cp"].size() > 1 else "dp"
        return get_flat_mesh(self.device_mesh, name).get_local_rank()

    # ── Forward / backward ───────────────────────────────────────────

    def _normalize_inputs(self, batch: Any) -> tuple[list[dict], bool]:
        """Return ``(microbatches, datum_input)``.

        Accepts a single batch dict, a list of microbatch dicts, or a list of
        :class:`Datum` (collated into one microbatch). ``datum_input`` signals
        that the caller wants :class:`ModelOutput` back.
        """
        if isinstance(batch, dict):
            return [batch], False
        if isinstance(batch, Sequence):
            return list(batch), False
        raise TypeError(f"Unsupported batch type: {type(batch)!r}")

    @staticmethod
    def _is_datum_input(batch: Any) -> bool:
        """True for ``list[Datum]`` or ``list[list[Datum]]`` (the tinker path)."""
        if not isinstance(batch, Sequence) or isinstance(batch, dict) or len(batch) == 0:
            return False
        head = batch[0]
        if isinstance(head, Datum):
            return True
        return isinstance(head, Sequence) and len(head) > 0 and isinstance(head[0], Datum)

    @staticmethod
    def _as_microbatch_datum_lists(batch: Any) -> list[list[Datum]]:
        """Normalize Datum input to ``list[list[Datum]]`` (one inner list per microbatch)."""
        return [list(batch)] if isinstance(batch[0], Datum) else [list(mb) for mb in batch]

    def forward_backward(
        self,
        batch: Any,
        loss_fn: "str | LossFn | Callable | None" = None,
        *,
        loss_kwargs: dict | None = None,
        num_microbatches: int | None = None,
        forward_only: bool = False,
        num_label_tokens: int | None = None,
    ) -> dict | ModelOutput:
        """Run forward + (optional) backward over one or more microbatches.

        Two input modes:

        * **Datum mode** (``list[Datum]`` or ``list[list[Datum]]``): the tinker
          path. ``loss_fn`` is a built-in name (``"cross_entropy"`` default,
          ``"importance_sampling"``, ``"ppo"``) or a ``LossFn`` callable
          ``(ModelOutput, Sequence[Datum]) -> Sequence[Tensor]``. The Engine owns
          weighting, the global token/sample denominator across data ranks, and
          backward. Returns a :class:`ModelOutput`.
        * **Dict mode** (``dict`` or ``list[dict]``): the legacy SFT path.
          ``loss_fn`` is an Automodel loss instance (or ``None`` to use the
          model's own loss); reduction goes through ``calculate_loss``. Returns
          ``{"loss", "metrics"}``.

        The microbatch lifecycle (grad-accumulation prep, MoE aux-loss scaling,
        final-backward hook) is handled here so callers don't reproduce it.
        """
        if self.pp_enabled:
            raise NotImplementedError(
                "Engine.forward_backward does not yet support pipeline parallelism; "
                "this is a planned follow-up. Use the recipe PP path for now."
            )

        if self._is_datum_input(batch):
            return self._forward_backward_datums(
                self._as_microbatch_datum_lists(batch),
                loss_fn if loss_fn is not None else "cross_entropy",
                loss_kwargs or {},
                forward_only=forward_only,
            )

        from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
        from nemo_automodel.components.distributed.utils import get_sync_ctx
        from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
        from nemo_automodel.components.loss.utils import calculate_loss
        from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
        from nemo_automodel.components.training.utils import (
            prepare_after_first_microbatch,
            prepare_for_final_backward,
            prepare_for_grad_accumulation,
        )
        from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

        microbatches, datum_input = self._normalize_inputs(batch)
        n = num_microbatches or len(microbatches)
        is_train = not forward_only
        loss_fn = loss_fn or self.config.loss_fn
        model = self.model_parts[0]
        device = self.device

        if is_train:
            prepare_for_grad_accumulation(self.model_parts, pp_enabled=False)
            self._set_moe_scale(num_label_tokens)

        losses: list[torch.Tensor] = []
        for i, mb in enumerate(microbatches):
            is_last = i == n - 1
            if is_train and is_last:
                prepare_for_final_backward(self.model_parts, pp_enabled=False)

            mb = self._batch_to_device(mb, device)
            train_ctx, mb = make_cp_batch_and_ctx(
                self.device_mesh,
                mb,
                use_te=self.config.cp_use_te,
                padding_token_id=self.config.cp_padding_token_id,
                num_chunks=self.config.cp_num_chunks,
            )
            labels = mb.pop("labels", None)
            sync_ctx = get_sync_ctx(model, is_last, self.config.defer_fsdp_grad_sync) if is_train else nullcontext()
            with train_ctx(), sync_ctx:
                fwd = filter_forward_kwargs(model, mb)
                if isinstance(loss_fn, FusedLinearCrossEntropy):
                    out = model(logits_to_keep=1, **fwd)
                else:
                    out = model(**fwd)

                if loss_fn is None:
                    # Model computed its own loss from labels in the batch.
                    local_loss = out.loss if hasattr(out, "loss") else out["loss"]
                else:
                    local_loss = calculate_loss(
                        loss_fn,
                        logits=getattr(out, "logits", out),
                        labels=labels,
                        model=model,
                        hidden_states=get_final_hidden_states(out),
                        num_label_tokens=num_label_tokens,
                    )
                if is_train:
                    local_loss.backward()
            losses.append(local_loss.detach())

            if is_train and i == 0:
                prepare_after_first_microbatch()

        loss_tensor = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=device)
        metrics = {"loss": float(loss_tensor)}
        if datum_input:
            return ModelOutput(loss=loss_tensor, metrics=metrics)
        return {"loss": loss_tensor, "metrics": metrics}

    @torch.no_grad()
    def forward(self, datums: Sequence[Datum], *, broadcast_output: bool = True) -> ModelOutput:
        """Forward-only pass returning per-datum logprobs and entropy.

        Reference-policy / critic evaluation. Runs the model over the collated
        ``datums``, then slices vocab outputs back to per-datum, per-token
        ``logprobs`` (of each datum's ``target_tokens``) and ``entropy``.

        Currently supports the non-PP path with full logits (no
        ``FusedLinearCrossEntropy``); ``broadcast_output`` is accepted for API
        forward-compatibility with the PP path.
        """
        if self.pp_enabled:
            raise NotImplementedError("Engine.forward does not yet support pipeline parallelism.")
        if not datums or not isinstance(datums[0], Datum):
            raise TypeError("Engine.forward requires a non-empty sequence of Datum.")

        from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
        from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

        datums = list(datums)
        model = self.model_parts[0]
        device = self.device
        batch = self._batch_to_device(collate_datums(datums), device)
        train_ctx, batch = make_cp_batch_and_ctx(
            self.device_mesh,
            batch,
            use_te=self.config.cp_use_te,
            padding_token_id=self.config.cp_padding_token_id,
            num_chunks=self.config.cp_num_chunks,
        )
        labels = batch.pop("labels", None)
        with train_ctx():
            out = model(**filter_forward_kwargs(model, batch))
        return self._build_model_output(getattr(out, "logits", out), datums, labels, detach=True)

    def _build_model_output(
        self,
        logits: torch.Tensor,
        datums: list[Datum],
        labels: torch.Tensor | None,
        *,
        detach: bool,
    ) -> ModelOutput:
        """Slice padded ``[B, T, V]`` logits into per-datum logprobs/entropy.

        ``logprobs[i]`` is the per-token logprob the model assigns to datum ``i``'s
        ``target_tokens``; ``entropy[i]`` the per-token entropy. With ``detach``
        the graph is dropped (forward-only); without it the tensors stay
        differentiable for the loss path.
        """
        per_logprobs: list[torch.Tensor] | None = None
        if labels is not None:
            token_lp = selected_token_logprobs(logits, labels.clamp_min(0))  # [B, T]
            per_logprobs = [token_lp[i, : d.seq_len] for i, d in enumerate(datums)]
        token_ent = compute_entropy(logits)  # [B, T]
        per_entropy = [token_ent[i, : d.seq_len] for i, d in enumerate(datums)]
        if detach:
            per_logprobs = [t.detach() for t in per_logprobs] if per_logprobs is not None else None
            per_entropy = [t.detach() for t in per_entropy]
        return ModelOutput(logprobs=per_logprobs, entropy=per_entropy, metrics={})

    def _forward_backward_datums(
        self,
        mb_datum_lists: list[list[Datum]],
        loss_fn: "str | LossFn",
        loss_kwargs: dict,
        *,
        forward_only: bool,
    ) -> ModelOutput:
        """Datum/LossFn path: forward → per-datum loss → reduce → backward.

        Owns the doc's normalization: applies ``weights``, divides by the global
        token (or sample) count across data ranks, and runs the microbatch
        lifecycle. Matches the recipe's non-PP loss formula (weighted sum of
        per-token losses / global token count), so DP scaling stays identical to
        the proven path.
        """
        from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
        from nemo_automodel.components.distributed.utils import get_sync_ctx
        from nemo_automodel.components.training.utils import (
            prepare_after_first_microbatch,
            prepare_for_final_backward,
            prepare_for_grad_accumulation,
        )
        from nemo_automodel.components.utils.model_utils import filter_forward_kwargs

        loss_fn = BUILTIN_LOSSES[loss_fn] if isinstance(loss_fn, str) else loss_fn
        all_datums = [d for mb in mb_datum_lists for d in mb]
        token_denom, sample_denom = self._global_denominator(all_datums)
        is_train = not forward_only
        model = self.model_parts[0]
        device = self.device
        n = len(mb_datum_lists)

        if is_train:
            prepare_for_grad_accumulation(self.model_parts, pp_enabled=False)
            self._set_moe_scale(int(token_denom))

        agg_logprobs: list[torch.Tensor] = []
        agg_entropy: list[torch.Tensor] = []
        loss_sum = torch.zeros((), device=device)
        for i, mb in enumerate(mb_datum_lists):
            is_last = i == n - 1
            if is_train and is_last:
                prepare_for_final_backward(self.model_parts, pp_enabled=False)

            batch = self._batch_to_device(collate_datums(mb), device)
            train_ctx, batch = make_cp_batch_and_ctx(
                self.device_mesh,
                batch,
                use_te=self.config.cp_use_te,
                padding_token_id=self.config.cp_padding_token_id,
                num_chunks=self.config.cp_num_chunks,
            )
            labels = batch.pop("labels", None)
            sync_ctx = get_sync_ctx(model, is_last, self.config.defer_fsdp_grad_sync) if is_train else nullcontext()
            fwd_ctx = nullcontext() if is_train else torch.no_grad()
            with fwd_ctx, train_ctx(), sync_ctx:
                out = model(**filter_forward_kwargs(model, batch))
                mo = self._build_model_output(getattr(out, "logits", out), mb, labels, detach=False)
                per_datum_losses = loss_fn(mo, mb, **loss_kwargs)
                loss = self._reduce_datum_losses(per_datum_losses, mb, token_denom, sample_denom)
                if is_train:
                    loss.backward()
            loss_sum = loss_sum + loss.detach()
            agg_logprobs.extend(t.detach() for t in (mo.logprobs or []))
            agg_entropy.extend(t.detach() for t in (mo.entropy or []))
            if is_train and i == 0:
                prepare_after_first_microbatch()

        return ModelOutput(
            loss=loss_sum,
            logprobs=agg_logprobs or None,
            entropy=agg_entropy or None,
            metrics={"loss": float(loss_sum)},
        )

    @staticmethod
    def _reduce_datum_losses(
        per_datum_losses: Sequence[torch.Tensor],
        datums: list[Datum],
        token_denom: float,
        sample_denom: float,
    ) -> torch.Tensor:
        """Weight + sum per-datum losses, normalized by the global denominator.

        Token-level (per-token tensors) are multiplied by ``weights`` and summed,
        divided by the global token count. Sample-level (scalar per datum) are
        summed and divided by the global sample count. Homogeneity is decided
        from the first datum's loss shape.
        """
        sample_level = per_datum_losses[0].numel() == 1
        total = None
        for loss_i, d in zip(per_datum_losses, datums):
            if sample_level:
                contrib = loss_i.reshape(())
            else:
                w = d.loss_inputs.get("weights")
                contrib = (loss_i * w.to(loss_i)).sum() if w is not None else loss_i.sum()
            total = contrib if total is None else total + contrib
        denom = sample_denom if sample_level else token_denom
        return total / max(denom, 1e-8)

    def _global_denominator(self, datums: list[Datum]) -> tuple[float, float]:
        """Global (token, sample) counts across data ranks for loss normalization.

        Token count = sum of ``weights`` (or sequence lengths when absent);
        sample count = number of datums. Both all-reduced over the DP(+CP) group.
        """
        token_local = 0.0
        for d in datums:
            w = d.loss_inputs.get("weights")
            token_local += float(w.sum()) if w is not None else float(d.seq_len)
        sample_local = float(len(datums))

        group = self._dp_group(include_cp=True)
        if group is not None and dist.is_initialized():
            t = torch.tensor([token_local, sample_local], device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM, group=group)
            token_local, sample_local = float(t[0]), float(t[1])
        return token_local, sample_local

    # ── Optimizer ────────────────────────────────────────────────────

    def zero_grad(self) -> None:
        for opt in self.optimizers:
            opt.zero_grad()

    def optimizer_step(self, num_label_tokens: int | None = None) -> tuple[bool, float]:
        """Scale + clip grads, then step every optimizer. Returns ``(ok, grad_norm)``."""
        if not self.optimizers:
            raise RuntimeError("no optimizers; build from config or inject them.")
        from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm

        moe_mesh = self.moe_mesh
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm=self.config.max_grad_norm,
            model_parts=self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=moe_mesh,
            ep_axis_name="ep" if moe_mesh is not None and "ep" in moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            num_label_tokens=num_label_tokens,
            dp_group_size=self.dp_size,
        )
        grad_norm_val = float(grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm)
        ok = math.isfinite(grad_norm_val)
        for opt in self.optimizers:
            if ok:
                opt.step()
            opt.zero_grad()
        return ok, grad_norm_val

    def lr_scheduler_step(self) -> float | list[float]:
        for sch in self.lr_schedulers:
            try:
                sch.step(1)
            except TypeError:
                sch.step()
        return self._current_lrs()

    def optim_step(self, *, lr: float, num_label_tokens: int | None = None) -> dict:
        """Tinker-style step: set ``lr`` on all param groups, then clip + step.

        Returns ``{"grad_norm", "lr", "update_succeeded"}``.
        """
        for opt in self.optimizers:
            for group in opt.param_groups:
                group["lr"] = lr
        ok, grad_norm = self.optimizer_step(num_label_tokens=num_label_tokens)
        return {"grad_norm": grad_norm, "lr": lr, "update_succeeded": ok}

    def _current_lrs(self) -> float | list[float]:
        lrs = [g["lr"] for opt in self.optimizers for g in opt.param_groups]
        return lrs[0] if len(lrs) == 1 else lrs

    # ── Mode switching ───────────────────────────────────────────────

    @contextmanager
    def train_mode(self):
        prev = [p.training for p in self.model_parts]
        for p in self.model_parts:
            p.train()
        try:
            yield
        finally:
            for p, was in zip(self.model_parts, prev):
                p.train(was)

    @contextmanager
    def eval_mode(self):
        prev = [p.training for p in self.model_parts]
        for p in self.model_parts:
            p.eval()
        try:
            yield
        finally:
            for p, was in zip(self.model_parts, prev):
                p.train(was)

    # ── Device movement ──────────────────────────────────────────────

    def to(self, device: str, *, model: bool = True, optimizer: bool = True) -> None:
        """Move model parts and/or optimizer state to ``device`` (``"cpu"``/``"cuda"``)."""
        from nemo_automodel.components.training.utils import move_to_device

        if model:
            for part in self.model_parts:
                move_to_device(part, device)
        if optimizer:
            for opt in self.optimizers:
                for state in opt.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(device)

    # ── Weight export (basic streaming) ──────────────────────────────

    def export_weights(self):
        """Yield ``(name, tensor)`` over model parameters, materializing DTensors.

        Minimal streaming export for inference refit. HF-name conversion and
        PEFT merging are a planned follow-up (Phase 5 ``export``).
        """
        for part in self.model_parts:
            for name, param in part.named_parameters():
                tensor = param.full_tensor() if hasattr(param, "full_tensor") else param
                yield name, tensor.detach()

    # ── Helpers ──────────────────────────────────────────────────────

    def _set_moe_scale(self, num_label_tokens: int | None) -> None:
        """Set the MoE aux-loss backward scale to cancel DP/PP grad scaling."""
        if self.moe_mesh is None:
            return
        from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

        scale = float(num_label_tokens) if num_label_tokens is not None else float(self.dp_size)
        MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(scale)

    @staticmethod
    def _batch_to_device(batch: dict, device: torch.device) -> dict:
        def move(v):
            if isinstance(v, torch.Tensor):
                return v.to(device, non_blocking=True)
            if isinstance(v, dict):
                return {k: move(x) for k, x in v.items() if x is not None}
            return v

        return {k: move(v) for k, v in batch.items()}
