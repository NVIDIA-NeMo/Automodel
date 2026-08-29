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

"""Model Optimizer DMD2 step distillation for the native diffusion trainer."""

from __future__ import annotations

from contextlib import contextmanager
from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.recipes.diffusion.train import build_diffusion_pipeline
from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe


def _require_fastgen() -> Any:
    """Import ModelOpt only when DMD2 is configured."""
    has_fastgen, fastgen = safe_import("modelopt.torch.fastgen")
    if not has_fastgen:
        raise ImportError(
            "DMD2 requires ModelOpt FastGen; install with `uv sync --extra dmd2`, which resolves the "
            "rev pinned in [tool.uv.sources]. Note that `pip install nemo_automodel[dmd2]` resolves a "
            "PyPI release instead, whose Qwen-Image plugin still passes `txt_seq_lens` and therefore "
            "does not work with the diffusers>=0.39 this package requires."
        )
    return fastgen


def _load_negative_prompt_embedding(path: str) -> torch.Tensor:
    """Load a user-generated static negative embedding with shape ``[S, D]``."""
    embedding_path = Path(path).expanduser()
    if not embedding_path.is_file():
        raise FileNotFoundError(f"DMD2 negative prompt embedding does not exist: {embedding_path}")
    embedding = torch.load(embedding_path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(embedding) or embedding.ndim != 2 or not embedding.is_floating_point():
        raise ValueError("DMD2 negative prompt embedding must be one floating-point tensor with shape [S, D].")
    if not torch.isfinite(embedding).all():
        raise ValueError(f"DMD2 negative prompt embedding contains non-finite values: {embedding_path}")
    return embedding.contiguous()


class _DMD2Objective:
    """Add full Model Optimizer DMD2 updates to ``TrainDiffusionRecipe``.

    Model Optimizer owns CFG, VSD/DSM, multi-step simulation, GAN/R1, and EMA.
    This object only owns the auxiliary models, optimizer alternation, replicated
    discriminator synchronization, and their checkpoint state.
    """

    use_distributed_checkpointing = True

    def __init__(self, cfg: ConfigNode) -> None:
        self.cfg = cfg
        fastgen = _require_fastgen()
        config = cfg.to_dict()
        for key in ("discriminator", "feature_capture", "negative_prompt_embedding_path", "pipeline"):
            config.pop(key, None)
        self.dmd_config = fastgen.DMDConfig(**config)
        if self.dmd_config.student_update_freq < 2:
            raise ValueError(
                "dmd2.student_update_freq must be at least 2, got "
                f"{self.dmd_config.student_update_freq}. The trainer runs a student update when "
                "`step % student_update_freq == 0` and a fake-score/discriminator update otherwise, "
                "so student_update_freq=1 makes every step a student step and never trains the fake "
                "score or discriminator."
            )

        self.dmd_pipeline: Any | None = None
        self.teacher: nn.Module | None = None
        self.fake_score: nn.Module | None = None
        self.student_optimizer: torch.optim.Optimizer | None = None
        self.fake_score_optimizer: torch.optim.Optimizer | None = None
        self.discriminator: nn.Module | None = None
        self.discriminator_optimizer: torch.optim.Optimizer | None = None
        self.negative_prompt_embedding: torch.Tensor | None = None
        self._feature_capture_shape: tuple[int, int] | None = None
        self._fake_score_state: ModelState | None = None
        self._fake_score_optimizer_state: OptimizerState | None = None
        self._discriminator_state: ModelState | None = None
        self._discriminator_optimizer_state: OptimizerState | None = None

    def configure(self, recipe: TrainDiffusionRecipe) -> None:
        """Validate the supported training topology before loading the student."""
        if recipe.cfg.get("model.mode", "finetune").lower() != "finetune":
            raise ValueError("DMD2 requires model.mode=finetune.")
        if recipe.cfg.get("peft", None) is not None:
            raise ValueError("DMD2 currently requires full-parameter training; remove the `peft` block.")
        if recipe.cfg.get("ddp", None) is not None or recipe.cfg.get("fsdp", None) is None:
            raise ValueError("DMD2 requires an `fsdp` block and does not support DDP.")

        fsdp_cfg = recipe.cfg.get("fsdp", {})
        unsupported = {
            name: int(fsdp_cfg.get(name, 1))
            for name in ("tp_size", "cp_size", "pp_size", "ep_size")
            if int(fsdp_cfg.get(name, 1)) != 1
        }
        if unsupported:
            values = ", ".join(f"{name}={value}" for name, value in unsupported.items())
            raise ValueError(f"DMD2 currently supports data-parallel FSDP2 only; got {values}.")
        if bool(recipe.cfg.get("data.dataloader.train_text_encoder", False)):
            raise ValueError("DMD2 requires AutoModel's precomputed text embeddings.")
        if recipe.cfg.get("model.stage", None) is not None:
            raise ValueError("DMD2 does not support model.stage.")
        if bool(recipe.cfg.get("model.transformer_engine_fp8", False)):
            raise ValueError("DMD2 does not yet support Transformer Engine FP8 state.")
        if recipe.cfg.get("model.attention_backend", None) == "flash":
            raise ValueError("DMD2 requires text masks, which the flash-attn 2 backend does not support.")
        if recipe.cfg.optimizer is None:
            raise ValueError("DMD2 requires the native top-level `optimizer` configuration.")

        ema = self.dmd_config.ema
        if ema is not None and (not ema.fsdp2 or ema.mode != "full_tensor"):
            raise ValueError("DMD2 checkpoint resume requires EMA with fsdp2=true and mode=full_tensor.")

    def primary_optimizer_steps(self, outer_steps: int) -> int:
        """Return the number of student updates in ``outer_steps``."""
        return ceil(outer_steps / int(self.dmd_config.student_update_freq))

    def build_lr_scheduler(self, recipe: TrainDiffusionRecipe) -> list[Any] | None:
        """Build the native student scheduler against the student-update budget."""
        if recipe.cfg.lr_scheduler is None:
            return None
        step_scheduler = recipe.step_scheduler
        if step_scheduler.epoch_len is not None:
            outer_steps = step_scheduler.num_epochs * step_scheduler.epoch_len
            if step_scheduler.max_steps is not None:
                outer_steps = min(outer_steps, step_scheduler.max_steps)
        else:
            outer_steps = step_scheduler.max_steps
        if outer_steps is None or outer_steps <= 0:
            raise ValueError("DMD2 could not resolve a positive outer-step budget for the LR scheduler.")
        return recipe.cfg.lr_scheduler.build(
            recipe.optimizer,
            step_scheduler,
            total_steps=self.primary_optimizer_steps(outer_steps),
        )

    def setup(self, recipe: TrainDiffusionRecipe) -> None:
        """Create DMD2 auxiliaries before the native checkpoint restore."""
        negative_path = self.cfg.get("negative_prompt_embedding_path", None)
        if self.dmd_config.guidance_scale is not None:
            if not negative_path:
                raise ValueError("DMD2 CFG requires dmd2.negative_prompt_embedding_path.")
            self.negative_prompt_embedding = _load_negative_prompt_embedding(str(negative_path)).to(
                recipe.device,
                dtype=recipe.compute_dtype,
            )

        self.student_optimizer = self._only_optimizer(recipe.optimizer, "student")
        self.teacher = self._build_auxiliary_transformer(recipe, trainable=False)
        self.fake_score = self._build_auxiliary_transformer(recipe, trainable=True)
        self.fake_score_optimizer = self._build_auxiliary_optimizer(recipe, self.fake_score, "fake-score")

        if self.dmd_config.gan_loss_weight_gen > 0:
            discriminator_cfg = self.cfg.get("discriminator", None)
            if discriminator_cfg is None:
                raise ValueError("DMD2 GAN requires dmd2.discriminator arguments.")
            if self.cfg.get("feature_capture", None) is None:
                raise ValueError("DMD2 GAN requires dmd2.feature_capture.")
            self.discriminator = discriminator_cfg.instantiate().to(
                device=recipe.device,
                dtype=recipe.compute_dtype,
            )
            self._broadcast_discriminator(recipe)
            self.discriminator_optimizer = self._build_auxiliary_optimizer(
                recipe,
                self.discriminator,
                "discriminator",
            )

        pipeline_cfg = self.cfg.get("pipeline", None)
        if pipeline_cfg is None:
            raise ValueError("DMD2 requires dmd2.pipeline.")
        self.dmd_pipeline = pipeline_cfg.instantiate(
            student=recipe.model,
            teacher=self.teacher,
            fake_score=self.fake_score,
            config=self.dmd_config,
            discriminator=self.discriminator,
        )

        self._fake_score_state = ModelState(self.fake_score, cpu_offload=recipe.cpu_offload)
        self._fake_score_optimizer_state = OptimizerState(
            self.fake_score,
            self.fake_score_optimizer,
            cpu_offload=recipe.cpu_offload,
        )
        if self.discriminator is not None and self.discriminator_optimizer is not None:
            self._discriminator_state = ModelState(self.discriminator, cpu_offload=recipe.cpu_offload)
            self._discriminator_optimizer_state = OptimizerState(
                self.discriminator,
                self.discriminator_optimizer,
                cpu_offload=recipe.cpu_offload,
            )

    def after_restore(self, recipe: TrainDiffusionRecipe) -> None:
        """Verify the restored student scheduler is on the DMD2 phase boundary."""
        if recipe.lr_scheduler is None:
            return
        completed_steps = int(recipe.step_scheduler.step)
        expected_student_steps = ceil(completed_steps / int(self.dmd_config.student_update_freq))
        restored_student_steps = int(recipe.lr_scheduler[0].num_steps)
        if restored_student_steps != expected_student_steps:
            raise ValueError(
                "DMD2 checkpoint has an inconsistent student LR scheduler: "
                f"expected {expected_student_steps}, restored {restored_student_steps}."
            )

    @contextmanager
    def _discriminator_trainable(self):
        """Expose the discriminator as trainable while DCP materializes its optimizer state.

        ``torch.distributed.checkpoint``'s ``_init_optim_state`` only zero-fills gradients for
        parameters with ``requires_grad=True``, so it creates no optimizer state for a frozen
        model. :meth:`_set_phase` leaves the discriminator frozen after every student phase,
        so without this a checkpoint taken during a student phase — the first outer step, a
        ``ckpt_every_steps: 1`` run, or any preemption — would persist ``param_groups`` with no
        ``state`` entries. The resume-side load rebuilds the discriminator trainable and then
        asks DCP for keys that were never written, failing with
        "Missing key in checkpoint state_dict: discriminator_optimizer.optim.state...".
        """
        if self.discriminator is None:
            yield
            return
        was_trainable = [parameter.requires_grad for parameter in self.discriminator.parameters()]
        self.discriminator.requires_grad_(True)
        try:
            yield
        finally:
            for parameter, trainable in zip(self.discriminator.parameters(), was_trainable):
                parameter.requires_grad_(trainable)

    def state_dict(self) -> dict[str, Any]:
        """Return DCP-compatible fake-score, discriminator, optimizer, and EMA state."""
        self._require_ready()
        assert self._fake_score_state is not None
        assert self._fake_score_optimizer_state is not None
        with self._discriminator_trainable():
            state = {
                "fake_score": self._fake_score_state.state_dict(),
                "fake_score_optimizer": self._fake_score_optimizer_state.state_dict(),
            }
            if self._discriminator_state is not None:
                state["discriminator"] = self._discriminator_state.state_dict()
            if self._discriminator_optimizer_state is not None:
                state["discriminator_optimizer"] = self._discriminator_optimizer_state.state_dict()
            if self.dmd_pipeline.ema is not None:
                state["ema"] = self.dmd_pipeline.ema.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore the DCP-populated DMD2 auxiliary state.

        Args:
            state: Mapping returned by :meth:`state_dict`. Fake-score and
                discriminator tensors use their logical parameter shapes,
                optimizer tensors preserve those axes, and EMA tensors match
                the student parameters they shadow.
        """
        self._require_ready()
        assert self._fake_score_state is not None
        assert self._fake_score_optimizer_state is not None
        with self._discriminator_trainable():
            self._fake_score_state.load_state_dict(state["fake_score"])
            self._fake_score_optimizer_state.load_state_dict(state["fake_score_optimizer"])
            if self._discriminator_state is not None:
                self._discriminator_state.load_state_dict(state["discriminator"])
            if self._discriminator_optimizer_state is not None:
                self._discriminator_optimizer_state.load_state_dict(state["discriminator_optimizer"])
            if self.dmd_pipeline.ema is not None:
                self.dmd_pipeline.ema.load_state_dict(state["ema"])

    def train_batch_group(
        self,
        recipe: TrainDiffusionRecipe,
        batch_group: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[float, float]:
        """Run one student or fake-score/discriminator phase.

        Args:
            recipe: Active recipe owning the student and distributed state.
            batch_group: Microbatches containing image latents ``[B,C,H,W]``,
                text embeddings ``[B,S,D]``, and masks ``[B,S]``.
            global_step: Outer step selecting the active DMD2 phase.

        Returns:
            Mean loss and active-model gradient norm scalars.
        """
        self._require_ready()
        assert self.fake_score is not None
        assert self.student_optimizer is not None
        assert self.fake_score_optimizer is not None
        assert self.teacher is not None
        if not batch_group:
            raise RuntimeError("DMD2 received an empty gradient-accumulation group.")

        student_phase = global_step % int(self.dmd_config.student_update_freq) == 0
        phase = "student" if student_phase else "fake_score"
        active_model = recipe.model if student_phase else self.fake_score
        active_optimizer = self.student_optimizer if student_phase else self.fake_score_optimizer
        assert active_model is not None
        assert active_optimizer is not None
        self._set_phase(recipe, student_phase)

        active_optimizer.zero_grad(set_to_none=True)
        if not student_phase and self.discriminator_optimizer is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        prepare_for_grad_accumulation([active_model], pp_enabled=False)
        total_losses: list[torch.Tensor] = []
        num_microbatches = len(batch_group)

        for index, micro_batch in enumerate(batch_group):
            if index == num_microbatches - 1:
                prepare_for_final_backward([active_model], pp_enabled=False)

            (
                latents,
                noise,
                text_embeddings,
                text_mask,
                negative_embeddings,
                negative_mask,
            ) = self._prepare_batch(recipe, micro_batch)
            kwargs = {
                "encoder_hidden_states": text_embeddings,
                "encoder_hidden_states_mask": text_mask,
            }
            if student_phase:
                losses = self.dmd_pipeline.compute_student_loss(
                    latents,
                    noise,
                    negative_encoder_hidden_states=negative_embeddings,
                    negative_encoder_hidden_states_mask=negative_mask,
                    **kwargs,
                )
            else:
                losses = self.dmd_pipeline.compute_fake_score_loss(latents, noise, **kwargs)

            total = losses["total"]
            if recipe.check_loss and not torch.isfinite(total).all():
                raise FloatingPointError(f"Non-finite DMD2 {phase} loss at step {global_step}.")
            (total / num_microbatches).backward()
            reported_total = total.detach()

            if not student_phase and self.discriminator_optimizer is not None:
                discriminator_losses = self.dmd_pipeline.compute_discriminator_loss(latents, noise, **kwargs)
                discriminator_total = discriminator_losses["total"]
                if recipe.check_loss and not torch.isfinite(discriminator_total).all():
                    raise FloatingPointError(f"Non-finite DMD2 discriminator loss at step {global_step}.")
                (discriminator_total / num_microbatches).backward()
                reported_total = reported_total + discriminator_total.detach()

            total_losses.append(reported_total)
            if index == 0:
                prepare_after_first_microbatch()

        grad_norm = clip_grad_norm(
            recipe.clip_grad_max_norm,
            [active_model],
            foreach=recipe.grad_clip_foreach,
        )
        grad_norm = float(grad_norm)

        if not student_phase and self.discriminator_optimizer is not None:
            self._synchronize_discriminator_gradients(recipe)
            clip_grad_norm(
                recipe.clip_grad_max_norm,
                [self.discriminator],
                foreach=recipe.grad_clip_foreach,
                use_torch_clip_grad_norm=True,
            )

        active_optimizer.step()
        if not student_phase and self.discriminator_optimizer is not None:
            self.discriminator_optimizer.step()
        if student_phase:
            if recipe.lr_scheduler is not None:
                recipe.lr_scheduler[0].step(1)
            self.dmd_pipeline.update_ema(iteration=global_step // int(self.dmd_config.student_update_freq) + 1)

        active_optimizer.zero_grad(set_to_none=True)
        if not student_phase and self.discriminator_optimizer is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        return float(torch.stack(total_losses).mean().item()), grad_norm

    def _build_auxiliary_transformer(
        self,
        recipe: TrainDiffusionRecipe,
        *,
        trainable: bool,
    ) -> nn.Module:
        """Build one teacher or fake-score transformer with the native diffusion loader."""
        pipe, _ = build_diffusion_pipeline(
            model_id=recipe.model_id,
            finetune_mode=True,
            device=recipe.device,
            dtype=recipe.model_dtype,
            compute_dtype=recipe.compute_dtype,
            cpu_offload=recipe.cpu_offload,
            fsdp_cfg=recipe.cfg.get("fsdp", None),
            ddp_cfg=None,
            attention_backend=recipe.attention_backend,
            transformer_engine_linear=recipe.transformer_engine_linear,
            transformer_engine_fp8_safe_only=recipe.transformer_engine_fp8,
            fuse_qkv_projections=recipe.fuse_qkv_projections,
            compact_fused_qkv_projections=recipe.compact_fused_qkv_projections,
            active_transformer=recipe.active_transformer,
        )
        model = pipe.transformer
        model.train(trainable)
        model.requires_grad_(trainable)
        return model

    @staticmethod
    def _only_optimizer(
        optimizers: list[torch.optim.Optimizer] | torch.optim.Optimizer,
        component: str,
    ) -> torch.optim.Optimizer:
        """Return the single optimizer allowed by the supported DP-only topology."""
        if isinstance(optimizers, torch.optim.Optimizer):
            return optimizers
        if len(optimizers) != 1:
            raise ValueError(f"DMD2 requires one {component} optimizer, got {len(optimizers)}.")
        return optimizers[0]

    def _build_auxiliary_optimizer(
        self,
        recipe: TrainDiffusionRecipe,
        model: nn.Module,
        component: str,
    ) -> torch.optim.Optimizer:
        """Build an auxiliary optimizer through AutoModel's typed optimizer config."""
        assert recipe.cfg.optimizer is not None
        optimizers = recipe.cfg.optimizer.build(model, device_mesh=recipe.device_mesh)
        return self._only_optimizer(optimizers, component)

    def _prepare_batch(
        self,
        recipe: TrainDiffusionRecipe,
        batch: dict[str, Any],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Move and validate one cached image batch.

        Args:
            recipe: Active recipe providing the target device and dtype.
            batch: Mapping with image latents ``[B,C,H,W]``, text embeddings
                ``[B,S,D]`` or ``[S,D]``, and masks ``[B,S]`` or ``[S]``.

        Returns:
            Image latents ``[B,C,H,W]``, same-shaped noise, positive embeddings
            ``[B,S,D]`` and mask ``[B,S]``, plus optional negative embeddings
            ``[B,S_negative,D]`` and mask ``[B,S_negative]``.
        """
        latents = batch["image_latents"].to(recipe.device, dtype=recipe.compute_dtype, non_blocking=True)
        text_embeddings = batch["text_embeddings"].to(
            recipe.device,
            dtype=recipe.compute_dtype,
            non_blocking=True,
        )
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        text_mask = batch.get("text_embeddings_mask")
        if text_mask is None:
            raise ValueError("DMD2 requires text_embeddings_mask from AutoModel's text-to-image collator.")
        text_mask = text_mask.to(recipe.device, non_blocking=True)
        if text_mask.ndim == 1:
            text_mask = text_mask.unsqueeze(0)
        if latents.ndim != 4 or text_embeddings.ndim != 3 or latents.shape[0] != text_embeddings.shape[0]:
            raise ValueError(
                "DMD2 expects image_latents [B,C,H,W] and text_embeddings [B,S,D]; "
                f"got {tuple(latents.shape)} and {tuple(text_embeddings.shape)}."
            )
        if text_mask.shape != text_embeddings.shape[:2]:
            raise ValueError(
                "DMD2 expects text_embeddings_mask [B,S] matching text_embeddings; "
                f"got {tuple(text_mask.shape)} and {tuple(text_embeddings.shape)}."
            )

        feature_shape = (latents.shape[-2], latents.shape[-1])
        if self.discriminator is not None and self._feature_capture_shape != feature_shape:
            self.cfg.feature_capture.instantiate(
                teacher=self.teacher,
                feature_indices=sorted(self.discriminator.feature_indices),
                h_lat=feature_shape[0],
                w_lat=feature_shape[1],
            )
            self._feature_capture_shape = feature_shape

        negative = None
        negative_mask = None
        if self.negative_prompt_embedding is not None:
            if self.negative_prompt_embedding.shape[-1] != text_embeddings.shape[-1]:
                raise ValueError("DMD2 positive and negative text embedding dimensions must match.")
            negative = self.negative_prompt_embedding.unsqueeze(0).expand(latents.shape[0], -1, -1)
            negative_mask = torch.ones(
                negative.shape[:2],
                dtype=text_mask.dtype,
                device=recipe.device,
            )
        return latents, torch.randn_like(latents), text_embeddings, text_mask, negative, negative_mask

    def _set_phase(self, recipe: TrainDiffusionRecipe, student_phase: bool) -> None:
        """Select the active model for this phase.

        Only ``train()``/``eval()`` is toggled on the FSDP2-sharded student and fake score;
        their ``requires_grad`` flags are deliberately left enabled. FSDP2 caches a parameter
        group's ``_orig_dtype``/``_reduce_dtype`` on the module's *first* forward, computed
        only from parameters with ``requires_grad=True``
        (``FSDPParamGroup._init_mp_dtypes``), and caches ``None`` when that set is empty.
        Step 0 is always a student phase and ModelOpt evaluates the fake score inside
        ``compute_student_loss``, so freezing here would make the fake score's first forward a
        frozen one and poison that cache for the whole run: its gradients would then be
        reduce-scattered in the parameter dtype instead of the configured
        ``fsdp.reduce_dtype``, and fp32 master weights would fail outright with
        "attempting to assign a gradient with dtype 'c10::BFloat16'".

        Gradients still cannot leak into the inactive transformer: ModelOpt wraps the
        fake-score forward in ``compute_student_loss`` and the student forward in both
        ``compute_fake_score_loss`` and ``compute_discriminator_loss`` in ``torch.no_grad()``.

        The discriminator is replicated rather than sharded, so it has no cached
        mixed-precision state to corrupt. It stays frozen outside its own phase so the student
        phase does not build discriminator gradients that the next phase would discard.
        """
        recipe.model.train(student_phase)
        self.fake_score.train(not student_phase)
        if self.discriminator is not None:
            self.discriminator.train(not student_phase)
            self.discriminator.requires_grad_(not student_phase)

    def _broadcast_discriminator(self, recipe: TrainDiffusionRecipe) -> None:
        if not dist.is_initialized() or recipe._get_dp_group_size() == 1:
            return
        with torch.no_grad():
            for tensor in (*self.discriminator.parameters(), *self.discriminator.buffers()):
                dist.broadcast(tensor, src=0, group=recipe._get_dp_group())

    def _synchronize_discriminator_gradients(self, recipe: TrainDiffusionRecipe) -> None:
        if not dist.is_initialized() or recipe._get_dp_group_size() == 1:
            return
        dp_size = recipe._get_dp_group_size()
        for parameter in self.discriminator.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM, group=recipe._get_dp_group())
                parameter.grad.div_(dp_size)

    def _require_ready(self) -> None:
        if (
            self.dmd_pipeline is None
            or self.fake_score is None
            or self.student_optimizer is None
            or self.fake_score_optimizer is None
        ):
            raise RuntimeError("DMD2 objective has not been set up.")
