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

"""Model Optimizer DMD2 objective for the native diffusion trainer."""

from __future__ import annotations

from math import ceil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
from torch import nn

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState, OptimizerState
from nemo_automodel.components.training.utils import (
    clip_grad_norm,
    prepare_after_first_microbatch,
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
)
from nemo_automodel.recipes.diffusion.train import (
    _build_diffusion_parallel_manager_args,
    _build_optimizer,
)

if TYPE_CHECKING:
    from nemo_automodel.components.config.loader import ConfigNode
    from nemo_automodel.recipes.diffusion.train import TrainDiffusionRecipe


def _require_qwen_fastgen() -> tuple[Any, Any, Any]:
    """Import ModelOpt only when Qwen-Image DMD2 is configured."""
    try:
        import modelopt.torch.fastgen as fastgen
        import modelopt.torch.fastgen.discriminators as fastgen_discriminators
        import modelopt.torch.fastgen.plugins.qwen_image as qwen_fastgen
    except ImportError as exc:
        raise ImportError("Qwen-Image DMD2 requires ModelOpt FastGen; install with `uv sync --extra dmd2`.") from exc
    return fastgen, fastgen_discriminators, qwen_fastgen


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


class DMD2Objective:
    """Add full Model Optimizer DMD2 updates to ``TrainDiffusionRecipe``.

    Model Optimizer owns CFG, VSD/DSM, multi-step simulation, GAN/R1, and EMA.
    This object only owns the auxiliary models, optimizer alternation, replicated
    discriminator synchronization, and their checkpoint state.
    """

    use_distributed_checkpointing = True

    def __init__(self, cfg: ConfigNode) -> None:
        self.cfg = cfg
        fastgen, _, _ = _require_qwen_fastgen()
        self.modelopt_config = fastgen.DMDConfig(**cfg.modelopt_config.to_dict())
        if self.modelopt_config.student_update_freq < 1:
            raise ValueError("dmd2.modelopt_config.student_update_freq must be at least 1.")

        self.modelopt_pipeline: Any | None = None
        self.teacher: nn.Module | None = None
        self.fake_score: nn.Module | None = None
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

        ema = self.modelopt_config.ema
        if ema is not None and (not ema.fsdp2 or ema.mode != "full_tensor"):
            raise ValueError("DMD2 checkpoint resume requires EMA with fsdp2=true and mode=full_tensor.")

    def primary_optimizer_steps(self, outer_steps: int) -> int:
        """Return the number of student updates in ``outer_steps``."""
        return ceil(outer_steps / int(self.modelopt_config.student_update_freq))

    def setup(self, recipe: TrainDiffusionRecipe) -> None:
        """Create DMD2 auxiliaries before the native checkpoint restore."""
        _, fastgen_discriminators, qwen_fastgen = _require_qwen_fastgen()

        negative_path = self.cfg.get("negative_prompt_embedding_path", None)
        if self.modelopt_config.guidance_scale is not None:
            if not negative_path:
                raise ValueError("DMD2 CFG requires dmd2.negative_prompt_embedding_path.")
            self.negative_prompt_embedding = _load_negative_prompt_embedding(str(negative_path)).to(
                recipe.device,
                dtype=recipe.compute_dtype,
            )

        parallel_scheme = {
            "transformer": _build_diffusion_parallel_manager_args(
                fsdp_cfg=recipe.cfg.get("fsdp", None),
                ddp_cfg=None,
                world_size=recipe.world_size,
                dtype=recipe.model_dtype,
                compute_dtype=recipe.compute_dtype,
                lora_enabled=False,
            )
        }
        self.teacher = self._load_transformer(recipe, parallel_scheme, trainable=False)
        self.fake_score = self._load_transformer(recipe, parallel_scheme, trainable=True)
        self.fake_score_optimizer = _build_optimizer(
            [parameter for parameter in self.fake_score.parameters() if parameter.requires_grad],
            recipe.cfg.get("optim.optimizer", {}),
            float(self.cfg.get("fake_score_lr", recipe.learning_rate)),
        )

        if self.modelopt_config.gan_loss_weight_gen > 0:
            discriminator_cfg = self.cfg.get("discriminator", None)
            if discriminator_cfg is None:
                raise ValueError("DMD2 GAN requires dmd2.discriminator arguments.")
            self.discriminator = fastgen_discriminators.Discriminator_ImageDiT(**discriminator_cfg.to_dict()).to(
                device=recipe.device,
                dtype=recipe.compute_dtype,
            )
            self._broadcast_discriminator(recipe)
            self.discriminator_optimizer = _build_optimizer(
                list(self.discriminator.parameters()),
                recipe.cfg.get("optim.optimizer", {}),
                float(self.cfg.get("discriminator_lr", recipe.learning_rate)),
            )

        self.modelopt_pipeline = qwen_fastgen.QwenImageDMDPipeline(
            student=recipe.model,
            teacher=self.teacher,
            fake_score=self.fake_score,
            config=self.modelopt_config,
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
        expected_student_steps = ceil(completed_steps / int(self.modelopt_config.student_update_freq))
        restored_student_steps = int(recipe.lr_scheduler[0].num_steps)
        if restored_student_steps != expected_student_steps:
            raise ValueError(
                "DMD2 checkpoint has an inconsistent student LR scheduler: "
                f"expected {expected_student_steps}, restored {restored_student_steps}."
            )

    def state_dict(self) -> dict[str, Any]:
        """Return DCP-compatible fake-score, discriminator, optimizer, and EMA state."""
        self._require_ready()
        assert self._fake_score_state is not None
        assert self._fake_score_optimizer_state is not None
        state = {
            "fake_score": self._fake_score_state.state_dict(),
            "fake_score_optimizer": self._fake_score_optimizer_state.state_dict(),
        }
        if self._discriminator_state is not None:
            state["discriminator"] = self._discriminator_state.state_dict()
        if self._discriminator_optimizer_state is not None:
            state["discriminator_optimizer"] = self._discriminator_optimizer_state.state_dict()
        if self.modelopt_pipeline.ema is not None:
            state["ema"] = self.modelopt_pipeline.ema.state_dict()
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore DMD2 auxiliary state populated by DCP."""
        self._require_ready()
        assert self._fake_score_state is not None
        assert self._fake_score_optimizer_state is not None
        self._fake_score_state.load_state_dict(state["fake_score"])
        self._fake_score_optimizer_state.load_state_dict(state["fake_score_optimizer"])
        if self._discriminator_state is not None:
            self._discriminator_state.load_state_dict(state["discriminator"])
        if self._discriminator_optimizer_state is not None:
            self._discriminator_optimizer_state.load_state_dict(state["discriminator_optimizer"])
        if self.modelopt_pipeline.ema is not None:
            self.modelopt_pipeline.ema.load_state_dict(state["ema"])

    def train_batch_group(
        self,
        recipe: TrainDiffusionRecipe,
        batch_group: list[dict[str, Any]],
        global_step: int,
    ) -> tuple[float, float]:
        """Run one student or fake-score/discriminator optimizer phase."""
        self._require_ready()
        assert self.fake_score is not None
        assert self.fake_score_optimizer is not None
        assert self.teacher is not None
        if not batch_group:
            raise RuntimeError("DMD2 received an empty gradient-accumulation group.")

        student_phase = global_step % int(self.modelopt_config.student_update_freq) == 0
        phase = "student" if student_phase else "fake_score"
        active_model = recipe.model if student_phase else self.fake_score
        active_optimizer = recipe.optimizer if student_phase else self.fake_score_optimizer
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

            latents, noise, text_embeddings, negative_embeddings = self._prepare_batch(recipe, micro_batch)
            kwargs = {"encoder_hidden_states": text_embeddings, "encoder_hidden_states_mask": None}
            if student_phase:
                losses = self.modelopt_pipeline.compute_student_loss(
                    latents,
                    noise,
                    negative_encoder_hidden_states=negative_embeddings,
                    **kwargs,
                )
            else:
                losses = self.modelopt_pipeline.compute_fake_score_loss(latents, noise, **kwargs)

            total = losses["total"]
            if recipe.check_loss and not torch.isfinite(total).all():
                raise FloatingPointError(f"Non-finite DMD2 {phase} loss at step {global_step}.")
            (total / num_microbatches).backward()
            reported_total = total.detach()

            if not student_phase and self.discriminator_optimizer is not None:
                discriminator_losses = self.modelopt_pipeline.compute_discriminator_loss(latents, noise, **kwargs)
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
        grad_norm = float(grad_norm) if torch.is_tensor(grad_norm) else float(grad_norm)

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
            self.modelopt_pipeline.update_ema(
                iteration=global_step // int(self.modelopt_config.student_update_freq) + 1
            )

        active_optimizer.zero_grad(set_to_none=True)
        if not student_phase and self.discriminator_optimizer is not None:
            self.discriminator_optimizer.zero_grad(set_to_none=True)

        return float(torch.stack(total_losses).mean().item()), grad_norm

    def _load_transformer(
        self,
        recipe: TrainDiffusionRecipe,
        parallel_scheme: dict[str, dict[str, Any]],
        *,
        trainable: bool,
    ) -> nn.Module:
        pipe, _ = NeMoAutoDiffusionPipeline.from_pretrained(
            recipe.model_id,
            torch_dtype=recipe.model_dtype,
            device=recipe.device,
            parallel_scheme=parallel_scheme,
            components_to_load=["transformer"],
            load_for_training=trainable,
            low_cpu_mem_usage=True,
            active_transformer=recipe.active_transformer,
            transformer_engine_linear=recipe.transformer_engine_linear,
            transformer_engine_fp8_safe_only=recipe.transformer_engine_fp8,
            fuse_qkv_projections=recipe.fuse_qkv_projections,
            compact_fused_qkv_projections=recipe.compact_fused_qkv_projections,
        )
        model = pipe.transformer
        if recipe.attention_backend is not None:
            getattr(model, "module", model).set_attention_backend(recipe.attention_backend)
        model.train(trainable)
        model.requires_grad_(trainable)
        return model

    def _prepare_batch(
        self,
        recipe: TrainDiffusionRecipe,
        batch: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        latents = batch["image_latents"].to(recipe.device, dtype=recipe.compute_dtype, non_blocking=True)
        text_embeddings = batch["text_embeddings"].to(
            recipe.device,
            dtype=recipe.compute_dtype,
            non_blocking=True,
        )
        if text_embeddings.ndim == 2:
            text_embeddings = text_embeddings.unsqueeze(0)
        if latents.ndim != 4 or text_embeddings.ndim != 3 or latents.shape[0] != text_embeddings.shape[0]:
            raise ValueError(
                "DMD2 expects image_latents [B,C,H,W] and text_embeddings [B,S,D]; "
                f"got {tuple(latents.shape)} and {tuple(text_embeddings.shape)}."
            )

        feature_shape = (latents.shape[-2], latents.shape[-1])
        if self.discriminator is not None and self._feature_capture_shape != feature_shape:
            _, _, qwen_fastgen = _require_qwen_fastgen()

            qwen_fastgen.attach_feature_capture(
                self.teacher,
                feature_indices=sorted(self.discriminator.feature_indices),
                h_lat=feature_shape[0],
                w_lat=feature_shape[1],
            )
            self._feature_capture_shape = feature_shape

        negative = None
        if self.negative_prompt_embedding is not None:
            if self.negative_prompt_embedding.shape[-1] != text_embeddings.shape[-1]:
                raise ValueError("DMD2 positive and negative text embedding dimensions must match.")
            negative = self.negative_prompt_embedding.unsqueeze(0).expand(latents.shape[0], -1, -1)
        return latents, torch.randn_like(latents), text_embeddings, negative

    def _set_phase(self, recipe: TrainDiffusionRecipe, student_phase: bool) -> None:
        recipe.model.train(student_phase)
        recipe.model.requires_grad_(student_phase)
        self.fake_score.train(not student_phase)
        self.fake_score.requires_grad_(not student_phase)
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
        if self.modelopt_pipeline is None or self.fake_score is None or self.fake_score_optimizer is None:
            raise RuntimeError("DMD2 objective has not been set up.")
