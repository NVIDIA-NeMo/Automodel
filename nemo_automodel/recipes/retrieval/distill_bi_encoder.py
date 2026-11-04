# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import logging
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.checkpoint.utils import find_latest_checkpoint, resolve_restore_from_to_checkpoint_dir
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.datasets.llm.retrieval_types import PASSAGE_MODALITY_NAMES
from nemo_automodel.components.distributed.tp_replicas import synchronize_tp_replica_gradients
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.embedding_distill import EmbeddingDistillLoss, EmbeddingMSELoss, ScoreDistillLoss
from nemo_automodel.components.loss.infonce import InfoNCEDistillLoss, InfoNCELoss
from nemo_automodel.components.loss.intermediate_distill import IntermediateDistillLoss
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe, _append_modality_loss_stats

logger = logging.getLogger(__name__)


def _build_or_none(cfg_section):
    if cfg_section is None:
        return None
    if hasattr(cfg_section, "instantiate"):
        return cfg_section.instantiate()
    return cfg_section


def _cfg_get_path(cfg, path: str, default=None):
    """Read either OmegaConf-style dotted keys or nested dict/config values."""
    try:
        value = cfg.get(path, None)
        if value is not None:
            return value
    except (AttributeError, KeyError, TypeError):
        pass

    cur = cfg
    for part in path.split("."):
        if cur is None:
            return default
        try:
            cur = cur.get(part, None)
        except (AttributeError, KeyError, TypeError):
            cur = getattr(cur, part, None)
    return default if cur is None else cur


def _get_optimizer_field(cfg, field_name: str, default=None):
    optimizer = getattr(cfg, "__dict__", {}).get("optimizer", None)
    optimizer_kwargs = getattr(optimizer, "kwargs", None)
    if isinstance(optimizer_kwargs, dict) and field_name in optimizer_kwargs:
        return optimizer_kwargs[field_name]
    if optimizer is not None and hasattr(optimizer, field_name):
        return getattr(optimizer, field_name)
    return _cfg_get_path(cfg, f"optimizer.{field_name}", default)


def _pop_optimizer_recipe_field(cfg, field_name: str, default=None):
    value = _cfg_get_path(cfg, f"optimizer.{field_name}", None)

    raw_cfg = getattr(cfg, "_raw", None)
    raw_optimizer = raw_cfg.get("optimizer", None) if raw_cfg is not None else None
    raw_dict = getattr(raw_optimizer, "__dict__", None)
    if isinstance(raw_dict, dict) and field_name in raw_dict:
        value = raw_dict.pop(field_name)
    raw_config = getattr(raw_optimizer, "_raw_config", None)
    if isinstance(raw_config, dict):
        raw_config.pop(field_name, None)

    optimizer = getattr(cfg, "__dict__", {}).get("optimizer", None)
    optimizer_kwargs = getattr(optimizer, "kwargs", None)
    if isinstance(optimizer, dict) and field_name in optimizer:
        value = optimizer.pop(field_name)
    elif isinstance(optimizer_kwargs, dict) and field_name in optimizer_kwargs:
        value = optimizer_kwargs.pop(field_name)
    elif optimizer is not None and hasattr(optimizer, field_name):
        value = getattr(optimizer, field_name)
        try:
            delattr(optimizer, field_name)
        except AttributeError:
            pass

    return default if value is None else value


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _unpack_prefixed_batch(
    batch: dict[str, Any],
    prefix: str,
    *,
    exclude: set[str] | None = None,
) -> dict[str, Any] | None:
    exclude = exclude or set()
    unpacked = {
        key[len(prefix) :]: value
        for key, value in batch.items()
        if key.startswith(prefix) and key[len(prefix) :] not in exclude
    }
    return unpacked or None


def _flatten_negative_batch(n_batch: dict[str, Any], bsz: int, num_neg: int) -> dict[str, Any]:
    flat_n_batch = {}
    for key, value in n_batch.items():
        if isinstance(value, torch.Tensor) and value.shape[:2] == (bsz, num_neg):
            flat_n_batch[key] = value.reshape(bsz * num_neg, *value.shape[2:])
        else:
            flat_n_batch[key] = value
    return flat_n_batch


def _unpack_qpn(batch: dict[str, Any]):
    q_batch = _unpack_prefixed_batch(batch, "q_")
    d_batch = _unpack_prefixed_batch(batch, "d_")
    if q_batch is None or d_batch is None:
        raise KeyError("Distillation batches must include q_* and d_* model inputs")

    if "n_input_ids" not in batch or "n_attention_mask" not in batch or "n_mask" not in batch:
        return q_batch, d_batch, None, None

    n_batch = _unpack_prefixed_batch(batch, "n_", exclude={"mask"})
    if n_batch is None:
        return q_batch, d_batch, None, None

    n_input_ids = n_batch.get("input_ids")
    n_mask = batch["n_mask"]

    if not isinstance(n_input_ids, torch.Tensor) or n_input_ids.dim() != 3:
        return q_batch, d_batch, None, None

    bsz, num_neg, seq_len = n_input_ids.shape
    if num_neg == 0 or seq_len == 0 or int(n_mask.sum().item()) == 0:
        return q_batch, d_batch, None, None

    flat_n_batch = _flatten_negative_batch(n_batch, bsz, num_neg)
    return q_batch, d_batch, flat_n_batch, n_mask


def _set_dummy_vision_flags(query: dict[str, Any] | None, doc: dict[str, Any] | None) -> None:
    if query is not None:
        query["run_dummy_vision"] = False
    if doc is not None:
        doc["run_dummy_vision"] = True


def _as_student_output(output) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    if not isinstance(output, tuple) or len(output) != 3:
        raise RuntimeError(
            "EmbeddingDistillRecipe expects the student model to return "
            "(pooled, projected, intermediate_outputs). Use RetrieverStudentWithProjection.build."
        )
    return output


def _as_teacher_output(output) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    if isinstance(output, tuple) and len(output) == 2:
        return output
    if isinstance(output, torch.Tensor):
        return output.float(), {}
    raise RuntimeError(
        "EmbeddingDistillRecipe expects the teacher model to return "
        "(pooled, intermediate_outputs). Use RetrieverTeacherEmbeddingEncoder.build."
    )


def _validate_distill_shapes(student: torch.Tensor, teacher: torch.Tensor) -> None:
    if student.shape != teacher.shape:
        raise RuntimeError(
            "Student and teacher distillation embedding shapes do not match: "
            f"student={tuple(student.shape)}, teacher={tuple(teacher.shape)}"
        )


def _repeat_queries_to_match_docs(
    student_query: torch.Tensor,
    teacher_query: torch.Tensor,
    student_doc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_query.shape[0] == student_doc.shape[0]:
        return student_query, teacher_query
    if student_doc.shape[0] % student_query.shape[0] != 0:
        raise RuntimeError(
            f"Cannot align query/doc distillation batches: query={student_query.shape[0]}, doc={student_doc.shape[0]}"
        )
    repeats = student_doc.shape[0] // student_query.shape[0]
    return student_query.repeat_interleave(repeats, dim=0), teacher_query.repeat_interleave(repeats, dim=0)


def _split_grouped_passages(
    reps: torch.Tensor,
    *,
    batch_size: int,
    n_passages: int,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if reps.dim() != 2:
        raise RuntimeError(f"{label} grouped-passage distillation requires 2-D embeddings, got {tuple(reps.shape)}")
    if reps.shape[0] == batch_size:
        return reps, None
    if reps.shape[0] != batch_size * n_passages:
        raise RuntimeError(
            f"Cannot reshape {label} passages: passages={reps.shape[0]}, queries={batch_size}, n_passages={n_passages}"
        )
    grouped = reps.reshape(batch_size, n_passages, reps.shape[-1])
    positives = grouped[:, 0, :]
    negatives = grouped[:, 1:, :] if n_passages > 1 else None
    return positives, negatives


def _mean_grouped_docs(reps: torch.Tensor, *, batch_size: int, n_passages: int, label: str) -> torch.Tensor:
    if reps.dim() != 1:
        raise RuntimeError(f"{label} per-sample loss must be 1-D, got {tuple(reps.shape)}")
    if reps.shape[0] == batch_size:
        return reps
    if reps.shape[0] != batch_size * n_passages:
        raise RuntimeError(
            f"Cannot reshape {label} passages: passages={reps.shape[0]}, queries={batch_size}, n_passages={n_passages}"
        )
    return reps.reshape(batch_size, n_passages).mean(dim=1)


def _embedding_distill_loss_per_query(
    *,
    s_q_proj: torch.Tensor,
    t_q_pool: torch.Tensor,
    s_d_proj: torch.Tensor,
    t_d_pool: torch.Tensor,
    n_passages: int,
    loss_weights: dict[str, float],
) -> torch.Tensor:
    batch_size = s_q_proj.shape[0]
    per_query = s_q_proj.new_zeros(batch_size)

    if loss_weights["distill"] > 0:
        q_cos = 1.0 - (F.normalize(s_q_proj.float(), dim=-1) * F.normalize(t_q_pool.float(), dim=-1)).sum(dim=-1)
        d_cos = 1.0 - (F.normalize(s_d_proj.float(), dim=-1) * F.normalize(t_d_pool.float(), dim=-1)).sum(dim=-1)
        per_query = per_query + loss_weights["distill"] * (
            q_cos + _mean_grouped_docs(d_cos, batch_size=batch_size, n_passages=n_passages, label="doc cosine")
        )

    if loss_weights["mse"] > 0:
        q_mse = (s_q_proj.float() - t_q_pool.float()).pow(2).mean(dim=-1)
        d_mse = (s_d_proj.float() - t_d_pool.float()).pow(2).mean(dim=-1)
        per_query = per_query + loss_weights["mse"] * (
            q_mse + _mean_grouped_docs(d_mse, batch_size=batch_size, n_passages=n_passages, label="doc mse")
        )

    if loss_weights["normalized_mse"] > 0:
        q_mse = (F.normalize(s_q_proj.float(), dim=-1) - F.normalize(t_q_pool.float(), dim=-1)).pow(2).mean(dim=-1)
        d_mse = (F.normalize(s_d_proj.float(), dim=-1) - F.normalize(t_d_pool.float(), dim=-1)).pow(2).mean(dim=-1)
        per_query = per_query + loss_weights["normalized_mse"] * (
            q_mse + _mean_grouped_docs(d_mse, batch_size=batch_size, n_passages=n_passages, label="doc normalized mse")
        )

    return per_query


def _count_trainable_parameters(module: torch.nn.Module | None) -> tuple[int, int]:
    if module is None:
        return 0, 0
    total = 0
    trainable = 0
    for param in module.parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
    return trainable, total


def _get_nested_module(model: torch.nn.Module, attr_name: str) -> torch.nn.Module | None:
    model = getattr(model, "module", model)
    candidates = (
        model,
        getattr(model, "model", None),
        getattr(model, "student", None),
        getattr(getattr(model, "student", None), "model", None),
    )
    for candidate in candidates:
        if candidate is not None and hasattr(candidate, attr_name):
            return getattr(candidate, attr_name)
    return None


def _dp_group_src_rank(group) -> int:
    if group is None:
        return 0
    try:
        return torch.distributed.get_process_group_ranks(group)[0]
    except Exception:
        return 0


class EmbeddingDistillRecipe(TrainBiEncoderRecipe):
    """Recipe for Stage-1 embedding distillation on bi-encoder backbones."""

    def setup(self):
        super().setup()

        model_target = ""
        model_cfg = self.cfg.get("model", None)
        if model_cfg is not None:
            model_target = model_cfg.get("_target_", "")
        if not isinstance(model_target, str):
            model_target = getattr(model_target, "__qualname__", str(model_target))

        if "RetrieverStudentWithProjection" not in model_target:
            logger.warning(
                "EmbeddingDistillRecipe expects model to be RetrieverStudentWithProjection; got %s",
                model_target or "<unknown>",
            )

        self.teacher_embeddings_cache = _cfg_get_path(self.cfg, "dataloader.collate_fn.teacher_embeddings_cache", None)
        if self.teacher_embeddings_cache is None:
            self.teacher_embeddings_cache = _cfg_get_path(self.cfg, "teacher_embeddings_cache", None)
        self.use_cached_teacher = bool(self.teacher_embeddings_cache)

        self.teacher_model = None
        if self.use_cached_teacher:
            logger.info(
                "Using cached teacher embeddings from %s; live teacher model will not be built.",
                self.teacher_embeddings_cache,
            )
        else:
            with torch.random.fork_rng(enabled=False):
                self.teacher_model = self.cfg.teacher_model.instantiate(
                    distributed_setup=self.distributed_setup,
                    peft_config=None,
                )
            self.teacher_model.eval()
            for param in self.teacher_model.parameters():
                param.requires_grad_(False)

        self.distill_loss = _build_or_none(self.cfg.get("distill_loss", None))
        self.mse_loss = _build_or_none(self.cfg.get("mse_loss", None))
        self.normalized_mse_loss = _build_or_none(self.cfg.get("normalized_mse_loss", None))
        self.score_loss = _build_or_none(self.cfg.get("score_loss", None))
        self.intermediate_loss = _build_or_none(self.cfg.get("intermediate_loss", None))
        self.infonce_loss = _build_or_none(self.cfg.get("infonce_loss", None))
        self.infonce_distill_loss = _build_or_none(self.cfg.get("infonce_distill_loss", None))

        self.loss_weights = {
            "distill": float(self.cfg.get("distill_loss_weight", 1.0)),
            "mse": float(self.cfg.get("mse_loss_weight", 0.0)),
            "normalized_mse": float(self.cfg.get("normalized_mse_loss_weight", 0.0)),
            "score": float(self.cfg.get("score_loss_weight", 0.0)),
            "intermediate": float(self.cfg.get("intermediate_loss_weight", 0.0)),
            "nce": float(self.cfg.get("nce_loss_weight", 0.0)),
            "nce_kd": float(self.cfg.get("nce_distill_loss_weight", 0.0)),
        }

        # Parse layer_pairs before building loss defaults so the default
        # IntermediateDistillLoss can be wired with the same pairs used to attach
        # the capture hooks below.
        self.layer_pairs = [tuple(int(x) for x in pair) for pair in self.cfg.get("layer_pairs", [])]
        if self.loss_weights["intermediate"] > 0 and not self.layer_pairs:
            raise ValueError("intermediate_loss_weight > 0 requires non-empty layer_pairs in config")

        if self.distill_loss is None:
            self.distill_loss = EmbeddingDistillLoss(reduction="mean")
        if self.mse_loss is None:
            self.mse_loss = EmbeddingMSELoss(normalize=False, reduction="mean")
        if self.normalized_mse_loss is None:
            self.normalized_mse_loss = EmbeddingMSELoss(normalize=True, reduction="mean")
        if self.score_loss is None:
            self.score_loss = ScoreDistillLoss(temperature=float(self.cfg.get("score_temperature", 0.02)))
        if self.intermediate_loss is None:
            self.intermediate_loss = IntermediateDistillLoss(layer_pairs=self.layer_pairs)
        if self.infonce_loss is None:
            self.infonce_loss = InfoNCELoss()
        if self.infonce_distill_loss is None:
            self.infonce_distill_loss = InfoNCEDistillLoss()

        # Guard against a configured intermediate loss whose layer_pairs are empty
        # while the weight is active: hooks would fire but the loss would silently
        # short-circuit to zero (contributing no gradient).
        if self.loss_weights["intermediate"] > 0:
            loss_pairs = getattr(self.intermediate_loss, "layer_pairs", None)
            if not loss_pairs:
                raise ValueError(
                    "intermediate_loss_weight > 0 but intermediate_loss has empty layer_pairs; "
                    "set layer_pairs on the intermediate_loss config or omit it to inherit the top-level layer_pairs."
                )
            # The capture hooks below are attached only for the top-level
            # layer_pairs. Any layer the loss reads that is not captured would
            # raise at lookup time, so require the loss pairs to reference only
            # captured student/teacher layers.
            captured_student = {s for s, _ in self.layer_pairs}
            captured_teacher = {t for _, t in self.layer_pairs}
            missing_student = sorted({s for s, _ in loss_pairs} - captured_student)
            missing_teacher = sorted({t for _, t in loss_pairs} - captured_teacher)
            if missing_student or missing_teacher:
                raise ValueError(
                    "intermediate_loss.layer_pairs reference layers that the top-level layer_pairs do not "
                    f"capture (missing student layers={missing_student}, teacher layers={missing_teacher}); "
                    "align intermediate_loss.layer_pairs with the top-level layer_pairs or omit it to inherit them."
                )
        if self.use_cached_teacher and self.loss_weights["intermediate"] > 0:
            raise ValueError(
                "intermediate_loss_weight > 0 is not supported with cached teacher embeddings; "
                "cached teacher stores pooled embeddings only. Disable intermediate loss or use a live teacher."
            )
        if self.loss_weights["intermediate"] > 0:
            student_layers = sorted({s for s, _ in self.layer_pairs})
            teacher_layers = sorted({t for _, t in self.layer_pairs})
            if hasattr(self.model_parts[0], "attach_intermediate_capture"):
                self.model_parts[0].attach_intermediate_capture(student_layers)
            if self.teacher_model is not None and hasattr(self.teacher_model, "attach_intermediate_capture"):
                self.teacher_model.attach_intermediate_capture(teacher_layers)

        self._sync_projection_parameters()
        self._apply_student_lr_overrides()
        self._log_trainable_parameter_summary()

    def _projection_parameters(self) -> list[torch.nn.Parameter]:
        model = self.model_parts[0]
        if isinstance(model, DistributedDataParallel):
            model = model.module
        projection = getattr(model, "projection", None)
        if projection is None:
            return []
        return [param for param in projection.parameters() if param.requires_grad]

    def _sync_projection_parameters(self) -> None:
        """Keep the rank-local projection head replicated across DP ranks."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        params = self._projection_parameters()
        if not params:
            return
        dp_group = self._get_dp_group(include_cp=True)
        dp_size = self._get_dp_group_size(include_cp=True)
        if dp_size <= 1:
            return
        src_rank = _dp_group_src_rank(dp_group)
        for param in params:
            torch.distributed.broadcast(param.data, src=src_rank, group=dp_group)

    def _sync_projection_gradients(self) -> None:
        """Average projection gradients that are outside FSDP/DDP wrapping."""
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        model = self.model_parts[0]
        if isinstance(model, DistributedDataParallel):
            return
        params = self._projection_parameters()
        if not params:
            return
        dp_group = self._get_dp_group(include_cp=True)
        dp_size = self._get_dp_group_size(include_cp=True)
        if dp_size <= 1:
            return
        for param in params:
            if param.grad is None:
                param.grad = torch.zeros_like(param.data)
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            param.grad.div_(dp_size)

    def _build_optimizer_param_groups(self) -> list[dict[str, Any]]:
        """Build optimizer groups with projection params isolated before checkpoint restore."""
        model = self.model_parts[0]
        decay_params = []
        no_decay_params = []
        proj_params = []
        optimizer_lr = float(_get_optimizer_field(self.cfg, "lr", 1.0e-4))
        distill_projection_lr = _pop_optimizer_recipe_field(self.cfg, "distill_projection_lr", None)
        self.student_lr_overrides = {
            "student_backbone_lr": _pop_optimizer_recipe_field(
                self.cfg, "student_backbone_lr", self.cfg.get("student_backbone_lr", None)
            ),
            "vision_tower_lr": _pop_optimizer_recipe_field(
                self.cfg, "vision_tower_lr", self.cfg.get("vision_tower_lr", None)
            ),
            "multi_modal_projector_lr": _pop_optimizer_recipe_field(
                self.cfg, "multi_modal_projector_lr", self.cfg.get("multi_modal_projector_lr", None)
            ),
            "language_model_lr": _pop_optimizer_recipe_field(
                self.cfg, "language_model_lr", self.cfg.get("language_model_lr", None)
            ),
        }

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "projection." in name:
                proj_params.append(param)
                continue
            name_l = name.lower()
            if name.endswith(".bias") or "norm" in name_l:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0, "wd_mult": 0.0})
        if proj_params:
            group = {"params": proj_params, "weight_decay": 0.0, "wd_mult": 0.0}
            if distill_projection_lr is not None:
                distill_projection_lr = float(distill_projection_lr)
                group["lr_mult"] = distill_projection_lr / optimizer_lr
                logger.info(
                    "Configured distill projection lr group: base_lr=%s distill_projection_lr=%s lr_mult=%s",
                    optimizer_lr,
                    distill_projection_lr,
                    group["lr_mult"],
                )
            param_groups.append(group)

        assert param_groups, "no trainable parameters found"
        logger.info(
            "Optimizer param groups: decay=%d, no_decay=%d, projection=%d",
            len(decay_params),
            len(no_decay_params),
            len(proj_params),
        )
        return param_groups

    def _apply_student_lr_overrides(self) -> None:
        """Apply module-specific absolute LRs while preserving decay/no-decay groups."""
        if not self.optimizer:
            return

        student_model = self.model_parts[0]
        lr_overrides = getattr(self, "student_lr_overrides", {})
        backbone_lr = lr_overrides.get("student_backbone_lr", self.cfg.get("student_backbone_lr", None))
        group_specs = (
            (
                "vision_tower",
                _get_nested_module(student_model, "vision_tower"),
                lr_overrides.get("vision_tower_lr", self.cfg.get("vision_tower_lr", backbone_lr)),
            ),
            (
                "multi_modal_projector",
                _get_nested_module(student_model, "multi_modal_projector"),
                lr_overrides.get("multi_modal_projector_lr", self.cfg.get("multi_modal_projector_lr", None)),
            ),
            (
                "language_model",
                _get_nested_module(student_model, "language_model"),
                lr_overrides.get("language_model_lr", self.cfg.get("language_model_lr", backbone_lr)),
            ),
        )

        opt = self.optimizer[0]
        base_lr = float(opt.param_groups[0]["lr"])

        for label, module, lr in group_specs:
            if lr is None or module is None:
                continue
            params = []
            seen = set()
            for param in module.parameters():
                if not param.requires_grad or id(param) in seen:
                    continue
                params.append(param)
                seen.add(id(param))
            if not params:
                logger.info("Skipping %s lr group because it has no trainable parameters", label)
                continue

            param_ids = {id(param) for param in params}
            override_groups = []
            removed = 0
            for group in opt.param_groups:
                matched_params = []
                remaining_params = []
                for param in group["params"]:
                    if id(param) in param_ids:
                        matched_params.append(param)
                    else:
                        remaining_params.append(param)
                if not matched_params:
                    continue

                group["params"] = remaining_params
                removed += len(matched_params)

                override_group = {key: value for key, value in group.items() if key != "params"}
                override_group["params"] = matched_params
                override_group["lr"] = float(lr)
                override_group["lr_mult"] = float(lr) / base_lr
                if override_group.get("weight_decay", None) == 0.0:
                    override_group["wd_mult"] = 0.0
                override_groups.append(override_group)

            for group in override_groups:
                opt.add_param_group(group)
            logger.info(
                "Configured optimizer lr group for %s: lr=%s params=%d groups=%d moved_from_existing_groups=%d",
                label,
                float(lr),
                len(params),
                len(override_groups),
                removed,
            )

    def _log_trainable_parameter_summary(self) -> None:
        student_model = self.model_parts[0]
        for label, module in (
            ("vision_tower", _get_nested_module(student_model, "vision_tower")),
            ("multi_modal_projector", _get_nested_module(student_model, "multi_modal_projector")),
            ("language_model", _get_nested_module(student_model, "language_model")),
            ("projection", getattr(getattr(student_model, "module", student_model), "projection", None)),
        ):
            trainable, total = _count_trainable_parameters(module)
            logger.info("Trainable parameter summary: %s trainable=%d total=%d", label, trainable, total)

    def load_checkpoint(self, restore_from: str | None = None):
        super().load_checkpoint(restore_from)

        if not self.checkpointer.config.enabled:
            return

        if restore_from:
            ckpt_dir = resolve_restore_from_to_checkpoint_dir(self.checkpointer.config.checkpoint_dir, restore_from)
        else:
            ckpt_dir = find_latest_checkpoint(self.checkpointer.config.checkpoint_dir)
        if ckpt_dir is None:
            return

        projection_path = Path(ckpt_dir) / "projection.pt"
        if not projection_path.is_file():
            return

        model = self.model_parts[0]
        if isinstance(model, DistributedDataParallel):
            model = model.module
        projection = getattr(model, "projection", None)
        if projection is None:
            return

        state = torch.load(projection_path, map_location="cpu", weights_only=True)
        projection_state = {"weight": state["weight"], "bias": state["bias"]}
        projection.load_state_dict(projection_state, strict=True)
        projection.to(device=projection.weight.device, dtype=projection.weight.dtype)
        logger.info("Restored distillation projection head from %s", projection_path)

    def _extract_scoring_reps(self, model_output):
        """Select the pooled student embedding for validation scoring.

        ``RetrieverStudentWithProjection.forward`` returns
        ``(pooled, projected, intermediate_outputs)``. The pooled embedding is the student's
        native retrieval representation (and what training's InfoNCE terms score with), so it
        is what the inherited validation loop should compare against.
        """
        if isinstance(model_output, tuple):
            return model_output[0]
        return model_output

    def _forward_backward_step(
        self,
        idx,
        batch,
        *,
        loss_buffer,
        num_batches,
        is_train: bool = True,
        modality_loss_buffers: dict[str, list[torch.Tensor]] | None = None,
    ):
        uses_hard_negatives = self.loss_weights["nce"] > 0 or self.loss_weights["nce_kd"] > 0
        if not uses_hard_negatives:
            batch = {key: value for key, value in batch.items() if not key.startswith("n_")}

        batch = _move_to_device(batch, self.dist_env.device)
        q_batch, d_batch, n_batch, n_mask = _unpack_qpn(batch)

        student_model = self.model_parts[0]
        train_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if torch.cuda.is_available() else nullcontext()
        sync_ctx = (
            get_sync_ctx(
                student_model,
                idx == num_batches - 1,
                defer_fsdp_grad_sync=getattr(self.distributed_config, "defer_fsdp_grad_sync", True),
            )
            if is_train
            else nullcontext()
        )

        with train_ctx, sync_ctx:
            if is_train:
                _set_dummy_vision_flags(q_batch, d_batch)

            s_q_pool, s_q_proj, s_q_inter = _as_student_output(student_model(q_batch))
            s_d_pool, s_d_proj, s_d_inter = _as_student_output(student_model(d_batch))

            s_n_pool = None
            if n_batch is not None and uses_hard_negatives:
                bsz, num_neg = n_mask.shape
                s_n_pool, _, _ = _as_student_output(student_model(n_batch))
                s_n_pool = s_n_pool.view(bsz, num_neg, s_n_pool.shape[-1])

            if self.use_cached_teacher:
                if "t_q_pool" not in batch or "t_d_pool" not in batch:
                    raise RuntimeError(
                        "Cached teacher mode is enabled, but the batch does not contain "
                        "'t_q_pool'/'t_d_pool'. Set dataloader.collate_fn.teacher_embeddings_cache "
                        "to a valid cache directory."
                    )
                t_q_pool = batch["t_q_pool"].float()
                t_d_pool = batch["t_d_pool"].float()
                t_q_inter = {}
                t_d_inter = {}
                t_n_pool = None
                if n_mask is not None and self.loss_weights["nce_kd"] > 0:
                    if "t_n_pool" not in batch:
                        raise RuntimeError(
                            "Cached InfoNCE distillation requires batch['t_n_pool']; "
                            "set dataloader.collate_fn.teacher_embeddings_cache."
                        )
                    t_n_pool = batch["t_n_pool"].float()
                if (
                    self.loss_weights["distill"] > 0
                    or self.loss_weights["mse"] > 0
                    or self.loss_weights["normalized_mse"] > 0
                ) and (t_q_pool.shape[-1] != s_q_proj.shape[-1] or t_d_pool.shape[-1] != s_d_proj.shape[-1]):
                    raise RuntimeError(
                        "Cached teacher embedding dim does not match student projection dim: "
                        f"t_q={tuple(t_q_pool.shape)}, s_q_proj={tuple(s_q_proj.shape)}, "
                        f"t_d={tuple(t_d_pool.shape)}, s_d_proj={tuple(s_d_proj.shape)}. "
                        "Set model.teacher_hidden_size to the cache embedding dimension."
                    )
            else:
                with torch.no_grad():
                    t_q_pool, t_q_inter = _as_teacher_output(self.teacher_model(q_batch))
                    t_d_pool, t_d_inter = _as_teacher_output(self.teacher_model(d_batch))
                    t_n_pool = None
                    if n_batch is not None and self.loss_weights["nce_kd"] > 0:
                        bsz, num_neg = n_mask.shape
                        t_n_pool, _ = _as_teacher_output(self.teacher_model(n_batch))
                        t_n_pool = t_n_pool.view(bsz, num_neg, t_n_pool.shape[-1])

            zero = s_q_proj.new_zeros(())
            _validate_distill_shapes(s_q_proj, t_q_pool)
            _validate_distill_shapes(s_d_proj, t_d_pool)
            s_q_proj_pair, t_q_pool_pair = _repeat_queries_to_match_docs(s_q_proj, t_q_pool, s_d_proj)

            l_distill = (
                self.distill_loss(s_q_proj_pair, t_q_pool_pair, s_d_proj, t_d_pool)
                if self.loss_weights["distill"] > 0
                else zero
            )
            l_mse = (
                self.mse_loss(s_q_proj_pair, t_q_pool_pair, s_d_proj, t_d_pool)
                if self.loss_weights["mse"] > 0
                else zero
            )
            l_normalized_mse = (
                self.normalized_mse_loss(s_q_proj_pair, t_q_pool_pair, s_d_proj, t_d_pool)
                if self.loss_weights["normalized_mse"] > 0
                else zero
            )

            s_pos_pool, s_grouped_neg_pool = _split_grouped_passages(
                s_d_pool,
                batch_size=s_q_pool.shape[0],
                n_passages=self.train_n_passages,
                label="student pooled",
            )
            t_pos_pool, t_grouped_neg_pool = _split_grouped_passages(
                t_d_pool,
                batch_size=t_q_pool.shape[0],
                n_passages=self.train_n_passages,
                label="teacher pooled",
            )
            l_score = (
                self.score_loss(s_q_pool, t_q_pool, s_pos_pool, t_pos_pool) if self.loss_weights["score"] > 0 else zero
            )

            if self.loss_weights["intermediate"] > 0:
                l_inter = self.intermediate_loss(
                    s_q_inter,
                    t_q_inter,
                    s_d_inter,
                    t_d_inter,
                    attn_q=q_batch["attention_mask"],
                    attn_d=d_batch["attention_mask"],
                    projector=getattr(student_model, "projection", None),
                )
            else:
                l_inter = zero

            if self.loss_weights["nce"] > 0:
                nce_docs = s_pos_pool
                nce_hard_negatives = s_n_pool if s_n_pool is not None else s_grouped_neg_pool
                nce_mask = n_mask
                l_nce = self.infonce_loss(
                    s_q_pool,
                    nce_docs,
                    hard_negatives=nce_hard_negatives,
                    hard_negatives_mask=nce_mask,
                )
            else:
                l_nce = zero

            if self.loss_weights["nce_kd"] > 0:
                nce_kd_docs = s_pos_pool
                nce_kd_teacher_docs = t_pos_pool
                nce_kd_hard_negatives = s_n_pool if s_n_pool is not None else s_grouped_neg_pool
                nce_kd_teacher_hard_negatives = t_n_pool if t_n_pool is not None else t_grouped_neg_pool
                l_nce_kd = self.infonce_distill_loss(
                    s_q_pool,
                    nce_kd_docs,
                    t_q_pool,
                    nce_kd_teacher_docs,
                    student_hard_negatives=nce_kd_hard_negatives,
                    teacher_hard_negatives=nce_kd_teacher_hard_negatives,
                    hard_negatives_mask=n_mask,
                )
            else:
                l_nce_kd = zero

            total_loss = (
                self.loss_weights["distill"] * l_distill
                + self.loss_weights["mse"] * l_mse
                + self.loss_weights["normalized_mse"] * l_normalized_mse
                + self.loss_weights["score"] * l_score
                + self.loss_weights["intermediate"] * l_inter
                + self.loss_weights["nce"] * l_nce
                + self.loss_weights["nce_kd"] * l_nce_kd
            )

            loss_buffer.append(total_loss.detach())
            if modality_loss_buffers is not None:
                per_query_loss = _embedding_distill_loss_per_query(
                    s_q_proj=s_q_proj,
                    t_q_pool=t_q_pool,
                    s_d_proj=s_d_proj,
                    t_d_pool=t_d_pool,
                    n_passages=self.train_n_passages,
                    loss_weights=self.loss_weights,
                )
                _append_modality_loss_stats(batch, per_query_loss, self.train_n_passages, modality_loss_buffers)
            self._loss_component_buffer.append(
                {
                    "loss_distill": l_distill.detach(),
                    "loss_mse": l_mse.detach(),
                    "loss_normalized_mse": l_normalized_mse.detach(),
                    "loss_score": l_score.detach(),
                    "loss_intermediate": l_inter.detach(),
                    "loss_nce": l_nce.detach(),
                    "loss_retrieval": l_nce.detach(),
                    "loss_nce_distill": l_nce_kd.detach(),
                }
            )

            if is_train:
                (total_loss / num_batches).backward()

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        self._loss_component_buffer: list[dict[str, torch.Tensor]] = []

        loss_buffer = []
        modality_loss_buffers = (
            {name: [] for name in PASSAGE_MODALITY_NAMES.values()} if self.log_modality_loss else None
        )
        for idx, batch in enumerate(batches):
            self._forward_backward_step(
                idx,
                batch,
                loss_buffer=loss_buffer,
                num_batches=len(batches),
                is_train=True,
                modality_loss_buffers=modality_loss_buffers,
            )

        self._sync_projection_gradients()

        synchronize_tp_replica_gradients(self.model_parts, self.device_mesh)
        grad_norm = scale_grads_and_clip_grad_norm(
            max_grad_norm,
            self.model_parts,
            norm_type=2.0,
            pp_enabled=self.pp_enabled,
            device_mesh=self.device_mesh,
            moe_mesh=self.moe_mesh,
            ep_axis_name="ep" if self.moe_mesh is not None and "ep" in self.moe_mesh.mesh_dim_names else None,
            pp_axis_name="pp" if self.pp_enabled else None,
            foreach=True,
            num_label_tokens=None,
            dp_group_size=self._get_dp_group_size(include_cp=True),
        )

        self.checkpointer.maybe_wait_for_staging()
        lr = self.optimizer[0].param_groups[0]["lr"]
        for opt in self.optimizer:
            opt.step()
            opt.zero_grad()

        if self.lr_scheduler is not None:
            for scheduler in self.lr_scheduler:
                scheduler.step(1)

        reporting_loss = torch.mean(torch.stack(loss_buffer))
        if torch.distributed.is_initialized():
            reporting_loss = self._dp_allreduce(reporting_loss, include_cp=True)
            reporting_loss = reporting_loss / self._get_dp_group_size(include_cp=True)
        reporting_loss_val = reporting_loss.cpu().item()
        self.loss_average_window.append(reporting_loss_val)
        average_loss = sum(self.loss_average_window) / len(self.loss_average_window)

        component_means: dict[str, float] = {}
        if self._loss_component_buffer:
            keys = self._loss_component_buffer[0].keys()
            for key in keys:
                value = torch.mean(torch.stack([item[key] for item in self._loss_component_buffer]))
                if torch.distributed.is_initialized():
                    value = self._dp_allreduce(value, include_cp=True)
                    value = value / self._get_dp_group_size(include_cp=True)
                component_means[key] = value.cpu().item()

        elapsed = time.perf_counter() - self.timestamp
        self.timestamp = time.perf_counter()
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

        metrics = {
            "loss": reporting_loss_val,
            "loss_avg_window": average_loss,
            "grad_norm": grad_norm,
            "lr": lr,
            "mem": mem_allocated,
            "time_per_step": elapsed,
        }
        metrics.update(component_means)
        if modality_loss_buffers is not None:
            for modality, stats_buffer in modality_loss_buffers.items():
                if stats_buffer:
                    stats = torch.stack(stats_buffer).sum(dim=0)
                else:
                    stats = torch.zeros(2, device=self.dist_env.device)
                if torch.distributed.is_initialized():
                    stats = self._dp_allreduce(stats, include_cp=True)
                loss_sum, count = stats[0].item(), stats[1].item()
                if count > 0:
                    metrics[f"loss_{modality}"] = loss_sum / count

        return MetricsSample(
            step=self.step_scheduler.step,
            epoch=self.step_scheduler.epoch,
            metrics=metrics,
        )

    def save_checkpoint(
        self,
        epoch: int,
        step: int,
        train_loss: float,
        val_loss: dict[str, float] | None = None,
        best_metric_key: str = "default",
    ):
        super().save_checkpoint(epoch, step, train_loss, val_loss=val_loss, best_metric_key=best_metric_key)

        if not self.checkpointer.config.enabled or not self.dist_env.is_main:
            return

        ckpt_root = Path(self.checkpointer.config.checkpoint_dir)
        epoch_ckpt = ckpt_root / f"epoch_{epoch}_step_{step}"
        if not epoch_ckpt.exists():
            return

        model = self.model_parts[0]
        if isinstance(model, DistributedDataParallel):
            model = model.module

        projection = getattr(model, "projection", None)
        if projection is not None:
            proj_state = {
                "weight": projection.weight.detach().cpu(),
                "bias": projection.bias.detach().cpu(),
                "in_features": int(projection.in_features),
                "out_features": int(projection.out_features),
            }
            torch.save(proj_state, epoch_ckpt / "projection.pt")


def main(default_config_path="examples/retrieval/distillation/ministral3_2b_distill.yaml"):
    """Entry point: load config, build the distillation recipe, and run training."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = EmbeddingDistillRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()
