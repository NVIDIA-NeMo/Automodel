# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed
from torch.nn.parallel import DistributedDataParallel

from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loggers.metric_logger import MetricsSample
from nemo_automodel.components.loss.embedding_distill import EmbeddingDistillLoss, EmbeddingMSELoss, ScoreDistillLoss
from nemo_automodel.components.loss.infonce import InfoNCEDistillLoss, InfoNCELoss
from nemo_automodel.components.loss.intermediate_distill import IntermediateDistillLoss
from nemo_automodel.components.training.utils import scale_grads_and_clip_grad_norm
from nemo_automodel.recipes.llm.train_ft import build_lr_scheduler
from nemo_automodel.recipes.retrieval.train_bi_encoder import TrainBiEncoderRecipe

logger = logging.getLogger(__name__)


def _build_or_none(cfg_section):
    if cfg_section is None:
        return None
    if hasattr(cfg_section, "instantiate"):
        return cfg_section.instantiate()
    return cfg_section


def _move_to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def _unpack_qpn(batch: dict[str, torch.Tensor]):
    q_batch = {
        "input_ids": batch["q_input_ids"],
        "attention_mask": batch["q_attention_mask"],
    }
    d_batch = {
        "input_ids": batch["d_input_ids"],
        "attention_mask": batch["d_attention_mask"],
    }

    if "n_input_ids" not in batch or "n_attention_mask" not in batch or "n_mask" not in batch:
        return q_batch, d_batch, None, None

    n_input_ids = batch["n_input_ids"]
    n_attention_mask = batch["n_attention_mask"]
    n_mask = batch["n_mask"]

    if n_input_ids.dim() != 3:
        return q_batch, d_batch, None, None

    bsz, num_neg, seq_len = n_input_ids.shape
    if num_neg == 0 or seq_len == 0 or int(n_mask.sum().item()) == 0:
        return q_batch, d_batch, None, None

    flat_n_batch = {
        "input_ids": n_input_ids.view(bsz * num_neg, seq_len),
        "attention_mask": n_attention_mask.view(bsz * num_neg, seq_len),
    }
    return q_batch, d_batch, flat_n_batch, n_mask


def _dp_group_src_rank(group) -> int:
    if group is None:
        return 0
    try:
        return torch.distributed.get_process_group_ranks(group)[0]
    except Exception:
        return 0


def _clean_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _copy_checkpoint_metadata(src_dir: Path, dst_dir: Path) -> None:
    """Copy HF metadata/tokenizer files needed by AutoModel.from_pretrained."""
    for src in src_dir.iterdir():
        if src.suffix == ".safetensors" or src.name.endswith(".safetensors.index.json"):
            continue
        if not src.is_file():
            continue
        shutil.copy2(src, dst_dir / src.name)


def _mirror_hf_metadata(src_model_dir: Path, dst_dir: Path, *, overwrite: bool = False) -> None:
    """Mirror non-weight HF artifacts from the original student checkpoint."""
    if not src_model_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in src_model_dir.iterdir():
        if not src.is_file():
            continue
        if src.suffix in {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}:
            continue
        if src.name.endswith((".bin.index.json", ".safetensors.index.json")):
            continue

        dst = dst_dir / src.name
        if overwrite or not dst.exists():
            shutil.copy2(src, dst)


def _strip_student_prefix(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return the HF backbone tensors from a StudentWithProjection checkpoint."""
    prefixes = ("student.model.", "module.student.model.", "model.")
    for prefix in prefixes:
        filtered = {key[len(prefix) :]: value for key, value in state.items() if key.startswith(prefix)}
        if filtered:
            return filtered

    # Already an HF backbone checkpoint. Drop projection tensors if present.
    return {key: value for key, value in state.items() if not key.startswith("projection.")}


def _export_hf_student_checkpoint(src_dir: Path, dst_dir: Path) -> None:
    """Materialize an evaluator-facing HF checkpoint from Automodel wrapper weights."""
    from safetensors.torch import load_file, save_file

    _clean_path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    _copy_checkpoint_metadata(src_dir, dst_dir)

    index_path = src_dir / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text())
        source_weight_files = sorted(set(index.get("weight_map", {}).values()))
    else:
        source_weight_files = sorted(path.name for path in src_dir.glob("*.safetensors"))

    if not source_weight_files:
        raise FileNotFoundError(f"No safetensors weights found in {src_dir}")

    output_weight_map: dict[str, str] = {}
    output_files: list[Path] = []
    multi_shard = len(source_weight_files) > 1

    for shard_idx, weight_file in enumerate(source_weight_files, start=1):
        shard_state = _strip_student_prefix(load_file(str(src_dir / weight_file)))
        if not shard_state:
            continue

        if multi_shard:
            out_name = f"model-{shard_idx:05d}-of-{len(source_weight_files):05d}.safetensors"
        else:
            out_name = "model.safetensors"
        out_path = dst_dir / out_name
        save_file(shard_state, str(out_path))
        output_files.append(out_path)
        output_weight_map.update({key: out_name for key in shard_state})

    if not output_weight_map:
        raise RuntimeError(f"No student backbone tensors found in {src_dir}")

    index = {
        "metadata": {"total_size": sum(path.stat().st_size for path in output_files)},
        "weight_map": output_weight_map,
    }
    (dst_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))


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

        if "StudentWithProjection" not in model_target:
            logger.warning(
                "EmbeddingDistillRecipe expects model to be StudentWithProjection; got %s",
                model_target or "<unknown>",
            )

        with torch.random.fork_rng(enabled=False):
            self.teacher_model = self.cfg.teacher_model.instantiate(
                device_mesh=self.device_mesh,
                moe_mesh=self.moe_mesh,
                distributed_config=self.distributed_config,
                peft_config=None,
            )
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad_(False)

        self.distill_loss = _build_or_none(self.cfg.get("distill_loss", None))
        self.mse_loss = _build_or_none(self.cfg.get("mse_loss", None))
        self.score_loss = _build_or_none(self.cfg.get("score_loss", None))
        self.intermediate_loss = _build_or_none(self.cfg.get("intermediate_loss", None))
        self.infonce_loss = _build_or_none(self.cfg.get("infonce_loss", None))
        self.infonce_distill_loss = _build_or_none(self.cfg.get("infonce_distill_loss", None))

        if self.distill_loss is None:
            self.distill_loss = EmbeddingDistillLoss(reduction="mean")
        if self.mse_loss is None:
            self.mse_loss = EmbeddingMSELoss(normalize=False, reduction="mean")
        if self.score_loss is None:
            self.score_loss = ScoreDistillLoss(temperature=float(self.cfg.get("score_temperature", 0.02)))
        if self.intermediate_loss is None:
            self.intermediate_loss = IntermediateDistillLoss(layer_pairs=[])
        if self.infonce_loss is None:
            self.infonce_loss = InfoNCELoss()
        if self.infonce_distill_loss is None:
            self.infonce_distill_loss = InfoNCEDistillLoss()

        self.loss_weights = {
            "distill": float(self.cfg.get("distill_loss_weight", 1.0)),
            "mse": float(self.cfg.get("mse_loss_weight", 0.0)),
            "score": float(self.cfg.get("score_loss_weight", 0.0)),
            "intermediate": float(self.cfg.get("intermediate_loss_weight", 0.0)),
            "nce": float(self.cfg.get("nce_loss_weight", 0.0)),
            "nce_kd": float(self.cfg.get("nce_distill_loss_weight", 0.0)),
        }

        self.layer_pairs = [tuple(int(x) for x in pair) for pair in self.cfg.get("layer_pairs", [])]
        if self.loss_weights["intermediate"] > 0 and not self.layer_pairs:
            raise ValueError(
                "intermediate_loss_weight > 0 requires non-empty layer_pairs in config"
            )

        if self.loss_weights["intermediate"] > 0:
            student_layers = sorted({s for s, _ in self.layer_pairs})
            teacher_layers = sorted({t for _, t in self.layer_pairs})
            if hasattr(self.model_parts[0], "attach_intermediate_capture"):
                self.model_parts[0].attach_intermediate_capture(student_layers)
            if hasattr(self.teacher_model, "attach_intermediate_capture"):
                self.teacher_model.attach_intermediate_capture(teacher_layers)

        projection_lr = self.cfg.get("projection_lr", None)
        restore_from = self.cfg.get("checkpoint.restore_from", None)
        self._sync_projection_parameters()
        if restore_from is None:
            self._rebuild_optimizer_with_projection_group(projection_lr)

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
                continue
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM, group=dp_group)
            param.grad.div_(dp_size)

    def _rebuild_optimizer_with_projection_group(self, projection_lr: float | None) -> None:
        model = self.model_parts[0]
        decay_params = []
        no_decay_params = []
        proj_params = []

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
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
        if proj_params:
            group = {"params": proj_params, "weight_decay": 0.0}
            if projection_lr is not None:
                group["lr"] = float(projection_lr)
            param_groups.append(group)

        new_optimizer = self.cfg.optimizer.instantiate(params=param_groups)
        if not isinstance(self.optimizer, list):
            raise TypeError(f"Expected optimizer to be a list, got {type(self.optimizer)}")
        self.optimizer[:] = [new_optimizer]

        new_lr_schedulers = build_lr_scheduler(self.cfg.get("lr_scheduler", None), self.optimizer, self.step_scheduler)
        if isinstance(self.lr_scheduler, list):
            self.lr_scheduler[:] = [] if new_lr_schedulers is None else list(new_lr_schedulers)

    def _forward_backward_step(self, idx, batch, *, loss_buffer, num_batches, is_train: bool = True):
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
            s_q_pool, s_q_proj, s_q_inter = student_model(q_batch)
            s_d_pool, s_d_proj, s_d_inter = student_model(d_batch)

            s_n_pool = None
            if n_batch is not None and uses_hard_negatives:
                bsz, num_neg = n_mask.shape
                s_n_pool, _, _ = student_model(n_batch)
                s_n_pool = s_n_pool.view(bsz, num_neg, s_n_pool.shape[-1])

            with torch.no_grad():
                t_q_pool, t_q_inter = self.teacher_model(q_batch)
                t_d_pool, t_d_inter = self.teacher_model(d_batch)
                t_n_pool = None
                if n_batch is not None and self.loss_weights["nce_kd"] > 0:
                    bsz, num_neg = n_mask.shape
                    t_n_pool, _ = self.teacher_model(n_batch)
                    t_n_pool = t_n_pool.view(bsz, num_neg, t_n_pool.shape[-1])

            zero = s_q_proj.new_zeros(())
            l_distill = self.distill_loss(s_q_proj, t_q_pool, s_d_proj, t_d_pool) if self.loss_weights["distill"] > 0 else zero
            l_mse = self.mse_loss(s_q_proj, t_q_pool, s_d_proj, t_d_pool) if self.loss_weights["mse"] > 0 else zero
            l_score = self.score_loss(s_q_pool, t_q_pool, s_d_pool, t_d_pool) if self.loss_weights["score"] > 0 else zero

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
                l_nce = self.infonce_loss(
                    s_q_pool,
                    s_d_pool,
                    hard_negatives=s_n_pool,
                    hard_negatives_mask=n_mask,
                )
            else:
                l_nce = zero

            if self.loss_weights["nce_kd"] > 0:
                l_nce_kd = self.infonce_distill_loss(
                    s_q_pool,
                    s_d_pool,
                    t_q_pool,
                    t_d_pool,
                    student_hard_negatives=s_n_pool,
                    teacher_hard_negatives=t_n_pool,
                    hard_negatives_mask=n_mask,
                )
            else:
                l_nce_kd = zero

            total_loss = (
                self.loss_weights["distill"] * l_distill
                + self.loss_weights["mse"] * l_mse
                + self.loss_weights["score"] * l_score
                + self.loss_weights["intermediate"] * l_inter
                + self.loss_weights["nce"] * l_nce
                + self.loss_weights["nce_kd"] * l_nce_kd
            )

            loss_buffer.append(total_loss.detach())
            self._loss_component_buffer.append(
                {
                    "loss_distill": l_distill.detach(),
                    "loss_mse": l_mse.detach(),
                    "loss_score": l_score.detach(),
                    "loss_intermediate": l_inter.detach(),
                    "loss_nce": l_nce.detach(),
                    "loss_nce_distill": l_nce_kd.detach(),
                }
            )

            if is_train:
                (total_loss / num_batches).backward()

    def _run_train_optim_step(self, batches, max_grad_norm=None):
        self._loss_component_buffer: list[dict[str, torch.Tensor]] = []

        loss_buffer = []
        for idx, batch in enumerate(batches):
            self._forward_backward_step(idx, batch, loss_buffer=loss_buffer, num_batches=len(batches), is_train=True)

        self._sync_projection_gradients()

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
            "grad_norm": grad_norm,
            "lr": lr,
            "mem": mem_allocated,
            "time_per_step": elapsed,
        }
        metrics.update(component_means)

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

        legacy_ckpt = ckpt_root / f"step_{step}"
        legacy_ckpt.mkdir(parents=True, exist_ok=True)

        if projection is not None:
            shutil.copy2(epoch_ckpt / "projection.pt", legacy_ckpt / "projection.pt")

        source_student_dir = epoch_ckpt / "model" / "consolidated"
        if not source_student_dir.is_dir():
            source_student_dir = epoch_ckpt / "model"

        # Ensure evaluator-facing HF metadata files are present in the saved
        # student directory. Some consolidated saves may omit config / custom
        # remote-code modules, which breaks AutoModel.from_pretrained at eval.
        src_model_dir = Path(str(self.cfg.get("model.pretrained_model_name_or_path", "")))
        _mirror_hf_metadata(src_model_dir, source_student_dir, overwrite=False)

        student_dir = legacy_ckpt / "student"
        _export_hf_student_checkpoint(source_student_dir, student_dir)
        _mirror_hf_metadata(src_model_dir, student_dir, overwrite=True)
        if projection is not None:
            shutil.copy2(epoch_ckpt / "projection.pt", student_dir / "projection.pt")

        meta = {"epoch": epoch, "step": step}
        (legacy_ckpt / "meta.json").write_text(json.dumps(meta, indent=2))
