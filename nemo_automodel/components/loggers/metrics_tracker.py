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

from __future__ import annotations

import os
import re
import time
from typing import Any, Iterable, Optional

import torch
import wandb


class WandBMetricsTracker:
    """
    Collects and logs training metrics to WandB in grouped sections.

    Handles:
    - time per step, tokens/s, samples/s, batches/s
    - micro/global batch sizes
    - TFLOPs/sec and MFU (needs peak TFLOPs per GPU provided)
    - per-device GPU peak memory (reserved/allocated/active)
    - duration elapsed/remaining and consumption counters
    - L2 norms of global grad (pass-through), embeddings, and attention Q-proj

    Usage:
      tracker = WandBMetricsTracker(...)
      tracker.update_after_step(time_delta, num_tokens_in_batch)
      tracker.log(step=..., epoch=..., base_log={...}, model_parts=[...], learning_rate=...)
    """

    def __init__(
        self,
        *,
        dp_world_size: int,
        step_scheduler: Any,
        micro_batch_size: int,
        model_parts: Iterable[torch.nn.Module] | None,
        peak_tflops_per_gpu: Optional[float] = None,
        count_trainable_params: Optional[int] = None,
        weight_norm_interval: Optional[int] = 50,
    ) -> None:
        self.dp_world_size = int(dp_world_size)
        self.step_scheduler = step_scheduler
        self.micro_batch_size = int(micro_batch_size)
        self.model_parts = list(model_parts) if model_parts is not None else []
        self.weight_norm_interval = int(weight_norm_interval)

        self.training_start_time = time.perf_counter()
        self.last_time_per_step: float | None = None
        self.cumulative_tokens: int = 0
        self.cumulative_samples: int = 0

        # Determine global batch size
        self.global_batch_size = int(self.step_scheduler.grad_acc_steps * self.micro_batch_size * max(1, self.dp_world_size))

        # Estimate total steps for ETA
        try:
            epoch_len = len(self.step_scheduler.dataloader)
        except Exception:
            epoch_len = None
        total_steps = None
        if epoch_len is not None and epoch_len > 0:
            total_steps = (self.step_scheduler.num_epochs * epoch_len) // self.step_scheduler.grad_acc_steps
            if getattr(self.step_scheduler, "max_steps", None) is not None:
                total_steps = min(total_steps, self.step_scheduler.max_steps)
        elif getattr(self.step_scheduler, "max_steps", None) is not None:
            total_steps = self.step_scheduler.max_steps
        self.planned_total_steps: Optional[int] = total_steps

        # Trainable params for FLOPs (allow injection for VLM)
        self.num_trainable_params = int(count_trainable_params) if count_trainable_params is not None else self._count_trainable_params()

        # MFU reference TFLOPs
        if peak_tflops_per_gpu is not None:
            self.peak_tflops_per_gpu = float(peak_tflops_per_gpu)
        else:
            env_val = os.getenv("PEAK_TFLOPS_PER_GPU", None)
            self.peak_tflops_per_gpu = float(env_val) if env_val is not None else None

        # Cache regexes
        self._qproj_pattern = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.q_proj\.weight$")
        self._embed_candidates = (
            "embed_tokens.weight",
            "wte.weight",
            "embeddings.word_embeddings.weight",
        )

    def _count_trainable_params(self) -> int:
        total = 0
        for mp in self.model_parts:
            for p in mp.parameters():
                if not p.requires_grad:
                    continue
                try:
                    total += p.numel()
                except Exception:
                    pass
        return int(total)

    def update_after_step(self, *, time_delta: float, num_tokens_in_batch: int) -> None:
        self.last_time_per_step = float(time_delta)
        self.cumulative_tokens += int(num_tokens_in_batch)
        self.cumulative_samples += int(self.global_batch_size)

    def _maybe_log_weight_norms(self, *, log_data: dict, model_parts: Iterable[torch.nn.Module]) -> None:
        if (self.step_scheduler.step % self.weight_norm_interval) != 0:
            return

        def _full_tensor_if_dtensor(param: torch.Tensor) -> torch.Tensor:
            return param.full_tensor() if hasattr(param, "full_tensor") else param

        # Embedding norm
        embed_l2: Optional[float] = None
        try:
            for mp in model_parts:
                for name, p in mp.named_parameters(remove_duplicate=False):
                    if any(name.endswith(s) or s in name for s in self._embed_candidates):
                        t = _full_tensor_if_dtensor(p).detach()
                        embed_l2 = float(torch.linalg.vector_norm(t.float(), 2).item())
                        break
                if embed_l2 is not None:
                    break
        except Exception:
            embed_l2 = None
        if embed_l2 is not None:
            log_data["weights/L2/embeddings"] = embed_l2

        # Q-proj norms per layer
        try:
            for mp in model_parts:
                for name, p in mp.named_parameters(remove_duplicate=False):
                    m = self._qproj_pattern.search(name)
                    if m is not None:
                        layer_idx = m.group(1)
                        t = _full_tensor_if_dtensor(p).detach()
                        q_l2 = float(torch.linalg.vector_norm(t.float(), 2).item())
                        log_data[f"weights/L2/attn/l{layer_idx}/q_proj"] = q_l2
        except Exception:
            pass

    def _add_throughput_and_batch(self, *, log_data: dict, tps: float | None) -> None:
        if self.last_time_per_step and self.last_time_per_step > 0:
            time_per_step = self.last_time_per_step
            samples_per_sec = self.global_batch_size / time_per_step
            global_batches_per_sec = 1.0 / time_per_step
            log_data.update(
                {
                    "throughput/time_per_step_sec": time_per_step,
                    "throughput/samples_per_sec": samples_per_sec,
                    "throughput/global_batches_per_sec": global_batches_per_sec,
                }
            )
        log_data["batch/micro_batch_size"] = self.micro_batch_size
        log_data["batch/global_batch_size"] = self.global_batch_size

        if self.num_trainable_params > 0 and tps is not None:
            total_tflops_per_sec = (6.0 * self.num_trainable_params * tps) / 1e12
            per_gpu_tflops_per_sec = total_tflops_per_sec / max(1, self.dp_world_size)
            log_data.update(
                {
                    "throughput/tflops_total": total_tflops_per_sec,
                    "throughput/tflops_per_gpu": per_gpu_tflops_per_sec,
                }
            )
            if self.peak_tflops_per_gpu and self.peak_tflops_per_gpu > 0:
                mfu = 100.0 * (per_gpu_tflops_per_sec / self.peak_tflops_per_gpu)
                log_data["throughput/MFU_percent"] = mfu

    def _add_duration(self, *, log_data: dict) -> None:
        elapsed_hours = (time.perf_counter() - self.training_start_time) / 3600.0
        log_data.update(
            {
                "duration/hours_elapsed": elapsed_hours,
                "duration/tokens_consumed": self.cumulative_tokens,
                "duration/samples_consumed": self.cumulative_samples,
                "duration/global_batches_consumed": self.step_scheduler.step,
            }
        )
        if self.planned_total_steps is not None and self.last_time_per_step and self.step_scheduler.step > 0:
            avg_time_per_step = elapsed_hours * 3600.0 / self.step_scheduler.step
            remaining_steps = max(0, int(self.planned_total_steps - self.step_scheduler.step))
            remaining_hours = (remaining_steps * avg_time_per_step) / 3600.0
            log_data["duration/hours_remaining"] = remaining_hours

    def _add_memory(self, *, log_data: dict) -> None:
        if not torch.cuda.is_available():
            return
        try:
            num_devices = torch.cuda.device_count()
            bytes_to_gb = 1024 ** 3
            for dev in range(num_devices):
                reserved_gb = torch.cuda.max_memory_reserved(dev) / bytes_to_gb
                allocated_gb = torch.cuda.max_memory_allocated(dev) / bytes_to_gb
                stats = torch.cuda.memory_stats(dev)
                active_peak = float(stats.get("active_bytes.all.peak", 0.0)) / bytes_to_gb
                prefix = f"memory/device_{dev}"
                log_data[f"{prefix}/peak_reserved_gb"] = reserved_gb
                log_data[f"{prefix}/peak_allocated_gb"] = allocated_gb
                log_data[f"{prefix}/peak_active_gb"] = active_peak
        except Exception:
            pass

    def log(
        self,
        *,
        step: int,
        epoch: int,
        base_log: dict,
        learning_rate: float,
        tps: Optional[float],
        grad_norm: Optional[float],
        model_parts: Iterable[torch.nn.Module] | None = None,
    ) -> None:
        """Compose and emit the WandB log payload with grouped metrics."""
        log_data = dict(base_log)
        log_data["step"] = step
        log_data["epoch"] = epoch
        if grad_norm is not None:
            log_data["gradients/L2/global_grad_norm"] = grad_norm
        log_data["learning_rate"] = learning_rate

        # Throughput + batch
        self._add_throughput_and_batch(log_data=log_data, tps=tps)
        # Duration
        self._add_duration(log_data=log_data)
        # Memory
        self._add_memory(log_data=log_data)
        # Weight norms (occasionally)
        if model_parts is None:
            model_parts = self.model_parts
        self._maybe_log_weight_norms(log_data=log_data, model_parts=model_parts)

        if wandb.run is not None:
            wandb.log(log_data, step=step) 