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
import io
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List

import torch
import torch.distributed as dist


@dataclass
class MetricsSample:
    step: int
    epoch: int
    metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: str = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
            "timestamp": self.timestamp,
        } | self.metrics

    def __post_init__(self):
        self.timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def stack_and_move_tensor_metrics_to_cpu(metric_vector: List[MetricsSample]) -> List[MetricsSample]:
    # Find all tensor metrics, stack them per metric name across samples, move to CPU,
    # then place CPU tensors back into the original metrics.
    def extract_tensor_metric_names(metric: MetricsSample) -> tuple:
        names = [name for name, value in metric.metrics.items() if isinstance(value, torch.Tensor)]
        # Sort for stable grouping and use tuple as a hashable key
        return tuple(sorted(names))

    if len(metric_vector) == 0:
        return metric_vector

    # Group sample indices by the exact set of tensor metric names they contain
    grouped_indices_by_names = {}
    for sample_index, metric in enumerate(metric_vector):
        names_key = extract_tensor_metric_names(metric)
        if names_key not in grouped_indices_by_names:
            grouped_indices_by_names[names_key] = []
        grouped_indices_by_names[names_key].append(sample_index)

    # For each group, stack tensors per metric name, move to CPU, and redistribute
    for names_key, indices in grouped_indices_by_names.items():
        if len(names_key) == 0:
            continue  # no tensor metrics in this group

        # Stack per metric name across all samples in the group
        stacked_by_name = {}
        for name in names_key:
            stacked = torch.stack([metric_vector[i].metrics[name] for i in indices])
            # Detach in case these require grad and move to CPU once
            stacked_by_name[name] = stacked.detach().cpu()

        # Write back the CPU tensors to each sample's metrics
        for pos, sample_index in enumerate(indices):
            for name in names_key:
                metric_vector[sample_index].metrics[name] = stacked_by_name[name][pos].item()

    return metric_vector


class MetricLogger:
    """
    Simple JSON Lines logger.

    - Appends one JSON object per line.
    - Thread-safe writes via an internal lock.
    - Creates parent directories as needed.
    - UTF-8 without BOM, newline per record.
    """

    def __init__(self, filepath: str, *, flush: bool = False, append: bool = True, buffer_size: int = 100) -> None:
        self.filepath = os.path.abspath(filepath)
        self.flush = flush
        self.buffer_size = buffer_size
        self.buffer = []
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        mode = "a" if append else "w"
        # Use buffered writer for performance; rely on flush flag when needed
        self._fp = io.open(self.filepath, mode, encoding="utf-8", buffering=1)

    def log(self, record: MetricsSample) -> None:
        self.buffer.append(record)
        if len(self.buffer) < self.buffer_size:
            return
        lines = self._move_to_cpu(self.buffer)
        self.buffer = []
        with self._lock:
            self._save(lines)

    def _move_to_cpu(self, buffer: List[MetricsSample]) -> List[MetricsSample]:
        lines = []
        for record in stack_and_move_tensor_metrics_to_cpu(buffer):
            lines.append(json.dumps(record.to_dict(), ensure_ascii=False))
        return lines

    def _save(self, lines: List[str]) -> None:
        if len(lines) == 0:
            return
        self._fp.write("\n".join(lines) + "\n")
        if self.flush:
            self._fp.flush()
            os.fsync(self._fp.fileno())

    def close(self) -> None:
        with self._lock:
            self._save(self._move_to_cpu(self.buffer))
            try:
                self._fp.flush()
            except Exception:
                pass
            try:
                self._fp.close()
            except Exception:
                pass

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class MetricLoggerDist(MetricLogger):
    def __init__(self, filepath: str, *, flush: bool = False, append: bool = True) -> None:
        super().__init__(filepath, flush=flush, append=append)
        assert dist.is_initialized(), "torch.distributed must be initialized with MetricLoggerDist"
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        # if not main rank, set log and close to no-op
        if self.rank != 0:
            self.log = lambda *args, **kwargs: None
            self.close = lambda: None
            self.__enter__ = lambda: None
            self.__exit__ = lambda: None


def build_metric_logger(filepath: str, *, flush: bool = False, append: bool = True) -> MetricLogger:
    if dist.is_initialized():
        return MetricLoggerDist(filepath, flush=flush, append=append)
    else:
        return MetricLogger(filepath, flush=flush, append=append)
