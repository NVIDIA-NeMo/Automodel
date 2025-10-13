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
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import torch.distributed as dist


@dataclass
class MetricsSample:
    step: int
    epoch: int
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "epoch": self.epoch,
        } | self.metrics


class MetricLogger:
    """
    Simple JSON Lines logger.

    - Appends one JSON object per line.
    - Thread-safe writes via an internal lock.
    - Creates parent directories as needed.
    - UTF-8 without BOM, newline per record.
    """

    def __init__(self, filepath: str, *, flush: bool = True, append: bool = True) -> None:
        self.filepath = os.path.abspath(filepath)
        self.flush = flush
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(self.filepath) or ".", exist_ok=True)
        mode = "a" if append else "w"
        # Use buffered writer for performance; rely on flush flag when needed
        self._fp = io.open(self.filepath, mode, encoding="utf-8", buffering=1)

    def log(self, record: Dict[str, Any], *, add_timestamp: bool = True) -> None:
        entry: Dict[str, Any] = dict(record)
        if add_timestamp and "timestamp" not in entry:
            entry["timestamp"] = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            if self.flush:
                self._fp.flush()
                os.fsync(self._fp.fileno())

    def close(self) -> None:
        with self._lock:
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

    def log(self, record: Dict[str, Any], *, add_timestamp: bool = True) -> None:
        super().log(record, add_timestamp=add_timestamp)
