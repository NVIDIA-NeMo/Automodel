# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Trackio run wrapper used by the training recipes."""

import logging
from collections.abc import Mapping
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Keys Trackio reserves for its own row fields (``trackio.utils.RESERVED_KEYS``). Trackio renames them to
# ``__<key>`` with a warning on every call; ``MetricsSample.to_dict()`` carries ``step`` and ``timestamp``, which
# Trackio already records itself, so they are dropped instead.
RESERVED_KEYS = frozenset({"project", "run", "timestamp", "step", "time", "metrics"})


class TrackioLogger:
    """Log training metrics to a Trackio run.

    Wraps the ``trackio.Run`` returned by ``trackio.init`` so recipes can log ``MetricsSample`` dictionaries
    directly: reserved keys are dropped, tensors are reduced to Python floats and non-numeric values are skipped.
    Built by :meth:`nemo_automodel.components.loggers.loggers.TrackioConfig.build` on rank 0 only.

    Args:
        run: The ``trackio.Run`` to log to.
    """

    def __init__(self, run: Any):
        self.run = run

    def log_metrics(self, metrics: Mapping[str, Any], step: int) -> None:
        """Log a metrics dictionary at ``step``.

        Args:
            metrics: Metric name to value. Values may be Python numbers or tensors; a tensor with more than one
                element is logged as its mean. Reserved Trackio keys (e.g. ``step``, ``timestamp``) and
                non-numeric values are skipped.
            step: Training step used as the x-axis.
        """
        payload = {}
        for key, value in metrics.items():
            if key in RESERVED_KEYS:
                continue
            if isinstance(value, torch.Tensor):
                payload[key] = float(value.item() if value.numel() == 1 else value.float().mean().item())
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                payload[key] = float(value)
            else:
                logger.debug("Skipping Trackio metric %s with unsupported type %s", key, type(value))
        if payload:
            self.run.log(payload, step=step)

    def finish(self) -> None:
        """Finish the run and flush pending metrics."""
        self.run.finish()
