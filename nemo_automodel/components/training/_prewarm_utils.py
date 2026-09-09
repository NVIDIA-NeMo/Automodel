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

"""Shared helpers for setup-time prewarms."""

import logging

import torch

logger = logging.getLogger(__name__)


def _resolve_cuda_device(device: torch.device | int | str | None, label: str) -> torch.device | None:
    """Normalize ``device`` and return it if it is a usable CUDA device, else None."""
    if device is None:
        logger.info("Skipping %s prewarm: no device assigned.", label)
        return None
    device = torch.device("cuda", device) if isinstance(device, int) else torch.device(device)
    if not torch.cuda.is_available() or device.type != "cuda":
        logger.info(
            "Skipping %s prewarm: device=%s cuda_available=%s",
            label,
            device,
            torch.cuda.is_available(),
        )
        return None
    torch.cuda.set_device(device)
    return device
