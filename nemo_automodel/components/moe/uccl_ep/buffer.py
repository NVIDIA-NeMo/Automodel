# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""UCCLBuffer: a DeepEP-compatible Buffer backed by UCCL-EP.

This module re-exports the canonical Buffer implementation under the UCCLBuffer
alias expected by nemo_automodel, with automatic intranode detection.
"""

import os

import torch

from nemo_automodel.components.moe.uccl_ep._buffer import Buffer, EventOverlap

try:
    from uccl.ep import Config, EventHandle  # noqa: F401
except ImportError:
    try:
        from ep import Config, EventHandle  # type: ignore[no-redef]  # noqa: F401
    except ImportError:
        class EventHandle:  # noqa: D101
            pass


class UCCLBuffer(Buffer):
    """Buffer subclass that auto-detects intranode mode.

    When all EP ranks fit on a single node (group_size <= LOCAL_WORLD_SIZE),
    RDMA is disabled and only NVLink is used, avoiding RDMA MR registration
    failures on single-node setups.
    """

    def __init__(
        self,
        group,
        num_nvl_bytes: int = 0,
        num_rdma_bytes: int = 0,
        low_latency_mode: bool = False,
        num_qps_per_rank: int = 24,
        allow_nvlink_for_low_latency_mode: bool = True,
        allow_mnnvl: bool = False,
        explicitly_destroy: bool = False,
        is_intranode: bool = False,
    ):
        # Auto-detect intranode: if all ranks fit on this node, use NVLink only.
        nproc_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count() or 1))
        if group.size() <= nproc_per_node:
            is_intranode = True
            num_rdma_bytes = 0

        super().__init__(
            group=group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=low_latency_mode,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=allow_nvlink_for_low_latency_mode,
            allow_mnnvl=allow_mnnvl,
            explicitly_destroy=explicitly_destroy,
            is_intranode=is_intranode,
        )


__all__ = ["UCCLBuffer", "Buffer", "EventOverlap", "EventHandle"]
