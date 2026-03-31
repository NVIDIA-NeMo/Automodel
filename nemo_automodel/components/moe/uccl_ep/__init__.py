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

"""UCCL-EP integration for expert parallelism.

UCCL-EP (https://github.com/uccl-project/uccl/tree/main/ep) is a community
fork of DeepEP that uses libibverbs RC QPs instead of NVSHMEM IBGDA, making
it compatible with environments where GPU-to-NIC PCIe P2P is unavailable
(e.g., Azure HPC VMs with Mellanox InfiniBand).

Usage:
    Set dispatcher: uccl_ep in the model backend config, or set
    AUTOMODEL_EP_BACKEND=uccl_ep environment variable.
"""

from nemo_automodel.components.moe.uccl_ep.buffer import UCCLBuffer

__all__ = ["UCCLBuffer"]
