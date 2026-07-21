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

"""Train-inference disaggregation for EAGLE-3 target serving.

Runs the frozen target model as a standalone inference server on separate
GPU(s) while the draft model trains elsewhere. The training side talks to the
server through :class:`RemoteEagle3TargetModel`, which implements the
:class:`~nemo_automodel.components.speculative.eagle.backend.Eagle3TargetBackend`
contract, so the EAGLE-3 recipe consumes a remote target exactly like the
co-located ``HFEagle3TargetModel``.

- HTTP is the control plane (input_ids up, tensor metadata down).
- NCCL is the data plane for the large supervision tensors (GPU-to-GPU,
  NVLink intra-node / RDMA inter-node), with a compact binary wire format
  fallback when NCCL is unavailable.
"""

from nemo_automodel.components.speculative.eagle.remote.client import RemoteEagle3TargetModel

__all__ = ["RemoteEagle3TargetModel"]
