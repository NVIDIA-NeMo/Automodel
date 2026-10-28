# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Step-5 Preview text and vision forward/training support.

The released primary decoder has 92 layers; the perception encoder has 47.
Checkpoint FP8 expert tensors are dequantized into the selected training dtype.
MTP auxiliary layers, incremental KV-cache generation, and TP are excluded.

The default CUDA attention backend is FA4 on SM90, with native 192-dimensional
heads. It requires flash-attention revision 3c2c6b31e730fd13e7d6f0d384af22684b24c532
or its SM90 GQA backward fix (#2888). CSA uses route-derived forward/reverse
block lists and compact region membership. It skips empty tiles; independent
query selections can still make almost every physical tile active.

CP uses contiguous query shards. SWA exchanges only the preceding window through
point-to-point transfers, with local masked zeros preserving global tile
alignment. CSA gathers original K/V and indexer K/Z; attention backward
reduce-scatters K/V gradients. Packed documents use resetting position IDs.

Validation entry points live in tests/functional_tests/models/step5 and
tests/functional_tests/context_parallel/run_step5_cp.py.
"""
