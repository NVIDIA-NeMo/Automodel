# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Vendored DeepSeek V4 TileLang kernels.

Attribution:
* Upstream project: Miles, https://github.com/yueming-yuan/miles
* Upstream revision: e561465d0b9bbf06188b7a5e2020dc7fd691f732, deepseek-v4 branch
* Upstream license: Apache-2.0, copyright 2025 Zhipu AI
* Source tree:
  https://github.com/yueming-yuan/miles/tree/e561465d0b9bbf06188b7a5e2020dc7fd691f732/miles_plugins/models/deepseek_v4/ops

The vendored files were adapted for AutoModel's DeepSeek V4 tensor layouts,
packed-sequence dispatch, and optional backend selection.  See
``ATTRIBUTION.md`` in this directory for the per-file source mapping.
"""
