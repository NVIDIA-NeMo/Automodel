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

MISSING_TRITON_MSG = "triton is not installed. Please install it with `pip install triton`."
MISSING_QWEN_VL_UTILS_MSG = "qwen_vl_utils is not installed. Please install it with `pip install qwen-vl-utils`."
MISSING_CUT_CROSS_ENTROPY_MSG = (
    "cut_cross_entropy is not installed. Please install it with `pip install cut-cross-entropy`."
)


def noop_decorator(func, *args, **kwargs):
    """A replacement for decorators that cannot be imported."""
    return func
