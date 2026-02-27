#!/usr/bin/env python3
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

"""Backward-compatibility shim -- the real implementation lives in cli.app."""

import warnings

warnings.warn(
    "nemo_automodel._cli.app has moved to cli.app. "
    "Update your imports. This shim will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from cli.app import build_parser, load_yaml, main  # noqa: F401, E402
