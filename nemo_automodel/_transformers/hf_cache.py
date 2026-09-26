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

"""Resolve Hugging Face configuration files from a warm cache before going online."""

import os
from collections.abc import Callable
from typing import Any

_HF_LOCAL_FILES_FIRST_ENV = "NEMO_AUTOMODEL_HF_LOCAL_FILES_FIRST"


def call_with_cached_files_first(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Avoid concurrent cache writes while loading an already cached config.

    A missing local file falls back to the original call, allowing downloads.
    Explicit ``local_files_only`` and ``force_download=True`` take precedence.
    Set ``NEMO_AUTOMODEL_HF_LOCAL_FILES_FIRST=0`` to restore online resolution.
    Cached revisions are preferred over checking the Hub for updates; cold
    cache downloads are not synchronized here.
    """
    if (
        os.environ.get(_HF_LOCAL_FILES_FIRST_ENV, "1") == "0"
        or "local_files_only" in kwargs
        or kwargs.get("force_download")
    ):
        return func(*args, **kwargs)
    try:
        return func(*args, local_files_only=True, **kwargs)
    except OSError:
        return func(*args, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'local_files_only'" not in str(exc):
            raise
        return func(*args, **kwargs)
