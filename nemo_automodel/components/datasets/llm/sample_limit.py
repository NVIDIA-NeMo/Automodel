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

"""Compatibility for the retired dataset sample-limit spelling."""

import warnings


def resolve_sample_limit(limit_dataset_samples: int | None, num_samples_limit: int | None) -> int | None:
    """Resolve the canonical limit and its deprecated alias without changing either input.

    The alias remains supported until the next breaking dataset API release.
    Conflicting values fail before loading data; equal values are accepted.
    """
    if num_samples_limit is not None:
        warnings.warn(
            "num_samples_limit is deprecated; use limit_dataset_samples instead. "
            "The alias will be removed in the next breaking dataset API release.",
            FutureWarning,
            stacklevel=3,
        )
        if limit_dataset_samples is not None and limit_dataset_samples != num_samples_limit:
            raise ValueError("limit_dataset_samples and num_samples_limit must agree when both are provided")
        return num_samples_limit
    return limit_dataset_samples
