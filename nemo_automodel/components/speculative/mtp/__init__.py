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

"""MTP (multi-token prediction) acceptance-length evaluation.

Scores a model's existing native MTP head (``nemo_automodel.components.models.common.mtp``)
against its own main-head greedy prediction under teacher forcing, to report
the self-speculative accept length a finetuned MTP head would achieve, without
adding a separate draft model or recipe.
"""

from nemo_automodel.components.speculative.mtp.core import (
    MTPAcceptanceMetrics,
    compute_mtp_prefix_hits,
    simulated_accept_length,
)

__all__ = [
    "MTPAcceptanceMetrics",
    "compute_mtp_prefix_hits",
    "simulated_accept_length",
]
