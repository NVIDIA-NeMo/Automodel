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

"""Linear-memory Titans model (Gated DeltaNet + data-dependent momentum + forget).

Importing this package registers ``TitansConfig`` / ``TitansForCausalLM`` with the
HuggingFace ``AutoConfig`` / ``AutoModel`` / ``AutoModelForCausalLM`` factories so
that ``AutoModelForCausalLM.from_config(TitansConfig(...))`` works out of the box.
"""

from nemo_automodel.components.models.titans.config import TitansConfig
from nemo_automodel.components.models.titans.layers import NeuralMemory, titans_delta_rule_recurrence
from nemo_automodel.components.models.titans.model import TitansForCausalLM, TitansModel

__all__ = [
    "TitansConfig",
    "TitansForCausalLM",
    "TitansModel",
    "NeuralMemory",
    "titans_delta_rule_recurrence",
]


def _register_with_hf() -> None:
    """Register Titans with HuggingFace Auto* factories (idempotent)."""
    try:
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        AutoConfig.register("titans", TitansConfig, exist_ok=True)
        AutoModel.register(TitansConfig, TitansModel, exist_ok=True)
        AutoModelForCausalLM.register(TitansConfig, TitansForCausalLM, exist_ok=True)
    except Exception:  # pragma: no cover - registration is best-effort
        import logging

        logging.getLogger(__name__).debug("Titans HF Auto* registration failed", exc_info=True)


_register_with_hf()
