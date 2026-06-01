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

import logging
import re
from typing import Any, Optional

import torch

from nemo_automodel.components.checkpoint.state_dict_adapter import StateDictAdapter
from nemo_automodel.components.models.common import BackendConfig

logger = logging.getLogger(__name__)


_THINKER_PREFIX = "thinker."
_DROP_PREFIXES = ("talker.", "token2wav.")

# Keys the NeMo Thinker class deletes at __init__ time (so the HF base
# checkpoint must not be allowed to repopulate them, otherwise load_state_dict
# would either raise on an unexpected key or silently re-insert the dead
# parameter). Matched as substrings against the post-prefix-strip key.
_DROP_THINKER_KEY_SUBSTRINGS = ("audio_tower.audio_bos_eos_token",)


class Qwen2_5OmniStateDictAdapter(StateDictAdapter):
    """HF Qwen2.5-Omni checkpoint adapter (thinker-only path).

    HF Qwen/Qwen2.5-Omni-* checkpoints store keys under three top-level
    prefixes: ``thinker.*``, ``talker.*``, ``token2wav.*``. For ASR/text
    fine-tuning we only train the Thinker, so this adapter:

    - on ``from_hf``: drops ``talker.*`` and ``token2wav.*`` keys and strips
      the ``thinker.`` prefix so keys align with our NeMo Thinker class.
    - on ``to_hf``: re-adds the ``thinker.`` prefix so the saved checkpoint
      can be merged back with the original talker/token2wav shards.

    Qwen2.5-Omni-3B is dense (no MoE), so no expert grouping logic is
    needed — this is a thin key-renaming adapter.
    """

    def __init__(
        self,
        config: Any,
        backend: BackendConfig | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.config = config
        self.backend = backend or BackendConfig()
        self.dtype = dtype
        self._uses_thinker_prefix = True

    def to_hf(
        self,
        state_dict: dict[str, Any],
        exclude_key_regex: Optional[str] = None,
        quantization: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        if self._uses_thinker_prefix:
            hf_state_dict = {_THINKER_PREFIX + k: v for k, v in state_dict.items()}
        else:
            hf_state_dict = dict(state_dict)

        if exclude_key_regex:
            pattern = re.compile(exclude_key_regex)
            hf_state_dict = {k: v for k, v in hf_state_dict.items() if not pattern.match(k)}
        return hf_state_dict

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        device_mesh: Optional[Any] = None,
        **kwargs,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        saw_thinker = False
        for key, value in hf_state_dict.items():
            if key.startswith(_DROP_PREFIXES):
                continue
            stripped = key[len(_THINKER_PREFIX) :] if key.startswith(_THINKER_PREFIX) else key
            # Drop keys for parameters the NeMo Thinker deletes at __init__.
            if any(sub in stripped for sub in _DROP_THINKER_KEY_SUBSTRINGS):
                continue
            if key.startswith(_THINKER_PREFIX):
                saw_thinker = True
            out[stripped] = value
        self._uses_thinker_prefix = saw_thinker or self._uses_thinker_prefix
        return out

    def convert_single_tensor_to_hf(self, fqn: str, tensor: Any, **kwargs) -> list[tuple[str, Any]]:
        exclude_key_regex = kwargs.get("exclude_key_regex", None)
        key = _THINKER_PREFIX + fqn if self._uses_thinker_prefix else fqn
        if exclude_key_regex:
            if re.match(exclude_key_regex, key):
                return []
        return [(key, tensor)]
