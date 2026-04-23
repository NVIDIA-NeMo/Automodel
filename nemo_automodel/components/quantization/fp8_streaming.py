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

"""Deprecated — replaced by the state-dict adapter pattern.

Devstral FP8 checkpoints are now loaded via
``nemo_automodel/components/models/devstral/state_dict_adapter.py``
(``DevstralFP8StateDictAdapter``) which plugs into the standard
``Checkpointer.load_model`` flow via ``_maybe_adapt_state_dict_{to,from}_hf``
— consistent with how ``deepseek_v3/state_dict_adapter.py`` handles its
own FP8 → bf16 dequant.
"""

__all__: list[str] = []
