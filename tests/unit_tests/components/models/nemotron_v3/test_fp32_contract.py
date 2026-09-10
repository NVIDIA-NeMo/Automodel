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

import types

from nemo_automodel.components.models.nemotron_v3.fp32_contract import get_force_hf_fp32_module_names


def test_force_hf_fp32_module_names_are_nemotron_specific():
    nemotron_config = types.SimpleNamespace(architectures=["NemotronHForCausalLM"])
    other_config = types.SimpleNamespace(architectures=["OtherForCausalLM"])

    assert get_force_hf_fp32_module_names(nemotron_config) == ("e_score_correction_bias",)
    assert get_force_hf_fp32_module_names(other_config) == ()
