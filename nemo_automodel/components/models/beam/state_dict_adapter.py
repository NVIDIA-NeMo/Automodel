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

"""State-dict adapter between Hugging Face Beam checkpoints and the native grouped-experts layout."""

from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter


class BeamStateDictAdapter(Qwen3MoeStateDictAdapter):
    """Converts between Hugging Face Beam checkpoints and the grouped-experts native format.

    Beam stores its experts exactly like Qwen3-MoE, so the Qwen3-MoE conversion (including fused PEFT v5
    ParamWrapper LoRA export/import and low-memory DCP loading) applies unchanged. The native module tree
    mirrors the HF one, so every tensor outside the experts keeps its HF name and passes through unchanged
    (attention projections and ``gate_bias``, q/k norms, the four per-layer norms, the dense MLP,
    ``mlp.gate.weight`` and the ``mlp.gate.e_score_correction_bias`` buffer, embeddings, final norm,
    ``lm_head``). Experts differ:

    HF checkpoint (per expert, the ``qwen2_moe`` conversion mapping)::

        model.layers.{L}.mlp.experts.{E}.gate_proj.weight   [moe_inter_dim, hidden]
        model.layers.{L}.mlp.experts.{E}.up_proj.weight     [moe_inter_dim, hidden]
        model.layers.{L}.mlp.experts.{E}.down_proj.weight   [hidden, moe_inter_dim]

    Native (grouped)::

        model.layers.{L}.mlp.experts.gate_and_up_projs      [n_experts, hidden, 2 * moe_inter_dim]
        model.layers.{L}.mlp.experts.down_projs             [n_experts, moe_inter_dim, hidden]

    ``to_hf`` therefore writes the per-expert layout that ``transformers`` itself produces with
    ``save_pretrained``, not the stacked in-memory layout.
    """
