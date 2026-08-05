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
"""CPU-only checks for Qwen3-MoE pipeline-parallel compatibility."""

from nemo_automodel.components.distributed.pipelining.hf_utils import model_keeps_self_forward
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM


def test_pp_keep_self_forward_is_declared():
    """Pipeline parallelism must preserve Qwen3-MoE's own forward.

    Qwen3-MoE uses the gpt_oss-style RoPE (``position_ids_to_freqs_cis`` +
    ``apply_rotary_emb_qk`` with ``cu_seqlens``/``cp_size``/``cp_rank``) and a
    ``freqs_cis`` decoder-layer API. Without ``_pp_keep_self_forward = True`` the
    PP builder replaces the forward with the generic HF one that calls
    ``rotary_emb(hidden_states, position_ids)`` and crashes inside
    ``apply_rotary_emb`` under THD + context parallelism
    (``RuntimeError: Sizes of tensors must match ... at torch.cat``). The model's
    own forward already handles PP stage routing (``embed_tokens``/``norm``/
    ``lm_head`` are ``None`` off the owning stage; hidden states arrive in the
    ``input_ids`` slot) and CP + THD.
    """
    assert getattr(Qwen3MoeForCausalLM, "_pp_keep_self_forward", False) is True
    # The pipeline build call site keys off model_keeps_self_forward(...).
    model = Qwen3MoeForCausalLM.__new__(Qwen3MoeForCausalLM)
    assert model_keeps_self_forward(model) is True
