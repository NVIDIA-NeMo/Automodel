# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import copy
from unittest.mock import patch

import pytest
import torch
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_moe.model import Qwen3MoeForCausalLM


def _model(dtype: torch.dtype) -> Qwen3MoeForCausalLM:
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        torch_dtype=dtype,
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    # Construction records a CUDA device for the lazy RoPE cache. Initialization
    # below selects CPU before the cache is built.
    with patch("torch.cuda.current_device", return_value=0):
        return Qwen3MoeForCausalLM(config, backend=backend)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("seed", [0, 42])
def test_initialization_matches_fp32_sampling(dtype: torch.dtype, seed: int) -> None:
    """Storage precision preserves the seeded fp32 initialization distribution."""
    model = _model(dtype)
    reference = _model(torch.float32)
    torch.manual_seed(seed)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=dtype)
    torch.manual_seed(seed)
    reference.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

    reference_parameters = dict(reference.named_parameters())
    for name, parameter in model.named_parameters():
        assert parameter.dtype == dtype
        assert torch.isfinite(parameter).all(), name
        torch.testing.assert_close(parameter, reference_parameters[name].to(dtype), rtol=0, atol=0, msg=name)


def test_initialized_model_trains_and_roundtrips() -> None:
    """Checkpoint-free weights support the first update and an exact state round trip."""
    torch.manual_seed(42)
    model = _model(torch.bfloat16)
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    input_ids = torch.randint(0, model.config.vocab_size, (2, 8))
    original_head = model.lm_head.weight.detach().clone()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    logits = model(input_ids).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1).float(), input_ids.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    optimizer.step()
    assert not torch.equal(original_head, model.lm_head.weight)

    restored = _model(torch.bfloat16)
    restored.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.bfloat16)
    restored.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(restored(input_ids).logits, model(input_ids).logits, rtol=0, atol=0)
