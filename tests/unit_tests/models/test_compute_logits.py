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

"""A forward called with ``compute_logits=False`` returns the final hidden states and empty logits.

Fused-linear losses project the hidden states with the head's weight themselves. Forwards that delegate to Hugging
Face, whose forward always runs the head, project a single position and drop it.
"""

import copy
import inspect
from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel._transformers.model_init import _get_mixin_wrapped_class
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.lm_head import LMHead
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep

pytestmark = pytest.mark.cpu_only


def _gemma4() -> tuple[nn.Module, int]:
    """Build a dense one-layer Gemma 4, whose forward delegates to Hugging Face.

    Returns:
        The model and its vocabulary size.
    """
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig

    from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration

    text_config = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=1,
        intermediate_size=64,
        max_position_embeddings=64,
        enable_moe_block=False,
        layer_types=["full_attention"],
        hidden_activation="gelu_pytorch_tanh",
    )
    config = Gemma4Config(text_config=text_config, vision_config=None, audio_config=None)
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", rope_fusion=False)
    return Gemma4ForConditionalGeneration(config, backend=backend), 64


def _gemma4_unified() -> tuple[nn.Module, int]:
    """Build a one-layer Gemma 4 Unified, whose forward delegates to Hugging Face without context parallelism.

    Returns:
        The model and its vocabulary size.
    """
    from nemo_automodel.components.models.gemma4_unified.model import Gemma4UnifiedForConditionalGeneration
    from tests.unit_tests.models.gemma4_unified.test_state_dict_adapter import _tiny_config

    return Gemma4UnifiedForConditionalGeneration(_tiny_config()), 32


def _qwen3_vl() -> tuple[nn.Module, int]:
    """Build a two-layer Qwen3-VL, whose text forward delegates to Hugging Face without context parallelism.

    Returns:
        The model and its vocabulary size.
    """
    from nemo_automodel.components.models.qwen3_vl.model import Qwen3VLForConditionalGeneration
    from tests.unit_tests.models.qwen3_vl.test_qwen3_vl_cp_preembed import _tiny_config

    return Qwen3VLForConditionalGeneration(_tiny_config()), 32


@pytest.mark.parametrize("build", [_gemma4, _gemma4_unified, _qwen3_vl], ids=["gemma4", "gemma4_unified", "qwen3_vl"])
def test_hugging_face_forward_projects_one_position_without_logits(
    build: Callable[[], tuple[nn.Module, int]],
):
    torch.manual_seed(0)
    model, vocab_size = build()
    model = model.float().eval()
    projected = []
    model.lm_head.register_forward_pre_hook(lambda module, args: projected.append(args[0].shape[:-1]))
    # Token ids below 29 avoid every model's image, video and special placeholder ids.
    input_ids = torch.randint(5, 29, (2, 6), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        with_logits = model(input_ids=input_ids, output_hidden_states=True)
        projected.clear()
        output = model(input_ids=input_ids, output_hidden_states=True, compute_logits=False)

    assert with_logits.logits.shape == (2, 6, vocab_size)
    # Hugging Face's forward runs the head, so it projects only the last position, and the logits stay empty.
    assert projected == [torch.Size([2, 1])]
    assert output.logits.shape[-1] == 0
    torch.testing.assert_close(get_final_hidden_states(output), get_final_hidden_states(with_logits), rtol=0, atol=0)


def test_kimi_vl_without_logits_returns_its_output_without_return_dict():
    from nemo_automodel.components.models.kimivl.model import KimiVLForConditionalGeneration

    # Kimi-VL's text model is a DeepSeek-V3 MoE, so a stub stands in for the decoder.
    hidden_states = torch.randn(2, 5, 8)
    head = LMHead(8, 16)
    model = SimpleNamespace(
        config=SimpleNamespace(text_config=None), model=lambda **kwargs: hidden_states, lm_head=head
    )
    input_ids = torch.zeros(2, 5, dtype=torch.long)

    logits = KimiVLForConditionalGeneration.forward(model, input_ids=input_ids, output_hidden_states=True)
    # The recipe passes no return_dict: empty logits alone would drop the hidden states.
    output = KimiVLForConditionalGeneration.forward(
        model, input_ids=input_ids, output_hidden_states=True, compute_logits=False
    )

    assert logits.shape == (2, 5, 16)
    assert output.logits.shape == (2, 5, 0)
    assert output.hidden_states is hidden_states


def test_loaded_hugging_face_class_takes_compute_logits():
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        use_cache=False,
    )
    torch.manual_seed(0)
    reference = LlamaForCausalLM(config).eval()
    model = copy.deepcopy(reference)
    # The loader builds every Hugging Face model with this class.
    model.__class__ = _get_mixin_wrapped_class(LlamaForCausalLM)
    projected = []
    model.lm_head.register_forward_pre_hook(lambda module, args: projected.append(args[0].shape[:-1]))
    input_ids = torch.randint(0, 64, (2, 6), generator=torch.Generator().manual_seed(1))

    with torch.no_grad():
        expected = reference(input_ids=input_ids, output_hidden_states=True)
        with_logits = model(input_ids=input_ids, output_hidden_states=True)
        projected.clear()
        output = model(input_ids=input_ids, output_hidden_states=True, compute_logits=False)

    assert type(model).__name__ == "LlamaForCausalLM"
    # Signature-based checks still see Hugging Face's parameters, and the new option.
    parameters = inspect.signature(model.forward).parameters
    assert "logits_to_keep" in parameters and parameters["compute_logits"].default is True
    assert _supports_logits_to_keep(model)
    torch.testing.assert_close(with_logits.logits, expected.logits, rtol=0, atol=0)
    assert projected == [torch.Size([2, 1])]
    assert output.logits.shape == (2, 1, 0)
    torch.testing.assert_close(output.hidden_states[-1], expected.hidden_states[-1], rtol=0, atol=0)


def test_hugging_face_forward_without_logits_to_keep_is_not_wrapped():
    from transformers import BertForMaskedLM

    # Without logits_to_keep the head cannot be limited, and the recipe keeps a logits loss for the model.
    assert "logits_to_keep" not in inspect.signature(BertForMaskedLM.forward).parameters
    assert _get_mixin_wrapped_class(BertForMaskedLM).forward is BertForMaskedLM.forward
