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

"""``logits_dtype`` through ``NeMoAutoModelForCausalLM``: a constructor argument of repo models."""

import logging
from unittest.mock import patch

import pytest
import torch
from torch import nn
from torch.nn.attention import SDPBackend
from transformers import LlamaConfig, PhiConfig, PretrainedConfig, Qwen2Config, Qwen3Config
from transformers import LlamaForCausalLM as HFLlamaForCausalLM

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.models.common.lm_head import LMHead

_VOCAB = 128


def _is_fp32_logits_head(head: nn.Module) -> bool:
    """Return whether ``head`` is an ``LMHead`` that writes FP32 logits."""
    return isinstance(head, LMHead) and head.logits_dtype == torch.float32


@pytest.fixture(autouse=True)
def _cpu_current_device(monkeypatch):
    # Build on the CPU on GPU runners too: the checkpoint loader queries the CUDA device when CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))


def _config(config_cls: type[PretrainedConfig], *, tied: bool) -> PretrainedConfig:
    return config_cls(
        architectures=[config_cls.__name__.removesuffix("Config") + "ForCausalLM"],
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
    )


def _build(config: PretrainedConfig, **kwargs) -> torch.nn.Module:
    torch.manual_seed(0)
    # Eager RMSNorm avoids a cold Inductor compile that the head does not depend on.
    backend = {"attn": "sdpa", "rms_norm": "torch"}
    return NeMoAutoModelForCausalLM.from_config(config, backend=backend, torch_dtype=torch.bfloat16, **kwargs)


@pytest.mark.parametrize("config_cls", [LlamaConfig, Qwen2Config, Qwen3Config])
@pytest.mark.parametrize("tied", [False, True])
def test_from_config_installs_an_fp32_logits_head(config_cls, tied):
    baseline = _build(_config(config_cls, tied=tied))
    model = _build(_config(config_cls, tied=tied), logits_dtype="float32")

    assert type(model).__name__ == type(baseline).__name__
    assert _is_fp32_logits_head(model.lm_head)
    assert (model.lm_head.weight is model.get_input_embeddings().weight) is tied
    assert list(model.state_dict()) == list(baseline.state_dict())
    assert model.lm_head.weight.dtype == torch.bfloat16

    input_ids = torch.randint(0, _VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    logits = model(input_ids=input_ids).logits
    hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    reference = hidden.double() @ model.lm_head.weight.double().t()
    assert logits.dtype == torch.float32
    torch.testing.assert_close(logits.double(), reference, rtol=0, atol=1e-5)
    assert (logits == logits.bfloat16().float()).float().mean() < 0.01

    logits.sum().backward()
    assert model.lm_head.weight.grad is not None and torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize("tied", [False, True])
def test_checkpoint_loads_into_the_fp32_logits_head(tmp_path, tied):
    # Custom models initialize on the meta device: the head is configured before the
    # checkpoint loads, so loading and re-tying must reach the FP32-logits head.
    torch.manual_seed(3)
    checkpoint = HFLlamaForCausalLM(_config(LlamaConfig, tied=tied)).to(torch.bfloat16)
    checkpoint.save_pretrained(tmp_path)
    config = _config(LlamaConfig, tied=tied)
    config._name_or_path = str(tmp_path)

    model = _build(config, logits_dtype="float32", load_base_model=True)

    assert _is_fp32_logits_head(model.lm_head)
    assert (model.lm_head.weight is model.get_input_embeddings().weight) is tied
    torch.testing.assert_close(model.lm_head.weight, checkpoint.lm_head.weight, rtol=0, atol=0)
    torch.testing.assert_close(
        model.get_input_embeddings().weight, checkpoint.get_input_embeddings().weight, rtol=0, atol=0
    )
    input_ids = torch.randint(0, _VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    output = model(input_ids=input_ids, output_hidden_states=True)
    reference = output.hidden_states[-1].double() @ checkpoint.lm_head.weight.double().t()
    assert output.logits.dtype == torch.float32
    torch.testing.assert_close(output.logits.double(), reference, rtol=0, atol=1e-5)


@pytest.mark.parametrize("tied", [False, True])
def test_vocabulary_resize_keeps_the_fp32_logits_head(tied):
    # Transformers builds a plain nn.Linear for a resized untied head; the LMHead takes its parameters instead.
    model = _build(_config(Qwen3Config, tied=tied), logits_dtype="float32")

    model.resize_token_embeddings(_VOCAB + 1)

    assert _is_fp32_logits_head(model.lm_head)
    assert model.lm_head.weight.shape == (_VOCAB + 1, 32)
    assert (model.lm_head.weight is model.get_input_embeddings().weight) is tied
    input_ids = torch.randint(0, _VOCAB + 1, (2, 6), generator=torch.Generator().manual_seed(1))
    output = model(input_ids=input_ids, output_hidden_states=True)
    reference = output.hidden_states[-1].double() @ model.lm_head.weight.double().t()
    assert output.logits.dtype == torch.float32
    torch.testing.assert_close(output.logits.double(), reference, rtol=0, atol=1e-5)


def test_logits_dtype_survives_the_build_retry():
    # A failed SDPA patch rebuilds the model through _build_model's retry path.
    with patch(
        "nemo_automodel._transformers.auto_model._patch_attention", side_effect=RuntimeError("unsupported")
    ) as patch_attention:
        model = _build(_config(Qwen3Config, tied=False), logits_dtype="float32", sdpa_method=[SDPBackend.MATH])

    assert patch_attention.call_count == 1
    assert _is_fp32_logits_head(model.lm_head)
    input_ids = torch.randint(0, _VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    assert model(input_ids=input_ids).logits.dtype == torch.float32


def _hf_config(config_cls: type[PretrainedConfig], **overrides) -> PretrainedConfig:
    return config_cls(
        architectures=[config_cls.__name__.removesuffix("Config") + "ForCausalLM"],
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        **overrides,
    )


@pytest.mark.parametrize(
    "config",
    [
        pytest.param(_hf_config(LlamaConfig, head_dim=8), id="llama-force-hf"),
        # Phi has no repo implementation, so the loader falls back to the Hugging Face class.
        pytest.param(_hf_config(PhiConfig), id="phi"),
    ],
)
def test_hugging_face_models_ignore_logits_dtype_with_a_warning(config, caplog):
    torch.manual_seed(0)
    with caplog.at_level(logging.WARNING):
        model = NeMoAutoModelForCausalLM.from_config(
            # Liger kernels would replace the HF norms with Triton kernels that cannot run on CPU, and the default
            # attention is FlashAttention 2 wherever flash-attn is installed.
            config,
            force_hf=True,
            torch_dtype=torch.bfloat16,
            logits_dtype="float32",
            use_liger_kernel=False,
            attn_implementation="eager",
        )

    assert "logits_dtype is ignored" in caplog.text
    assert type(model.get_output_embeddings()) is nn.Linear
    input_ids = torch.randint(0, _VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    assert model(input_ids=input_ids).logits.dtype == torch.bfloat16


def test_any_logits_dtype_is_applied():
    model = _build(_config(Qwen3Config, tied=False), logits_dtype="bfloat16")

    assert isinstance(model.lm_head, LMHead) and model.lm_head.logits_dtype == torch.bfloat16
    input_ids = torch.randint(0, _VOCAB, (2, 6), generator=torch.Generator().manual_seed(1))
    assert model(input_ids=input_ids).logits.dtype == torch.bfloat16
