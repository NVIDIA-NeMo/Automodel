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

"""Unit tests for the NeMo AutoModel Inkling MoE VLM implementation."""

import pytest
import torch
import torch.nn.functional as F

transformers_inkling = pytest.importorskip("transformers.models.inkling.modeling_inkling")

from transformers.models.inkling.modeling_inkling import (  # noqa: E402
    InklingForConditionalGeneration as HFInklingForConditionalGeneration,
)
from transformers.models.inkling.modeling_inkling import InklingMoE as HFInklingMoE  # noqa: E402

from nemo_automodel.components.models.common import BackendConfig  # noqa: E402
from nemo_automodel.components.models.inkling.layers import InklingMoE  # noqa: E402
from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration  # noqa: E402

from .parity_check_inkling import build_tiny_config  # noqa: E402


def _build_models():
    cfg = build_tiny_config()
    hf = HFInklingForConditionalGeneration(cfg).to(dtype=torch.float32).eval()
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch")
    nemo = InklingForConditionalGeneration.from_config(cfg, backend=backend).to(dtype=torch.float32).eval()
    return cfg, hf, nemo


def test_sparse_layers_use_inkling_moe():
    cfg, _, nemo = _build_models()
    layers = nemo.model.language_model.layers
    mlp_types = cfg.text_config.mlp_layer_types
    for i, layer in enumerate(layers):
        if mlp_types[i] == "sparse":
            assert isinstance(layer.mlp, InklingMoE)
            assert not isinstance(layer.mlp, HFInklingMoE)
        else:
            assert not isinstance(layer.mlp, InklingMoE)


def test_state_dict_adapter_roundtrip_exact():
    _, hf, nemo = _build_models()
    adapter = nemo.state_dict_adapter
    hf_sd = hf.state_dict()
    roundtrip = adapter.to_hf(adapter.from_hf(hf_sd))
    assert set(roundtrip.keys()) == set(hf_sd.keys())
    for k in hf_sd:
        assert torch.equal(hf_sd[k], roundtrip[k]), f"round-trip mismatch for {k}"


def test_from_hf_load_has_no_missing_or_unexpected_keys():
    _, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    missing, unexpected = nemo.load_state_dict(native_sd, strict=False)
    ignore = lambda ks: [k for k in ks if "rotary" not in k and "inv_freq" not in k]
    assert ignore(missing) == []
    assert ignore(unexpected) == []


def test_logit_parity_kl_below_1e_3():
    cfg, hf, nemo = _build_models()
    native_sd = nemo.state_dict_adapter.from_hf(hf.state_dict())
    nemo.load_state_dict(native_sd, strict=False)

    torch.manual_seed(0)
    input_ids = torch.randint(0, cfg.text_config.vocab_size, (1, 24))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        hf_logits = hf(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        nemo_logits = nemo(input_ids=input_ids, attention_mask=attention_mask).logits.float()

    assert hf_logits.shape == nemo_logits.shape
    kl = F.kl_div(
        F.log_softmax(nemo_logits, dim=-1),
        F.log_softmax(hf_logits, dim=-1),
        log_target=True,
        reduction="batchmean",
    ).item()
    assert kl < 1e-3, f"KL divergence too high: {kl}"
    assert (hf_logits - nemo_logits).abs().max().item() < 1e-2
