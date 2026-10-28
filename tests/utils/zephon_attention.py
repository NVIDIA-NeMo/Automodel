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

"""CPU oracle for real decoder consumption of Zephon Flash attention metadata."""

from collections.abc import Mapping

import pytest
import torch
from transformers import PreTrainedModel
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.shared.packed_seq import indexed_mask_to_4d_block_causal


def compare_decoder_attention(
    model: PreTrainedModel,
    dense: Mapping[str, torch.Tensor],
    flash: Mapping[str, torch.Tensor],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare real VLM outputs/gradients under dense and public Flash metadata.

    Both batches carry IDs/labels [B,S], positions [B,S] or [3,B,S], and
    processor media tensors. The production hook constructs either [B,1,S,S]
    masks or cumulative lengths; only the unavailable GPU kernel is replaced
    with an SDPA oracle. This verifies metadata routing, not Flash kernels.
    """
    for name in dense.keys() - {"attention_mask"}:
        torch.testing.assert_close(flash[name], dense[name], atol=0, rtol=0)
    if "attention_mask" in dense:
        assert dense["attention_mask"].shape == dense["input_ids"].shape
        assert dense["attention_mask"].all()
    assert "attention_mask" not in flash and "cu_seq_lens_q" not in flash
    assert flash["_packed_seq_ids"].shape == flash["input_ids"].shape
    decoder = model.get_decoder()
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    seen = []

    def cpu_varlen_attention(module, query, key, value, attention_mask, **kwargs):
        # Q/K/V are [B,H,S,D]; public cumulative lengths cover all B*S tokens.
        assert attention_mask is None
        cu = kwargs.pop("cu_seq_lens_q")
        torch.testing.assert_close(cu, kwargs.pop("cu_seq_lens_k"))
        assert cu.dtype == torch.int32 and cu[0] == 0
        assert cu[-1] == query.shape[0] * query.shape[2]
        assert kwargs.pop("max_length_q") == kwargs.pop("max_length_k") == int(cu.diff().max())
        seen.append(cu.tolist())
        attention_ids = torch.repeat_interleave(torch.arange(1, len(cu)), cu.diff().long())
        mask = indexed_mask_to_4d_block_causal(attention_ids.view(query.shape[0], query.shape[2]))
        kwargs.pop("is_causal", None)
        return original(module, query, key, value, mask, is_causal=False, **kwargs)

    def forward(batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False).logits

    def loss(logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), dense["labels"].flatten(), reduction="sum")

    model.zero_grad()
    configure_packing("sdpa", model=model)
    expected = forward(dense)
    expected_loss = loss(expected)
    expected_loss.backward()
    gradients = {
        name: parameter.grad.clone() for name, parameter in model.named_parameters() if parameter.grad is not None
    }
    model.zero_grad()
    configure_packing("flash_attention_2", model=model)
    with monkeypatch.context() as scoped:
        # Construction and vision encoding stay real; replace only the unavailable
        # GPU kernel dispatch after selecting the real decoder's Flash mask path.
        scoped.setattr(decoder.config, "_attn_implementation", "flash_attention_2")
        scoped.setitem(ALL_ATTENTION_FUNCTIONS._local_mapping, "flash_attention_2", cpu_varlen_attention)
        actual = forward(flash)
        actual_loss = loss(actual)
        actual_loss.backward()
    assert seen
    valid = dense["_packed_seq_ids"].gt(0)
    torch.testing.assert_close(actual[valid], expected[valid], atol=2e-6, rtol=1e-5)
    torch.testing.assert_close(actual_loss, expected_loss)
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name], atol=2e-5, rtol=2e-4, msg=name)
        else:
            assert parameter.grad is None
