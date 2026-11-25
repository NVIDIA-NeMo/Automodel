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

import torch

from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM
from nemo_automodel.components.models.hy_v4_flash.state_dict_adapter import HY4FlashStateDictAdapter

from .tiny import CPU_BACKEND, tiny_config


def _model():
    torch.manual_seed(0)
    model = HY4FlashForCausalLM(tiny_config(vision={"num_hidden_layers": 0}), backend=CPU_BACKEND).float().eval()
    model.initialize_weights(dtype=torch.float32)
    return model


def test_packed_documents_equal_separate_forwards():
    """Packing must not let documents see each other: packed logits equal each document run alone."""
    model = _model()
    generator = torch.Generator().manual_seed(2)
    lengths = [21, 12, 9]
    docs = [torch.randint(3, 280, (1, n), generator=generator) for n in lengths]
    with torch.no_grad():
        alone = torch.cat([model(d).logits for d in docs], dim=1)
        packed = model(
            torch.cat(docs, dim=1),
            seq_lens=torch.tensor([lengths]),
            qkv_format="thd",
        ).logits
    torch.testing.assert_close(packed, alone, rtol=2e-4, atol=2e-4)


def test_padded_batch_rows_equal_separate_forwards():
    model = _model()
    generator = torch.Generator().manual_seed(3)
    a, b = torch.randint(3, 280, (1, 20), generator=generator), torch.randint(3, 280, (1, 13), generator=generator)
    batch = torch.full((2, 20), 2)
    batch[0], batch[1, :13] = a[0], b[0]
    mask = torch.zeros(2, 20, dtype=torch.long)
    mask[0], mask[1, :13] = 1, 1
    with torch.no_grad():
        out = model(batch, attention_mask=mask).logits
        torch.testing.assert_close(out[0], model(a).logits[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(out[1, :13], model(b).logits[0], rtol=2e-4, atol=2e-4)


def test_adapter_round_trip_restores_released_names_and_values():
    model = _model()
    adapter = HY4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    native = {k: v.detach().clone() for k, v in model.state_dict().items()}
    released = adapter.to_hf(native)
    assert not [k for k in released if "sinks_param" in k or "fc1_latent" in k or "attn_hc" in k], "native names leaked"
    assert "model.layers.0.self_attn.learnable_sink_param" in released
    assert (
        "model.layers.0.mlp.post_fc2_norm.weight" in released
        and "model.layers.1.mlp.combine_output_norm.weight" in released
    )
    assert released["model.layers.1.mlp.experts.gate_up_proj"].shape == (8, 64, 24)  # [experts, 2 * inter, latent]
    again = adapter.from_hf({k: v.clone() for k, v in released.items()})
    assert set(again) == set(native)
    for key, value in native.items():
        torch.testing.assert_close(again[key], value, rtol=0, atol=0, msg=lambda m, key=key: f"{key}: {m}")


def test_released_expert_layout_is_transposed_not_reshaped():
    """gate_up_proj[e] is [2 * inter, latent] (gate rows first); native gate_and_up_projs[e] is its transpose."""
    model = _model()
    adapter = HY4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    released = torch.arange(8 * 64 * 24, dtype=torch.float32).reshape(8, 64, 24)
    native = adapter.from_hf({"model.layers.1.mlp.experts.gate_up_proj": released})
    assert torch.equal(native["model.layers.1.mlp.experts.gate_and_up_projs"][3], released[3].T)


def test_drafter_and_extra_state_keys_are_skipped_in_both_directions():
    """The released checkpoint carries 87 ``model.dfly_layers.*`` keys (the serving-only drafter) the model has no
    parameters for; TE ``_extra_state`` buffers have no released counterpart."""
    model = _model()
    adapter = HY4FlashStateDictAdapter(model.config, model.moe_config, CPU_BACKEND, dtype=torch.float32)
    kept = "model.layers.0.self_attn.linear_gate.weight"
    native = adapter.from_hf(
        {
            "model.dfly_layers.0.input_layernorm.weight": torch.ones(4),
            "model.dfly_layers.1.mlp.down_proj.weight": torch.ones(2, 2),
            "model.layers.0.mlp._extra_state": torch.ones(1),
            kept: torch.ones(3, 3),
        }
    )
    assert set(native) == {"model.layers.0.self_attn.gate_proj.weight"}
    assert adapter.convert_single_tensor_to_hf("model.dfly_layers.0.input_layernorm.weight", torch.ones(4)) == []
    assert adapter.convert_single_tensor_to_hf("model.layers.0.mlp._extra_state", torch.ones(1)) == []
    assert adapter.forced_hf_dtype_mapping({"model.dfly_layers.0.hc_fn": torch.ones(2, dtype=torch.float32)}) == {}
