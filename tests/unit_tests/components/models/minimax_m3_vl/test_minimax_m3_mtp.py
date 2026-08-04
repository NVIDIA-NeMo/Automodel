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

"""Stage-4 tests: DeepSeek-V3-style MTP module.

sglang skips MTP at load (inference-only), so there is no sglang MTP forward to
diff against. These tests therefore validate the MTP *orchestration* (roll
direction, enorm/hnorm concat order, final_layernorm, shared lm_head) and the
state-dict round-trip; the inner submodules (attention, MoE, Gemma norm, sparse
indexer) are independently parity-tested against the sglang transcription in the
Stage-1/2 tests."""

import torch

from nemo_automodel.components.models.common.mtp.mtp import roll_tensor
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3Indexer
from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3CausalLMOutput


def test_mtp_module_present_and_sparse(mtp_model):
    mtp = mtp_model.model.mtp
    assert mtp is not None and len(mtp.layers) == 1
    block = mtp.layers[0]
    # The MTP transformer_layer is a full MoE + sparse-attention decoder block.
    assert block.transformer_layer.is_moe_layer
    assert isinstance(block.transformer_layer.self_attn.indexer, MiniMaxM3Indexer)


def test_train_returns_mtp_output_eval_returns_tensor(mtp_model):
    cfg = mtp_model.config
    ids = torch.randint(2, cfg.vocab_size, (2, 16))
    mtp_model.eval()
    with torch.no_grad():
        eval_out = mtp_model(ids)
    assert isinstance(eval_out, torch.Tensor)

    mtp_model.train()
    train_out = mtp_model(ids)
    assert isinstance(train_out, MiniMaxM3CausalLMOutput)
    assert len(train_out.mtp_per_depth_logits) == 1
    assert train_out.mtp_per_depth_logits[0].shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(train_out.mtp_per_depth_logits[0]).all()


def test_mtp_forward_orchestration(mtp_model):
    """mtp_logits matches an explicit hand-wired DeepSeek-V3 MTP orchestration of the
    same submodules (eval = deterministic): checks roll direction, enorm/hnorm concat
    order, final_layernorm placement and the shared lm_head -- not the submodule math."""
    mtp_model.eval()
    cfg = mtp_model.config
    ids = torch.randint(2, cfg.vocab_size, (2, 16))
    block = mtp_model.model.mtp.layers[0]

    with torch.no_grad():
        hidden = mtp_model.model(ids)  # final normed hidden states
        mine = mtp_model.model.mtp_logits(hidden, ids, mtp_model.lm_head)[0]

        # Independent reference: roll -> embed -> eh_proj(cat[enorm(emb), hnorm(h)])
        # -> transformer_layer -> final_layernorm -> shared lm_head.
        position_ids = torch.arange(ids.shape[1]).unsqueeze(0).expand(ids.shape[0], -1)
        freqs_cis = mtp_model.model.make_freqs_cis(position_ids)
        emb = mtp_model.model.embed_tokens(roll_tensor(ids, shifts=-1, dim=-1))
        x = block.eh_proj(torch.cat([block.enorm(emb), block.hnorm(hidden)], dim=-1))
        x = block.transformer_layer(x=x, freqs_cis=freqs_cis)
        ref = mtp_model.lm_head(block.final_layernorm(x))

    assert torch.allclose(mine, ref, atol=1e-5), (mine - ref).abs().max().item()


def test_mtp_adapter_roundtrip_and_naming(mtp_model):
    adapter = mtp_model.state_dict_adapter
    native = {k: v.clone() for k, v in mtp_model.state_dict().items()}
    assert any("model.mtp.layers.0.transformer_layer.mlp.experts" in k for k in native)
    assert "model.mtp.layers.0.eh_proj.weight" in native

    hf = adapter.to_hf(native)
    # MTP transformer_layer experts split under block_sparse_moe in HF layout.
    assert any(k.startswith("model.mtp.layers.0.transformer_layer.block_sparse_moe.experts.0.w1") for k in hf)
    assert "model.mtp.layers.0.enorm.weight" in hf
    assert any("model.mtp.layers.0.transformer_layer.self_attn.index_q_proj.weight" == k for k in hf)

    back = adapter.from_hf(hf)
    assert set(back.keys()) == set(native.keys()), (
        set(native) - set(back),
        set(back) - set(native),
    )
    for key in native:
        assert torch.allclose(native[key].float(), back[key].float(), atol=1e-6), key


def test_from_hf_drops_mtp_when_disabled(model):
    """A model without MTP (num_mtp_modules=0) drops any MTP tensors on load."""
    adapter = model.state_dict_adapter
    assert not adapter._mtp_enabled
    hf = adapter.to_hf(model.state_dict())
    hf["model.mtp.layers.0.enorm.weight"] = torch.zeros(model.config.hidden_size)
    native = adapter.from_hf(hf)
    assert not any(".mtp." in k for k in native)
