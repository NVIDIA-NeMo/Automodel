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

"""Kimi K3.1 gated DSA: config, indexer selection and the sparse MLA paths."""

import copy
from datetime import timedelta
from pathlib import Path
from unittest import mock

import pytest
import torch

import nemo_automodel.components.models.kimi_k3.cp as kimi_cp
import nemo_automodel.components.models.kimi_k3.model as kimi_model
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3Config, KimiK3TextConfig, KimiK31Config
from nemo_automodel.components.models.kimi_k3.cp import KimiPackedContext
from nemo_automodel.components.models.kimi_k3.model import (
    KimiDSAIndexer,
    KimiDSAIndexerLayerNorm,
    KimiK3ForCausalLM,
    KimiMLAAttention,
    KimiRMSNorm,
)
from nemo_automodel.components.models.kimi_k3.multimodal import KimiK31ForConditionalGeneration

# Released ``moonshotai/Kimi-K3.1-Pro`` text_config.dsa_config.
_RELEASED_DSA_CONFIG = {
    "dsa_layers": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93],
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_num_heads": 32,
    "index_rope_dim": 0,
    "index_k_norm_type": "layernorm",
    "index_topk_freq": 1,
    "index_skip_topk_offset": 1,
}


def _backend(attn: str = "eager") -> BackendConfig:
    return BackendConfig(
        attn=attn,
        linear="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=False,
    )


def _tiny_text_config(dsa_config=None, *, num_hidden_layers: int = 2, q_lora_rank: int | None = 16):
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=32,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=None,
        num_shared_experts=0,
        first_k_dense_replace=num_hidden_layers,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": list(range(1, num_hidden_layers + 1)),
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        dsa_config=dsa_config,
        attn_res_block_size=None,
    )


def _tiny_dsa_config(**overrides):
    return {"dsa_layers": [2], "index_topk": 3, "index_head_dim": 8, "index_num_heads": 2, **overrides}


def _tiny_dsa_attention(attn: str = "eager", **dsa_overrides) -> KimiMLAAttention:
    torch.manual_seed(0)
    config = _tiny_text_config(_tiny_dsa_config(**dsa_overrides))
    attention = KimiMLAAttention(config, layer_idx=1, backend=_backend(attn))
    attention.init_weights(torch.device("cpu"), init_std=0.2)
    return attention


def _causal_mask(batch_size: int, seq_length: int) -> torch.Tensor:
    blocked = torch.ones(seq_length, seq_length, dtype=torch.bool).triu(1)
    mask = torch.zeros(seq_length, seq_length).masked_fill(blocked, torch.finfo(torch.float32).min)
    return mask[None, None].expand(batch_size, 1, -1, -1)


def _reference_scores(indexer: KimiDSAIndexer, hidden_states: torch.Tensor, query_input: torch.Tensor) -> torch.Tensor:
    """Compute the indexer score ``sum_h relu(q_h . k) * w_h`` densely with one einsum.

    Args:
        indexer: Indexer whose projections are evaluated.
        hidden_states: Tensor of shape [batch, sequence, hidden].
        query_input: Tensor of shape [batch, sequence, q_lora_rank].

    Returns:
        Tensor of shape [batch, query_sequence, key_sequence].
    """
    batch_size, seq_length = hidden_states.shape[:2]
    queries = indexer.wq_b(query_input).view(batch_size, seq_length, indexer.num_heads, indexer.head_dim)
    keys = indexer.k_norm(indexer.wk(hidden_states))
    weights = indexer.weights_proj(hidden_states) * (indexer.head_dim * indexer.num_heads) ** -0.5
    logits = torch.einsum("bshd,btd->bsht", queries, keys).relu()
    return (logits * weights[..., None]).sum(2)


# --- config ---------------------------------------------------------------------------------------


def test_k3_config_has_no_dsa_layers():
    config = KimiK3TextConfig()
    assert config.dsa_config is None
    assert not any(config.is_dsa_layer(layer_idx) for layer_idx in range(config.num_hidden_layers))


def test_k31_config_defaults_match_released_checkpoint():
    config = KimiK31Config()
    text_config = config.text_config
    assert config.model_type == "kimi_k31"
    assert config.architectures == ["KimiK31ForConditionalGeneration"]
    assert text_config.dsa_config == _RELEASED_DSA_CONFIG
    assert text_config.dsa_config["dsa_layers"] == text_config.linear_attn_config["full_attn_layers"]
    dsa_layer_indices = [layer_idx for layer_idx in range(93) if text_config.is_dsa_layer(layer_idx)]
    assert dsa_layer_indices == [layer - 1 for layer in _RELEASED_DSA_CONFIG["dsa_layers"]]


def test_k31_config_round_trips_through_dict():
    config = KimiK31Config(text_config=_tiny_text_config(_tiny_dsa_config()).to_dict())
    restored = KimiK31Config(**config.to_dict())
    assert restored.text_config.dsa_config == config.text_config.dsa_config
    assert restored.text_config.is_dsa_layer(1) and not restored.text_config.is_dsa_layer(0)


def test_dsa_config_fills_defaults():
    config = _tiny_text_config({})
    assert config.dsa_config["dsa_layers"] == [1, 2]
    assert config.dsa_config["index_topk"] == 2048


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dsa_layers": [3]}, "full-attention layers"),
        ({"index_topk": 0}, "index_topk must be positive"),
        ({"index_k_norm_type": "batchnorm"}, "index_k_norm_type"),
        ({"index_rope_dim": 64}, "index_rope_dim=0"),
        ({"index_topk_freq": 2}, "index_topk_freq=1"),
    ],
)
def test_dsa_config_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _tiny_text_config(_tiny_dsa_config(**overrides))


def test_truncated_model_drops_dsa_layers_beyond_the_cut():
    config = _tiny_text_config(_tiny_dsa_config(dsa_layers=[1, 2]))
    model = KimiK3ForCausalLM(config, backend=_backend(), num_hidden_layers=1)
    assert model.config.dsa_config["dsa_layers"] == [1]
    assert config.dsa_config["dsa_layers"] == [1, 2]


def test_dflash_draft_config_drops_target_dsa_layers():
    from nemo_automodel.components.speculative.dflash.draft_kimi_k3 import build_kimi_k3_dflash_draft_config

    target_config = _tiny_text_config(_tiny_dsa_config())
    draft_config = build_kimi_k3_dflash_draft_config(
        target_config,
        num_draft_layers=2,
        num_target_layers=2,
        block_size=4,
        dflash_config={"target_layer_ids": [0, 1]},
        attention_backend="sdpa",
    )
    assert draft_config.dsa_config is None
    assert target_config.dsa_config is not None


# --- model structure ------------------------------------------------------------------------------


def test_only_dsa_layers_own_an_indexer_with_checkpoint_names():
    model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
    assert model.model.layers["0"].self_attn.indexer is None
    indexer = model.model.layers["1"].self_attn.indexer
    assert isinstance(indexer, KimiDSAIndexer)
    assert isinstance(indexer.k_norm, KimiDSAIndexerLayerNorm)
    indexer_keys = sorted(key for key in model.state_dict() if ".indexer." in key)
    assert indexer_keys == [
        "model.layers.1.self_attn.indexer.k_norm.bias",
        "model.layers.1.self_attn.indexer.k_norm.weight",
        "model.layers.1.self_attn.indexer.weights_proj.weight",
        "model.layers.1.self_attn.indexer.wk.weight",
        "model.layers.1.self_attn.indexer.wq_b.weight",
    ]
    assert indexer.wq_b.weight.shape == (2 * 8, 16)
    assert indexer.wk.weight.shape == (8, 32)
    assert indexer.weights_proj.weight.shape == (2, 32)


def test_rmsnorm_key_norm_and_query_from_hidden_without_q_lora():
    config = _tiny_text_config(_tiny_dsa_config(index_k_norm_type="rmsnorm"), q_lora_rank=None)
    attention = KimiMLAAttention(config, layer_idx=1, backend=_backend())
    assert isinstance(attention.indexer.k_norm, KimiRMSNorm)
    assert attention.indexer.wq_b.weight.shape == (2 * 8, 32)
    hidden_states = torch.randn(1, 6, 32)
    output = attention(hidden_states, attention_mask=_causal_mask(1, 6))
    assert output.shape == (1, 6, 32) and torch.isfinite(output).all()


def test_indexer_init_resets_key_norm():
    attention = _tiny_dsa_attention()
    with torch.no_grad():
        attention.indexer.k_norm.weight.fill_(3.0)
        attention.indexer.k_norm.bias.fill_(1.0)
    attention.init_weights(torch.device("cpu"), init_std=0.02)
    torch.testing.assert_close(attention.indexer.k_norm.weight, torch.ones(8))
    torch.testing.assert_close(attention.indexer.k_norm.bias, torch.zeros(8))


def test_indexer_layernorm_computes_in_fp32():
    norm = KimiDSAIndexerLayerNorm(8, eps=1e-5, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 8, dtype=torch.bfloat16)
    expected = torch.nn.functional.layer_norm(hidden_states.float(), (8,), eps=1e-5).bfloat16()
    output = norm(hidden_states)
    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, expected)


# --- indexer selection ----------------------------------------------------------------------------


def test_indexer_selects_the_highest_scoring_visible_keys():
    attention = _tiny_dsa_attention()
    indexer = attention.indexer
    hidden_states = torch.randn(2, 9, 32)
    query_input, _ = attention._project_queries(hidden_states)
    allowed = torch.ones(9, 9, dtype=torch.bool).tril().expand(2, -1, -1)

    selected = indexer.select(hidden_states, query_input, indexer.project_keys(hidden_states), allowed)

    assert not (selected & ~allowed).any()
    torch.testing.assert_close(selected.sum(-1), allowed.sum(-1).clamp(max=indexer.index_topk))
    scores = _reference_scores(indexer, hidden_states, query_input)
    # Tie-robust top-k check: no unselected visible key outscores a selected one.
    lowest_selected = scores.masked_fill(~selected, float("inf")).amin(-1)
    highest_rejected = scores.masked_fill(~(allowed & ~selected), float("-inf")).amax(-1)
    assert (highest_rejected <= lowest_selected).all()


def test_indexer_scores_query_chunks_independently(monkeypatch):
    attention = _tiny_dsa_attention()
    indexer = attention.indexer
    hidden_states = torch.randn(1, 10, 32)
    query_input, _ = attention._project_queries(hidden_states)
    keys = indexer.project_keys(hidden_states)
    allowed = torch.ones(10, 10, dtype=torch.bool).tril()[None]
    full = indexer.select(hidden_states, query_input, keys, allowed)
    monkeypatch.setattr(KimiDSAIndexer, "_query_chunk_size", 3)
    chunked = indexer.select(hidden_states, query_input, keys, allowed)
    # ReLU makes zero-score ties common, so compare the selected scores rather than the indices.
    scores = _reference_scores(indexer, hidden_states, query_input)
    torch.testing.assert_close(chunked.sum(-1), full.sum(-1))
    torch.testing.assert_close((scores * chunked).sum(-1), (scores * full).sum(-1))


def test_indexer_keeps_every_visible_key_when_the_sequence_fits_the_budget():
    attention = _tiny_dsa_attention(index_topk=8)
    allowed = torch.ones(8, 8, dtype=torch.bool).tril()[None]
    hidden_states = torch.randn(1, 8, 32)
    with mock.patch.object(attention.indexer.wq_b, "forward", side_effect=AssertionError("must not score")):
        selected = attention.indexer.select(hidden_states, hidden_states, torch.randn(1, 8, 8), allowed)
    assert selected is allowed


def test_indexer_runs_without_autograd():
    attention = _tiny_dsa_attention()
    hidden_states = torch.randn(1, 7, 32, requires_grad=True)
    output = attention(hidden_states, attention_mask=_causal_mask(1, 7))
    output.sum().backward()
    assert hidden_states.grad is not None
    assert attention.q_b_proj.weight.grad is not None
    assert all(parameter.grad is None for parameter in attention.indexer.parameters())


# --- sparse attention paths -----------------------------------------------------------------------


def _dense_attention_like(attention: KimiMLAAttention) -> KimiMLAAttention:
    dense = KimiMLAAttention(_tiny_text_config(None), layer_idx=1, backend=attention.backend)
    dense.load_state_dict(attention.state_dict(), strict=False)
    return dense


def test_dsa_equals_dense_mla_when_sequence_fits_the_budget():
    attention = _tiny_dsa_attention(index_topk=16)
    hidden_states = torch.randn(2, 8, 32)
    mask = _causal_mask(2, 8)
    torch.testing.assert_close(
        attention(hidden_states, attention_mask=mask),
        _dense_attention_like(attention)(hidden_states, attention_mask=mask),
    )


def test_dsa_restricts_each_query_to_its_selected_keys():
    attention = _tiny_dsa_attention(index_topk=3)
    hidden_states = torch.randn(1, 8, 32)
    mask = _causal_mask(1, 8)
    output = attention(hidden_states, attention_mask=mask)
    query_input, _ = attention._project_queries(hidden_states)
    keep = attention._select_sparse_keys(hidden_states, query_input, mask[:, 0].eq(0)).unsqueeze(1)
    sparse_mask = torch.zeros(keep.shape).masked_fill(~keep, torch.finfo(torch.float32).min)
    expected = _dense_attention_like(attention)(hidden_states, attention_mask=sparse_mask)
    torch.testing.assert_close(output, expected)
    # Rows past the budget really are sparse, so the result differs from dense attention.
    dense = _dense_attention_like(attention)(hidden_states, attention_mask=mask)
    assert not torch.allclose(output[:, 3:], dense[:, 3:])
    torch.testing.assert_close(output[:, :3], dense[:, :3])


def test_dsa_without_mask_assumes_causal_attention():
    attention = _tiny_dsa_attention()
    hidden_states = torch.randn(2, 7, 32)
    torch.testing.assert_close(attention(hidden_states), attention(hidden_states, attention_mask=_causal_mask(2, 7)))


def test_dsa_sdpa_matches_eager():
    eager = _tiny_dsa_attention("eager")
    sdpa = _tiny_dsa_attention("sdpa")
    sdpa.load_state_dict(eager.state_dict())
    hidden_states = torch.randn(2, 9, 32)
    mask = _causal_mask(2, 9)
    torch.testing.assert_close(sdpa(hidden_states, attention_mask=mask), eager(hidden_states, attention_mask=mask))


def test_dsa_te_receives_an_arbitrary_mask_of_masked_positions():
    attention = _tiny_dsa_attention()
    hidden_states = torch.randn(1, 6, 32)
    mask = _causal_mask(1, 6)
    query_input, _ = attention._project_queries(hidden_states)
    keep = attention._select_sparse_keys(hidden_states, query_input, mask[:, 0].eq(0)).unsqueeze(1)
    captured = {}

    def fake_te(query, key, value, **kwargs):
        """Record the TE keyword arguments and return zeros.

        Args:
            query: Tensor of shape [batch, sequence, heads, q_head_dim].
            key: Tensor of shape [batch, sequence, heads, q_head_dim].
            value: Tensor of shape [batch, sequence, heads, v_head_dim].
            **kwargs: TE attention options.

        Returns:
            Tensor of shape [batch, sequence, heads * v_head_dim].
        """
        captured.update(kwargs)
        return torch.zeros(*query.shape[:2], query.shape[2] * 8)

    attention.backend = copy.copy(attention.backend)
    attention.backend.attn = "te"
    attention.attn_func = fake_te
    attention(hidden_states, attention_mask=mask)
    assert captured["attn_mask_type"] == "arbitrary"
    torch.testing.assert_close(captured["attention_mask"], keep.logical_not())


def test_dsa_packed_documents_never_select_across_documents():
    attention = _tiny_dsa_attention(index_topk=2)
    doc_ids = torch.tensor([[1, 1, 1, 1, 2, 2, 2, 2, 2]])
    mask = kimi_cp.build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)
    hidden_states = torch.randn(1, 9, 32)
    query_input, _ = attention._project_queries(hidden_states)
    keep = attention._select_sparse_keys(hidden_states, query_input, mask[:, 0].eq(0))
    same_document = doc_ids[:, :, None] == doc_ids[:, None, :]
    assert not (keep & ~same_document).any()
    torch.testing.assert_close(keep.sum(-1), torch.tensor([[1, 2, 2, 2, 1, 2, 2, 2, 2]]))


def test_dsa_context_parallel_path_matches_full_sequence(monkeypatch):
    from torch.nn.attention.flex_attention import flex_attention

    attention = _tiny_dsa_attention()
    doc_ids = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2]])
    hidden_states = torch.randn(1, 8, 32)
    mask = kimi_cp.build_document_causal_mask(doc_ids, doc_ids, q_global_start=0, dtype=torch.float32)
    with torch.no_grad():
        expected = attention(hidden_states, attention_mask=mask)

    monkeypatch.setattr(kimi_cp, "_compiled_flex_attention", lambda: flex_attention)
    monkeypatch.setattr(kimi_model, "all_gather_sequence", lambda tensor, group, dim: tensor)
    attention.setup_cp_attention(mock.Mock())
    # FlexAttention has no CPU backward, so this compares the forward only.
    with torch.no_grad():
        actual = attention._forward_with_cp(hidden_states, KimiPackedContext(doc_ids=doc_ids))
    torch.testing.assert_close(actual, expected)


def test_sparse_keep_block_mask_reads_the_keep_tensor():
    keep = torch.rand(2, 4, 6) > 0.5
    block_mask = kimi_cp.sparse_keep_block_mask(keep)
    for batch_idx in range(2):
        actual = block_mask.mask_mod(
            torch.tensor(batch_idx), torch.tensor(0), torch.arange(4)[:, None], torch.arange(6)[None, :]
        )
        torch.testing.assert_close(actual, keep[batch_idx])


def test_full_model_forward_and_backward_with_dsa():
    torch.manual_seed(0)
    model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"))
    indexer = model.model.layers["1"].self_attn.indexer
    assert all(torch.isfinite(parameter).all() for parameter in indexer.parameters())
    torch.testing.assert_close(indexer.k_norm.bias, torch.zeros_like(indexer.k_norm.bias))
    input_ids = torch.randint(1, 64, (2, 9))
    logits = model(input_ids=input_ids).logits
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), input_ids.flatten())
    loss.backward()
    assert torch.isfinite(loss)
    assert model.model.layers["1"].self_attn.q_b_proj.weight.grad is not None


# --- registration and FLOPs -----------------------------------------------------------------------


def test_k31_is_registered():
    from nemo_automodel._transformers.registry import ModelRegistry, resolve_custom_config_cls
    from nemo_automodel._transformers.tokenization.registry import TokenizerRegistry

    assert resolve_custom_config_cls("kimi_k31") is KimiK31Config
    assert ModelRegistry.model_arch_name_to_cls["KimiK31ForConditionalGeneration"] is KimiK31ForConditionalGeneration
    assert TokenizerRegistry.has_custom_tokenizer("kimi_k31")


def test_k31_checkpoint_config_json_resolves_the_local_config(tmp_path):
    import json

    from nemo_automodel._transformers.model_init import get_hf_config

    text_config = _tiny_text_config(_tiny_dsa_config()).to_dict()
    text_config["architectures"] = ["KimiLinearForCausalLM"]
    checkpoint_config = {
        "architectures": ["KimiK31ForConditionalGeneration"],
        "auto_map": {"AutoConfig": "configuration_kimi_k3.KimiK3Config"},
        "model_type": "kimi_k31",
        "text_config": text_config,
        "vision_config": {"model_type": "kimi_k3_vision", "vt_hidden_size": 16, "text_hidden_size": 32},
    }
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_config))

    loaded = get_hf_config(str(tmp_path), "eager", trust_remote_code=False)

    assert isinstance(loaded, KimiK31Config)
    assert loaded.architectures == ["KimiK31ForConditionalGeneration"]
    assert loaded.text_config.dsa_config == {
        **_tiny_dsa_config(),
        "index_rope_dim": 0,
        "index_k_norm_type": "layernorm",
        "index_topk_freq": 1,
        "index_skip_topk_offset": 1,
    }


def test_k31_from_pretrained_reads_the_k31_config():
    config = KimiK31Config()
    with (
        mock.patch.object(KimiK31Config, "from_pretrained", return_value=config) as from_pretrained,
        mock.patch.object(KimiK31ForConditionalGeneration, "from_config", return_value="model") as from_config,
    ):
        assert KimiK31ForConditionalGeneration.from_pretrained("moonshotai/Kimi-K3.1-Pro") == "model"
    from_pretrained.assert_called_once_with("moonshotai/Kimi-K3.1-Pro")
    from_config.assert_called_once_with(config)


def test_flops_cost_dsa_layers_as_sparse_attention_plus_indexer():
    from nemo_automodel.components.utils.flops_utils import (
        _mla_attention_per_layer_flops,
        get_flops_formula_for_hf_config,
        kimi_k3_flops,
    )

    k3 = KimiK3Config()
    k31 = KimiK31Config()
    seq_len = 32768
    assert get_flops_formula_for_hf_config(k31) is kimi_k3_flops
    text = k31.text_config
    mla_args = (
        1,
        seq_len,
        text.hidden_size,
        text.num_attention_heads,
        text.q_lora_rank,
        text.kv_lora_rank,
        text.qk_rope_head_dim,
        text.qk_nope_head_dim,
        text.v_head_dim,
    )
    # Sparse attention over 2048 keys replaces causal attention; the indexer runs forward-only (2x).
    indexer = 2 * seq_len * ((1536 * 32 * 128 + 7168 * (128 + 32)) + 32 * 128 * seq_len)
    per_layer_delta = (
        _mla_attention_per_layer_flops(*mla_args, index_topk=2048) - _mla_attention_per_layer_flops(*mla_args) + indexer
    )
    delta = kimi_k3_flops(k31, seq_len=seq_len) - kimi_k3_flops(k3, seq_len=seq_len)
    assert delta == pytest.approx(24 * per_layer_delta)


# --- review follow-ups ----------------------------------------------------------------------------


def test_indexer_parameters_are_frozen():
    attention = _tiny_dsa_attention()
    assert not any(parameter.requires_grad for parameter in attention.indexer.parameters())
    assert attention.q_b_proj.weight.requires_grad


def test_short_sequence_takes_the_dense_causal_path():
    attention = _tiny_dsa_attention("sdpa", index_topk=8)
    hidden_states = torch.randn(1, 8, 32)
    captured = {}
    original = attention.attn_func

    def spy(query, key, value, **kwargs):
        """Record the SDPA keyword arguments.

        Args:
            query: Tensor of shape [batch, heads, sequence, q_head_dim].
            key: Tensor of shape [batch, heads, sequence, q_head_dim].
            value: Tensor of shape [batch, heads, sequence, v_head_dim].
            **kwargs: SDPA options.

        Returns:
            Tensor of shape [batch, heads, sequence, v_head_dim].
        """
        captured.update(kwargs)
        return original(query, key, value, **kwargs)

    attention.attn_func = spy
    with mock.patch.object(KimiDSAIndexer, "select", side_effect=AssertionError("must not select")):
        attention(hidden_states, attention_mask=_causal_mask(1, 8))
    assert captured["is_causal"] is True and "attn_mask" not in captured


def test_short_context_parallel_sequence_uses_the_dense_document_mask(monkeypatch):
    attention = _tiny_dsa_attention(index_topk=8)
    doc_ids = torch.tensor([[1, 1, 1, 2, 2, 2, 2, 2]])
    context = KimiPackedContext(doc_ids=doc_ids)
    monkeypatch.setattr(kimi_model, "all_gather_sequence", lambda tensor, group, dim: tensor)
    flex = mock.Mock(side_effect=lambda query, key, value, **kwargs: torch.zeros(*query.shape[:3], 8))
    monkeypatch.setattr(kimi_model, "document_causal_flex_attention", flex)
    attention.setup_cp_attention(mock.Mock())
    with mock.patch.object(KimiDSAIndexer, "select", side_effect=AssertionError("must not select")):
        attention._forward_with_cp(torch.randn(1, 8, 32), context)
    assert flex.call_args.kwargs["block_mask"] is context.mla_block_mask


def test_k31_config_defaults_a_missing_dsa_block():
    text_config = _tiny_text_config(None).to_dict()
    config = KimiK31Config(text_config=text_config)
    assert config.text_config.dsa_config["dsa_layers"] == [1, 2]
    with pytest.raises(ValueError, match="requires a text_config with a dsa_config"):
        KimiK31Config(text_config=_tiny_text_config(None))


def test_flops_short_sequences_cost_dsa_layers_as_dense_mla():
    from nemo_automodel.components.utils.flops_utils import kimi_k3_flops

    assert kimi_k3_flops(KimiK31Config(), seq_len=2048) == kimi_k3_flops(KimiK3Config(), seq_len=2048)
    assert kimi_k3_flops(KimiK31Config(), seq_len=2049) != kimi_k3_flops(KimiK3Config(), seq_len=2049)


@pytest.mark.cpu_only
@pytest.mark.parametrize("targeting", [{}, {"target_modules": ["*_proj"]}, {"match_all_linear": True}])
def test_indexer_stays_frozen_after_peft_trainability_rebinding(targeting: dict[str, list[str] | bool]) -> None:
    from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
    from nemo_automodel.shared.peft_config import PeftConfig

    model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
    apply_lora_to_linear_modules(model, PeftConfig(dim=2, alpha=2, **targeting))
    indexer = model.model.layers["1"].self_attn.indexer
    assert any("lora_" in name for name, _ in indexer.named_parameters())

    # Loading or sharding can replace Parameters and restore requires_grad=True.
    indexer.wk.weight = torch.nn.Parameter(indexer.wk.weight.detach().clone())
    for _ in range(2):
        _apply_trainability_policy(model, peft_enabled=True, freeze_config=None, strict=True)
        assert not any(parameter.requires_grad for parameter in indexer.parameters())
        assert model.model.layers["1"].self_attn.q_b_proj.lora_A.weight.requires_grad


@pytest.mark.cpu_only
def test_indexer_freeze_overrides_explicit_unfreeze_on_wrapped_models() -> None:
    from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
    from nemo_automodel.components.utils.model_utils import parse_freeze_config

    model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
    wrapped = torch.nn.Sequential(model)
    freeze_config = parse_freeze_config({"unfreeze_modules": [{"path": "0.model.layers.1.self_attn.indexer"}]})
    _apply_trainability_policy(wrapped, peft_enabled=False, freeze_config=freeze_config, strict=True)
    assert not any(parameter.requires_grad for parameter in model.model.layers["1"].self_attn.indexer.parameters())
    assert model.model.layers["1"].self_attn.q_b_proj.weight.requires_grad


@pytest.mark.cpu_only
def test_indexer_lora_checkpoint_resume_matches_next_step(tmp_path: Path) -> None:
    from nemo_automodel._transformers.infrastructure import _apply_trainability_policy
    from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
    from nemo_automodel.shared.peft_config import PeftConfig

    torch.manual_seed(11)
    model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
    model.initialize_weights(buffer_device=torch.device("cpu"))
    apply_lora_to_linear_modules(model, PeftConfig(dim=2, alpha=2))
    _apply_trainability_policy(model, peft_enabled=True, freeze_config=None, strict=True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    input_ids = torch.arange(1, 10).unsqueeze(0)
    model(input_ids=input_ids).logits.square().mean().backward()
    optimizer.step()
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint_path)

    restored = copy.deepcopy(model)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    # assign=True models checkpoint loading that replaces the Parameter objects.
    restored.load_state_dict(checkpoint["model"], assign=True)
    _apply_trainability_policy(restored, peft_enabled=True, freeze_config=None, strict=True)
    restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad], lr=0.01)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    for current_model, current_optimizer in ((model, optimizer), (restored, restored_optimizer)):
        current_optimizer.zero_grad(set_to_none=True)
        current_model(input_ids=input_ids).logits.square().mean().backward()
        current_optimizer.step()
        assert not any(p.requires_grad for p in current_model.model.layers["1"].self_attn.indexer.parameters())
        assert set(current_optimizer.state) == {p for p in current_model.parameters() if p.requires_grad}
    for (name, parameter), (restored_name, restored_parameter) in zip(
        model.named_parameters(), restored.named_parameters()
    ):
        assert name == restored_name
        torch.testing.assert_close(parameter, restored_parameter, rtol=0, atol=0)


def _run_indexer_lora_ddp(rank: int, world_size: int, init_file: str) -> None:
    import torch.distributed as dist

    from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
    from nemo_automodel.components.distributed import DDPConfig, MeshContext
    from nemo_automodel.shared.peft_config import PeftConfig

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    try:
        torch.manual_seed(7)
        model = KimiK3ForCausalLM(_tiny_text_config(_tiny_dsa_config()), backend=_backend())
        model.initialize_weights(buffer_device=torch.device("cpu"))
        # Compare DDP reductions with a single-process FP32 reference.
        model.float()
        model = apply_model_infrastructure(
            model=model,
            is_meta_device=False,
            device=torch.device("cpu"),
            load_base_model=False,
            model_wrapper=MeshContext(strategy_config=DDPConfig()),
            peft_config=PeftConfig(dim=2, alpha=2, lora_dtype=torch.float32),
        )
        # A single-rank strategy can return the unwrapped model. Always exercise
        # a real reducer, including its second-forward unused-parameter check.
        if not isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = torch.nn.parallel.DistributedDataParallel(model)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=0.01)
        reference = copy.deepcopy(model.module)
        reference_params = [parameter for parameter in reference.parameters() if parameter.requires_grad]
        reference_optimizer = torch.optim.AdamW(reference_params, lr=0.01)
        indexer_before = {
            name: parameter.detach().clone() for name, parameter in model.named_parameters() if ".indexer." in name
        }
        before = [parameter.detach().clone() for parameter in trainable]
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            reference_optimizer.zero_grad(set_to_none=True)
            input_ids = (torch.arange(9).unsqueeze(0) + rank + 1) % 64
            loss = model(input_ids=input_ids).logits.square().mean()
            loss.backward()
            global_ids = (torch.arange(9).unsqueeze(0) + torch.arange(world_size).unsqueeze(1) + 1) % 64
            reference(input_ids=global_ids).logits.square().mean().backward()
            assert torch.isfinite(loss)
            for name, parameter in model.named_parameters():
                if ".indexer." in name:
                    assert not parameter.requires_grad and parameter.grad is None
                elif parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter, expected in zip(trainable, reference_params):
                torch.testing.assert_close(parameter.grad, expected.grad, rtol=1e-4, atol=1e-7)
            optimizer.step()
            reference_optimizer.step()
            for parameter, expected in zip(trainable, reference_params):
                torch.testing.assert_close(parameter, expected, rtol=1e-4, atol=1e-7)
        assert any(not torch.equal(old, new) for old, new in zip(before, trainable))
        assert set(optimizer.state) == set(trainable)
        for name, parameter in model.named_parameters():
            if name in indexer_before:
                torch.testing.assert_close(parameter, indexer_before[name], rtol=0, atol=0)
        for parameter in trainable:
            peers = [torch.empty_like(parameter) for _ in range(world_size)]
            dist.all_gather(peers, parameter.detach())
            for peer in peers:
                torch.testing.assert_close(peer, parameter, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.cpu_only
@pytest.mark.parametrize("world_size", [1, 2])
def test_indexer_lora_ddp_completes_two_optimizer_steps(tmp_path: Path, world_size: int) -> None:
    torch.multiprocessing.spawn(
        _run_indexer_lora_ddp, args=(world_size, str(tmp_path / "gloo_init")), nprocs=world_size, join=True
    )


@pytest.mark.parametrize("query_start", [0, 8])
def test_indexer_all_padding_query_chunks(query_start: int) -> None:
    """A trailing empty chunk or CP shard must select no padded keys."""
    indexer = KimiDSAIndexer(_tiny_text_config(_tiny_dsa_config(index_topk=4)), torch.float32)
    indexer._query_chunk_size = 4
    x = torch.randn(1, 12, 32)
    q = torch.randn(1, 12, 16)
    positions = torch.arange(12)
    allowed = (positions[:, None] >= positions[None, :]) & (positions[:, None] < 6) & (positions[None, :] < 6)
    allowed = allowed[None, query_start:]
    selected = indexer.select(x[:, query_start:], q[:, query_start:], indexer.project_keys(x), allowed)
    assert not (selected & ~allowed).any()
    torch.testing.assert_close(selected.sum(-1), allowed.sum(-1).clamp_max(4))


def test_cudnn_metadata_rejects_interior_masked_holes() -> None:
    """A masked hole must not silently split one document into two contexts."""
    context = KimiPackedContext(torch.tensor([[1, 1, 0, 1, 1]], dtype=torch.int32))
    with pytest.raises(ValueError, match="interior masked holes"):
        context.cudnn_dsa_metadata(0)


def test_cudnn_indexer_dense_prefix_skips_scoring() -> None:
    """Keep all visible prefix keys without invoking a CUDA scoring kernel."""
    from nemo_automodel.components.models.glm_moe_dsa.kernels.cudnn_dsa import CudnnDsaPackedMetadata

    metadata = CudnnDsaPackedMetadata(
        starts=torch.zeros(4, dtype=torch.int32),
        causal_lengths=torch.tensor([1, 2, 3, 0], dtype=torch.int32),
        query_valid=torch.tensor([True, True, True, False]),
        valid_row_indices=torch.tensor([0, 1, 2]),
        segment_cu_q=torch.tensor([0, 4], dtype=torch.int32),
        segment_cu_k=torch.tensor([0, 4], dtype=torch.int32),
        q_causal_offsets=torch.zeros(1, dtype=torch.int32),
        key_source_indices=None,
        max_seqlen_q=4,
        max_seqlen_k=4,
        total_key_tokens=4,
        all_rows_nonempty=False,
    )
    indexer = KimiDSAIndexer(_tiny_text_config(_tiny_dsa_config(index_topk=4)), torch.float32)
    with mock.patch.object(indexer.wq_b, "forward", side_effect=AssertionError("scoring is unnecessary")):
        actual = indexer.select_indices(torch.zeros(4, 32), torch.zeros(4, 16), torch.zeros(4, 8), metadata)
    expected = torch.tensor(
        [[[0, -1, -1, -1]], [[0, 1, -1, -1]], [[0, 1, 2, -1]], [[-1, -1, -1, -1]]], dtype=torch.int32
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("kv_lora_rank", 16, "kv_lora_rank"),
        ("qk_rope_head_dim", 8, "qk_rope_head_dim"),
        ("attention_dropout", 0.1, "dropout"),
    ],
)
def test_cudnn_rejects_unsupported_attention_settings(field: str, value: int | float, message: str) -> None:
    config = _tiny_text_config(_tiny_dsa_config(index_topk=64, index_head_dim=128, index_num_heads=32))
    config.kv_lora_rank = 512
    config.qk_rope_head_dim = 64
    setattr(config, field, value)
    with pytest.raises(ValueError, match=message):
        KimiMLAAttention(config, 1, _backend("cudnn"))


def test_cudnn_sparse_forward_head_padding_is_limited_to_validated_kimi_shape() -> None:
    from nemo_automodel.components.models.common.cudnn_sparse_attention import _padded_head_count

    assert _padded_head_count(96, 10) == 128
    for heads in (3, 48, 80, 112):
        with pytest.raises(ValueError, match="query-head count"):
            _padded_head_count(heads, 10)
    with pytest.raises(ValueError, match="query-head count"):
        _padded_head_count(96, 9)
