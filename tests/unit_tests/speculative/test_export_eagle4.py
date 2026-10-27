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

"""Export of an Automodel EAGLE-4 draft to the ``LlamaForCausalLMEagle4`` serving layout."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import LlamaConfig, Qwen3Config

from nemo_automodel.components.speculative.eagle.draft_llama import LlamaEagle3DraftModel
from nemo_automodel.components.speculative.eagle.eagle4 import Eagle4CandidateHeadConfig, Eagle4Config
from nemo_automodel.components.speculative.eagle.export_eagle4 import export_eagle4_draft, main

# Tensor names of the released LlamaForCausalLMEagle4 drafts (fc_norm enabled).
SERVING_KEYS = {
    "efficient_lm_head.weight",
    "efficient_lm_head_down_proj.weight",
    "embed_tokens.weight",
    "fc.weight",
    "fc_norm.0.weight",
    "fc_norm.1.weight",
    "fc_norm.2.weight",
    "layers.0.hidden_norm.weight",
    "layers.0.input_layernorm.weight",
    "layers.0.mlp.down_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "lm_head.weight",
    "norm.weight",
}
# Config keys of the released drafts (an export may write a subset).
SERVING_CONFIG_KEYS = {
    "_torchspec_version",
    "architectures",
    "attention_bias",
    "attention_dropout",
    "bos_token_id",
    "draft_vocab_size",
    "dtype",
    "eagle_config",
    "efficient_lm_head_config",
    "eos_token_id",
    "fc_norm",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "initializer_range",
    "intermediate_size",
    "layer_types",
    "max_position_embeddings",
    "mlp_bias",
    "model_type",
    "norm_output",
    "num_attention_heads",
    "num_aux_hidden_states",
    "num_hidden_layers",
    "num_key_value_heads",
    "pad_token_id",
    "pretraining_tp",
    "rms_norm_eps",
    "rope_parameters",
    "sliding_window",
    "target_hidden_size",
    "tie_word_embeddings",
    "transformers_version",
    "use_cache",
    "use_sliding_window",
    "vocab_size",
    "torch_dtype",
}


def _training_draft(candidate_head: bool = True, **config_overrides) -> LlamaEagle3DraftModel:
    """Build a draft the way TrainEagle3Recipe does: the target config dict plus the recipe's draft keys, as a LlamaConfig."""
    target = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        vocab_size=64,
        max_position_embeddings=64,
    )
    config = target.to_dict()
    config.update(
        draft_vocab_size=64,
        target_hidden_size=32,
        architectures=["LlamaEagle3DraftModel"],
        tie_word_embeddings=False,
        attn_implementation="eager",
        fc_norm=True,
        norm_output=True,
        parallel_drafting=False,
    )
    head = Eagle4CandidateHeadConfig(rank=4, num_candidates=8) if candidate_head else None
    config.update(Eagle4Config(sliding_window=16, candidate_head=head).draft_config_updates(aux_layer_ids=[1, 2, 3]))
    config.update(config_overrides)
    torch.manual_seed(0)
    return LlamaEagle3DraftModel(LlamaConfig.from_dict(config)).to(torch.bfloat16)


def _write_checkpoint(draft, root, *, num_shards=1, state_dict=None, config=None):
    """Write ``draft`` like a consolidated checkpoint: config.json plus (sharded) safetensors."""
    consolidated = root / "model" / "consolidated"
    consolidated.mkdir(parents=True)
    (consolidated / "config.json").write_text(config or draft.config.to_json_string(use_diff=False))
    state = state_dict if state_dict is not None else draft.state_dict()
    keys = sorted(state)
    for i in range(num_shards):
        shard = {k: state[k].contiguous() for k in keys[i::num_shards]}
        save_file(shard, str(consolidated / f"model-{i + 1:05d}-of-{num_shards:05d}.safetensors"))
    return consolidated


def test_export_writes_the_serving_layout(tmp_path):
    draft = _training_draft()
    _write_checkpoint(draft, tmp_path / "step", num_shards=2)
    out = export_eagle4_draft(tmp_path / "step", tmp_path / "export")  # resolves model/consolidated

    tensors = load_file(str(out / "model.safetensors"))
    assert set(tensors) == SERVING_KEYS
    for key, tensor in draft.state_dict().items():
        torch.testing.assert_close(tensors[key.removeprefix("model.")], tensor)

    config = json.loads((out / "config.json").read_text())
    assert set(config) <= SERVING_CONFIG_KEYS
    assert config["architectures"] == ["LlamaForCausalLMEagle4"]
    assert config["model_type"] == "llama"
    assert config["num_hidden_layers"] == 1
    assert config["draft_vocab_size"] == config["vocab_size"] == 64
    assert config["head_dim"] == 8
    assert config["layer_types"] == ["sliding_attention"] and config["sliding_window"] == 16
    assert config["eagle_config"]["eagle_aux_hidden_state_layer_ids"] == [1, 2, 3]
    assert config["efficient_lm_head_config"] == {"type": "low_rank", "rank": 4, "num_candidates": 8}
    assert config["torch_dtype"] == "bfloat16"
    assert config["tie_word_embeddings"] is False


def test_exported_draft_loads_back_and_computes_the_same_logits(tmp_path):
    draft = _training_draft()
    out = export_eagle4_draft(_write_checkpoint(draft, tmp_path / "ckpt"), tmp_path / "export")
    tensors = load_file(str(out / "model.safetensors"))
    reloaded = _training_draft()
    with torch.no_grad():
        for p in reloaded.parameters():
            p.zero_()
    top_level = ("lm_head.", "efficient_lm_head")
    reloaded.load_state_dict({(k if k.startswith(top_level) else f"model.{k}"): v for k, v in tensors.items()})
    hidden = torch.randn(1, 4, 32, dtype=torch.bfloat16)
    torch.testing.assert_close(reloaded.compute_candidate_logits(hidden), draft.compute_candidate_logits(hidden))


def test_cli_entry_point(tmp_path):
    main([str(_write_checkpoint(_training_draft(), tmp_path / "ckpt")), "--output-dir", str(tmp_path / "export")])
    assert (tmp_path / "export" / "config.json").is_file()


def _config_json(**overrides):
    config = json.loads(_training_draft().config.to_json_string(use_diff=False))
    config.update(overrides)
    return json.dumps(config)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"architectures": ["LlamaEagle3DraftModel"]}, "Not an EAGLE-4 draft"),
        ({"efficient_lm_head_config": None}, "no candidate head"),
        ({"draft_vocab_size": 32}, "full vocabulary"),
        ({"parallel_drafting": True}, "parallel_drafting"),
        ({"rope_parameters": None}, "rope_parameters"),
        ({"num_aux_hidden_states": 2}, "three aux hidden states"),
        ({"target_hidden_size": 64}, "target_hidden_size"),
        ({"attention_bias": True}, "bias-free SiLU"),
        ({"hidden_act": "gelu"}, "bias-free SiLU"),
    ],
)
def test_export_rejects_drafts_the_serving_engines_do_not_build(tmp_path, overrides, match):
    draft = _training_draft()
    ckpt = _write_checkpoint(draft, tmp_path / "ckpt", config=_config_json(**overrides))
    with pytest.raises(ValueError, match=match):
        export_eagle4_draft(ckpt, tmp_path / "export")


def test_export_rejects_mismatched_tensors(tmp_path):
    draft = _training_draft()
    state = draft.state_dict()
    state.pop("efficient_lm_head.weight")
    state["model.extra.weight"] = torch.zeros(1)
    ckpt = _write_checkpoint(draft, tmp_path / "ckpt", state_dict=state)
    with pytest.raises(ValueError, match=r"missing \['efficient_lm_head.weight'\], unexpected \['extra.weight'\]"):
        export_eagle4_draft(ckpt, tmp_path / "export")


def test_export_rejects_bad_paths(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.json"):
        export_eagle4_draft(tmp_path, tmp_path / "export")
    ckpt = _write_checkpoint(_training_draft(), tmp_path / "ckpt")
    with pytest.raises(ValueError, match="must differ"):
        export_eagle4_draft(ckpt, ckpt)
    for shard in ckpt.glob("*.safetensors"):
        shard.unlink()
    with pytest.raises(FileNotFoundError, match="No .safetensors"):
        export_eagle4_draft(ckpt, tmp_path / "export")
