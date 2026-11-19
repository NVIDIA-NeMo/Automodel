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

"""Unit tests for the ERNIE 5.5 checkpoint name tables and eager Hugging Face conversions."""

import copy
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch

import nemo_automodel
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.ernie5_5.checkpoint_layout import (
    DSAKVLayout,
    map_common_from_hf,
    map_common_to_hf,
    map_dsa_from_hf,
    map_dsa_to_hf,
    map_hca_from_hf,
    map_model_from_hf,
    map_model_to_hf,
    map_moe_from_hf,
    map_moe_to_hf,
    pack_experts,
    unpack_experts,
)
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config
from nemo_automodel.components.models.ernie5_5.dsa import Ernie5_5DSAAttention
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM

pytestmark = pytest.mark.cpu_only


def test_common_key_and_tensor_round_trip():
    torch.manual_seed(5584)
    state = {
        "model.embed_tokens.weight": torch.randn(7, 4, dtype=torch.bfloat16),
        "lm_head.weight": torch.randn(7, 4, dtype=torch.bfloat16),
        "model.norm.weight": torch.randn(4, dtype=torch.bfloat16),
        "model.lm_head.multimax_ranges": torch.randn(4, dtype=torch.bfloat16),
        "model.lm_head.multimax_ts": torch.randn(4, dtype=torch.bfloat16),
        "model.mhc_contract.hc_head_fn": torch.randn(8, 2),
        "model.mhc_contract.hc_head_base": torch.randn(2),
        "model.mhc_contract.hc_head_scale": torch.randn(1),
    }
    native = map_common_from_hf(state, vocab=7, hidden=4, streams=2)
    restored = map_common_to_hf(native, vocab=7, hidden=4, streams=2)
    assert set(restored) == set(state)
    assert all(torch.equal(value, restored[key]) for key, value in state.items())
    # The released hyper-head matrix is a transposed view of the native one.
    head = restored["model.mhc_contract.hc_head_fn"]
    assert head.untyped_storage().data_ptr() == native["hyper_head.hc_fn"].untyped_storage().data_ptr()
    with pytest.raises(ValueError, match="shape"):
        map_common_from_hf(state, vocab=8, hidden=4, streams=2)
    with pytest.raises(ValueError, match="exactly"):
        map_common_from_hf({**state, "unrecognized": torch.zeros(1)}, vocab=7, hidden=4, streams=2)


def _released_layer_zero_metadata() -> dict[str, tuple[torch.dtype, tuple[int, ...]]]:
    return {
        "core_attention.attn_sink": (torch.float32, (64,)),
        "core_attention.compressor.ape": (torch.float32, (128, 512)),
        "core_attention.compressor.linear_wgate.weight": (torch.bfloat16, (512, 4096)),
        "core_attention.compressor.linear_wkv.weight": (torch.bfloat16, (512, 4096)),
        "core_attention.compressor.norm.weight": (torch.bfloat16, (512,)),
        "gate_proj.weight": (torch.bfloat16, (8192, 1024)),
        "kv_layernorm.weight": (torch.bfloat16, (512,)),
        "linear_kv_proj.weight": (torch.bfloat16, (512, 4096)),
        "linear_o_group_proj": (torch.bfloat16, (8192, 4096)),
        "linear_q_down_proj.weight": (torch.bfloat16, (1024, 4096)),
        "linear_q_up_proj.weight": (torch.bfloat16, (32768, 1024)),
        "o_proj.weight": (torch.bfloat16, (4096, 8192)),
        "q_layernorm.weight": (torch.bfloat16, (1024,)),
        "vha_postmix_U": (torch.bfloat16, (64, 16)),
        "vha_postmix_V": (torch.bfloat16, (64, 16)),
    }


def test_hca_key_map_has_released_shapes_and_loads_strictly_on_meta() -> None:
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=4096,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=64,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=1024,
        o_lora_rank=1024,
        o_groups=8,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        max_position_embeddings=1048576,
        compress_ratios=[128],
        sliding_window=128,
        num_hash_layers=0,
        hc_mult=1,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="bfloat16",
    )
    with torch.device("meta"):
        module = Ernie5_5HCAAttention(config, layer_idx=0, vha_rank=16)
    hf_state = {
        key: torch.empty(shape, dtype=dtype, device="meta")
        for key, (dtype, shape) in _released_layer_zero_metadata().items()
    }
    native = map_hca_from_hf(hf_state)
    assert native["vha.vha_postmix_U"] is hf_state["vha_postmix_U"]
    result = module.load_state_dict(native, strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys


def test_hca_key_map_rejects_partial_and_extra_state() -> None:
    with pytest.raises(ValueError, match="Invalid HCA checkpoint keys"):
        map_hca_from_hf({"unexpected": torch.empty(0)})


def test_dsa_key_mapping_round_trip_and_strict_load():
    torch.manual_seed(5569)
    model = Ernie5_5DSAAttention(8, 2, 6, 4, 3, 2, 3, 2, 4, 1, window=2, remote_topk=1, dtype=torch.float32)
    layout = DSAKVLayout(heads=2, latent=4, key_channels=3, value_channels=3)
    native = model.state_dict()
    hf = map_dsa_to_hf(native, layout)
    restored = map_dsa_from_hf(hf, layout)
    model.load_state_dict(restored, strict=True)
    assert set(restored) == set(native)
    for key, tensor in native.items():
        assert torch.equal(restored[key], tensor), key
    # Per-head HF stores [K rows | V rows], while native K is transposed.
    first = hf["kv_b_proj.weight"].reshape(2, 6, 4)[0]
    torch.testing.assert_close(first[:3], native["k_b_proj"].reshape(2, 4, 3)[0].T, rtol=0, atol=0)
    torch.testing.assert_close(first[3:], native["v_b_proj"].reshape(2, 3, 4)[0], rtol=0, atol=0)


def test_dsa_mapping_rejects_missing_extra_or_bad_kv_shape():
    model = Ernie5_5DSAAttention(8, 2, 6, 4, 3, 2, 3, 2, 4, 1, window=2, remote_topk=1, dtype=torch.float32)
    layout = DSAKVLayout(2, 4, 3, 3)
    hf = map_dsa_to_hf(model.state_dict(), layout)
    with pytest.raises(ValueError, match="Invalid DSA layer HF keys"):
        map_dsa_from_hf({key: value for key, value in hf.items() if key != "q_a_proj.weight"}, layout)
    with pytest.raises(ValueError, match="Invalid DSA layer HF keys"):
        map_dsa_from_hf({**hf, "unexpected": torch.zeros(1)}, layout)
    with pytest.raises(ValueError, match="Expected combined KV shape"):
        map_dsa_from_hf({**hf, "kv_b_proj.weight": torch.zeros(2, 2)}, layout)


def _expert_state(ids):
    result = {}
    for expert in ids:
        result[f"experts.{expert}.w1.weight"] = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 100 * expert
        result[f"experts.{expert}.w2.weight"] = torch.arange(12, dtype=torch.float32).reshape(3, 4) + 200 * expert
        result[f"experts.{expert}.w3.weight"] = torch.arange(12, dtype=torch.float32).reshape(4, 3) + 300 * expert
    return result


def test_selected_expert_order_and_round_trip():
    ids = (7, 2)
    state = _expert_state(ids)
    gate_up, down = pack_experts(state, ids)
    assert gate_up.shape == (2, 3, 8) and down.shape == (2, 4, 3)
    torch.testing.assert_close(gate_up[0, :, :4], state["experts.7.w1.weight"].T, rtol=0, atol=0)
    torch.testing.assert_close(gate_up[1, :, 4:], state["experts.2.w3.weight"].T, rtol=0, atol=0)
    restored = unpack_experts(gate_up, down, ids)
    assert all(torch.equal(restored[key], value) for key, value in state.items())
    # The per-expert matrices are views of the grouped tensors, not copies: writing one writes its expert only.
    restored["experts.2.w2.weight"].zero_()
    assert not down[1].any() and down[0].any()


def test_complete_local_moe_mapping_round_trip():
    ids = (0, 1)
    state = _expert_state(ids)
    for name in (
        "gate.weight",
        "gate.weight_1",
        "gate.routed_scaling_factor_param",
        "e_score_correction_bias",
        "fc1_latent_proj.weight",
        "fc2_latent_proj.weight",
        "latent_norm.weight",
        "shared_experts.w1.weight",
        "shared_experts.w2.weight",
        "shared_experts.w3.weight",
    ):
        state[name] = torch.randn(2, 3)
    native = map_moe_from_hf(state, ids)
    restored = map_moe_to_hf(native, ids)
    assert set(restored) == set(state)
    assert all(torch.equal(restored[key], value) for key, value in state.items())


def test_moe_mapping_rejects_invalid_ids_and_shapes():
    state = _expert_state((0,))
    with pytest.raises(ValueError, match="unique"):
        pack_experts(state, (0, 0))
    with pytest.raises(ValueError, match="Missing selected expert"):
        pack_experts(state, (1,))
    with pytest.raises(ValueError, match="Invalid projection shapes"):
        pack_experts({**state, "experts.0.w2.weight": torch.zeros(2, 2)}, (0,))


def test_tiny_decoder_only_model_conversion_and_reload():
    torch.manual_seed(5588)
    config = Ernie5_5Config(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=1,
        compress_ratios=[-2],
        n_routed_experts=3,
        num_nextn_predict_layers=0,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        hc_mult=2,
        hybrid_mla_num_attention_heads=2,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        sliding_window=2,
        torch_dtype="float32",
    )
    backend = BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")
    entry = Ernie5_5_MoeForCausalLM(config, backend)
    entry.initialize_weights(dtype=torch.float32)
    model = entry.model
    # The eager conversions name the feed-forward of a layer ``feed_forward``; the model registers it as ``mlp``.
    original = {
        name.replace("layers.0.mlp.", "layers.0.feed_forward."): value for name, value in model.state_dict().items()
    }
    hf = map_model_to_hf(original, config)
    recovered = map_model_from_hf(hf, config)
    assert set(original) == set(recovered)
    assert all(torch.equal(value, recovered[key]) for key, value in original.items())
    restored = copy.deepcopy(model)
    with torch.no_grad():
        restored.embed_tokens.weight.zero_()
    restored.load_state_dict(
        {name.replace("layers.0.feed_forward.", "layers.0.mlp."): value for name, value in recovered.items()},
        strict=True,
    )
    tokens = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(restored(tokens), model(tokens), rtol=0, atol=0)
    with pytest.raises(ValueError, match="exactly the common"):
        map_model_from_hf({**hf, "model.layers.1.mtp_fake": torch.zeros(1)}, config)


@pytest.mark.parametrize("mapper", [map_model_from_hf, map_model_to_hf])
def test_model_mapping_rejects_config_with_mtp(mapper):
    config = Ernie5_5Config(
        num_hidden_layers=1,
        compress_ratios=[128, -2],
        num_nextn_predict_layers=1,
        first_k_dense_replace=1,
        moe_layer_freq=1,
    )
    with pytest.raises(NotImplementedError, match="cannot preserve MTP"):
        mapper({}, config)


# Runs in a fresh interpreter so that each run has its own string hash seed. It converts the shape-only
# Hugging Face state of a three-layer model (HCA + dense, HCA + MoE, DSA + MoE) to native names and back,
# and prints both ordered key lists.
_KEY_ORDER_SCRIPT = """
import json
import sys

sys.path.insert(0, sys.argv[1])

import torch

from nemo_automodel.components.models.ernie5_5 import checkpoint_layout as layout
from nemo_automodel.components.models.ernie5_5.config import Ernie5_5Config

VOCAB, HIDDEN, STREAMS, EXPERTS, HEADS, LATENT, KEY, VALUE, INTERMEDIATE, MOE_LATENT = 8, 4, 2, 2, 2, 2, 2, 2, 3, 2
MIX = (2 + STREAMS) * STREAMS


def meta(*shape):
    return torch.empty(shape, dtype=torch.float32, device="meta")


config = Ernie5_5Config(
    vocab_size=VOCAB,
    hidden_size=HIDDEN,
    num_hidden_layers=3,
    compress_ratios=[128, 128, -2],
    n_routed_experts=EXPERTS,
    num_nextn_predict_layers=0,
    first_k_dense_replace=1,
    moe_layer_freq=1,
    hc_mult=STREAMS,
    hybrid_mla_num_attention_heads=HEADS,
    hybrid_mla_kv_lora_rank=LATENT,
    hybrid_mla_qk_nope_head_dim=KEY,
    hybrid_mla_v_head_dim=VALUE,
)
state = {
    "model.embed_tokens.weight": meta(VOCAB, HIDDEN),
    "lm_head.weight": meta(VOCAB, HIDDEN),
    "model.norm.weight": meta(HIDDEN),
    "model.lm_head.multimax_ranges": meta(4),
    "model.lm_head.multimax_ts": meta(4),
    "model.mhc_contract.hc_head_fn": meta(STREAMS * HIDDEN, STREAMS),
    "model.mhc_contract.hc_head_base": meta(STREAMS),
    "model.mhc_contract.hc_head_scale": meta(1),
}
attention_names = (layout.HCA_HF_TO_NATIVE, layout.HCA_HF_TO_NATIVE, layout.DSA_HF_TO_NATIVE)
for index, names in enumerate(attention_names):
    prefix = f"model.layers.{index}."
    state[prefix + "input_layernorm.weight"] = meta(HIDDEN)
    state[prefix + "post_attention_layernorm.weight"] = meta(HIDDEN)
    for unit in ("self_attention_hyper_connection.", "mlp_hyper_connection."):
        state[prefix + unit + "mapping_proj.weight"] = meta(STREAMS * HIDDEN, MIX)
        state[prefix + unit + "bias"] = meta(MIX)
        for alpha in ("alpha_pre", "alpha_post", "alpha_res"):
            state[prefix + unit + alpha] = meta(1)
    for name in names:
        state[prefix + "self_attn." + name] = meta(1)
    if index == 2:
        state[prefix + "self_attn.kv_b_proj.weight"] = meta(HEADS * (KEY + VALUE), LATENT)
    if index == 0:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            state[prefix + f"mlp.{projection}.weight"] = meta(1)
    else:
        for name in layout.MOE_HF_TO_NATIVE:
            state[prefix + "block_sparse_moe." + name] = meta(1)
        for expert in range(EXPERTS):
            expert_prefix = f"{prefix}block_sparse_moe.experts.{expert}."
            state[expert_prefix + "w1.weight"] = meta(INTERMEDIATE, MOE_LATENT)
            state[expert_prefix + "w2.weight"] = meta(MOE_LATENT, INTERMEDIATE)
            state[expert_prefix + "w3.weight"] = meta(INTERMEDIATE, MOE_LATENT)
native = layout.map_model_from_hf(state, config)
restored = layout.map_model_to_hf(native, config)
print(json.dumps({"native": list(native), "hf": list(restored), "source": sorted(state)}))
"""


# Seconds that the interpreters of one test may take together; below the hard watchdog of the unit suite.
_INTERPRETER_TIMEOUT = 60.0


def _run_interpreters(arguments: list[str], hash_seeds: tuple[str | None, ...]) -> dict[str | None, str]:
    """Run one Python script in a fresh interpreter per hash seed, all at once, and return what each printed.

    No interpreter outlives the call: when one does not finish in time, it and the others are killed and reaped
    before the exception leaves this function.

    Args:
        arguments: Path of the script, followed by its arguments.
        hash_seeds: ``PYTHONHASHSEED`` of each interpreter; ``None`` leaves the variable unset.

    Returns:
        The standard output of each interpreter, by its hash seed.

    Raises:
        subprocess.TimeoutExpired: If the interpreters take longer than ``_INTERPRETER_TIMEOUT`` together.
        AssertionError: If an interpreter exits with a nonzero status; the message is its standard error.
    """
    base = {name: value for name, value in os.environ.items() if name != "PYTHONHASHSEED"}
    processes: dict[str | None, subprocess.Popen[str]] = {}
    finished: dict[str | None, tuple[str, str]] = {}
    deadline = time.monotonic() + _INTERPRETER_TIMEOUT
    try:
        for seed in hash_seeds:
            environment = base if seed is None else {**base, "PYTHONHASHSEED": seed}
            processes[seed] = subprocess.Popen(
                [sys.executable, *arguments], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
        for seed, process in processes.items():
            finished[seed] = process.communicate(timeout=max(1.0, deadline - time.monotonic()))
    finally:
        for seed, process in processes.items():
            if seed not in finished:
                process.kill()
                process.communicate()
    for seed, process in processes.items():
        assert process.returncode == 0, finished[seed][1]
    return {seed: output for seed, (output, _) in finished.items()}


def test_converted_key_order_does_not_depend_on_the_hash_seed(tmp_path: Path) -> None:
    """Every process must produce the same key order: the ranks of a distributed save are separate processes."""
    script = tmp_path / "key_order.py"
    script.write_text(_KEY_ORDER_SCRIPT, encoding="utf-8")
    repository_root = str(Path(nemo_automodel.__file__).parents[1])
    outputs = _run_interpreters([str(script), repository_root], ("0", "1", "2", None))
    results = {seed: json.loads(output.splitlines()[-1]) for seed, output in outputs.items()}
    first = results["0"]
    assert sorted(first["hf"]) == first["source"]
    assert len(first["hf"]) == 125 and len(first["native"]) == 106
    for seed, result in results.items():
        assert result["hf"] == first["hf"], f"Hugging Face key order differs with PYTHONHASHSEED={seed}"
        assert result["native"] == first["native"], f"native key order differs with PYTHONHASHSEED={seed}"
