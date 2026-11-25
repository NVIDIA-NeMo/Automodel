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

"""Independent whole-state references and helpers for sharded checkpoint tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5 import (
    mtp_checkpoint_adapter as candidate_module,
)
from nemo_automodel.components.models.ernie5_5.config import (
    Ernie5_5Config,
    mtp_decoder_config,
)
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.mtp_checkpoint_adapter import (
    Ernie5_5MTPCheckpointAdapter,
)
from nemo_automodel.components.models.ernie5_5.mtp_state_dict_adapter import (
    _OWNED,
    Ernie5_5MTPStateDictAdapter,
)
from nemo_automodel.components.models.ernie5_5.mtp_training import (
    Ernie5_5MTPForCausalLM,
)

# Tiny fixture: two DSA layers (dense then MoE), followed by MTP sharing the last layer.
BODY_PREFIX = "model.layers.1."
MTP_PREFIX = "model.layers.2."
BIAS_SUFFIX = "block_sparse_moe.e_score_correction_bias"
OWNED_HF = frozenset(MTP_PREFIX + suffix for suffix in _OWNED.values())
EXPERT_HF_MARK = ".block_sparse_moe.experts."
# Tensor families stored as FP32 in the release; other families are BF16.
F32_MARKERS = (
    "hyper_connection.",
    "block_sparse_moe.gate.",
    "hc_head_",
    "e_score_correction_bias",
    "mhc_contract.",
    "attn_sink",
    "compressor.ape",
)
# Six corruption targets: five alias destinations and the last decoder owner.
CHANGED = {
    "alias_plain": MTP_PREFIX + "self_attn.q_a_proj.weight",
    "alias_mhc_alpha": MTP_PREFIX + "transformer_layer.self_attention_hyper_connection.alpha_pre",
    "alias_mhc_projection": MTP_PREFIX + "transformer_layer.mlp_hyper_connection.mapping_proj.weight",
    "alias_fused_kv": MTP_PREFIX + "self_attn.kv_b_proj.weight",
    "alias_expert": MTP_PREFIX + "block_sparse_moe.experts.3.w2.weight",
    "body_owner": BODY_PREFIX + "self_attn.q_a_proj.weight",
}


def candidate_identity() -> dict:
    """Record the tested adapter path and hash in each result to identify the source."""
    path = Path(candidate_module.__file__)
    return {"file": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def tiny_config(dtype: str = "bfloat16", heads: int = 2) -> Ernie5_5Config:
    return Ernie5_5Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=24,
        num_hidden_layers=2,
        num_nextn_predict_layers=1,
        mtp_shared_last_layer=True,
        compress_ratios=[-2, -2, -2],
        max_position_embeddings=1024,
        num_attention_heads=2,
        head_dim=8,
        qk_rope_head_dim=4,
        q_lora_rank=6,
        o_lora_rank=4,
        o_groups=2,
        sliding_window=8,
        hybrid_mla_num_attention_heads=heads,
        hybrid_mla_q_lora_rank=6,
        hybrid_mla_kv_lora_rank=4,
        hybrid_mla_qk_nope_head_dim=4,
        hybrid_mla_qk_rope_head_dim=2,
        hybrid_mla_v_head_dim=3,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        rope_interleave=True,
        hc_mult=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=12,
        moe_latent_size=8,
        first_k_dense_replace=1,
        pad_token_id=0,
        torch_dtype=dtype,
    )


def backend() -> BackendConfig:
    return BackendConfig(attn="eager", linear="torch", experts="torch", dispatcher="torch")


def stored_dtype(name: str) -> torch.dtype:
    return torch.float32 if any(mark in name for mark in F32_MARKERS) else torch.bfloat16


def fill(model: torch.nn.Module, seed: int) -> None:
    """Fill every state tensor with BF16-representable random values, including distinct nonzero decoder and MTP biases."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for tensor in model.state_dict().values():
            value = torch.randn(tensor.shape, generator=generator)
            tensor.copy_(value.to(torch.bfloat16).to(tensor.dtype))


def attach(model: Ernie5_5MTPForCausalLM) -> Ernie5_5MTPForCausalLM:
    """Assert that the model already owns the native checkpoint adapter and return it."""
    assert isinstance(model.state_dict_adapter, Ernie5_5MTPCheckpointAdapter)
    return model


def new_model(seed: int, dtype: str = "bfloat16", heads: int = 2) -> Ernie5_5MTPForCausalLM:
    model = Ernie5_5MTPForCausalLM(tiny_config(dtype, heads), backend())
    fill(model, seed)
    return attach(model)


def meta_model(dtype: str = "bfloat16", heads: int = 2) -> Ernie5_5MTPForCausalLM:
    with torch.device("meta"):
        model = Ernie5_5MTPForCausalLM(tiny_config(dtype, heads), backend())
    return attach(model)


def meta_decoder_model() -> Ernie5_5_MoeForCausalLM:
    with torch.device("meta"):
        return Ernie5_5_MoeForCausalLM(mtp_decoder_config(tiny_config()), backend())


def parent_export(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Export the whole mapping with the frozen parent adapter, independently of the checkpoint adapter under test."""
    return Ernie5_5MTPStateDictAdapter(model.config).to_hf(model.state_dict())


def reference_tensors(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Return independent tensors in release layouts and release-family dtypes."""
    return {
        name: value.detach().to(stored_dtype(name)).clone().contiguous() for name, value in parent_export(model).items()
    }


def write_reference(directory: Path, model: torch.nn.Module, mutate=None) -> dict[str, torch.Tensor]:
    """Write a release-layout HF directory with one safetensors file, index and config."""
    tensors = reference_tensors(model)
    if mutate is not None:
        mutate(tensors)
    directory.mkdir(parents=True)
    save_file(tensors, str(directory / "model.safetensors"))
    index = {"metadata": {}, "weight_map": dict.fromkeys(tensors, "model.safetensors")}
    (directory / "model.safetensors.index.json").write_text(json.dumps(index))
    model.config.save_pretrained(directory)
    return tensors


def change(key: str):
    """Return a mutation that adds one to every element of a stored tensor."""

    def mutate(tensors: dict[str, torch.Tensor]) -> None:
        tensors[key] = tensors[key] + 1

    return mutate


def drop(predicate):
    """Return a mutation that removes stored tensors matching the predicate."""

    def mutate(tensors: dict[str, torch.Tensor]) -> None:
        for name in [name for name in tensors if predicate(name)]:
            del tensors[name]

    return mutate


def read_into(destinations: dict[str, torch.Tensor], checkpoint: dict[str, torch.Tensor]) -> None:
    """Copy stored values into destinations in place with dtype conversion, as the checkpoint reader does."""
    if destinations.keys() != checkpoint.keys():
        raise AssertionError("destination keys differ from the checkpoint keys")
    with torch.no_grad():
        for name, destination in destinations.items():
            destination.copy_(checkpoint[name])


def load_checkpointer(reference: Path, tmp: Path):
    return CheckpointingConfig(
        enabled=True,
        checkpoint_dir="",
        model_save_format="safetensors",
        model_cache_dir=str(tmp / "cache"),
        model_repo_id=str(reference),
        save_consolidated=False,
    ).build(0, 0, 0)


def save_checkpointer(reference: Path, tmp: Path, run: str, consolidate: str):
    return CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(tmp / run),
        model_save_format="safetensors",
        model_cache_dir=str(tmp / "cache"),
        model_repo_id=str(reference),
        save_consolidated=consolidate,
        is_async=False,
    ).build(0, 0, 0)


def base_load(model: torch.nn.Module, reference: Path, tmp: Path) -> None:
    """Materialize a meta model and load base weights through Checkpointer in one process."""
    checkpointer = load_checkpointer(reference, tmp)
    try:
        checkpointer.initialize_model_weights(model, torch.device("cpu"))
        checkpointer.load_base_model(
            model,
            torch.device("cpu"),
            str(tmp / "cache"),
            str(reference),
            load_base_model=True,
        )
    finally:
        checkpointer.close()


def state_difference(actual: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> dict:
    """Count differences between tensor mappings in keys, dtypes, shapes or values."""
    dtype_bad = [name for name in expected if name in actual and actual[name].dtype != expected[name].dtype]
    value_bad = [
        name
        for name in expected
        if name in actual
        and (
            actual[name].shape != expected[name].shape
            or not torch.equal(actual[name].to(torch.float32), expected[name].to(torch.float32))
        )
    ]
    return {
        "same_names": set(actual) == set(expected),
        "tensors": len(expected),
        "dtype_mismatches": len(dtype_bad),
        "value_mismatches": len(value_bad),
        "first_mismatch": (dtype_bad + value_bad)[:1],
    }


def exact(difference: dict) -> bool:
    return difference["same_names"] and difference["dtype_mismatches"] == 0 and difference["value_mismatches"] == 0


def read_safetensors(directory: Path) -> dict[str, torch.Tensor]:
    stored: dict[str, torch.Tensor] = {}
    for file in sorted(directory.glob("*.safetensors")):
        tensors = load_file(str(file))
        if not stored.keys().isdisjoint(tensors):
            raise ValueError(f"A tensor name is stored twice across files: {file.name}")
        stored.update(tensors)
    return stored


def body_name(alias: str) -> str:
    return BODY_PREFIX + alias[len(MTP_PREFIX) :].removeprefix("transformer_layer.")


def alias_report(stored: dict[str, torch.Tensor]) -> dict:
    """Check stored aliases for counts, exact equality with the last decoder owner and matching dtypes."""
    aliases = [name for name in stored if name.startswith(MTP_PREFIX) and name not in OWNED_HF]
    unequal = [
        alias
        for alias in aliases
        if body_name(alias) in stored
        and (
            stored[alias].dtype != stored[body_name(alias)].dtype
            or stored[alias].shape != stored[body_name(alias)].shape
            or not torch.equal(stored[alias], stored[body_name(alias)])
        )
    ]
    bias, decoder_bias = MTP_PREFIX + BIAS_SUFFIX, BODY_PREFIX + BIAS_SUFFIX
    return {
        "keys": len(stored),
        "aliases": len(aliases),
        "aliases_with_transformer_layer_prefix": sum("transformer_layer." in name for name in aliases),
        "owned_present": len(OWNED_HF & stored.keys()),
        "aliases_unequal_to_body": len(unequal),
        "aliases_without_source": sum(body_name(alias) not in stored for alias in aliases),
        "decoder_bias_copied_to_mtp_bias": bool(
            bias in stored and decoder_bias in stored and torch.equal(stored[bias], stored[decoder_bias])
        ),
        "hc_head_fn_shape": list(stored[MTP_PREFIX + "hc_head_fn"].shape)
        if MTP_PREFIX + "hc_head_fn" in stored
        else None,
    }


def raised(function) -> dict:
    """Run a call and record its exception type, last message line and notes, or None.

    DCP CheckpointException inherits BaseException; control interrupts still propagate.
    Retain strings only so traceback references cannot keep load sessions alive.
    """
    try:
        function()
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as error:  # noqa: BLE001
        lines = [line.strip() for line in str(error).splitlines() if line.strip()]
        return {
            "raised": type(error).__name__,
            "is_exception_subclass": isinstance(error, Exception),
            "message": (lines[-1] if lines else "")[:300],
            "notes": [str(note)[:200] for note in getattr(error, "__notes__", [])],
        }
    return {"raised": None}


def blocked(adapter, state: dict[str, torch.Tensor]) -> dict:
    """Record three blocked operations after failure; calling this on a healthy adapter would open a session."""
    return {
        "save_to_hf": raised(lambda: adapter.to_hf(state))["raised"],
        "load_to_hf": raised(lambda: adapter.to_hf(state, for_checkpoint_load=True))["raised"],
        "key_plan": raised(lambda: adapter.get_hf_state_dict_keys(state))["raised"],
    }


def decoder_state_flags(adapter) -> dict:
    """Read decoder session flags without invoking adapter operations."""
    decoder = adapter.decoder
    return {
        "decoder_poisoned": bool(getattr(decoder, "_poisoned", False)),
        "decoder_session_open": getattr(decoder, "_token", None) is not None,
    }
