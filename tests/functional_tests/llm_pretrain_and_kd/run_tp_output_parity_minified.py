#!/usr/bin/env python
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

"""Standalone distributed TP output parity test (minified models).

Validates that TP=2 produces the same logits as TP=1 for tiny ("2-layer thin")
variants of:
- Qwen3ForCausalLM (HF)
- Qwen3ForSequenceClassification (HF)
- Ministral3ForCausalLM (NeMo Automodel custom)
- LlamaForCausalLM (NeMo Automodel custom, combined QKV + gate_up projections)
- Qwen2ForCausalLM (NeMo Automodel custom, combined QKV + gate_up projections)
- Nemotron Super (LlamaForCausalLM, 10-layer full-hidden, llama_nemotron_super_tp_plan)

It also validates both tensor-parallel plans:
- sequence_parallel=False
- sequence_parallel=True

Usage:
    torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py

    # Optional: select models / SP mode
    torchrun --nproc_per_node=2 tests/functional_tests/llm_pretrain_and_kd/run_tp_output_parity_minified.py \\
        --models qwen3 qwen3_seq_cls ministral3 nemotron \\
        --sequence_parallel both \\
        --kl_threshold 2e-6
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Literal, Sequence, cast

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from torch.distributed.tensor.placement_types import Replicate
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM, Qwen3ForSequenceClassification

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches
from nemo_automodel.components.distributed.parallelizer import _get_parallel_plan, _update_attention_head_counts_for_tp
from nemo_automodel.components.models.baichuan.configuration import BaichuanConfig
from nemo_automodel.components.models.baichuan.model import BaichuanForCausalLM
from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.models.llama.model import LlamaConfig, LlamaForCausalLM
from nemo_automodel.components.models.mistral3.model import Ministral3Config, Ministral3ForCausalLM
from nemo_automodel.components.models.qwen2.model import Qwen2Config, Qwen2ForCausalLM

ModelKind = Literal["qwen3", "qwen3_seq_cls", "ministral3", "llama", "qwen2", "nemotron", "baichuan"]
SPMode = Literal["true", "false", "both"]


def _is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_distributed() else 0


def _world_size() -> int:
    return dist.get_world_size() if _is_distributed() else 1


def _init_distributed() -> tuple[str, torch.device, str]:
    """Init process group if launched via torchrun.

    Returns:
        backend, device, device_type
    """
    if not dist.is_available():
        return "none", torch.device("cpu"), "cpu"
    if dist.is_initialized():
        # Best-effort infer device_type.
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        return (
            dist.get_backend(),
            torch.device("cuda", torch.cuda.current_device()) if device_type == "cuda" else torch.device("cpu"),
            device_type,
        )

    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return "none", torch.device("cpu"), "cpu"

    if torch.cuda.is_available():
        backend = "nccl"
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        device_type = "cuda"
    else:
        # Ensure gloo binds to loopback on minimal containers.
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
        backend = "gloo"
        device = torch.device("cpu")
        device_type = "cpu"

    dist.init_process_group(backend=backend)
    return backend, device, device_type


def _maybe_gather_dtensor_to_replicated_local(x: torch.Tensor | DTensor, *, tp_mesh: DeviceMesh) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.redistribute(device_mesh=tp_mesh, placements=[Replicate()]).to_local()
    return cast(torch.Tensor, x)


def _kl_divergence_from_logits(*, reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """Return KL(reference || candidate), averaged per token.

    Both inputs are expected to be full (non-sharded) logits with shape [B, T, V].
    """
    assert reference_logits.shape == candidate_logits.shape
    vocab_size = reference_logits.shape[-1]
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_log_probs = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    # F.kl_div expects input=log(q), target=log(p) when log_target=True → KL(p || q)
    return F.kl_div(cand_log_probs, ref_log_probs, reduction="none", log_target=True).sum(-1)


@dataclass(frozen=True)
class _Case:
    kind: ModelKind
    sequence_parallel: bool

    def name(self) -> str:
        return f"{self.kind}/sequence_parallel={self.sequence_parallel}"


def _nemotron_10l_config_dict() -> dict:
    """Config dict for a 10-layer slice of nvidia/Llama-3_3-Nemotron-Super-49B-v1_5.

    First 10 block_configs give a hybrid mix: 8 attention layers (GQA,
    n_heads_in_group=8) and 2 no-op attention layers, with varying ffn_mult.
    The auto_map repo prefixes let HF resolve the DeciLM remote code from its
    local cache without hitting the hub.
    """

    def _block(attn_no_op: bool, n_heads_in_group: int | None, ffn_mult: float) -> dict:
        return {
            "attention": {
                "n_heads_in_group": n_heads_in_group,
                "no_op": attn_no_op,
                "replace_with_linear": False,
                "sparsify": None,
                "num_sink_tokens": None,
                "unshifted_sink": False,
                "use_prefill_window_in_sink_attention": False,
                "window_length": None,
            },
            "ffn": {
                "ffn_mult": ffn_mult,
                "no_op": False,
                "replace_with_linear": False,
                "sparsify": None,
            },
        }

    return {
        "architectures": ["DeciLMForCausalLM"],
        "model_type": "nemotron-nas",
        "auto_map": {
            "AutoConfig": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5--configuration_decilm.DeciLMConfig",
            "AutoModelForCausalLM": "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5--modeling_decilm.DeciLMForCausalLM",
        },
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": None,
        "intermediate_size": None,
        "num_hidden_layers": 10,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-05,
        "vocab_size": 128256,
        "max_position_embeddings": 131072,
        "tie_word_embeddings": False,
        "use_cache": False,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "initializer_range": 0.02,
        "mlp_bias": False,
        "pretraining_tp": 1,
        "rope_theta": 500000.0,
        "rope_scaling": {
            "factor": 16.0,
            "high_freq_factor": 4.0,
            "low_freq_factor": 1.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        },
        "torch_dtype": "bfloat16",
        "bos_token_id": 128000,
        "eos_token_id": [128001, 128008, 128009],
        "block_configs": [
            _block(False, 8, 2.625),  # L0: attn, small FFN
            _block(False, 8, 5.25),  # L1-5: attn, large FFN
            _block(False, 8, 5.25),
            _block(False, 8, 5.25),
            _block(False, 8, 5.25),
            _block(False, 8, 5.25),
            _block(True, None, 2.625),  # L6-7: no-op attn, small FFN
            _block(True, None, 2.625),
            _block(False, 8, 5.25),  # L8-9: attn, large FFN
            _block(False, 8, 5.25),
        ],
    }


def _is_nemotron_remote_code_available() -> bool:
    """Return True if the DeciLM remote code is importable from the HF cache."""
    try:
        import json
        import tempfile

        from transformers import AutoConfig

        config_dict = _nemotron_10l_config_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            AutoConfig.from_pretrained(tmpdir, trust_remote_code=True)
        return True
    except Exception:
        return False


def _build_minified_model(kind: ModelKind):
    if kind == "ministral3":
        cfg = Ministral3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 8,
                "max_position_embeddings": 128,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
            },
        )
        return cfg, Ministral3ForCausalLM(cfg)

    if kind == "qwen3":
        num_layers = 2
        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            attention_bias=False,
            use_sliding_window=False,
            layer_types=["full_attention"] * num_layers,
        )
        return cfg, Qwen3ForCausalLM(cfg)

    if kind == "qwen3_seq_cls":
        num_layers = 2
        cfg = Qwen3Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            attention_bias=False,
            use_sliding_window=False,
            layer_types=["full_attention"] * num_layers,
            num_labels=2,  # must be divisible by TP size (2) if sharded
            pad_token_id=0,  # required for batch_size>1 pooling
        )
        return cfg, Qwen3ForSequenceClassification(cfg)

    if kind == "llama":
        num_layers = 2
        backend = BackendConfig(rms_norm="torch")
        cfg = LlamaConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            attention_bias=False,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        return cfg, LlamaForCausalLM(cfg, backend=backend)

    if kind == "nemotron":
        import json
        import tempfile

        from transformers import AutoConfig
        from transformers import AutoModelForCausalLM as _HFAutoCausalLM

        config_dict = _nemotron_10l_config_dict()
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(os.path.join(tmpdir, "config.json"), "w") as f:
                json.dump(config_dict, f)
            cfg = AutoConfig.from_pretrained(tmpdir, trust_remote_code=True)
        cfg._attn_implementation = "eager"
        cfg._name_or_path = "nvidia/Llama-3_3-Nemotron-Super-49B-v1_5"
        model = _HFAutoCausalLM.from_config(cfg, trust_remote_code=True)
        return cfg, model

    if kind == "qwen2":
        num_layers = 2
        backend = BackendConfig(rms_norm="torch")
        cfg = Qwen2Config(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=True,
            use_sliding_window=False,
            sliding_window=None,
            layer_types=["full_attention"] * num_layers,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )
        model = Qwen2ForCausalLM(cfg, backend=backend)
        with torch.no_grad():
            for _, module in model.named_modules():
                if isinstance(module, torch.nn.Linear) and module.bias is not None:
                    torch.nn.init.normal_(module.bias, mean=0.1, std=0.1)
        return cfg, model

    if kind == "baichuan":
        cfg = BaichuanConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            max_position_embeddings=128,
            use_cache=False,
            tie_word_embeddings=False,
        )
        return cfg, BaichuanForCausalLM(cfg)

    raise ValueError(f"Unknown model kind: {kind}")


def _baichuan_tp_plan() -> dict:
    """TP plan for Baichuan2.

    Only the MLP is sharded.  The attention path stays fully replicated
    because W_pack uses a non-interleaved [Q|K|V] layout (ColwiseParallel
    would split it incorrectly) and NormHead (lm_head) is not nn.Linear
    (ColwiseParallel is unsupported).  Keeping the attention replicated
    also avoids DTensor from_local misinterpretation at o_proj.
    """
    return {
        "model.layers.*.mlp.gate_proj": ColwiseParallel(),
        "model.layers.*.mlp.up_proj": ColwiseParallel(),
        "model.layers.*.mlp.down_proj": RowwiseParallel(),
    }


_SP_UNSUPPORTED_MODELS: set[ModelKind] = {"baichuan"}


def _run_case(
    case: _Case,
    *,
    device: torch.device,
    device_type: str,
    kl_threshold: float,
    dtype: torch.dtype,
) -> tuple[bool, float]:
    """Return (ok, kl_divergence)."""
    world_size = _world_size()
    assert world_size == 2, f"This test is intended for TP=2; got world_size={world_size}"

    # Use the same initial weights for baseline and TP models.
    torch.manual_seed(1234)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(1234)

    cfg, baseline = _build_minified_model(case.kind)
    baseline = baseline.to(device=device, dtype=dtype)
    baseline.eval()

    # Deterministic inputs across ranks.
    torch.manual_seed(999)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(999)
    # Keep this small for a fast functional test. Also avoid 0 to keep seq-cls pad-token
    # pooling deterministic.
    input_ids = torch.randint(1, int(cfg.vocab_size), (2, 1024), dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        baseline_logits = cast(
            torch.Tensor,
            baseline(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits,
        )

    # Rebuild model with identical weights, then TP-parallelize.
    torch.manual_seed(1234)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(1234)
    _, tp_model = _build_minified_model(case.kind)
    tp_model = tp_model.to(device=device, dtype=dtype)
    tp_model.eval()

    if case.kind == "nemotron":
        tp_shard_plan = "llama_nemotron_super_tp_plan"
    elif case.kind == "baichuan":
        tp_shard_plan = _baichuan_tp_plan()
    else:
        tp_shard_plan = None
    tp_mesh = DeviceMesh(device_type, torch.arange(world_size, device="cpu"), mesh_dim_names=("tp",))
    plan = _get_parallel_plan(tp_model, sequence_parallel=case.sequence_parallel, tp_shard_plan=tp_shard_plan)
    parallelize_module(tp_model, tp_mesh, plan)
    if case.kind == "nemotron":
        _update_attention_head_counts_for_tp(tp_model, tp_mesh.size())

    with torch.inference_mode():
        tp_logits = tp_model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    tp_logits_full = _maybe_gather_dtensor_to_replicated_local(tp_logits, tp_mesh=tp_mesh)

    kl = _kl_divergence_from_logits(reference_logits=baseline_logits, candidate_logits=tp_logits_full)
    # NOTE: keep this threshold loose; different TP reduction orders can introduce tiny numeric drift.
    ok = torch.all(kl <= kl_threshold)
    return ok, kl.view(-1).max().item()


def main(argv: Sequence[str] | None = None) -> int:
    # Ensure any required transformers compatibility patches are applied before we build models.
    apply_cache_compatibility_patches()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["qwen3", "qwen3_seq_cls", "ministral3", "llama", "qwen2", "nemotron"],
        choices=["qwen3", "qwen3_seq_cls", "ministral3", "llama", "qwen2", "nemotron", "baichuan"],
        help="Which models to test. 'nemotron' uses 10-layer full-hidden LlamaForCausalLM with llama_nemotron_super_tp_plan.",
    )
    parser.add_argument(
        "--sequence_parallel",
        default="both",
        choices=["true", "false", "both"],
        help="Run sequence_parallel=True/False/both.",
    )
    parser.add_argument(
        "--kl_threshold",
        type=float,
        default=2e-6,
        help="Fail if KL(TP=1 || TP=2) exceeds this threshold (per token). Default is bf16-appropriate.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Data type to use for the models.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    _init_distributed()
    rank = _rank()
    world_size = _world_size()

    if world_size != 2:
        if rank == 0:
            print(f"ERROR: This test requires world_size=2 (TP=2), got {world_size}", file=sys.stderr)
        return 1

    # Derive per-rank device after init.
    if torch.cuda.is_available() and dist.get_backend() == "nccl":
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        device = torch.device(f"cuda:{local_rank}")
        device_type = "cuda"
    else:
        device = torch.device("cpu")
        device_type = "cpu"

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.sequence_parallel == "both":
        sp_flags = [False, True]
    else:
        sp_flags = [args.sequence_parallel == "true"]

    cases = [
        _Case(kind=cast(ModelKind, k), sequence_parallel=sp)
        for k in args.models
        for sp in sp_flags
        if not (sp and k in _SP_UNSUPPORTED_MODELS)
    ]

    if any(c.kind == "nemotron" for c in cases) and not _is_nemotron_remote_code_available():
        cases = [c for c in cases if c.kind != "nemotron"]
        if rank == 0:
            print("SKIP: nemotron (DeciLM remote code not cached; run once with HF_HUB_OFFLINE=0 to cache)")

    all_ok = True

    for case in cases:
        kl_threshold = args.kl_threshold if case.kind != "baichuan" else 1e-4
        # Keep ranks roughly in sync for cleaner output.
        dist.barrier()
        ok, kl = _run_case(case, device=device, device_type=device_type, kl_threshold=kl_threshold, dtype=dtype)

        ok_tensor = torch.tensor(1 if ok else 0, device=device, dtype=torch.int)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)
        all_ok = all_ok and bool(ok_tensor.item())

        kl_tensor = torch.tensor(kl, device=device, dtype=torch.float32)
        dist.all_reduce(kl_tensor, op=dist.ReduceOp.MAX)

        if rank == 0:
            status = "PASS" if bool(ok_tensor.item()) else "FAIL"
            print(f"{status}: {case.name()} (kl_div={kl_tensor.item():.6g}, threshold={args.kl_threshold:g})")

    if rank == 0 and all_ok:
        print("PASS: all TP parity checks passed")

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
