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

"""Numerical parity against the released DeepSeek V4.1 reference implementation.

The reference is the ``inference/`` package shipped with
``deepseek-ai/DeepSeek-V4.1-Flash`` (``model.py``, ``engram.py``, ``kernel.py``,
``vision.py``, ``image_processor.py``).  It needs CUDA and TileLang, so this test
is opt-in:

    DSV41_REFERENCE_DIR=/path/to/DeepSeek-V4.1-Flash \\
    DSV41_TOKENIZER_DIR=/path/to/DeepSeek-V4.1-Flash \\
    pytest tests/functional_tests/models/deepseek_v41/test_dsv41_reference_parity.py -s

A tiny randomly initialised reference ``Transformer`` (bf16 weights, no DSpark,
no vision) is converted through the state-dict adapter into
``DeepseekV41ForCausalLM`` and the full-sequence logits are compared.
"""

from __future__ import annotations

import functools
import importlib
import os
import sys

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.engram import EngramLayout
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

REFERENCE_DIR = os.environ.get("DSV41_REFERENCE_DIR")
TOKENIZER_DIR = os.environ.get("DSV41_TOKENIZER_DIR")

pytestmark = pytest.mark.skipif(
    not REFERENCE_DIR or not torch.cuda.is_available(),
    reason="set DSV41_REFERENCE_DIR to the released inference package and run on a CUDA device",
)

VOCAB = 129280  # the released tokenizer size; Engram hashing needs the real vocabulary
COMPRESSED_VOCAB = 99092
SHARED = dict(
    hidden=128,
    moe_inter=64,
    n_layers=6,
    n_heads=4,
    head_dim=64,
    rope_head_dim=16,
    q_lora_rank=32,
    o_lora_rank=16,
    o_groups=2,
    n_experts=8,
    n_active=2,
    window=8,
    compress_ratios=(0, 0, 2, 2, 1, 1),
    kv_sources=(2, 4),
    index_sources=(2, 4, 5),
    index_heads=2,
    index_head_dim=32,
    index_topk=6,
    candidate_layer=4,
    candidate_blocks=2,
    candidate_block_size=4,
    yarn_original=64,
    yarn_factor=4,
)


def _load_reference_module():
    inference_dir = os.path.join(REFERENCE_DIR, "inference")
    if inference_dir not in sys.path:
        sys.path.insert(0, inference_dir)
    return importlib.import_module("model")


def _engram_layout(engram: bool) -> EngramLayout | None:
    if not engram:
        return None
    probe = DeepseekV41Config(
        engram_layer_ids=[1],
        engram_num_embeddings=[1],
        engram_max_ngram_size=3,
        engram_vocab_size=500,
        engram_n_heads=2,
        engram_head_dim=32,
    )
    return EngramLayout.from_config(probe)


def _reference_args(ref, engram: bool):
    layout = _engram_layout(engram)
    engram_kwargs = (
        dict(
            engram_layer_ids=(1,),
            engram_num_embeddings=(layout.bucket_span(0),),
            engram_max_ngram_size=3,
            engram_vocab_size=500,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_pad_id=2,
            engram_compressed_vocab_size=COMPRESSED_VOCAB,
        )
        if engram
        else {}
    )
    return ref.ModelArgs(
        max_batch_size=2,
        max_seq_len=256,
        dtype="bf16",
        expert_dtype=None,
        vocab_size=VOCAB,
        dim=SHARED["hidden"],
        moe_inter_dim=SHARED["moe_inter"],
        n_layers=SHARED["n_layers"],
        n_mtp_layers=0,
        n_heads=SHARED["n_heads"],
        n_routed_experts=SHARED["n_experts"],
        n_shared_experts=1,
        n_activated_experts=SHARED["n_active"],
        score_func="sqrtsoftplus",
        norm_topk_prob=True,
        route_scale=1.5,
        swiglu_limit=10.0,
        q_lora_rank=SHARED["q_lora_rank"],
        head_dim=SHARED["head_dim"],
        rope_head_dim=SHARED["rope_head_dim"],
        norm_eps=1e-20,
        o_groups=SHARED["o_groups"],
        o_lora_rank=SHARED["o_lora_rank"],
        window_size=SHARED["window"],
        compress_ratios=SHARED["compress_ratios"],
        kv_source_layers=SHARED["kv_sources"],
        index_source_layers=SHARED["index_sources"],
        compress_rope_theta=160000.0,
        original_seq_len=SHARED["yarn_original"],
        rope_theta=10000.0,
        rope_factor=SHARED["yarn_factor"],
        beta_fast=32,
        beta_slow=1,
        index_n_heads=SHARED["index_heads"],
        index_head_dim=SHARED["index_head_dim"],
        index_topk=SHARED["index_topk"],
        candidate_source_layer=SHARED["candidate_layer"],
        candidate_topk_blocks=SHARED["candidate_blocks"],
        candidate_block_size=SHARED["candidate_block_size"],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        dspark_block_size=0,
        **engram_kwargs,
    )


def _automodel_config(engram: bool) -> DeepseekV41Config:
    layout = _engram_layout(engram)
    engram_kwargs = (
        dict(
            engram_layer_ids=[1],
            engram_num_embeddings=[layout.bucket_span(0)],
            engram_max_ngram_size=3,
            engram_vocab_size=500,
            engram_n_heads=2,
            engram_head_dim=32,
            engram_pad_token_id=2,
            engram_compressed_vocab_size=COMPRESSED_VOCAB,
        )
        if engram
        else dict(engram_layer_ids=[], engram_num_embeddings=[])
    )
    return DeepseekV41Config(
        vocab_size=VOCAB,
        hidden_size=SHARED["hidden"],
        moe_intermediate_size=SHARED["moe_inter"],
        num_hidden_layers=SHARED["n_layers"],
        num_attention_heads=SHARED["n_heads"],
        head_dim=SHARED["head_dim"],
        qk_rope_head_dim=SHARED["rope_head_dim"],
        q_lora_rank=SHARED["q_lora_rank"],
        o_lora_rank=SHARED["o_lora_rank"],
        o_groups=SHARED["o_groups"],
        n_routed_experts=SHARED["n_experts"],
        n_shared_experts=1,
        num_experts_per_tok=SHARED["n_active"],
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        rms_norm_eps=1e-20,
        max_position_embeddings=256,
        rope_theta=10000.0,
        compress_rope_theta=160000.0,
        rope_scaling={
            "rope_type": "yarn",
            "factor": SHARED["yarn_factor"],
            "original_max_position_embeddings": SHARED["yarn_original"],
            "beta_fast": 32,
            "beta_slow": 1,
        },
        sliding_window=SHARED["window"],
        compress_ratios=list(SHARED["compress_ratios"]),
        kv_source_layer_ids=list(SHARED["kv_sources"]),
        index_source_layer_ids=list(SHARED["index_sources"]),
        index_n_heads=SHARED["index_heads"],
        index_head_dim=SHARED["index_head_dim"],
        index_topk=SHARED["index_topk"],
        candidate_source_layer_id=SHARED["candidate_layer"],
        candidate_topk_blocks=SHARED["candidate_blocks"],
        candidate_block_size=SHARED["candidate_block_size"],
        hc_mult=4,
        hc_sinkhorn_iters=20,
        hc_eps=1e-6,
        torch_dtype="bfloat16",
        **engram_kwargs,
    )


@torch.no_grad()
def _randomize_reference(ref_model: torch.nn.Module, seed: int) -> None:
    """Fill the reference's uninitialised parameters with a deterministic random model."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    for name, param in ref_model.named_parameters():
        if name.endswith(("hc_attn_scale", "hc_ffn_scale")):
            param.fill_(1.0)
        elif name.endswith(("hc_attn_base", "hc_ffn_base")):
            param.zero_()
        elif name.endswith("gate.bias"):
            param.zero_()
        elif name.endswith(("q_weight", "k_weight")):
            param.fill_(1.0)
        elif name.endswith("engram.embed.scale"):
            param.copy_(torch.full(param.shape, 127, dtype=torch.uint8, device=param.device).view(param.dtype))
        elif "norm" in name and param.dim() == 1:
            param.fill_(1.0)
        elif name.endswith("attn_sink"):
            param.copy_(torch.randn(param.shape, generator=generator, device=param.device) * 0.1)
        else:
            values = torch.randn(param.shape, generator=generator, device=param.device, dtype=torch.float32)
            std = 0.5 if name.endswith("engram.embed.weight") else 0.02
            param.copy_((values * std).to(param.dtype))


def _build_pair(engram: bool, attn_backend: str, seed: int = 0):
    ref = _load_reference_module()
    tokenizer = None
    if engram:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR or REFERENCE_DIR)
    args = _reference_args(ref, engram)
    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("cuda"):
            ref_model = ref.Transformer(args, tokenizer)
    finally:
        torch.set_default_dtype(prev_dtype)
    _randomize_reference(ref_model, seed)
    # Full-sequence logits instead of the last position only.
    ref_model.head.forward = functools.partial(ref_model.head.forward, full_logits=True)

    config = _automodel_config(engram)
    backend = BackendConfig(
        attn=attn_backend,
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch_mm",
        enable_hf_state_dict_adapter=True,
        # The reference scores the router in fp32 (``linear(x.float(), weight.float())``).
        gate_precision=torch.float32,
    )
    model = DeepseekV41ForCausalLM(config, backend=backend)
    cast_model_to_dtype(model, torch.bfloat16)
    hf_state = {k: v.detach().clone().cpu() for k, v in ref_model.state_dict().items()}
    converted = model.state_dict_adapter.from_hf(hf_state)
    missing, unexpected = model.load_state_dict(converted, strict=False)
    assert not unexpected, unexpected
    assert not missing, missing
    model = model.cuda().eval()
    if engram:
        model.set_engram_tokenizer(tokenizer)
    return ref_model, model


def _compare(ref_logits: torch.Tensor, logits: torch.Tensor, label: str) -> None:
    ref_logits = ref_logits.float()
    logits = logits.float()
    diff = (ref_logits - logits).abs()
    cosine = torch.nn.functional.cosine_similarity(ref_logits, logits, dim=-1)
    top1 = (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
    print(
        f"[{label}] max_abs={diff.max().item():.4f} mean_abs={diff.mean().item():.5f} "
        f"min_cos={cosine.min().item():.5f} mean_cos={cosine.mean().item():.5f} top1_agree={top1:.4f} "
        f"ref_std={ref_logits.std().item():.4f}"
    )
    assert cosine.min().item() > 0.99, f"{label}: logits diverge from the reference (min cosine {cosine.min().item()})"
    assert top1 > 0.95, f"{label}: top-1 agreement {top1}"


@pytest.mark.parametrize("attn_backend", ["sdpa", "tilelang"])
@pytest.mark.parametrize("engram", [False, True])
def test_logits_match_reference(engram: bool, attn_backend: str):
    if engram and not (TOKENIZER_DIR or os.path.exists(os.path.join(REFERENCE_DIR, "tokenizer.json"))):
        pytest.skip("Engram parity needs DSV41_TOKENIZER_DIR (or tokenizer files next to the reference)")
    if attn_backend == "tilelang":
        from nemo_automodel.components.models.deepseek_v4.optimized_kernels import is_dsv4_kernel_available

        if not is_dsv4_kernel_available("sparse_attn"):
            pytest.skip("TileLang sparse attention unavailable")
    ref_model, model = _build_pair(engram, attn_backend)
    torch.manual_seed(1234)
    tokens = torch.randint(0, VOCAB, (2, 37), device="cuda")

    # Per-layer diagnostics: capture the HC streams leaving every block on both sides.
    ref_streams: list[torch.Tensor] = []
    our_streams: list[torch.Tensor] = []
    hooks = [
        layer.register_forward_hook(lambda _m, _i, out: ref_streams.append(out[0].detach().float()))
        for layer in ref_model.layers
    ]
    hooks += [
        layer.register_forward_hook(lambda _m, _i, out: our_streams.append(out[0].detach().float()))
        for layer in model.model.layers.values()
    ]
    prev_device = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        with torch.inference_mode():
            _, ref_logits, _ = ref_model(tokens, 0)
    finally:
        torch.set_default_device(prev_device)
    with torch.no_grad():
        logits = model(tokens).logits
    for hook in hooks:
        hook.remove()
    for idx, (ref_h, our_h) in enumerate(zip(ref_streams, our_streams)):
        cos = torch.nn.functional.cosine_similarity(ref_h.flatten(2), our_h.flatten(2), dim=-1)
        rel = ((ref_h - our_h).norm() / ref_h.norm()).item()
        print(f"  layer {idx} ({model.config.csa2_mode(idx)}): min_cos={cos.min().item():.5f} rel_err={rel:.5f}")
    assert logits.shape == ref_logits.shape == (2, 37, VOCAB)
    _compare(ref_logits, logits, f"engram={engram} attn={attn_backend}")
