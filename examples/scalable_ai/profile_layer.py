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

"""
Layer profiling script for DeepSeek-V4 / Moonlight-V4 models (scalable-ai, 2026 September edition).

Profiles one layer at a time (attention, MoE, full block, RMSNorm, hyper-connection mixer) with nsys / NVTX
ranges, comparing NeMo Automodel's native DeepSeek-V4 layers with the stock transformers layers.

DeepSeek-V4 has three attention kinds selected per layer by ``compress_ratios``: 0 = sliding-window only,
4 = Compressed Sparse Attention (ratio-4 overlapped compression + lightning indexer), 128 = Heavily Compressed
Attention. ``--compress-ratio`` picks which kind to profile (``--layer-idx`` overrides the layer explicitly).

Usage:
    # Automodel CSA attention layer, eager backend (any GPU) vs TileLang kernels (Hopper-class GPUs)
    nsys profile -c cudaProfilerApi -t cuda,nvtx -o attn_csa_eager \\
        python profile_layer.py --layer attn --compress-ratio 4 --backend-attn eager
    nsys profile -c cudaProfilerApi -t cuda,nvtx -o attn_csa_tilelang \\
        python profile_layer.py --layer attn --compress-ratio 4 --backend-attn tilelang

    # Stock transformers layers (sliding-window attention; see the note on --use-hf below)
    python profile_layer.py --layer attn --compress-ratio 0 --use-hf --no-nsys

    # MoE (torch._grouped_mm experts) vs transformers MoE; hyper-connection mixer; full block
    python profile_layer.py --layer moe --no-nsys
    python profile_layer.py --layer moe --use-hf --no-nsys
    python profile_layer.py --layer hc --no-nsys
    python profile_layer.py --layer block --compress-ratio 128 --no-nsys

Note on --use-hf: transformers' DeepSeek-V4 attention is inference-oriented. Without a KV cache its CSA layers
gather the per-query top-k entries into an ``S x k`` key axis (memory ~ S^2 * k) and its compressed entries carry
no causal mask, so HF profiles are meaningful for ratio 0 (sliding window) and, for timing only, ratio 128.
"""

import argparse
import dataclasses
import inspect
import json
import logging
import os
from typing import Literal

import torch
from huggingface_hub import hf_hub_download

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import initialize_rms_norm_module
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4HyperConnection,
    DeepseekV4RotaryEmbedding,
    _dsv4_kernel_backend,
    build_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4Block
from nemo_automodel.components.moe.layers import MoE, MoEConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

LayerType = Literal["attn", "moe", "block", "rmsnorm", "hc"]
RATIO_NAMES = {0: "sliding-window (SWA)", 4: "compressed sparse (CSA)", 128: "heavily compressed (HCA)"}

# Tensors the native implementation keeps in fp32 (see DeepseekV4ForCausalLM._keep_in_fp32_modules_strict).
FP32_NAME_TAGS = (
    "attn_hc",
    "ffn_hc",
    "hc_head",
    "sinks_param",
    "ape_param",
    "compressor.wkv",
    "compressor.wgate",
    "indexer.wkv",
    "indexer.wgate",
)


# ----------------------------------------------------------------------------------------------------------------
# Config handling
# ----------------------------------------------------------------------------------------------------------------
def load_config_dict(model_id: str) -> dict:
    """Load the raw ``config.json`` of a local model directory or a Hub repo."""
    path = (
        os.path.join(model_id, "config.json") if os.path.isdir(model_id) else hf_hub_download(model_id, "config.json")
    )
    return json.load(open(path))


def pick_layer_idx(compress_ratios: list[int], compress_ratio: int, layer_idx: int | None) -> int:
    """Return ``layer_idx`` if given, else the first layer whose compress ratio equals ``compress_ratio``."""
    if layer_idx is not None:
        return layer_idx
    for i, r in enumerate(compress_ratios):
        if r == compress_ratio:
            return i
    raise ValueError(f"no layer with compress ratio {compress_ratio} in {compress_ratios}")


def automodel_config(cfg_dict: dict) -> DeepseekV4Config:
    """Build NeMo Automodel's DeepSeek-V4 config for layer profiling (learned routing everywhere, bf16)."""
    cfg = DeepseekV4Config(
        **{k: v for k, v in cfg_dict.items() if k not in ("architectures", "model_type", "transformers_version")}
    )
    cfg.num_hash_layers = 0  # profile learned routing; the hash layer's fixed table is a from-config artefact
    cfg.torch_dtype = "bfloat16"
    return cfg


def hf_config(cfg_dict: dict, compress_ratio: int):
    """A one-layer transformers config of the requested attention kind (transformers maps ratios 0/4/128 only)."""
    from transformers import DeepseekV4Config as HFDeepseekV4Config

    d = {
        k: v
        for k, v in cfg_dict.items()
        if k not in ("architectures", "model_type", "torch_dtype", "transformers_version")
    }
    d.update(
        num_hidden_layers=1,
        compress_ratios=[compress_ratio],
        num_hash_layers=0,
        num_nextn_predict_layers=0,
        use_cache=False,
    )
    return HFDeepseekV4Config(**d)


# ----------------------------------------------------------------------------------------------------------------
# Layer construction
# ----------------------------------------------------------------------------------------------------------------
def moe_config_from(cfg: DeepseekV4Config, dtype: torch.dtype) -> MoEConfig:
    """Mirror DeepseekV4Model.__init__'s MoE defaults (no group-limited routing, sqrt-softplus, clamped SwiGLU)."""
    fields = {f.name for f in dataclasses.fields(MoEConfig)}
    d = dict(
        dim=cfg.hidden_size,
        inter_dim=cfg.moe_intermediate_size,
        moe_inter_dim=cfg.moe_intermediate_size,
        n_routed_experts=cfg.n_routed_experts,
        n_shared_experts=cfg.n_shared_experts,
        n_activated_experts=cfg.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=1e-3,
        score_func="sqrtsoftplus",
        route_scale=cfg.routed_scaling_factor,
        aux_loss_coeff=0,
        norm_topk_prob=cfg.norm_topk_prob,
        dtype=dtype,
        swiglu_limit=float(getattr(cfg, "swiglu_limit", 0.0) or 0.0),
    )
    return MoEConfig(**{k: v for k, v in d.items() if k in fields})


def _cast_like_model(layer: torch.nn.Module, dtype: torch.dtype) -> None:
    """Cast to ``dtype`` except the tensors the full model keeps in fp32."""
    for name, module in layer.named_modules():
        if isinstance(module, DeepseekV4HyperConnection) or any(tag in name for tag in FP32_NAME_TAGS):
            continue
        for pname, p in module.named_parameters(recurse=False):
            if p.is_floating_point() and not any(tag in f"{name}.{pname}" for tag in FP32_NAME_TAGS):
                p.data = p.data.to(dtype)


def create_automodel_layer(
    layer_type: LayerType,
    cfg: DeepseekV4Config,
    backend: BackendConfig,
    layer_idx: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.nn.Module:
    """Construct, initialise (under no_grad) and dtype-cast one native NeMo Automodel layer."""
    with torch.device(device):
        if layer_type == "attn":
            layer = DeepseekV4Attention(cfg, layer_idx, backend)
        elif layer_type == "moe":
            layer = MoE(moe_config_from(cfg, dtype), backend)
        elif layer_type == "block":
            layer = DeepseekV4Block(layer_idx, cfg, moe_config_from(cfg, dtype), backend)
        elif layer_type == "rmsnorm":
            layer = initialize_rms_norm_module(backend.rms_norm, cfg.hidden_size, eps=cfg.rms_norm_eps, dtype=dtype)
        elif layer_type == "hc":
            layer = DeepseekV4HyperConnection(
                cfg.hc_mult,
                cfg.hidden_size,
                cfg.hc_sinkhorn_iters,
                float(cfg.hc_eps),
                float(cfg.rms_norm_eps),
                sinkhorn_backend=_dsv4_kernel_backend(backend),
            )
        else:
            raise ValueError(layer_type)
    if any(p.device.type == "meta" for p in layer.parameters()):  # e.g. TE modules
        layer = layer.to_empty(device=device)
    init = getattr(layer, "init_weights", None) or getattr(layer, "reset_parameters", None)
    with torch.no_grad():  # the model-level initialize_weights runs under no_grad; the MoE initialiser writes in place
        if init is not None:
            params = inspect.signature(init).parameters
            kwargs = {}
            if "buffer_device" in params:
                kwargs["buffer_device"] = device
            if "init_std" in params:
                kwargs["init_std"] = cfg.initializer_range
            init(**kwargs)
        else:
            for p in layer.parameters():
                torch.nn.init.normal_(p, mean=0.0, std=cfg.initializer_range)
    _cast_like_model(layer, dtype)
    return layer


def create_hf_layer(layer_type: LayerType, hf_cfg, dtype: torch.dtype, device: torch.device):
    """Build a one-layer transformers model (proper _init_weights) and extract the requested sub-module."""
    from transformers import DeepseekV4ForCausalLM

    with torch.device(device):
        model = DeepseekV4ForCausalLM(hf_cfg)
    dec = model.model.layers[0]
    layer = {"attn": dec.self_attn, "moe": dec.mlp, "block": dec, "rmsnorm": dec.input_layernorm, "hc": dec.attn_hc}[
        layer_type
    ]
    rotary = model.model.rotary_emb
    layer = layer.to(dtype)  # HF runs the mHC / Sinkhorn math in fp32 internally
    return layer, rotary


# ----------------------------------------------------------------------------------------------------------------
# Inputs and forward calls
# ----------------------------------------------------------------------------------------------------------------
def create_inputs(
    layer_type: LayerType,
    cfg,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype,
    use_hf: bool,
    hf_rotary=None,
    hf_layer: torch.nn.Module | None = None,
) -> dict:
    """Random hidden states plus the rotary / mask tensors the chosen layer needs.

    Args:
        hf_rotary: The transformers rotary embedding module, when ``use_hf``.
        hf_layer: The transformers layer, when ``use_hf``; used to discover which rope layer type
            it indexes ``position_embeddings`` by, so the matching rotary is built for it.

    Returns:
        Mapping of keyword name to input. ``x`` is of shape [batch, sequence, hidden], or
        [batch, sequence, hc_mult, hidden] for the ``block`` and ``hc`` layers which carry
        manifold-constrained hyper-connection residual streams. ``position_ids`` is
        [batch, sequence]; ``attention_mask`` is [batch, 1, sequence, sequence];
        ``position_embeddings`` is a ``(cos, sin)`` pair of [batch, sequence, head_dim] tensors,
        keyed by rope layer type on the transformers path.
    """
    D, hc = cfg.hidden_size, cfg.hc_mult
    inputs = {}
    if layer_type in ("block", "hc"):
        inputs["x"] = torch.randn(batch_size, seq_len, hc, D, device=device, dtype=dtype, requires_grad=True)
    else:
        inputs["x"] = torch.randn(batch_size, seq_len, D, device=device, dtype=dtype, requires_grad=True)
    if layer_type in ("attn", "block"):
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        ref = torch.empty(batch_size, seq_len, D, device=device, dtype=dtype)
        inputs["position_ids"] = position_ids
        if use_hf:
            from transformers.masking_utils import create_sliding_window_causal_mask

            # transformers indexes position_embeddings by the layer's own rope type
            # (modeling_deepseek_v4.py: ``cos, sin = position_embeddings[self.rope_layer_type]``),
            # where sliding-window layers use "main" and compressed ones "compress" -- different
            # theta and scaling.  Build the rotary for whichever the layer declares, so CSA/HCA are
            # not silently profiled with the sliding-window frequencies.
            rope_layer_type = getattr(getattr(hf_layer, "self_attn", hf_layer), "rope_layer_type", "main")
            inputs["position_embeddings"] = {rope_layer_type: hf_rotary(ref, position_ids, layer_type=rope_layer_type)}
            inputs["attention_mask"] = create_sliding_window_causal_mask(
                config=cfg, inputs_embeds=ref, attention_mask=None, past_key_values=None, position_ids=position_ids
            )
        else:
            prf = cfg.qk_rope_head_dim / cfg.head_dim
            rotary_main = DeepseekV4RotaryEmbedding(
                rope_theta=float(cfg.rope_theta),
                head_dim=cfg.head_dim,
                partial_rotary_factor=prf,
                device=device,
                rope_scaling=None,
            )
            rotary_compress = DeepseekV4RotaryEmbedding(
                rope_theta=float(cfg.compress_rope_theta),
                head_dim=cfg.head_dim,
                partial_rotary_factor=prf,
                device=device,
                rope_scaling=getattr(cfg, "rope_scaling", None),
            )
            inputs["position_embeddings"] = rotary_main(ref, position_ids)
            inputs["position_embeddings_compress"] = rotary_compress(ref, position_ids)
            inputs["rotary_compress"] = rotary_compress
            inputs["attention_mask"] = build_causal_padding_mask(
                None, seq_len, dtype, device, batch_size=batch_size, sliding_window=cfg.sliding_window
            )
    return inputs


def run_forward(layer, inputs: dict, layer_type: LayerType, use_hf: bool):
    """Call the layer with the inputs each implementation expects and return its main output tensor."""
    x = inputs["x"]
    if layer_type == "attn":
        if use_hf:
            out = layer(
                hidden_states=x,
                position_embeddings=inputs["position_embeddings"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
            )
        else:
            out = layer(
                x,
                position_embeddings=inputs["position_embeddings"],
                attention_mask=inputs["attention_mask"],
                position_embeddings_compress=inputs["position_embeddings_compress"],
                rotary_compress=inputs["rotary_compress"],
                position_ids=inputs["position_ids"],
            )
        return out[0] if isinstance(out, tuple) else out
    if layer_type == "block":
        if use_hf:
            return layer(
                x,
                input_ids=None,
                position_embeddings=inputs["position_embeddings"],
                position_ids=inputs["position_ids"],
                attention_mask=inputs["attention_mask"],
            )
        return layer(
            x,
            position_embeddings=inputs["position_embeddings"],
            position_ids=inputs["position_ids"],
            position_embeddings_compress=inputs["position_embeddings_compress"],
            rotary_compress=inputs["rotary_compress"],
            attention_mask=inputs["attention_mask"],
        )
    if layer_type == "moe":
        if use_hf:
            return layer(x)
        return layer(x.reshape(-1, x.shape[-1])).view(x.shape)
    return layer(x)  # rmsnorm, hc


def loss_of(output) -> torch.Tensor:
    """Scalar loss over every differentiable output tensor (mean, in fp32)."""
    outs = output if isinstance(output, (tuple, list)) else (output,)
    return sum(o.float().mean() for o in outs if isinstance(o, torch.Tensor) and o.requires_grad)


def run_forward_backward(layer, inputs, layer_type, iteration, use_nvtx, use_hf):
    """One forward + backward pass wrapped in NVTX ranges."""
    if use_nvtx:
        torch.cuda.nvtx.range_push(f"iter_{iteration}_forward")
    output = run_forward(layer, inputs, layer_type, use_hf)
    if use_nvtx:
        torch.cuda.nvtx.range_pop()
        torch.cuda.nvtx.range_push(f"iter_{iteration}_backward")
    loss = loss_of(output)
    loss.backward()
    if use_nvtx:
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    return loss


# ----------------------------------------------------------------------------------------------------------------
def main():
    """Parse arguments, build the requested layer and run the profiling loop."""
    parser = argparse.ArgumentParser(description="Profile individual layers of DeepSeek-V4 / Moonlight-V4 models")
    parser.add_argument(
        "--model-id", type=str, default="akoumpa/Moonlight-V4-16B-A3B", help="HF model id or local dir with config.json"
    )
    parser.add_argument("--layer", type=str, choices=["attn", "moe", "block", "rmsnorm", "hc"], default="attn")
    parser.add_argument(
        "--compress-ratio",
        type=int,
        choices=[0, 4, 128],
        default=4,
        help="attention kind for attn/block: 0 sliding-window, 4 CSA (+indexer), 128 HCA",
    )
    parser.add_argument(
        "--layer-idx", type=int, default=None, help="explicit layer index (default: first layer of the requested kind)"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--warmup-iters", type=int, default=5)
    parser.add_argument("--profile-iters", type=int, default=10)
    parser.add_argument("--nsys-start", type=int, default=None)
    parser.add_argument("--nsys-end", type=int, default=None)
    parser.add_argument("--no-nsys", action="store_true", help="disable cudaProfilerApi / NVTX calls")
    parser.add_argument(
        "--backend-attn",
        type=str,
        default="eager",
        choices=["eager", "tilelang"],
        help="Automodel attention path: eager (dense masked attention with sinks) or TileLang sparse kernels",
    )
    parser.add_argument("--backend-rms-norm", type=str, default="torch_fp32", choices=["torch", "torch_fp32", "te"])
    parser.add_argument("--backend-experts", type=str, default="torch_mm", choices=["torch", "torch_mm", "gmm", "te"])
    parser.add_argument("--backend-linear", type=str, default="torch", choices=["torch", "te"])
    parser.add_argument(
        "--use-hf", action="store_true", help="profile the stock transformers layer instead of NeMo Automodel's"
    )
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    cfg_dict = load_config_dict(args.model_id)
    layer_idx = pick_layer_idx(cfg_dict["compress_ratios"], args.compress_ratio, args.layer_idx)
    ratio = cfg_dict["compress_ratios"][layer_idx]
    logger.info(
        f"Model {args.model_id}: hidden {cfg_dict['hidden_size']}, {cfg_dict['num_attention_heads']} heads x {cfg_dict['head_dim']}, "
        f"{cfg_dict['n_routed_experts']} experts top-{cfg_dict['num_experts_per_tok']}, hc_mult {cfg_dict['hc_mult']}"
    )
    logger.info(
        f"Layer {args.layer} at index {layer_idx} ({RATIO_NAMES.get(ratio, ratio)}); source: {'transformers' if args.use_hf else 'NeMo Automodel'}"
    )

    hf_rotary = None
    if args.use_hf:
        if args.layer in ("attn", "block") and ratio == 4:
            logger.warning(
                "transformers CSA gathers S x k keys per layer without a cache; expect very high memory and untrustworthy timings"
            )
        cfg = hf_config(cfg_dict, ratio)
        layer, hf_rotary = create_hf_layer(args.layer, cfg, dtype, device)
    else:
        cfg = automodel_config(cfg_dict)
        backend = BackendConfig(
            attn=args.backend_attn,
            linear=args.backend_linear,
            rms_norm=args.backend_rms_norm,
            rope_fusion=False,
            experts=args.backend_experts,
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        logger.info(
            f"Backend: attn={backend.attn} linear={backend.linear} rms_norm={backend.rms_norm} experts={backend.experts}"
        )
        layer = create_automodel_layer(args.layer, cfg, backend, layer_idx, dtype, device)
    layer.train()

    n_params = sum(p.numel() for p in layer.parameters())
    n_fp32 = sum(p.numel() for p in layer.parameters() if p.dtype == torch.float32)
    logger.info(f"Parameters: {n_params:,} ({n_fp32:,} kept in fp32)")

    inputs = create_inputs(
        args.layer, cfg, args.batch_size, args.seq_len, device, dtype, args.use_hf, hf_rotary, hf_layer=layer
    )
    nsys_start = args.nsys_start if args.nsys_start is not None else args.warmup_iters
    nsys_end = args.nsys_end if args.nsys_end is not None else args.warmup_iters + args.profile_iters - 1
    total_iters = args.warmup_iters + args.profile_iters
    use_nvtx = not args.no_nsys
    logger.info(
        f"Running {total_iters} iterations ({args.warmup_iters} warmup + {args.profile_iters} profile), "
        f"batch {args.batch_size} x seq {args.seq_len}, nsys window [{nsys_start}, {nsys_end}]"
    )

    iter_times = []
    torch.cuda.synchronize()
    for i in range(total_iters):
        is_warmup = i < args.warmup_iters
        if use_nvtx and i == nsys_start:
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()
        for p in layer.parameters():
            p.grad = None
        inputs["x"] = torch.randn_like(inputs["x"], requires_grad=True)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if use_nvtx:
            torch.cuda.nvtx.range_push(f"iteration_{i}_{'warmup' if is_warmup else 'profile'}")
        start.record()
        loss = run_forward_backward(layer, inputs, args.layer, i, use_nvtx, args.use_hf)
        end.record()
        if use_nvtx:
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        t = start.elapsed_time(end)
        if not is_warmup:
            iter_times.append(t)
        logger.info(
            f"Iter {i:3d} {'[warmup] ' if is_warmup else '[profile]'}: {t:8.3f} ms | loss={loss.item():.4f} | "
            f"peak mem={torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )
        if use_nvtx and i == nsys_end:
            torch.cuda.cudart().cudaProfilerStop()

    if iter_times:
        logger.info("=" * 60)
        logger.info(
            f"{args.layer} ({RATIO_NAMES.get(ratio, ratio)}) | {'transformers' if args.use_hf else 'NeMo Automodel ' + args.backend_attn} | "
            f"{args.model_id} | batch {args.batch_size} x seq {args.seq_len}"
        )
        logger.info(f"Parameters: {n_params:,} | iterations: {len(iter_times)}")
        logger.info(
            f"Average fwd+bwd: {sum(iter_times) / len(iter_times):.3f} ms (min {min(iter_times):.3f}, max {max(iter_times):.3f})"
        )
        logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
