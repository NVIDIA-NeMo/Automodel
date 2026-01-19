# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
Layer Profiling Script for DeepSeek V3 / Moonlight Models

This script profiles individual layers (MLA, MLP, MoE, Block) using nsys and nvtx tags.
It initializes layers from an AutoConfig and runs forward/backward passes with profiling.

Supports both custom NeMo Automodel layers and stock HuggingFace layers for comparison.

Usage:
    # Profile custom MoE layer with nsys
    nsys profile -c cudaProfilerApi -t cuda,nvtx -o moe_profile \
        python profile_layer.py --model-id moonshotai/Moonlight-16B-A3B --layer moe

    # Profile stock HuggingFace MoE layer
    nsys profile -c cudaProfilerApi -t cuda,nvtx -o moe_hf_profile \
        python profile_layer.py --model-id moonshotai/Moonlight-16B-A3B --layer moe --use-hf

    # Profile MLA attention layer (custom vs HF)
    python profile_layer.py --layer mla --no-nsys
    python profile_layer.py --layer mla --use-hf --no-nsys

    # Profile full transformer block
    python profile_layer.py --layer block --no-nsys
    python profile_layer.py --layer block --use-hf --no-nsys

    # Run without nsys (for debugging)
    python profile_layer.py --model-id moonshotai/Moonlight-16B-A3B --layer moe --no-nsys
"""

import argparse
import logging
from typing import Literal

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from nemo_automodel.components.models.deepseek_v3.layers import MLA
from nemo_automodel.components.models.deepseek_v3.model import Block
from nemo_automodel.components.models.deepseek_v3.rope_utils import freqs_cis_from_position_ids, precompute_freqs_cis
from nemo_automodel.components.moe.layers import MLP, MoE, MoEConfig
from nemo_automodel.components.moe.utils import BackendConfig, initialize_rms_norm_module

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


LayerType = Literal["mla", "mlp", "moe", "block", "rmsnorm"]
LayerSource = Literal["custom", "hf"]


def create_moe_config_from_hf(config, dtype: torch.dtype = torch.bfloat16) -> MoEConfig:
    """Create MoEConfig from HuggingFace config."""
    return MoEConfig(
        dim=config.hidden_size,
        inter_dim=config.intermediate_size,
        moe_inter_dim=config.moe_intermediate_size,
        n_routed_experts=config.n_routed_experts,
        n_shared_experts=config.n_shared_experts,
        n_activated_experts=config.num_experts_per_tok,
        n_expert_groups=config.n_group,
        n_limited_groups=config.topk_group,
        train_gate=True,
        gate_bias_update_factor=0.001,
        score_func="sigmoid",
        route_scale=config.routed_scaling_factor,
        aux_loss_coeff=0,
        norm_topk_prob=config.norm_topk_prob,
        dtype=dtype,
    )


def create_layer(
    layer_type: LayerType,
    config,
    backend: BackendConfig,
    layer_idx: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Create a single custom NeMo Automodel layer based on layer type."""
    moe_config = create_moe_config_from_hf(config, dtype=dtype)

    if layer_type == "mla":
        return MLA(config, backend)
    elif layer_type == "mlp":
        return MLP(config.hidden_size, config.intermediate_size, backend.linear, dtype=dtype)
    elif layer_type == "moe":
        return MoE(moe_config, backend)
    elif layer_type == "block":
        # Use layer_idx >= first_k_dense_replace to get MoE block
        return Block(layer_idx, config, moe_config, backend)
    elif layer_type == "rmsnorm":
        return initialize_rms_norm_module(
            rms_norm_impl=backend.rms_norm,
            dim=config.hidden_size,
            eps=config.rms_norm_eps,
            dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def create_hf_layer(
    layer_type: LayerType,
    config,
    layer_idx: int = 3,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.nn.Module:
    """Create a stock HuggingFace layer from AutoModel.from_config.

    This extracts individual components from the full HuggingFace model
    to allow profiling of specific layer types.
    """
    # Create the full model on meta device first to get layer structure
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(config)

    # Get the model's inner transformer (usually model.model for CausalLM)
    if hasattr(model, "model"):
        inner_model = model.model
    else:
        inner_model = model

    # Get the layers list
    if hasattr(inner_model, "layers"):
        layers = inner_model.layers
    elif hasattr(inner_model, "decoder") and hasattr(inner_model.decoder, "layers"):
        layers = inner_model.decoder.layers
    else:
        raise ValueError(f"Could not find layers in model: {type(model)}")

    # Select the appropriate layer based on layer_type
    if layer_type == "block":
        # Return the full decoder layer
        layer = layers[layer_idx]
    elif layer_type == "mla":
        # Return just the attention component
        decoder_layer = layers[layer_idx]
        if hasattr(decoder_layer, "self_attn"):
            layer = decoder_layer.self_attn
        else:
            raise ValueError(f"Could not find attention in decoder layer: {type(decoder_layer)}")
    elif layer_type == "mlp":
        # Return just the MLP component (from a dense layer, not MoE)
        # Use layer 0 which is typically dense
        decoder_layer = layers[0]
        if hasattr(decoder_layer, "mlp"):
            layer = decoder_layer.mlp
        else:
            raise ValueError(f"Could not find mlp in decoder layer: {type(decoder_layer)}")
    elif layer_type == "moe":
        # Return just the MoE component (from an MoE layer)
        decoder_layer = layers[layer_idx]
        if hasattr(decoder_layer, "mlp"):
            layer = decoder_layer.mlp
        else:
            raise ValueError(f"Could not find mlp/moe in decoder layer: {type(decoder_layer)}")
    elif layer_type == "rmsnorm":
        # Return an RMSNorm layer from the model (e.g., input_layernorm)
        decoder_layer = layers[layer_idx]
        if hasattr(decoder_layer, "input_layernorm"):
            layer = decoder_layer.input_layernorm
        elif hasattr(inner_model, "norm"):
            # Use the final norm layer
            layer = inner_model.norm
        else:
            raise ValueError(f"Could not find rmsnorm in decoder layer: {type(decoder_layer)}")
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")

    return layer


def create_inputs(
    layer_type: LayerType,
    config,
    batch_size: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    use_hf: bool = False,
    rope_fusion: bool = False,
) -> dict:
    """Create input tensors for the layer."""
    hidden_size = config.hidden_size

    # Main hidden states input
    x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype, requires_grad=True)

    inputs = {"x": x}

    if use_hf:
        # HuggingFace layers use position_embeddings (cos, sin) for attention
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        inputs["position_ids"] = position_ids
        # No attention mask (causal mask is applied internally)
        inputs["attention_mask"] = None

        if layer_type in ("mla", "block"):
            # HF attention needs position_embeddings as (cos, sin) tuple
            # Compute rotary embeddings similar to HF's approach
            position_embeddings = _compute_hf_position_embeddings(config, position_ids, device, dtype)
            inputs["position_embeddings"] = position_embeddings

        if layer_type == "moe":
            # HF MoE doesn't need position embeddings, just hidden_states
            pass
    else:
        # Custom NeMo Automodel layers
        if layer_type in ("mla", "block"):
            # MLA and Block need freqs_cis for rotary embeddings
            # First precompute the base frequencies
            freqs = precompute_freqs_cis(
                config.qk_rope_head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                config.rope_scaling,
            ).to(device)
            # Create position_ids and convert to proper freqs_cis format
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
            freqs_cis = freqs_cis_from_position_ids(position_ids, freqs, for_fused_rope=rope_fusion)
            inputs["freqs_cis"] = freqs_cis
            inputs["attention_mask"] = None

        if layer_type == "moe":
            # MoE needs padding_mask (None = no padding)
            inputs["padding_mask"] = None

    return inputs


def _compute_hf_position_embeddings(config, position_ids, device, dtype):
    """Compute position embeddings (cos, sin) for HuggingFace attention layers."""
    # Based on HuggingFace DeepseekV3RotaryEmbedding
    dim = config.qk_rope_head_dim
    base = config.rope_theta

    # Compute inverse frequencies
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))

    # Handle rope scaling if present
    if config.rope_scaling is not None:
        factor = config.rope_scaling.get("factor", 1.0)
        inv_freq = inv_freq / factor

    # Compute position embeddings
    # position_ids: [batch, seq_len]
    # inv_freq: [dim // 2]
    freqs = torch.outer(position_ids[0].float(), inv_freq)  # [seq_len, dim // 2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq_len, dim]

    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)

    # Expand for batch size
    cos = cos.unsqueeze(0)  # [1, seq_len, dim]
    sin = sin.unsqueeze(0)  # [1, seq_len, dim]

    return (cos, sin)


def run_forward_backward(
    layer: torch.nn.Module,
    inputs: dict,
    layer_type: LayerType,
    iteration: int,
    use_nvtx: bool = True,
    use_hf: bool = False,
) -> torch.Tensor:
    """Run forward and backward pass with optional NVTX annotations."""
    x = inputs["x"]

    # Forward pass
    if use_nvtx:
        torch.cuda.nvtx.range_push(f"iter_{iteration}_forward")

    if use_hf:
        # HuggingFace layer forward pass
        output = _run_hf_forward(layer, x, inputs, layer_type)
    else:
        # Custom NeMo Automodel layer forward pass
        output = _run_custom_forward(layer, x, inputs, layer_type)

    if use_nvtx:
        torch.cuda.nvtx.range_pop()

    # Backward pass
    if use_nvtx:
        torch.cuda.nvtx.range_push(f"iter_{iteration}_backward")

    # Create a simple loss for backward
    # Handle tuple outputs (some HF layers return tuples)
    if isinstance(output, tuple):
        output = output[0]
    loss = output.sum()
    loss.backward()

    if use_nvtx:
        # Synchronize to ensure backward kernels finish before popping range
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

    return loss


def _run_custom_forward(layer, x, inputs, layer_type):
    """Run forward pass for custom NeMo Automodel layers."""
    if layer_type == "mla":
        return layer(x, inputs["freqs_cis"], inputs.get("attention_mask"))
    elif layer_type == "mlp":
        return layer(x)
    elif layer_type == "moe":
        # Reshape for MoE: (batch, seq, hidden) -> (batch * seq, hidden)
        x_flat = x.view(-1, x.shape[-1])
        output = layer(x_flat, inputs.get("padding_mask"))
        # Reshape back
        return output.view(x.shape)
    elif layer_type == "block":
        return layer(x, inputs["freqs_cis"], inputs.get("attention_mask"))
    elif layer_type == "rmsnorm":
        return layer(x)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def _run_hf_forward(layer, x, inputs, layer_type):
    """Run forward pass for HuggingFace layers."""
    attention_mask = inputs.get("attention_mask")
    position_embeddings = inputs.get("position_embeddings")

    if layer_type == "mla":
        # HF attention layer - DeepseekV3Attention/FlashAttention2
        # Takes hidden_states, position_embeddings (cos, sin), attention_mask
        output = layer(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
        )
        # Returns tuple: (attn_output, attn_weights, past_key_value)
        return output[0] if isinstance(output, tuple) else output
    elif layer_type == "mlp":
        # HF MLP layer - just takes hidden_states
        return layer(x)
    elif layer_type == "moe":
        # HF MoE layer - DeepseekV3MoE
        # Takes hidden_states, returns hidden_states
        return layer(x)
    elif layer_type == "block":
        # HF decoder layer - DeepseekV3DecoderLayer
        # Takes hidden_states, attention_mask, position_embeddings
        output = layer(
            hidden_states=x,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        # Returns tuple: (hidden_states, ...) or just hidden_states
        return output[0] if isinstance(output, tuple) else output
    elif layer_type == "rmsnorm":
        # HF RMSNorm layer - just takes hidden_states
        return layer(x)
    else:
        raise ValueError(f"Unknown layer type: {layer_type}")


def main():
    parser = argparse.ArgumentParser(description="Profile individual layers from DeepSeek V3 / Moonlight models")

    # Model configuration
    parser.add_argument(
        "--model-id",
        type=str,
        default="moonshotai/Moonlight-16B-A3B",
        help="HuggingFace model ID or path to config",
    )
    parser.add_argument(
        "--layer",
        type=str,
        choices=["mla", "mlp", "moe", "block", "rmsnorm"],
        default="moe",
        help="Type of layer to profile",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=3,
        help="Layer index (used for Block to determine if MoE or dense)",
    )

    # Input configuration
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=4096, help="Sequence length")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16", "float32"])

    # Profiling configuration
    parser.add_argument("--warmup-iters", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--profile-iters", type=int, default=10, help="Number of profiling iterations")
    parser.add_argument(
        "--nsys-start", type=int, default=None, help="Iteration to start nsys profiling (default: after warmup)"
    )
    parser.add_argument(
        "--nsys-end", type=int, default=None, help="Iteration to end nsys profiling (default: last iteration)"
    )
    parser.add_argument("--no-nsys", action="store_true", help="Disable nsys profiling API calls")

    # Backend configuration (only used for custom layers)
    parser.add_argument("--backend-linear", type=str, default="torch", choices=["torch", "te"])
    parser.add_argument(
        "--backend-attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "te", "flex"],
        help="Attention backend: sdpa (default, most compatible), te (TransformerEngine), flex",
    )
    parser.add_argument("--backend-rms-norm", type=str, default="torch", choices=["torch", "te"])

    # Layer source configuration
    parser.add_argument(
        "--use-hf",
        action="store_true",
        help="Use stock HuggingFace layers from AutoModel.from_config instead of custom NeMo Automodel layers",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available, profiling will be limited")

    # Parse dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    # Load config
    logger.info(f"Loading config from {args.model_id}")
    config = AutoConfig.from_pretrained(args.model_id)
    logger.info(f"Config: hidden_size={config.hidden_size}, n_routed_experts={config.n_routed_experts}")

    # Determine layer source
    use_hf = args.use_hf
    layer_source = "HuggingFace" if use_hf else "NeMo Automodel"
    logger.info(f"Layer source: {layer_source}")

    # Create layer based on source
    rope_fusion = False  # Only used for custom layers
    if use_hf:
        logger.info(f"Creating HuggingFace {args.layer} layer (layer_idx={args.layer_idx})")
        layer = create_hf_layer(args.layer, config, layer_idx=args.layer_idx, dtype=dtype)
        # Materialize from meta device
        layer = layer.to_empty(device=device)
        # Initialize weights randomly
        for param in layer.parameters():
            if param.requires_grad:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
    else:
        # Setup backend for custom layers
        backend = BackendConfig(enable_deepep=False)
        backend.linear = args.backend_linear
        backend.attn = args.backend_attn
        backend.rms_norm = args.backend_rms_norm
        rope_fusion = backend.rope_fusion
        logger.info(
            f"Backend: linear={backend.linear}, attn={backend.attn}, rms_norm={backend.rms_norm}, rope_fusion={rope_fusion}"
        )

        logger.info(f"Creating custom {args.layer} layer (layer_idx={args.layer_idx})")
        layer = create_layer(args.layer, config, backend, layer_idx=args.layer_idx, dtype=dtype)
        # Check if layer is on meta device (e.g., TE RMSNorm) and needs to_empty
        if any(p.device.type == "meta" for p in layer.parameters()):
            layer = layer.to_empty(device=device)
            # Initialize weights for layers created on meta device
            for param in layer.parameters():
                if param.requires_grad:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
        else:
            layer = layer.to(device)

    layer.train()

    # Count parameters
    total_params = sum(p.numel() for p in layer.parameters())
    trainable_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create inputs
    logger.info(f"Creating inputs: batch_size={args.batch_size}, seq_len={args.seq_len}")
    inputs = create_inputs(
        args.layer, config, args.batch_size, args.seq_len, device, dtype, use_hf=use_hf, rope_fusion=rope_fusion
    )

    # Set default nsys start/end
    nsys_start = args.nsys_start if args.nsys_start is not None else args.warmup_iters
    nsys_end = args.nsys_end if args.nsys_end is not None else args.warmup_iters + args.profile_iters - 1

    total_iters = args.warmup_iters + args.profile_iters
    use_nvtx = not args.no_nsys

    logger.info(f"Running {total_iters} iterations ({args.warmup_iters} warmup + {args.profile_iters} profile)")
    logger.info(f"nsys profiling: start={nsys_start}, end={nsys_end}, nvtx={use_nvtx}")

    # Warmup CUDA
    torch.cuda.synchronize()

    # Timing storage
    iter_times = []

    # Main loop
    for i in range(total_iters):
        is_warmup = i < args.warmup_iters

        # Start nsys profiling
        if not args.no_nsys and i == nsys_start:
            logger.info(f"Starting nsys profiling at iteration {i}")
            torch.cuda.cudart().cudaProfilerStart()
            torch.autograd.profiler.emit_nvtx(record_shapes=True).__enter__()

        # Zero gradients
        for param in layer.parameters():
            if param.grad is not None:
                param.grad.zero_()

        # Recreate input with fresh gradients
        inputs["x"] = torch.randn_like(inputs["x"], requires_grad=True)

        # Create CUDA events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        # Mark iteration in NVTX
        if use_nvtx:
            phase = "warmup" if is_warmup else "profile"
            torch.cuda.nvtx.range_push(f"iteration_{i}_{phase}")

        # Record start time
        start_event.record()

        # Run forward/backward
        loss = run_forward_backward(layer, inputs, args.layer, i, use_nvtx=use_nvtx, use_hf=use_hf)

        # Record end time
        end_event.record()

        # Pop NVTX range before synchronize
        if use_nvtx:
            torch.cuda.nvtx.range_pop()

        # Synchronize and compute elapsed time
        torch.cuda.synchronize()
        iter_time = start_event.elapsed_time(end_event)  # Already in ms

        # Store timing (only for profile iterations)
        if not is_warmup:
            iter_times.append(iter_time)

        # Log
        phase_str = "[warmup]" if is_warmup else "[profile]"
        mem_allocated = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(
            f"Iter {i:3d} {phase_str}: {iter_time:.3f} ms | loss={loss.item():.4f} | mem={mem_allocated:.2f} GB"
        )

        # Stop nsys profiling
        if not args.no_nsys and i == nsys_end:
            logger.info(f"Stopping nsys profiling at iteration {i}")
            torch.cuda.cudart().cudaProfilerStop()

    # Print summary
    if iter_times:
        avg_time = sum(iter_times) / len(iter_times)
        min_time = min(iter_times)
        max_time = max(iter_times)

        logger.info("=" * 60)
        logger.info("Profiling Summary")
        logger.info("=" * 60)
        logger.info(f"Layer type: {args.layer}")
        logger.info(f"Layer source: {layer_source}")
        logger.info(f"Model: {args.model_id}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Sequence length: {args.seq_len}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Profile iterations: {len(iter_times)}")
        logger.info(f"Average iteration time: {avg_time:.3f} ms")
        logger.info(f"Min iteration time: {min_time:.3f} ms")
        logger.info(f"Max iteration time: {max_time:.3f} ms")
        logger.info(f"Peak memory: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
