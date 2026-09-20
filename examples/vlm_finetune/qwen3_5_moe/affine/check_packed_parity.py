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
"""Assert that a packed batch produces the same result as its documents run alone.

This is the gate for `neat` packing: if any layer lets information cross a document
boundary, a packed forward will not match the per-document forwards, and the loss will
still look plausible. The check covers both halves of the hybrid backbone -- the
GatedDeltaNet layers, whose recurrent state must reset on ``cu_seqlens``, and the full
attention layers, which must see per-document causality.

It runs one small stack of real ``Qwen3_5MoeBlock`` layers, so the real kernels are
exercised. Rotary frequencies are supplied directly and sliced per document rather than
recomputed, so a mismatch can only come from leakage, not from a rotary convention
difference between the two paths.

Usage::

    # On the GPU box, the configuration the recipe actually runs:
    python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py --attn te

    # CPU sanity run of the mask logic only. The GatedDeltaNet layers need a real
    # causal_conv1d / FLA build, so restrict the stack to full_attention here.
    TORCHDYNAMO_DISABLE=1 python examples/vlm_finetune/qwen3_5_moe/affine/check_packed_parity.py \\
        --attn sdpa --device cpu --dtype float32 --head-dim 16 \\
        --layers full_attention full_attention

``TORCHDYNAMO_DISABLE=1`` is only needed where no C++ compiler is available for the MoE
layer's inductor path; on the container it can be omitted.

Exit code is non-zero if any layer leaks.
"""

import argparse

import torch

from nemo_automodel.components.datasets.utils import _indexed_mask_to_4d_block_causal
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeBlock
from nemo_automodel.components.moe.layers import MoEConfig

try:
    from transformers.models.qwen3_5_moe import Qwen3_5MoeTextConfig
except ImportError as exc:  # pragma: no cover - transformers is a hard requirement here
    raise SystemExit(f"transformers with qwen3_5_moe support is required: {exc}")


def _build_configs(hidden: int, head_dim: int, layer_types: list[str]):
    """Build a small text config and matching MoE config.

    Args:
        hidden: Hidden size.
        head_dim: Per-head dimension. Keep 256 to match the real checkpoint, since that
            is the value that forces TE >= 2.18 + cuDNN >= 9.23 for fused attention.
        layer_types: Per-layer type strings, e.g. three ``linear_attention`` then one
            ``full_attention``.

    Returns:
        ``(text_config, moe_config)``.
    """
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=hidden,
        num_hidden_layers=len(layer_types),
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=head_dim,
        intermediate_size=2 * hidden,
        moe_intermediate_size=hidden,
        shared_expert_intermediate_size=hidden,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        rms_norm_eps=1e-6,
        router_aux_loss_coef=0.01,
        pad_token_id=0,
        layer_types=layer_types,
    )
    moe_config = MoEConfig(
        dim=hidden,
        inter_dim=hidden,
        moe_inter_dim=text_config.moe_intermediate_size,
        n_routed_experts=text_config.num_experts,
        n_shared_experts=1,
        n_activated_experts=text_config.num_experts_per_tok,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=text_config.router_aux_loss_coef,
        norm_topk_prob=True,
        expert_bias=False,
        router_bias=False,
        expert_activation="swiglu",
    )
    return text_config, moe_config


def _masks_for(document_ids: torch.Tensor, attn: str) -> tuple[torch.Tensor, dict]:
    """Build the mask the given backend expects, mirroring the packed collater.

    Args:
        document_ids: Indexed document map of shape [batch, sequence], zero for padding.
        attn: Attention backend name.

    Returns:
        ``(attention_mask, extra_kwargs)``. For TE the mask is the indexed map itself and
        the model derives ``cu_seqlens`` from it. For SDPA the mask is the dense
        ``[batch, 1, sequence, sequence]`` block-causal form and the indexed map travels
        beside it as ``_packed_seq_ids``, exactly as ``neat_packed_vlm_collater`` emits.
    """
    if attn == "te":
        return document_ids, {}
    return _indexed_mask_to_4d_block_causal(document_ids), {"_packed_seq_ids": document_ids}


def main() -> None:
    """Compare packed and per-document forwards and exit non-zero on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn", default="te", choices=["te", "sdpa"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--head-dim", type=int, default=256)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        help="Layer types to stack; the default mirrors the real [linear x3, full] cycle.",
    )
    parser.add_argument("--doc-lens", type=int, nargs="+", default=[37, 61, 19])
    parser.add_argument("--pad", type=int, default=5, help="Trailing padding tokens in the pack.")
    parser.add_argument("--atol", type=float, default=None, help="Defaults by dtype.")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    atol = args.atol if args.atol is not None else (2e-2 if dtype is torch.bfloat16 else 1e-4)

    text_config, moe_config = _build_configs(args.hidden, args.head_dim, list(args.layers))
    backend = BackendConfig(attn=args.attn, linear="torch", rms_norm="torch", rope_fusion=False)

    torch.manual_seed(0)
    blocks = torch.nn.ModuleList(
        Qwen3_5MoeBlock(idx, text_config, moe_config, backend) for idx in range(len(args.layers))
    ).to(device=device, dtype=dtype)
    # Blocks are constructed with uninitialized parameters; the real model calls this.
    # init_weights writes in place, so it must run outside autograd tracking.
    with torch.no_grad():
        for block in blocks:
            block.init_weights(device)
    blocks.eval()
    bad = [name for name, p in blocks.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise SystemExit(f"non-finite parameters after init_weights: {bad[:5]}")

    doc_lens = list(args.doc_lens)
    total = sum(doc_lens) + args.pad

    # Indexed document map and per-document position ids, as `neat` builds them.
    document_ids = torch.zeros(1, total, dtype=torch.long, device=device)
    position_ids = torch.zeros(1, total, dtype=torch.long, device=device)
    offset = 0
    bounds = []
    for index, length in enumerate(doc_lens, start=1):
        document_ids[0, offset : offset + length] = index
        position_ids[0, offset : offset + length] = torch.arange(length, device=device)
        bounds.append((offset, offset + length))
        offset += length

    hidden_states = torch.randn(1, total, args.hidden, device=device, dtype=dtype)
    # Non-fused rope consumes [batch, sequence, head_dim] holding concatenated (cos, sin),
    # each head_dim // 2 wide. Supplied once and sliced per document, so both paths see
    # identical rotary input for corresponding tokens; zeros would make rope degenerate
    # and hide position errors.
    angles = torch.randn(1, total, args.head_dim // 2, device=device, dtype=torch.float32)
    freqs_cis = torch.cat([angles.cos(), angles.sin()], dim=-1).to(dtype)

    packed_mask, packed_extra = _masks_for(document_ids, args.attn)
    packed = hidden_states
    with torch.no_grad():
        for block in blocks:
            packed = block(
                packed,
                freqs_cis=freqs_cis,
                attention_mask=packed_mask,
                # Supplied explicitly, as the real forward does. Left None, the block
                # would derive it from the mask, which is wrong for a 4-D SDPA mask.
                padding_mask=document_ids.eq(0),
                position_ids=position_ids,
                **packed_extra,
            )

    print(f"backend={args.attn} device={device} dtype={args.dtype} layers={args.layers}")
    print(f"pack: {len(doc_lens)} documents {doc_lens} + {args.pad} padding = {total} tokens, atol={atol}")

    failures = 0
    for index, (start, end) in enumerate(bounds, start=1):
        length = end - start
        solo_ids = torch.ones(1, length, dtype=torch.long, device=device)
        solo_mask, solo_extra = _masks_for(solo_ids, args.attn)
        solo = hidden_states[:, start:end]
        with torch.no_grad():
            for block in blocks:
                solo = block(
                    solo,
                    freqs_cis=freqs_cis[:, start:end],
                    attention_mask=solo_mask,
                    padding_mask=torch.zeros_like(solo_ids, dtype=torch.bool),
                    position_ids=position_ids[:, start:end],
                    **solo_extra,
                )

        delta = (packed[:, start:end].float() - solo.float()).abs().max().item()
        status = "ok" if delta <= atol else "LEAK"
        if status != "ok":
            failures += 1
        print(f"  document {index}: tokens={length:5d} max|packed - standalone| = {delta:.3e}  {status}")

    if args.pad:
        # Padding is not a document; it must not pick up content from one.
        tail = packed[:, sum(doc_lens) :]
        print(f"  padding tail: max|value| = {tail.float().abs().max().item():.3e} (informational)")

    if failures:
        raise SystemExit(f"{failures}/{len(doc_lens)} documents differ when packed - attention or state leaks")
    print(f"\nOK - all {len(doc_lens)} documents are unaffected by being packed together")


if __name__ == "__main__":
    main()
