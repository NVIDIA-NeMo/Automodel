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

"""Interleaved reference/generation forward for Wan-Animate-2 training.

The Diffusers integration exposes ``forward_ref`` and ``forward_gen`` as two
separate entry points, each walking all forty blocks. That is correct for
inference, where the reference pass runs once under ``no_grad`` and its cache is
reused across denoising steps. It does not survive training under FSDP2:

* Each block writes its keys and values by mutating a caller-owned mapping. With
  gradient enabled, the pre-forward path maps over the arguments and rebuilds
  recognised containers, so every block receives a copy. Measured: all forty
  blocks receive the correct index, yet the caller's dict holds one entry.
* Walking the blocks twice per step means FSDP2 sees two forwards per module. It
  reshards after the first, freeing weights the second pass and its backward
  still need, which surfaces as ``setStorage: ... storage of size 0``. Keeping
  the parameters gathered instead costs the full unsharded model per rank.

The reference training implementation (DiffSynth-Studio) avoids all of this by
interleaving: one call per block that runs the reference pass and the generation
pass back to back against a block-local cache. FSDP2 then sees exactly one
forward per module, and nothing crosses a wrapper boundary. This module ports
that structure, reusing the block methods the integration already provides.
"""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["forward_origin", "install_forward_origin", "supports_interleaved_forward"]


def _block_forward_origin(
    self: torch.nn.Module,
    x: torch.Tensor,
    x_ref: torch.Tensor,
    ref_args: dict[str, Any],
    gen_args: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one block's reference pass and generation pass against a local cache.

    Installed as a method on the block class so the block is entered through its
    ``__call__`` exactly once per step. Calling ``forward_ref``/``forward_gen``
    directly would skip the block's own FSDP2 hooks, leaving its parameters
    sharded and failing with "got mixed torch.Tensor and DTensor".

    Args:
        self: One transformer block.
        x: Generation-stream activations for this block.
        x_ref: Reference-stream activations for this block.
        ref_args: Keyword arguments for the block's ``forward_ref``.
        gen_args: Keyword arguments for the block's ``forward_gen``.

    Returns:
        ``(x, x_ref)`` after this block.
    """
    block = self
    # Index 0 rather than the block's position: the cache lives and dies inside
    # this call, so there is exactly one entry and nothing to disambiguate. This
    # is what makes the container-copying failure structurally impossible.
    key_cache: dict[int, torch.Tensor] = {}
    value_cache: dict[int, torch.Tensor] = {}
    x_ref = block.forward_ref(x_ref, 0, key_cache, value_cache, **ref_args)
    x = block.forward_gen(x, 0, key_cache, value_cache, **gen_args)
    return x, x_ref


def _precompile_attention_without_cudagraphs() -> None:
    """Compile the attention helper with autotuning but without CUDA graphs.

    The integration compiles its attention with ``mode="max-autotune"``, which
    also records CUDA graphs, and caches the result in a module-level global. A
    recorded region cannot be recomputed: checkpointing a block frees its
    activations after the forward and recreates them in the backward, which the
    recorder reports as "an input tensor deallocate during graph recording that
    did not occur during replay". Selective checkpointing does not help, because
    the flex-attention entry point is a higher-order op and so is not on the
    save list that covers ``aten.mm`` and friends.

    Populating the cache first with ``max-autotune-no-cudagraphs`` keeps the
    kernel autotuning and drops only the graph capture. This is scoped to this
    model's own compiled helper; it does not touch the Inductor configuration
    that every other model in the process shares.
    """
    try:
        from diffusers.models.transformers import transformer_wan_animate_2 as upstream
    except ImportError:
        # The integration is not in a released diffusers yet. Nothing to
        # pre-compile; the model load itself will report the missing support.
        return

    if getattr(upstream, "_flex_compiled", None) is not None:
        return
    upstream._flex_compiled = torch.compile(
        upstream._flex_attention_raw,
        dynamic=False,
        mode="max-autotune-no-cudagraphs",
        fullgraph=True,
    )


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    """Return the module underneath any activation-checkpoint wrappers.

    Args:
        module: A block, possibly wrapped for activation checkpointing.

    Returns:
        The innermost wrapped module.
    """
    seen = 0
    while hasattr(module, "_checkpoint_wrapped_module") and seen < 8:
        module = module._checkpoint_wrapped_module
        seen += 1
    return module


def install_forward_origin(model: torch.nn.Module) -> bool:
    """Attach :func:`forward_origin` to the transformer class if it is missing.

    The Diffusers integration ships ``forward_ref`` and ``forward_gen`` but not
    the interleaved traversal that training needs. Installing it on the class
    lets the model's own ``forward`` dispatcher reach it by name, so the call
    goes through ``__call__`` and FSDP2's hooks run.

    Args:
        model: The Wan-Animate-2 transformer.

    Returns:
        True when the interleaved forward is available after this call.
    """
    inner = getattr(model, "module", model)
    if not supports_interleaved_forward(inner):
        return False
    _precompile_attention_without_cudagraphs()
    cls = type(inner)
    if not hasattr(cls, "forward_origin"):
        cls.forward_origin = forward_origin
    # Install on the block itself, not on whatever wraps it. By the time this
    # runs the blocks may sit inside an activation-checkpoint wrapper, whose
    # forward forwards to the module underneath; installing on the wrapper's
    # class would leave the real block without the method.
    block_cls = type(_unwrap_module(inner.blocks[0]))
    if not hasattr(block_cls, "forward_origin"):
        block_cls.forward_origin = _block_forward_origin
    return True


def supports_interleaved_forward(model: torch.nn.Module) -> bool:
    """Report whether the transformer exposes what the interleaved path needs.

    Args:
        model: The Wan-Animate-2 transformer, possibly FSDP-wrapped.

    Returns:
        True when every attribute and block method this module relies on exists.
    """
    inner = getattr(model, "module", model)
    needed = (
        "blocks",
        "patch_embedding",
        "time_embedding",
        "time_projection",
        "text_embedding",
        "head",
        "unpatchify",
        "create_mask",
        "block_masks",
    )
    if not all(hasattr(inner, name) for name in needed):
        return False
    blocks = getattr(inner, "blocks", None)
    if not blocks:
        return False
    first = _unwrap_module(blocks[0])
    return hasattr(first, "forward_ref") and hasattr(first, "forward_gen")


def forward_origin(self: torch.nn.Module, inputs: dict[str, Any]) -> list[torch.Tensor]:
    """Run both streams through every block once, in a single traversal.

    Mirrors the reference implementation's ``forward_origin``. The embedding work
    of ``forward_ref`` and ``forward_gen`` is performed up front, then the blocks
    are walked once, each handling both streams.

    Installed as a method on the transformer class rather than called as a free
    function, so that invoking it through ``model(inputs, method="forward_origin")``
    passes through the module's ``__call__``. That matters under FSDP2: entering
    by any other route skips the root pre-forward hook, leaving the parameters as
    sharded DTensors, and the first convolution fails with "got mixed
    torch.Tensor and DTensor".

    Args:
        self: The Wan-Animate-2 transformer.
        inputs: The mapping produced by ``WanAnimate2Adapter.prepare_inputs``.

    Returns:
        Per-sample velocity predictions, as ``forward_gen`` returns them.
    """
    from diffusers.models.transformers.transformer_wan_animate_2 import (
        rope_params,
        sinusoidal_embedding_1d,
    )

    net = self
    device = net.patch_embedding.weight.device
    dim_head = net.dim // net.num_heads

    def _embed(latents: list[torch.Tensor], condition: list[torch.Tensor], seq_len: int):
        """Patch-embed one stream and pad it to its packed sequence length."""
        merged = [torch.cat([u, v], dim=0) for u, v in zip(latents, condition)]
        embedded = [net.patch_embedding(u.unsqueeze(0)) for u in merged]
        grid = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in embedded])
        flat = [u.flatten(2).transpose(1, 2) for u in embedded]
        padded = torch.cat([torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1) for u in flat])
        return padded, grid

    def _text(context: list[torch.Tensor], clip_features: torch.Tensor | None) -> torch.Tensor:
        """Embed the caption and prepend CLIP image tokens when the model uses them."""
        embedded = net.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(net.text_len - u.size(0), u.size(1))]) for u in context])
        )
        if net.use_img_emb and clip_features is not None:
            embedded = torch.concat([net.img_emb(clip_features), embedded], dim=1)
        return embedded

    def _rope(offset_t: int, offset_h: int, offset_w: int) -> torch.Tensor:
        """Build the RoPE frequency table for one stream's spatial offsets."""
        freqs = torch.cat(
            [
                rope_params(512, dim_head - 4 * (dim_head // 6), offset=offset_t),
                rope_params(512, 2 * (dim_head // 6), offset=offset_h),
                rope_params(512, 2 * (dim_head // 6), offset=offset_w),
            ],
            dim=1,
        )
        return freqs.to(device) if freqs.device != device else freqs

    # --- reference stream -----------------------------------------------------
    x_ref, grid_sizes_ref = _embed(inputs["x_ref"], inputs["condition_y"], inputs["seq_len_ref"])

    # The reference RoPE offsets are latched on the first forward and never
    # reset, which is why a run must stay on a single resolution bucket. Latch
    # them from the reference grid, exactly as forward_ref does when it runs
    # first in the two-pass order.
    reference_grid = inputs["grid_sizes_ref"]
    if net.refer_offset_t < 0:
        net.refer_offset_t = reference_grid[0][0].item()
    if net.refer_offset_h < 0:
        net.refer_offset_h = reference_grid[0][1].item()
    if net.refer_offset_w < 0:
        net.refer_offset_w = reference_grid[0][2].item()
    net.freqs_ref = _rope(net.refer_offset_t, net.refer_offset_h, net.refer_offset_w)

    with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
        e_ref = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, inputs["timestep"] * 0 + 1).float())
        e0_ref = net.time_projection(e_ref).unflatten(1, (6, net.dim))
    context_ref = _text(inputs["context_ref"], inputs.get("clip_fea_ref"))

    ref_args = {
        "e_ref": e0_ref,
        "grid_sizes_ref": grid_sizes_ref,
        "freqs_ref": net.freqs_ref,
        "context_ref": context_ref,
        "context_lens": None,
    }

    # --- generation stream ----------------------------------------------------
    x, grid_sizes = _embed(inputs["x"], inputs["y"], inputs["seq_len"])
    net.freqs = _rope(0, 0, 0)

    with torch.amp.autocast(device_type=device.type, dtype=torch.float32):
        e = net.time_embedding(sinusoidal_embedding_1d(net.freq_dim, inputs["timestep"]).float())
        e0 = net.time_projection(e).unflatten(1, (6, net.dim))
    context = _text(inputs["context"], inputs.get("clip_fea"))

    origin_len = inputs["origin_len"]
    origin_area = inputs["origin_area"]
    mask_id = (origin_len, origin_area[0], origin_area[1])
    if mask_id not in net.block_masks:
        net.block_masks[mask_id] = net.create_mask(origin_len, origin_area, x.device)

    gen_args = {
        "e": e0,
        "block_mask": net.block_masks[mask_id],
        "grid_sizes": grid_sizes,
        "freqs": net.freqs,
        "context": context,
        "grid_sizes_ref": grid_sizes_ref,
        "freqs_ref": net.freqs_ref,
        "context_lens": None,
        "origin_area": origin_area,
        "origin_len": origin_len,
    }

    # --- one traversal, both streams -----------------------------------------
    # Checkpoint the whole block, both passes together, as the reference
    # training implementation does. Interleaving keeps the reference stream's
    # activations live alongside the generation stream's for the backward, which
    # the frozen-reference path never paid for and which does not fit otherwise.
    # Recomputing a block is far cheaper than storing two streams of activations
    # across forty of them.
    for block in net.blocks:
        # Through __call__, with the block's own `method` dispatcher, so FSDP2
        # gathers this block's parameters before either pass runs, and so any
        # checkpoint wrapper the parallelization strategy installed around the
        # block is honoured. Checkpointing is the strategy's concern, not this
        # function's.
        x, x_ref = block(x, x_ref, ref_args, gen_args, method="forward_origin")

    x = net.head(x, e)
    x = net.unpatchify(x, grid_sizes)
    return [u.float() for u in x]
