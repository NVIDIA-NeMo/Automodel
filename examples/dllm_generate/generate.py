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

"""Inference script for Automodel dLLM checkpoints.

Provides ``DLLMSampler`` (core logic) with preset subclasses:

- ``LLaDASampler``: no-cache, full-forward defaults.
- ``NemotronLabsDLLMSampler``: KV-cache block-diffusion defaults.

Usage
-----
LLaDA generation::

    python examples/dllm_generate/generate.py \
        --checkpoint <path> \
        --prompt "Explain what a neural network is." \
        --sampler llada

Nemotron-Labs-Diffusion generation::

    python examples/dllm_generate/generate.py \
        --checkpoint <path> \
        --prompt "What is 2+2?" \
        --sampler nemotron

Override preset defaults::

    python examples/dllm_generate/generate.py \
        --checkpoint <path> \
        --sampler nemotron --temperature 0.5 --steps 2048

Infilling (any sampler)::

    python examples/dllm_generate/generate.py \
        --checkpoint <path> \
        --prompt "The capital of France is [MASK] and it is known for [MASK]." \
        --infill

Checkpoint path resolution
--------------------------
The ``--checkpoint`` flag accepts flexible paths:

- ``.../consolidated`` (direct HF-format dir)
- ``.../model`` (finds ``consolidated/`` inside)
- ``.../LATEST`` (finds ``model/consolidated/`` inside)
- ``.../epoch_0_step_312/model/consolidated`` (intermediate steps)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from typing import Optional

import torch
from utils import (
    get_num_transfer_tokens,
    get_transfer_index,
    load_model_and_tokenizer,
    resolve_checkpoint,
    trim_response,
)

# ---------------------------------------------------------------------------
# Sampler config
# ---------------------------------------------------------------------------


@dataclass
class SamplerConfig:
    """Configuration for dLLM generation."""

    steps: int = 128
    max_new_tokens: int = 128
    block_size: int = 32
    temperature: float = 0.0
    remasking: str = "low_confidence"
    use_kv_cache: bool = False
    threshold: Optional[float] = None
    causal_context: bool = False
    eos_token_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Core sampler
# ---------------------------------------------------------------------------


class DLLMSampler:
    """Core dLLM sampler supporting both no-cache and KV-cache generation.

    Subclass and set :attr:`default_config` to create model-specific presets.
    Users can override any field at construction or at sample time.
    """

    default_config: SamplerConfig = SamplerConfig()

    def __init__(self, model, mask_id: int, eos_id: int, **overrides):
        self.model = model
        self.mask_id = mask_id
        self.eos_id = eos_id
        self.device = next(model.parameters()).device
        if overrides:
            self.default_config = replace(self.default_config, **overrides)

    def _set_diffusion_lm(self, enabled: bool):
        """Toggle the ``diffusion_lm`` flag on attention layers.

        Only meaningful for models whose attention modules expose this flag
        (e.g. Nemotron-Labs-Diffusion's ``NemotronLabsDiffusionModel``).
        """
        m = self.model.module if hasattr(self.model, "module") else self.model
        if not hasattr(m, "encoder"):
            return
        for layer in m.encoder.layers:
            if hasattr(layer.self_attn, "diffusion_lm"):
                layer.self_attn.diffusion_lm = enabled

    @torch.no_grad()
    def sample(
        self,
        inputs,
        config: SamplerConfig | None = None,
        **overrides,
    ) -> torch.Tensor:
        """Generate text via iterative block-wise denoising.

        Args:
            inputs: List of prompt token tensors or lists.
            config: Full config. If ``None``, uses :attr:`default_config`.
            **overrides: Override individual fields on the config.

        Returns:
            Token tensor of shape ``[B, prompt_len + gen_len]``.
        """
        cfg = config or self.default_config
        if overrides:
            cfg = replace(cfg, **overrides)

        use_kv_cache = cfg.use_kv_cache
        block_size = cfg.block_size

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.device) for p in inputs]
        prompt_lens = [p.shape[0] for p in inputs]
        max_prompt_len = max(prompt_lens)
        B = len(inputs)

        if use_kv_cache:
            gen_length = (cfg.max_new_tokens // block_size) * block_size or block_size
            num_blocks = gen_length // block_size
            steps = (cfg.steps // num_blocks) * num_blocks or num_blocks
        else:
            gen_length = cfg.max_new_tokens
            num_blocks = math.ceil(gen_length / block_size)
            steps = cfg.steps
        steps_per_block = steps // num_blocks if use_kv_cache else math.ceil(steps / num_blocks)

        T = max_prompt_len + gen_length
        x = torch.full((B, T), self.eos_id, dtype=torch.long, device=self.device)
        for i, p in enumerate(inputs):
            x[i, : prompt_lens[i]] = p
            x[i, prompt_lens[i] : prompt_lens[i] + gen_length] = self.mask_id

        attention_mask = None
        if not use_kv_cache:
            attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.device)
            for i, pl in enumerate(prompt_lens):
                attention_mask[i, : min(pl + gen_length, T)] = 1

        past_key_values = None
        if use_kv_cache:
            if cfg.causal_context:
                self._set_diffusion_lm(False)
            output = self.model(x[:, :max_prompt_len], use_cache=True, use_causal_mask=cfg.causal_context)
            past_key_values = output.past_key_values
            if cfg.causal_context:
                self._set_diffusion_lm(True)

        for b in range(num_blocks):
            block_start = max_prompt_len + b * block_size
            block_end = min(block_start + block_size, T)
            block_slice = slice(block_start, block_end)
            actual_block_len = block_end - block_start

            block_mask = torch.zeros((B, actual_block_len), dtype=torch.bool, device=self.device)
            for j in range(B):
                s = prompt_lens[j] + b * block_size
                e = min(s + block_size, prompt_lens[j] + gen_length, T)
                if s < e:
                    off = max(s - block_start, 0)
                    w = min(e, block_end) - max(s, block_start)
                    if w > 0:
                        block_mask[j, off : off + w] = (
                            x[j, max(s, block_start) : max(s, block_start) + w] == self.mask_id
                        )

            num_transfer_tokens = get_num_transfer_tokens(block_mask, steps_per_block)

            for i in range(num_transfer_tokens.size(1)):
                if use_kv_cache:
                    mask_idx = x[:, block_slice] == self.mask_id
                    if mask_idx.sum() == 0:
                        break
                    logits = self.model(
                        x[:, block_slice],
                        past_key_values=past_key_values,
                        use_cache=False,
                    ).logits
                    x0, transfer_idx = get_transfer_index(
                        logits,
                        cfg.temperature,
                        cfg.remasking,
                        mask_idx,
                        x[:, block_slice],
                        num_transfer_tokens=num_transfer_tokens[:, i],
                        threshold=cfg.threshold,
                    )
                    cur = x[:, block_slice].clone()
                    cur[transfer_idx] = x0[transfer_idx]
                    x[:, block_slice] = cur
                else:
                    mask_idx = x == self.mask_id
                    logits = self.model(x, attention_mask=attention_mask).logits
                    x0, transfer_idx = get_transfer_index(
                        logits,
                        cfg.temperature,
                        cfg.remasking,
                        mask_idx,
                        x,
                        num_transfer_tokens=num_transfer_tokens[:, i],
                        threshold=cfg.threshold,
                    )
                    for j in range(B):
                        transfer_idx[j, :block_start] = False
                        transfer_idx[j, block_end:] = False
                    x[transfer_idx] = x0[transfer_idx]

                if cfg.eos_token_id is not None:
                    block_tokens = x[:, block_slice]
                    eos_mask = block_tokens == cfg.eos_token_id
                    any_eos = eos_mask.any(dim=1)
                    if any_eos.any():
                        after_eos = eos_mask.cumsum(dim=1).bool()
                        mask_before = (block_tokens == self.mask_id) & ~after_eos
                        if (any_eos & ~mask_before.any(dim=1)).any():
                            break

            if use_kv_cache:
                if cfg.causal_context:
                    self._set_diffusion_lm(False)
                output = self.model(
                    x[:, block_slice],
                    past_key_values=past_key_values,
                    use_cache=True,
                    use_causal_mask=cfg.causal_context,
                )
                past_key_values = output.past_key_values
                if cfg.causal_context:
                    self._set_diffusion_lm(True)

            if cfg.eos_token_id is not None:
                gen_so_far = x[:, max_prompt_len:]
                is_eos = gen_so_far == cfg.eos_token_id
                has_eos = is_eos.any(dim=1)
                if has_eos.all():
                    first_eos_pos = is_eos.to(torch.int64).argmax(dim=1)
                    max_eos = first_eos_pos.max().item()
                    return x[:, : max_prompt_len + max_eos + 1]

        return x

    @torch.no_grad()
    def infill(
        self,
        inputs,
        config: SamplerConfig | None = None,
        **overrides,
    ) -> torch.Tensor:
        """Fill ``[MASK]`` tokens in-place via full-forward denoising."""
        cfg = config or self.default_config
        if overrides:
            cfg = replace(cfg, **overrides)

        if isinstance(inputs[0], list):
            inputs = [torch.as_tensor(p, dtype=torch.long, device=self.device) for p in inputs]

        B = len(inputs)
        seq_lens = [t.shape[0] for t in inputs]
        T = max(seq_lens)
        block_size = cfg.block_size or T

        x = torch.full((B, T), self.eos_id, dtype=torch.long, device=self.device)
        for i, t in enumerate(inputs):
            x[i, : seq_lens[i]] = t
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=self.device)
        for i, L in enumerate(seq_lens):
            if L > 0:
                attention_mask[i, :L] = 1

        num_blocks = math.ceil(T / block_size)
        steps_per_block = math.ceil(cfg.steps / num_blocks)

        for b in range(num_blocks):
            start = b * block_size
            stop = min(start + block_size, T)
            block_mask = torch.zeros((B, block_size), dtype=torch.bool, device=self.device)
            widths = []
            for j in range(B):
                width = max(0, min(seq_lens[j], stop) - start)
                widths.append(width)
                if width > 0:
                    block_mask[j, :width] = x[j, start : start + width] == self.mask_id

            transfer_schedule = get_num_transfer_tokens(block_mask, steps_per_block)
            for s in range(transfer_schedule.size(1)):
                mask_full = x == self.mask_id
                logits = self.model(x, attention_mask=attention_mask).logits
                x0, transfer_index = get_transfer_index(
                    logits,
                    cfg.temperature,
                    cfg.remasking,
                    mask_full,
                    x,
                    num_transfer_tokens=transfer_schedule[:, s],
                )
                for j in range(B):
                    transfer_index[j, :start] = False
                    transfer_index[j, start + widths[j] :] = False
                x[transfer_index] = x0[transfer_index]

        return x


# ---------------------------------------------------------------------------
# Preset subclasses
# ---------------------------------------------------------------------------


class LLaDASampler(DLLMSampler):
    """DLLMSampler with LLaDA defaults: no cache, full-forward, linear schedule."""

    default_config = SamplerConfig(
        steps=128,
        max_new_tokens=128,
        block_size=128,
        temperature=0.0,
        remasking="low_confidence",
        use_kv_cache=False,
        threshold=None,
        causal_context=False,
        eos_token_id=None,
    )


class NemotronLabsDLLMSampler(DLLMSampler):
    """DLLMSampler with Nemotron-Labs-Diffusion defaults: KV cache, causal context, threshold.

    For Nemotron-Labs-Diffusion, the CLI in ``main`` routes generation through
    the model's built-in ``model.generate(...)`` (which has the AR-seed
    mechanism), so the inherited ``sample`` method here is unused on the
    Nemotron path. This class is kept as a config-preset holder and a
    reference implementation of the standalone sampler.
    """

    default_config = SamplerConfig(
        steps=1024,
        max_new_tokens=1024,
        block_size=32,
        temperature=0.0,
        remasking="low_confidence",
        use_kv_cache=True,
        threshold=0.9,
        causal_context=True,
        eos_token_id=None,  # resolved from tokenizer at runtime
    )


SAMPLERS = {
    "llada": LLaDASampler,
    "nemotron": NemotronLabsDLLMSampler,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    """Run diffusion-language-model text generation from the CLI."""
    parser = argparse.ArgumentParser(
        description="Generate text from Automodel dLLM checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", action="append", required=True)
    parser.add_argument(
        "--sampler",
        default="llada",
        choices=list(SAMPLERS.keys()),
        help="Sampler preset (default: llada)",
    )
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--remasking", default=None, choices=["low_confidence", "random"])
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument(
        "--no_kv_cache",
        action="store_true",
        help="Disable KV cache (also disables causal context)",
    )
    parser.add_argument("--raw", action="store_true", help="No chat template")
    parser.add_argument("--infill", action="store_true", help="Infilling mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        checkpoint_path = resolve_checkpoint(args.checkpoint)
    except FileNotFoundError:
        checkpoint_path = args.checkpoint

    print(f"Loading: {checkpoint_path} (sampler={args.sampler})")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model, tokenizer, mask_id, eos_id = load_model_and_tokenizer(checkpoint_path, sampler_name=args.sampler)

    overrides = {}
    for key in ["steps", "max_new_tokens", "block_size", "temperature", "remasking", "threshold"]:
        val = getattr(args, key)
        if val is not None:
            overrides[key] = val
    # use_kv_cache and causal_context are tied: disabling KV cache also disables
    # causal context, since causal prompt encoding only works with the KV path.
    if args.no_kv_cache:
        overrides["use_kv_cache"] = False
        overrides["causal_context"] = False
    if args.sampler == "nemotron" and "eos_token_id" not in overrides:
        overrides["eos_token_id"] = eos_id

    sampler_cls = SAMPLERS[args.sampler]
    sampler = sampler_cls(model, mask_id=mask_id, eos_id=eos_id, **overrides)
    print(f"Model on {sampler.device}, mask_id={mask_id}, eos_id={eos_id}")
    print(f"Config: {sampler.default_config}")

    if args.infill:
        print(f"\n{'=' * 80}\n{'INFILLING MODE':^80}\n{'=' * 80}")
        messages_list = []
        for prompt in args.prompt:
            parts = prompt.split("[MASK]")
            content = (tokenizer.mask_token * 20).join(parts)
            messages_list.append([{"role": "user", "content": content}])
        encoded = tokenizer.apply_chat_template(
            messages_list,
            add_generation_prompt=False,
            tokenize=True,
            return_tensors=None,
        )
        outputs = sampler.infill(encoded["input_ids"])
        for i, prompt in enumerate(args.prompt):
            print(f"\n{'─' * 80}\n[Prompt {i}] {prompt}\n{'─' * 80}")
            print(f"[Filled] {tokenizer.decode(outputs[i], skip_special_tokens=True)}")
    else:
        gen_mode = "RAW" if args.raw else "CHAT"
        print(f"\n{'=' * 80}\n{f'{gen_mode} GENERATION ({args.sampler})':^80}\n{'=' * 80}")
        if args.raw:
            inputs = [tokenizer.encode(p, add_special_tokens=True) for p in args.prompt]
        else:
            messages_list = [[{"role": "user", "content": p}] for p in args.prompt]
            encoded = tokenizer.apply_chat_template(
                messages_list,
                add_generation_prompt=True,
                tokenize=True,
                return_tensors=None,
            )
            inputs = encoded["input_ids"]

        if args.sampler == "nemotron":
            # Use the model's built-in block-diffusion generate (with the
            # AR-seed mechanism: each block's first token is sampled from the
            # causal-forward's last logit, then diffusion fills the rest).
            # This matches the upstream usage snippet for Nemotron-Labs-Diffusion
            # and produces noticeably better outputs than the standalone
            # ``DLLMSampler.sample`` reimplementation.
            device = next(model.parameters()).device
            cfg = sampler.default_config
            # The model's built-in ``generate`` asserts ``max_new_tokens % block_length == 0``.
            # Round down to a multiple of block_size to match the forgiving
            # behavior of the standalone sampler.
            gen_length = (cfg.max_new_tokens // cfg.block_size) * cfg.block_size or cfg.block_size
            sequences = []
            for prompt_ids in inputs:
                prompt_tensor = torch.as_tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)
                with torch.no_grad():
                    out_ids, _nfe = model.generate(
                        prompt_tensor,
                        max_new_tokens=gen_length,
                        block_length=cfg.block_size,
                        threshold=cfg.threshold,
                        temperature=cfg.temperature,
                        causal_context=cfg.causal_context,
                        eos_token_id=cfg.eos_token_id,
                    )
                generated = out_ids[0, prompt_tensor.shape[1] :]
                sequences.append(tokenizer.decode(generated, skip_special_tokens=True))
        else:
            # LLaDA path: LLaDA checkpoints don't ship a built-in ``generate``
            # method, so fall back to the standalone ``DLLMSampler`` here.
            outputs = sampler.sample(inputs)
            sequences = trim_response(tokenizer, outputs.tolist(), inputs)
        for i, (prompt, response) in enumerate(zip(args.prompt, sequences)):
            print(f"\n{'─' * 80}\n[Prompt {i}] {prompt}\n{'─' * 80}")
            print(response.strip() or "<empty>")

    print(f"\n{'=' * 80}")


if __name__ == "__main__":
    main()
