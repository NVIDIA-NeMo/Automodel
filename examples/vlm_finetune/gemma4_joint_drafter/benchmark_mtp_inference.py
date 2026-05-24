# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""MTP-vs-baseline inference benchmark for joint-finetuned Gemma 4 4B + drafter.

This script measures the inference speed-up obtained by enabling the Gemma 4
multi-token-prediction (MTP) candidate generator (HuggingFace
``SinglePositionMultiTokenCandidateGenerator``) on a base + drafter pair
produced by NeMo AutoModel's joint fine-tuning recipe
(``examples/vlm_finetune/gemma4/gemma4_4b_joint_drafter_text_mix.yaml``).

It runs the same set of MT-Bench-style single-turn prompts twice on the same
GPU:

1. ``without MTP`` -- standard greedy decoding on the base model alone.
2. ``with MTP``    -- greedy decoding on the base model with the drafter passed
   as ``assistant_model``. Transformers automatically picks up the
   ``SinglePositionMultiTokenCandidateGenerator`` when the assistant class
   name starts with ``Gemma4Assistant``
   (see ``transformers/generation/utils.py::_get_candidate_generator``).

Per-prompt and aggregate stats are printed:

* wall-clock decode time
* generated tokens
* tokens/sec
* number of *target-model* forward calls (instrumented via a forward hook)
* average accepted tokens per target-model step (= tokens / target-forwards)

Default checkpoint roots point at ``epoch_0_step_999`` (the last fully-saved
joint checkpoint; ``step_1249`` lost its drafter to a mid-save crash).

Usage:

.. code-block:: bash

    python examples/vlm_finetune/gemma4/benchmark_mtp_inference.py \
        --base-ckpt  vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v6/epoch_0_step_999/base/model/consolidated \
        --drafter-ckpt vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v6/epoch_0_step_999/drafter/model/consolidated \
        --max-new-tokens 256 \
        --num-assistant-tokens 4

Requires:
    * transformers >= 5.8.0.dev (Gemma 4 assistant + MTP candidate generator).
    * Single GPU with bf16 support (assisted decoding is hard-coded to
      ``batch_size == 1`` in ``transformers``).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import torch


# 20-item MT-Bench-style single-turn prompt set. These are paraphrased prompts
# in the same spirit as the MT-Bench v1 single-turn questions and cover the
# eight MT-Bench categories (writing, roleplay, reasoning, math, coding,
# extraction, stem, humanities) so the benchmark sees a representative spread
# of decode lengths and content types.
DEFAULT_PROMPTS: list[str] = [
    # writing
    "Compose a short engaging blog post (about 200 words) about a recent trip to Hawaii, highlighting cultural experiences and must-see attractions.",
    "Draft a professional email apologizing to a client for a delayed delivery, while reassuring them that the issue is being resolved.",
    "Write a vivid description of a busy market scene in Marrakesh using strong sensory detail.",
    # roleplay
    "You are a seasoned mountain guide. A novice asks how to prepare for a first multi-day hike. Reply in character with concrete, actionable advice.",
    "Pretend to be a 19th-century naturalist on an Amazon expedition writing the day's field journal entry.",
    # reasoning
    "Three switches outside a sealed room each control one of three bulbs inside the room. You may enter the room exactly once. How do you determine which switch controls which bulb? Explain your reasoning.",
    "A bat and a ball cost 1.10 dollars in total. The bat costs 1.00 dollar more than the ball. How much does the ball cost? Show your reasoning step by step.",
    # math
    "Solve for x: 2*x^2 - 5*x - 3 = 0. Show every step.",
    "Compute the definite integral of sin(x)^2 from 0 to pi. Show the algebra.",
    # coding
    "Write a Python function `is_palindrome(s)` that returns True if the string is a palindrome, ignoring case and non-alphanumeric characters. Include three test cases.",
    "Implement merge sort in Python with type hints and a short docstring.",
    "Write a SQL query that, given a table `orders(id, user_id, amount, created_at)`, returns the top 5 users by total amount spent in the last 30 days.",
    # extraction
    "Extract the names of every company mentioned in the following text and return them as a JSON list: 'In Q4, Apple, Microsoft, and a small startup called Lumina partnered with Nvidia on a new AI accelerator. Google was not involved.'",
    "From this paragraph, list (a) the date, (b) the location, and (c) every numerical figure: 'On March 14, 2023, the conference in Berlin drew 4,500 attendees from 62 countries, with 318 speakers across 12 tracks.'",
    # stem
    "Explain in clear terms how mRNA vaccines work and why they were developed so quickly during the COVID-19 pandemic.",
    "Describe the difference between supervised, unsupervised, and reinforcement learning, with one concrete example of each.",
    "What is the second law of thermodynamics and why does it imply that perpetual motion machines are impossible?",
    # humanities
    "Discuss two major causes of the fall of the Western Roman Empire and weigh their relative importance.",
    "Compare and contrast utilitarianism and deontological ethics, illustrating with a single moral dilemma.",
    "Summarize the plot of Shakespeare's Hamlet in under 150 words while preserving the central themes.",
]


# ---------------------------------------------------------------------------
# transformers import + sanity check
# ---------------------------------------------------------------------------


def _import_transformers() -> Any:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover -- env error
        raise SystemExit(
            "transformers is not importable. This benchmark requires "
            "transformers>=5.8.0.dev with Gemma4Assistant support."
        ) from exc

    # Hard-require the MTP candidate generator added in 5.8.x.
    try:
        from transformers.generation.candidate_generator import (  # noqa: F401
            SinglePositionMultiTokenCandidateGenerator,
        )
    except ImportError as exc:
        raise SystemExit(
            "Installed transformers does not export "
            "`SinglePositionMultiTokenCandidateGenerator`. Update to "
            "transformers>=5.8.0.dev (the branch under transformers-tot/)."
        ) from exc

    return transformers


# ---------------------------------------------------------------------------
# Forward-call counter
# ---------------------------------------------------------------------------


class _ForwardCounter:
    """Pre-forward hook that counts the number of times ``module(...)`` runs.

    Used to count target-model forward calls during ``.generate()``. The
    baseline path triggers one forward per generated token; the MTP path
    triggers one forward per accepted *window* of up to ``num_assistant_tokens
    + 1`` tokens, so the ratio ``new_tokens / forward_calls`` is the average
    accepted run length.
    """

    def __init__(self) -> None:
        self.count = 0

    def reset(self) -> None:
        self.count = 0

    def __call__(self, _module: torch.nn.Module, _args: Any, _kwargs: Any) -> None:
        self.count += 1


# ---------------------------------------------------------------------------
# Generation runners
# ---------------------------------------------------------------------------


@dataclass
class RunStats:
    """Aggregated decode statistics for a single (mode, prompt) run."""

    prompt_idx: int
    new_tokens: int
    decode_time_s: float
    target_forwards: int
    output_text: str
    output_ids: list[int] = field(default_factory=list)

    @property
    def tokens_per_sec(self) -> float:
        return self.new_tokens / self.decode_time_s if self.decode_time_s > 0 else 0.0

    @property
    def accepted_tokens_per_step(self) -> float:
        return self.new_tokens / self.target_forwards if self.target_forwards > 0 else 0.0


def _format_prompt(tokenizer: Any, prompt_text: str) -> torch.Tensor:
    """Apply the model's chat template to ``prompt_text``.

    Falls back to a raw tokenization when no chat template is registered.
    """
    messages = [{"role": "user", "content": prompt_text}]
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template may return a LongTensor or a BatchEncoding.
        return ids if isinstance(ids, torch.Tensor) else ids["input_ids"]
    return tokenizer(prompt_text, return_tensors="pt").input_ids


@torch.no_grad()
def _generate(
    base: torch.nn.Module,
    tokenizer: Any,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    assistant_model: torch.nn.Module | None,
    num_assistant_tokens: int,
    target_counter: _ForwardCounter,
) -> tuple[torch.Tensor, float, int]:
    """Run a single ``model.generate`` call and return ``(output_ids, elapsed, forwards)``.

    ``output_ids`` excludes the prompt. ``elapsed`` is GPU-synchronized wall time.
    ``forwards`` is the number of base-model forward calls observed via the hook.
    """
    if assistant_model is not None:
        # transformers reads `num_assistant_tokens` from the assistant's
        # generation_config; bump it (default is 20) before each call.
        assistant_model.generation_config.num_assistant_tokens = num_assistant_tokens
        assistant_model.generation_config.num_assistant_tokens_schedule = "constant"

    target_counter.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    generate_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if assistant_model is not None:
        generate_kwargs["assistant_model"] = assistant_model

    out = base.generate(input_ids=input_ids.to(base.device), **generate_kwargs)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    prompt_len = input_ids.shape[1]
    new_ids = out[:, prompt_len:]
    return new_ids, elapsed, target_counter.count


def _run_one_prompt(
    *,
    prompt_idx: int,
    prompt_text: str,
    base: torch.nn.Module,
    drafter: torch.nn.Module | None,
    tokenizer: Any,
    max_new_tokens: int,
    num_assistant_tokens: int,
    target_counter: _ForwardCounter,
    label: str,
) -> RunStats:
    """Run a single prompt and return its ``RunStats``."""
    input_ids = _format_prompt(tokenizer, prompt_text)
    new_ids, elapsed, forwards = _generate(
        base,
        tokenizer,
        input_ids,
        max_new_tokens=max_new_tokens,
        assistant_model=drafter,
        num_assistant_tokens=num_assistant_tokens,
        target_counter=target_counter,
    )
    new_ids_list = new_ids[0].tolist()
    text = tokenizer.decode(new_ids_list, skip_special_tokens=True)
    stats = RunStats(
        prompt_idx=prompt_idx,
        new_tokens=len(new_ids_list),
        decode_time_s=elapsed,
        target_forwards=forwards,
        output_text=text,
        output_ids=new_ids_list,
    )
    print(
        f"  [{label}] prompt {prompt_idx:2d}: "
        f"{stats.new_tokens:4d} tok / {stats.decode_time_s:6.2f}s = "
        f"{stats.tokens_per_sec:6.1f} tok/s  "
        f"(forwards={stats.target_forwards:4d}, "
        f"accept/step={stats.accepted_tokens_per_step:4.2f})",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _aggregate(runs: list[RunStats]) -> dict[str, float]:
    total_new = sum(r.new_tokens for r in runs)
    total_time = sum(r.decode_time_s for r in runs)
    total_forwards = sum(r.target_forwards for r in runs)
    return {
        "n_prompts": len(runs),
        "total_new_tokens": total_new,
        "total_decode_time_s": total_time,
        "aggregate_tokens_per_sec": total_new / total_time if total_time > 0 else 0.0,
        "total_target_forwards": total_forwards,
        "mean_accepted_per_step": (total_new / total_forwards) if total_forwards > 0 else 0.0,
        "mean_tokens_per_sec": sum(r.tokens_per_sec for r in runs) / max(1, len(runs)),
    }


def _print_summary(no_mtp: dict[str, float], mtp: dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("Summary".center(78))
    print("=" * 78)
    header = f"{'metric':<32} {'no MTP':>20} {'with MTP':>20}"
    print(header)
    print("-" * 78)

    def _row(name: str, key: str, fmt: str = "{:.2f}") -> None:
        a, b = no_mtp.get(key, 0.0), mtp.get(key, 0.0)
        print(f"{name:<32} {fmt.format(a):>20} {fmt.format(b):>20}")

    _row("# prompts", "n_prompts", "{:.0f}")
    _row("total new tokens", "total_new_tokens", "{:.0f}")
    _row("total decode time (s)", "total_decode_time_s")
    _row("aggregate tokens/sec", "aggregate_tokens_per_sec")
    _row("mean per-prompt tokens/sec", "mean_tokens_per_sec")
    _row("total target forwards", "total_target_forwards", "{:.0f}")
    _row("mean accepted tokens/step", "mean_accepted_per_step", "{:.3f}")

    if no_mtp.get("aggregate_tokens_per_sec", 0.0) > 0:
        speedup = mtp.get("aggregate_tokens_per_sec", 0.0) / no_mtp["aggregate_tokens_per_sec"]
        print("-" * 78)
        print(f"{'wall-clock speed-up (MTP / no MTP)':<32} {speedup:>41.2f}x")
    print("=" * 78)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_DEFAULT_BASE = (
    # "vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v6/epoch_0_step_999/base/model/consolidated"
    "vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v9_gbs64_val50/epoch_0_step_499/base/model/consolidated"
)
_DEFAULT_DRAFTER = (
    # "vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v6/epoch_0_step_999/drafter/model/consolidated"
    "vlm_checkpoints/gemma4_4b_joint_drafter_text_mix_v9_gbs64_val50/epoch_0_step_499/drafter/model/consolidated"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-ckpt", default=_DEFAULT_BASE, help="HF consolidated dir for base.")
    p.add_argument("--drafter-ckpt", default=_DEFAULT_DRAFTER, help="HF consolidated dir for drafter.")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--num-assistant-tokens",
        type=int,
        default=4,
        help="K -- number of draft tokens per assisted step (MTP only).",
    )
    p.add_argument(
        "--num-prompts",
        type=int,
        default=len(DEFAULT_PROMPTS),
        help="How many prompts from the built-in MT-Bench-style set to run.",
    )
    p.add_argument(
        "--mode",
        choices=("both", "no_mtp", "mtp"),
        default="both",
        help="Which decode paths to time (default: both, then compare).",
    )
    p.add_argument(
        "--attn",
        choices=("sdpa", "eager", "flash_attention_2"),
        default="eager",
        help="Attention implementation. Eager is safest; sdpa is fastest on stock H100.",
    )
    p.add_argument(
        "--save-json",
        default=None,
        help="Optional path to dump full per-prompt results as JSON.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required.")

    _import_transformers()
    from transformers import AutoTokenizer, Gemma4AssistantForCausalLM, Gemma4ForConditionalGeneration

    torch.manual_seed(args.seed)

    for path, name in ((args.base_ckpt, "base"), (args.drafter_ckpt, "drafter")):
        if not os.path.isdir(path):
            raise SystemExit(f"{name} checkpoint dir not found: {path}")
        if not os.path.exists(os.path.join(path, "config.json")):
            raise SystemExit(f"{name} dir does not look like an HF consolidated dump (missing config.json): {path}")

    print(f"Loading base from:    {args.base_ckpt}", flush=True)
    base = (
        Gemma4ForConditionalGeneration.from_pretrained(
            args.base_ckpt,
            dtype=torch.bfloat16,
            attn_implementation=args.attn,
        )
        .to("cuda")
        .eval()
    )

    print(f"Loading drafter from: {args.drafter_ckpt}", flush=True)
    drafter = (
        Gemma4AssistantForCausalLM.from_pretrained(
            args.drafter_ckpt,
            dtype=torch.bfloat16,
            attn_implementation=args.attn,
        )
        .to("cuda")
        .eval()
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_ckpt)

    target_counter = _ForwardCounter()
    base.register_forward_pre_hook(target_counter, with_kwargs=True)

    prompts = DEFAULT_PROMPTS[: args.num_prompts]
    print(f"\nRunning {len(prompts)} prompts, max_new_tokens={args.max_new_tokens}, K={args.num_assistant_tokens}.\n")

    # Warm-up: one short call per mode to amortize CUDA allocator + kernel
    # autotune (sdpa) costs that would otherwise be attributed to the first
    # timed run.
    print("Warm-up (excluded from stats)...", flush=True)
    warmup_ids = _format_prompt(tokenizer, "Hello, please introduce yourself in one sentence.")
    if args.mode in ("both", "no_mtp"):
        _generate(
            base,
            tokenizer,
            warmup_ids,
            max_new_tokens=16,
            assistant_model=None,
            num_assistant_tokens=args.num_assistant_tokens,
            target_counter=target_counter,
        )
    if args.mode in ("both", "mtp"):
        _generate(
            base,
            tokenizer,
            warmup_ids,
            max_new_tokens=16,
            assistant_model=drafter,
            num_assistant_tokens=args.num_assistant_tokens,
            target_counter=target_counter,
        )

    no_mtp_runs: list[RunStats] = []
    mtp_runs: list[RunStats] = []

    if args.mode in ("both", "no_mtp"):
        print("\n=== Decoding without MTP ===", flush=True)
        for i, prompt in enumerate(prompts):
            no_mtp_runs.append(
                _run_one_prompt(
                    prompt_idx=i,
                    prompt_text=prompt,
                    base=base,
                    drafter=None,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    num_assistant_tokens=args.num_assistant_tokens,
                    target_counter=target_counter,
                    label="no MTP",
                )
            )

    if args.mode in ("both", "mtp"):
        print("\n=== Decoding with MTP ===", flush=True)
        for i, prompt in enumerate(prompts):
            mtp_runs.append(
                _run_one_prompt(
                    prompt_idx=i,
                    prompt_text=prompt,
                    base=base,
                    drafter=drafter,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    num_assistant_tokens=args.num_assistant_tokens,
                    target_counter=target_counter,
                    label="MTP   ",
                )
            )

    no_mtp_agg = _aggregate(no_mtp_runs) if no_mtp_runs else {}
    mtp_agg = _aggregate(mtp_runs) if mtp_runs else {}
    _print_summary(no_mtp_agg, mtp_agg)

    if args.save_json:
        payload = {
            "args": vars(args),
            "no_mtp": {
                "aggregate": no_mtp_agg,
                "runs": [r.__dict__ for r in no_mtp_runs],
            },
            "mtp": {
                "aggregate": mtp_agg,
                "runs": [r.__dict__ for r in mtp_runs],
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.save_json)) or ".", exist_ok=True)
        with open(args.save_json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote per-prompt results to {args.save_json}.")

    del base, drafter
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
