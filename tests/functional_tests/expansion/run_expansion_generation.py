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

"""Generation from an expanded model, on a real checkpoint and a real GPU.

Three properties, in the order they matter.

Decoding from a KV cache must be refused. An expanded layer runs twice, and stream B's
keys and values differ from stream A's once the expansion weights learn, so one cache
cannot serve both. Before this was caught, ``generate()`` -- whose ``use_cache`` defaults
to true -- returned well-formed tokens computed against the wrong history, with no error.

Decoding without a cache must be correct, not merely non-crashing. It is checked against a
decode loop written out by hand, one full forward per step, because that is the reference
the recommendation in the refusal message implicitly promises.

At initialization the expanded model must generate exactly what its parent generates.

Every check perturbs the expansion weights first, except the last one, which is about the
unperturbed state. At their initial values the expanded model *is* its parent, so any
generation check passes no matter how broken stream B is.

Usage:
    python tests/functional_tests/expansion/run_expansion_generation.py \\
        --model /path/to/Llama-3.2-1B-Instruct

Single GPU; generation here is not distributed.
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Importable from a source checkout with the package uninstalled, the way the pytest
# conftest puts the repo root on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from nemo_automodel.components.expansion import (  # noqa: E402
    ExpansionConfig,
    apply_expansion,
    expansion_parameters,
)

PROMPT = "The capital of France is"
STEPS = 12


def build(model_path: str, layers: list[int] | None, perturb: float, dtype: torch.dtype) -> torch.nn.Module:
    """Load the pretrained model, optionally expand it and move its expansion weights.

    Args:
        model_path: Local path or hub id of the pretrained causal LM.
        layers: Decoder-layer indices to expand, or ``None`` to leave it unexpanded.
        perturb: Standard deviation of noise added to every expansion weight. Non-zero
            makes stream B observable; at zero the output projections discard it.
        dtype: Parameter dtype.

    Returns:
        The model, in eval mode on the current CUDA device.
    """
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).cuda().eval()
    if layers is not None:
        apply_expansion(model, ExpansionConfig(enabled=True, layers=layers))
    if perturb:
        generator = torch.Generator(device="cuda").manual_seed(3)
        with torch.no_grad():
            for _, param in expansion_parameters(model):
                param.add_(torch.randn(param.shape, generator=generator, device="cuda") * perturb)
    return model


def greedy_without_cache(model: torch.nn.Module, input_ids: torch.Tensor, steps: int) -> torch.Tensor:
    """Decode greedily with a full forward per step and no cache anywhere.

    Args:
        model: A causal LM.
        input_ids: Token ids of shape ``[batch, sequence]``.
        steps: How many tokens to append.

    Returns:
        Token ids of shape ``[batch, sequence + steps]``.
    """
    for _ in range(steps):
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        input_ids = torch.cat([input_ids, logits[:, -1:].argmax(-1)], dim=1)
    return input_ids


def main() -> None:
    """Run the three generation checks and print a decoded sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="local path or hub id of the pretrained model")
    parser.add_argument("--layers", type=int, nargs="+", default=[8, 12])
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids.cuda()

    print(f"model={args.model} layers={args.layers} dtype={args.dtype}")
    print(f"prompt: {PROMPT!r}\n")

    expanded = build(args.model, args.layers, perturb=0.02, dtype=dtype)

    print("[1] decoding from a KV cache is refused")
    try:
        with torch.no_grad():
            expanded.generate(input_ids, max_new_tokens=STEPS, do_sample=False, use_cache=True)
    except NotImplementedError as error:
        print(f"  refused, as it must be: {str(error)[:110]}...")
    else:
        raise AssertionError("cached generation was not refused; it returns tokens computed on the wrong history")

    print("\n[2] decoding without a cache matches a hand-rolled reference")
    reference = greedy_without_cache(expanded, input_ids, STEPS)
    with torch.no_grad():
        generated = expanded.generate(input_ids, max_new_tokens=STEPS, do_sample=False, use_cache=False)
    assert torch.equal(generated, reference), (
        f"generate() and the reference disagree:\n  {generated.tolist()}\n  {reference.tolist()}"
    )
    print(f"  match; expanded model says: {tokenizer.decode(generated[0], skip_special_tokens=True)!r}")

    print("\n[3] at initialization the expanded model generates exactly like its parent")
    with torch.no_grad():
        parent_out = build(args.model, None, 0.0, dtype).generate(
            input_ids, max_new_tokens=STEPS, do_sample=False, use_cache=False
        )
        fresh_out = build(args.model, args.layers, 0.0, dtype).generate(
            input_ids, max_new_tokens=STEPS, do_sample=False, use_cache=False
        )
    assert torch.equal(fresh_out, parent_out), "an unperturbed expanded model diverged from its parent"
    print(f"  match; parent says: {tokenizer.decode(parent_out[0], skip_special_tokens=True)!r}")

    print("\n[generation] OK")


if __name__ == "__main__":
    main()
