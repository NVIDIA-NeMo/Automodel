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

"""Test that a checkpoint can be loaded by both HF and vLLM and produces identical greedy outputs."""

import sys

import pytest
import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain quantum computing in simple terms:",
]


def _extract_custom_args(argv):
    custom_keys = {"--model_path", "--tokenizer", "--max_new_tokens"}
    custom = {}
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] in custom_keys:
            custom[argv[i].lstrip("-")] = argv[i + 1]
            i += 2
        else:
            remaining.append(argv[i])
            i += 1
    return custom, remaining


# Extract custom args at module level before pytest processes them.
_custom_args, _remaining_argv = _extract_custom_args(sys.argv)
sys.argv = _remaining_argv


def test_vllm_greedy_matches_hf():
    """Load a checkpoint with HF and vLLM, then verify greedy outputs match token-for-token."""
    pytest.importorskip("vllm")

    model_path = _custom_args["model_path"]
    tokenizer_path = _custom_args.get("tokenizer", model_path)
    max_new_tokens = int(_custom_args.get("max_new_tokens", "20"))

    # HF greedy decoding
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HF] Loading model from {model_path} on {device}")
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)
    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    hf_outputs = []
    for idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = hf_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        hf_outputs.append(generated.tolist())
        print(f"[HF] Prompt {idx}: generated {len(generated)} tokens")

    del hf_model
    torch.cuda.empty_cache()

    # vLLM greedy decoding
    from vllm import LLM, SamplingParams

    print(f"[vLLM] Loading model from {model_path}")
    llm = LLM(model=model_path, model_impl="transformers")
    sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    vllm_results = llm.generate(PROMPTS, sampling_params)

    vllm_outputs = []
    for idx, result in enumerate(vllm_results):
        tokens = list(result.outputs[0].token_ids)
        vllm_outputs.append(tokens)
        print(f"[vLLM] Prompt {idx}: generated {len(tokens)} tokens")

    # Token-for-token comparison
    for i, prompt in enumerate(PROMPTS):
        hf_tokens = hf_outputs[i]
        vllm_tokens = vllm_outputs[i]
        min_len = min(len(hf_tokens), len(vllm_tokens))
        assert hf_tokens[:min_len] == vllm_tokens[:min_len], (
            f"Token mismatch for prompt {i}: {prompt!r}\nHF:   {hf_tokens[:20]}...\nvLLM: {vllm_tokens[:20]}..."
        )
        print(f"Prompt {i}: PASS ({min_len} tokens match)")
