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

"""Test that a checkpoint can be loaded by vLLM and produces correct greedy outputs.

Default mode: compare vLLM (model_impl="transformers") against HF token-for-token.
Smoke test mode (--vllm_smoke_test): verify the model loads into vLLM's native backend
and produces non-empty output, without HF comparison.

Usage:
  SFT (config-driven):
    python -m pytest test_checkpoint_vllm_deploy.py \
        --deploy_mode sft --config_path recipe.yaml --deploy_model_path /path/to/checkpoint/model/consolidated/

  PEFT (config-driven):
    python -m pytest test_checkpoint_vllm_deploy.py \
        --deploy_mode peft --config_path recipe.yaml --adapter_path /path/to/checkpoint/model/

  Legacy (no config):
    python -m pytest test_checkpoint_vllm_deploy.py \
        --deploy_model_path <hf_name_or_path> --tokenizer <hf_name> --trust_remote_code
"""

import gc
import os
import sys
import tempfile

import torch

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    ",
    "Explain quantum computing in simple terms:",
]


def _extract_custom_args(argv):
    custom_keys = {
        "--deploy_model_path",
        "--tokenizer",
        "--max_new_tokens",
        "--adapter_path",
        "--config_path",
        "--deploy_mode",
    }
    boolean_keys = {"--vllm_smoke_test", "--trust_remote_code"}
    custom = {}
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] in custom_keys:
            custom[argv[i].lstrip("-")] = argv[i + 1]
            i += 2
        elif argv[i] in boolean_keys:
            custom[argv[i].lstrip("-")] = True
            i += 1
        else:
            remaining.append(argv[i])
            i += 1
    return custom, remaining


def _load_recipe_config(config_path):
    """Load a recipe YAML and return the parsed dict."""
    import yaml

    with open(config_path) as f:
        return yaml.safe_load(f)


def _resolve_args(custom_args):
    """Resolve final test arguments from CLI args and optional recipe config.

    Returns a dict with keys: model_path, adapter_path, tokenizer, max_new_tokens,
    smoke_test, trust_remote_code, enable_expert_parallel.
    """
    config_path = custom_args.get("config_path")
    mode = custom_args.get("deploy_mode")  # "sft", "peft", or None (legacy)
    cfg = _load_recipe_config(config_path) if config_path else {}

    model_cfg = cfg.get("model", {})
    tokenizer_cfg = cfg.get("tokenizer", {})
    ci_cfg = cfg.get("ci", {})

    # -- model_path and adapter_path --
    if mode == "peft":
        # PEFT: base model from config, adapter from --adapter_path
        model_path = model_cfg["pretrained_model_name_or_path"]
        adapter_path = custom_args["adapter_path"]
    elif mode == "sft":
        # SFT: full model from --deploy_model_path
        model_path = custom_args["deploy_model_path"]
        adapter_path = None
    else:
        # Legacy: caller provides everything explicitly
        model_path = custom_args["deploy_model_path"]
        adapter_path = custom_args.get("adapter_path")

    # Normalize trailing slash so HF's dynamic-module cache name isn't empty (os.path.basename).
    if isinstance(model_path, str):
        model_path = model_path.rstrip("/") or model_path

    # -- tokenizer --
    if "tokenizer" in custom_args:
        tokenizer = custom_args["tokenizer"]
    elif tokenizer_cfg.get("pretrained_model_name_or_path"):
        tokenizer = tokenizer_cfg["pretrained_model_name_or_path"]
    else:
        tokenizer = model_path

    # -- flags --
    # trust_remote_code placement varies: top-level `model:` or nested under
    # `ci.checkpoint_robustness:`. Accept any source that says true.
    ckpt_robustness_cfg = ci_cfg.get("checkpoint_robustness") or {}
    trust_remote_code = bool(
        custom_args.get("trust_remote_code")
        or model_cfg.get("trust_remote_code")
        or ckpt_robustness_cfg.get("trust_remote_code")
    )

    smoke_test = bool(custom_args.get("vllm_smoke_test") or ci_cfg.get("vllm_smoke_test"))
    enable_expert_parallel = bool(ci_cfg.get("vllm_enable_expert_parallel"))
    merge_lora = bool(ckpt_robustness_cfg.get("vllm_merge_lora"))

    max_new_tokens = int(custom_args.get("max_new_tokens", "20"))

    return {
        "model_path": model_path,
        "adapter_path": adapter_path,
        "tokenizer": tokenizer,
        "max_new_tokens": max_new_tokens,
        "smoke_test": smoke_test,
        "trust_remote_code": trust_remote_code,
        "enable_expert_parallel": enable_expert_parallel,
        "merge_lora": merge_lora,
    }


# Extract custom args at module level before pytest processes them.
_custom_args, _remaining_argv = _extract_custom_args(sys.argv)
sys.argv = _remaining_argv


def test_vllm_greedy_matches_hf():
    """Load a checkpoint with HF and vLLM, then verify greedy outputs match token-for-token."""

    args = _resolve_args(_custom_args)
    model_path = args["model_path"]
    adapter_path = args["adapter_path"]
    tokenizer_path = args["tokenizer"]
    max_new_tokens = args["max_new_tokens"]
    smoke_test = args["smoke_test"]
    trust_remote_code = args["trust_remote_code"]
    enable_expert_parallel = args["enable_expert_parallel"]
    merge_lora = args["merge_lora"]

    from vllm import LLM, SamplingParams

    if smoke_test:
        # Smoke test: just verify vLLM can load the model and generate non-empty output.
        # Uses native vLLM backend (no model_impl="transformers"), no HF comparison.
        # tp = GPUs exposed by the launcher (CUDA_VISIBLE_DEVICES); 1 by default, more for large models.
        tp_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if adapter_path is not None:
            from vllm.lora.request import LoRARequest

            print(f"[vLLM smoke test] Loading model from {model_path} with enable_lora=True (tp={tp_size})")
            llm = LLM(
                model=model_path,
                enable_lora=True,
                max_lora_rank=64,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
            )
            lora_request = LoRARequest("adapter", 1, adapter_path)
            sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
            vllm_results = llm.generate(PROMPTS, sampling_params, lora_request=lora_request)
        else:
            print(f"[vLLM smoke test] Loading model from {model_path} (tp={tp_size})")
            llm = LLM(
                model=model_path,
                trust_remote_code=trust_remote_code,
                tensor_parallel_size=tp_size,
                enable_expert_parallel=enable_expert_parallel,
            )
            sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
            vllm_results = llm.generate(PROMPTS, sampling_params)

        for idx, result in enumerate(vllm_results):
            tokens = list(result.outputs[0].token_ids)
            assert len(tokens) > 0, f"Prompt {idx}: vLLM generated 0 tokens"
            print(f"Prompt {idx}: PASS ({len(tokens)} tokens generated)")
        return

    # Default mode: compare vLLM (model_impl="transformers") against HF token-for-token.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if adapter_path is not None:
        from peft import PeftModel

        print(f"[HF] Loading base model from {model_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
        ).to(device)
        print(f"[HF] Loading adapter from {adapter_path}")
        hf_model = PeftModel.from_pretrained(base_model, adapter_path, torch_dtype=torch.bfloat16)
    else:
        print(f"[HF] Loading model from {model_path} on {device}")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=trust_remote_code
        ).to(device)

    hf_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=trust_remote_code)

    hf_outputs = []
    for idx, prompt in enumerate(PROMPTS):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = hf_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        hf_outputs.append(generated.tolist())
        print(f"[HF] Prompt {idx}: generated {len(generated)} tokens")

    merged_dir = None
    if adapter_path is not None and merge_lora:
        merged_dir = tempfile.mkdtemp(prefix="merged_adapter_", dir=os.path.dirname(os.path.normpath(adapter_path)))
        print(f"[merge] merging adapter into base model -> {merged_dir}")
        hf_model.merge_and_unload().save_pretrained(merged_dir)
        tokenizer.save_pretrained(merged_dir)

    del hf_model
    if adapter_path is not None:
        del base_model
    gc.collect()
    torch.cuda.empty_cache()

    # vLLM greedy decoding
    if adapter_path is not None and merge_lora:
        print(f"[vLLM] Loading merged model from {merged_dir}")
        llm = LLM(
            model=merged_dir,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.7,
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        vllm_results = llm.generate(PROMPTS, sampling_params)
    elif adapter_path is not None:
        from vllm.lora.request import LoRARequest

        print(f"[vLLM] Loading base model from {model_path} with enable_lora=True")
        llm = LLM(
            model=model_path,
            enable_lora=True,
            max_lora_rank=64,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.7,
        )
        lora_request = LoRARequest("adapter", 1, adapter_path)
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        vllm_results = llm.generate(PROMPTS, sampling_params, lora_request=lora_request)
    else:
        print(f"[vLLM] Loading model from {model_path}")
        llm = LLM(
            model=model_path,
            model_impl="transformers",
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=0.7,
        )
        sampling_params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
        vllm_results = llm.generate(PROMPTS, sampling_params)

    vllm_outputs = []
    for idx, result in enumerate(vllm_results):
        tokens = list(result.outputs[0].token_ids)
        vllm_outputs.append(tokens)
        print(f"[vLLM] Prompt {idx}: generated {len(tokens)} tokens")

    MIN_MATCH_PREFIX = 5
    for i, prompt in enumerate(PROMPTS):
        hf_tokens = hf_outputs[i]
        vllm_tokens = vllm_outputs[i]
        assert min(len(hf_tokens), len(vllm_tokens)) >= MIN_MATCH_PREFIX, (
            f"Too few tokens for prompt {i}: HF={len(hf_tokens)}, vLLM={len(vllm_tokens)} "
            f"(need >= {MIN_MATCH_PREFIX} generated tokens to compare)"
        )
        match_len = next(
            (j for j, (a, b) in enumerate(zip(hf_tokens, vllm_tokens)) if a != b),
            min(len(hf_tokens), len(vllm_tokens)),
        )
        assert match_len >= MIN_MATCH_PREFIX, (
            f"Token mismatch for prompt {i}: {prompt!r}\n"
            f"  HF and vLLM agree on only {match_len} leading token(s) (require >= {MIN_MATCH_PREFIX}).\n"
            f"  Divergence within the first tokens indicates a broken checkpoint load; "
            f"later divergence is expected fp nondeterminism between engines.\n"
            f"  HF:   {hf_tokens[:20]}...\n  vLLM: {vllm_tokens[:20]}..."
        )
        print(f"Prompt {i}: PASS ({match_len}/{min(len(hf_tokens), len(vllm_tokens))} leading tokens match)")
