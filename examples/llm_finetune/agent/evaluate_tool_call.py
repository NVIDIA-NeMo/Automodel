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

"""Offline tool-call accuracy evaluation for agent SFT checkpoints.

Training only logs ``val_loss``, which cannot tell "the model learned the
format" from "the model emits the wrong tool name / malformed argument JSON".
The FSDP2 training recipe also skips in-loop generation eval by default. This
script closes that gap: load a finished checkpoint (or any HF model) and run
:class:`ToolCallAccuracyEvaluator` over a held-out function-calling set, then
print the ``tool_call/*`` metrics.

Usage:
    # Consolidated (HuggingFace-compatible) checkpoint from SFT training
    python examples/llm_finetune/agent/evaluate_tool_call.py \
        --checkpoint-path /path/to/checkpoint/epoch_X_step_Y \
        --data-path /path/to/heldout_function_calling.jsonl

    # Distributed (DCP) or LoRA checkpoint: also pass the base model
    python examples/llm_finetune/agent/evaluate_tool_call.py \
        --checkpoint-path /path/to/checkpoint/epoch_X_step_Y \
        --base-model-path Qwen/Qwen2.5-3B \
        --data-path /path/to/heldout_function_calling.jsonl

    # Baseline an untrained model straight from the Hub
    python examples/llm_finetune/agent/evaluate_tool_call.py \
        --base-model-path Qwen/Qwen2.5-3B \
        --dataset-name llamafactory/glaive_toolcall_en --split "test"

Prefer a genuinely held-out split (a separate ``--data-path`` jsonl, or a
``--split`` the model was not trained on). ``train[:N]`` is fine for a smoke
test but measures memorization, not generalization.
"""

import argparse
import glob
import json
import logging
import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from nemo_automodel._transformers import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.eval.tool_call_evaluator import ToolCallAccuracyEvaluator
from nemo_automodel.components.loggers.log_utils import setup_logging


def _get_checkpoint_type(checkpoint_path: str) -> str:
    """Return ``"safetensors"`` if the checkpoint holds safetensors, else ``"torch_save"`` (DCP)."""
    safetensors = glob.glob(os.path.join(checkpoint_path, "model", "*.safetensors"))
    return "safetensors" if safetensors else "torch_save"


def _is_peft_checkpoint(checkpoint_path: str) -> bool:
    """True if the checkpoint stores LoRA adapter weights."""
    return os.path.exists(os.path.join(checkpoint_path, "model", "adapter_model.safetensors"))


def _is_consolidated_safetensors_checkpoint(checkpoint_path: str) -> bool:
    """True if the checkpoint is a consolidated (HF-loadable) safetensors checkpoint."""
    return os.path.exists(os.path.join(checkpoint_path, "model", "model.safetensors.index.json"))


def _apply_peft_to_model(model: NeMoAutoModelForCausalLM, checkpoint_path: str) -> None:
    """Re-apply the LoRA structure recorded in the checkpoint before loading adapter weights."""
    peft_dict = {}
    with open(os.path.join(checkpoint_path, "model", "adapter_config.json"), "r") as f:
        restored = json.load(f)
        peft_dict["dim"] = restored["r"]
        peft_dict["alpha"] = restored["lora_alpha"]
    with open(os.path.join(checkpoint_path, "model", "automodel_peft_config.json"), "r") as f:
        peft_dict |= json.load(f)
    apply_lora_to_linear_modules(model, PeftConfig.from_dict(peft_dict))


def load_model(checkpoint_path: Optional[str], base_model_path: Optional[str]) -> NeMoAutoModelForCausalLM:
    """Load a causal LM for evaluation from a checkpoint or straight from the Hub.

    Mirrors ``examples/vlm_generate/generate.py``: a consolidated safetensors
    checkpoint loads directly; a DCP / LoRA checkpoint loads the base model and
    restores weights on top; with no ``checkpoint_path`` the base model is loaded
    as-is (a baseline).
    """
    from nemo_automodel.components.distributed.init_utils import initialize_distributed

    initialize_distributed(backend="nccl" if torch.cuda.is_available() else "gloo", timeout_minutes=10)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if checkpoint_path is None:
        if base_model_path is None:
            raise ValueError("Provide --checkpoint-path and/or --base-model-path")
        return NeMoAutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)

    model_path = os.path.join(checkpoint_path, "model")
    if _is_consolidated_safetensors_checkpoint(checkpoint_path):
        return NeMoAutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(device)

    if base_model_path is None:
        raise ValueError("--base-model-path is required to restore a non-consolidated (DCP/LoRA) checkpoint")

    model = NeMoAutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    if _is_peft_checkpoint(checkpoint_path):
        _apply_peft_to_model(model, checkpoint_path)
    checkpointer = Checkpointer(
        config=CheckpointingConfig(
            enabled=True,
            checkpoint_dir=checkpoint_path,
            model_save_format=_get_checkpoint_type(checkpoint_path),
            model_cache_dir="",
            model_repo_id=base_model_path,
            save_consolidated=False,
            is_peft=_is_peft_checkpoint(checkpoint_path),
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
    )
    checkpointer.load_model(model, model_path)
    return model


def main() -> None:
    """Load a checkpoint, run the tool-call evaluator, and print the metrics."""
    setup_logging()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="NeMo checkpoint dir (contains model/).")
    parser.add_argument("--base-model-path", type=str, default=None, help="HF id or local base model.")
    parser.add_argument("--tokenizer-path", type=str, default=None, help="Defaults to base model / checkpoint.")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-name", type=str, default=None, help="HF Hub function-calling dataset id.")
    source.add_argument("--data-path", type=str, default=None, help="Local eval JSON/JSONL file (held-out).")

    parser.add_argument("--split", type=str, default="train", help="Split for --dataset-name (e.g. 'test').")
    parser.add_argument("--limit-dataset-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=None)
    parser.add_argument("--metric-prefix", type=str, default="tool_call")
    parser.add_argument("--output-file", type=str, default=None, help="Optional path to write metrics as JSON.")
    args = parser.parse_args()

    model = load_model(args.checkpoint_path, args.base_model_path)
    model.eval()

    tokenizer_path = args.tokenizer_path or args.base_model_path
    if tokenizer_path is None and args.checkpoint_path is not None:
        tokenizer_path = os.path.join(args.checkpoint_path, "model")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    evaluator = ToolCallAccuracyEvaluator(
        dataset_name=args.dataset_name,
        path=args.data_path,
        split=args.split,
        limit_dataset_samples=args.limit_dataset_samples,
        max_eval_samples=args.max_eval_samples,
        max_new_tokens=args.max_new_tokens,
        max_prompt_tokens=args.max_prompt_tokens,
        metric_prefix=args.metric_prefix,
    )
    metrics = evaluator.evaluate(model, tokenizer)

    logging.info("tool-call evaluation results:")
    for key in sorted(metrics):
        logging.info("  %s = %.4f", key, metrics[key])

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logging.info("wrote metrics to %s", args.output_file)


if __name__ == "__main__":
    main()
