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

"""Merge a LoRA/QLoRA adapter into a base HuggingFace model and save the result.

Supports both dense and Mixture-of-Experts (MoE) models. For QLoRA adapters the
base model is loaded in 4-bit, dequantized, and only then merged so that the
adapter delta is applied to the correct weight representation (avoids the
"naive merge" quality degradation described in
https://kaitchup.substack.com/p/lora-adapters-when-a-naive-merge).

Usage examples
--------------

Standard LoRA merge (dense)::

    python tools/merge_lora.py \
        --base-model meta-llama/Llama-3.2-1B \
        --adapter-path checkpoints/adapter/ \
        --output-dir merged_model/

QLoRA merge (dequantize then merge)::

    python tools/merge_lora.py \
        --base-model meta-llama/Llama-3.2-1B \
        --adapter-path checkpoints/adapter/ \
        --output-dir merged_model/ \
        --qlora

MoE model merge::

    python tools/merge_lora.py \
        --base-model deepseek-ai/DeepSeek-V2-Lite \
        --adapter-path checkpoints/adapter/ \
        --output-dir merged_model/

Embedding / non-CausalLM model merge (auto-detected from adapter task_type)::

    python tools/merge_lora.py \
        --base-model BAAI/bge-base-en-v1.5 \
        --adapter-path checkpoints/adapter/ \
        --output-dir merged_model/

Explicit model class override::

    python tools/merge_lora.py \
        --base-model BAAI/bge-base-en-v1.5 \
        --adapter-path checkpoints/adapter/ \
        --output-dir merged_model/ \
        --model-class AutoModel
"""

import argparse
import copy
import gc
import json
import logging
import os

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TASK_TYPE_TO_AUTO_CLASS = {
    "CAUSAL_LM": "AutoModelForCausalLM",
    "SEQ_CLS": "AutoModelForSequenceClassification",
    "SEQ_2_SEQ_LM": "AutoModelForSeq2SeqLM",
    "TOKEN_CLS": "AutoModelForTokenClassification",
    "QUESTION_ANS": "AutoModelForQuestionAnswering",
    "FEATURE_EXTRACTION": "AutoModel",
}


def _resolve_auto_cls(adapter_path: str, model_class: str | None = None):
    """Return the ``transformers.AutoModel*`` class to use for loading.

    Resolution order:
    1. Explicit *model_class* string (e.g. ``"AutoModel"``).
    2. ``task_type`` field in the adapter's ``adapter_config.json``.
    3. Fall back to ``AutoModelForCausalLM``.
    """
    import transformers

    if model_class is not None:
        cls = getattr(transformers, model_class, None)
        if cls is None:
            raise ValueError(
                f"Unknown model class '{model_class}'. "
                f"Must be an attribute of the `transformers` package "
                f"(e.g. AutoModelForCausalLM, AutoModel)."
            )
        return cls

    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            adapter_cfg = json.load(f)
        task_type = adapter_cfg.get("task_type")
        if task_type and task_type in TASK_TYPE_TO_AUTO_CLASS:
            cls_name = TASK_TYPE_TO_AUTO_CLASS[task_type]
            logger.info("Detected task_type=%s → using %s", task_type, cls_name)
            return getattr(transformers, cls_name)
        if task_type:
            logger.warning(
                "Unrecognised task_type '%s' in adapter config; falling back to AutoModelForCausalLM.",
                task_type,
            )

    return transformers.AutoModelForCausalLM


def dequantize_model(model, dtype=torch.float16, device="cpu"):
    """Replace every ``bitsandbytes.nn.Linear4bit`` with a plain ``nn.Linear``.

    The 4-bit weights are dequantized using the stored ``quant_state`` so that
    the full-precision delta from the LoRA adapter is applied correctly.

    Args:
        model: A HuggingFace model loaded with 4-bit quantization.
        dtype: Target dtype for the dequantized weights.
        device: Device to place the new linear layers on.

    Returns:
        The same model object, modified in-place.
    """
    import bitsandbytes as bnb
    from bitsandbytes.functional import dequantize_4bit

    with torch.no_grad():
        for name, module in model.named_modules():
            if isinstance(module, bnb.nn.Linear4bit):
                logger.info("Dequantizing %s", name)
                quant_state = copy.deepcopy(module.weight.quant_state)
                quant_state[2] = dtype
                weights = dequantize_4bit(module.weight.data, quant_state=quant_state, quant_type="nf4").to(dtype)

                new_module = torch.nn.Linear(
                    module.in_features, module.out_features, bias=module.bias is not None, dtype=dtype
                )
                new_module.weight = torch.nn.Parameter(weights)
                if module.bias is not None:
                    new_module.bias = torch.nn.Parameter(module.bias.data.to(dtype))
                new_module.to(device=device, dtype=dtype)

                # Walk to the parent and swap the child module.
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent = model.get_submodule(parts[0])
                    setattr(parent, parts[1], new_module)
                else:
                    setattr(model, name, new_module)

    model.is_loaded_in_4bit = False
    return model


def _clean_quantization_config(output_dir):
    """Remove ``quantization_config`` from the saved ``config.json``."""
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.exists(config_path):
        return
    with open(config_path, "r") as f:
        config_data = json.load(f)
    changed = False
    for key in ("quantization_config", "pretraining_tp"):
        if key in config_data:
            config_data.pop(key)
            changed = True
    if changed:
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)


def merge_lora(
    base_model: str,
    adapter_path: str,
    output_dir: str,
    *,
    qlora: bool = False,
    dtype: str = "float16",
    device: str = "auto",
    save_tokenizer: bool = True,
    trust_remote_code: bool = False,
    model_class: str | None = None,
):
    """Load a base model, apply a LoRA adapter, merge, and save.

    Args:
        base_model: HuggingFace model name or path.
        adapter_path: Path to the PEFT adapter directory.
        output_dir: Where to write the merged model.
        qlora: If True, load the base model in 4-bit and dequantize before merging.
        dtype: Weight dtype for the merged model (``float16``, ``bfloat16``, ``float32``).
        device: ``auto``, ``cpu``, or ``cuda:N``.
        save_tokenizer: Whether to save the tokenizer alongside the model.
        trust_remote_code: Passed through to ``from_pretrained``.
        model_class: Explicit ``transformers`` Auto class name (e.g.
            ``"AutoModel"``, ``"AutoModelForCausalLM"``).  When ``None``
            the class is inferred from the adapter's ``task_type``.
    """
    from peft import PeftModel
    from transformers import AutoTokenizer

    auto_cls = _resolve_auto_cls(adapter_path, model_class)
    torch_dtype = getattr(torch, dtype)

    # --- Load base model ---
    load_kwargs = dict(
        trust_remote_code=trust_remote_code,
    )

    if qlora:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        load_kwargs["quantization_config"] = bnb_config
        load_kwargs["device_map"] = {"": 0}
        logger.info("Loading base model in 4-bit for QLoRA merge (%s): %s", auto_cls.__name__, base_model)
    else:
        load_kwargs["torch_dtype"] = torch_dtype
        load_kwargs["device_map"] = device
        logger.info("Loading base model (%s): %s", auto_cls.__name__, base_model)

    model = auto_cls.from_pretrained(base_model, **load_kwargs)

    # --- Dequantize if QLoRA ---
    if qlora:
        logger.info("Dequantizing model weights from 4-bit to %s", dtype)
        model = dequantize_model(model, dtype=torch_dtype, device="cuda")

    # --- Apply adapter and merge ---
    logger.info("Loading adapter from %s", adapter_path)
    model = PeftModel.from_pretrained(model, adapter_path)

    logger.info("Merging adapter into base model")
    model = model.merge_and_unload()

    # --- Save ---
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving merged model to %s", output_dir)
    model.save_pretrained(output_dir, safe_serialization=True)

    if qlora:
        _clean_quantization_config(output_dir)

    if save_tokenizer:
        try:
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=trust_remote_code)
            tokenizer.save_pretrained(output_dir)
            logger.info("Tokenizer saved to %s", output_dir)
        except Exception as e:
            logger.warning("Could not save tokenizer: %s", e)

    logger.info("Merge complete.")

    # Free memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge a LoRA / QLoRA adapter into a HuggingFace base model.")
    parser.add_argument(
        "--base-model",
        "-m",
        required=True,
        help="HuggingFace model name/path (e.g. meta-llama/Llama-3.2-1B or /path/to/model).",
    )
    parser.add_argument(
        "--adapter-path",
        "-a",
        required=True,
        help="Path to the PEFT adapter directory (must contain adapter_config.json).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        required=True,
        help="Directory to save the merged model.",
    )
    parser.add_argument(
        "--qlora",
        action="store_true",
        default=False,
        help="Load base model in 4-bit and dequantize before merging (for QLoRA adapters).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Weight dtype for the merged model (default: float16).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device map for model loading (default: auto). Use 'cpu' for CPU-only.",
    )
    parser.add_argument(
        "--no-save-tokenizer",
        action="store_true",
        default=False,
        help="Skip saving the tokenizer.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=False,
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--model-class",
        default=None,
        help=(
            "Explicit transformers Auto class name (e.g. AutoModel, AutoModelForCausalLM). "
            "When omitted, inferred from the adapter's task_type in adapter_config.json."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    merge_lora(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        output_dir=args.output_dir,
        qlora=args.qlora,
        dtype=args.dtype,
        device=args.device,
        save_tokenizer=not args.no_save_tokenizer,
        trust_remote_code=args.trust_remote_code,
        model_class=args.model_class,
    )


if __name__ == "__main__":
    main()
