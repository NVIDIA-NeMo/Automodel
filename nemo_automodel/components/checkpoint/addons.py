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

from typing import Protocol, TYPE_CHECKING, Optional, Any
import os
import torch
from torch import nn
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
import json

if TYPE_CHECKING:
    from peft import PeftConfig

class CheckpointAddon(Protocol):
    """
    Optional hooks that run around backend IO (used for PEFT and consolidated HF metadata).
    """
    def pre_save(self, **kwargs) -> None: ...

class ConsolidatedHFAddon:
    """
    Addon for consolidated HF metadata.
    """
    def pre_save(self, **kwargs):
        model_state = kwargs["model_state"]
        consolidated_model_path = kwargs["consolidated_path"]
        tokenizer = kwargs["tokenizer"]
        model_part = model_state.model[0]  # ModelState already converts to list if needed

        # Perform save operations on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # save the config.json file
            if hasattr(model_part, "config"):
                with open(os.path.join(consolidated_model_path, "config.json"), "w") as f:
                    f.write(model_part.config.to_json_string())
            # save the generation_config.json file
            if hasattr(model_part, "generation_config"):
                with open(os.path.join(consolidated_model_path, "generation_config.json"), "w") as f:
                    f.write(model_part.generation_config.to_json_string())

            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(consolidated_model_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


class PeftAddon:
    """
    Addon for PEFT metadata.
    """
    def pre_save(self, **kwargs):
        model_path = kwargs["model_path"]
        tokenizer = kwargs["tokenizer"]
        model_state = kwargs["model_state"]
        peft_config = kwargs["peft_config"]
        hf_peft_config = _get_hf_peft_config(peft_config, model_state)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)
            # save in HF format. Only keys that are needed for PEFT module loading will be saved here.
            with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
                json.dump(hf_peft_config, f, indent=2, sort_keys=True)
            # save the full PEFT config for inference loading inside Automodel.
            with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
                json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState) -> dict:
    """
    Get the PEFT config in the format expected by Hugging Face.
    """
    MODEL_TYPE_TO_PEFT_TASK_TYPE = {
        "SequenceClassification": "SEQ_CLS",
        "Seq2SeqLM": "SEQ_2_SEQ_LM",
        "CausalLM": "CAUSAL_LM",
        "TokenClassification": "TOKEN_CLS",
        "QuestionAnswering": "QUESTION_ANS",
        "FeatureExtraction": "FEATURE_EXTRACTION",
    }
    model_part = model_state.model[0]
    target_modules = _extract_target_modules(model_part)
    try:
        model_task = model_part.config.architectures[0].split("For")[-1]
    except (AttributeError, IndexError, TypeError):
        model_task = "N/A"

    try:
        name_or_path = model_part.config.name_or_path
    except (AttributeError, TypeError):
        name_or_path = "N/A"

    try:
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except KeyError:
        task_type = "CAUSAL_LM"

    return {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": name_or_path,
    }

def _get_automodel_peft_metadata(peft_config: "PeftConfig") -> dict:
    """
    Get the PEFT metadata in the format expected by Automodel.
    """
    PEFT_KEYS = {"dim", "alpha"}
    return {k: v for k, v in peft_config.to_dict().items() if k not in PEFT_KEYS}

def _extract_target_modules(model: nn.Module) -> list[str]:
    """
    Extract the target modules from the model.

    Note: When torch.compile is used, module names get prefixed with '_orig_mod.'.
    This function strips those prefixes to get the original module names.
    """
    final_target_modules = set()
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            # Remove the torch.compile _orig_mod prefix if present
            target_name = name.rsplit(".", 1)[0]
            if target_name.startswith("_orig_mod."):
                target_name = target_name[len("_orig_mod.") :]
            final_target_modules.add(target_name)
    return sorted(list(final_target_modules))