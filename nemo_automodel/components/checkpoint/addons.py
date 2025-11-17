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

import glob
import json
import os
import shutil
from typing import TYPE_CHECKING, Protocol

import torch
from torch import nn

from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState

if TYPE_CHECKING:
    from peft import PeftConfig


class CheckpointAddon(Protocol):
    """
    Optional hooks that run around backend IO (used for PEFT and consolidated HF metadata).
    """

    def pre_save(self, **kwargs) -> None: ...

    def post_save(self, **kwargs) -> None: ...


class ConsolidatedHFAddon:
    """
    Addon that writes consolidated Hugging Face metadata alongside sharded weights.

    On rank 0, this saves `config.json`, `generation_config.json`, and tokenizer
    artifacts into the provided consolidated directory, then synchronizes ranks.
    """

    def pre_save(self, **kwargs) -> None:
        """
        Pre-save hook to emit consolidated HF artifacts.

        Expected kwargs:
            model_state (ModelState): Wrapper holding the model parts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
            tokenizer (PreTrainedTokenizerBase | None): Optional tokenizer to save.
        """
        model_state = kwargs["model_state"]
        hf_metadata_dir = kwargs["hf_metadata_dir"]
        fqn_to_file_index_mapping = kwargs["fqn_to_file_index_mapping"]
        tokenizer = kwargs.get("tokenizer", None)
        model_part = model_state.model[0]  # ModelState already converts to list if needed
        original_model_path = kwargs["original_model_path"]

        # Perform save operations on rank 0
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # if the HF model has custom model code, we need to save it as part of the checkpoint
            _maybe_save_custom_model_code(original_model_path, hf_metadata_dir)
            # save the config.json file
            if hasattr(model_part, "config"):
                with open(os.path.join(hf_metadata_dir, "config.json"), "w") as f:
                    f.write(model_part.config.to_json_string())
            # save the generation_config.json file
            if getattr(model_part, "generation_config", None) is not None:
                with open(os.path.join(hf_metadata_dir, "generation_config.json"), "w") as f:
                    f.write(model_part.generation_config.to_json_string())

            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(hf_metadata_dir)

            # save the fqn_to_file_index_mapping file
            with open(os.path.join(hf_metadata_dir, "fqn_to_file_index_mapping.json"), "w") as f:
                json.dump(fqn_to_file_index_mapping, f, indent=2, sort_keys=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def post_save(self, **kwargs) -> None:
        """
        Move the saved HF metadata to the consolidated directory.

        The reason we keep it this way is because the HF metadata needs to be available
        for offline consolidation, otherwise any changes made to the config during training
        will be lost.

        Expected kwargs:
            consolidated_path (str): Target directory for consolidated artifacts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
        """
        consolidated_path = kwargs["consolidated_path"]
        hf_metadata_path = kwargs["hf_metadata_path"]
        if not consolidated_path:
            # in this case we are just saving the sharded HF safetensors
            return

        if (not torch.distributed.is_initialized()) or (torch.distributed.get_rank() == 0):
            # Move each item inside hf_metadata_dir into consolidated_path
            for item_name in os.listdir(hf_metadata_path):
                if item_name == "fqn_to_file_index_mapping.json":
                    continue  # this is saved by the consolidation step
                src_path = os.path.join(hf_metadata_path, item_name)
                dst_path = os.path.join(consolidated_path, item_name)
                shutil.move(src_path, dst_path)
            shutil.rmtree(hf_metadata_path, ignore_errors=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


class PeftAddon:
    """
    Addon that writes PEFT-specific metadata and tokenizer alongside adapter weights.

    On rank 0, this saves `adapter_config.json`, `automodel_peft_config.json`,
    the tokenizer (if provided), and synchronizes all ranks afterward.
    """

    def pre_save(self, **kwargs) -> None:
        """
        Pre-save hook to emit PEFT artifacts.

        Expected kwargs:
            model_path (str): Directory in which to save PEFT files.
            tokenizer (PreTrainedTokenizerBase | None): Optional tokenizer to save.
            model_state (ModelState): Wrapper holding the model parts.
            peft_config (PeftConfig): PEFT configuration for serialization.
        """
        model_path = kwargs["model_path"]
        tokenizer = kwargs.get("tokenizer", None)
        model_state = kwargs["model_state"]
        peft_config = kwargs["peft_config"]
        original_model_path = kwargs["original_model_path"]
        hf_peft_config = _get_hf_peft_config(peft_config, model_state)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # if the HF model has custom model code, we need to save it as part of the checkpoint
            _maybe_save_custom_model_code(original_model_path, model_path)
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

    def post_save(self, **kwargs) -> None:
        pass


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState) -> dict:
    """
    Get the minimal PEFT config in the format expected by Hugging Face.

    Args:
        peft_config: Source PEFT configuration.
        model_state: Model wrapper used to infer target modules and model task.

    Returns:
        A dictionary containing the minimal HF-compatible PEFT configuration
        (e.g., task type, LoRA rank/alpha, and discovered target modules).
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

    Args:
        peft_config: Source PEFT configuration.

    Returns:
        A dict containing Automodel-specific PEFT metadata fields filtered from
        the full PEFT configuration.
    """
    PEFT_KEYS = {"dim", "alpha"}
    return {k: v for k, v in peft_config.to_dict().items() if k not in PEFT_KEYS}


def _extract_target_modules(model: nn.Module) -> list[str]:
    """
    Extract the target modules from the model used by LoRA/PEFT layers.

    Note:
        When torch.compile is used, module names get prefixed with `_orig_mod.`.
        This function strips those prefixes to get the original module names.

    Args:
        model: The model whose named modules are scanned.

    Returns:
        A sorted list of unique module name prefixes that contain LoRA layers.
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


def _maybe_save_custom_model_code(original_model_path: str | None, hf_metadata_dir: str) -> None:
    """
    Save the custom model code if it exists. This function preserves the original directory structure.
    """
    if original_model_path is None:
        return
    if not os.path.isdir(original_model_path):
        return
    pattern = os.path.join(original_model_path, "**", "*.py")
    for src_path in glob.glob(pattern, recursive=True):
        # Skip any .hidden paths
        rel_path = os.path.relpath(src_path, original_model_path)
        dst_path = os.path.join(hf_metadata_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
