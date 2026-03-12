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
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin

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
                v4_compatible = kwargs.get("v4_compatible", False)
                if (
                    v4_compatible
                    and original_model_path is not None
                    and _config_exists(original_model_path, "config.json")
                ):
                    _save_original_config_json(original_model_path, hf_metadata_dir, "config.json")
                else:
                    _maybe_strip_quantization_config(model_part)
                    with open(os.path.join(hf_metadata_dir, "config.json"), "w") as f:
                        f.write(model_part.config.to_json_string())
            # save the generation_config.json file
            if getattr(model_part, "generation_config", None) is not None:
                if (
                    v4_compatible
                    and original_model_path is not None
                    and _config_exists(original_model_path, "generation_config.json")
                ):
                    _save_original_config_json(original_model_path, hf_metadata_dir, "generation_config.json")
                else:
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
        arch_name = model_part.config.architectures[0]
        # "LlamaForCausalLM".split("For") → ["Llama", "CausalLM"]
        # "LlamaBidirectionalModel".split("For") → ["LlamaBidirectionalModel"]
        parts = arch_name.split("For")
        model_task = parts[-1] if len(parts) > 1 else "FeatureExtraction"
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
        "use_dora": peft_config.use_dora,
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

    Combined-projection module names (e.g. ``qkv_proj``, ``gate_up_proj``) are
    expanded to the individual Hugging Face projection names so that the saved
    ``adapter_config.json`` is compatible with vLLM, TensorRT-LLM and the
    Hugging Face PEFT library.

    For MoE expert LoRA (GroupedExpertsLoRA / GroupedExpertsDeepEPLoRA), the
    grouped 3-D adapter parameters are expanded to per-expert HF projection
    names (e.g. ``model.layers.0.mlp.experts.0.gate_proj``).

    Note:
        When torch.compile is used, module names get prefixed with `_orig_mod.`.
        This function strips those prefixes to get the original module names.

    Args:
        model: The model whose named modules are scanned.

    Returns:
        A sorted list of unique module name prefixes that contain LoRA layers.
    """
    # Mapping from combined projection names to their HF-compatible split names.
    _COMBINED_TO_SPLIT = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    _MOE_LORA_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")

    final_target_modules = set()
    for name, _ in model.named_modules():
        if "lora" in name.lower():
            # Remove the torch.compile _orig_mod prefix if present
            target_name = name.rsplit(".", 1)[0]
            if target_name.startswith("_orig_mod."):
                target_name = target_name[len("_orig_mod.") :]

            # Expand combined projection names to individual HF projection names
            last_component = target_name.rsplit(".", 1)[-1]
            if last_component in _COMBINED_TO_SPLIT:
                parent = target_name.rsplit(".", 1)[0] if "." in target_name else ""
                for split_name in _COMBINED_TO_SPLIT[last_component]:
                    expanded = f"{parent}.{split_name}" if parent else split_name
                    final_target_modules.add(expanded)
            else:
                final_target_modules.add(target_name)

    # Detect MoE expert LoRA: adapter weights stored as nn.Parameter (not
    # nn.Module) so they don't appear in named_modules(). Scan parameters
    # and expand to per-expert HF projection names.
    # Only applies to models that use split-expert state dict conversion
    # (MoESplitExpertsStateDictMixin); models with natively merged experts
    # (e.g. Qwen 3.5) don't need per-expert expansion.
    if hasattr(model, "state_dict_adapter") and isinstance(model.state_dict_adapter, MoESplitExpertsStateDictMixin):
        seen_expert_groups: set[tuple[str, str]] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            for lora_suffix in _MOE_LORA_SUFFIXES:
                if name.endswith(f".{lora_suffix}"):
                    expert_path = name[: -len(f".{lora_suffix}")]
                    if expert_path.startswith("_orig_mod."):
                        expert_path = expert_path[len("_orig_mod.") :]

                    group = "gate_and_up" if "gate_and_up" in lora_suffix else "down"
                    if (expert_path, group) in seen_expert_groups:
                        break
                    seen_expert_groups.add((expert_path, group))

                    n_experts = param.shape[0]
                    for expert_id in range(n_experts):
                        if group == "gate_and_up":
                            final_target_modules.add(f"{expert_path}.{expert_id}.gate_proj")
                            final_target_modules.add(f"{expert_path}.{expert_id}.up_proj")
                        else:
                            final_target_modules.add(f"{expert_path}.{expert_id}.down_proj")
                    break

    # When the model is a biencoder, LoRA target modules are discovered under
    # the "lm_q." prefix (the biencoder query-encoder wrapper).  The standalone
    # HF base model (LlamaBidirectionalModel) uses bare names without any
    # wrapper prefix (e.g. "layers.0.self_attn.q_proj").  Strip "lm_q." so
    # the saved adapter_config.json is compatible with merge_lora / HF PEFT.
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is not None:
        from nemo_automodel.components.models.common.bidirectional import BiencoderStateDictAdapter

        if isinstance(adapter, BiencoderStateDictAdapter):
            remapped = set()
            for name in final_target_modules:
                if name.startswith("lm_q."):
                    remapped.add(name[len("lm_q.") :])
                else:
                    remapped.add(name)
            final_target_modules = remapped

    return sorted(list(final_target_modules))


def _maybe_strip_quantization_config(model_part: nn.Module) -> None:
    """Remove ``quantization_config`` from the HF config when no parameters are quantized.

    Models loaded from quantized checkpoints (e.g. mxfp4 GPT-OSS) carry a
    ``quantization_config`` on their ``config`` object.  After dequantization
    all parameters are standard floating-point, but the stale config entry would
    still be written to the saved ``config.json``.  This strips it so the output
    checkpoint is a clean bf16 checkpoint, consistent with e.g.
    ``unsloth/gpt-oss-20b-BF16``.
    """
    config = getattr(model_part, "config", None)
    if config is None or not hasattr(config, "quantization_config"):
        return

    _QUANTIZED_DTYPES = frozenset({torch.uint8, torch.int8})
    if any(p.dtype in _QUANTIZED_DTYPES for p in model_part.parameters()):
        return

    delattr(config, "quantization_config")


def _config_exists(original_model_path: nn.Module, config_name: str) -> bool:
    src = os.path.join(original_model_path, config_name)
    return os.path.isfile(src)


def _save_original_config_json(original_model_path: str, hf_metadata_dir: str, config_name: str) -> None:
    """Copy the original pretrained ``config.json`` with ``quantization_config`` stripped.

    This is used in v4-compatible mode so that downstream consumers (e.g. vLLM)
    that expect a transformers-v4-style config receive the file verbatim from the
    original checkpoint, minus any quantization metadata (since saved weights are
    always bf16).
    """
    src = os.path.join(original_model_path, config_name)
    if not os.path.isfile(src):
        return
    with open(src) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    dst = os.path.join(hf_metadata_dir, config_name)
    with open(dst, "w") as f:
        json.dump(cfg, f, indent=2)


def _maybe_save_custom_model_code(original_model_path: str | None, hf_metadata_dir: str) -> None:
    """
    Save the custom model code if it exists. This function preserves the original directory structure.
    """
    if original_model_path is None:
        return
    if os.path.isfile(original_model_path):
        pattern = original_model_path
    elif os.path.isdir(original_model_path):
        pattern = os.path.join(original_model_path, "**", "*.py")
    else:
        return
    for src_path in glob.glob(pattern, recursive=True):
        rel_path = os.path.relpath(src_path, original_model_path)
        if os.path.basename(src_path) == "__init__.py":
            continue
        dst_path = os.path.join(hf_metadata_dir, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
