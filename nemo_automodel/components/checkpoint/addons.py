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
            _maybe_save_custom_model_code(original_model_path, hf_metadata_dir, model_part=model_part)
            # save the config.json file
            if hasattr(model_part, "config"):
                v4_compatible = kwargs.get("v4_compatible", False)
                config_name = "config.json"
                if v4_compatible and _config_exists(original_model_path, config_name):
                    _save_original_config_json(original_model_path, hf_metadata_dir, config_name)
                    config_name = "config.v5.json"

                _maybe_strip_quantization_config(model_part)
                with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
                    if hasattr(model_part.config, "to_json_string"):
                        # Use ``use_diff=False`` so the full config (not the
                        # diff against class defaults) is serialized. For
                        # remote-code configs registered via
                        # ``register_for_auto_class`` (e.g. DeciLM /
                        # Llama-Nemotron-Super-49B ``model_type='nemotron-nas'``),
                        # ``to_diff_dict`` sees the class-level ``model_type``
                        # attribute as equal to the class default and drops
                        # it from the serialized JSON. Reloading via
                        # ``AutoConfig.from_pretrained`` on the resulting
                        # consolidated directory then raises
                        # ``Unrecognized model ... Should have a 'model_type'
                        # key``. Writing the full dict guarantees
                        # ``model_type``, ``architectures`` and ``auto_map``
                        # land in the saved config regardless of class defaults.
                        f.write(model_part.config.to_json_string(use_diff=False))
                    else:
                        # Diffusers models use FrozenDict for config instead of PretrainedConfig
                        json.dump(dict(model_part.config), f, indent=2, default=str)

            # save the generation_config.json file
            if getattr(model_part, "generation_config", None) is not None:
                config_name = "generation_config.json"
                if v4_compatible and _config_exists(original_model_path, config_name):
                    _save_original_config_json(original_model_path, hf_metadata_dir, config_name)
                    config_name = "generation_config.v5.json"
                with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
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
        v4_compatible = kwargs.get("v4_compatible", False)
        hf_peft_config = _get_hf_peft_config(peft_config, model_state, v4_compatible=v4_compatible)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            # if the HF model has custom model code, we need to save it as part of the checkpoint
            model_part = model_state.model[0] if model_state is not None else None
            _maybe_save_custom_model_code(original_model_path, model_path, model_part=model_part)
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


def _get_hf_peft_config(peft_config: "PeftConfig", model_state: ModelState, v4_compatible: bool = False) -> dict:
    """
    Get the minimal PEFT config in the format expected by Hugging Face.

    Args:
        peft_config: Source PEFT configuration.
        model_state: Model wrapper used to infer target modules and model task.
        v4_compatible: When True, use legacy per-expert expansion format.

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
    target_modules = _extract_target_modules(model_part, v4_compatible=v4_compatible)
    target_parameters = _extract_target_parameters(model_part, v4_compatible=v4_compatible)
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

    config = {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": peft_config.dim,
        "lora_alpha": peft_config.alpha,
        "use_dora": peft_config.use_dora,
        "target_modules": target_modules,
        "bias": "none",
        "base_model_name_or_path": name_or_path,
    }
    if target_parameters:
        config["target_parameters"] = target_parameters
    return config


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


def _is_qwen3_moe(model: nn.Module) -> bool:
    """Check whether *model* uses the Qwen3 MoE state-dict adapter."""
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        return False
    from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter

    return isinstance(adapter, Qwen3MoeStateDictAdapter)


def _extract_target_parameters(model: nn.Module, v4_compatible: bool = False) -> list[str]:
    """Extract ``target_parameters`` for PEFT v0.18+ ParamWrapper format.

    Returns fused expert parameter paths for Qwen3 MoE when not in legacy mode,
    or an empty list otherwise.
    """
    if v4_compatible:
        return []
    if _is_qwen3_moe(model):
        return ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    return []


def _extract_target_modules(model: nn.Module, v4_compatible: bool = False) -> list[str]:
    """
    Extract the target modules from the model used by LoRA/PEFT layers.

    Combined-projection module names (e.g. ``qkv_proj``, ``gate_up_proj``) are
    expanded to the individual HF projection names for adapter_config.json
    compatibility with vLLM, TensorRT-LLM, and HF PEFT.

    For MoE expert LoRA, grouped 3-D adapter parameters are expanded to
    per-expert HF projection names unless the model is Qwen3 MoE in
    non-legacy mode (where ``target_parameters`` is used instead).

    Strips ``_orig_mod.`` (torch.compile) and ``_checkpoint_wrapped_module.``
    (activation checkpointing) prefixes from module names.
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
            target_name = name.rsplit(".", 1)[0]
            if target_name.startswith("_orig_mod."):
                target_name = target_name[len("_orig_mod.") :]
            target_name = target_name.replace("_checkpoint_wrapped_module.", "")

            # Expand combined projection names to individual HF projection names
            last_component = target_name.rsplit(".", 1)[-1]
            if last_component in _COMBINED_TO_SPLIT:
                parent = target_name.rsplit(".", 1)[0] if "." in target_name else ""
                for split_name in _COMBINED_TO_SPLIT[last_component]:
                    expanded = f"{parent}.{split_name}" if parent else split_name
                    final_target_modules.add(expanded)
            else:
                final_target_modules.add(target_name)

    # MoE expert LoRA: adapter weights are nn.Parameter (not nn.Module) so
    # they don't appear in named_modules(). Expand to per-expert HF names,
    # unless Qwen3 MoE in non-legacy mode (uses target_parameters instead).
    _has_split_expert_mixin = hasattr(model, "state_dict_adapter") and isinstance(
        model.state_dict_adapter, MoESplitExpertsStateDictMixin
    )
    _skip_for_qwen3 = not v4_compatible and _is_qwen3_moe(model)
    if _has_split_expert_mixin and not _skip_for_qwen3:
        seen_expert_groups: set[tuple[str, str]] = set()
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            for lora_suffix in _MOE_LORA_SUFFIXES:
                if name.endswith(f".{lora_suffix}"):
                    expert_path = name[: -len(f".{lora_suffix}")]
                    if expert_path.startswith("_orig_mod."):
                        expert_path = expert_path[len("_orig_mod.") :]
                    expert_path = expert_path.replace("_checkpoint_wrapped_module.", "")

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

    # Strip "model." prefix for encoder adapters so adapter_config.json
    # is compatible with HF PEFT / merge_lora.
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is not None:
        from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

        if isinstance(adapter, EncoderStateDictAdapter):
            final_target_modules = {
                name[len("model.") :] if name.startswith("model.") else name for name in final_target_modules
            }

    return sorted(final_target_modules)


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


def _config_exists(original_model_path: str, config_name: str) -> bool:
    if original_model_path is None or not os.path.isdir(original_model_path):
        return False
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


def _maybe_save_custom_model_code(
    original_model_path: str | None,
    hf_metadata_dir: str,
    model_part: nn.Module | None = None,
) -> None:
    """
    Save the custom model code if it exists. This function preserves the original directory structure.

    When ``original_model_path`` is a local dir, copy its ``.py`` files. When it is an HF
    hub id (e.g. ``nvidia/Nemotron-Flash-1B``) and the loaded model has ``auto_map`` custom
    code, copy the ``.py`` files from the cached ``transformers_modules`` directory so the
    consolidated checkpoint carries ``modeling_*.py`` locally and reloads without needing
    ``trust_remote_code=True``.
    """
    copied: set[str] = set()

    def _copy_py_tree(src_dir: str) -> None:
        for src_path in glob.glob(os.path.join(src_dir, "**", "*.py"), recursive=True):
            if os.path.basename(src_path) == "__init__.py":
                continue
            rel_path = os.path.relpath(src_path, src_dir)
            dst_path = os.path.join(hf_metadata_dir, rel_path)
            if dst_path in copied:
                continue
            os.makedirs(os.path.dirname(dst_path) or hf_metadata_dir, exist_ok=True)
            shutil.copy2(src_path, dst_path)
            copied.add(dst_path)

    if original_model_path is not None:
        if os.path.isfile(original_model_path):
            dst_path = os.path.join(hf_metadata_dir, os.path.basename(original_model_path))
            os.makedirs(hf_metadata_dir, exist_ok=True)
            shutil.copy2(original_model_path, dst_path)
            copied.add(dst_path)
        elif os.path.isdir(original_model_path):
            _copy_py_tree(original_model_path)

    # Fallback: HF hub id path — resolve custom code via the model class's module file.
    # Needed for trust_remote_code models (e.g. Nemotron-Flash) so reloads from the
    # consolidated dir have has_local_code=True and don't require trust_remote_code.
    if model_part is not None and not copied:
        custom_dirs: set[str] = set()
        for cls in _iter_custom_code_classes(model_part):
            try:
                import inspect

                src_file = inspect.getfile(cls)
            except (TypeError, OSError):
                continue
            module_name = getattr(cls, "__module__", "") or ""
            if not module_name.startswith("transformers_modules."):
                continue
            custom_dirs.add(os.path.dirname(src_file))
        for src_dir in custom_dirs:
            _copy_py_tree(src_dir)


def _iter_custom_code_classes(model_part: nn.Module):
    """Yield classes referenced by ``config.auto_map`` (and the model's own class).

    Walks the full MRO so wrappers like FSDP2 (which add mixins / rename the
    top-level class) don't hide the original ``transformers_modules.*`` class.
    """
    seen: set[type] = set()
    custom_pkg = ""
    for base in type(model_part).__mro__:
        mod = getattr(base, "__module__", "") or ""
        if mod.startswith("transformers_modules."):
            if base not in seen:
                seen.add(base)
                yield base
            # Record the package path to resolve auto_map entries relative to it.
            if not custom_pkg:
                custom_pkg = ".".join(mod.split(".")[:-1])

    config = getattr(model_part, "config", None)
    auto_map = getattr(config, "auto_map", None) if config is not None else None
    if not isinstance(auto_map, dict) or not custom_pkg:
        return
    import importlib

    for value in auto_map.values():
        candidates = value if isinstance(value, (list, tuple)) else [value]
        for ref in candidates:
            if not isinstance(ref, str) or "." not in ref:
                continue
            module_path, class_name = ref.rsplit(".", 1)
            # auto_map entries are like "modeling_nemotron_flash.NemotronFlashForCausalLM";
            # the module lives under the same transformers_modules package as the model class.
            full_module = f"{custom_pkg}.{module_path}"
            try:
                mod = importlib.import_module(full_module)
                target = getattr(mod, class_name, None)
            except Exception:
                continue
            if isinstance(target, type) and target not in seen:
                seen.add(target)
                yield target
