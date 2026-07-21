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

from nemo_automodel.components.checkpoint._backports.hf_utils import (
    FQN_TO_DTYPE_MAPPING_FILENAME,
    FQN_TO_FILE_INDEX_MAPPING_FILENAME,
)
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.moe.state_dict_mixin import MoESplitExpertsStateDictMixin
from nemo_automodel.shared.utils import unwrap_model

if TYPE_CHECKING:
    from peft import PeftConfig
    from torch.distributed import ProcessGroup


def _is_group_rank_0(process_group: "ProcessGroup | None") -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank(group=process_group) == 0


def _group_barrier(process_group: "ProcessGroup | None") -> None:
    if torch.distributed.is_initialized():
        torch.distributed.barrier(group=process_group)


_SENTENCE_TRANSFORMER_POOLING_KEYS = {
    "avg": "pooling_mode_mean_tokens",
    "cls": "pooling_mode_cls_token",
    "last": "pooling_mode_lasttoken",
}
_SOURCE_LEGAL_ASSET_PREFIXES = ("license", "notice")
_TEXT_EXPORT_STALE_PROCESSOR_ASSETS = ("processor_config.json", "preprocessor_config.json")


def _write_json(path: str, value) -> None:
    """Write deterministic, human-readable JSON metadata."""
    with open(path, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def _copy_source_legal_assets(
    original_model_path: str | None,
    hf_metadata_dir: str,
    source_repository_path: str | None = None,
) -> None:
    """Preserve repository- and model-root legal notices without inheriting other source semantics."""
    source_roots = []
    for source_root in (source_repository_path, original_model_path):
        if source_root is None or not os.path.isdir(source_root):
            continue
        if any(os.path.samefile(source_root, existing_root) for existing_root in source_roots):
            continue
        source_roots.append(source_root)

    for source_root in source_roots:
        for item in os.scandir(source_root):
            normalized_name = item.name.lower()
            is_legal_asset = normalized_name == ".gitattributes" or normalized_name.startswith(
                _SOURCE_LEGAL_ASSET_PREFIXES
            )
            if item.is_file() and is_legal_asset:
                shutil.copy2(item.path, os.path.join(hf_metadata_dir, item.name))


def _remove_stale_text_processor_assets(hf_metadata_dir: str) -> None:
    """Remove source multimodal processor metadata from a text-only export."""
    for asset_name in _TEXT_EXPORT_STALE_PROCESSOR_ASSETS:
        asset_path = os.path.join(hf_metadata_dir, asset_name)
        if os.path.isfile(asset_path):
            os.remove(asset_path)


def _restore_source_tokenizer_serialization_state(original_model_path: str | None, hf_metadata_dir: str) -> None:
    """Remove training-time tokenizer state while retaining tokenizer edits."""
    tokenizer_json_path = os.path.join(hf_metadata_dir, "tokenizer.json")
    source_tokenizer_json_path = (
        os.path.join(original_model_path, "tokenizer.json") if original_model_path is not None else None
    )
    if os.path.isfile(tokenizer_json_path):
        with open(tokenizer_json_path) as f:
            tokenizer_json = json.load(f)
        source_tokenizer_json = {}
        if source_tokenizer_json_path is not None and os.path.isfile(source_tokenizer_json_path):
            with open(source_tokenizer_json_path) as f:
                source_tokenizer_json = json.load(f)
        for key in ("truncation", "padding"):
            tokenizer_json[key] = source_tokenizer_json.get(key)
        _write_json(tokenizer_json_path, tokenizer_json)

    tokenizer_config_path = os.path.join(hf_metadata_dir, "tokenizer_config.json")
    source_tokenizer_config_path = (
        os.path.join(original_model_path, "tokenizer_config.json") if original_model_path is not None else None
    )
    if os.path.isfile(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tokenizer_config = json.load(f)
        source_tokenizer_config = {}
        if source_tokenizer_config_path is not None and os.path.isfile(source_tokenizer_config_path):
            with open(source_tokenizer_config_path) as f:
                source_tokenizer_config = json.load(f)
        if "local_files_only" in source_tokenizer_config:
            tokenizer_config["local_files_only"] = source_tokenizer_config["local_files_only"]
        else:
            tokenizer_config.pop("local_files_only", None)
        tokenizer_config.pop("processor_class", None)
        _write_json(tokenizer_config_path, tokenizer_config)


def _read_source_sentence_transformer_max_seq_length(original_model_path: str | None) -> int | None:
    """Read the source Sentence Transformers deployment limit when present."""
    if original_model_path is None:
        return None
    config_path = os.path.join(original_model_path, "sentence_bert_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as f:
        source_config = json.load(f)
    value = source_config.get("max_seq_length")
    if value is None:
        return None
    max_seq_length = int(value)
    if max_seq_length <= 0:
        raise ValueError("Source sentence_bert_config.json max_seq_length must be positive.")
    return max_seq_length


def _resolve_sentence_transformer_max_seq_length(
    model_part: nn.Module,
    export_config,
    tokenizer,
    original_model_path: str | None = None,
) -> int:
    """Resolve deployment sequence length without using training-time truncation."""
    if export_config.max_seq_length is not None:
        max_seq_length = int(export_config.max_seq_length)
        if max_seq_length <= 0:
            raise ValueError("sentence_transformer_max_seq_length must be positive.")
        model_max_seq_length = getattr(getattr(model_part, "config", None), "max_position_embeddings", None)
        if model_max_seq_length is not None:
            model_max_seq_length = int(model_max_seq_length)
            if 0 < model_max_seq_length < 1_000_000_000 and max_seq_length > model_max_seq_length:
                raise ValueError("sentence_transformer_max_seq_length exceeds the model's max_position_embeddings.")
        return max_seq_length

    source_max_seq_length = _read_source_sentence_transformer_max_seq_length(original_model_path)
    if source_max_seq_length is not None:
        model_max_seq_length = getattr(getattr(model_part, "config", None), "max_position_embeddings", None)
        if model_max_seq_length is not None:
            model_max_seq_length = int(model_max_seq_length)
            if 0 < model_max_seq_length < 1_000_000_000 and source_max_seq_length > model_max_seq_length:
                raise ValueError("Source Sentence Transformers max_seq_length exceeds max_position_embeddings.")
        return source_max_seq_length

    candidates = [
        getattr(tokenizer, "model_max_length", None),
        getattr(getattr(model_part, "config", None), "max_position_embeddings", None),
    ]
    finite_candidates = []
    for candidate in candidates:
        if candidate is None:
            continue
        max_seq_length = int(candidate)
        if 0 < max_seq_length < 1_000_000_000:
            finite_candidates.append(max_seq_length)
    if finite_candidates:
        return min(finite_candidates)
    raise ValueError("Unable to determine a finite Sentence Transformers deployment max_seq_length.")


def _validate_sentence_transformer_export(
    model_part: nn.Module,
    export_config,
    tokenizer,
    original_model_path: str | None = None,
) -> None:
    """Validate generated metadata inputs before rank-zero-only checkpoint I/O."""
    pooling = getattr(model_part, "pooling", None)
    if pooling not in _SENTENCE_TRANSFORMER_POOLING_KEYS:
        raise ValueError(f"Pooling mode {pooling!r} cannot be represented by standard Sentence Transformers metadata.")

    model_config = getattr(model_part, "config", None)
    embedding_dimension = getattr(model_config, "hidden_size", None)
    if not isinstance(embedding_dimension, int) or embedding_dimension <= 0:
        raise ValueError("Bi-encoder config must expose a positive hidden_size for Sentence Transformers export.")

    if tokenizer is None:
        raise ValueError("A tokenizer is required to export a loadable Sentence Transformers checkpoint.")

    normalize = bool(getattr(model_part, "l2_normalize", False))
    expected_similarity = "cosine" if normalize else "dot"
    similarity_fn_name = export_config.similarity_fn_name
    if similarity_fn_name is not None and similarity_fn_name != expected_similarity:
        raise ValueError(
            f"similarity_fn_name={similarity_fn_name!r} does not match l2_normalize={normalize}; "
            f"expected {expected_similarity!r}."
        )
    if export_config.do_lower_case:
        raise ValueError(
            "do_lower_case=True is unsupported because the NeMo training pipeline does not lowercase text."
        )

    _resolve_sentence_transformer_max_seq_length(model_part, export_config, tokenizer, original_model_path)


def _save_generated_sentence_transformer_assets(
    model_part: nn.Module,
    export_config,
    original_model_path: str | None,
    hf_metadata_dir: str,
    tokenizer,
) -> None:
    """Generate Sentence Transformers metadata from the effective bi-encoder behavior."""
    _validate_sentence_transformer_export(model_part, export_config, tokenizer, original_model_path)
    pooling = getattr(model_part, "pooling", None)
    model_config = getattr(model_part, "config", None)
    embedding_dimension = getattr(model_config, "hidden_size", None)

    query_prompt = export_config.query_prompt or ""
    document_prompt = export_config.document_prompt or ""

    normalize = bool(getattr(model_part, "l2_normalize", False))
    similarity_fn_name = export_config.similarity_fn_name
    if similarity_fn_name is None:
        similarity_fn_name = "cosine" if normalize else "dot"

    modules = [
        {
            "idx": 0,
            "name": "0",
            "path": "",
            "type": "sentence_transformers.models.Transformer",
        },
        {
            "idx": 1,
            "name": "1",
            "path": "1_Pooling",
            "type": "sentence_transformers.models.Pooling",
        },
    ]
    if normalize:
        modules.append(
            {
                "idx": 2,
                "name": "2",
                "path": "2_Normalize",
                "type": "sentence_transformers.models.Normalize",
            }
        )

    pooling_config = {
        "word_embedding_dimension": embedding_dimension,
        "pooling_mode_cls_token": False,
        "pooling_mode_max_tokens": False,
        "pooling_mode_mean_tokens": False,
        "pooling_mode_mean_sqrt_len_tokens": False,
        "pooling_mode_weightedmean_tokens": False,
        "pooling_mode_lasttoken": False,
        "include_prompt": bool(export_config.include_prompt),
    }
    pooling_config[_SENTENCE_TRANSFORMER_POOLING_KEYS[pooling]] = True

    max_seq_length = _resolve_sentence_transformer_max_seq_length(
        model_part, export_config, tokenizer, original_model_path
    )
    do_lower_case = False

    os.makedirs(os.path.join(hf_metadata_dir, "1_Pooling"), exist_ok=True)
    _write_json(os.path.join(hf_metadata_dir, "modules.json"), modules)
    _write_json(
        os.path.join(hf_metadata_dir, "config_sentence_transformers.json"),
        {
            "prompts": {"query": query_prompt, "document": document_prompt},
            "default_prompt_name": None,
            "similarity_fn_name": similarity_fn_name,
        },
    )
    _write_json(
        os.path.join(hf_metadata_dir, "sentence_bert_config.json"),
        {"max_seq_length": max_seq_length, "do_lower_case": bool(do_lower_case)},
    )
    _write_json(os.path.join(hf_metadata_dir, "1_Pooling", "config.json"), pooling_config)
    _restore_source_tokenizer_serialization_state(original_model_path, hf_metadata_dir)
    _remove_stale_text_processor_assets(hf_metadata_dir)
    _copy_source_legal_assets(
        original_model_path,
        hf_metadata_dir,
        source_repository_path=getattr(model_part, "source_repository_path", None),
    )


def _save_generated_hf_assets(
    model_part: nn.Module,
    metadata_reference_path: str | None,
    hf_metadata_dir: str,
    tokenizer,
    v4_compatible: bool,
    model_config=None,
    save_custom_model_code: bool = True,
) -> None:
    """Run the existing generated Hugging Face metadata export path."""
    if save_custom_model_code:
        _maybe_save_custom_model_code(metadata_reference_path, hf_metadata_dir, model_part=model_part)

    config = model_config if model_config is not None else getattr(model_part, "config", None)
    if config is not None:
        config_name = "config.json"
        if v4_compatible and _config_exists(metadata_reference_path, config_name):
            _save_original_config_json(metadata_reference_path, hf_metadata_dir, config_name)
            config_name = "config.v5.json"

        _maybe_strip_quantization_config(model_part, config=config)
        with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
            if hasattr(config, "to_json_string"):
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
                f.write(config.to_json_string(use_diff=False))
            else:
                # Diffusers models use FrozenDict for config instead of PretrainedConfig
                json.dump(dict(config), f, indent=2, default=str)

    if getattr(model_part, "generation_config", None) is not None:
        config_name = "generation_config.json"
        if v4_compatible and _config_exists(metadata_reference_path, config_name):
            _save_original_config_json(metadata_reference_path, hf_metadata_dir, config_name)
            config_name = "generation_config.v5.json"
        with open(os.path.join(hf_metadata_dir, config_name), "w") as f:
            f.write(model_part.generation_config.to_json_string())

    if tokenizer is not None:
        tokenizer.save_pretrained(hf_metadata_dir)


class CheckpointAddon(Protocol):
    """
    Optional hooks that run around backend IO (used for PEFT and consolidated HF metadata).
    """

    def pre_save(self, **kwargs) -> None: ...

    def post_save(self, **kwargs) -> None: ...


class ConsolidatedHFAddon:
    """
    Addon that writes consolidated Hugging Face metadata alongside sharded weights.

    Bi-encoder checkpoints generate Sentence Transformers metadata from effective
    runtime semantics while retaining source legal notices.
    Other models retain the generated config, custom-code, and tokenizer path.
    Rank 0 writes the artifacts, then synchronizes ranks.
    """

    def pre_save(self, **kwargs) -> None:
        """
        Pre-save hook to emit consolidated HF artifacts.

        Expected kwargs:
            model_state (ModelState): Wrapper holding the model parts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
            tokenizer (PreTrainedTokenizerBase | None): Optional tokenizer to save.
            fqn_to_dtype_mapping (dict[str, str] | None): Original HF safetensors dtype map.
            original_model_path (str | None): Authoritative source checkpoint snapshot.
            generated_metadata_path (str | None): Existing model/code reference for generated metadata fallback.
        """
        model_state = kwargs["model_state"]
        hf_metadata_dir = kwargs["hf_metadata_dir"]
        fqn_to_file_index_mapping = kwargs["fqn_to_file_index_mapping"]
        fqn_to_dtype_mapping = kwargs.get("fqn_to_dtype_mapping", None)
        tokenizer = kwargs.get("tokenizer", None)
        model_part = model_state.model[0]  # ModelState already converts to list if needed
        original_model_path = kwargs["original_model_path"]
        generated_metadata_path = kwargs.get("generated_metadata_path", original_model_path)

        export_model = unwrap_model(model_part)
        sentence_transformer_export_config = getattr(
            export_model,
            "sentence_transformer_export_config",
            None,
        )
        if sentence_transformer_export_config is not None:
            _validate_sentence_transformer_export(
                export_model,
                sentence_transformer_export_config,
                tokenizer,
                original_model_path,
            )

        process_group = kwargs.get("process_group")

        # Perform save operations on rank 0
        if _is_group_rank_0(process_group):
            if sentence_transformer_export_config is not None:
                deploy_config = export_model.get_hf_export_config()
                _save_generated_hf_assets(
                    export_model,
                    generated_metadata_path,
                    hf_metadata_dir,
                    tokenizer,
                    # Bi-encoder metadata always describes the effective trained model;
                    # copying a source config in v4 mode could restore stale semantics.
                    v4_compatible=False,
                    model_config=deploy_config,
                    save_custom_model_code=bool(getattr(deploy_config, "auto_map", None)),
                )
                _save_generated_sentence_transformer_assets(
                    export_model,
                    sentence_transformer_export_config,
                    original_model_path,
                    hf_metadata_dir,
                    tokenizer,
                )
            else:
                _save_generated_hf_assets(
                    model_part,
                    generated_metadata_path,
                    hf_metadata_dir,
                    tokenizer,
                    v4_compatible=kwargs.get("v4_compatible", False),
                )

            # save the fqn_to_file_index_mapping file
            with open(os.path.join(hf_metadata_dir, FQN_TO_FILE_INDEX_MAPPING_FILENAME), "w") as f:
                json.dump(fqn_to_file_index_mapping, f, indent=2, sort_keys=True)
            if fqn_to_dtype_mapping:
                with open(os.path.join(hf_metadata_dir, FQN_TO_DTYPE_MAPPING_FILENAME), "w") as f:
                    json.dump(fqn_to_dtype_mapping, f, indent=2, sort_keys=True)
        _group_barrier(process_group)

    def post_save(self, **kwargs) -> None:
        """
        Copy the saved HF metadata to the consolidated directory.

        The reason we keep it this way is because the HF metadata needs to stay
        available for offline consolidation and re-export, otherwise any changes
        made to the config during training will be lost.

        Expected kwargs:
            consolidated_path (str): Target directory for consolidated artifacts.
            hf_metadata_dir (str): Target directory for HF metadata artifacts.
        """
        consolidated_path = kwargs["consolidated_path"]
        hf_metadata_path = kwargs["hf_metadata_path"]
        process_group = kwargs.get("process_group")
        if not consolidated_path:
            # in this case we are just saving the sharded HF safetensors
            return

        if _is_group_rank_0(process_group):
            # Copy each public metadata item into consolidated_path while keeping
            # .hf_metadata intact for the offline consolidation helper.
            for item_name in os.listdir(hf_metadata_path):
                if item_name in {FQN_TO_FILE_INDEX_MAPPING_FILENAME, FQN_TO_DTYPE_MAPPING_FILENAME}:
                    continue  # internal helper metadata, not part of the HF output
                src_path = os.path.join(hf_metadata_path, item_name)
                dst_path = os.path.join(consolidated_path, item_name)
                if os.path.isdir(src_path):
                    shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_path, dst_path)
        _group_barrier(process_group)


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
        generated_metadata_path = kwargs.get("generated_metadata_path", original_model_path)
        v4_compatible = kwargs.get("v4_compatible", False)
        process_group = kwargs.get("process_group")
        hf_peft_config = _get_hf_peft_config(peft_config, model_state, v4_compatible=v4_compatible)
        automodel_peft_metadata = _get_automodel_peft_metadata(peft_config)
        if _is_group_rank_0(process_group):
            # if the HF model has custom model code, we need to save it as part of the checkpoint
            model_part = model_state.model[0] if model_state is not None else None
            _maybe_save_custom_model_code(generated_metadata_path, model_path, model_part=model_part)
            # save the tokenizer
            if tokenizer is not None:
                tokenizer.save_pretrained(model_path)
            # save in HF format. Only keys that are needed for PEFT module loading will be saved here.
            with open(os.path.join(model_path, "adapter_config.json"), "w") as f:
                json.dump(hf_peft_config, f, indent=2, sort_keys=True)
            # save the full PEFT config for inference loading inside Automodel.
            with open(os.path.join(model_path, "automodel_peft_config.json"), "w") as f:
                json.dump(automodel_peft_metadata, f, indent=2, sort_keys=True)
        _group_barrier(process_group)

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
    # Walk ALL local model parts, not just model[0]: under pipeline parallelism
    # ModelState.model is the list of this rank's virtual stages (each holding a
    # subset of layers). Using only model[0] would capture just the first virtual
    # stage's layers, so adapter_config.json would under-list target_modules even
    # after the cross-PP union. _extract_target_modules / _extract_target_parameters
    # union across the parts internally. The model-class properties (config,
    # architecture) are identical across parts, so read those from model[0].
    model_parts = model_state.model
    model_part = model_parts[0]
    pp_group = getattr(model_state, "pp_group", None)
    target_modules = _extract_target_modules(model_parts, v4_compatible=v4_compatible, pp_group=pp_group)
    target_parameters = _extract_target_parameters(model_parts, v4_compatible=v4_compatible)
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
    result = {}
    for k, v in peft_config.to_dict().items():
        if k in PEFT_KEYS:
            continue
        if isinstance(v, torch.dtype):
            v = str(v)
        result[k] = v
    return result


def _is_qwen3_moe(model: nn.Module) -> bool:
    """Check whether *model* uses the Qwen3 MoE state-dict adapter."""
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        return False
    from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter

    return isinstance(adapter, Qwen3MoeStateDictAdapter)


def _extract_target_parameters(model: "nn.Module | list[nn.Module]", v4_compatible: bool = False) -> list[str]:
    """Extract ``target_parameters`` for PEFT v0.18+ ParamWrapper format.

    Returns fused expert parameter paths for Qwen3 MoE when not in legacy mode,
    or an empty list otherwise.

    ``model`` may be a single module or a list of PP parts; the check is a
    per-model-class property, so the first part is representative.
    """
    model = model[0] if isinstance(model, (list, tuple)) else model
    if v4_compatible:
        return []
    if _is_qwen3_moe(model):
        return ["mlp.experts.gate_up_proj", "mlp.experts.down_proj"]
    return []


def _extract_target_modules(
    model: "nn.Module | list[nn.Module]",
    v4_compatible: bool = False,
    pp_group: "torch.distributed.ProcessGroup | None" = None,
) -> list[str]:
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

    ``model`` may be a single module or a list of modules. Under pipeline
    parallelism the caller passes the full list of this rank's virtual stages,
    so names are unioned across all local parts (using only the first part would
    miss the other stages' layers). When ``pp_group`` is also provided, the
    discovered names are additionally unioned across the PP group so the result
    covers every rank's layers too.
    """
    model_parts = model if isinstance(model, (list, tuple)) else [model]
    # Mapping from combined projection names to their HF-compatible split names.
    _COMBINED_TO_SPLIT = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    _MOE_LORA_SUFFIXES = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")

    final_target_modules = set()
    for _mp in model_parts:
        for name, _ in _mp.named_modules():
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
    # The mixin / qwen3 checks are per-model-class properties (identical across
    # all PP parts), so check the first part.
    _has_split_expert_mixin = hasattr(model_parts[0], "state_dict_adapter") and isinstance(
        model_parts[0].state_dict_adapter, MoESplitExpertsStateDictMixin
    )
    _skip_for_qwen3 = not v4_compatible and _is_qwen3_moe(model_parts[0])
    if _has_split_expert_mixin and not _skip_for_qwen3:
        seen_expert_groups: set[tuple[str, str]] = set()
        for _mp in model_parts:
            for name, param in _mp.named_parameters():
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
    adapter = getattr(model_parts[0], "state_dict_adapter", None)
    if adapter is not None:
        from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

        if isinstance(adapter, EncoderStateDictAdapter):
            final_target_modules = {
                name[len("model.") :] if name.startswith("model.") else name for name in final_target_modules
            }

    # Under pipeline parallelism each rank only holds the local stage's layers,
    # so named_modules() above yields layer-specific target names for that stage
    # only. Union the sets across the PP group so adapter_config.json lists every
    # trained layer, matching the gathered adapter weights (see
    # ModelState.state_dict). Without this the config would advertise only ~1/pp
    # of the layers.
    if pp_group is not None and torch.distributed.get_world_size(group=pp_group) > 1:
        world = torch.distributed.get_world_size(group=pp_group)
        gathered: list[list[str]] = [None] * world
        torch.distributed.all_gather_object(gathered, sorted(final_target_modules), group=pp_group)
        for part in gathered:
            if part:
                final_target_modules.update(part)

    return sorted(final_target_modules)


def _maybe_strip_quantization_config(model_part: nn.Module, config=None) -> None:
    """Remove ``quantization_config`` from the HF config when no parameters are quantized.

    Models loaded from quantized checkpoints (e.g. mxfp4 GPT-OSS) carry a
    ``quantization_config`` on their ``config`` object.  After dequantization
    all parameters are standard floating-point, but the stale config entry would
    still be written to the saved ``config.json``.  This strips it so the output
    checkpoint is a clean bf16 checkpoint, consistent with e.g.
    ``unsloth/gpt-oss-20b-BF16``.
    """
    if config is None:
        config = getattr(model_part, "config", None)
    if config is None or not hasattr(config, "quantization_config"):
        return

    _QUANTIZED_DTYPES = frozenset({torch.uint8, torch.int8})
    if any(p.dtype in _QUANTIZED_DTYPES for p in model_part.parameters()):
        return

    delattr(config, "quantization_config")


def _config_exists(original_model_path: str | None, config_name: str) -> bool:
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


# Symbols some trust_remote_code models import at module load that newer transformers
# removed. Map symbol -> (module, fallback literal) so a consolidated checkpoint reloads
# under a deploy container whose transformers dropped them (e.g. DeciLM/Nemotron-Super
# imports NEED_SETUP_CACHE_CLASSES_MAPPING, gone since transformers 4.57).
_REMOVED_TRANSFORMERS_SYMBOLS = {
    "NEED_SETUP_CACHE_CLASSES_MAPPING": ("transformers.generation.utils", "{}"),
}


def _apply_transformers_compat_guards(py_path: str) -> None:
    """Guard imports of transformers symbols removed in newer versions.

    For each copied ``.py`` that does ``from <module> import ... <symbol> ...`` where
    ``<symbol>`` was removed upstream, insert a preamble defining the symbol on
    ``<module>`` if absent, so the subsequent import resolves. Files that don't
    reference such symbols are left byte-for-byte unchanged.
    """
    try:
        with open(py_path, encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return

    # Seed with symbols already guarded (cross-call idempotency); also dedups multiple
    # import lines of the same symbol within one file.
    guarded = {sym for sym in _REMOVED_TRANSFORMERS_SYMBOLS if f"_nemo_compat_mod.{sym}" in text}

    out: list[str] = []
    changed = False
    for line in text.splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith("from ") and " import " in stripped:
            for symbol, (module, fallback) in _REMOVED_TRANSFORMERS_SYMBOLS.items():
                if symbol not in guarded and symbol in stripped and f"from {module} import" in stripped:
                    indent = line[: len(line) - len(stripped)]
                    out.append(
                        f"{indent}import {module} as _nemo_compat_mod  # noqa\n"
                        f'{indent}if not hasattr(_nemo_compat_mod, "{symbol}"):\n'
                        f"{indent}    _nemo_compat_mod.{symbol} = {fallback}\n"
                        f"{indent}del _nemo_compat_mod\n"
                    )
                    guarded.add(symbol)
                    changed = True
                    break
        out.append(line)

    if changed:
        with open(py_path, "w", encoding="utf-8") as f:
            f.writelines(out)


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
            _apply_transformers_compat_guards(dst_path)
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
