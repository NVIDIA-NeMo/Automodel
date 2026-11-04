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

"""Create the Ministral-3-8B Instruct VLM with BF16 weights and the regular processor."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import logging
import os
import stat
import struct
import tempfile
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from transformers import Mistral3Model, PixtralImageProcessor
    from transformers.processing_utils import ProcessorMixin
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger(__name__)

PROCESSOR_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
REVISION = "main"
REQUIRED_TRANSFORMERS_VERSION = "5.12.1"
MAX_SEQUENCE_LENGTH = 32_768
TEXT_HIDDEN_SIZE = 4_096
VISION_HIDDEN_SIZE = 1_024
TEXT_ROPE_THETA = 1_000_000.0
VISION_ROPE_THETA = 10_000.0
SAFETENSORS_HEADER_MAX_BYTES = 64 * 1024 * 1024
NESTED_LANGUAGE_MODEL_PREFIX = "language_model.model."
BASE_LANGUAGE_MODEL_PREFIX = "language_model."
HEADER_BACKUP_SUFFIX = ".header-backup-language-model-model-prefix"
INDEX_BACKUP_SUFFIX = ".backup-language-model-model-prefix"
EXPECTED_STATIC_OUTPUT_FILES = {
    "config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
}


def _require_mistral_common_absent() -> None:
    """Fail before importing Transformers when mistral-common is installed or importable."""
    try:
        installed_distribution = distribution("mistral-common")
    except PackageNotFoundError:
        installed_distribution = None
    module_spec = importlib.util.find_spec("mistral_common")
    if installed_distribution is not None or module_spec is not None:
        distribution_detail = (
            f"distribution version {installed_distribution.version}"
            if installed_distribution is not None
            else "no distribution metadata"
        )
        module_detail = f"importable from {module_spec.origin}" if module_spec is not None else "not importable"
        raise RuntimeError(
            "mistral-common must be completely absent before this conversion starts; "
            f"found {distribution_detail}, module is {module_detail}. Remove mistral-common and rerun."
        )


def _require_transformers_version() -> None:
    """Require the Transformers version used to validate this conversion."""
    try:
        installed_version = distribution("transformers").version
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"This conversion requires transformers=={REQUIRED_TRANSFORMERS_VERSION}, but Transformers is absent"
        ) from error
    if installed_version != REQUIRED_TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"This conversion requires transformers=={REQUIRED_TRANSFORMERS_VERSION}, got {installed_version}"
        )


def _load_source_json(filename: str) -> dict[str, object]:
    """Download a bounded JSON object from the regular Instruct repository."""
    from huggingface_hub import hf_hub_download

    source_path = Path(hf_hub_download(PROCESSOR_MODEL_ID, filename, revision=REVISION))
    source_size = source_path.stat().st_size
    if source_size <= 0 or source_size > 1_048_576:
        raise ValueError(f"Refusing to load unexpected {filename} size ({source_size} bytes)")
    with source_path.open(encoding="utf-8") as source_file:
        source_config = json.load(source_file)
    if not isinstance(source_config, dict):
        raise TypeError(f"Expected a JSON object in {source_path}")
    return source_config


def _load_source_tokenizer_config() -> dict[str, object]:
    """Load and validate the tokenizer config that the output must preserve."""
    tokenizer_config = _load_source_json("tokenizer_config.json")
    expected = {
        "backend": "tokenizers",
        "tokenizer_class": "TokenizersBackend",
        "processor_class": "PixtralProcessor",
    }
    actual = {name: tokenizer_config.get(name) for name in expected}
    if actual != expected or "chat_template" in tokenizer_config:
        raise ValueError(f"Unexpected source tokenizer config: expected {expected}, got {actual}")
    return tokenizer_config


def _expected_output_tokenizer_config(source_config: dict[str, object]) -> dict[str, object]:
    """Apply the reference checkpoint's sequence-length and padding overrides."""
    output_config = copy.deepcopy(source_config)
    output_config["model_max_length"] = MAX_SEQUENCE_LENGTH
    output_config["padding_side"] = "right"
    return output_config


def _load_source_processor_config() -> dict[str, object]:
    """Load and validate the processor config from the regular Instruct repository."""
    processor_config = _load_source_json("processor_config.json")
    expected = {
        "processor_class": "PixtralProcessor",
        "image_token": "[IMG]",
        "image_break_token": "[IMG_BREAK]",
        "image_end_token": "[IMG_END]",
        "patch_size": 14,
        "spatial_merge_size": 2,
    }
    actual = {name: processor_config.get(name) for name in expected}
    if actual != expected or "chat_template" in processor_config:
        raise ValueError(f"Unexpected source processor config: expected {expected}, got {actual}")
    return processor_config


def _write_json(path: Path, value: dict[str, object]) -> None:
    """Write a JSON object using stable reference-checkpoint formatting."""
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(value, output_file, ensure_ascii=False, indent=2, sort_keys=True)
        output_file.write("\n")


def _load_tokenizer() -> PreTrainedTokenizerBase:
    """Load the tokenizer.json-backed regular Instruct tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        PROCESSOR_MODEL_ID,
        revision=REVISION,
        trust_remote_code=False,
        model_max_length=MAX_SEQUENCE_LENGTH,
        padding_side="right",
    )
    tokenizer.model_max_length = MAX_SEQUENCE_LENGTH
    tokenizer.padding_side = "right"
    tokenizer.chat_template = None
    tokenizer.init_kwargs["model_max_length"] = MAX_SEQUENCE_LENGTH
    tokenizer.init_kwargs["padding_side"] = "right"
    tokenizer.init_kwargs.pop("chat_template", None)
    _validate_tokenizer(tokenizer)
    return tokenizer


def _build_image_processor(source_config: dict[str, object]) -> PixtralImageProcessor:
    """Build the Pixtral image processor from the regular Instruct config."""
    from transformers import PixtralImageProcessor

    image_config = source_config.get("image_processor")
    if not isinstance(image_config, dict):
        raise ValueError("Missing nested image_processor config")
    if image_config.get("image_processor_type") != "PixtralImageProcessorFast":
        raise ValueError(f"Unexpected image processor type: {image_config.get('image_processor_type')!r}")
    image_processor = PixtralImageProcessor.from_dict(image_config)
    patch_size = image_processor.patch_size
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    elif isinstance(patch_size, dict):
        patch_height, patch_width = patch_size.get("height"), patch_size.get("width")
    else:
        patch_height = getattr(patch_size, "height", None)
        patch_width = getattr(patch_size, "width", None)
    if (patch_height, patch_width) != (14, 14):
        raise ValueError(f"Expected image processor patch size 14, got {image_processor.patch_size}")
    image_size = image_processor.size
    if isinstance(image_size, int):
        longest_edge = image_size
    elif isinstance(image_size, dict):
        longest_edge = image_size.get("longest_edge")
    else:
        longest_edge = getattr(image_size, "longest_edge", None)
    if longest_edge != 1540:
        raise ValueError(f"Expected longest edge 1540, got {image_processor.size}")
    return image_processor


def _validate_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Validate the tokenizer contract required by the composed VLM."""
    if type(tokenizer).__name__ != "TokenizersBackend":
        raise TypeError(f"Expected TokenizersBackend, got {type(tokenizer).__name__}")
    if tokenizer.vocab_size != 131_072 or len(tokenizer) != 131_072:
        raise ValueError(f"Expected vocabulary size 131072, got {tokenizer.vocab_size}/{len(tokenizer)}")
    expected_ids = {"<unk>": 0, "<s>": 1, "</s>": 2, "[IMG]": 10, "<pad>": 11, "[IMG_BREAK]": 12, "[IMG_END]": 13}
    actual_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in expected_ids}
    if actual_ids != expected_ids:
        raise ValueError(f"Unexpected special-token mapping: expected {expected_ids}, got {actual_ids}")
    if tokenizer.model_max_length != MAX_SEQUENCE_LENGTH or tokenizer.padding_side != "right":
        raise ValueError("Tokenizer must use a 32768-token limit and right padding")


def _validate_model(model: Mistral3Model) -> None:
    """Validate the base VLM architecture with no causal language-modeling head."""
    from transformers import Ministral3Model, Mistral3Model

    if not isinstance(model, Mistral3Model):
        raise TypeError(f"Expected Mistral3Model, got {type(model).__name__}")
    if not isinstance(model.language_model, Ministral3Model):
        raise TypeError(f"Expected Ministral3Model, got {type(model.language_model).__name__}")
    if model.language_model.config is not model.config.text_config:
        raise ValueError("Outer text_config and language_model.config must share identity")
    expected_outer = {
        "architectures": ["Mistral3Model"],
        "image_token_index": 10,
        "multimodal_projector_bias": False,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": 2,
        "tie_word_embeddings": False,
        "vision_feature_layer": -1,
    }
    actual_outer = {name: getattr(model.config, name) for name in expected_outer}
    if actual_outer != expected_outer:
        raise ValueError(f"Unexpected outer config: expected {expected_outer}, got {actual_outer}")
    if model.config.rope_theta != TEXT_ROPE_THETA:
        raise ValueError(f"Unexpected outer rope_theta: {model.config.rope_theta}")
    text_config = model.config.text_config
    expected_text = {
        "hidden_size": TEXT_HIDDEN_SIZE,
        "intermediate_size": 14_336,
        "max_position_embeddings": 262_144,
        "num_attention_heads": 32,
        "num_hidden_layers": 34,
        "num_key_value_heads": 8,
        "vocab_size": 131_072,
    }
    actual_text = {name: getattr(text_config, name) for name in expected_text}
    if actual_text != expected_text:
        raise ValueError(f"Unexpected text config: expected {expected_text}, got {actual_text}")
    if text_config.rope_parameters["rope_theta"] != TEXT_ROPE_THETA:
        raise ValueError(f"Unexpected text rope_theta: {text_config.rope_parameters['rope_theta']}")
    vision_config = model.config.vision_config
    expected_vision = {
        "hidden_size": VISION_HIDDEN_SIZE,
        "image_size": 1_540,
        "intermediate_size": 4_096,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "patch_size": 14,
    }
    actual_vision = {name: getattr(vision_config, name) for name in expected_vision}
    if actual_vision != expected_vision:
        raise ValueError(f"Unexpected vision config: expected {expected_vision}, got {actual_vision}")
    if vision_config.rope_parameters["rope_theta"] != VISION_ROPE_THETA:
        raise ValueError(f"Unexpected vision rope_theta: {vision_config.rope_parameters['rope_theta']}")
    if getattr(model.config, "quantization_config", None) is not None:
        raise ValueError("The BF16 checkpoint must not retain quantization_config")
    if hasattr(model, "lm_head"):
        raise ValueError("The base Mistral3Model must not contain a causal language-modeling head")
    non_bf16 = [
        f"{name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    if non_bf16:
        raise TypeError("All floating-point parameters must be BF16: " + ", ".join(non_bf16[:20]))


def _tensor_digest(tensor: torch.Tensor, *, chunk_elements: int = 1_048_576) -> str:
    """Hash a contiguous tensor in bounded CPU chunks.

    Args:
        tensor: Contiguous tensor of shape [...], with arbitrary dimensions.
        chunk_elements: Maximum source elements copied to CPU per chunk.

    Returns:
        SHA-256 digest covering dtype, shape, and exact storage bytes.
    """
    if chunk_elements <= 0 or not tensor.is_contiguous():
        raise ValueError("Tensor hashing requires positive chunks and contiguous storage")
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    flattened = tensor.detach().view(-1)
    for start in range(0, flattened.numel(), chunk_elements):
        chunk = flattened[start : start + chunk_elements].to(device="cpu", copy=True)
        digest.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_fingerprints(model: nn.Module) -> dict[str, str]:
    """Fingerprint every parameter and persistent buffer in bounded CPU chunks."""
    return {name: _tensor_digest(tensor) for name, tensor in model.state_dict(keep_vars=True).items()}


def _config_snapshot(model: Mistral3Model) -> dict[str, object]:
    """Capture serialized config values except checkpoint-location metadata."""

    def _normalize(value: object) -> object:
        if isinstance(value, dict):
            return {key: _normalize(item) for key, item in value.items() if key != "_name_or_path"}
        if isinstance(value, list):
            return [_normalize(item) for item in value]
        return value

    snapshot = _normalize(model.config.to_dict())
    if not isinstance(snapshot, dict):
        raise TypeError("Expected model config to serialize as a JSON object")
    return snapshot


def _validate_multimodal_forward(
    model: Mistral3Model,
    processor: ProcessorMixin,
    device: torch.device,
) -> None:
    """Run a tiny image-first base-model forward.

    Args:
        model: BF16 base VLM producing hidden states of shape [batch, sequence, hidden].
        processor: Pixtral processor producing padded image-text tensors.
        device: CUDA device used for validation.
    """
    batch = processor(
        images=[torch.zeros((3, 56, 56), dtype=torch.uint8)],
        text=["[IMG]validation"],
        padding=True,
        return_tensors="pt",
    )
    required = {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
    if missing := sorted(required - batch.keys()):
        raise KeyError(f"Processor omitted required multimodal inputs: {missing}")
    inputs = {
        name: value.to(device=device, dtype=torch.bfloat16) if value.is_floating_point() else value.to(device)
        for name, value in batch.items()
    }
    model.eval()
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False, return_dict=True)
    expected_shape = (1, inputs["input_ids"].shape[1], TEXT_HIDDEN_SIZE)
    if tuple(outputs.last_hidden_state.shape) != expected_shape or not torch.isfinite(outputs.last_hidden_state).all():
        raise ValueError(f"Invalid hidden states: shape={tuple(outputs.last_hidden_state.shape)}")


def _prepare_output_directory(output_path: Path) -> None:
    """Create an empty setgid output directory without overwriting a checkpoint."""
    if output_path.is_symlink():
        raise ValueError(f"Refusing to write through symlink {output_path}")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise FileExistsError(f"Refusing to overwrite non-empty output {output_path}")
    output_path.mkdir(mode=0o2775, parents=True, exist_ok=True)
    output_path.chmod(0o2775)


def _make_group_writable(root: Path) -> None:
    """Grant group read/write access and directory traversal under ``root``."""
    for path in (root, *root.rglob("*")):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode | (stat.S_IRGRP | stat.S_IWGRP | (stat.S_IXGRP | stat.S_ISGID if path.is_dir() else 0)))


def _read_safetensors_header(path: Path) -> tuple[bytes, dict[str, object]]:
    """Read a bounded safetensors header without loading tensor payloads."""
    with path.open("rb") as checkpoint_file:
        length_bytes = checkpoint_file.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"Truncated safetensors length prefix in {path}")
        header_length = struct.unpack("<Q", length_bytes)[0]
        if header_length == 0 or header_length > SAFETENSORS_HEADER_MAX_BYTES:
            raise ValueError(f"Invalid safetensors header length {header_length} in {path}")
        header_bytes = checkpoint_file.read(header_length)
    if len(header_bytes) != header_length:
        raise ValueError(f"Truncated safetensors header in {path}")
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise TypeError(f"Expected a JSON object in the safetensors header for {path}")
    return length_bytes + header_bytes, header


def _write_exact_backup(path: Path, content: bytes) -> None:
    """Create an exact backup, or validate an existing one."""
    try:
        file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Existing backup does not match the source: {path}")
        return
    with os.fdopen(file_descriptor, "wb") as backup_file:
        backup_file.write(content)
        backup_file.flush()
        os.fsync(backup_file.fileno())


def _renamed_language_model_key(name: str) -> str:
    """Map ``language_model.model.*`` to the stock base-model namespace."""
    if name.startswith(NESTED_LANGUAGE_MODEL_PREFIX):
        return BASE_LANGUAGE_MODEL_PREFIX + name.removeprefix(NESTED_LANGUAGE_MODEL_PREFIX)
    return name


def _normalize_safetensors_header(path: Path) -> tuple[int, set[str]]:
    """Flatten language-model keys while preserving tensor payload bytes."""
    original_size = path.stat().st_size
    original_header_blob, original_header = _read_safetensors_header(path)
    header_length = len(original_header_blob) - 8
    normalized_header: dict[str, object] = {}
    renamed_count = 0
    for name, descriptor in original_header.items():
        normalized_name = name if name == "__metadata__" else _renamed_language_model_key(name)
        if normalized_name in normalized_header:
            raise ValueError(f"Normalizing {name!r} would collide with {normalized_name!r} in {path}")
        normalized_header[normalized_name] = descriptor
        renamed_count += normalized_name != name

    if renamed_count:
        serialized_header = json.dumps(normalized_header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        if len(serialized_header) > header_length:
            raise ValueError(f"Normalized header grew from {header_length} to {len(serialized_header)} bytes in {path}")
        padded_header = serialized_header + b" " * (header_length - len(serialized_header))
        _write_exact_backup(Path(f"{path}{HEADER_BACKUP_SUFFIX}"), original_header_blob)
        with path.open("r+b") as checkpoint_file:
            checkpoint_file.write(struct.pack("<Q", header_length))
            checkpoint_file.write(padded_header)
            checkpoint_file.flush()
            os.fsync(checkpoint_file.fileno())

    if path.stat().st_size != original_size:
        raise ValueError(f"Header normalization changed checkpoint size for {path}")
    _, verified_header = _read_safetensors_header(path)
    if verified_header != normalized_header:
        raise ValueError(f"Header verification failed after normalizing {path}")
    tensor_names = {name for name in normalized_header if name != "__metadata__"}
    if any(name.startswith(NESTED_LANGUAGE_MODEL_PREFIX) for name in tensor_names):
        raise ValueError(f"Nested language-model keys remain in {path}")
    return renamed_count, tensor_names


def _replace_json_file(path: Path, content: bytes) -> None:
    """Atomically replace a JSON file while preserving its permissions."""
    original_mode = stat.S_IMODE(path.stat().st_mode)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as temporary_file:
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.chmod(original_mode)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _normalize_saved_mistral3_state_dict(output_path: Path) -> int:
    """Flatten saved text keys in every shard header and the safetensors index."""
    index_path = output_path / "model.safetensors.index.json"
    index_bytes = index_path.read_bytes()
    if len(index_bytes) > SAFETENSORS_HEADER_MAX_BYTES:
        raise ValueError(f"Unexpectedly large safetensors index: {index_path}")
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError(f"Expected a safetensors index with a weight_map in {index_path}")
    weight_map = index["weight_map"]
    normalized_weight_map: dict[str, object] = {}
    index_renamed_count = 0
    for name, shard_name in weight_map.items():
        normalized_name = _renamed_language_model_key(name)
        if normalized_name in normalized_weight_map:
            raise ValueError(f"Normalizing {name!r} would collide with {normalized_name!r}")
        normalized_weight_map[normalized_name] = shard_name
        index_renamed_count += normalized_name != name

    shard_names = set(weight_map.values())
    if not all(isinstance(shard_name, str) for shard_name in shard_names):
        raise TypeError(f"Safetensors index contains a non-string shard name in {index_path}")
    all_tensor_names: set[str] = set()
    header_renamed_count = 0
    for shard_name in sorted(shard_names):
        shard_path = output_path / shard_name
        if shard_path.parent != output_path or not shard_path.is_file():
            raise ValueError(f"Invalid or missing safetensors shard {shard_name!r}")
        shard_renamed_count, tensor_names = _normalize_safetensors_header(shard_path)
        if duplicates := all_tensor_names & tensor_names:
            raise ValueError(f"Tensor keys occur in multiple shards: {sorted(duplicates)[:5]}")
        all_tensor_names.update(tensor_names)
        header_renamed_count += shard_renamed_count
    if header_renamed_count != index_renamed_count or all_tensor_names != set(normalized_weight_map):
        raise ValueError("Safetensors headers and index disagree after key normalization")
    if any("lm_head" in name for name in normalized_weight_map):
        raise ValueError("The saved base-model checkpoint unexpectedly contains LM-head weights")
    if index_renamed_count:
        _write_exact_backup(Path(f"{index_path}{INDEX_BACKUP_SUFFIX}"), index_bytes)
        index["weight_map"] = normalized_weight_map
        _replace_json_file(index_path, json.dumps(index, ensure_ascii=False, indent=2).encode() + b"\n")
    return header_renamed_count


def _prune_and_validate_output_files(output_path: Path) -> set[str]:
    """Keep exactly the reference checkpoint's file layout."""
    shard_names = {path.name for path in output_path.glob("model-*-of-*.safetensors") if path.is_file()}
    if not shard_names:
        raise FileNotFoundError(f"Expected sharded safetensors in {output_path}")
    expected = EXPECTED_STATIC_OUTPUT_FILES | shard_names
    for path in output_path.iterdir():
        if path.is_dir() or path.is_symlink():
            raise ValueError(f"Unexpected non-file checkpoint artifact: {path}")
        if path.name not in expected:
            path.unlink()
    actual = {path.name for path in output_path.iterdir()}
    if actual != expected:
        raise ValueError(
            f"Unexpected output layout: missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    return actual


def _validate_saved_processor_files(
    output_path: Path,
    source_tokenizer_config: dict[str, object],
    source_processor_config: dict[str, object],
) -> None:
    """Validate exact tokenizer and processor files in the output."""
    with (output_path / "tokenizer_config.json").open(encoding="utf-8") as tokenizer_file:
        tokenizer_config = json.load(tokenizer_file)
    if tokenizer_config != _expected_output_tokenizer_config(source_tokenizer_config):
        raise ValueError("Saved tokenizer config differs from the regular Instruct source plus overrides")
    with (output_path / "processor_config.json").open(encoding="utf-8") as processor_file:
        processor_config = json.load(processor_file)
    if processor_config != source_processor_config:
        raise ValueError("Saved processor config differs from the regular Instruct source")
    if not (output_path / "tokenizer.json").is_file() or (output_path / "tokenizer.json").stat().st_size == 0:
        raise FileNotFoundError("Missing non-empty tokenizer.json")


def compose_checkpoint(output_path: Path) -> None:
    """Compose, save, reload, and validate the requested Hugging Face checkpoint."""
    _require_mistral_common_absent()
    _require_transformers_version()
    from transformers import AutoModel, AutoProcessor, AutoTokenizer, PixtralProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for bounded-memory checkpoint composition")
    output_path = output_path.expanduser()
    if output_path.is_symlink():
        raise ValueError(f"Refusing to write through symlink {output_path}")
    output_path = output_path.resolve()
    previous_umask = os.umask(0o002)
    prepared = False
    try:
        _prepare_output_directory(output_path)
        prepared = True
        device_index = torch.cuda.current_device()
        device_map = {"": device_index}
        tokenizer_config = _load_source_tokenizer_config()
        processor_config = _load_source_processor_config()
        tokenizer = _load_tokenizer()
        image_processor = _build_image_processor(processor_config)

        logger.info("Loading BF16 base VLM without the causal LM head from %s", MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            revision=REVISION,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        model.config.architectures = ["Mistral3Model"]
        model.config.dtype = torch.bfloat16
        model.config.rope_theta = TEXT_ROPE_THETA
        _validate_model(model)
        expected_config = _config_snapshot(model)
        expected_fingerprints = _state_fingerprints(model)
        processor = PixtralProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=model.config.vision_config.patch_size,
            spatial_merge_size=model.config.spatial_merge_size,
            chat_template=None,
            image_token="[IMG]",
            image_break_token="[IMG_BREAK]",
            image_end_token="[IMG_END]",
        )

        model.save_pretrained(output_path, safe_serialization=True, max_shard_size="4GB")
        renamed_key_count = _normalize_saved_mistral3_state_dict(output_path)
        logger.info("Flattened %d saved language-model tensor keys", renamed_key_count)
        processor.save_pretrained(output_path)
        _write_json(output_path / "tokenizer_config.json", _expected_output_tokenizer_config(tokenizer_config))
        _write_json(output_path / "processor_config.json", processor_config)
        _prune_and_validate_output_files(output_path)
        _validate_saved_processor_files(output_path, tokenizer_config, processor_config)
        _make_group_writable(output_path)
        del processor, image_processor, tokenizer, model
        gc.collect()
        torch.cuda.empty_cache()

        reloaded_model = AutoModel.from_pretrained(
            output_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        _validate_model(reloaded_model)
        if _config_snapshot(reloaded_model) != expected_config:
            raise ValueError("Save/reload changed model configuration values")
        actual_fingerprints = _state_fingerprints(reloaded_model)
        if actual_fingerprints != expected_fingerprints:
            mismatches = sorted(
                name
                for name in set(expected_fingerprints) | set(actual_fingerprints)
                if actual_fingerprints.get(name) != expected_fingerprints.get(name)
            )
            raise ValueError(f"Save/reload changed state-tensor bytes for: {mismatches}")
        reloaded_tokenizer = AutoTokenizer.from_pretrained(output_path, local_files_only=True, trust_remote_code=False)
        _validate_tokenizer(reloaded_tokenizer)
        reloaded_processor = AutoProcessor.from_pretrained(output_path, local_files_only=True, trust_remote_code=False)
        if not isinstance(reloaded_processor, PixtralProcessor):
            raise TypeError(f"Expected PixtralProcessor, got {type(reloaded_processor).__name__}")
        _validate_tokenizer(reloaded_processor.tokenizer)
        if reloaded_processor.image_token_ids != [10, 12, 13]:
            raise ValueError(f"Expected image token IDs [10, 12, 13], got {reloaded_processor.image_token_ids}")
        _validate_multimodal_forward(reloaded_model, reloaded_processor, torch.device("cuda", device_index))
        _prune_and_validate_output_files(output_path)
        logger.info("Checkpoint composition and validation completed successfully")
    finally:
        os.umask(previous_umask)
        if prepared and output_path.exists() and output_path.is_dir():
            _make_group_writable(output_path)


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_path", type=Path, help="new directory for the composed checkpoint")
    return parser.parse_args()


def main() -> int:
    """Run checkpoint composition from the command line."""
    compose_checkpoint(_parse_args().output_path)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
