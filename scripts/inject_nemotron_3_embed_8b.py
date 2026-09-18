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

"""Create a BF16 Pixtral VLM with the Nemotron-3-Embed-8B text backbone."""

from __future__ import annotations

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


logger = logging.getLogger(__name__)

TOKENIZER_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512"
VISION_MODEL_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
TEXT_MODEL_ID = "nvidia/Nemotron-3-Embed-8B-BF16"
REVISION = "main"
REQUIRED_TRANSFORMERS_VERSION = "5.12.1"
OUTPUT_PATH = Path(
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/"
    "nemotron_4_embed_vl/pixtral_nemotron_3_embed_8B"
)
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


def _load_source_tokenizer_config() -> dict[str, object]:
    """Download and validate the tokenizer config that the output must preserve."""
    from huggingface_hub import hf_hub_download

    tokenizer_config_path = Path(
        hf_hub_download(
            repo_id=TOKENIZER_MODEL_ID,
            filename="tokenizer_config.json",
            revision=REVISION,
        )
    )
    tokenizer_config_size = tokenizer_config_path.stat().st_size
    if tokenizer_config_size > 1_048_576:
        raise ValueError(f"Refusing to load unexpectedly large tokenizer config ({tokenizer_config_size} bytes)")
    with tokenizer_config_path.open(encoding="utf-8") as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
    if not isinstance(tokenizer_config, dict):
        raise TypeError(f"Expected a JSON object in {tokenizer_config_path}")

    expected_source_values = {
        "backend": "tokenizers",
        "tokenizer_class": "TokenizersBackend",
        "processor_class": "PixtralProcessor",
    }
    actual_source_values = {name: tokenizer_config.get(name) for name in expected_source_values}
    if actual_source_values != expected_source_values:
        raise ValueError(
            f"Unexpected source tokenizer config: expected {expected_source_values}, got {actual_source_values}"
        )
    if "chat_template" in tokenizer_config:
        raise ValueError("The source tokenizer_config.json unexpectedly contains chat_template")
    return tokenizer_config


def _expected_output_tokenizer_config(source_tokenizer_config: dict[str, object]) -> dict[str, object]:
    """Apply only the requested context-limit and padding overrides to the source config."""
    expected_tokenizer_config = copy.deepcopy(source_tokenizer_config)
    expected_tokenizer_config["model_max_length"] = MAX_SEQUENCE_LENGTH
    expected_tokenizer_config["padding_side"] = "right"
    return expected_tokenizer_config


def _write_output_tokenizer_config(
    output_path: Path,
    source_tokenizer_config: dict[str, object],
) -> None:
    """Write the source tokenizer config with only the two requested overrides."""
    tokenizer_config_path = output_path / "tokenizer_config.json"
    expected_tokenizer_config = _expected_output_tokenizer_config(source_tokenizer_config)
    with tokenizer_config_path.open("w", encoding="utf-8") as tokenizer_config_file:
        json.dump(expected_tokenizer_config, tokenizer_config_file, ensure_ascii=False, indent=2, sort_keys=True)
        tokenizer_config_file.write("\n")


def _load_source_processor_config() -> dict[str, object]:
    """Download and validate the BF16 VLM processor configuration."""
    from huggingface_hub import hf_hub_download

    processor_config_path = Path(
        hf_hub_download(
            repo_id=VISION_MODEL_ID,
            filename="processor_config.json",
            revision=REVISION,
        )
    )
    processor_config_size = processor_config_path.stat().st_size
    if processor_config_size > 1_048_576:
        raise ValueError(f"Refusing to load unexpectedly large processor config ({processor_config_size} bytes)")
    with processor_config_path.open(encoding="utf-8") as processor_config_file:
        processor_config = json.load(processor_config_file)
    if not isinstance(processor_config, dict):
        raise TypeError(f"Expected a JSON object in {processor_config_path}")

    expected_source_values = {
        "processor_class": "PixtralProcessor",
        "image_token": "[IMG]",
        "image_break_token": "[IMG_BREAK]",
        "image_end_token": "[IMG_END]",
        "patch_size": 14,
        "spatial_merge_size": 2,
    }
    actual_source_values = {name: processor_config.get(name) for name in expected_source_values}
    if actual_source_values != expected_source_values:
        raise ValueError(
            f"Unexpected source processor config: expected {expected_source_values}, got {actual_source_values}"
        )
    if "chat_template" in processor_config:
        raise ValueError("The source processor_config.json unexpectedly contains chat_template")
    return processor_config


def _write_output_processor_config(
    output_path: Path,
    source_processor_config: dict[str, object],
) -> None:
    """Preserve the complete processor configuration from the BF16 VLM."""
    processor_config_path = output_path / "processor_config.json"
    with processor_config_path.open("w", encoding="utf-8") as processor_config_file:
        json.dump(source_processor_config, processor_config_file, ensure_ascii=False, indent=2, sort_keys=True)
        processor_config_file.write("\n")


def _load_tokenizer() -> PreTrainedTokenizerBase:
    """Load and validate the tokenizer.json-backed Ministral tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_MODEL_ID,
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


def _build_image_processor(source_processor_config: dict[str, object]) -> PixtralImageProcessor:
    """Build the Transformers 5.12.1 Pixtral image processor from the source config."""
    from transformers import PixtralImageProcessor

    image_processor_config = source_processor_config.get("image_processor")
    if not isinstance(image_processor_config, dict):
        raise ValueError("Missing nested image_processor config in source processor_config.json")
    if image_processor_config.get("image_processor_type") != "PixtralImageProcessorFast":
        raise ValueError(
            "Expected image_processor_type='PixtralImageProcessorFast', got "
            f"{image_processor_config.get('image_processor_type')!r}"
        )
    image_processor = PixtralImageProcessor.from_dict(image_processor_config)
    patch_size = image_processor.patch_size
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    elif isinstance(patch_size, dict):
        patch_height = patch_size.get("height")
        patch_width = patch_size.get("width")
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
        raise ValueError(f"Expected image processor longest edge 1540, got {image_processor.size}")
    return image_processor


def _validate_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Validate the tokenizer contract required by the composed VLM."""
    if type(tokenizer).__name__ != "TokenizersBackend":
        raise TypeError(
            f"Expected tokenizer class TokenizersBackend from {TOKENIZER_MODEL_ID}, got {type(tokenizer).__name__}"
        )
    if tokenizer.vocab_size != 131_072 or len(tokenizer) != 131_072:
        raise ValueError(
            f"Expected a 131072-token vocabulary, got vocab_size={tokenizer.vocab_size} and len={len(tokenizer)}"
        )

    expected_token_ids = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "[IMG]": 10,
        "<pad>": 11,
        "[IMG_BREAK]": 12,
        "[IMG_END]": 13,
        "<SPECIAL_34>": 34,
        "<SPECIAL_35>": 35,
    }
    actual_token_ids = {token: tokenizer.convert_tokens_to_ids(token) for token in expected_token_ids}
    if actual_token_ids != expected_token_ids:
        raise ValueError(f"Unexpected special-token mapping: expected {expected_token_ids}, got {actual_token_ids}")
    if tokenizer.model_max_length != MAX_SEQUENCE_LENGTH:
        raise ValueError(f"Expected tokenizer model_max_length={MAX_SEQUENCE_LENGTH}, got {tokenizer.model_max_length}")
    if tokenizer.padding_side != "right":
        raise ValueError(f"Expected right padding, got {tokenizer.padding_side!r}")


def _validate_composed_model(model: Mistral3Model) -> None:
    """Validate architecture, config, shape, and dtype invariants."""
    from transformers import Ministral3Model

    if not isinstance(model.language_model, Ministral3Model):
        raise TypeError(f"Expected a Ministral3Model text backbone, got {type(model.language_model).__name__}")
    text_config = model.config.text_config
    if model.language_model.config is not text_config:
        raise ValueError("Outer text_config and language_model.config must reference the same config object")
    expected_text_values = {
        "architectures": ["Ministral3Model"],
        "attention_dropout": 0.0,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": TEXT_HIDDEN_SIZE,
        "intermediate_size": 14_336,
        "is_causal": False,
        "num_hidden_layers": 34,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "pad_token_id": 11,
        "pooling": "avg",
        "rms_norm_eps": 1e-5,
        "sliding_window": None,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 131_072,
    }
    actual_text_values = {name: getattr(text_config, name) for name in expected_text_values}
    if actual_text_values != expected_text_values:
        raise ValueError(f"Unexpected text config: expected {expected_text_values}, got {actual_text_values}")
    if text_config.max_position_embeddings != MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected text max_position_embeddings={MAX_SEQUENCE_LENGTH}, got {text_config.max_position_embeddings}"
        )
    expected_outer_values = {
        "architectures": ["Mistral3Model"],
        "image_token_index": 10,
        "multimodal_projector_bias": False,
        "projector_hidden_act": "gelu",
        "spatial_merge_size": 2,
        "tie_word_embeddings": False,
        "vision_feature_layer": -1,
    }
    actual_outer_values = {name: getattr(model.config, name) for name in expected_outer_values}
    if actual_outer_values != expected_outer_values:
        raise ValueError(f"Unexpected outer config: expected {expected_outer_values}, got {actual_outer_values}")
    if model.config.rope_theta != TEXT_ROPE_THETA:
        raise ValueError(f"Expected outer rope_theta={TEXT_ROPE_THETA} for vLLM, got {model.config.rope_theta}")
    if text_config.rope_parameters["rope_theta"] != TEXT_ROPE_THETA:
        raise ValueError(
            f"Unexpected text RoPE theta: expected {TEXT_ROPE_THETA}, got {text_config.rope_parameters['rope_theta']}"
        )
    if model.config.vision_config.rope_parameters["rope_theta"] != VISION_ROPE_THETA:
        raise ValueError(
            "Unexpected vision RoPE theta: "
            f"expected {VISION_ROPE_THETA}, got "
            f"{model.config.vision_config.rope_parameters['rope_theta']}"
        )
    expected_vision_values = {
        "attention_dropout": 0.0,
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": VISION_HIDDEN_SIZE,
        "image_size": 1_540,
        "intermediate_size": 4_096,
        "num_attention_heads": 16,
        "num_channels": 3,
        "num_hidden_layers": 24,
        "patch_size": 14,
    }
    actual_vision_values = {name: getattr(model.config.vision_config, name) for name in expected_vision_values}
    if actual_vision_values != expected_vision_values:
        raise ValueError(f"Unexpected vision config: expected {expected_vision_values}, got {actual_vision_values}")
    if getattr(model.config, "quantization_config", None) is not None:
        raise ValueError("The composed BF16 checkpoint must not retain quantization_config")
    if getattr(text_config, "quantization_config", None) is not None:
        raise ValueError("The injected BF16 text config must not retain quantization_config")
    if hasattr(model, "lm_head"):
        raise ValueError("The embedding VLM must not contain a causal language-modeling head")
    projector_norm_eps = model.multi_modal_projector.norm.variance_epsilon
    if projector_norm_eps != text_config.rms_norm_eps:
        raise ValueError(
            f"Projector norm eps {projector_norm_eps} does not match injected text rms_norm_eps "
            f"{text_config.rms_norm_eps}"
        )

    expected_projector_shapes = {
        "linear_1": (TEXT_HIDDEN_SIZE, VISION_HIDDEN_SIZE),
        "linear_2": (TEXT_HIDDEN_SIZE, TEXT_HIDDEN_SIZE),
    }
    actual_projector_shapes = {
        "linear_1": tuple(model.multi_modal_projector.linear_1.weight.shape),
        "linear_2": tuple(model.multi_modal_projector.linear_2.weight.shape),
    }
    if actual_projector_shapes != expected_projector_shapes:
        raise ValueError(
            f"Unexpected projector shapes: expected {expected_projector_shapes}, got {actual_projector_shapes}"
        )
    if model.multi_modal_projector.linear_1.bias is not None or model.multi_modal_projector.linear_2.bias is not None:
        raise ValueError("Expected the BF16 source multimodal projector to remain bias-free")

    non_bf16_parameters = [
        f"{name}={parameter.dtype}"
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16
    ]
    if non_bf16_parameters:
        raise TypeError(
            "All floating-point model parameters must be BF16; mismatches: " + ", ".join(non_bf16_parameters[:20])
        )


def _tensor_digest(tensor: torch.Tensor, *, chunk_elements: int = 1_048_576) -> str:
    """Hash a tensor without staging its complete storage in system memory.

    Args:
        tensor: Contiguous tensor of shape [...], with arbitrary dimensions. The
            tensor is read without mutation and transferred to CPU in bounded
            flattened chunks.
        chunk_elements: Maximum number of source tensor elements copied to CPU
            per chunk.

    Returns:
        SHA-256 digest covering the tensor dtype, shape, and exact storage bytes.
    """
    if chunk_elements <= 0:
        raise ValueError(f"chunk_elements must be positive, got {chunk_elements}")
    if not tensor.is_contiguous():
        raise ValueError("Tensor fingerprinting requires contiguous storage")

    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    flattened = tensor.detach().view(-1)
    for start in range(0, flattened.numel(), chunk_elements):
        chunk = flattened[start : start + chunk_elements].to(device="cpu", copy=True)
        digest.update(chunk.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _state_fingerprints(model: nn.Module) -> dict[str, str]:
    """Fingerprint every parameter and persistent buffer in bounded CPU chunks."""
    state = model.state_dict(keep_vars=True)
    return {name: _tensor_digest(tensor) for name, tensor in state.items()}


def _normalize_config_value(value: object) -> object:
    """Remove only non-serialized checkpoint-location metadata from a config value."""
    if isinstance(value, dict):
        return {
            key: _normalize_config_value(nested_value) for key, nested_value in value.items() if key != "_name_or_path"
        }
    if isinstance(value, list):
        return [_normalize_config_value(item) for item in value]
    return value


def _config_snapshot(model: Mistral3Model) -> dict[str, object]:
    """Capture all serializable model configuration values except checkpoint location."""
    snapshot = _normalize_config_value(model.config.to_dict())
    if not isinstance(snapshot, dict):
        raise TypeError(f"Expected model config to serialize as a JSON object, got {type(snapshot).__name__}")
    return snapshot


def _validate_multimodal_forward(
    model: Mistral3Model,
    processor: ProcessorMixin,
    device: torch.device,
) -> None:
    """Run a tiny image-first, right-padded image-text batch through the reloaded VLM."""
    padding_probe = processor.tokenizer(
        ["short", "a deliberately longer validation sequence"],
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    padding_mask = padding_probe["attention_mask"]
    sequence_lengths = padding_mask.sum(dim=-1)
    if sequence_lengths.min() == sequence_lengths.max():
        raise ValueError(f"Padding probe unexpectedly produced equal-length sequences: {sequence_lengths.tolist()}")
    shorter_index = int(sequence_lengths.argmin().item())
    shorter_mask = padding_mask[shorter_index]
    first_padding_position = int(sequence_lengths[shorter_index].item())
    if first_padding_position >= shorter_mask.numel():
        raise ValueError(f"Padding probe did not produce padding: attention_mask={padding_mask.tolist()}")
    if not torch.all(shorter_mask[:first_padding_position] == 1) or not torch.all(
        shorter_mask[first_padding_position:] == 0
    ):
        raise ValueError(f"Tokenizer did not apply right padding: attention_mask={padding_mask.tolist()}")

    image = torch.zeros((3, 56, 56), dtype=torch.uint8)
    batch = processor(
        images=[image],
        text=["[IMG]validation"],
        padding=True,
        return_tensors="pt",
    )
    required_inputs = {"input_ids", "attention_mask", "pixel_values", "image_sizes"}
    missing_inputs = sorted(required_inputs - batch.keys())
    if missing_inputs:
        raise KeyError(f"Processor omitted required multimodal inputs: {missing_inputs}")

    input_ids = batch["input_ids"][0]
    image_token_mask = (input_ids == 10) | (input_ids == 12) | (input_ids == 13)
    image_positions = image_token_mask.nonzero(as_tuple=False).flatten()
    if image_positions.numel() == 0:
        raise ValueError("Processor did not expand the leading [IMG] placeholder")
    prefix = input_ids[: int(image_positions[0].item())]
    bos_token_id = processor.tokenizer.bos_token_id
    if prefix.numel() > 0 and (bos_token_id is None or not torch.all(prefix == bos_token_id)):
        raise ValueError(f"Non-image content preceded the image placeholder: token IDs {prefix.tolist()}")

    model_inputs: dict[str, torch.Tensor] = {}
    for name, value in batch.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected processor output {name!r} to be a tensor, got {type(value).__name__}")
        if value.is_floating_point():
            model_inputs[name] = value.to(device=device, dtype=torch.bfloat16)
        else:
            model_inputs[name] = value.to(device=device)

    model.eval()
    with torch.inference_mode():
        outputs = model(**model_inputs, use_cache=False, return_dict=True)
    hidden_states = outputs.last_hidden_state
    expected_shape = (1, model_inputs["input_ids"].shape[1], TEXT_HIDDEN_SIZE)
    if tuple(hidden_states.shape) != expected_shape:
        raise ValueError(f"Expected multimodal hidden-state shape {expected_shape}, got {tuple(hidden_states.shape)}")
    if not torch.isfinite(hidden_states).all():
        raise ValueError("Multimodal validation forward produced non-finite hidden states")


def _prepare_output_directory(output_path: Path) -> None:
    """Create an empty setgid output directory without overwriting a checkpoint."""
    if output_path.is_symlink():
        raise ValueError(f"Refusing to write a checkpoint through symlink {output_path}")
    if output_path.exists():
        if not output_path.is_dir():
            raise NotADirectoryError(f"Output path exists but is not a directory: {output_path}")
        if any(output_path.iterdir()):
            raise FileExistsError(f"Refusing to mix checkpoint files into non-empty directory {output_path}")
    else:
        output_path.mkdir(mode=0o2775, parents=True)
    output_path.chmod(0o2775)


def _make_group_writable(root: Path) -> None:
    """Grant group read/write access and directory traversal under ``root``."""

    def _apply_permissions(path: Path) -> None:
        if path.is_symlink():
            return
        current_mode = stat.S_IMODE(path.stat().st_mode)
        if path.is_dir():
            path.chmod(current_mode | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_ISGID)
        else:
            path.chmod(current_mode | stat.S_IRGRP | stat.S_IWGRP)

    _apply_permissions(root)
    for path in root.rglob("*"):
        _apply_permissions(path)


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
    """Create an immutable-by-convention backup, or validate an existing one."""
    try:
        file_descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o664)
    except FileExistsError:
        if path.read_bytes() != content:
            raise ValueError(f"Existing backup does not match the current source header: {path}")
        return
    with os.fdopen(file_descriptor, "wb") as backup_file:
        backup_file.write(content)
        backup_file.flush()
        os.fsync(backup_file.fileno())


def _renamed_language_model_key(name: str) -> str:
    """Map the nested text-wrapper namespace to the stock Mistral3Model namespace."""
    if name.startswith(NESTED_LANGUAGE_MODEL_PREFIX):
        return BASE_LANGUAGE_MODEL_PREFIX + name.removeprefix(NESTED_LANGUAGE_MODEL_PREFIX)
    return name


def _normalize_safetensors_header(path: Path) -> tuple[int, set[str]]:
    """Normalize one safetensors header while preserving all tensor payload bytes."""
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
        serialized_header = json.dumps(
            normalized_header,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
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
        raise ValueError(f"Header normalization changed the checkpoint size for {path}")
    _, verified_header = _read_safetensors_header(path)
    if verified_header != normalized_header:
        raise ValueError(f"Header verification failed after normalizing {path}")
    tensor_names = {name for name in normalized_header if name != "__metadata__"}
    if any(name.startswith(NESTED_LANGUAGE_MODEL_PREFIX) for name in tensor_names):
        raise ValueError(f"Nested language-model keys remain in {path}")
    return renamed_count, tensor_names


def _replace_json_file(path: Path, content: bytes) -> None:
    """Atomically replace a JSON file while preserving its permission bits."""
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
    """Normalize saved base-model keys for stock ``AutoModel`` loading.

    Supports a single ``model.safetensors`` file or an indexed sharded export.
    Safetensors tensor descriptors and payload offsets remain unchanged.

    Returns:
        Number of tensor keys renamed from ``language_model.model.*`` to
        ``language_model.*``.
    """
    single_checkpoint = output_path / "model.safetensors"
    index_path = output_path / "model.safetensors.index.json"
    if single_checkpoint.is_file() == index_path.is_file():
        raise ValueError(f"Expected exactly one of {single_checkpoint.name} or {index_path.name} in {output_path}")

    if single_checkpoint.is_file():
        renamed_count, _ = _normalize_safetensors_header(single_checkpoint)
        return renamed_count

    index_bytes = index_path.read_bytes()
    if len(index_bytes) > SAFETENSORS_HEADER_MAX_BYTES:
        raise ValueError(f"Unexpectedly large safetensors index: {index_path}")
    index = json.loads(index_bytes)
    if not isinstance(index, dict) or not isinstance(index.get("weight_map"), dict):
        raise TypeError(f"Expected a safetensors index with a weight_map object in {index_path}")
    weight_map = index["weight_map"]
    normalized_weight_map: dict[str, object] = {}
    index_renamed_count = 0
    for name, shard_name in weight_map.items():
        normalized_name = _renamed_language_model_key(name)
        if normalized_name in normalized_weight_map:
            raise ValueError(f"Normalizing {name!r} would collide with {normalized_name!r} in {index_path}")
        normalized_weight_map[normalized_name] = shard_name
        index_renamed_count += normalized_name != name

    unsorted_shard_names = set(weight_map.values())
    if not all(isinstance(shard_name, str) for shard_name in unsorted_shard_names):
        raise TypeError(f"Safetensors index contains a non-string shard name in {index_path}")
    shard_names = sorted(unsorted_shard_names)
    all_tensor_names: set[str] = set()
    header_renamed_count = 0
    for shard_name in shard_names:
        shard_path = output_path / shard_name
        if shard_path.parent != output_path or not shard_path.is_file():
            raise ValueError(f"Invalid or missing safetensors shard {shard_name!r} in {index_path}")
        shard_renamed_count, tensor_names = _normalize_safetensors_header(shard_path)
        duplicates = all_tensor_names & tensor_names
        if duplicates:
            raise ValueError(f"Tensor keys occur in multiple safetensors shards: {sorted(duplicates)[:5]}")
        all_tensor_names.update(tensor_names)
        header_renamed_count += shard_renamed_count

    if header_renamed_count != index_renamed_count:
        raise ValueError(
            f"Header/index rename count mismatch: headers={header_renamed_count}, index={index_renamed_count}"
        )
    if all_tensor_names != set(normalized_weight_map):
        raise ValueError("Safetensors shard headers do not match the normalized index weight map")
    if index_renamed_count:
        _write_exact_backup(Path(f"{index_path}{INDEX_BACKUP_SUFFIX}"), index_bytes)
        index["weight_map"] = normalized_weight_map
        normalized_index = json.dumps(index, ensure_ascii=False, indent=2).encode("utf-8") + b"\n"
        _replace_json_file(index_path, normalized_index)
    return header_renamed_count


def _validate_saved_processor_files(
    output_path: Path,
    *,
    source_tokenizer_config: dict[str, object],
    source_processor_config: dict[str, object],
) -> None:
    """Validate exact tokenizer and processor config contracts in the saved directory."""
    tokenizer_json = output_path / "tokenizer.json"
    tokenizer_config_path = output_path / "tokenizer_config.json"
    processor_config_path = output_path / "processor_config.json"
    if not tokenizer_json.is_file() or tokenizer_json.stat().st_size == 0:
        raise FileNotFoundError(f"Missing non-empty tokenizer.json in {output_path}")
    if not tokenizer_config_path.is_file():
        raise FileNotFoundError(f"Missing tokenizer_config.json in {output_path}")
    if not processor_config_path.is_file():
        raise FileNotFoundError(f"Missing processor_config.json in {output_path}")
    tekken_path = next(output_path.rglob("tekken.json"), None)
    if tekken_path is not None:
        raise ValueError(f"tekken.json must not be present in the checkpoint; found {tekken_path}")
    for chat_template_name in ("chat_template.jinja", "chat_template.json"):
        chat_template_path = next(output_path.rglob(chat_template_name), None)
        if chat_template_path is not None:
            raise ValueError(f"The generative chat template must not be present; found {chat_template_path}")

    with tokenizer_config_path.open(encoding="utf-8") as tokenizer_config_file:
        tokenizer_config = json.load(tokenizer_config_file)
    if not isinstance(tokenizer_config, dict):
        raise TypeError(f"Expected a JSON object in {tokenizer_config_path}")
    expected_tokenizer_config = _expected_output_tokenizer_config(source_tokenizer_config)
    if tokenizer_config != expected_tokenizer_config:
        missing_keys = sorted(expected_tokenizer_config.keys() - tokenizer_config.keys())
        extra_keys = sorted(tokenizer_config.keys() - expected_tokenizer_config.keys())
        changed_keys = sorted(
            name
            for name in expected_tokenizer_config.keys() & tokenizer_config.keys()
            if tokenizer_config[name] != expected_tokenizer_config[name]
        )
        raise ValueError(
            "Saved tokenizer_config.json differs from the source config plus the requested overrides: "
            f"missing={missing_keys}, extra={extra_keys}, changed={changed_keys}"
        )

    with processor_config_path.open(encoding="utf-8") as processor_config_file:
        processor_config = json.load(processor_config_file)
    if not isinstance(processor_config, dict):
        raise TypeError(f"Expected a JSON object in {processor_config_path}")
    if processor_config != source_processor_config:
        missing_keys = sorted(source_processor_config.keys() - processor_config.keys())
        extra_keys = sorted(processor_config.keys() - source_processor_config.keys())
        changed_keys = sorted(
            name
            for name in source_processor_config.keys() & processor_config.keys()
            if processor_config[name] != source_processor_config[name]
        )
        raise ValueError(
            "Saved processor_config.json differs from the BF16 VLM source config: "
            f"missing={missing_keys}, extra={extra_keys}, changed={changed_keys}"
        )


def main() -> int:
    """Compose, save, reload, and validate the requested Hugging Face checkpoint."""
    _require_mistral_common_absent()
    _require_transformers_version()
    from transformers import (
        AutoModel,
        AutoProcessor,
        AutoTokenizer,
        Ministral3Model,
        Mistral3Model,
        PixtralProcessor,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for bounded-memory checkpoint composition")

    previous_umask = os.umask(0o002)
    output_prepared = False
    try:
        _prepare_output_directory(OUTPUT_PATH)
        output_prepared = True
        device_index = torch.cuda.current_device()
        device_map = {"": device_index}
        logger.info("Loading tokenizer config from %s", TOKENIZER_MODEL_ID)
        source_tokenizer_config = _load_source_tokenizer_config()
        logger.info("Loading tokenizer from %s", TOKENIZER_MODEL_ID)
        tokenizer = _load_tokenizer()

        logger.info("Loading Pixtral processor config from %s", VISION_MODEL_ID)
        source_processor_config = _load_source_processor_config()
        image_processor = _build_image_processor(source_processor_config)

        logger.info("Loading BF16 vision tower and projector from %s", VISION_MODEL_ID)
        model = AutoModel.from_pretrained(
            VISION_MODEL_ID,
            revision=REVISION,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if not isinstance(model, Mistral3Model):
            raise TypeError(f"Expected Mistral3Model from {VISION_MODEL_ID}, got {type(model).__name__}")

        logger.info("Loading BF16 bidirectional text model from %s", TEXT_MODEL_ID)
        text_model = AutoModel.from_pretrained(
            TEXT_MODEL_ID,
            revision=REVISION,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if not isinstance(text_model, Ministral3Model):
            raise TypeError(f"Expected Ministral3Model from {TEXT_MODEL_ID}, got {type(text_model).__name__}")

        text_config = text_model.config
        text_config.max_position_embeddings = MAX_SEQUENCE_LENGTH
        logger.info("Fingerprinting BF16 vision and projector source weights")
        expected_vision_fingerprints = _state_fingerprints(model.vision_tower)
        expected_projector_fingerprints = _state_fingerprints(model.multi_modal_projector)
        original_language_model = model.language_model
        model.language_model = text_model
        model.language_model.config = text_config
        del original_language_model
        del text_model
        gc.collect()
        torch.cuda.empty_cache()

        model.config.text_config = text_config
        model.config.architectures = ["Mistral3Model"]
        model.config.dtype = torch.bfloat16
        model.config.rope_theta = TEXT_ROPE_THETA
        _validate_composed_model(model)
        actual_vision_fingerprints = _state_fingerprints(model.vision_tower)
        if actual_vision_fingerprints != expected_vision_fingerprints:
            raise ValueError("Injecting the text model changed BF16 vision-tower weights")
        actual_projector_fingerprints = _state_fingerprints(model.multi_modal_projector)
        if actual_projector_fingerprints != expected_projector_fingerprints:
            raise ValueError("Injecting the text model changed BF16 multimodal-projector weights")
        expected_config = _config_snapshot(model)
        logger.info("Fingerprinting all composed model state tensors")
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

        logger.info("Saving composed checkpoint to %s", OUTPUT_PATH)
        model.save_pretrained(
            OUTPUT_PATH,
            safe_serialization=True,
            max_shard_size="4GB",
        )
        renamed_key_count = _normalize_saved_mistral3_state_dict(OUTPUT_PATH)
        logger.info("Normalized %d saved language-model tensor keys", renamed_key_count)
        processor.save_pretrained(OUTPUT_PATH)
        _write_output_tokenizer_config(OUTPUT_PATH, source_tokenizer_config)
        _write_output_processor_config(OUTPUT_PATH, source_processor_config)
        (OUTPUT_PATH / "tekken.json").unlink(missing_ok=True)
        (OUTPUT_PATH / "chat_template.jinja").unlink(missing_ok=True)
        (OUTPUT_PATH / "chat_template.json").unlink(missing_ok=True)
        _validate_saved_processor_files(
            OUTPUT_PATH,
            source_tokenizer_config=source_tokenizer_config,
            source_processor_config=source_processor_config,
        )
        _make_group_writable(OUTPUT_PATH)

        del processor
        del image_processor
        del tokenizer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Reloading the saved model for structural and byte-level validation")
        reloaded_model = AutoModel.from_pretrained(
            OUTPUT_PATH,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
            device_map=device_map,
            low_cpu_mem_usage=True,
        )
        if not isinstance(reloaded_model, Mistral3Model):
            raise TypeError(
                f"Expected saved checkpoint to reload as Mistral3Model, got {type(reloaded_model).__name__}"
            )
        _validate_composed_model(reloaded_model)
        actual_config = _config_snapshot(reloaded_model)
        if actual_config != expected_config:
            missing_config_keys = sorted(expected_config.keys() - actual_config.keys())
            extra_config_keys = sorted(actual_config.keys() - expected_config.keys())
            changed_config_keys = sorted(
                name
                for name in expected_config.keys() & actual_config.keys()
                if actual_config[name] != expected_config[name]
            )
            raise ValueError(
                "Save/reload changed model config values: "
                f"missing={missing_config_keys}, extra={extra_config_keys}, changed={changed_config_keys}"
            )
        logger.info("Fingerprinting all reloaded model state tensors")
        actual_fingerprints = _state_fingerprints(reloaded_model)
        if actual_fingerprints != expected_fingerprints:
            mismatches = sorted(set(expected_fingerprints) | set(actual_fingerprints))
            mismatches = [
                name for name in mismatches if actual_fingerprints.get(name) != expected_fingerprints.get(name)
            ]
            raise ValueError(f"Save/reload changed state-tensor bytes for: {mismatches}")

        logger.info("Reloading tokenizer and processor without remote code or mistral-common")
        reloaded_tokenizer = AutoTokenizer.from_pretrained(
            OUTPUT_PATH,
            local_files_only=True,
            trust_remote_code=False,
        )
        _validate_tokenizer(reloaded_tokenizer)
        reloaded_processor = AutoProcessor.from_pretrained(
            OUTPUT_PATH,
            local_files_only=True,
            trust_remote_code=False,
        )
        if not isinstance(reloaded_processor, PixtralProcessor):
            raise TypeError(
                f"Expected saved processor to reload as PixtralProcessor, got {type(reloaded_processor).__name__}"
            )
        _validate_tokenizer(reloaded_processor.tokenizer)
        if reloaded_processor.image_token_ids != [10, 12, 13]:
            raise ValueError(f"Expected image token IDs [10, 12, 13], got {reloaded_processor.image_token_ids}")

        logger.info("Running a tiny image-first multimodal validation forward")
        _validate_multimodal_forward(reloaded_model, reloaded_processor, torch.device("cuda", device_index))
        _validate_saved_processor_files(
            OUTPUT_PATH,
            source_tokenizer_config=source_tokenizer_config,
            source_processor_config=source_processor_config,
        )
        _make_group_writable(OUTPUT_PATH)
        del reloaded_processor
        del reloaded_tokenizer
        del reloaded_model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Checkpoint composition and validation completed successfully")
        return 0
    finally:
        os.umask(previous_umask)
        if output_prepared and OUTPUT_PATH.exists() and OUTPUT_PATH.is_dir():
            _make_group_writable(OUTPUT_PATH)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(main())
