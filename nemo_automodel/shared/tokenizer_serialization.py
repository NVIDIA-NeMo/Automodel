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

"""Serialization cleanup shared by portable tokenizer exporters."""

import json
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def _write_json(path: str, contents: dict) -> None:
    """Write tokenizer metadata as formatted JSON."""
    with open(path, "w") as stream:
        json.dump(contents, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def restore_source_tokenizer_truncation_and_padding(original_model_path: str | None, save_directory: str) -> None:
    """Remove training-time truncation and padding from a saved ``tokenizer.json``.

    A fast tokenizer keeps the truncation and padding of its last call, and ``save_pretrained`` writes them to
    ``tokenizer.json``, where the ``tokenizers`` library applies them to every input of a consumer that loads the
    file directly. The two entries of the saved file are reset to those of the source snapshot, or to None when
    there is no source file. The tokenizer object is not read or changed, and a file that already holds these
    values is not rewritten.

    Args:
        original_model_path: Local source snapshot, or None to clear the state.
        save_directory: Directory containing the serialized tokenizer.
    """
    tokenizer_json_path = os.path.join(save_directory, "tokenizer.json")
    if not os.path.isfile(tokenizer_json_path):
        return
    with open(tokenizer_json_path, encoding="utf-8") as f:
        tokenizer_json = json.load(f)
    source_tokenizer_json = {}
    if original_model_path is not None:
        source_tokenizer_json_path = os.path.join(original_model_path, "tokenizer.json")
        if os.path.isfile(source_tokenizer_json_path):
            with open(source_tokenizer_json_path, encoding="utf-8") as f:
                source_tokenizer_json = json.load(f)
    source_state = {key: source_tokenizer_json.get(key) for key in ("truncation", "padding")}
    if all(tokenizer_json.get(key) == value for key, value in source_state.items()):
        return
    tokenizer_json.update(source_state)
    with open(tokenizer_json_path, "w", encoding="utf-8") as f:
        json.dump(tokenizer_json, f, indent=2, ensure_ascii=False)
        f.write("\n")


def restore_source_tokenizer_serialization_state(
    original_model_path: str | None,
    hf_metadata_dir: str,
    tokenizer: "PreTrainedTokenizerBase",
) -> None:
    """Remove training-time tokenizer state while retaining tokenizer edits.

    Args:
        original_model_path: Local source snapshot, or None to clear transient state.
        hf_metadata_dir: Export directory containing the serialized tokenizer.
        tokenizer: Tokenizer supplying the current padding token.
    """
    pad_token = getattr(tokenizer, "pad_token", None)
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
        if pad_token is not None:
            tokenizer_config["pad_token"] = pad_token
        _write_json(tokenizer_config_path, tokenizer_config)

    special_tokens_map_path = os.path.join(hf_metadata_dir, "special_tokens_map.json")
    if pad_token is not None and os.path.isfile(special_tokens_map_path):
        with open(special_tokens_map_path) as f:
            special_tokens_map = json.load(f)
        special_tokens_map["pad_token"] = pad_token
        _write_json(special_tokens_map_path, special_tokens_map)
