# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Normalize fetched image documents without filesystem or network access."""

from __future__ import annotations

import io
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import cast

from PIL import Image

from nemo_automodel.components.datasets.vlm.conversation import ConversationMessage, ConversationSample, MediaValue
from nemo_automodel.components.datasets.vlm.datasets import VlmMediaDecodeError


@dataclass(frozen=True, kw_only=True)
class VlmInputOptions:
    """Per-source schema adaptation before processor encoding.

    ``column_mapping`` maps canonical names (messages, images, question, answer)
    to source columns. A custom normalizer receives only the fetched payload and
    must return a materialized conversation without opening files or URLs. Use a
    deterministic, serializable module-level callable for worker execution.
    """

    column_mapping: dict[str, str] = field(default_factory=dict)
    normalizer: Callable[[Mapping[str, MediaValue]], ConversationSample] | None = None

    def __post_init__(self) -> None:
        unknown = set(self.column_mapping) - {"messages", "images", "question", "answer"}
        if unknown:
            raise ValueError(f"Unknown VLM input column_mapping keys: {sorted(unknown)}")
        names = list(self.column_mapping.values())
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("VLM input column names must be nonempty, distinct strings")
        if ("question" in self.column_mapping) != ("answer" in self.column_mapping):
            raise ValueError("Map both question and answer columns together")
        if "messages" in self.column_mapping and "question" in self.column_mapping:
            raise ValueError("Map either messages or question/answer columns")
        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("VLM input normalizer must be a callable")


def decode_embedded_image(value: MediaValue | bytearray | memoryview) -> Image.Image:
    """Decode embedded bytes or a PIL image into an owned, materialized RGB image.

    HF image structs use their bytes even when a path is present. Paths and URLs
    are never opened. Corrupt bytes raise VlmMediaDecodeError; an unsupported
    storage/schema raises ValueError and is never silently dropped.
    """
    image = _embedded_image(value)
    try:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        with Image.open(io.BytesIO(bytes(image))) as decoded:
            return decoded.convert("RGB")
    except (OSError, Image.DecompressionBombError) as exc:
        raise VlmMediaDecodeError("Cannot decode embedded image bytes") from exc


def _embedded_image(value: MediaValue | bytearray | memoryview) -> Image.Image | bytes:
    if isinstance(value, dict) and value.get("bytes") is not None:
        value = value["bytes"]
    if isinstance(value, (Image.Image, bytes)):
        return value
    if isinstance(value, (bytearray, memoryview)):
        return bytes(value)
    raise ValueError(
        "Zephon VLM requires embedded image bytes or materialized PIL images, not paths/URLs. "
        "Export external media into self-contained Parquet or LitData shards before training; "
        "see examples/vlm_finetune/zephon/data.md."
    )


def _column(
    payload: Mapping[str, MediaValue], options: VlmInputOptions, name: str, aliases: tuple[str, ...]
) -> MediaValue:
    if name in options.column_mapping:
        key = options.column_mapping[name]
        if key not in payload:
            raise ValueError(f"Missing configured VLM {name} column {key!r}")
        return payload[key]
    present = [key for key in aliases if payload.get(key) is not None]
    if len(present) > 1:
        raise ValueError(f"Ambiguous VLM {name} columns {present}; select one with input.column_mapping")
    return payload[present[0]] if present else None


def normalize_image_document(
    payload: Mapping[str, MediaValue], *, options: VlmInputOptions | None = None, source_name: str = "VLM source"
) -> ConversationSample:
    """Normalize common fetched image/chat schemas to AutoModel conversations.

    Native conversation, messages, ShareGPT from/value, and image/question/answer
    records are accepted. Separate images bind to explicit <image> markers or
    structured image placeholders in encounter order; question/answer records
    place their images before the question. No input containers are mutated and
    no file or URL is opened. Schema errors include the source name.
    """
    options = VlmInputOptions() if options is None else options
    if options.normalizer is not None:
        # Specialized binary layouts are decoded by the caller's pure callback.
        payload = cast(Mapping[str, MediaValue], options.normalizer(payload))
        options = VlmInputOptions()
    try:
        return _normalize(payload, options)
    except ValueError as exc:
        raise ValueError(f"{source_name}: {exc}") from exc


def _normalize(payload: Mapping[str, MediaValue], options: VlmInputOptions) -> ConversationSample:
    messages = _column(payload, options, "messages", ("conversation", "messages", "conversations"))
    # Recognize common pointer-column names so they fail with migration guidance
    # instead of making an image/question/answer record appear text-only.
    images = _column(
        payload, options, "images", ("images", "image", "image_path", "image_paths", "image_url", "image_urls")
    )
    pending = list(images) if isinstance(images, (list, tuple)) else ([] if images is None else [images])
    if messages is None:
        question = _column(payload, options, "question", ("question",))
        answer = _column(payload, options, "answer", ("answer",))
        if not isinstance(question, str) or not isinstance(answer, str):
            raise ValueError("Expected conversation/messages/conversations or string question and answer columns")
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": image} for image in pending]
                + [{"type": "text", "text": question}],
            },
            {"role": "assistant", "content": answer},
        ]
        pending = []
    if isinstance(messages, str):
        try:
            messages = json.loads(messages)
        except json.JSONDecodeError as exc:
            raise ValueError("Conversation string must be a JSON array") from exc
    if not isinstance(messages, list) or not messages:
        raise ValueError("Conversation must be a nonempty list of messages")
    # Reject unsupported modalities before attempting any image decode, including
    # when corrupt-image dropping is enabled on the encoding operator.
    for message in messages:
        if isinstance(message, dict) and isinstance(message.get("content"), list):
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") not in ("text", "image"):
                    raise ValueError(
                        f"Zephon image conversations support only image and text content; got {block.get('type')!r}"
                    )
    for image in pending:
        _embedded_image(image)
    cursor = 0

    def take_image() -> MediaValue:
        nonlocal cursor
        if cursor >= len(pending):
            raise ValueError("Image placeholder has no matching embedded image")
        image = pending[cursor]
        cursor += 1
        return image

    def content_blocks(value: MediaValue) -> list[dict[str, MediaValue]]:
        if isinstance(value, str):
            parts = value.split("<image>")
            result = []
            for index, text in enumerate(parts):
                if text:
                    result.append({"type": "text", "text": text})
                if index < len(parts) - 1:
                    result.append({"type": "image", "image": _embedded_image(take_image())})
            return result
        if not isinstance(value, list):
            raise ValueError("Message content must be a string or a list of text/image blocks")
        result = []
        for block in value:
            if not isinstance(block, dict):
                raise ValueError("Each content block must be a mapping")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                result.extend(content_blocks(block["text"]))
            elif kind == "image":
                image = block.get("image")
                result.append({"type": "image", "image": _embedded_image(take_image() if image is None else image)})
            else:
                raise ValueError("Zephon image conversations support only image and text content blocks")
        return result

    conversation: list[ConversationMessage] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Each message must be a mapping")
        role = message.get("role", message.get("from"))
        if not isinstance(role, str) or not role:
            raise ValueError("Each message needs a role (or ShareGPT from)")
        role = {"human": "user", "gpt": "assistant"}.get(role, role)
        conversation.append({"role": role, "content": content_blocks(message.get("content", message.get("value")))})
    if cursor != len(pending):
        raise ValueError("Embedded images remain unused; put an <image> marker at each image location")
    result: ConversationSample = {"conversation": conversation}
    # Preserve already fetched metadata used by native processor/template logic.
    if "mm_inputs_meta" in payload:
        metadata = payload["mm_inputs_meta"]
        if not isinstance(metadata, dict):
            raise ValueError("mm_inputs_meta must be a mapping")
        result["mm_inputs_meta"] = metadata
    if "_text_tokens" in payload:
        tokens = payload["_text_tokens"]
        if not isinstance(tokens, int):
            raise ValueError("_text_tokens must be an integer")
        result["_text_tokens"] = tokens
    # Validate the entire document before decoding any image: corrupt-media drop
    # must never conceal a later schema error (such as a second image pointer).
    for message in conversation:
        content = message["content"]
        assert isinstance(content, list)
        for block in content:
            if block["type"] == "image":
                block["image"] = decode_embedded_image(block["image"])
    return result
