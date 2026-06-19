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

"""Pixtral processor extensions for Ministral3 bi-encoder retrieval."""

from __future__ import annotations

import base64
import os
from io import BytesIO
from typing import Any, Literal

import requests
import torch
from PIL import Image
from transformers import BatchEncoding, PixtralProcessor


def load_image(image: Any) -> Image.Image:
    """Load an image from a PIL object, local path, URL, base64 string, or bytes."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    if isinstance(image, dict):
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        if "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        if "url" in image:
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        if "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
    raise ValueError(f"Invalid image: {image}")


class Ministral3BiEncoderProcessor(PixtralProcessor):
    """Pixtral processor with retrieval-specific query/document batching helpers."""

    def __init__(
        self,
        image_processor: Any = None,
        tokenizer: Any = None,
        patch_size: int = 16,
        spatial_merge_size: int = 1,
        chat_template: str | None = None,
        image_token: str = "[IMG]",
        image_break_token: str = "[IMG_BREAK]",
        image_end_token: str = "[IMG_END]",
        q_max_length: int | None = None,
        p_max_length: int | None = None,
        q_max_len: int | None = None,
        p_max_len: int | None = None,
        pad_to_multiple_of: int | None = None,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        padding: bool | str = True,
        image_longest_edge: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
            chat_template=chat_template,
            image_token=image_token,
            image_break_token=image_break_token,
            image_end_token=image_end_token,
            **kwargs,
        )
        self.q_max_length = q_max_length if q_max_length is not None else q_max_len
        self.p_max_length = p_max_length if p_max_length is not None else p_max_len
        self.pad_to_multiple_of = pad_to_multiple_of
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        self.image_longest_edge = image_longest_edge

    def process_queries(
        self,
        queries: list[str],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        **kwargs: Any,
    ) -> BatchEncoding:
        """Process query strings into tokenized model inputs."""
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        query_prompts = [f"{self.query_prefix} {query}" if self.query_prefix else query for query in queries]
        return self._process_text(
            query_prompts,
            max_length=self.q_max_length,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )

    def process_documents(
        self,
        documents: dict[str, list[Any]] | list[dict[str, Any]],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process text and image documents into model inputs.

        Args:
            documents: Either a dict with ``images`` and ``texts`` lists, or a list
                of dicts with ``image`` and ``text`` keys.
            return_tensors: Output format, ``"pt"`` or ``"np"``.
            padding: Padding strategy for tokenization.
            truncation: Whether to truncate document tokens.
            **kwargs: Extra keyword arguments forwarded to ``PixtralProcessor``.

        Returns:
            A dictionary with token fields and, when images are present,
            ``pixel_values`` and ``image_sizes``.
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        images, texts = self._extract_document_fields(documents)
        contents = []
        processor_images = []
        for image, text in zip(images, texts):
            content = "" if text is None else str(text)
            if image is not None and image != "":
                processor_images.append(load_image(image).convert("RGB"))
                content = f"{self.image_token} {content}".strip()
            if self.passage_prefix:
                content = f"{self.passage_prefix} {content}".strip()
            contents.append(content)

        batch = self._process_text(
            contents,
            images=processor_images or None,
            max_length=self.p_max_length,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if "pixel_values" not in batch:
            batch["pixel_values"] = None
        if "image_sizes" not in batch:
            batch["image_sizes"] = None
        return batch

    def process_queries_documents_biencoder(self, features: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        """Process grouped retrieval examples into the bi-encoder batch format."""
        queries = []
        pos_neg_text_batch = []
        pos_neg_image_batch = []
        for feature in features:
            queries.append(feature["question"])
            pos_neg_text_batch.extend(feature["doc_text"])
            pos_neg_image_batch.extend(feature["doc_image"])

        query_batch_dict = self.process_queries(queries, **kwargs)
        doc_batch_dict = self.process_documents(
            {"images": pos_neg_image_batch, "texts": pos_neg_text_batch},
            **kwargs,
        )
        merged_batch_dict = self.merge_batch_dict(query_batch_dict, doc_batch_dict)
        return self.add_dummy_labels(queries, merged_batch_dict)

    def merge_batch_dict(self, query_batch_dict: dict[str, Any], doc_batch_dict: dict[str, Any]) -> dict[str, Any]:
        """Prefix and merge query/document processor outputs."""
        merged_batch_dict = {}
        for key, value in query_batch_dict.items():
            merged_batch_dict[f"q_{key}"] = value
        for key, value in doc_batch_dict.items():
            merged_batch_dict[f"d_{key}"] = value
        return merged_batch_dict

    def add_dummy_labels(self, questions: list[str], merged_batch_dict: dict[str, Any]) -> dict[str, Any]:
        """Add dummy labels expected by the retrieval training loop."""
        merged_batch_dict["labels"] = torch.zeros(len(questions), dtype=torch.long)
        return merged_batch_dict

    def _process_text(
        self,
        text: list[str],
        max_length: int | None,
        return_tensors: Literal["pt", "np"],
        padding: bool | str | None,
        truncation: bool,
        images: list[Image.Image] | None = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if padding is None:
            padding = self.padding
        if truncation and max_length is None:
            max_length = getattr(self.tokenizer, "model_max_length", None)
        if images is not None and self.image_longest_edge is not None and "size" not in kwargs:
            kwargs["size"] = {"longest_edge": self.image_longest_edge}

        return self(
            images=images,
            text=text,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            **kwargs,
        )

    @staticmethod
    def _extract_document_fields(documents: dict[str, list[Any]] | list[dict[str, Any]]) -> tuple[list[Any], list[Any]]:
        if isinstance(documents, dict):
            images = documents["images"]
            texts = documents["texts"]
            if len(texts) != len(images):
                raise ValueError(f"Got {len(texts)} texts and {len(images)} images.")
            return images, texts
        if isinstance(documents, list):
            return [pair["image"] for pair in documents], [pair["text"] for pair in documents]
        raise ValueError("The documents need to be a dict or list of dicts.")


def _register_with_hf_auto_classes() -> None:
    Ministral3BiEncoderProcessor.register_for_auto_class("AutoProcessor")


_register_with_hf_auto_classes()
