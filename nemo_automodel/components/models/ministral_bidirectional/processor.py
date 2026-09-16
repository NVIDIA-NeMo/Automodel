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

"""Pixtral processor extensions for Ministral3 retrieval models."""

from __future__ import annotations

import base64
import os
from enum import IntEnum
from io import BytesIO
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image
from tokenizers.normalizers import Replace, Sequence
from transformers import BatchEncoding, PixtralProcessor
from transformers.tokenization_utils_tokenizers import TokenizersBackend

_CONTROL_TOKEN_ESCAPE = "\u200c"
_CONTROL_TOKEN_ESCAPE_POLICY = "mistral_retrieval_zwnj_prefix_v1"
_CONTROL_TOKEN_PREFIXES = ("[", "<")

_MISTRAL_RETRIEVAL_CHAT_TEMPLATE = """
{# nemo-mistral-retrieval-v1
   Derived from the Mistral Ministral-3 instruct chat template render_content
   macro and its image-before-text canonicalization. Retrieval adaptation:
   system content is the retrieval prefix; generation wrappers and BOS/EOS are
   deliberately omitted so message inference matches training token for token.
#}
{%- macro mark_user_text(text) -%}
{{- text | replace("[", "[\u200c") | replace("<", "<\u200c") -}}
{%- endmacro -%}
{%- macro render_text_content(content, context_name) -%}
{%- if content is string -%}
    {{- content -}}
{%- elif content is sequence -%}
    {%- set ns = namespace(has_output=false) -%}
    {%- for block in content -%}
        {%- if block["type"] == "text" -%}
            {%- if ns.has_output %}{{ "\\n\\n" }}{% endif -%}
            {{- block["text"] -}}
            {%- set ns.has_output = true -%}
        {%- else -%}
            {{- raise_exception(context_name + " may contain only text blocks") -}}
        {%- endif -%}
    {%- endfor -%}
{%- else -%}
    {{- raise_exception(context_name + " must be a string or a list of text blocks") -}}
{%- endif -%}
{%- endmacro -%}
{%- macro render_user_content(content) -%}
{%- if content is string -%}
    {{- mark_user_text(content) -}}
{%- elif content is sequence -%}
    {%- set ns = namespace(has_output=false) -%}
    {%- for block in content -%}
        {%- if block["type"] in ["image", "image_url"] -%}
            {%- if ns.has_output %}{{ " " }}{% endif -%}
            {{- "[IMG]" -}}
            {%- set ns.has_output = true -%}
        {%- endif -%}
    {%- endfor -%}
    {%- for block in content -%}
        {%- if block["type"] == "text" -%}
            {%- if ns.has_output %}{{ " " }}{% endif -%}
            {{- mark_user_text(block["text"]) -}}
            {%- set ns.has_output = true -%}
        {%- elif block["type"] not in ["image", "image_url"] -%}
            {{- raise_exception("User content supports only text and image blocks") -}}
        {%- endif -%}
    {%- endfor -%}
{%- else -%}
    {{- raise_exception("User content must be a string or a list of text and image blocks") -}}
{%- endif -%}
{%- endmacro -%}
{%- if messages | length == 1 and messages[0]["role"] == "user" -%}
    {{- render_user_content(messages[0]["content"]) -}}
{%- elif messages | length == 2
    and messages[0]["role"] == "system"
    and messages[1]["role"] == "user" -%}
    {{- render_text_content(messages[0]["content"], "System content") -}}
    {{- " " -}}
    {{- render_user_content(messages[1]["content"]) -}}
{%- else -%}
    {{- raise_exception("Retrieval inputs require one user message with an optional leading system prefix") -}}
{%- endif -%}
""".strip()


def _apply_mistral_retrieval_ownership_policy(tokenizer: TokenizersBackend) -> str:
    """Configure persistent tokenizer and retrieval-template ownership behavior."""
    tokenizer.split_special_tokens = False
    tokenizer.init_kwargs["split_special_tokens"] = False
    if tokenizer.init_kwargs.get("nemo_control_token_escape_policy") != _CONTROL_TOKEN_ESCAPE_POLICY:
        existing_normalizer = tokenizer.backend_tokenizer.normalizer
        ownership_normalizers = [
            Replace(f"{prefix}{_CONTROL_TOKEN_ESCAPE}", prefix) for prefix in _CONTROL_TOKEN_PREFIXES
        ]
        if existing_normalizer is not None:
            ownership_normalizers.append(existing_normalizer)
        tokenizer.backend_tokenizer.normalizer = Sequence(ownership_normalizers)
        tokenizer.init_kwargs["nemo_control_token_escape_policy"] = _CONTROL_TOKEN_ESCAPE_POLICY
    return _MISTRAL_RETRIEVAL_CHAT_TEMPLATE


class PassageModality(IntEnum):
    """Document modality encoded in a multimodal retrieval batch."""

    TEXT_ONLY = 0
    IMAGE_ONLY = 1
    IMAGE_TEXT = 2


def load_image(image: Any) -> Image.Image:
    """Load one retrieval image without fetching remote content.

    Args:
        image: PIL image, local path, or a mapping containing ``disk_path``,
            ``base64``, or ``bytes``. Encoded payloads require the mapping form.

    Returns:
        The decoded PIL image.

    Raises:
        ValueError: If the input is unsupported or contains a remote URL.
    """
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
            raise ValueError("Remote image URLs are not supported; use disk_path, base64, or bytes.")
        if "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
    raise ValueError(f"Invalid image: {image}")


class Mistral3BiEncoderProcessor(PixtralProcessor):
    """Pixtral processor with retrieval-specific query/document batching helpers.

    The persistent ownership policy is applied when the processor is constructed:
    its tokenizer treats raw control markers as processor-owned, while its chat
    template marks externally supplied text for ordinary tokenization. Direct
    retrieval query/document helpers apply the same text marker because they do
    not render the chat template. Saving defaults to the stock Pixtral class for
    portable bi-encoder checkpoints; cross-encoder recipes preserve this class
    because their evaluation path requires its custom batching helper.
    """

    _export_as_stock_processor = True

    def check_argument_for_proper_class(
        self,
        argument_name: str,
        argument: Any,
    ) -> type | tuple[type, ...]:
        """Validate the retrieval tokenizer without importing MistralCommonBackend.

        Transformers validates processor tokenizers against every generally supported
        tokenizer class, which imports ``mistral-common`` even when the selected tokenizer
        is already a ``TokenizersBackend``. This processor supports only the latter so its
        image markers and escaped user text retain the retrieval-specific semantics.

        Args:
            argument_name: Processor attribute being validated.
            argument: Runtime processor component.

        Returns:
            The accepted component class, or the superclass validation result.

        Raises:
            TypeError: If a non-TokenizersBackend tokenizer is supplied.
        """
        if argument_name == "tokenizer":
            if not isinstance(argument, TokenizersBackend):
                raise TypeError(
                    "Mistral3BiEncoderProcessor requires a Transformers TokenizersBackend tokenizer; "
                    f"got {type(argument).__name__}. Load it with "
                    "Mistral3BiEncoderProcessor.from_pretrained(...), which selects the required backend "
                    "automatically."
                )
            return TokenizersBackend
        return super().check_argument_for_proper_class(argument_name, argument)

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls,
        sub_processor_type: str,
        pretrained_model_name_or_path: str,
        subfolder: str = "",
        **kwargs: Any,
    ) -> TokenizersBackend:
        """Load the processor tokenizer directly, bypassing automatic backend selection."""
        tokenizer_subfolder = (
            subfolder if sub_processor_type == "tokenizer" else os.path.join(subfolder, sub_processor_type)
        )
        if "backend" in kwargs and kwargs["backend"] != "tokenizers":
            raise ValueError(
                f"backend must be 'tokenizers' for Mistral retrieval processors, got {kwargs['backend']!r}."
            )
        if "split_special_tokens" in kwargs and kwargs["split_special_tokens"] is not False:
            raise ValueError("split_special_tokens must be False for Mistral retrieval processors.")
        kwargs.pop("backend", None)
        kwargs.pop("split_special_tokens", None)
        return TokenizersBackend.from_pretrained(
            pretrained_model_name_or_path,
            subfolder=tokenizer_subfolder,
            fix_mistral_regex=True,
            split_special_tokens=False,
            **kwargs,
        )

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
        rerank_max_length: int | None = None,
        q_max_len: int | None = None,
        p_max_len: int | None = None,
        pad_to_multiple_of: int | None = None,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        padding: bool | str = True,
        image_longest_edge: int | None = None,
        use_prompt_template: bool = False,
        export_as_stock_processor: bool = True,
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
        self.chat_template = _apply_mistral_retrieval_ownership_policy(self.tokenizer)
        self.q_max_length = q_max_length if q_max_length is not None else q_max_len
        self.p_max_length = p_max_length if p_max_length is not None else p_max_len
        self.rerank_max_length = rerank_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.padding = padding
        if image_longest_edge is not None:
            self.image_longest_edge = image_longest_edge
        self.use_prompt_template = use_prompt_template
        self.export_as_stock_processor = export_as_stock_processor

    @property
    def image_longest_edge(self) -> int | None:
        """Return the canonical longest edge used by the image processor."""
        return self.image_processor.size.get("longest_edge")

    @image_longest_edge.setter
    def image_longest_edge(self, value: int | None) -> None:
        self.image_processor.size["longest_edge"] = value

    def process_queries(
        self,
        queries: list[str],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        **kwargs: Any,
    ) -> BatchEncoding:
        """Process query strings into tokenized model inputs.

        Args:
            queries: Query texts for a batch of size ``batch``.
            return_tensors: Output format, ``"pt"`` or ``"np"``.
            padding: Padding strategy for tokenization.
            truncation: Whether to truncate query tokens.
            **kwargs: Extra keyword arguments forwarded to ``PixtralProcessor``.

        Returns:
            A mapping with integer ``input_ids`` and ``attention_mask`` of shape
            [batch, query_sequence], padded according to ``padding``. Values are
            PyTorch tensors or NumPy arrays according to ``return_tensors``.
            Unpadded ragged NumPy token fields are object arrays of shape [batch],
            containing one token array per query; PyTorch requires rectangular batches.
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        query_prompts = [
            f"{self.query_prefix} {self._mark_user_text_ownership(query)}"
            if self.query_prefix
            else self._mark_user_text_ownership(query)
            for query in queries
        ]
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
            A mapping with integer ``input_ids`` and ``attention_mask`` of shape
            [documents, document_sequence]. Image inputs add channels-first
            ``pixel_values`` of shape [images, channels, height, width] and
            ``image_sizes`` of shape [images, 2], storing height then width before
            batch padding. Both image fields are None for text-only batches.
            Values are PyTorch tensors or NumPy arrays according to ``return_tensors``.
            Unpadded ragged NumPy token fields are object arrays of shape [documents],
            containing one token array per document; PyTorch requires rectangular batches.
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        images, texts = self._extract_document_fields(documents)
        contents = []
        processor_images = []
        for image, text in zip(images, texts):
            content = "" if text is None else self._mark_user_text_ownership(str(text))
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

    def process_queries_documents_biencoder(
        self,
        features: list[dict[str, Any]],
        return_tensors: Literal["pt", "np"] = "pt",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process grouped retrieval examples into the bi-encoder batch format.

        Args:
            features: Query examples with aligned ``doc_text`` and ``doc_image``
                candidate lists.
            return_tensors: Output format, ``"pt"`` or ``"np"``.
            **kwargs: Extra keyword arguments forwarded to the query and document processors.

        Returns:
            A mapping with ``q_input_ids`` and ``q_attention_mask`` of shape
            [batch, query_sequence], and ``d_input_ids`` and ``d_attention_mask``
            of shape [documents, document_sequence]. Documents are flattened in
            query order, retaining each query's candidate order. ``d_pixel_values``
            has shape [images, channels, height, width] and ``d_image_sizes`` has
            shape [images, 2] in height-width order; both are None without images.
            Integer ``passage_modality`` has shape [documents] and zero ``labels``
            has shape [batch]. Values use the requested PyTorch or NumPy backend.
            Unpadded ragged NumPy token fields are object arrays of shape [batch]
            or [documents], containing per-example token arrays; PyTorch requires rectangular batches.
        """
        queries = []
        pos_neg_text_batch = []
        pos_neg_image_batch = []
        for feature in features:
            queries.append(feature["question"])
            pos_neg_text_batch.extend(feature["doc_text"])
            pos_neg_image_batch.extend(feature["doc_image"])
        doc_modalities = []
        for text, image in zip(pos_neg_text_batch, pos_neg_image_batch):
            has_text = text is not None and not (isinstance(text, str) and text == "")
            has_image = image is not None and not (isinstance(image, str) and image == "")
            if has_text and has_image:
                doc_modalities.append(PassageModality.IMAGE_TEXT)
            elif has_image:
                doc_modalities.append(PassageModality.IMAGE_ONLY)
            else:
                doc_modalities.append(PassageModality.TEXT_ONLY)

        query_batch_dict = self.process_queries(queries, return_tensors=return_tensors, **kwargs)
        doc_batch_dict = self.process_documents(
            {"images": pos_neg_image_batch, "texts": pos_neg_text_batch},
            return_tensors=return_tensors,
            **kwargs,
        )
        merged_batch_dict = self.merge_batch_dict(query_batch_dict, doc_batch_dict)
        if return_tensors == "pt":
            merged_batch_dict["passage_modality"] = torch.tensor(doc_modalities, dtype=torch.long)
        elif return_tensors == "np":
            merged_batch_dict["passage_modality"] = np.asarray(doc_modalities, dtype=np.int64)
        return self.add_dummy_labels(queries, merged_batch_dict, return_tensors=return_tensors)

    def prompt_template_question_passage(self, question: str, text: str) -> str:
        """Format one question-passage pair for cross-encoder scoring.

        Args:
            question: Query text.
            text: Passage text, including an image placeholder when applicable.

        Returns:
            The formatted question-passage text.
        """
        return f"query: {question}\n\npassage: {text}"

    def process_queries_documents_crossencoder(
        self,
        features: list[dict[str, Any]],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Process flattened query-document pairs for vision cross-encoder training.

        Each feature contains one ``question``, ``doc_text``, and optional
        ``doc_image``. Image, text, and image-plus-text documents are supported.

        Args:
            features: Flattened query-document examples. ``num_labels`` contains
                the number of original query groups when labels are requested.
            return_tensors: Output format, ``"pt"`` or ``"np"``.
            padding: Padding strategy for tokenization.
            truncation: Whether to truncate the combined sequences.
            **kwargs: Extra keyword arguments forwarded to ``PixtralProcessor``.

        Returns:
            A mapping with integer ``input_ids`` and ``attention_mask`` of shape
            [pairs, sequence], retaining flattened query-document pair order.
            ``pixel_values`` has shape [images, channels, height, width] and
            ``image_sizes`` has shape [images, 2] in height-width order; both are
            None without images. Zero integer ``labels`` has shape [query_groups]
            when ``num_labels`` is present. Values use the requested PyTorch or NumPy backend.
            Unpadded ragged NumPy token fields are object arrays of shape [pairs],
            containing one token array per pair; PyTorch requires rectangular batches.
        """

        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        contents = []
        processor_images = []
        for feature in features:
            question = self._mark_user_text_ownership(str(feature["question"]))
            document_text = (
                "" if feature["doc_text"] is None else self._mark_user_text_ownership(str(feature["doc_text"]))
            )
            if self.use_prompt_template:
                content = self.prompt_template_question_passage(question, document_text)
            else:
                content = f"{question}\n{document_text}"

            image = feature["doc_image"]
            if image is not None and image != "":
                processor_images.append(load_image(image).convert("RGB"))
                content = f"{self.image_token} {content}".strip()
            contents.append(content)

        batch = self._process_text(
            contents,
            images=processor_images or None,
            max_length=self.rerank_max_length,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            **kwargs,
        )
        if "pixel_values" not in batch:
            batch["pixel_values"] = None
        if "image_sizes" not in batch:
            batch["image_sizes"] = None
        if features and "num_labels" in features[0]:
            num_queries = int(features[0]["num_labels"])
            if return_tensors == "pt":
                batch["labels"] = torch.zeros(num_queries, dtype=torch.long)
            else:
                batch["labels"] = np.zeros(num_queries, dtype=np.int64)

        return batch

    def merge_batch_dict(self, query_batch_dict: dict[str, Any], doc_batch_dict: dict[str, Any]) -> dict[str, Any]:
        """Prefix and merge query and document processor outputs.

        Args:
            query_batch_dict: Query mapping, typically containing ``input_ids``
                and ``attention_mask`` of shape [batch, query_sequence]. Tensor
                or NumPy values of arbitrary rank and non-array values are accepted.
            doc_batch_dict: Document mapping, typically containing ``input_ids``
                and ``attention_mask`` of shape [documents, document_sequence],
                ``pixel_values`` of shape [images, channels, height, width], and
                ``image_sizes`` of shape [images, 2] in height-width order. Image
                fields may be None; other arbitrary-rank values are also accepted.

        Returns:
            One mapping with query keys prefixed by ``q_`` and document keys by ``d_``.
            Values retain their original shapes, dtypes and devices, and alias the
            input values without copying. Neither input mapping is mutated.
        """
        merged_batch_dict = {}
        for key, value in query_batch_dict.items():
            merged_batch_dict[f"q_{key}"] = value
        for key, value in doc_batch_dict.items():
            merged_batch_dict[f"d_{key}"] = value
        return merged_batch_dict

    def add_dummy_labels(
        self,
        questions: list[str],
        merged_batch_dict: dict[str, Any],
        *,
        return_tensors: Literal["pt", "np"] = "pt",
    ) -> dict[str, Any]:
        """Add dummy labels expected by the retrieval training loop.

        Args:
            questions: Query strings defining the batch size.
            merged_batch_dict: Merged processor output. Tensor or NumPy array values may have arbitrary shapes and
                are preserved without modification.
            return_tensors: Backend for the labels, ``"pt"`` or ``"np"``.

        Returns:
            The input mapping, mutated in place with integer labels of shape [batch].
        """
        if return_tensors == "pt":
            merged_batch_dict["labels"] = torch.zeros(len(questions), dtype=torch.long)
        elif return_tensors == "np":
            merged_batch_dict["labels"] = np.zeros(len(questions), dtype=np.int64)
        else:
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")
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
        """Tokenize text and optional images with the retrieval length policy.

        Args:
            text: Text strings in batch order.
            max_length: Maximum sequence length when truncation is enabled.
            return_tensors: Output backend, ``"pt"`` or ``"np"``.
            padding: Padding strategy, or None to use the processor default.
            truncation: Whether to truncate text tokens.
            images: Optional PIL images aligned with image placeholders in ``text``.
            **kwargs: Additional arguments forwarded to ``PixtralProcessor``.

        Returns:
            Mapping with integer ``input_ids`` and ``attention_mask`` of shape
            [batch, sequence]. Image batches also contain ``pixel_values`` of shape
            [images, channels, height, width] and ``image_sizes`` of shape [images, 2]
            in height-width order. Values use the requested PyTorch or NumPy backend.
            Image fields are absent without images. Unpadded ragged NumPy token fields
            are object arrays of shape [batch] containing per-example token arrays;
            PyTorch requires rectangular batches. Additional requested tokenizer outputs
            follow the Pixtral/Transformers-defined layouts.
        """
        if padding is None:
            padding = self.padding
        if truncation and max_length is None:
            max_length = getattr(self.tokenizer, "model_max_length", None)

        try:
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
        except ValueError as error:
            if images and "Mismatch in `image` token count" in str(error):
                raise ValueError(
                    "A complete image token structure plus the passage prefix cannot fit within "
                    f"p_max_length={max_length}; increase p_max_length or reduce the processed image size."
                ) from error
            raise

    def get_hf_export_processor(self) -> PixtralProcessor:
        """Return the behaviorally equivalent stock processor for portable export.

        Subclasses must explicitly re-assert ``_export_as_stock_processor = True``.
        This prevents inherited export from silently discarding custom image or
        text preprocessing that stock ``PixtralProcessor`` cannot reproduce.

        Returns:
            Stock processor configured with the persistent retrieval behavior.

        Raises:
            TypeError: If a subclass has not explicitly opted into stock export.
        """
        if type(self).__dict__.get("_export_as_stock_processor") is not True:
            raise TypeError(
                f"{type(self).__name__} must explicitly opt in to stock processor export after verifying that "
                "its preprocessing behavior is representable by PixtralProcessor assets."
            )

        return PixtralProcessor(
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            patch_size=self.patch_size,
            spatial_merge_size=self.spatial_merge_size,
            chat_template=self.chat_template,
            image_token=self.image_token,
            image_break_token=self.image_break_token,
            image_end_token=self.image_end_token,
        )

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        push_to_hub: bool = False,
        **kwargs: Any,
    ) -> list[str]:
        """Save either a stock Pixtral processor or this custom processor.

        Bi-encoder processors default to a portable stock export. Cross-encoder
        recipes disable stock export because evaluation requires
        ``process_queries_documents_crossencoder`` from this class.

        Args:
            save_directory: Directory for the standard Transformers processor assets.
            push_to_hub: Whether to upload the saved assets to the Hugging Face Hub.
            **kwargs: Additional arguments forwarded to the standard processor save.

        Returns:
            Paths written by the selected processor save operation.
        """
        if self.export_as_stock_processor:
            return self.get_hf_export_processor().save_pretrained(
                save_directory,
                push_to_hub=push_to_hub,
                **kwargs,
            )
        return PixtralProcessor.save_pretrained(self, save_directory, push_to_hub=push_to_hub, **kwargs)

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

    @staticmethod
    def _mark_user_text_ownership(text: str) -> str:
        """Mark user-owned control-token prefixes for ordinary tokenization."""
        for prefix in _CONTROL_TOKEN_PREFIXES:
            text = text.replace(prefix, f"{prefix}{_CONTROL_TOKEN_ESCAPE}")
        return text


def _register_with_hf_auto_classes() -> None:
    Mistral3BiEncoderProcessor.register_for_auto_class("AutoProcessor")


_register_with_hf_auto_classes()
