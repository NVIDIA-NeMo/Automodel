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

"""Checkpoint-owned retrieval preprocessing for mining and evaluation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image
from transformers.utils.hub import cached_file

from nemo_automodel.shared.import_utils import safe_import

if TYPE_CHECKING:
    from nemo_automodel._transformers.retrieval import BiEncoderModel

logger = logging.getLogger(__name__)


class _RetrievalProcessor(Protocol):
    def process_queries(self, queries: list[str], *, return_tensors: Literal["pt"]) -> dict[str, Any]:
        """Process query strings.

        Args:
            queries: Query strings in input order.
            return_tensors: Tensor backend.

        Returns:
            Token IDs and attention mask tensors of shape [batch, sequence].
        """
        ...

    def process_documents(self, documents: list[dict[str, Any]], *, return_tensors: Literal["pt"]) -> dict[str, Any]:
        """Process text/image documents.

        Args:
            documents: Mappings containing text and optional decoded or local images.
            return_tensors: Tensor backend.

        Returns:
            Token IDs/masks of shape [batch, sequence], optional pixel tensors of
            shape [images, channels, height, width], and image sizes of shape [images, 2].
        """
        ...


def _prepare_document(document: dict[str, Any], *, use_text_in_document: bool, use_images: bool) -> tuple[Any, Any]:
    """Apply the same mining content policy for both inference backends."""
    source_image = document.get("image")
    source_has_image = source_image is not None and not (isinstance(source_image, str) and source_image == "")
    image = source_image if use_images else None
    text = document.get("text")
    if source_has_image and not use_text_in_document:
        text = ""
    title = document.get("title")
    if text and title:
        text = f"{title} {text}".strip()
    has_image = image is not None and not (isinstance(image, str) and image == "")
    has_text = text is not None and str(text).strip() != ""
    if not has_image and not has_text:
        document_id = document.get("_mining_document_id", "<unknown>")
        raise ValueError(
            f"Document {document_id!r} has no encodable text or image under the configured multimodal mining policy."
        )
    return image, text


@dataclass(frozen=True)
class CheckpointMiningEncoderConfig:
    """Optional runtime overrides for the checkpoint's model-owned retrieval processor.

    Omitted overrides preserve saved processor settings and Sentence Transformers
    prompts. An explicitly empty prefix disables that prompt. The processor is
    loaded from the same resolved snapshot as the model, not a second model path.
    """

    q_max_length: int | None = None
    p_max_length: int | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None
    image_longest_edge: int | None = None
    use_text_in_document: bool = True
    use_images: bool = True

    def build(
        self,
        *,
        device: torch.device,
        model_name_or_path: str | None = None,
        model: "BiEncoderModel | None" = None,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> "CheckpointMiningEncoder | SentenceTransformerMiningEncoder":
        """Build an encoder using Sentence Transformers metadata or the checkpoint processor.

        Args:
            device: Device on which model inputs and embeddings are computed.
            model_name_or_path: Local checkpoint directory or Hugging Face model ID.
            model: Already loaded bi-encoder, for callers that own model loading.
            trust_remote_code: Whether model loading may execute remote code.
            attn_implementation: Optional attention backend for model loading.

        Returns:
            A configured mining encoder.
        """
        if model is None:
            if model_name_or_path is None:
                raise ValueError("Either model_name_or_path or model must be provided for mining.")
            modules_file = cached_file(model_name_or_path, "modules.json", _raise_exceptions_for_missing_entries=False)
            if modules_file is not None:
                config_file = cached_file(
                    model_name_or_path, "config_sentence_transformers.json", _raise_exceptions_for_missing_entries=False
                )
                if config_file is not None:
                    with open(config_file, encoding="utf-8") as file:
                        model_type = json.load(file).get("model_type", "SentenceTransformer")
                    if model_type != "SentenceTransformer":
                        raise ValueError(
                            f"Checkpoint {model_name_or_path!r} is a {model_type}, not a SentenceTransformer "
                            "embedding model."
                        )
                available, sentence_transformers = safe_import("sentence_transformers")
                if not available:
                    raise ImportError("Mining this checkpoint requires the sentence-transformers package.")
                model_kwargs = {"attn_implementation": attn_implementation} if attn_implementation else None
                sentence_transformer = sentence_transformers.SentenceTransformer(
                    model_name_or_path,
                    device=str(device),
                    trust_remote_code=trust_remote_code,
                    model_kwargs=model_kwargs,
                )
                if self.image_longest_edge is not None:
                    image_processor = getattr(
                        getattr(sentence_transformer[0], "processor", None), "image_processor", None
                    )
                    if image_processor is None or "longest_edge" not in image_processor.size:
                        raise ValueError("This SentenceTransformer checkpoint has no configurable image longest edge.")
                    image_processor.size["longest_edge"] = self.image_longest_edge
                logger.info("Using Sentence Transformers inference for mining checkpoint %s", model_name_or_path)
                return SentenceTransformerMiningEncoder(
                    model=sentence_transformer,
                    q_max_length=self.q_max_length,
                    p_max_length=self.p_max_length,
                    query_prefix=self.query_prefix,
                    passage_prefix=self.passage_prefix,
                    use_text_in_document=self.use_text_in_document,
                    use_images=self.use_images,
                )

            from nemo_automodel._transformers.auto_model import NeMoAutoModelBiEncoder

            model_kwargs = {"use_liger_kernel": False, "use_sdpa_patching": True}
            if trust_remote_code:
                model_kwargs["trust_remote_code"] = True
            if attn_implementation is not None:
                model_kwargs["attn_implementation"] = attn_implementation
            model = NeMoAutoModelBiEncoder.from_pretrained(model_name_or_path, **model_kwargs).to(device)
            model.eval()
            logger.info("Using AutoModel inference for mining checkpoint %s", model_name_or_path)

        processor_target = getattr(model.model, "retrieval_processor_target", None)
        if processor_target is None:
            raise ValueError("This checkpoint's backbone does not declare a supported retrieval processor.")
        module_name, class_name = processor_target.rsplit(".", 1)
        processor_class = getattr(import_module(module_name), class_name)
        overrides = {
            "q_max_length": self.q_max_length,
            "p_max_length": self.p_max_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "image_longest_edge": self.image_longest_edge,
        }
        loading_options = {key: value for key, value in overrides.items() if value is not None}
        if model.config._commit_hash is not None:
            loading_options["revision"] = model.config._commit_hash
        processor = processor_class.from_pretrained(
            model.config.name_or_path or model.source_model_path, **loading_options
        )
        logger.info("Resolved checkpoint retrieval processor: %s", processor)
        return CheckpointMiningEncoder(
            model=model,
            processor=processor,
            device=device,
            use_text_in_document=self.use_text_in_document,
            use_images=self.use_images,
        )


class CheckpointMiningEncoder:
    """Encode mining queries and documents through the checkpoint's retrieval processor."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: _RetrievalProcessor,
        device: torch.device,
        use_text_in_document: bool = True,
        use_images: bool = True,
    ) -> None:
        """Initialize a multimodal mining encoder.

        Args:
            model: Bi-encoder module whose ``encode`` method returns a tensor of shape [batch, hidden].
            processor: Processor that creates text tensors of shape [batch, sequence] and optional image tensors of
                shape [images, channels, height, width].
            device: Device on which model inputs and embeddings are computed.
            use_text_in_document: Whether image-bearing documents retain their text.
            use_images: Whether document images are forwarded to the processor.
        """
        self.model: torch.nn.Module | None = model
        self.processor = processor
        self.device = device
        self.use_text_in_document = use_text_in_document
        self.use_images = use_images

    @property
    def pooling(self) -> str:
        """Pooling mode used by the checkpoint."""
        return self.model.pooling

    @property
    def l2_normalize(self) -> bool:
        """Whether the checkpoint normalizes embeddings."""
        return self.model.l2_normalize

    def _encode_batch(self, inputs: dict[str, Any]) -> np.ndarray:
        """Encode one processor batch.

        Args:
            inputs: Model inputs containing token tensors of shape [batch, sequence] and optional image tensors of
                shape [images, channels, height, width] plus image sizes of shape [images, 2].

        Returns:
            Embeddings of shape [batch, hidden].

        Raises:
            RuntimeError: If the model has been released or returns no embeddings.
        """
        if self.model is None:
            raise RuntimeError("The multimodal mining model has already been released.")
        device_inputs = {
            key: value.to(self.device) if isinstance(value, torch.Tensor) else value for key, value in inputs.items()
        }
        parameter = next(self.model.parameters(), None)
        dtype = parameter.dtype if parameter is not None else torch.float32
        with (
            torch.no_grad(),
            torch.amp.autocast(
                self.device.type,
                dtype=dtype,
                enabled=self.device.type == "cuda" and dtype in (torch.float16, torch.bfloat16),
            ),
        ):
            embeddings = self.model.encode(device_inputs)
        if embeddings is None:
            raise RuntimeError("The multimodal mining model returned no embeddings.")
        expected_batch_size = inputs["input_ids"].shape[0]
        if embeddings.ndim != 2 or embeddings.shape[0] != expected_batch_size:
            raise ValueError(
                "The multimodal mining model must return embeddings of shape [batch, hidden]; "
                f"got {tuple(embeddings.shape)} for batch size {expected_batch_size}."
            )
        if not torch.isfinite(embeddings).all():
            raise ValueError("The multimodal mining model returned non-finite embeddings.")
        return embeddings.cpu().float().numpy()

    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        """Encode query text in stable input order.

        Args:
            queries: Query strings for a batch of size ``queries``.
            batch_size: Maximum number of queries processed per model call.

        Returns:
            Embeddings of shape [queries, hidden].
        """
        embeddings = []
        for start in range(0, len(queries), batch_size):
            inputs = self.processor.process_queries(queries[start : start + batch_size], return_tensors="pt")
            embeddings.append(self._encode_batch(dict(inputs)))
        return np.concatenate(embeddings, axis=0)

    def encode_documents(self, documents: list[dict[str, Any]], *, batch_size: int) -> np.ndarray:
        """Encode text, image, and image-text documents in stable input order.

        Args:
            documents: Document mappings for a batch of size ``documents``, each with ``text`` and ``image`` fields.
            batch_size: Maximum number of documents processed per model call.

        Returns:
            Embeddings of shape [documents, hidden].
        """
        embeddings = []
        for start in range(0, len(documents), batch_size):
            processor_documents = []
            for document in documents[start : start + batch_size]:
                image, text = _prepare_document(
                    document, use_text_in_document=self.use_text_in_document, use_images=self.use_images
                )
                if isinstance(image, (bytes, bytearray, memoryview)):
                    image = {"bytes": bytes(image)}
                processor_documents.append({"image": image, "text": text})
            inputs = self.processor.process_documents(processor_documents, return_tensors="pt")
            embeddings.append(self._encode_batch(inputs))
        return np.concatenate(embeddings, axis=0)

    def release_model(self) -> None:
        """Release the encoder's model reference after embedding generation."""
        self.model = None


class SentenceTransformerMiningEncoder:
    """Adapt Sentence Transformers query/document inference to the mining corpus."""

    def __init__(
        self,
        *,
        model: Any,
        q_max_length: int | None,
        p_max_length: int | None,
        query_prefix: str | None,
        passage_prefix: str | None,
        use_text_in_document: bool,
        use_images: bool,
    ) -> None:
        """Keep runtime input policies separate from checkpoint-owned preprocessing."""
        self.model = model
        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.use_text_in_document = use_text_in_document
        self.use_images = use_images
        pooling = next((module.pooling_mode for module in model if hasattr(module, "pooling_mode")), None)
        self.pooling = "avg" if pooling == "mean" else pooling
        self.l2_normalize = any(type(module).__name__ == "Normalize" for module in model)

    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        """Encode query strings with saved or explicitly overridden prompts."""
        if self.model is None:
            raise RuntimeError("The multimodal mining model has already been released.")
        options: dict[str, Any] = {}
        if self.query_prefix is not None:
            options["prompt"] = self.query_prefix
        if self.q_max_length is not None:
            options["processing_kwargs"] = {"text": {"max_length": self.q_max_length, "truncation": True}}
        embeddings = np.asarray(self.model.encode_query(queries, batch_size=batch_size, **options), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(queries) or not np.isfinite(embeddings).all():
            raise ValueError("Sentence Transformers must return one finite embedding per mining query.")
        return embeddings

    def encode_documents(self, documents: list[dict[str, Any]], *, batch_size: int) -> np.ndarray:
        """Encode text, image, and image-text corpus records in input order."""
        if self.model is None:
            raise RuntimeError("The multimodal mining model has already been released.")
        groups: dict[str, list[tuple[int, Any]]] = {"text": [], "image": [], "message": []}
        for index, document in enumerate(documents):
            image, text = _prepare_document(
                document, use_text_in_document=self.use_text_in_document, use_images=self.use_images
            )
            if isinstance(image, (bytes, bytearray, memoryview)):
                with Image.open(BytesIO(image)) as decoded:
                    image = decoded.convert("RGB")
            has_image = image is not None and not (isinstance(image, str) and image == "")
            has_text = text is not None and str(text).strip() != ""
            if has_image and has_text:
                groups["message"].append(
                    (
                        index,
                        [
                            {
                                "role": "user",
                                "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
                            }
                        ],
                    )
                )
            elif has_image:
                groups["image"].append((index, image))
            else:
                groups["text"].append((index, text))
        options: dict[str, Any] = {}
        if self.passage_prefix is not None:
            options["prompt"] = self.passage_prefix
        if self.p_max_length is not None:
            options["processing_kwargs"] = {"text": {"max_length": self.p_max_length, "truncation": True}}
        embeddings: list[np.ndarray | None] = [None] * len(documents)
        for group in groups.values():
            if not group:
                continue
            group_embeddings = np.asarray(
                self.model.encode_document([value for _, value in group], batch_size=batch_size, **options),
                dtype=np.float32,
            )
            if (
                group_embeddings.ndim != 2
                or group_embeddings.shape[0] != len(group)
                or not np.isfinite(group_embeddings).all()
            ):
                raise ValueError("Sentence Transformers must return one finite embedding per mining document.")
            for (index, _), embedding in zip(group, group_embeddings):
                embeddings[index] = embedding
        return np.stack(embeddings)

    def release_model(self) -> None:
        """Release the Sentence Transformers model after embedding generation."""
        self.model = None
