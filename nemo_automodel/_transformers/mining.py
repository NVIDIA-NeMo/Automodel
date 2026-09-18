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

import logging
from dataclasses import dataclass
from importlib import import_module
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
import torch

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

    def build(self, *, model: "BiEncoderModel", device: torch.device) -> "CheckpointMiningEncoder":
        """Build an encoder using the processor declared by the loaded backbone.

        Args:
            model: Bi-encoder module whose ``encode`` method returns a tensor of shape [batch, hidden].
            device: Device on which model inputs and embeddings are computed.

        Returns:
            A configured multimodal mining encoder.
        """
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
        with (
            torch.no_grad(),
            torch.amp.autocast(self.device.type, dtype=torch.float16, enabled=self.device.type == "cuda"),
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
                source_image = document.get("image")
                source_has_image = source_image is not None and not (
                    isinstance(source_image, str) and source_image == ""
                )
                image = source_image if self.use_images else None
                if isinstance(image, (bytes, bytearray, memoryview)):
                    image = {"bytes": bytes(image)}
                text = document.get("text")
                if source_has_image and not self.use_text_in_document:
                    text = ""
                title = document.get("title")
                if text and title:
                    text = f"{title} {text}".strip()
                has_image = image is not None and not (isinstance(image, str) and image == "")
                has_text = text is not None and str(text).strip() != ""
                if not has_image and not has_text:
                    document_id = document.get("_mining_document_id", "<unknown>")
                    raise ValueError(
                        f"Document {document_id!r} has no encodable text or image under the configured multimodal "
                        "mining policy."
                    )
                processor_documents.append({"image": image, "text": text})
            inputs = self.processor.process_documents(processor_documents, return_tensors="pt")
            embeddings.append(self._encode_batch(inputs))
        return np.concatenate(embeddings, axis=0)

    def release_model(self) -> None:
        """Release the encoder's model reference after embedding generation."""
        self.model = None
