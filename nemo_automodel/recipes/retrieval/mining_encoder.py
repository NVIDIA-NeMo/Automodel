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

"""Checkpoint inference adapters for the hard-negative mining recipe."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from importlib import import_module
from io import BytesIO
from typing import Any, Literal, Protocol

import numpy as np
import torch
from PIL import Image
from transformers.utils.hub import cached_file

from nemo_automodel.shared.import_utils import safe_import

logger = logging.getLogger(__name__)


class _RetrievalProcessor(Protocol):
    def process_queries(self, queries: list[str], *, return_tensors: Literal["pt"]) -> dict[str, Any]: ...

    def process_documents(
        self, documents: list[dict[str, Any]], *, return_tensors: Literal["pt"]
    ) -> dict[str, Any]: ...


def _prepare_document(document: dict[str, Any]) -> tuple[Any, str]:
    """Use all available document content, normalizing absent images and blank text."""
    image = document.get("image")
    if isinstance(image, str) and image == "":
        image = None
    text = document.get("text")
    title = document.get("title")
    if text and title:
        text = f"{title} {text}".strip()
    text = str(text) if text is not None and str(text).strip() else ""
    if image is None and not text:
        document_id = document.get("_mining_document_id", "<unknown>")
        raise ValueError(f"Document {document_id!r} has no encodable text or image.")
    return image, text


@dataclass(frozen=True)
class CheckpointMiningEncoderConfig:
    """Build mining inference using the checkpoint's saved prompts and preprocessing."""

    def build(
        self,
        *,
        device: torch.device,
        model_name_or_path: str,
        trust_remote_code: bool = False,
        attn_implementation: str | None = None,
    ) -> "CheckpointMiningEncoder | SentenceTransformerMiningEncoder":
        """Build an encoder using Sentence Transformers metadata or the checkpoint processor.

        Args:
            device: Device on which model inputs and embeddings are computed.
            model_name_or_path: Local checkpoint directory or Hugging Face model ID.
            trust_remote_code: Whether model loading may execute remote code.
            attn_implementation: Optional attention backend for model loading.

        Returns:
            A configured mining encoder.
        """
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
            logger.info("Using Sentence Transformers inference for mining checkpoint %s", model_name_or_path)
            return SentenceTransformerMiningEncoder(model=sentence_transformer)

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
        loading_options = {}
        commit_hash = getattr(model.config, "_commit_hash", None)
        if commit_hash is not None:
            loading_options["revision"] = commit_hash
        processor = processor_class.from_pretrained(
            model.config.name_or_path or model.source_model_path, **loading_options
        )
        logger.info("Resolved checkpoint retrieval processor: %s", processor)
        return CheckpointMiningEncoder(model=model, processor=processor, device=device)


class CheckpointMiningEncoder:
    """Encode mining queries and documents through the checkpoint's retrieval processor."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        processor: _RetrievalProcessor,
        device: torch.device,
    ) -> None:
        """Keep the AutoModel backbone and its saved processor together for mining."""
        self.model: torch.nn.Module | None = model
        self.processor = processor
        self.device = device

    @property
    def pooling(self) -> str:
        """Pooling mode used by the checkpoint."""
        return self.model.pooling

    @property
    def l2_normalize(self) -> bool:
        """Whether the checkpoint normalizes embeddings."""
        return self.model.l2_normalize

    def _encode_batch(self, inputs: dict[str, Any]) -> np.ndarray:
        """Return finite embeddings of shape [batch, hidden] for one processor batch."""
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
        """Encode queries in input order with bounded processor batches."""
        embeddings = []
        for start in range(0, len(queries), batch_size):
            inputs = self.processor.process_queries(queries[start : start + batch_size], return_tensors="pt")
            embeddings.append(self._encode_batch(dict(inputs)))
        return np.concatenate(embeddings, axis=0)

    def encode_documents(self, documents: list[dict[str, Any]], *, batch_size: int) -> np.ndarray:
        """Encode corpus documents in input order with bounded processor batches."""
        embeddings = []
        for start in range(0, len(documents), batch_size):
            processor_documents = []
            for document in documents[start : start + batch_size]:
                image, text = _prepare_document(document)
                if isinstance(image, (bytes, bytearray, memoryview)):
                    image = {"bytes": bytes(image)}
                processor_documents.append({"image": image, "text": text})
            inputs = self.processor.process_documents(processor_documents, return_tensors="pt")
            embeddings.append(self._encode_batch(inputs))
        return np.concatenate(embeddings, axis=0)

    def release_model(self) -> None:
        """Move the model to CPU and release it after embedding generation; safe to repeat."""
        if self.model is not None:
            self.model.cpu()
            self.model = None


class SentenceTransformerMiningEncoder:
    """Adapt Sentence Transformers query/document inference to the mining corpus."""

    def __init__(self, *, model: Any) -> None:
        """Keep the checkpoint's inference pipeline and its embedding metadata together."""
        self.model = model
        pooling = next((module.pooling_mode for module in model if hasattr(module, "pooling_mode")), None)
        self.pooling = "avg" if pooling == "mean" else pooling
        self.l2_normalize = any(type(module).__name__ == "Normalize" for module in model)

    def encode_queries(self, queries: list[str], *, batch_size: int) -> np.ndarray:
        """Encode query strings with the checkpoint's saved prompts and sequence limits."""
        if self.model is None:
            raise RuntimeError("The multimodal mining model has already been released.")
        embeddings = np.asarray(self.model.encode_query(queries, batch_size=batch_size), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(queries) or not np.isfinite(embeddings).all():
            raise ValueError("Sentence Transformers must return one finite embedding per mining query.")
        return embeddings

    def encode_documents(self, documents: list[dict[str, Any]], *, batch_size: int) -> np.ndarray:
        """Encode text, image, and image-text corpus records in input order."""
        if self.model is None:
            raise RuntimeError("The multimodal mining model has already been released.")
        groups: dict[str, list[tuple[int, Any]]] = {"text": [], "image": [], "message": []}
        for index, document in enumerate(documents):
            image, text = _prepare_document(document)
            if isinstance(image, (bytes, bytearray, memoryview)):
                with Image.open(BytesIO(image)) as decoded:
                    image = decoded.convert("RGB")
            if image is not None and text:
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
            elif image is not None:
                groups["image"].append((index, image))
            else:
                groups["text"].append((index, text))
        embeddings: list[np.ndarray | None] = [None] * len(documents)
        for group in groups.values():
            if not group:
                continue
            group_embeddings = np.asarray(
                self.model.encode_document([value for _, value in group], batch_size=batch_size),
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
        """Move the model to CPU and release it after embedding generation; safe to repeat."""
        if self.model is not None:
            self.model.cpu()
            self.model = None
