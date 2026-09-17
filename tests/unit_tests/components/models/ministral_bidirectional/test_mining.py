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

"""CPU behavior tests for model-owned multimodal mining encoding."""

from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.models.ministral_bidirectional.mining import (
    Mistral3MultimodalMiningEncoder,
    Mistral3MultimodalMiningEncoderConfig,
)
from nemo_automodel.components.models.ministral_bidirectional.processor import load_image
from nemo_automodel.recipes.retrieval.mine_hard_negatives import MineHardNegativesRecipe


class _PixelProcessor:
    def __init__(self) -> None:
        self.documents = []

    def process_queries(self, queries, return_tensors):
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1.0, 0.0] for _ in queries])}

    def process_documents(self, documents, return_tensors):
        assert return_tensors == "pt"
        self.documents.extend(documents)
        pixels = torch.tensor([float(document["image"]) for document in documents]).reshape(-1, 1, 1, 1)
        return {
            "input_ids": torch.ones((len(documents), 2)),
            "pixel_values": pixels,
            "image_sizes": torch.ones((len(documents), 2), dtype=torch.long),
        }


class _PixelModel(torch.nn.Module):
    def encode(self, inputs):
        pixels = inputs.get("pixel_values")
        if pixels is None:
            return inputs["input_ids"]
        values = pixels.flatten(start_dim=1).mean(dim=1)
        return torch.stack((values, 1.0 - values), dim=1)


class _ImageProcessor:
    def __init__(self) -> None:
        self.documents = []

    def process_documents(self, documents, return_tensors):
        assert return_tensors == "pt"
        self.documents.extend(documents)
        pixels = [load_image(document["image"]).convert("RGB").getpixel((0, 0))[0] / 255 for document in documents]
        return {
            "input_ids": torch.ones((len(documents), 2)),
            "pixel_values": torch.tensor(pixels).reshape(-1, 1, 1, 1),
            "image_sizes": torch.ones((len(documents), 2), dtype=torch.long),
        }


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_typed_config_owns_processor_construction(monkeypatch):
    processor = _PixelProcessor()
    from_pretrained = MagicMock(return_value=processor)
    monkeypatch.setattr(
        "nemo_automodel.components.models.ministral_bidirectional.mining.Mistral3BiEncoderProcessor.from_pretrained",
        from_pretrained,
    )
    config = ConfigNode(
        {
            "_target_": Mistral3MultimodalMiningEncoderConfig,
            "processor_name_or_path": "/processor",
            "p_max_length": 2048,
            "use_text_in_document": False,
        }
    ).instantiate()

    encoder = config.build(model=_PixelModel(), device=torch.device("cpu"))

    assert encoder.processor is processor
    assert encoder.use_text_in_document is False
    assert from_pretrained.call_args.kwargs["p_max_length"] == 2048


def test_pixels_change_same_text_embeddings_and_ranking_excludes_positive():
    processor = _PixelProcessor()
    encoder = Mistral3MultimodalMiningEncoder(
        model=_PixelModel(),
        processor=processor,
        device=torch.device("cpu"),
    )
    documents = [
        {"text": "same", "image": 0.99},
        {"text": "same", "image": 0.80},
        {"text": "same", "image": 0.20},
    ]

    query_embeddings = encoder.encode_queries(["query"], batch_size=1)
    document_embeddings = encoder.encode_documents(documents, batch_size=3)

    assert np.isfinite(document_embeddings).all()
    assert not np.array_equal(document_embeddings[1], document_embeddings[2])
    assert processor.documents == documents

    recipe = MineHardNegativesRecipe.__new__(MineHardNegativesRecipe)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"))
    negative_indices, _, _ = recipe._mine_hard_negatives(
        query_embeddings,
        document_embeddings,
        [[0]],
        batch_size=1,
        num_negs=1,
    )
    assert negative_indices == [[1]]


def test_image_only_and_mixed_documents_preserve_configured_content_policy():
    processor = _PixelProcessor()
    encoder = Mistral3MultimodalMiningEncoder(
        model=_PixelModel(),
        processor=processor,
        device=torch.device("cpu"),
        use_text_in_document=True,
        use_images=True,
    )

    embeddings = encoder.encode_documents(
        [{"text": "", "image": 0.25}, {"title": "A", "text": "caption", "image": 0.75}],
        batch_size=2,
    )

    assert embeddings.shape == (2, 2)
    assert processor.documents == [
        {"image": 0.25, "text": ""},
        {"image": 0.75, "text": "A caption"},
    ]


def test_missing_image_path_fails_with_configured_path(tmp_path):
    missing_path = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match=str(missing_path)):
        load_image(str(missing_path))


def test_unsupported_image_diagnostic_does_not_expose_payload():
    payload = b"private-image-payload"

    with pytest.raises(ValueError, match="Invalid image type: bytes") as error:
        load_image(payload)

    assert "private-image-payload" not in str(error.value)


@pytest.mark.parametrize("binary_type", [bytes, bytearray, memoryview])
def test_binary_images_are_wrapped_for_the_strict_processor(binary_type):
    payload = binary_type(_png_bytes())
    processor = _ImageProcessor()
    encoder = Mistral3MultimodalMiningEncoder(
        model=_PixelModel(),
        processor=processor,
        device=torch.device("cpu"),
    )

    embeddings = encoder.encode_documents([{"text": "caption", "image": payload}], batch_size=1)

    assert embeddings.shape == (1, 2)
    assert np.isfinite(embeddings).all()
    assert processor.documents == [{"text": "caption", "image": {"bytes": bytes(payload)}}]


def test_policy_rejects_document_without_usable_text_or_image():
    encoder = Mistral3MultimodalMiningEncoder(
        model=_PixelModel(),
        processor=_PixelProcessor(),
        device=torch.device("cpu"),
        use_images=False,
    )

    with pytest.raises(ValueError, match="doc-7.*no encodable text or image"):
        encoder.encode_documents(
            [{"_mining_document_id": "doc-7", "text": "", "image": 0.5}],
            batch_size=1,
        )
