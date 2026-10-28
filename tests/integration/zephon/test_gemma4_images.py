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

"""Real Gemma4 image encoding, packed attention and process-worker replay on CPU."""

import copy
import json
from contextlib import closing

import pytest
import torch
from PIL import Image

from nemo_automodel._transformers.registry import resolve_image_position_state
from nemo_automodel.components.datasets.vlm.datasets import encode_vlm_document
from nemo_automodel.components.datasets.zephon.config import (
    ZephonOptions,
    ZephonPackingOptions,
    ZephonRuntimeOptions,
    ZephonSource,
    ZephonSourceOptions,
    ZephonTrainingOptions,
    ZephonWorkerOptions,
)
from nemo_automodel.components.datasets.zephon.vlm_config import (
    ZephonVlmDataloaderConfig,
    ZephonVlmSourceConfig,
)
from nemo_automodel.components.distributed.pipelining.hf_utils import (
    validate_hf_model_for_pipeline_support,
)
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import (
    configure_packing,
    configure_packing_for_models,
)
from nemo_automodel.components.models.gemma4_moe.model import (
    Gemma4ForConditionalGeneration,
)
from tests.utils.gemma4_vlm_fixtures import gemma4_processor, tiny_gemma4_config
from tests.utils.zephon_positions import assert_joined_position_ranges


@pytest.fixture
def source(tmp_path):
    processor = gemma4_processor()
    processor.save_pretrained(tmp_path / "processor")
    images = []
    for color in ("red", "blue"):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (8, 8), color).save(path)
        images.append(str(path))
    documents = []
    for paths in ([images[0]], [images[1]], [], images):
        documents.append(
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": path} for path in paths]
                        + [{"type": "text", "text": "Describe image"}],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "the answer"}]},
                ]
            }
        )
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "images.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents * 2))
    return tmp_path, documents, processor


def _model(backend="sdpa"):
    config = tiny_gemma4_config()
    config._attn_implementation = backend
    return (
        Gemma4ForConditionalGeneration(
            config,
            backend=BackendConfig(attn=backend, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch"),
        )
        .float()
        .train()
    )


def _loader(source, model, *, runner="inline", fake=False, max_documents=2, xdoc=False):
    directory, _, _ = source
    config = ZephonVlmDataloaderConfig(
        dataset_config=ZephonVlmSourceConfig(inject_fake_images=fake),
        zephon=ZephonOptions(
            sources=(ZephonSource(name="images", path=str(directory / "corpus"), fmt="jsonl"),),
            packing=ZephonPackingOptions(
                seq_len=255,
                pack_algorithm="first_fit",
                pack_num_bins=1,
                max_num_documents=max_documents,
                max_num_documents_scope="bin",
                shuffle_after_pack=False,
            ),
            source=ZephonSourceOptions(
                chunk_size=4, shuffle_shards=False, shuffle_within_shard=False, shuffle_block_size=None
            ),
            training=ZephonTrainingOptions(mask_segment_starts=True, allow_cross_document_attention=xdoc),
            workers=ZephonWorkerOptions(tokenize_parallelism=1),
            runtime=ZephonRuntimeOptions(runner=runner, mtp_mode=False),
        ),
    )
    return config.build(
        pretrained_model_name_or_path=str(directory / "processor"),
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        position_state_factory=lambda: resolve_image_position_state(model),
        packing_contract=configure_packing_for_models([model]),
    ).dataloader


def _forward(model, batch):
    """Run IDs [B,S], scalar positions [B,S], compact IDs and processor image tensors.

    Args:
        model: Real Gemma4 model.
        batch: Collated token/media batch, including ignored labels [B,S].

    Returns:
        Model logits [B,S,V].
    """
    configure_packing(model.config._attn_implementation, model=model)
    return model(**{key: value for key, value in batch.items() if key != "labels"}, use_cache=False).logits


@pytest.mark.parametrize("backend", ["sdpa", "eager"])
def test_packed_images_match_separate_native_documents_and_gradients(source, backend):
    torch.manual_seed(42)
    model = _model(backend)
    reference = copy.deepcopy(model)
    with closing(_loader(source, model)) as loader:
        batch = next(loader)
    _, documents, processor = source
    encoded = [encode_vlm_document(document, processor, inject_fake_images=False) for document in documents[:2]]
    valid = batch["_packed_seq_ids"].gt(0)
    expected_ids = torch.cat([document["input_ids"] for document in encoded])
    torch.testing.assert_close(batch["input_ids"][valid], expected_ids)
    torch.testing.assert_close(batch["image_position_ids"], torch.cat([doc["image_position_ids"] for doc in encoded]))
    valid_patches = batch["image_position_ids"].ge(0).all(-1).sum(-1)
    expected_image_tokens = (valid_patches // model.config.vision_config.pooling_kernel_size**2).sum()
    assert batch["input_ids"].eq(model.config.image_token_id).sum() == expected_image_tokens
    assert model.model.language_model.embed_tokens.weight is model.lm_head.weight
    separate = [
        reference(
            input_ids=document["input_ids"][None],
            attention_mask=document["attention_mask"][None],
            pixel_values=document["pixel_values"],
            image_position_ids=document["image_position_ids"],
            mm_token_type_ids=document["mm_token_type_ids"].unsqueeze(0),
            use_cache=False,
        ).logits
        for document in encoded
    ]
    expected = torch.cat(separate, dim=1).squeeze(0)
    actual = _forward(model, batch)[valid]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    labels = batch["labels"][valid]
    actual_loss = torch.nn.functional.cross_entropy(actual, labels, reduction="sum")
    expected_loss = torch.nn.functional.cross_entropy(expected, labels, reduction="sum")
    torch.testing.assert_close(actual_loss, expected_loss)
    actual_loss.backward()
    expected_loss.backward()
    assert model.model.vision_tower.patch_embedder.input_proj.weight.grad.abs().sum() > 0
    for (name, parameter), (_, baseline) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, baseline.grad, atol=2e-5, rtol=2e-4, msg=name)
        else:
            assert baseline.grad is None


def test_xdoc_retains_local_geometry_media_supervision_and_native_pp_restriction(source):
    torch.manual_seed(43)
    model = _model()
    with closing(_loader(source, model)) as loader:
        batch = next(loader)
    with closing(_loader(source, model, xdoc=True)) as xdoc_loader:
        xdoc = next(xdoc_loader)
    assert_joined_position_ranges(batch["position_ids"], xdoc["position_ids"], batch["_packed_seq_ids"])
    for key in batch.keys() - {"_packed_seq_ids", "position_ids"}:
        torch.testing.assert_close(xdoc[key], batch[key])
    assert "document_ids" in xdoc
    torch.testing.assert_close(xdoc["document_ids"], batch["document_ids"])
    assert xdoc["_packed_seq_ids"][xdoc["_packed_seq_ids"].gt(0)].eq(1).all()
    assert xdoc["document_ids"].gt(1).any()
    before = _forward(model, batch)
    before_xdoc = _forward(model, xdoc)
    changed_ids = batch["input_ids"].clone()
    changed_ids[batch["_packed_seq_ids"].eq(1) & changed_ids.eq(5)] = 8
    second = batch["_packed_seq_ids"].eq(2)
    torch.testing.assert_close(_forward(model, {**batch, "input_ids": changed_ids})[second], before[second])
    assert not torch.allclose(_forward(model, {**xdoc, "input_ids": changed_ids})[second], before_xdoc[second])
    with pytest.raises(ValueError, match="tie_word_embeddings=True"):
        validate_hf_model_for_pipeline_support(model)


def test_process_loader_restores_images_and_fake_token_exclusions(source):
    model = _model()
    original, restored = (
        _loader(source, model, runner="process", fake=True, max_documents=1),
        _loader(source, model, runner="process", fake=True, max_documents=1),
    )
    try:
        next(original)
        checkpoint = original.state_dict()
        expected = list(original)
        assert expected
        restored.load_state_dict(checkpoint)
        actual = list(restored)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        assert any((batch["_packed_seq_ids"].eq(0) & ~batch["padding_mask"]).any() for batch in actual)
        assert torch.isfinite(_forward(model, actual[0])).all()
    finally:
        original.close()
        restored.close()
