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

"""Adversarial packed-image boundaries against independent full-document forwards."""

import copy

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.datasets.vlm.collate_fns import packed_sequence_thd_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.models.deepseek_v41.cp import shard_cp_batch
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.processing import IMAGE, IMAGE_END, IMAGE_NEW_LINE, IMAGE_START
from nemo_automodel.components.models.deepseek_v41.vision import DeepseekV41VisionAligner, DeepseekV41VisionTransformer
from tests.unit_tests.models.deepseek_v41.test_engram import _tokenizer
from tests.unit_tests.models.deepseek_v41.test_model import _backend, _tiny_config
from tests.unit_tests.models.deepseek_v41.test_vision import _config, _reference_vision

pytestmark = [pytest.mark.cpu_only, pytest.mark.timeout(120)]


def _model(full_schedule: bool = False, ratio: int = 2, engram: bool = False) -> DeepseekV41ForCausalLM:
    config = _tiny_config()
    config.dtype = "float32"
    config.image_token_id = 8
    config.vision_config = _config().vision_config
    config.vision_config.downsample_ratio = ratio
    config.text_config.initializer_range = 0.08
    config.text_config.sliding_window = 4
    if full_schedule:
        text = config.text_config
        text.num_hidden_layers = 40
        text.compress_ratios = [0, 0] + [2] * 18 + [1] * 20
        text.kv_source_layer_ids = [2, 8, 14, 20]
        text.index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]
        text.candidate_source_layer_id = 20
        text.candidate_block_size = 8
    if engram:
        text = config.text_config
        text.engram_layer_ids = [1, 14] if full_schedule else [1, 3]
        text.engram_num_embeddings = [113, 253]
        text.engram_vocab_size = 11
        text.engram_compressed_vocab_size = 6
        text.engram_max_ngram_size = 4
        text.engram_n_heads = 2
        text.engram_head_dim = 4
        text.engram_pad_token_id = 2
    model = DeepseekV41ForCausalLM(config, backend=_backend(), tokenizer=_tokenizer() if engram else None)
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    return model


def _sample(
    grids: tuple[tuple[int, int], ...], *, prefix: int = 1, suffix: int = 3, ratio: int = 2
) -> dict[str, torch.Tensor]:
    types = [-1] * prefix
    for h, w in grids:
        types += (
            [IMAGE_START]
            + ([IMAGE] * ((w + ratio - 1) // ratio) + [IMAGE_NEW_LINE]) * ((h + ratio - 1) // ratio)
            + [IMAGE_END]
        )
        types += [-1]
    types += [-1] * suffix
    types = torch.tensor(types)
    ids = torch.where(types >= 0, 8, torch.arange(len(types)) % 7 + 1)
    result = {
        "input_ids": ids,
        "labels": ids.clone(),
        "attention_mask": torch.ones_like(ids),
        "vision_token_types": types,
    }
    result["labels"][types >= 0] = -100
    if grids:
        result["pixel_values"] = torch.randn(sum(h * w for h, w in grids), 3, 2, 2).bfloat16()
        result["image_grid_hws"] = torch.tensor(grids)
    return result


def _batch(samples: list[dict[str, torch.Tensor]], *, length: int | None = None, alignment: int = 1) -> dict:
    """Pack unshifted samples; token tensors are [sequence], patches [patches, 3, 2, 2].

    Args:
        samples: Samples with token-aligned fields and image grids [images, 2].
        length: Optional physical row length.
        alignment: Per-document physical alignment.

    Returns:
        A collated native VLM batch with token axes [1, sequence].
    """
    pack = _build_packed_vlm_sample([_shift_sample(s) for s in samples], 4096, 0, sequence_alignment=alignment)
    return packed_sequence_thd_vlm_collater([pack], max_length=length)


@pytest.mark.parametrize("grids", [((1, 1),), ((3, 5), (2, 4), (1, 7)), ((2, 4), (2, 4))])
@pytest.mark.parametrize("ratio", [2, 3])
def test_batched_vision_matches_independent_equations_and_all_gradients(
    grids: tuple[tuple[int, int], ...], ratio: int
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(180)
    config = _config()
    config.vision_config.downsample_ratio = ratio
    vision, aligner = DeepseekV41VisionTransformer(config), DeepseekV41VisionAligner(config)
    params = {
        **{f"vision.{k}": v for k, v in vision.named_parameters()},
        **{f"aligner.{k}": v for k, v in aligner.named_parameters()},
    }
    ref_params = {k: v.detach().clone().requires_grad_() for k, v in params.items()}
    pixels = torch.randn(sum(h * w for h, w in grids), 3, 2, 2, requires_grad=True)
    ref_pixels = pixels.detach().clone().requires_grad_()
    grid = torch.tensor(grids)
    encoded = vision(pixels.flatten(1), grid_thw=torch.cat([torch.ones_like(grid[:, :1]), grid], 1), return_dict=True)
    actual = aligner(encoded.pooler_output, image_grid_hws=grid)
    expected = torch.cat(
        [
            _reference_vision(patch, ref_params, config, height=h, width=w)
            for patch, (h, w) in zip(ref_pixels.split([h * w for h, w in grids]), grids, strict=True)
        ]
    )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(pixels.grad, ref_pixels.grad, atol=2e-6, rtol=2e-5)
    for name, param in params.items():
        torch.testing.assert_close(param.grad, ref_params[name].grad, atol=3e-6, rtol=3e-5, msg=name)


@pytest.mark.parametrize("alignment", [1, 4, 64])
@pytest.mark.parametrize("full_schedule", [False, True])
@pytest.mark.parametrize("ratio,engram", [(2, False), (3, True)])
def test_packed_images_match_independent_documents_and_updates(
    alignment: int, full_schedule: bool, ratio: int, engram: bool
) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(202)
    model = _model(full_schedule, ratio=ratio, engram=engram)
    reference = copy.deepcopy(model)
    samples = [_sample(((3, 5),), prefix=2, ratio=ratio), _sample(()), _sample(((1, 1), (2, 3)), prefix=3, ratio=ratio)]
    batch = _batch(samples, length=256, alignment=alignment)
    _, batch, layout = shard_cp_batch(None, None, batch, packed_alignment=2, pad_multiple=2)
    labels = batch.pop("labels")
    batch.pop("n_images_per_sample", None)
    actual = model(**batch).logits
    loss = F.cross_entropy(actual.flatten(0, 1), labels.flatten(), ignore_index=-100)
    loss.backward()

    expected_parts = []
    reference_loss = 0
    nlabels = 0
    for sample in samples:
        shifted = _shift_sample(sample)
        kwargs = {key: value for key, value in shifted.items() if key not in ("labels", "attention_mask")}
        kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
        kwargs["vision_token_types"] = kwargs["vision_token_types"].unsqueeze(0)
        output = reference(**kwargs).logits
        expected_parts.append(output)
        reference_loss += F.cross_entropy(output.flatten(0, 1), shifted["labels"], ignore_index=-100, reduction="sum")
        nlabels += int((shifted["labels"] != -100).sum())
    reference_loss = reference_loss / nlabels
    reference_loss.backward()
    for doc_id, expected in enumerate(expected_parts, 1):
        torch.testing.assert_close(actual[batch["packed_seq_ids"] == doc_id], expected[0], atol=8e-6, rtol=8e-5)
    torch.testing.assert_close(loss, reference_loss, atol=2e-6, rtol=2e-5)
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        if param.grad is None or ref.grad is None:
            assert param.grad is None and ref.grad is None, name
        else:
            torch.testing.assert_close(param.grad, ref.grad, atol=1e-5, rtol=3e-4, msg=name)
    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    ref_opt = torch.optim.SGD(reference.parameters(), lr=0.03)
    opt.step()
    ref_opt.step()
    for (name, param), (_, ref) in zip(model.named_parameters(), reference.named_parameters(), strict=True):
        torch.testing.assert_close(param, ref, atol=1e-6, rtol=1e-5, msg=name)


def test_previous_document_image_and_length_cannot_change_next_document() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(93)
    model = _model(True)
    target = _sample(((3, 3), (1, 1)), prefix=1)
    previous = [_sample(((2, 2),), prefix=1), _sample(((5, 7), (1, 3)), prefix=8)]
    outputs = []
    for source in previous:
        _, batch, _ = shard_cp_batch(None, None, _batch([source, target], length=256), packed_alignment=2)
        batch.pop("labels")
        batch.pop("n_images_per_sample", None)
        outputs.append(model(**batch).logits[batch["packed_seq_ids"] == 2])
    torch.testing.assert_close(outputs[0], outputs[1], atol=5e-6, rtol=5e-5)


@pytest.mark.parametrize(
    "corruption",
    ["missing_pixels", "extra_grid", "missing_patch", "bad_grid", "bad_type", "wrong_id", "cross_doc", "image_padding"],
)
def test_malformed_images_are_rejected(corruption: str) -> None:
    torch.set_num_threads(1)
    model = _model()
    _, batch, _ = shard_cp_batch(None, None, _batch([_sample(((3, 5),))], length=32), packed_alignment=2)
    batch.pop("labels")
    batch.pop("n_images_per_sample", None)
    if corruption == "missing_pixels":
        del batch["pixel_values"]
    elif corruption == "extra_grid":
        batch["image_grid_hws"] = batch["image_grid_hws"].repeat(2, 1)
    elif corruption == "missing_patch":
        batch["pixel_values"] = batch["pixel_values"][:-1]
    elif corruption == "bad_grid":
        batch["image_grid_hws"][0, 0] = 0
    elif corruption == "bad_type":
        batch["vision_token_types"][batch["vision_token_types"] == IMAGE_END] = -1
    elif corruption == "wrong_id":
        batch["input_ids"][batch["vision_token_types"] == IMAGE_START] = 1
    elif corruption == "cross_doc":
        batch["packed_seq_ids"][:, 4:] = 2
    else:
        batch["packed_seq_ids"][batch["vision_token_types"] >= 0] = 0
    with pytest.raises(ValueError):
        model(**batch)


def test_shift_cannot_cut_an_image_and_collator_cannot_overflow() -> None:
    sample = _sample(((1, 1),))
    end = int(torch.where(sample["vision_token_types"] == IMAGE_END)[0][-1]) + 1
    for key in ("input_ids", "labels", "attention_mask", "vision_token_types"):
        sample[key] = sample[key][:end]
    with pytest.raises(ValueError, match="truncate an image"):
        _shift_sample(sample)
    with pytest.raises(ValueError, match="exceeding"):
        _batch([_sample(((3, 5),))], length=1)


def test_text_only_and_all_padding_have_zero_visual_gradients() -> None:
    torch.set_num_threads(1)
    model = _model()
    hidden = model.model._image_embeddings(torch.ones(2, 16, dtype=torch.long), None, None, torch.full((2, 16), -1))
    hidden.square().sum().backward()
    for name, param in model.named_parameters():
        if name.startswith(("model.vision.", "model.aligner.")):
            assert param.grad is not None, name
            assert torch.count_nonzero(param.grad) == 0, name


def test_batch_rows_keep_distinct_image_and_document_order() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(197)
    model = _model()
    rows = [
        [_sample(((1, 1), (3, 5))), _sample(())],
        [_sample(()), _sample(((5, 3),), prefix=2)],
    ]
    packs = [_build_packed_vlm_sample([_shift_sample(s) for s in row], 256, 0, sequence_alignment=4) for row in rows]
    batch = packed_sequence_thd_vlm_collater(packs, max_length=128)
    _, batch, _ = shard_cp_batch(None, None, batch, packed_alignment=2)
    batch.pop("labels")
    batch.pop("n_images_per_sample")
    actual = model(**batch).logits
    for row_index, samples in enumerate(rows):
        _, separate, _ = shard_cp_batch(None, None, _batch(samples, length=128, alignment=4), packed_alignment=2)
        separate.pop("labels")
        separate.pop("n_images_per_sample")
        expected = model(**separate).logits
        torch.testing.assert_close(actual[row_index], expected[0], atol=5e-6, rtol=5e-5)


def test_random_pack_padding_preserves_every_token_label_and_image_marker() -> None:
    torch.set_num_threads(1)
    import random

    rng = random.Random(731)
    for _ in range(80):
        samples = [
            _sample(
                tuple((rng.randint(1, 7), rng.randint(1, 7)) for _ in range(rng.randint(0, 3))),
                prefix=rng.randint(0, 7),
                suffix=rng.randint(1, 5),
            )
            for _ in range(rng.randint(1, 5))
        ]
        batch = _batch(samples, alignment=rng.choice((1, 2, 4, 16, 64)))
        _, batch, _ = shard_cp_batch(None, None, batch, packed_alignment=2, pad_multiple=2)
        for i, sample in enumerate(samples, 1):
            expected = _shift_sample(sample)
            mask = batch["packed_seq_ids"][0] == i
            for key in ("input_ids", "labels", "vision_token_types"):
                torch.testing.assert_close(batch[key][0, mask], expected[key], rtol=0, atol=0)
            positions = batch["position_ids"][0, mask]
            torch.testing.assert_close(positions, torch.arange(len(expected["input_ids"])), rtol=0, atol=0)
        pad = batch["packed_seq_ids"] == 0
        assert (batch["vision_token_types"][pad] == -1).all()
        assert (batch["labels"][pad] == -100).all()
        assert not batch["attention_mask"][pad].any()


def test_padding_only_full_model_backward_is_finite_and_zero() -> None:
    torch.set_num_threads(1)
    model = _model()
    ids = torch.zeros((1, 32), dtype=torch.long)
    batch = dict(
        input_ids=ids,
        seq_lens=torch.tensor([[0]]),
        seq_lens_padded=torch.tensor([[32]]),
        labels=torch.full_like(ids, -100),
        vision_token_types=torch.full_like(ids, -1),
    )
    _, batch, _ = shard_cp_batch(None, None, batch, packed_alignment=2)
    labels = batch.pop("labels")
    logits = model(**batch).logits
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=-100, reduction="sum")
    loss.backward()
    assert loss.item() == 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all(), name
            assert torch.count_nonzero(param.grad) == 0, name


@pytest.mark.parametrize("length", [None, 32])
def test_one_token_document_becomes_safe_padding_after_shift(length: int | None) -> None:
    sample = _sample((), prefix=0, suffix=1)
    _, batch, _ = shard_cp_batch(None, None, _batch([sample], length=length), packed_alignment=2)
    assert batch["input_ids"].shape == (1, length or 2)
    assert not batch["packed_seq_ids"].any()
    assert (batch["labels"] == -100).all()
    assert (batch["vision_token_types"] == -1).all()


@pytest.mark.parametrize("mode,size", [("L", (1, 7)), ("RGBA", (7, 1)), ("RGB", (5, 7)), ("RGB", (40, 50))])
def test_exact_pack_length_uses_headers_without_decoding(mode: str, size: tuple[int, int], monkeypatch) -> None:
    import io

    from PIL import Image

    from nemo_automodel.components.models.deepseek_v41.processing import DeepseekV41Processor
    from tests.unit_tests.models.deepseek_v41.test_vision import _tokenizer as vision_tokenizer

    processor = DeepseekV41Processor(vision_tokenizer(), _config())
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    image = Image.open(io.BytesIO(buffer.getvalue()))
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "one two"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "tail"}]},
    ]
    with monkeypatch.context() as patch:
        patch.setattr(image, "load", lambda *a, **k: pytest.fail("Length estimation decoded image pixels"))
        expected_length, expected_media = processor.estimate_packed_tokens(messages)
    result = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
    assert result["input_ids"].numel() == expected_length
    assert int((result["vision_token_types"] >= 0).sum()) == expected_media
