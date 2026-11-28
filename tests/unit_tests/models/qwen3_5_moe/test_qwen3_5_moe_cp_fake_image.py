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

"""Qwen3.5-MoE context-parallel M-RoPE positions for batches that hold the loader's masked fake image.

``prepare_model_inputs_for_cp`` computes the full-sequence positions for ``cp_size > 1``. The pretokenized
loader gives a text-only sample a fake image, masks its vision tokens and keeps its grid in
``image_grid_thw``; the tests check that every attended image is still laid out on its own grid.

Expected positions are written from the M-RoPE definition while a row is built, with each image's grid taken
from the row's description and never from ``image_grid_thw``: a counter ``c`` walks the attended tokens, a
text token takes ``(c, c, c)`` and advances ``c`` by one, an image or video frame of ``height x width`` tokens
takes ``(c, c + row, c + column)`` and advances ``c`` by ``max(height, width)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from nemo_automodel.components.datasets.vlm.fake_image import mask_fake_vision_tokens_single
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
from tests.unit_tests.models.qwen3_5_moe.test_qwen3_5_moe_mtp import _backend, _tiny_vl_config

# Token ids: padding, image placeholder, video placeholder, vision start, vision end; text tokens start at 10.
_PAD, _IMAGE, _VIDEO, _START, _END = 0, 5, 6, 7, 8
# What the loader's fake-image masking reads from the processor.
_PROCESSOR = SimpleNamespace(
    image_token_id=_IMAGE, video_token_id=_VIDEO, vision_start_token_id=_START, vision_end_token_id=_END
)


@dataclass(frozen=True)
class _Text:
    tokens: int


@dataclass(frozen=True)
class _Image:
    height: int
    width: int


@dataclass(frozen=True)
class _FakeImage:
    """The image the loader injects into a text-only sample (here 2 x 2 tokens); its tokens get masked."""

    height: int = 2
    width: int = 2


@dataclass(frozen=True)
class _Video:
    """Per frame: one timestamp text token, vision start, ``height x width`` video tokens, vision end."""

    frames: int
    height: int
    width: int


_Segment = _Text | _Image | _FakeImage | _Video


@dataclass
class _Row:
    input_ids: list[int] = field(default_factory=list)
    # Hand-written (temporal, height, width) position per token; None for a token of the masked fake image.
    positions: list[tuple[int, int, int] | None] = field(default_factory=list)
    image_grids: list[list[int]] = field(default_factory=list)
    video_grids: list[list[int]] = field(default_factory=list)


def _build_row(segments: list[_Segment]) -> _Row:
    row = _Row()
    counter = 0

    def text(token: int) -> None:
        nonlocal counter
        row.input_ids.append(token)
        row.positions.append((counter, counter, counter))
        counter += 1

    def media(token: int, height: int, width: int) -> None:
        nonlocal counter
        text(_START)
        for line in range(height):
            for column in range(width):
                row.input_ids.append(token)
                row.positions.append((counter, counter + line, counter + column))
        counter += max(height, width)
        text(_END)

    for segment in segments:
        if isinstance(segment, _Text):
            for _ in range(segment.tokens):
                text(10 + len(row.input_ids))
        elif isinstance(segment, _FakeImage):
            row.input_ids.extend([_START, *[_IMAGE] * (segment.height * segment.width), _END])
            row.positions.extend([None] * (segment.height * segment.width + 2))
            row.image_grids.append([1, segment.height, segment.width])
        elif isinstance(segment, _Image):
            media(_IMAGE, segment.height, segment.width)
            row.image_grids.append([1, segment.height, segment.width])
        else:
            for _ in range(segment.frames):
                text(10 + len(row.input_ids))
                media(_VIDEO, segment.height, segment.width)
            row.video_grids.append([segment.frames, segment.height, segment.width])
    return row


def _build_batch(
    rows: list[list[_Segment]], *, device: str, pad_left: bool
) -> tuple[dict[str, torch.Tensor], list[list[tuple[int, int, int] | None]]]:
    """Collate rows into the hook's batch and the hand-written positions of every token.

    Returns:
        The batch: ``input_ids`` and ``attention_mask`` of shape [batch, sequence], ``image_grid_thw`` of shape
        [images, 3] with the fake images' grids included in batch order, ``video_grid_thw`` of shape [videos, 3].
        The expected positions per row and token, None for a padded or masked token.
    """
    built = [_build_row(segments) for segments in rows]
    length = max(len(row.input_ids) for row in built)
    input_ids, attention_mask, expected = [], [], []
    for segments, row in zip(rows, built):
        sample = {
            "input_ids": torch.tensor(row.input_ids),
            "attention_mask": torch.ones(len(row.input_ids), dtype=torch.long),
        }
        if any(isinstance(segment, _FakeImage) for segment in segments):
            # The pretokenized loader masks the vision tokens of a sample it gave a fake image.
            mask_fake_vision_tokens_single(sample, _PROCESSOR)
        assert sample["attention_mask"].tolist() == [int(position is not None) for position in row.positions]
        padding = length - len(row.input_ids)
        ids, mask = sample["input_ids"].tolist(), sample["attention_mask"].tolist()
        if pad_left:
            input_ids.append([_PAD] * padding + ids)
            attention_mask.append([0] * padding + mask)
            expected.append([None] * padding + row.positions)
        else:
            input_ids.append(ids + [_PAD] * padding)
            attention_mask.append(mask + [0] * padding)
            expected.append(row.positions + [None] * padding)
    batch = {
        "input_ids": torch.tensor(input_ids, device=device),
        "attention_mask": torch.tensor(attention_mask, device=device),
    }
    image_grids = [grid for row in built for grid in row.image_grids]
    video_grids = [grid for row in built for grid in row.video_grids]
    if image_grids:
        batch["image_grid_thw"] = torch.tensor(image_grids, device=device)
    if video_grids:
        batch["video_grid_thw"] = torch.tensor(video_grids, device=device)
    return batch, expected


def _assert_positions(position_ids: torch.Tensor, expected: list[list[tuple[int, int, int] | None]]) -> None:
    """Compare the attended tokens of every row with the hand-written positions.

    Args:
        position_ids: Tensor of shape [3, batch, sequence]; axis 0 is temporal, height, width.
        expected: Per row and token the (temporal, height, width) position, None where it is not compared.
    """
    assert position_ids.shape == (3, len(expected), len(expected[0]))
    for row_index, row in enumerate(expected):
        attended = [index for index, position in enumerate(row) if position is not None]
        got = [tuple(position_ids[:, row_index, index].tolist()) for index in attended]
        assert got == [row[index] for index in attended], f"row {row_index}"


_FAKE_ROW = [_Text(2), _FakeImage(), _Text(3)]

# Batches in which a masked fake image comes before an attended image.
_FAKE_BEFORE_REAL = {
    "fake_row_first_other_token_count": [_FAKE_ROW, [_Text(1), _Image(2, 3), _Text(2)]],
    "fake_row_first_same_token_count": [_FAKE_ROW, [_Text(1), _Image(1, 4), _Text(2)]],
    "fake_row_between_two_real_rows": [
        [_Text(1), _Image(2, 3), _Text(1)],
        _FAKE_ROW,
        [_Text(1), _Image(3, 1), _Text(2)],
    ],
    "fake_row_between_two_real_rows_same_token_count": [
        [_Text(1), _Image(4, 1), _Text(1)],
        _FAKE_ROW,
        [_Text(1), _Image(1, 4), _Text(2)],
    ],
    "two_fake_rows_then_real": [
        [_Text(2), _FakeImage(), _Text(1)],
        [_Text(1), _FakeImage(), _Text(2)],
        [_Text(1), _Image(1, 4), _Text(2)],
    ],
    "fake_row_then_two_images_in_a_row": [_FAKE_ROW, [_Text(1), _Image(1, 4), _Text(1), _Image(2, 1), _Text(2)]],
    "fake_row_then_image_and_video": [_FAKE_ROW, [_Text(1), _Image(3, 1), _Video(2, 2, 1), _Text(1)]],
    "video_row_then_fake_row_then_image": [
        [_Text(1), _Video(2, 1, 2), _Text(1)],
        [_Text(2), _FakeImage(), _Text(1)],
        [_Text(1), _Image(1, 4), _Text(1)],
    ],
}

# Batches whose grids are in walk order without any selection: no attended image follows a masked one.
_CONTROLS = {
    "control_real_row_first_fake_row_last": [[_Text(1), _Image(2, 3), _Text(2)], _FAKE_ROW],
    "control_fake_row_then_video_row": [_FAKE_ROW, [_Text(1), _Video(2, 1, 3), _Text(2)]],
    "control_only_fake_rows": [_FAKE_ROW, [_Text(1), _FakeImage(), _Text(2)]],
    "control_two_image_rows": [[_Text(1), _Image(1, 4), _Text(2)], [_Text(2), _Image(2, 2)]],
    "control_video_row_and_two_image_row": [
        [_Text(1), _Video(2, 1, 3), _Text(2)],
        [_Image(2, 2), _Text(1), _Image(1, 2)],
    ],
    "control_text_rows": [[_Text(5)], [_Text(3)]],
}

_ON_CPU_AND_CUDA = pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ],
)


@pytest.fixture(scope="module")
def model() -> Qwen3_5MoeForConditionalGeneration:
    cfg = _tiny_vl_config()
    cfg.image_token_id, cfg.video_token_id = _IMAGE, _VIDEO
    cfg.vision_start_token_id, cfg.vision_end_token_id = _START, _END
    # The hook reads the config and the batch only, so the weights stay on the CPU.
    return Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())


class TestFakeImageRowUnderContextParallelism:
    @_ON_CPU_AND_CUDA
    @pytest.mark.parametrize("pad_left", [False, True], ids=["right_padded", "left_padded"])
    @pytest.mark.parametrize(
        "rows", [*_FAKE_BEFORE_REAL.values(), *_CONTROLS.values()], ids=[*_FAKE_BEFORE_REAL, *_CONTROLS]
    )
    def test_every_attended_image_is_laid_out_on_its_own_grid(self, model, rows, pad_left, device):
        batch, expected = _build_batch(rows, device=device, pad_left=pad_left)
        grids = {key: batch[key].clone() for key in ("image_grid_thw", "video_grid_thw") if key in batch}

        prepared = model.prepare_model_inputs_for_cp(batch)

        assert prepared["position_ids"].device.type == device
        _assert_positions(prepared["position_ids"], expected)
        # The vision tower still encodes the fake image, so the batch keeps every grid.
        assert "image_grid_thw" not in prepared
        for key, grid in grids.items():
            torch.testing.assert_close(batch[key], grid, rtol=0, atol=0)

    @_ON_CPU_AND_CUDA
    @pytest.mark.parametrize(
        ("real_grid", "real_first", "expected"),
        [
            # 2 x 3 image: six tokens, so the fake image's 2 x 2 grid does not fit it.
            ((2, 3), False, [[0, 1, 1, 1, 1, 1, 1, 4], [0, 1, 1, 1, 2, 2, 2, 4], [0, 1, 2, 3, 1, 2, 3, 4]]),
            # 1 x 4 image: four tokens like the fake image, laid out differently.
            ((1, 4), False, [[0, 1, 1, 1, 1, 5, 6, 7], [0, 1, 1, 1, 1, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]),
            # Control: with the real row first, its grid is the next one in any walk.
            ((1, 4), True, [[0, 1, 1, 1, 1, 5, 6, 7], [0, 1, 1, 1, 1, 5, 6, 7], [0, 1, 2, 3, 4, 5, 6, 7]]),
        ],
        ids=["other_token_count", "same_token_count", "control_real_row_first"],
    )
    def test_smallest_batch_with_positions_written_out(self, model, device, real_grid, real_first, expected):
        """A text-only row with a fake image, and a row of vision start, one image, vision end and text."""
        height, width = real_grid
        rows = [[_FakeImage(), _Text(2)], [_Image(height, width), _Text(6 - height * width)]]
        if real_first:
            rows.reverse()
        real_row = 0 if real_first else 1
        batch, _ = _build_batch(rows, device=device, pad_left=False)
        assert batch["attention_mask"][1 - real_row].tolist() == [0, 0, 0, 0, 0, 0, 1, 1]

        position_ids = model.prepare_model_inputs_for_cp(batch)["position_ids"]

        # position_ids has shape [axes, batch, sequence].
        assert position_ids[:, real_row].tolist() == expected
        # The masked fake image takes no positions: the two text tokens of its row count from zero.
        assert position_ids[:, 1 - real_row, 6:].tolist() == [[0, 1]] * 3

    def test_height_width_grids_are_promoted_with_the_fake_images_grid(self, model):
        batch, expected = _build_batch(
            _FAKE_BEFORE_REAL["fake_row_first_same_token_count"], device="cpu", pad_left=False
        )
        batch["image_grid_hws"] = batch.pop("image_grid_thw")[:, 1:]

        prepared = model.prepare_model_inputs_for_cp(batch)

        _assert_positions(prepared["position_ids"], expected)
        assert prepared["image_grid_thw"].tolist() == [[1, 2, 2], [1, 1, 4]]
        assert prepared["image_grid_hws"] is None

    @pytest.mark.parametrize("rows", list(_CONTROLS.values())[3:], ids=list(_CONTROLS)[3:])
    def test_batch_without_a_masked_image_walks_with_the_batch_grids(self, model, rows, monkeypatch):
        walked_with: list[torch.Tensor | None] = []
        get_rope_index = model.model.get_rope_index

        def recording_get_rope_index(
            input_ids: torch.Tensor,
            mm_token_type_ids: torch.Tensor,
            image_grid_thw: torch.Tensor | None = None,
            video_grid_thw: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Record the image grids of the walk.

            Args:
                input_ids: Tensor of shape [batch, sequence].
                mm_token_type_ids: Tensor of shape [batch, sequence], 0 text, 1 image, 2 video.
                image_grid_thw: Tensor of shape [images, 3], or None.
                video_grid_thw: Tensor of shape [videos, 3], or None.
                attention_mask: Tensor of shape [batch, sequence], or None.

            Returns:
                Position ids of shape [3, batch, sequence] and position deltas of shape [batch, 1].
            """
            walked_with.append(image_grid_thw)
            return get_rope_index(
                input_ids,
                mm_token_type_ids=mm_token_type_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                attention_mask=attention_mask,
            )

        monkeypatch.setattr(model.model, "get_rope_index", recording_get_rope_index)
        batch, expected = _build_batch(rows, device="cpu", pad_left=False)

        prepared = model.prepare_model_inputs_for_cp(batch)

        _assert_positions(prepared["position_ids"], expected)
        assert len(walked_with) == 1
        assert walked_with[0] is batch.get("image_grid_thw")
