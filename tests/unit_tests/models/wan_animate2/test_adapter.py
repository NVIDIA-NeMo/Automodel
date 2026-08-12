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

"""CPU unit tests for the Wan-Animate-2 flow-matching adapter.

Every test here is CPU-only and stubs the transformer: the upstream
``WanAnimate2Transformer3DModel`` attention kernels hard-assert
``q.device.type == "cuda"``, so no real forward pass can run on CPU.
``WanAnimate2Adapter.forward`` also resolves the attention backend through
nothing here depends on the fork being installed.

The geometry expectations come from how the upstream transformer *consumes* the
adapter's outputs: its ``(1, 2, 2)`` patch embedding turns every latent frame
into ``(latent_height // 2) * (latent_width // 2)`` tokens, and
``WanAnimate2Transformer3DModel.create_mask`` recovers the driving latent frame
count as ``origin_len // 4 + 1`` and the per-frame token count as
``prod(origin_area) // 256``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.pipeline import create_adapter
from nemo_automodel.components.models.wan_animate2.adapter import WanAnimate2Adapter
from nemo_automodel.components.models.wan_animate2.interleaved import _block_forward_origin

TARGET_FRAMES = 3
LATENT_HEIGHT = 4
LATENT_WIDTH = 6
LATENT_CHANNELS = 16
CONDITIONING_CHANNELS = 20
MASK_CHANNELS = 4
TEXT_TOKENS = 5
REF_TEXT_TOKENS = 4
CLIP_TOKENS = 257
CLIP_DIM = 1280
TEXT_DIM = 4096
# The upstream (1, 2, 2) patch embedding halves both spatial axes.
TOKENS_PER_FRAME = (LATENT_HEIGHT // 2) * (LATENT_WIDTH // 2)


def _make_batch(
    *,
    batch_size: int = 1,
    driving_frames: int = TARGET_FRAMES,
    latent_height: int = LATENT_HEIGHT,
    latent_width: int = LATENT_WIDTH,
) -> dict[str, torch.Tensor]:
    """Build a cached Wan-Animate-2 triplet batch of tiny tensors.

    Args:
        batch_size: Size of axis 0 for every tensor in the batch.
        driving_frames: Latent frame count of the driving video.
        latent_height: Latent grid height.
        latent_width: Latent grid width.

    Returns:
        Mapping of the cache keys the adapter reads: the four latent entries are
        [batch, 16, frames, latent_height, latent_width] with TARGET_FRAMES
        frames (one for ``reference_latents``, ``driving_frames`` for
        ``driving_latents``), the two CLIP entries [batch, 257, 1280] and the two
        text entries [batch, tokens, 4096].
    """
    return {
        "video_latents": torch.randn(batch_size, LATENT_CHANNELS, TARGET_FRAMES, latent_height, latent_width),
        "reference_latents": torch.randn(batch_size, LATENT_CHANNELS, 1, latent_height, latent_width),
        "driving_latents": torch.randn(batch_size, LATENT_CHANNELS, driving_frames, latent_height, latent_width),
        "cond_zero_latents": torch.zeros(batch_size, LATENT_CHANNELS, TARGET_FRAMES, latent_height, latent_width),
        "clip_fea": torch.randn(batch_size, CLIP_TOKENS, CLIP_DIM),
        "clip_fea_ref": torch.randn(batch_size, CLIP_TOKENS, CLIP_DIM),
        "text_embeddings": torch.randn(batch_size, TEXT_TOKENS, TEXT_DIM),
        "prompt_ref_embeddings": torch.randn(batch_size, REF_TEXT_TOKENS, TEXT_DIM),
    }


def _make_context(
    batch: dict[str, torch.Tensor],
    *,
    sigma_value: float = 0.5,
    dtype: torch.dtype = torch.float32,
) -> FlowMatchingContext:
    """Wrap a cached batch in a CPU flow-matching context.

    Args:
        batch: Cached batch laid out as documented by :func:`_make_batch`.
        sigma_value: Flow-matching noise level used for every sample.
        dtype: Compute dtype the adapter must cast its outputs to. The cached
            tensors stay float32, matching a cache read back for bf16 training.

    Returns:
        Context whose float32 ``noisy_latents`` and ``latents`` have the shape of
        ``video_latents`` and whose ``timesteps`` and ``sigma`` have shape
        [batch].
    """
    latents = batch["video_latents"]
    batch_size = latents.shape[0]
    sigma = torch.full((batch_size,), sigma_value)
    noise = torch.randn_like(latents)
    broadcast_sigma = sigma.view(batch_size, 1, 1, 1, 1)
    return FlowMatchingContext(
        noisy_latents=(1.0 - broadcast_sigma) * latents + broadcast_sigma * noise,
        latents=latents,
        timesteps=sigma * 1000.0,
        sigma=sigma,
        task_type="i2v",
        data_type="video",
        device=torch.device("cpu"),
        dtype=dtype,
        batch=batch,
    )


@dataclass
class _RecordedCall:
    """One recorded call into :class:`_RecordingTransformer`, as observed on entry."""

    method: str
    grad_enabled: bool
    # First positional argument: per-sample tensors of shape
    # [channels, frames, latent_height, latent_width].
    stream: list[torch.Tensor]
    # Every other upstream keyword, including the ``_ref`` / non-``_ref`` pairs.
    kwargs: dict[str, Any]
    key_cache: dict[int, torch.Tensor]
    value_cache: dict[int, torch.Tensor]
    cached_entries_on_entry: int


class _RecordingBlock(nn.Module):
    """A block exposing the two per-block passes the interleaved traversal calls."""

    def forward_ref(self, x_ref: torch.Tensor, index: int, k_cache: dict, v_cache: dict, **kwargs: Any):
        """Write this block's keys and values, then return the reference stream."""
        k_cache[index] = x_ref
        v_cache[index] = x_ref
        return x_ref

    def forward_gen(self, x: torch.Tensor, index: int, k_cache: dict, v_cache: dict, **kwargs: Any):
        """Read this block's keys and values, then return the generation stream."""
        _ = k_cache[index], v_cache[index]
        return x


class _RecordingTransformer(nn.Module):
    """Stub for the ``method``-dispatched upstream transformer.

    The reference phase writes one key/value tensor per block into the caches and
    returns nothing; the generation phase scales its input stream by a trainable
    parameter and returns the per-sample list the adapter expects.
    """

    def __init__(self, *, num_blocks: int = 2, scale: float = 2.0) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale))
        self.num_blocks = num_blocks
        self.calls: list[_RecordedCall] = []
        # `install_forward_origin` refuses to run against a transformer that does
        # not expose the interleaved traversal's dependencies, so the stub has to
        # carry them even though this test never exercises the real traversal.
        self.blocks = nn.ModuleList(_RecordingBlock() for _ in range(num_blocks))
        self.patch_embedding = nn.Conv3d(36, 8, kernel_size=1)
        self.time_embedding = nn.Identity()
        self.time_projection = nn.Identity()
        self.text_embedding = nn.Identity()
        self.head = nn.Identity()
        self.block_masks: dict[Any, Any] = {}

    def unpatchify(self, x: torch.Tensor, grid_sizes: torch.Tensor) -> torch.Tensor:
        """Present the upstream method name; unused by these tests."""
        return x

    def create_mask(self, origin_len: int, origin_area: list[int], device: torch.device) -> None:
        """Present the upstream method name; unused by these tests."""
        return None

    def forward_origin(self, inputs: dict[str, Any]) -> list[torch.Tensor]:
        """Record the call and emulate one interleaved traversal.

        Args:
            inputs: The mapping produced by ``prepare_inputs``.

        Returns:
            Per-sample tensors of shape [16, frames, latent_height,
            latent_width], one per sample.
        """
        self.calls.append(
            _RecordedCall(
                method="forward_origin",
                grad_enabled=torch.is_grad_enabled(),
                stream=inputs["x"],
                kwargs=inputs,
                key_cache={},
                value_cache={},
                cached_entries_on_entry=0,
            )
        )
        return [sample * self.scale for sample in inputs["x"]]

    def forward(self, *args: Any, method: str, **kwargs: Any) -> list[torch.Tensor]:
        """Dispatch on ``method`` exactly as the upstream transformer does."""
        return getattr(self, method)(*args, **kwargs)


def test_prepare_inputs_stream_shapes_and_channel_layout() -> None:
    """The generation stream gains one frame; conditioning is 4 mask + 16 latent."""
    torch.manual_seed(0)
    batch = _make_batch()
    context = _make_context(batch)

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    generation_stream = inputs["x"][0]
    conditioning = inputs["y"][0]
    driving_stream = inputs["x_ref"][0]
    driving_conditioning = inputs["condition_y"][0]

    assert len(inputs["x"]) == 1
    assert generation_stream.shape == (LATENT_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH)
    assert conditioning.shape == (CONDITIONING_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH)
    # The driving stream carries no reference slot, so it keeps its own frames.
    assert driving_stream.shape == (LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    assert driving_conditioning.shape == (CONDITIONING_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)

    # Only the leading (reference-slot) frame is new; the rest is the noisy target.
    torch.testing.assert_close(generation_stream[:, 1:], context.noisy_latents[0])
    torch.testing.assert_close(
        conditioning[MASK_CHANNELS:],
        torch.cat([batch["reference_latents"][0], batch["cond_zero_latents"][0]], dim=1),
    )
    torch.testing.assert_close(driving_conditioning[MASK_CHANNELS:], batch["driving_latents"][0])

    # Text and CLIP conditioning must not be swapped between the two phases.
    torch.testing.assert_close(inputs["context"][0], batch["text_embeddings"][0])
    torch.testing.assert_close(inputs["context_ref"][0], batch["prompt_ref_embeddings"][0])
    torch.testing.assert_close(inputs["clip_fea"], batch["clip_fea"])
    torch.testing.assert_close(inputs["clip_fea_ref"], batch["clip_fea_ref"])
    torch.testing.assert_close(inputs["timestep"], context.timesteps)


def test_prepare_inputs_conditioning_mask_marks_only_the_reference_frame() -> None:
    """The generation mask is on for the reference slot and off for target frames."""
    torch.manual_seed(1)
    context = _make_context(_make_batch())

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    mask = inputs["y"][0][:MASK_CHANNELS]
    assert torch.equal(mask[:, 0], torch.ones(MASK_CHANNELS, LATENT_HEIGHT, LATENT_WIDTH))
    assert torch.equal(mask[:, 1:], torch.zeros(MASK_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH))
    # Every driving frame is clean conditioning, so its mask is fully on.
    driving_mask = inputs["condition_y"][0][:MASK_CHANNELS]
    assert torch.equal(driving_mask, torch.ones(MASK_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH))


def test_prepare_inputs_geometry_matches_the_upstream_patch_embedding_and_block_mask() -> None:
    """Sequence lengths, the reference grid and the origin metadata agree with upstream."""
    torch.manual_seed(2)
    context = _make_context(_make_batch())

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    # The patch embedding consumes 16 latent + 20 conditioning channels and emits
    # one token per (1, 2, 2) patch of the concatenated stream.
    assert inputs["x"][0].shape[0] + inputs["y"][0].shape[0] == 36
    assert inputs["seq_len"] == (TARGET_FRAMES + 1) * TOKENS_PER_FRAME
    assert inputs["seq_len_ref"] == TARGET_FRAMES * TOKENS_PER_FRAME
    # grid_sizes_ref indexes the reference RoPE tables, so it is the post-patch grid.
    assert inputs["grid_sizes_ref"].dtype == torch.long
    assert inputs["grid_sizes_ref"].tolist() == [[TARGET_FRAMES, LATENT_HEIGHT // 2, LATENT_WIDTH // 2]]

    # WanAnimate2Transformer3DModel.create_mask recovers the driving frame count
    # and per-frame token count from origin_len/origin_area, then sizes the query
    # stream as (origin_latent_f + 1) * hw against a key stream of
    # origin_latent_f * hw. origin_area is the pixel resolution behind the latent
    # grid (8x VAE).
    origin_latent_frames = inputs["origin_len"] // 4 + 1
    tokens_per_frame = math.prod(inputs["origin_area"]) // 256
    assert origin_latent_frames == TARGET_FRAMES
    assert tokens_per_frame == TOKENS_PER_FRAME
    assert (origin_latent_frames + 1) * tokens_per_frame == inputs["seq_len"]
    assert origin_latent_frames * tokens_per_frame == inputs["seq_len_ref"]
    assert inputs["origin_area"] == [LATENT_HEIGHT * 8, LATENT_WIDTH * 8]


def test_prepare_inputs_reference_slot_lies_on_the_flow_matching_path() -> None:
    """The reference slot interpolates the clean reference latent with noise."""
    adapter = WanAnimate2Adapter()
    torch.manual_seed(4)
    batch = _make_batch()
    reference_latent = batch["reference_latents"][0][:, 0]

    torch.manual_seed(5)
    clean_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=0.0))["x"][0][:, 0]
    torch.manual_seed(5)
    noise_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=1.0))["x"][0][:, 0]
    torch.manual_seed(5)
    half_slot = adapter.prepare_inputs(_make_context(batch, sigma_value=0.5))["x"][0][:, 0]

    # sigma=0 is the clean reference frame, sigma=1 carries none of it, and the
    # midpoint is the average of the two endpoints.
    torch.testing.assert_close(clean_slot, reference_latent)
    assert not torch.allclose(noise_slot, reference_latent)
    torch.testing.assert_close(half_slot, 0.5 * clean_slot + 0.5 * noise_slot)


def test_prepare_inputs_casts_every_model_input_to_the_context_dtype() -> None:
    """The cache is float32 while training runs in bf16, so every input is cast."""
    torch.manual_seed(21)
    context = _make_context(_make_batch(), dtype=torch.bfloat16)

    inputs = WanAnimate2Adapter().prepare_inputs(context)

    for key in ("x", "y", "x_ref", "condition_y", "context", "context_ref"):
        assert {tensor.dtype for tensor in inputs[key]} == {torch.bfloat16}, key
    for key in ("clip_fea", "clip_fea_ref", "timestep"):
        assert inputs[key].dtype == torch.bfloat16, key
    # grid_sizes_ref indexes the reference RoPE tables and must stay integral.
    assert inputs["grid_sizes_ref"].dtype == torch.long


def test_prepare_inputs_rejects_batch_size_greater_than_one() -> None:
    """Batched key/value packing is unsupported upstream, so batch > 1 must fail."""
    torch.manual_seed(6)
    context = _make_context(_make_batch(batch_size=2))

    with pytest.raises(ValueError, match="local_batch_size=1"):
        WanAnimate2Adapter().prepare_inputs(context)


def test_prepare_inputs_rejects_driving_latents_with_a_foreign_frame_count() -> None:
    """A driving clip of another length silently mis-sizes the upstream block mask.

    ``create_mask`` derives the flex-attention query span from the *driving*
    frame count while ``forward_gen`` embeds ``target_latent_frames + 1`` frames.
    A mismatch must be rejected here rather than produce a ``seq_len`` that
    disagrees with the block mask.
    """
    torch.manual_seed(7)
    batch = _make_batch(driving_frames=TARGET_FRAMES - 1)

    with pytest.raises(ValueError, match="same latent frame count"):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("mutate", "error", "message"),
    [
        (lambda batch: batch.pop("clip_fea_ref"), TypeError, "clip_fea_ref"),
        (
            lambda batch: batch.update(text_embeddings=batch["text_embeddings"].unsqueeze(-1)),
            ValueError,
            "'text_embeddings' must have 3 dimensions",
        ),
        (
            lambda batch: batch.update(driving_latents=torch.cat([batch["driving_latents"]] * 2)),
            ValueError,
            "'driving_latents' must have batch size 1",
        ),
    ],
    ids=["missing", "wrong-rank", "foreign-batch-size"],
)
def test_prepare_inputs_rejects_a_malformed_cached_tensor(
    mutate: Callable[[dict[str, torch.Tensor]], object], error: type[Exception], message: str
) -> None:
    """One shared check requires every cached key, its rank, and its batch size."""
    torch.manual_seed(8)
    batch = _make_batch()
    mutate(batch)

    with pytest.raises(error, match=message):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("key", "shape", "message"),
    [
        ("reference_latents", (1, LATENT_CHANNELS, 2, LATENT_HEIGHT, LATENT_WIDTH), "exactly one latent frame"),
        (
            "cond_zero_latents",
            (1, LATENT_CHANNELS, TARGET_FRAMES + 1, LATENT_HEIGHT, LATENT_WIDTH),
            "must match the target latent frame count",
        ),
        ("reference_latents", (1, LATENT_CHANNELS, 1, LATENT_HEIGHT + 2, LATENT_WIDTH), "spatial dims"),
        ("driving_latents", (1, LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH + 2), "spatial dims"),
    ],
)
def test_prepare_inputs_rejects_cached_geometry_that_disagrees_with_the_target(
    key: str, shape: tuple[int, ...], message: str
) -> None:
    """Cached conditioning is concatenated with the target, so its geometry must align."""
    torch.manual_seed(11)
    batch = _make_batch()
    batch[key] = torch.randn(shape)

    with pytest.raises(ValueError, match=message):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


@pytest.mark.parametrize(
    ("mangle", "message"),
    [
        (lambda latents: latents[:, :, 0], "noisy latents"),
        (lambda latents: latents[:, :8], "16 latent channels"),
    ],
    ids=["image-shaped", "half-the-latent-channels"],
)
def test_prepare_inputs_rejects_noisy_latents_with_a_foreign_layout(
    mangle: Callable[[torch.Tensor], torch.Tensor], message: str
) -> None:
    """A video/image mix-up or a non-Wan VAE cache must fail loudly."""
    torch.manual_seed(12)
    context = _make_context(_make_batch())
    context.noisy_latents = mangle(context.noisy_latents)

    with pytest.raises(ValueError, match=message):
        WanAnimate2Adapter().prepare_inputs(context)


def test_prepare_inputs_rejects_a_latent_grid_the_patch_size_cannot_tile() -> None:
    """An odd latent height cannot be tiled by the (1, 2, 2) patch embedding."""
    torch.manual_seed(13)
    batch = _make_batch(latent_height=LATENT_HEIGHT + 1)

    with pytest.raises(ValueError, match=r"divisible by the \(2, 2\) patch size"):
        WanAnimate2Adapter().prepare_inputs(_make_context(batch))


def test_forward_dispatches_one_interleaved_traversal_per_step() -> None:
    """One call per step, through the model's own ``method`` dispatcher, with gradient.

    The two passes used to be issued separately, which under FSDP2 meant two
    forwards per module and a cache that did not survive the call boundary. The
    adapter now issues a single interleaved traversal instead.
    """
    torch.manual_seed(16)
    adapter = WanAnimate2Adapter()
    model = _RecordingTransformer()
    inputs = adapter.prepare_inputs(_make_context(_make_batch()))

    adapter.forward(model, inputs)
    adapter.forward(model, inputs)

    assert [call.method for call in model.calls] == ["forward_origin", "forward_origin"]

    first, _ = model.calls
    assert first.stream is inputs["x"]
    # Both streams and both conditionings reach the traversal in one mapping.
    assert first.kwargs["x_ref"] is inputs["x_ref"]
    assert first.kwargs["condition_y"] is inputs["condition_y"]
    assert first.kwargs["context_ref"] is inputs["context_ref"]
    assert first.kwargs["context"] is inputs["context"]
    # The reference stream is trained, so the traversal runs with gradient.
    assert first.grad_enabled is True


def test_block_traversal_gives_each_block_an_isolated_cache() -> None:
    """A block's keys and values live and die inside its own call.

    This is what keeps the traversal correct under wrappers that rebuild
    container arguments: nothing is handed across a call boundary.
    """
    block = _RecordingBlock()
    stream = torch.zeros(2, 3)
    reference = torch.ones(2, 3)

    _, _ = _block_forward_origin(block, stream, reference, {}, {})

    # forward_gen reading index 0 would raise if the cache were not the one
    # forward_ref just wrote, so reaching here is the assertion.
    assert True


def test_forward_drops_the_reference_slot_and_keeps_the_target_frames() -> None:
    """The prediction covers the target frames only and stays differentiable."""
    torch.manual_seed(18)
    adapter = WanAnimate2Adapter()
    model = _RecordingTransformer(scale=2.0)
    context = _make_context(_make_batch())
    inputs = adapter.prepare_inputs(context)

    prediction = adapter.forward(model, inputs)

    assert prediction.shape == (1, LATENT_CHANNELS, TARGET_FRAMES, LATENT_HEIGHT, LATENT_WIDTH)
    # The stub scales its input, so the surviving frames must be the noisy target
    # frames: had the trailing frame been dropped instead, this would fail.
    torch.testing.assert_close(prediction, context.noisy_latents * 2.0)

    prediction.sum().backward()
    assert model.scale.grad is not None
    assert torch.isfinite(model.scale.grad).all()


def test_create_adapter_resolves_the_wan_animate2_adapter() -> None:
    """The recipe-facing factory name maps to this adapter."""
    assert isinstance(create_adapter("wan_animate2"), WanAnimate2Adapter)
