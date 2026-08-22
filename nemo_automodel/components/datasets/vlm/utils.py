# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import io
import logging
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import lmdb

    HAVE_LMDB = True
except ImportError:
    HAVE_LMDB = False


_lmdb_env_cache: dict[str, "lmdb.Environment"] = {}


def _merge_media_values(values: list[Any], *, field_name: str) -> torch.Tensor | list[Any]:
    """Merge media values while preserving the explicit ragged-list contract.

    Four-dimensional ``pixel_values`` use ``[media, channels, height, width]``.
    Some processors emit a different image resolution for each Datum, so pad
    those tensors on the right and bottom to the largest spatial extent before
    concatenating their media axes. Other tensor fields retain their processor-
    defined layout and must already be concatenable.

    Args:
        values: Non-empty per-sample media values. Tensor ``pixel_values``
            use ``[media, channels, height, width]``; other tensor fields use
            ``[media, ...]``. List or tuple items retain their processor-defined
            layout.
        field_name: Processor field being merged.

    Returns:
        One BF16 tensor on the input device, concatenated on its media axis.
        Four-dimensional ``pixel_values`` have shape
        ``[sum_media, channels, max_height, max_width]``; other tensor fields
        have shape ``[sum_media, ...]``. Processors that explicitly represent
        variable-resolution media as lists produce one flattened list whose
        tensor items are converted to BF16 on their original devices.
    """
    if not values:
        raise ValueError("Media merge requires at least one value.")
    if all(isinstance(value, torch.Tensor) for value in values):
        tensors = values
        if field_name == "pixel_values" and all(value.ndim == 4 for value in tensors):
            max_height = max(value.shape[-2] for value in tensors)
            max_width = max(value.shape[-1] for value in tensors)
            tensors = [
                F.pad(value, (0, max_width - value.shape[-1], 0, max_height - value.shape[-2])) for value in tensors
            ]
        return torch.cat(tensors, dim=0).to(torch.bfloat16)
    if all(isinstance(value, (list, tuple)) for value in values):
        return [
            item.to(torch.bfloat16) if isinstance(item, torch.Tensor) else item for value in values for item in value
        ]
    raise TypeError(
        "VLM media values must be consistently tensors or variable-resolution lists, "
        f"got {[type(value).__name__ for value in values]}."
    )


def _resolve_processor_token_id(processor, attr_names, token_names):
    """Resolve a model-specific media token id from processor/config/tokenizer."""
    tokenizer = getattr(processor, "tokenizer", processor)
    unk_id = getattr(tokenizer, "unk_token_id", None)

    config = getattr(processor, "config", None)
    for source in (processor, config, tokenizer, getattr(tokenizer, "config", None)):
        if source is None:
            continue
        for attr in attr_names:
            value = getattr(source, attr, None)
            if isinstance(value, int):
                return value
            if isinstance(value, str):
                try:
                    token_id = tokenizer.convert_tokens_to_ids(value)
                except Exception:
                    continue
                if isinstance(token_id, int) and token_id != unk_id:
                    return token_id
    for token in token_names:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id != unk_id:
            return token_id
    return None


def _grid_media_token_count(grid, merge_size: int) -> int:
    """Return the number of merged media tokens described by a processor grid."""
    if grid is None:
        return 0
    grid_t = torch.as_tensor(grid)
    if grid_t.numel() == 0:
        return 0
    if grid_t.ndim == 1:
        grid_t = grid_t.view(1, -1)
    merge_len = int(merge_size) ** 2
    return int((grid_t.to(torch.long).prod(dim=-1) // merge_len).sum().item())


def _media_token_mismatch(input_ids, processor_inputs, processor) -> str | None:
    """Return a mismatch description for resolvable media grids and token ids.

    Args:
        input_ids: Canonical processor token ids with shape ``[sequence]``.
        processor_inputs: Processor-ready model inputs containing optional image
            or video grids.
        processor: Processor/tokenizer used to resolve model-specific media ids
            and spatial merge factors.

    Returns:
        A short mismatch description, or ``None`` when the grid/token counts
        agree or the processor does not expose enough information to check.
    """
    image_token_id = _resolve_processor_token_id(
        processor,
        ("image_token_id", "image_token_index", "image_token"),
        ("<|image_pad|>", "<image>", "<|image|>"),
    )
    video_token_id = _resolve_processor_token_id(
        processor,
        ("video_token_id", "video_token_index", "video_token"),
        ("<|video_pad|>", "<video>", "<|video|>"),
    )

    image_grid = processor_inputs.get("image_grid_thw")
    if image_grid is not None and image_token_id is not None:
        image_merge_size = getattr(getattr(processor, "image_processor", None), "merge_size", 2)
        expected = _grid_media_token_count(image_grid, image_merge_size)
        actual = int((input_ids == image_token_id).sum().item())
        if actual != expected:
            return f"image tokens={actual}, expected={expected}"

    video_grid = processor_inputs.get("video_grid_thw")
    if video_grid is not None and video_token_id is not None:
        video_merge_size = getattr(getattr(processor, "video_processor", None), "merge_size", 2)
        expected = _grid_media_token_count(video_grid, video_merge_size)
        actual = int((input_ids == video_token_id).sum().item())
        if actual != expected:
            return f"video tokens={actual}, expected={expected}"

    return None


def _resolve_lmdb_image(path):
    """Read an image from an LMDB database.

    Paths use the format ``<lmdb_dir>::<key>``, e.g.
    ``/data/my_db.lmdb::0000000087``.

    Returns:
        PIL.Image.Image: The decoded image.
    """
    if not HAVE_LMDB:
        raise ImportError("lmdb package is required to load LMDB images: pip install lmdb")

    lmdb_path, key = path.split("::", 1)
    if lmdb_path not in _lmdb_env_cache:
        _lmdb_env_cache[lmdb_path] = lmdb.open(lmdb_path, readonly=True, lock=False)
    env = _lmdb_env_cache[lmdb_path]
    with env.begin() as txn:
        data = txn.get(key.encode())
    if data is None:
        raise KeyError(f"Key '{key}' not found in LMDB database '{lmdb_path}'")
    return Image.open(io.BytesIO(data)).convert("RGB")


def _read_video_frames(video_path, processor=None, frame_indices=None, return_metadata=False):
    """Read and sample video frames from a video file using decord.

    If *frame_indices* is provided (e.g. from a dataset annotation), those
    exact frame numbers are used.  Otherwise, frame sampling uses the same
    ``smart_nframes`` + ``linspace`` strategy as ``qwen_vl_utils`` to ensure
    that preloaded frames are identical to those produced by the processor's
    own video pipeline.

    Args:
        video_path: Path to the video file.
        processor: HuggingFace processor whose ``video_processor`` supplies
            default fps / max_frames / min_frames.
        frame_indices: Explicit list of 0-based frame indices to extract.
        return_metadata: If True, return ``(frames, video_fps, used_indices)``
            so callers can preserve timing information for timestamp calculation.

    Returns:
        list[PIL.Image.Image]: Sampled video frames as RGB PIL Images.
        If *return_metadata* is True, returns
        ``(frames, video_fps, used_indices)`` instead.
    """
    try:
        import decord
    except ImportError as exc:
        raise RuntimeError(
            "decord is required to read video files; install it with: pip install nemo-automodel[vlm-media]"
        ) from exc
    import torch as _torch

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path)
    total_frames = len(vr)
    if total_frames == 0:
        raise ValueError(f"Video has no frames: {video_path}")

    video_fps = vr.get_avg_fps()

    # Determine temporal_patch_size for even-frame alignment (default 2)
    temporal_patch_size = 2
    if processor is not None and hasattr(processor, "video_processor"):
        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)

    if frame_indices is not None:
        # Use explicitly specified frame indices, clamp to valid range
        indices = [min(i, total_frames - 1) for i in frame_indices]
        # Pad to temporal_patch_size alignment
        remainder = len(indices) % temporal_patch_size
        if remainder != 0:
            indices.extend([indices[-1]] * (temporal_patch_size - remainder))
    else:
        # Get frame sampling config from processor
        target_fps = None
        max_frames = None
        min_frames = 4
        if processor is not None and hasattr(processor, "video_processor"):
            vp = processor.video_processor
            target_fps = getattr(vp, "fps", None)
            max_frames = getattr(vp, "max_frames", None)
            min_frames = getattr(vp, "min_frames", min_frames)

        # ---------------------------------------------------------------
        # Calculate target frame count using the same algorithm as
        # qwen_vl_utils.smart_nframes:
        #   nframes = total_frames / video_fps * target_fps
        #   clamped to [min_frames, min(max_frames, total_frames)]
        #   rounded UP to temporal_patch_size alignment
        # ---------------------------------------------------------------
        if target_fps is not None and video_fps > 0:
            nframes = total_frames / video_fps * target_fps
        else:
            nframes = float(total_frames)

        nframes = max(nframes, min_frames)
        if max_frames is not None:
            nframes = min(nframes, max_frames)
        nframes = min(nframes, total_frames)

        # Round UP to temporal_patch_size boundary (matches sampler,
        # HF video processor, and LLaMA-Factory).  When nframes exceeds
        # total_frames after rounding, linspace naturally repeats the
        # last frame indices which is the standard padding approach.
        nframes = int(nframes)
        remainder = nframes % temporal_patch_size
        if remainder != 0:
            nframes += temporal_patch_size - remainder
        nframes = max(nframes, temporal_patch_size)

        # Uniformly sample frame indices (matching qwen_vl_utils linspace).
        indices = _torch.linspace(0, total_frames - 1, nframes).round().long().tolist()

    frames = vr.get_batch(indices).asnumpy()
    pil_frames = [Image.fromarray(f).convert("RGB") for f in frames]
    if return_metadata:
        return pil_frames, video_fps, indices
    return pil_frames


def _preload_media(example, processor=None, preserve_video_metadata=False):
    """Pre-load image and video files in a conversation example.

    Images are loaded as PIL RGB Images.
    Videos are decoded into lists of PIL RGB Images (sampled frames).

    When *preserve_video_metadata* is ``True``, the original video fps and
    the sampled frame indices are stored on each video content item as
    ``_video_fps`` and ``_frame_indices``.  This allows downstream code
    (e.g. :func:`_build_video_metadata`) to construct ``VideoMetadata``
    for the processor so it inserts correct timestamps.
    """
    conversation = example.get("conversation")
    if not conversation:
        return example
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            media_type = item.get("type")
            if media_type == "image":
                img = item.get("image")
                if isinstance(img, str):
                    if "::" in img:
                        item["image"] = _resolve_lmdb_image(img)
                    else:
                        item["image"] = Image.open(img).convert("RGB")
                elif isinstance(img, Image.Image):
                    item["image"] = img.convert("RGB")
            elif media_type == "video":
                vid = item.get("video")
                if isinstance(vid, str):
                    if preserve_video_metadata:
                        frames, fps, indices = _read_video_frames(
                            vid,
                            processor,
                            frame_indices=item.get("frame_indices"),
                            return_metadata=True,
                        )
                        item["video"] = frames
                        item["_video_fps"] = fps
                        item["_frame_indices"] = indices
                    else:
                        item["video"] = _read_video_frames(
                            vid,
                            processor,
                            frame_indices=item.get("frame_indices"),
                        )
                elif isinstance(vid, (list, tuple)) and vid and all(isinstance(f, str) for f in vid):
                    # Pre-extracted frame sequence: every entry is an image path,
                    # so frames load directly -- no video decoder required.
                    temporal_patch_size = 2
                    if processor is not None and hasattr(processor, "video_processor"):
                        temporal_patch_size = getattr(processor.video_processor, "temporal_patch_size", 2)
                    frames = [Image.open(f).convert("RGB") for f in vid]
                    indices = list(range(len(frames)))
                    # Pad to temporal_patch_size alignment by repeating the last
                    # frame, mirroring the decord path above.
                    remainder = len(indices) % temporal_patch_size
                    if remainder != 0:
                        pad = temporal_patch_size - remainder
                        frames.extend([frames[-1]] * pad)
                        indices.extend([indices[-1]] * pad)
                    item["video"] = frames
                    if preserve_video_metadata:
                        fps = item.get("fps")
                        if fps is None and processor is not None and hasattr(processor, "video_processor"):
                            fps = getattr(processor.video_processor, "fps", None)
                        if fps is None:
                            raise ValueError(
                                "fps is required for pre-extracted frame sequences when "
                                "preserve_video_metadata is enabled, but none could be resolved. "
                                "Set an 'fps' field on the video content item, or configure 'fps' "
                                "on the processor's video_processor so that correct timestamps can "
                                "be built for the extracted frames."
                            )
                        item["_video_fps"] = fps
                        item["_frame_indices"] = indices
    return example


def _build_video_metadata(conversation):
    """Build a list of ``VideoMetadata`` from preserved ``_video_fps`` / ``_frame_indices``.

    ``_preload_media(preserve_video_metadata=True)`` stores these on each
    video content item.  Passing the resulting metadata to the processor
    ensures correct timestamps and prevents double frame-sampling.

    Returns an empty list if no video metadata is found.
    """
    from transformers.video_utils import VideoMetadata

    metadata_list = []
    for msg in conversation:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video":
                continue
            fps = item.get("_video_fps")
            indices = item.get("_frame_indices")
            video = item.get("video")
            if fps is not None and indices is not None and video is not None:
                metadata_list.append(
                    VideoMetadata(
                        total_num_frames=len(video),
                        fps=fps,
                        frames_indices=list(indices),
                    )
                )
    return metadata_list


def default_stop_tokens(processor) -> Iterable[str]:
    """Return default generation stop tokens for a processor tokenizer."""
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token = getattr(tokenizer, "eos_token", None) if tokenizer is not None else None
    candidates = [
        "<end_of_turn>",
        "<|im_end|>",
        "<|eot_id|>",
    ]
    if eos_token is not None:
        candidates.append(eos_token)
    return tuple(candidates)


def json2token(obj, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence.

    From NeMo's automodel_datasets.py
    """
    if type(obj) is dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        output = ""
        keys = sorted(obj.keys(), reverse=True) if sort_json_key else obj.keys()
        for k in keys:
            output += rf"<s_{k}>" + json2token(obj[k], sort_json_key) + rf"</s_{k}>"
        return output
    if type(obj) is list:
        return r"<sep/>".join([json2token(item, sort_json_key) for item in obj])
    return str(obj)


def process_text_batch(
    processor,
    texts: list[str],
    images: list | None = None,
) -> dict[str, torch.Tensor]:
    """
    Process a batch of texts and optionally images.

    Args:
        processor: The processor to use for tokenization and image processing
        texts: List of text strings to process
        images: Optional list of images to process

    Returns:
        Dict containing processed batch data
    """
    if images is not None:
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )
        if "pixel_values" in batch:
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    else:
        batch = processor(
            text=texts,
            padding=True,
            return_tensors="pt",
        )

    return batch


def set_image_pixel_bounds(processor, *, max_pixels: int | None = None, min_pixels: int | None = None) -> None:
    """Clamp a VLM processor's per-image resolution.

    A draft-training batch is truncated to ``seq_length`` and the assistant turn
    the draft is supervised on sits at the *end* of the sequence. Qwen VL
    processors default to a per-image ``max_pixels`` large enough to emit more
    vision tokens than the whole budget, in which case truncation removes the
    assistant turn outright, the label scan finds no assistant marker, and the
    resulting ``loss_mask`` is all zeros.

    Answer regeneration must apply the same bounds as training: the answer is
    conditioned on whatever resolution the target saw, so regenerating at a
    higher one supervises the draft against an image it never gets.

    Args:
        processor: A HuggingFace processor; a no-op when it has no image processor.
        max_pixels: Per-image pixel ceiling, or None to keep the default.
        min_pixels: Per-image pixel floor, or None to keep the default.
    """
    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return
    for attribute, value in (("max_pixels", max_pixels), ("min_pixels", min_pixels)):
        if value is None:
            continue
        setattr(image_processor, attribute, int(value))
        # transformers >= 5 reads the bounds off ``size`` when it is present,
        # falling back to the flat attributes only when it is not.
        size = getattr(image_processor, "size", None)
        if isinstance(size, dict) and attribute in size:
            size[attribute] = int(value)
        logger.info("Capped processor %s at %d", attribute, int(value))
