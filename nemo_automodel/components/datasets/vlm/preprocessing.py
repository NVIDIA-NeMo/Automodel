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

"""Shared document/media assembly for native and streaming VLM pipelines."""

from collections.abc import Mapping, Sequence

import torch


def merge_vlm_documents(
    samples: Sequence[Mapping[str, object]],
    *,
    padding_idx: int,
    has_mrope: bool = False,
    sequence_alignment: int = 1,
    return_labels: bool = True,
    return_attention_mask: bool = True,
) -> dict[str, object]:
    """Assemble ordered VLM documents without shifting or selecting bins.

    Native packing supplies already shifted documents; a streaming pipeline can
    supply unshifted documents and shift later in its training converter.

    Args:
        samples: Ordered document mappings. IDs, labels and token-type tensors have
            shape [sequence]; optional scalar positions [sequence] or mRoPE positions
            [3, sequence]. Pixels have
            shape [patches, features], [images, patches, features], or
            [images, channels, height, width], or are a list of image tensors
            [channels, height, width]. Grids are [media, 3], image positions
            [images, patches, coordinate_axes], image sizes [images, 2], and
            timestamps [videos].
        padding_idx: Token ID used only for per-document alignment padding.
        has_mrope: Concatenate supplied multi-axis positions along the sequence axis.
            Scalar positions are also preserved when supplied, otherwise reset per document.
        sequence_alignment: Pad each document to this multiple; mRoPE requires 1.
        return_labels: Concatenate native target labels. Disable when a downstream
            converter constructs labels from the original documents' loss masks.
        return_attention_mask: Construct native indexed attention IDs. Disable when
            a downstream converter derives document IDs from scalar positions.

    Returns:
        Concatenated IDs and token types [tokens]; labels and indexed document mask
        [tokens] are included only when requested. Also returns
        positions [tokens] or [3, tokens]; real and padded subdocument lengths;
        media counts; and media fields concatenated along their leading axis.
        Variable-resolution tensors with image sizes are zero-padded to a common
        height/width; list-layout media stays a list in input order. Inputs are not
        mutated. No final capacity padding, label shift or token batch is made.
    """
    if sequence_alignment < 1:
        raise ValueError(f"sequence_alignment must be at least 1, got {sequence_alignment}.")
    if has_mrope and sequence_alignment > 1:
        raise NotImplementedError("Context-parallel THD packing for multi-axis mRoPE VLMs is not yet implemented.")

    all_input_ids: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    all_attention_mask: list[torch.Tensor] = []
    all_mm_token_type_ids: list[torch.Tensor] = []
    all_position_ids_1d: list[torch.Tensor] = []
    mrope_position_ids_list: list[torch.Tensor] = []
    seq_lens: list[int] = []
    seq_lens_padded: list[int] = []

    pixel_values_list: list[torch.Tensor | list[torch.Tensor]] = []
    patch_pixel_values_list: list[torch.Tensor] = []
    num_patches_list: list[torch.Tensor] = []
    image_grid_thw_list: list[torch.Tensor] = []
    image_position_ids_list: list[torch.Tensor] = []
    image_sizes_list: list[torch.Tensor] = []
    pixel_values_videos_list: list[torch.Tensor] = []
    video_grid_thw_list: list[torch.Tensor] = []
    second_per_grid_ts_list: list[torch.Tensor] = []
    n_images = 0
    n_videos = 0

    for seq_idx, sample in enumerate(samples, start=1):
        ids = torch.as_tensor(sample["input_ids"], dtype=torch.long)
        seq_len = ids.numel()
        padded_seq_len = ((seq_len + sequence_alignment - 1) // sequence_alignment) * sequence_alignment
        pad = padded_seq_len - seq_len
        seq_lens.append(seq_len)
        seq_lens_padded.append(padded_seq_len)
        all_input_ids.append(torch.nn.functional.pad(ids, (0, pad), value=padding_idx))
        if return_labels:
            labels = torch.as_tensor(sample["labels"], dtype=torch.long)
            all_labels.append(torch.nn.functional.pad(labels, (0, pad), value=-100))
        if return_attention_mask:
            all_attention_mask.append(ids.new_full((padded_seq_len,), seq_idx))

        token_types = sample.get("mm_token_type_ids")
        token_types = ids.new_zeros(seq_len) if token_types is None else torch.as_tensor(token_types, dtype=torch.long)
        all_mm_token_type_ids.append(torch.nn.functional.pad(token_types, (0, pad)))

        if has_mrope and "position_ids" in sample:
            mrope_position_ids_list.append(sample["position_ids"])
        elif sample.get("position_ids") is not None:
            positions = torch.as_tensor(sample["position_ids"], dtype=torch.long)
            all_position_ids_1d.append(
                torch.cat((positions, torch.arange(seq_len, padded_seq_len, device=positions.device)))
            )
        else:
            all_position_ids_1d.append(torch.arange(padded_seq_len, device=ids.device))

        if "pixel_values" in sample and sample["pixel_values"] is not None:
            pixel_values_list.append(sample["pixel_values"])
        if "patch_pixel_values" in sample and sample["patch_pixel_values"] is not None:
            patch_pixel_values_list.append(sample["patch_pixel_values"])
        if "num_patches" in sample and sample["num_patches"] is not None:
            num_patches_list.append(torch.as_tensor(sample["num_patches"], dtype=torch.long))
        if "image_grid_thw" in sample and sample["image_grid_thw"] is not None:
            image_grid_thw_list.append(sample["image_grid_thw"])
        if "image_position_ids" in sample and sample["image_position_ids"] is not None:
            image_position_ids_list.append(sample["image_position_ids"])
        if sample.get("image_sizes") is not None:
            image_sizes_list.append(sample["image_sizes"])
        for key in ("image_grid_thw", "image_position_ids", "image_sizes"):
            if sample.get(key) is not None:
                n_images += sample[key].shape[0]
                break
        if "pixel_values_videos" in sample and sample["pixel_values_videos"] is not None:
            pixel_values_videos_list.append(sample["pixel_values_videos"])
        if "video_grid_thw" in sample and sample["video_grid_thw"] is not None:
            n_videos += sample["video_grid_thw"].shape[0]
            video_grid_thw_list.append(sample["video_grid_thw"])
        if "second_per_grid_ts" in sample and sample["second_per_grid_ts"] is not None:
            second_per_grid_ts_list.append(sample["second_per_grid_ts"])

    # No padding here — collater pads to batch-max for efficiency.
    packed = {
        "input_ids": torch.cat(all_input_ids) if all_input_ids else torch.empty(0, dtype=torch.long),
    }
    if return_labels:
        packed["labels"] = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
    if return_attention_mask:
        packed["attention_mask"] = (
            torch.cat(all_attention_mask) if all_attention_mask else torch.empty(0, dtype=torch.long)
        )
    packed.update(
        mm_token_type_ids=torch.cat(all_mm_token_type_ids)
        if all_mm_token_type_ids
        else torch.empty(0, dtype=torch.long),
        seq_lens=seq_lens,
        seq_lens_padded=seq_lens_padded,
        n_images=n_images,
        n_videos=n_videos,
    )

    if has_mrope and mrope_position_ids_list:
        packed["position_ids"] = torch.cat(mrope_position_ids_list, dim=1)
    else:
        packed["position_ids"] = (
            torch.cat(all_position_ids_1d) if all_position_ids_1d else torch.empty(0, dtype=torch.long)
        )

    if (
        image_sizes_list
        and pixel_values_list
        and all(isinstance(value, torch.Tensor) and value.ndim == 4 for value in pixel_values_list)
    ):
        packed["pixel_values"] = pad_image_tensors(pixel_values_list)
    elif pixel_values_list and all(isinstance(value, torch.Tensor) for value in pixel_values_list):
        if all(value.shape[1:] == pixel_values_list[0].shape[1:] for value in pixel_values_list):
            packed["pixel_values"] = torch.cat(pixel_values_list, dim=0)
        else:
            # Variable-resolution images (e.g. dynamic-resolution processors): keep
            # one tensor per image; the THD collater and the model's feature
            # extractor accept the list form.
            packed["pixel_values"] = [image for value in pixel_values_list for image in value]
    elif pixel_values_list and all(isinstance(value, (list, tuple)) for value in pixel_values_list):
        packed["pixel_values"] = [item for value in pixel_values_list for item in value]
    elif pixel_values_list:
        raise TypeError("Packed VLM pixel_values must be consistently tensors or variable-resolution lists.")
    else:
        packed["pixel_values"] = None
    if patch_pixel_values_list:
        packed["patch_pixel_values"] = torch.cat(patch_pixel_values_list, dim=0)
    if num_patches_list:
        packed["num_patches"] = torch.cat(num_patches_list, dim=0)
    if image_sizes_list:
        packed["image_sizes"] = torch.cat(image_sizes_list, dim=0)
    packed["image_grid_thw"] = torch.cat(image_grid_thw_list, dim=0) if image_grid_thw_list else None
    packed["image_position_ids"] = torch.cat(image_position_ids_list, dim=0) if image_position_ids_list else None
    packed["pixel_values_videos"] = torch.cat(pixel_values_videos_list, dim=0) if pixel_values_videos_list else None
    packed["video_grid_thw"] = torch.cat(video_grid_thw_list, dim=0) if video_grid_thw_list else None
    packed["second_per_grid_ts"] = torch.cat(second_per_grid_ts_list, dim=0) if second_per_grid_ts_list else None

    return packed


def pad_image_tensors(values: Sequence[torch.Tensor]) -> torch.Tensor:
    """Zero-pad processor images to a common spatial size without resizing.

    Args:
        values: Image tensors [images, channels, height, width], with matching
            channel counts. Original unpadded sizes travel in `image_sizes`.

    Returns:
        Concatenated images [total_images, channels, max_height, max_width].
        Pixels stay at their original top-left coordinates; inputs are unchanged.
    """
    if not values or any(value.ndim != 4 for value in values):
        raise ValueError("Image padding requires nonempty [images, channels, height, width] tensors")
    height = max(value.shape[-2] for value in values)
    width = max(value.shape[-1] for value in values)
    return torch.cat(
        [torch.nn.functional.pad(value, (0, width - value.shape[-1], 0, height - value.shape[-2])) for value in values],
        dim=0,
    )


def merge_media_values(
    values: Sequence[torch.Tensor | Sequence[torch.Tensor]],
    *,
    dtype: torch.dtype | None = None,
) -> torch.Tensor | list[torch.Tensor]:
    """Collate fixed-shape patches or variable-resolution images without token slicing.

    Args:
        values: Ordered per-document media. Tensors have shape [patches, features],
            [images, patches, features], or [images, channels, height, width] with
            matching trailing dimensions.
            Ragged media use lists of [channels, height, width] tensors.
        dtype: Optional caller-owned conversion; None retains processor precision.

    Returns:
        A tensor concatenated along the leading axis, or a flattened list of image
        tensors in document/image order. Inputs are not mutated; ragged list entries
        can alias inputs when no dtype conversion is needed.
    """
    if not values:
        raise ValueError("Media merge requires at least one value.")
    if all(isinstance(value, torch.Tensor) for value in values):
        return torch.cat(values, dim=0).to(dtype=dtype)
    # At least one pack carries variable-resolution media as a list: flatten
    # every pack to one tensor per image (a stacked [N, ...] tensor splits along dim 0).
    flat = []
    for value in values:
        if isinstance(value, torch.Tensor):
            flat.extend(value.to(dtype=dtype))
        elif isinstance(value, (list, tuple)):
            flat.extend(item.to(dtype=dtype) if isinstance(item, torch.Tensor) else item for item in value)
        else:
            raise TypeError(f"VLM media values must be tensors or lists, got {type(value).__name__}.")
    return flat


def collate_position_ids(values: Sequence[torch.Tensor], *, sequence_length: int) -> torch.Tensor:
    """Pad scalar or multi-axis positions and stack documents without changing positions.

    Args:
        values: Position tensors [sequence] or [position_axes, sequence], with the
            same rank and number of axes across documents. Input tensors are not mutated.
        sequence_length: Padded sequence length, at least each document's length.

    Returns:
        Positions [batch, sequence_length] for scalar inputs, or
        [position_axes, batch, sequence_length] for multi-axis inputs. Padding is zero;
        original dtype and device are preserved.
    """
    if not values:
        raise ValueError("Position collation requires at least one document.")
    rank = values[0].ndim
    if rank not in (1, 2):
        raise ValueError("Position IDs must have shape [sequence] or [position_axes, sequence].")
    padded = []
    for value in values:
        if value.ndim != rank or value.shape[:-1] != values[0].shape[:-1]:
            raise ValueError("Documents must have matching position-axis layouts.")
        padding = sequence_length - value.shape[-1]
        if padding < 0:
            raise ValueError("Position IDs exceed the requested sequence_length.")
        padded.append(torch.cat((value, value.new_zeros((*value.shape[:-1], padding))), dim=-1))
    return torch.stack(padded, dim=0 if rank == 1 else 1)
