# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0.

import base64
import dataclasses
import os
from dataclasses import field
from io import BytesIO
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import requests
import torch
from PIL import Image
from transformers import BatchEncoding, PretrainedConfig, ProcessorMixin
from transformers.image_processing_base import BatchFeature
from transformers.image_processing_utils_fast import BaseImageProcessorFast, divide_to_patches
from transformers.image_utils import (
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    SizeDict,
    get_image_size,
    make_list_of_images,
)
from transformers.utils import TensorType

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


@dataclasses.dataclass
class Conversation:
    """Manages prompt construction with system messages and multi-turn dialogues."""

    # System instruction prepended to prompts
    system_message: str = ""
    # Role identifiers for dialogue turns
    roles: Tuple[str, str] = ("", "")
    # Message history as (role, content) pairs
    messages: List[List[str]] = field(default_factory=list)
    # Separator token between messages
    sep: str = ""
    # Token IDs that trigger generation stopping
    stop_token_ids: List[int] = None

    def get_prompt(self) -> str:
        """Construct the formatted prompt string from system message and dialogue history."""
        ret = self.system_message + self.sep
        for role, message in self.messages:
            if message:
                ret += role + message + self.sep
            else:
                ret += role
        return ret

    def append_message(self, role: str, message: str):
        """Add a message turn to the dialogue history."""
        self.messages.append([role, message])


def get_conv_template(name: str) -> Conversation:
    """Initialize a conversation instance with default configuration."""
    return Conversation(
        stop_token_ids=[128259, 128001],
    )


def load_image(image):
    """Load an image from a file, a URL, a base64 string, or a bytes object."""
    if isinstance(image, Image.Image):
        return image
    elif isinstance(image, str) and os.path.exists(image):
        return Image.open(image)
    elif isinstance(image, dict):
        if "disk_path" in image:
            return Image.open(image["disk_path"])
        elif "base64" in image:
            return Image.open(BytesIO(base64.b64decode(image["base64"])))
        elif "url" in image:
            response = requests.get(image["url"])
            return Image.open(BytesIO(response.content))
        elif "bytes" in image:
            return Image.open(BytesIO(image["bytes"]))
        else:
            raise ValueError(f"Invalid image: {image}")
    else:
        raise ValueError(f"Invalid image: {image}")


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    """
    previous version mainly foucs on ratio.
    We also consider area ratio here.
    """
    best_factor = float("-inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        area_ratio = (ratio[0] * ratio[1] * image_size * image_size) / area
        # new area > 60% of original image area is enough.
        factor_based_on_area_n_ratio = min(area_ratio, 0.6) * min(
            target_aspect_ratio / aspect_ratio, aspect_ratio / target_aspect_ratio
        )

        if factor_based_on_area_n_ratio > best_factor:
            best_factor = factor_based_on_area_n_ratio
            best_ratio = ratio

    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    """Dynamically preprocess an image into a list of image tiles, with a thumbnail if needed."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


class LlamaNemotronVLImageProcessor(BaseImageProcessorFast):
    """Fast batched image processor for Llama Nemotron VL retrieval inputs."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 512,
        max_num_tiles: int = 6,
        use_thumbnail: bool = True,
        dynamic_image_size: bool = True,
        norm_type: str = "siglip",
        resample: Optional[Union[PILImageResampling, int]] = None,
        **kwargs,
    ):
        if norm_type == "imagenet":
            image_mean, image_std = IMAGENET_MEAN, IMAGENET_STD
        elif norm_type == "siglip":
            image_mean, image_std = SIGLIP_MEAN, SIGLIP_STD
        else:
            raise ValueError(f"Invalid norm_type: {norm_type!r}. Must be 'imagenet' or 'siglip'.")

        kwargs.setdefault("do_rescale", True)
        kwargs.setdefault("rescale_factor", 1 / 255)
        kwargs.setdefault("do_normalize", True)
        kwargs.setdefault("image_mean", image_mean)
        kwargs.setdefault("image_std", image_std)
        kwargs.setdefault("resample", resample if resample is not None else PILImageResampling.BICUBIC)

        super().__init__(**kwargs)
        self.image_size = image_size
        self.max_num_tiles = max_num_tiles
        self.use_thumbnail = use_thumbnail
        self.dynamic_image_size = dynamic_image_size
        self.norm_type = norm_type

    def dynamic_preprocess(
        self,
        image: torch.Tensor,
        image_size: int = 512,
        max_num_tiles: int = 6,
        use_thumbnail: bool = True,
        resample: Optional[Union[PILImageResampling, int]] = None,
    ) -> List[torch.Tensor]:
        """Split one channel-first image tensor into dynamically sized square tiles."""
        resample = resample if resample is not None else self.resample
        orig_height, orig_width = get_image_size(image, channel_dim=ChannelDimension.FIRST)
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(1, max_num_tiles + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num_tiles and i * j >= 1
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size,
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        resized_img = self.resize(image, SizeDict(height=target_height, width=target_width), resample=resample)
        patches = divide_to_patches(resized_img, image_size)
        if use_thumbnail and len(patches) != 1:
            patches.append(self.resize(image, SizeDict(height=image_size, width=image_size), resample=resample))

        return patches

    def _preprocess(
        self,
        images: ImageInput,
        image_size: Optional[int] = None,
        max_num_tiles: Optional[int] = None,
        use_thumbnail: Optional[bool] = None,
        dynamic_image_size: Optional[bool] = None,
        do_rescale: Optional[bool] = None,
        rescale_factor: Optional[float] = None,
        do_normalize: Optional[bool] = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        resample: Optional[Union[PILImageResampling, int]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchFeature:
        image_size = image_size if image_size is not None else self.image_size
        max_num_tiles = max_num_tiles if max_num_tiles is not None else self.max_num_tiles
        use_thumbnail = use_thumbnail if use_thumbnail is not None else self.use_thumbnail
        dynamic_image_size = dynamic_image_size if dynamic_image_size is not None else self.dynamic_image_size
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        resample = resample if resample is not None else self.resample

        all_patches = []
        num_patches = []
        for image in make_list_of_images(images):
            if dynamic_image_size:
                patches = self.dynamic_preprocess(image, image_size, max_num_tiles, use_thumbnail, resample=resample)
            else:
                patches = [self.resize(image, SizeDict(height=image_size, width=image_size), resample=resample)]
            all_patches.extend(patches)
            num_patches.append(len(patches))

        pixel_values = torch.stack(all_patches, dim=0)
        pixel_values = self.rescale_and_normalize(
            pixel_values,
            do_rescale,
            rescale_factor,
            do_normalize=do_normalize,
            image_mean=image_mean,
            image_std=image_std,
        )

        return BatchFeature(data={"pixel_values": pixel_values, "num_patches": num_patches}, tensor_type=return_tensors)


class LlamaNemotronVLProcessorConfig(PretrainedConfig):
    """Dummy Configuration for LlamaNemotronVLProcessor,
    just to register the processor with AutoProcessor."""

    pass


class LlamaNemotronVLProcessor(ProcessorMixin):
    """Processor for LlamaNemotronVL model."""

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        tokenizer: Any,
        config: Optional[LlamaNemotronVLProcessorConfig] = None,
        q_max_length: Optional[int] = None,
        p_max_length: Optional[int] = None,
        pad_to_multiple_of: Optional[int] = None,
        query_prefix: str = "query:",
        passage_prefix: str = "passage:",
        max_input_tiles: int = 6,
        num_image_token: int = 256,
        dynamic_image_size: bool = True,
        image_size: int = 512,
        use_thumbnail: bool = True,
        template: str = "bidirectional-llama-retriever",
        num_channels: int = 3,
        norm_type: str = "siglip",
        system_message: str = "",
        padding: Union[bool, str] = True,
        **kwargs,
    ):
        tokenizer.padding_side = "left"
        tokenizer.model_input_names = tokenizer.model_input_names + ["pixel_values"]
        self.tokenizer = tokenizer

        self.q_max_length = q_max_length
        self.p_max_length = p_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.max_input_tiles = max_input_tiles
        self.num_image_token = num_image_token
        self.dynamic_image_size = dynamic_image_size
        self.image_size = image_size
        self.use_thumbnail = use_thumbnail
        self.template = template
        self.num_channels = num_channels
        self.norm_type = norm_type
        self.system_message = system_message
        self.padding = padding
        self.image_processor = LlamaNemotronVLImageProcessor(
            image_size=image_size,
            max_num_tiles=max_input_tiles,
            use_thumbnail=use_thumbnail,
            dynamic_image_size=dynamic_image_size,
            norm_type=norm_type,
        )

        super().__init__(self.tokenizer)

    def __call__(
        self,
        text: Optional[List[str]] = None,
        images: Optional[List[Any]] = None,
        text_kwargs: Optional[Dict[str, Any]] = None,
        images_kwargs: Optional[Dict[str, Any]] = None,
        common_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process text and/or image inputs into model-ready features.
        This method provides compatibility with the standard HuggingFace processor interface
        used by Sentence Transformers. For image inputs, it delegates to process_documents.
        For text-only inputs, it tokenizes directly (assuming any task prefix has already been
        applied by the caller).
        Args:
            text: List of text strings. For text-only inputs, these should already include
                any task prefix (e.g. "query: " or "passage: ").
            images: List of PIL Images for document encoding.
            text_kwargs: Keyword arguments for text processing (e.g. padding, truncation).
            images_kwargs: Keyword arguments for image processing (unused, for API compat).
            common_kwargs: Common keyword arguments (e.g. return_tensors).
            **kwargs: Additional keyword arguments (ignored).
        Returns:
            Dict with "input_ids", "attention_mask", and optionally "pixel_values".
        """
        text_kwargs = text_kwargs or {}
        common_kwargs = common_kwargs or {}
        tokenizer_kwargs = {**common_kwargs, **text_kwargs}
        return_tensors = tokenizer_kwargs.pop("return_tensors", "pt")
        padding = tokenizer_kwargs.pop("padding", self.padding)
        truncation = tokenizer_kwargs.pop("truncation", True)

        if images is not None:
            # Image or image+text: delegate to process_documents which handles
            # image tiling, image token creation, and the passage prefix.
            if text is not None:
                documents = [{"image": img, "text": t} for img, t in zip(images, text)]
            else:
                documents = [{"image": img, "text": ""} for img in images]
            return self.process_documents(
                documents,
                return_tensors=return_tensors,
                padding=padding,
                truncation=truncation,
            )

        # Text-only: just tokenize (the caller has already applied any prompt prefix)
        max_length = None
        if truncation:
            max_length = self.p_max_length or self.tokenizer.model_max_length
        return self.tokenizer(
            text,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

    def process_documents(
        self,
        documents: Union[Dict, List[Dict]],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        pixel_values_layout: Literal["per_image", "flat_tiles"] = "flat_tiles",
        **kwargs,
    ) -> Dict[str, Any]:
        """Process documents into model inputs with tokenized text and pixel values.
        Args:
            documents: Either a dict with "images" and "texts" lists, or a list of
                dicts each with "image" and "text" keys. Images can be PIL Images,
                file paths, or None/empty string for text-only documents.
            return_tensors: Output format — "pt" for PyTorch tensors, "np" for numpy arrays.
            padding: Padding strategy passed to the tokenizer. Defaults to the value
                set in the processor constructor.
            truncation: Whether to truncate sequences to p_max_length.
            pixel_values_layout: How to structure the pixel values output:
                - "flat_tiles": All image tiles concatenated into a single tensor of shape
                  (total_tiles, C, H, W). Different images may contribute different numbers
                  of tiles. None if no images are present. This is the format expected by
                  the model's forward() method.
                - "per_image": A list aligned with the input documents, where each entry
                  is either a tensor of shape (num_tiles, C, H, W) or None.
        Returns:
            Dict with "input_ids", "attention_mask", and "pixel_values".
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        if isinstance(documents, dict):
            images = documents["images"]
            texts = documents["texts"]
            assert len(texts) == len(images)
        elif isinstance(documents, list):
            images = [pair["image"] for pair in documents]
            texts = [pair["text"] for pair in documents]
        else:
            raise ValueError("The documents need to be a dict or list of dicts")

        contents = []
        pil_images_by_idx = {}
        max_input_tile_by_idx = {}
        for idx, (image, text) in enumerate(zip(images, texts)):
            prefix = ""
            if image is not None and image != "":
                pil_image = load_image(image)
                pil_images_by_idx[idx] = pil_image.convert("RGB") if pil_image.mode != "RGB" else pil_image
                prefix = "<image>"
                max_input_tile_by_idx[idx] = self.max_input_tiles

            # ToDo: Order is hardcoded and different than before. No \n after <image>
            content = text
            if prefix != "":
                content = prefix + " " + content
            if self.passage_prefix:
                content = self.passage_prefix + " " + content
            contents.append(content)

        assert len(max_input_tile_by_idx) == len(pil_images_by_idx), (
            "The number of max_input_tile_by_idx and pil_images_by_idx should be the same."
        )

        pixel_values_by_idx = {}
        if pil_images_by_idx:
            image_indices = list(pil_images_by_idx.keys())
            self.image_processor.image_size = self.image_size
            self.image_processor.max_num_tiles = self.max_input_tiles
            self.image_processor.use_thumbnail = self.use_thumbnail
            self.image_processor.dynamic_image_size = self.dynamic_image_size
            image_features = self.image_processor(
                [pil_images_by_idx[idx] for idx in image_indices], return_tensors="pt"
            )
            pixel_values = image_features["pixel_values"].to(dtype=torch.bfloat16)
            num_patches = image_features["num_patches"]
            if isinstance(num_patches, torch.Tensor):
                num_patches = num_patches.tolist()

            offset = 0
            for idx, patch_count in zip(image_indices, num_patches):
                patch_count = int(patch_count)
                pixel_values_by_idx[idx] = pixel_values[offset : offset + patch_count]
                offset += patch_count

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        IMG_START_TOKEN = "<img>"
        IMG_END_TOKEN = "</img>"
        IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

        content_prompts = []
        pixel_values_list = []
        for i, content in enumerate(contents):
            pixel_values = pixel_values_by_idx.get(i)
            pixel_values_list.append(pixel_values)

            if pixel_values is not None and "<image>" not in content:
                content = "<image> " + content

            # Reseting conversation messages
            template.messages.clear()

            # TODO: do we need this template?
            template.append_message(template.roles[0], content)  # user
            template.append_message(template.roles[1], None)  # assistant
            content_prompt = template.get_prompt()

            if pixel_values is not None:
                num_patches = pixel_values.shape[0]
                image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
                content_prompt = content_prompt.replace("<image>", image_tokens, 1)
            else:
                content_prompt = content_prompt.replace("<image>", "", 1)

            content_prompts.append(content_prompt)

        max_length = None
        if truncation:
            max_length = self.p_max_length or self.tokenizer.model_max_length

        if padding is None:
            padding = self.padding

        model_inputs = self.tokenizer(
            content_prompts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        image_token_indices = None
        if return_tensors == "pt" and pil_images_by_idx:
            img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
            selected = (model_inputs["input_ids"] == img_context_token_id).nonzero(as_tuple=False)
            if selected.numel() > 0:
                image_token_indices = selected[:, 0] * model_inputs["input_ids"].shape[1] + selected[:, 1]
            else:
                image_token_indices = torch.empty(0, dtype=torch.long)

        if pixel_values_layout == "flat_tiles":
            pixel_values_list = [pv for pv in pixel_values_list if pv is not None]
            if len(pixel_values_list) > 1:
                pixel_values_squeezed = torch.concat(pixel_values_list, axis=0)
            elif len(pixel_values_list) == 1:
                pixel_values_squeezed = pixel_values_list[0]
            else:
                pixel_values_squeezed = None

            if pixel_values_squeezed is not None and return_tensors == "np":
                pixel_values_return_value = pixel_values_squeezed.to(dtype=torch.float16).cpu().numpy()
            else:
                pixel_values_return_value = pixel_values_squeezed
        elif pixel_values_layout == "per_image":
            if return_tensors == "np":
                pixel_values_return_value = [
                    pv.to(dtype=torch.float16).cpu().numpy() if pv is not None else None for pv in pixel_values_list
                ]
            else:
                pixel_values_return_value = pixel_values_list
        else:
            raise ValueError(
                f"Invalid pixel_values_layout: {pixel_values_layout!r}. Must be 'flat_tiles' or 'per_image'."
            )

        batch_docs = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "pixel_values": pixel_values_return_value,
        }
        if image_token_indices is not None:
            batch_docs["image_token_indices"] = image_token_indices

        return batch_docs

    def process_queries(
        self,
        queries: List[str],
        return_tensors: Literal["pt", "np"] = "pt",
        padding: bool | str | None = None,
        truncation: bool = True,
        **kwargs,
    ) -> BatchEncoding:
        """Process queries into model inputs with tokenized text.
        Args:
            queries: List of query strings.
            return_tensors: Output format — "pt" for PyTorch tensors, "np" for numpy arrays.
            padding: Padding strategy passed to the tokenizer. Defaults to the value
                set in the processor constructor.
            truncation: Whether to truncate sequences to q_max_length.
        Returns:
            Dict with "input_ids" and "attention_mask".
        """
        if return_tensors not in ("pt", "np"):
            raise ValueError(f"Invalid return_tensors: {return_tensors!r}. Must be 'pt' or 'np'.")

        template = get_conv_template(self.template)
        template.system_message = self.system_message

        query_prompts = []
        for query in queries:
            if self.query_prefix:
                query = f"{self.query_prefix} {query}"

            # Reseting conversation messages
            template.messages.clear()

            template.append_message(template.roles[0], query)  # user
            template.append_message(template.roles[1], None)  # assistant
            query_prompt = template.get_prompt()

            query_prompts.append(query_prompt)

        max_length = None
        if truncation:
            max_length = self.q_max_length or self.tokenizer.model_max_length

        if padding is None:
            padding = self.padding

        batch_query = self.tokenizer(
            query_prompts,
            truncation=truncation,
            max_length=max_length,
            padding=padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        return batch_query

    def process_queries_documents_biencoder(self, features: Dict, **kwargs) -> Dict[str, Any]:
        """
        (Pdb) features
        [{'image': [<PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C3A0>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C580>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C940>], 'text': ['passage: ', 'passage: ', 'passage: '], 'question': "query: What change did Carl Rey suggest for the Strategic Plan's website objective deadline?"}, {'image': [<PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C0D0>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5DC00>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5EBF0>], 'text': ['passage: ', 'passage: ', 'passage: '], 'question': 'query: What are the name and TIN requirements for individuals with real estate transactions?'}, {'image': [<PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5D390>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C850>, <PIL.Image.Image image mode=RGB size=1275x1650 at 0x155059A5C070>], 'text': ['passage: ', 'passage: ', 'passage: '], 'question': 'query: How does Richard Hooker view human inclinations?'}]
        """
        queries = []
        pos_neg_text_batch = []
        pos_neg_image_batch = []
        for feature in features:
            queries.append(feature["question"])
            pos_neg_text_batch.extend(feature["doc_text"])
            pos_neg_image_batch.extend(feature["doc_image"])

        query_batch_dict = self.process_queries(queries, **kwargs)
        doc_batch_dict = self.process_documents({"images": pos_neg_image_batch, "texts": pos_neg_text_batch}, **kwargs)

        merged_batch_dict = self.merge_batch_dict(query_batch_dict, doc_batch_dict)
        merged_batch_dict = self.add_dummy_labels(queries, merged_batch_dict)
        return merged_batch_dict

    def merge_batch_dict(self, query_batch_dict, doc_batch_dict):
        q_prefix, d_prefix = "q_", "d_"
        # merge into a single BatchEncoding by adding prefix
        merged_batch_dict = {}
        for k in list(query_batch_dict.keys()):
            merged_batch_dict[q_prefix + k] = query_batch_dict[k]
            del query_batch_dict[k]
        for k in list(doc_batch_dict.keys()):
            merged_batch_dict[d_prefix + k] = doc_batch_dict[k]
            del doc_batch_dict[k]
        return merged_batch_dict

    def add_dummy_labels(self, questions, merged_batch_dict):
        # dummy placeholder for field "labels", won't use it to compute loss
        labels = torch.zeros(len(questions), dtype=torch.long)
        merged_batch_dict["labels"] = labels
        return merged_batch_dict


def _register_with_hf_auto_classes():
    LlamaNemotronVLProcessor.register_for_auto_class("AutoProcessor")


_register_with_hf_auto_classes()
