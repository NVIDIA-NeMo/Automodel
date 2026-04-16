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

"""
Mock VLM conversation dataset for benchmarking and testing.

Generates synthetic image(s) and minimal conversations in the standard
Automodel conversation format, compatible with ``PreTokenizedDatasetWrapper``
and any HF ``AutoProcessor`` that supports the conversation schema.

The images are random-noise PIL images — no real data download is needed.
The processor / vision encoder processes them through the normal pipeline,
so this exercises the full VLM training path end-to-end.

When used with ``pretokenize: true``, ``truncate: true``, and ``max_length``
in the dataset config, ``PreTokenizedDatasetWrapper`` tokenizes each sample
and truncates to exactly ``max_length`` tokens.  The mock response is
sized from ``max_length`` so that truncation always produces a full-length
sequence.
"""

from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

_WORD_POOL = (
    "the image shows a landscape with mountains and rivers flowing through "
    "green valleys while clouds drift across the blue sky and birds fly "
    "overhead casting shadows on the ground below where flowers bloom in "
    "vibrant colors and trees sway gently in the warm breeze as sunlight "
    "filters through the leaves creating patterns of light and shadow on "
    "the forest floor where small animals scurry about gathering food for "
    "the coming winter season while insects buzz around the wildflowers"
).split()


def _make_random_image(
    rng: np.random.Generator,
    size: Tuple[int, int] = (256, 256),
) -> Image.Image:
    """Create a random-noise RGB PIL image."""
    w, h = size
    pixels = rng.integers(low=0, high=256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def _generate_response(rng: np.random.Generator, num_words: int) -> str:
    """Generate a dummy response of *num_words* words from a fixed pool."""
    words = [_WORD_POOL[i] for i in rng.integers(0, len(_WORD_POOL), size=num_words)]
    return " ".join(words)


def build_mock_vlm_dataset(
    *,
    num_samples: int = 10,
    num_images_per_sample: int = 1,
    image_size: Tuple[int, int] = (256, 256),
    prompt: str = "Describe this image.",
    responses: Optional[List[str]] = None,
    max_length: Optional[int] = None,
    seed: int = 0,
    **kwargs,
) -> list:
    """Build a mock VLM dataset in Automodel conversation format.

    Each sample is a dict with a ``"conversation"`` key whose value is a list
    of user/assistant message dicts.  User messages contain one or more
    ``{"type": "image", "image": <PIL.Image>}`` items followed by a text prompt.
    Assistant messages contain a single text response.

    This is the same format produced by ``make_rdr_dataset``,
    ``make_unimm_chat_dataset``, and ``make_meta_dataset``, so the returned
    list can be fed directly to ``PreTokenizedDatasetWrapper``.

    When ``max_length`` is set and ``responses`` is ``None``, each sample's
    assistant response is generated with ``max_length`` words — guaranteed
    to exceed ``max_length`` tokens so that ``PreTokenizedDatasetWrapper``
    with ``truncate=True`` produces exactly ``max_length`` tokens per sample.

    Args:
        num_samples: Number of conversation examples to generate.
        num_images_per_sample: Number of random images per user turn.
        image_size: ``(width, height)`` of each generated image.
        prompt: Text prompt appended after the image(s) in the user turn.
        responses: Optional list of assistant responses.  Cycled over samples.
        max_length: Target sequence length.  When set (and ``responses`` is
            ``None``), generates a response of ``max_length`` words per sample
            so the tokenized sequence always exceeds ``max_length`` tokens.
        seed: Random seed for reproducibility.

    Returns:
        A list of dicts, each with a single ``"conversation"`` key.
    """
    rng = np.random.default_rng(seed=seed)
    response_words = max_length if max_length is not None else 256
    dataset = []

    for i in range(num_samples):
        images = [_make_random_image(rng, image_size) for _ in range(num_images_per_sample)]

        user_content = [{"type": "image", "image": img} for img in images]
        user_content.append({"type": "text", "text": prompt})

        if responses is not None:
            response_text = responses[i % len(responses)]
        else:
            response_text = _generate_response(rng, response_words)

        sample = {
            "conversation": [
                {"role": "user", "content": user_content},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": response_text}],
                },
            ],
        }
        dataset.append(sample)

    return dataset
