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

"""Processor parity: native ``Inkling2Processor`` vs the Transformers reference.

Runs the native processor on synthetic images (RGB, RGBA, grayscale, NumPy channels-last, sizes
that are not tile multiples), on text with image placeholders, and on chats with images through
``apply_chat_template``, and requires ``input_ids``, ``attention_mask``, ``pixel_values`` and
``image_grid_thw`` to be identical to Transformers' ``Inkling2Processor``.

The reference needs a Transformers build with ``inkling2`` (huggingface/transformers@109a7d19bf or
later), while the native processor is meant for the pinned 5.18.0. To compare across the two
installs, save the native outputs with the pinned release and compare them under the reference::

    python tests/unit_tests/models/inkling2/parity_check_inkling2_processing.py --model DIR --save native.pt
    python tests/unit_tests/models/inkling2/parity_check_inkling2_processing.py --model DIR --compare native.pt

``DIR`` is a model ID or a snapshot with the tokenizer, chat template and ``processor_config.json``.
Without ``--compare``, a reference-capable install compares in-process native outputs instead.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image

from nemo_automodel.components.models.inkling2.processing import Inkling2Processor

_IMAGE = "<|unused_200054|>"


def build_images() -> dict[str, Image.Image | np.ndarray]:
    """Return deterministic synthetic images covering each conversion path."""
    rng = np.random.default_rng(0)
    rgba = rng.integers(0, 256, size=(37, 50, 4), dtype=np.uint8)
    rgba[..., 3] = np.linspace(0, 255, 50, dtype=np.uint8)[None, :]
    return {
        "rgb_83x121": Image.fromarray(rng.integers(0, 256, size=(83, 121, 3), dtype=np.uint8)),
        "rgba_37x50": Image.fromarray(rgba, mode="RGBA"),
        "gray_40x80": Image.fromarray(rng.integers(0, 256, size=(40, 80), dtype=np.uint8), mode="L"),
        "numpy_rgb_45x45": rng.integers(0, 256, size=(45, 45, 3), dtype=np.uint8),
        "numpy_rgba_41x39": rng.integers(0, 256, size=(41, 39, 4), dtype=np.uint8),
    }


def _chat(images: list[Image.Image | np.ndarray], question: str, answer: str) -> list[dict]:
    content = [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": question}]
    return [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]


def run_cases(processor: object) -> dict[str, dict[str, torch.Tensor]]:
    """Run every case and return ``{case: {output name: tensor}}``."""
    images = build_images()
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for name, image in images.items():
        result = processor.image_processor(image, return_tensors="pt")
        outputs[f"image/{name}"] = {key: result[key] for key in ("pixel_values", "image_grid_thw")}

    text = f"Compare {_IMAGE} with {_IMAGE} and {_IMAGE}."
    result = processor(
        text=[text],
        images=[images["rgb_83x121"], images["rgba_37x50"], images["gray_40x80"]],
        return_tensors="pt",
    )
    outputs["text_with_placeholders"] = dict(result)

    chats = {
        "chat_one_image": _chat([images["rgb_83x121"]], "What is shown?", "Noise."),
        "chat_two_images": _chat([images["numpy_rgba_41x39"], images["gray_40x80"]], "Which is larger?", "The 2nd."),
    }
    for name, chat in chats.items():
        result = processor.apply_chat_template(chat, tokenize=True, return_dict=True, return_tensors="pt")
        outputs[name] = dict(result)
    return outputs


def compare(native: dict[str, dict[str, torch.Tensor]], reference: dict[str, dict[str, torch.Tensor]]) -> bool:
    """Print one row per output and return whether every output is identical."""
    ok = native.keys() == reference.keys()
    print("| case | output | shape | identical |")
    print("|---|---|---|---|")
    for case in reference:
        if native.get(case, {}).keys() != reference[case].keys():
            print(f"| {case} | keys differ: {sorted(native.get(case, {}))} vs {sorted(reference[case])} | | no |")
            ok = False
            continue
        for key, expected in reference[case].items():
            actual = native[case][key]
            same = actual.shape == expected.shape and actual.dtype == expected.dtype and torch.equal(actual, expected)
            ok &= same
            print(f"| {case} | {key} | {list(expected.shape)} | {'yes' if same else 'no'} |")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="thinkingmachines/Inkling-2-Small", help="model ID or snapshot directory")
    parser.add_argument("--save", help="write the native outputs to this file")
    parser.add_argument("--compare", help="compare the reference against native outputs saved with --save")
    args = parser.parse_args()

    native_outputs = (
        torch.load(args.compare, weights_only=True)
        if args.compare
        else run_cases(Inkling2Processor.from_pretrained(args.model))
    )
    if args.save:
        torch.save(native_outputs, args.save)
        print(f"saved native outputs of {len(native_outputs)} cases to {args.save}")
        return

    from transformers.models.inkling2.processing_inkling2 import Inkling2Processor as HFInkling2Processor

    reference_outputs = run_cases(HFInkling2Processor.from_pretrained(args.model))
    if not compare(native_outputs, reference_outputs):
        raise SystemExit("FAILED: native and reference processor outputs differ")
    print("PASSED: all outputs identical")


if __name__ == "__main__":
    main()
