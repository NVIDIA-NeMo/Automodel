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

"""Arabic OCR dataset: paired page images and their transcriptions.

Expects a flat directory of ``<id>.png`` / ``<id>.md`` pairs, where the ``.md``
file holds the ground-truth transcription of the page image. Despite the
extension the transcriptions are plain text, not structured Markdown.

Emits the same ``{"conversation": [...]}`` shape as the other VLM datasets in
this package, so it works with ``default_collate_fn`` unchanged. Images are
opened lazily in ``__getitem__`` (``Image.open`` reads only the header) and
decoded by the DataLoader worker, so building the dataset does not hold 2000
decoded pages -- or 2000 open file handles -- in memory.

Downscaling is deliberately NOT done here: cap the pixel budget on the
processor instead (``size.longest_edge``) so the image is resampled once, by
the image processor, rather than twice.
"""

import logging
import os
import random
from dataclasses import dataclass, field
from functools import partial

from PIL import Image
from torch.utils.data import Dataset

from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn

logger = logging.getLogger(__name__)

# Instruction shown to the model as the system turn. The rules describe what the
# reference transcriptions in this corpus actually do, so the prompt and the
# targets agree: the pages are plain text (Markdown headings appear in 0.2% of
# them and tables in 0.4%), figures are stood in for by an "image : N" line
# (37.0%), struck-out personal details become [REDACTED] (11.4%), and paragraphs
# are separated by blank lines (76.8%).
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert Arabic OCR system. You read a scanned page and transcribe "
    "its text exactly as it appears.\n"
    "Rules:\n"
    "- Transcribe verbatim. Preserve tashkeel, hamza forms, and punctuation as "
    "written, and do not normalize, correct, or modernize the spelling.\n"
    "- Keep the reading order, the line breaks, and the blank lines between "
    "paragraphs.\n"
    "- Write numerals exactly as printed, whether Arabic-Indic (٠-٩) or Latin "
    "(0-9). Never convert between the two.\n"
    "- Keep page numbers, headers, and footnote markers in place.\n"
    "- Where a figure, photo, or illustration appears, write 'image : N' on its "
    "own line, numbering figures from 1 down the page.\n"
    "- Where personal information has been struck out, write [REDACTED].\n"
    "- Output the transcription only. Do not translate, summarize, comment, or "
    "wrap the result in a code fence."
)

# Sampled per example so the model does not overfit to one phrasing of the task.
# Mixed English/Arabic because requests at inference time may come in either.
# None of these say "Markdown": the targets are plain text, and asking for
# Markdown would promise structure the references do not contain.
DEFAULT_PROMPTS = (
    "Transcribe all the text in this image.",
    "Extract the Arabic text from this page, keeping its original layout.",
    "Perform OCR on this page and return the text exactly as written.",
    "Read this scanned page and write out its full contents.",
    "What text appears in this image? Transcribe it in full.",
    "Transcribe this page, preserving its line and paragraph breaks.",
    "استخرج النص العربي الموجود في هذه الصورة.",
    "اقرأ هذه الصفحة واكتب محتواها كاملاً.",
    "انسخ نص هذه الصفحة كما هو مع الحفاظ على ترتيب الأسطر والفقرات.",
)


def arabic_ocr_collate_fn(examples, processor, **kwargs):
    """``default_collate_fn`` with the chat template's reasoning preamble disabled.

    Qwen3.5's chat template defaults to ``reasoning_effort='xhigh'`` and prepends
    "Reasoning effort is set to xhigh. Please think carefully..." to the system
    turn -- while the assistant turn it builds for a plain text target is
    ``<think>\\n\\n</think>`` (empty). Training on that pairing teaches the model
    to ignore an explicit instruction to reason, and spends ~50 tokens of every
    sample's context saying so. ``enable_thinking=False`` drops the preamble; the
    empty ``<think>`` marker stays either way, since the template always emits it
    for the final assistant turn.

    Implemented by binding the flag onto the processor's bound method for the
    duration of the call rather than by forking ``default_collate_fn``, which
    would duplicate ~80 lines of label-building and media handling that we want
    to keep tracking upstream.
    """
    original = processor.apply_chat_template
    processor.apply_chat_template = partial(original, enable_thinking=False)
    try:
        return default_collate_fn(examples, processor, **kwargs)
    finally:
        processor.apply_chat_template = original


def _collect_pairs(root: str) -> list[str]:
    """Return sorted ids that have both a ``.png`` and a non-empty ``.md``."""
    ids, missing_png, empty_md = [], 0, 0
    for entry in sorted(os.listdir(root)):
        if not entry.endswith(".md"):
            continue
        stem = entry[: -len(".md")]
        if not os.path.exists(os.path.join(root, stem + ".png")):
            missing_png += 1
            continue
        # An empty transcription yields an assistant turn with no tokens, which
        # would leave the whole sample masked to -100 and contribute no loss.
        with open(os.path.join(root, entry), encoding="utf-8") as fh:
            if not fh.read().strip():
                empty_md += 1
                continue
        ids.append(stem)

    if missing_png or empty_md:
        logger.warning(
            "Arabic OCR: skipped %d sample(s) with no matching .png and %d with an empty .md",
            missing_png,
            empty_md,
        )
    if not ids:
        raise ValueError(f"No usable <id>.png / <id>.md pairs found under {root}")
    return ids


def make_arabic_ocr_dataset(
    path_or_dataset: str,
    split: str = "train",
    val_fraction: float = 0.0,
    seed: int = 1234,
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
    prompts: tuple[str, ...] = DEFAULT_PROMPTS,
) -> "ArabicOcrDataset":
    """Build the Arabic OCR conversation dataset.

    Args:
        path_or_dataset: Directory holding the ``<id>.png`` / ``<id>.md`` pairs.
        split: ``"train"`` or ``"validation"``. With the default
            ``val_fraction=0.0`` the validation split is empty and ``"train"``
            covers every pair.
        val_fraction: Fraction of pairs held out for validation.
        seed: Seed for the held-out shuffle and for per-example prompt choice.
        system_prompt: System turn prepended to every conversation. ``None``
            omits the system turn entirely.
        prompts: Instructions to sample from for the user turn.

    Returns:
        An indexable dataset whose items are ``{"conversation": [...]}``.
    """
    return ArabicOcrDataset(
        root=path_or_dataset,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
        system_prompt=system_prompt,
        prompts=tuple(prompts),
    )


class ArabicOcrDataset(Dataset):
    """Paired page-image / Markdown-transcription dataset for OCR fine-tuning."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        val_fraction: float = 0.0,
        seed: int = 1234,
        system_prompt: str | None = DEFAULT_SYSTEM_PROMPT,
        prompts: tuple[str, ...] = DEFAULT_PROMPTS,
    ):
        if split not in ("train", "validation"):
            raise ValueError(f"split must be 'train' or 'validation', got {split!r}")
        if not 0.0 <= val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0.0, 1.0), got {val_fraction}")
        if not prompts:
            raise ValueError("prompts must not be empty")

        self.root = root
        self.system_prompt = system_prompt
        self.prompts = tuple(prompts)
        self.seed = seed

        ids = _collect_pairs(root)
        # Shuffle a copy under a fixed seed so the split is identical on every
        # rank and across restarts, without depending on filesystem ordering.
        shuffled = list(ids)
        random.Random(seed).shuffle(shuffled)
        n_val = int(len(shuffled) * val_fraction)
        self.ids = shuffled[n_val:] if split == "train" else shuffled[:n_val]

        logger.info(
            "Arabic OCR [%s]: %d/%d pairs from %s",
            split,
            len(self.ids),
            len(ids),
            root,
        )

    def __len__(self) -> int:
        return len(self.ids)

    def _prompt_for(self, idx: int) -> str:
        # Keyed on the seed and index so a given page always gets the same
        # instruction across epochs and restarts -- reproducible, and it keeps
        # the prompt stable when a run resumes mid-epoch.
        return self.prompts[random.Random((self.seed, idx).__hash__()).randrange(len(self.prompts))]

    def __getitem__(self, idx: int) -> dict:
        stem = self.ids[idx]
        with open(os.path.join(self.root, stem + ".md"), encoding="utf-8") as fh:
            answer = fh.read().strip()
        # Lazy: reads the PNG header only. The collate_fn converts to RGB and the
        # DataLoader worker does the pixel decode.
        image = Image.open(os.path.join(self.root, stem + ".png"))

        conversation = []
        if self.system_prompt:
            conversation.append(
                {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]}
            )
        conversation.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self._prompt_for(idx)},
                ],
            }
        )
        conversation.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
        return {"conversation": conversation}


@dataclass
class ArabicOcrDatasetConfig:
    """Construction-time configuration for the Arabic OCR dataset."""

    path_or_dataset: str = ""
    """Directory holding the <id>.png / <id>.md pairs."""
    split: str = "train"
    """Split to load: 'train' or 'validation'."""
    val_fraction: float = 0.0
    """Fraction of pairs held out for validation."""
    seed: int = 1234
    """Seed for the held-out shuffle and per-example prompt choice."""
    system_prompt: str | None = DEFAULT_SYSTEM_PROMPT
    """System turn prepended to every conversation; None omits it."""
    prompts: tuple[str, ...] = field(default_factory=lambda: DEFAULT_PROMPTS)
    """Instructions to sample from for the user turn."""

    def build(self) -> ArabicOcrDataset:
        """Build the Arabic OCR conversation dataset."""
        return make_arabic_ocr_dataset(
            path_or_dataset=self.path_or_dataset,
            split=self.split,
            val_fraction=self.val_fraction,
            seed=self.seed,
            system_prompt=self.system_prompt,
            prompts=self.prompts,
        )
