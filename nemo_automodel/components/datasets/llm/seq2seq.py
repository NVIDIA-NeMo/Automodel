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
"""Seq2seq (encoder-decoder) fine-tuning dataset for AutoModelForSeq2SeqLM.

Encoder-decoder models such as T5 and BART differ from decoder-only models in
how the training batch is laid out:

- The encoder reads ``input_ids`` (the source) with its own ``attention_mask``.
- ``labels`` are the target tokens, kept at full length and **not** shifted.
  The model's loss aligns ``logits[i]`` with ``labels[i]`` directly.
- ``decoder_input_ids`` is the right-shifted copy of ``labels`` (teacher
  forcing). HuggingFace builds this internally when ``labels`` is passed, but
  the training loop pops ``labels`` before calling the model, so we build it
  here and put it in the batch so it survives.

This is the opposite of the causal SFT path (see
``formatting_utils._package_tokenized_example``), which concatenates prompt and
answer into one stream and pre-shifts ``input_ids``/``labels`` by one position.

The produced per-sample dict reuses the ``___PAD_TOKEN_IDS___`` convention so
that ``utils.default_collater`` pads each field with the right value
(``labels`` -> -100, the rest -> the pad id / 0).
"""

import logging

from datasets import load_dataset

from nemo_automodel.components.datasets.lazy_mapped_dataset import LazyMappedDataset
from nemo_automodel.components.datasets.llm.formatting_utils import _add_pad_token

logger = logging.getLogger(__name__)


def _shift_right(token_ids, decoder_start_token_id):
    """Right-shift target tokens to form decoder inputs (teacher forcing).

    Mirrors HuggingFace ``shift_tokens_right`` / T5 ``_shift_right``: prepend
    ``decoder_start_token_id`` and drop the final token, so position ``i`` of the
    result is the input that should predict ``token_ids[i]``.

    Args:
        token_ids: The (unshifted) target token ids.
        decoder_start_token_id: The token the decoder starts from.

    Returns:
        A list of the same length as ``token_ids``.
    """
    return [decoder_start_token_id] + list(token_ids[:-1])


def _extract_target_text(value):
    """Pull a target string out of a dataset field.

    Supports plain strings and the SQuAD ``answers`` layout
    (``{"text": [...], "answer_start": [...]}``).
    """
    if isinstance(value, dict) and "text" in value:
        texts = value["text"]
        return texts[0].strip() if texts else ""
    return value.strip() if isinstance(value, str) else str(value)


def _format_seq2seq_example(
    example,
    tokenizer,
    source_template,
    source_key,
    target_key,
    decoder_start_token_id,
    pad_token_id,
    seq_length=None,
    truncation=True,
):
    """Turn one raw example into the seq2seq batch fields.

    Returns a dict with ``input_ids``, ``attention_mask``, ``labels`` (unshifted)
    and ``decoder_input_ids`` (right-shifted ``labels``), plus the
    ``___PAD_TOKEN_IDS___`` metadata used by ``default_collater``.
    """
    if source_template is not None:
        source = source_template.format(**example)
    else:
        source = example[source_key]
    source = source if isinstance(source, str) else str(source)
    target = _extract_target_text(example[target_key])

    tok_kwargs = {}
    if seq_length is not None:
        tok_kwargs["max_length"] = seq_length
        tok_kwargs["truncation"] = bool(truncation)

    enc = tokenizer(source, **tok_kwargs)
    # ``text_target`` tokenizes the target in target mode (decoder vocab / eos)
    # without applying any shift.
    dec = tokenizer(text_target=target, **tok_kwargs)

    labels = list(dec["input_ids"])
    decoder_input_ids = _shift_right(labels, decoder_start_token_id)

    return {
        "input_ids": list(enc["input_ids"]),
        "attention_mask": list(enc["attention_mask"]),
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
        "___PAD_TOKEN_IDS___": {
            "input_ids": pad_token_id,
            "attention_mask": 0,
            "labels": -100,
            "decoder_input_ids": pad_token_id,
        },
    }


def make_seq2seq_dataset(
    tokenizer,
    seq_length=None,
    limit_dataset_samples=None,
    split="train",
    dataset_name="rajpurkar/squad",
    source_template="question: {question}  context: {context}",
    source_key="question",
    target_key="answers",
    decoder_start_token_id=None,
    truncation=True,
):
    """Load and preprocess a dataset for encoder-decoder (seq2seq) fine-tuning.

    Each example is tokenized into an encoder source (``input_ids`` +
    ``attention_mask``) and a decoder target. The target becomes the unshifted
    ``labels``; ``decoder_input_ids`` is its right-shifted copy. ``default_collater``
    pads ``labels`` with -100 and the id fields with the pad id.

    Args:
        tokenizer: A HuggingFace tokenizer (injected by the recipe). Must support
            the ``text_target`` argument for target-side tokenization.
        seq_length (int, optional): If set, truncate source and target to this
            length.
        limit_dataset_samples (int, optional): If set, only load this many
            examples from the split.
        split (str): Dataset split to load (e.g. "train", "validation").
        dataset_name (str): HuggingFace dataset identifier. Defaults to SQuAD,
            framed as a question+context -> answer seq2seq task.
        source_template (str, optional): ``str.format`` template applied to each
            example to build the source text. If None, ``source_key`` is used
            verbatim.
        source_key (str): Field used as the source when ``source_template`` is
            None.
        target_key (str): Field holding the target. Supports plain strings and
            the SQuAD ``answers`` dict layout.
        decoder_start_token_id (int, optional): Token the decoder starts from.
            If None, defaults to the tokenizer's pad id (correct for T5/mT5).
            Models with a different convention (e.g. BART uses eos) should set
            this explicitly.
        truncation (bool): Whether to truncate to ``seq_length`` when it is set.

    Returns:
        A ``LazyMappedDataset`` yielding the per-sample seq2seq fields.
    """
    if limit_dataset_samples is not None:
        assert isinstance(limit_dataset_samples, int), "Expected limit_dataset_samples to be an int"
        if "[" not in split:
            split = f"{split}[:{limit_dataset_samples}]"
        else:
            logging.warning(f"Dataset split {split} already has a slice, skipping limit_dataset_samples")
    dataset = load_dataset(dataset_name, split=split)

    eos_token_id = getattr(tokenizer, "eos_token_id", 0)
    # NOTE: do not use ``_add_pad_token(...) or eos_token_id`` here. T5's pad id
    # is 0, which is falsy, so the ``or`` would wrongly fall back to eos. Only
    # fall back when there genuinely is no pad id.
    pad_token_id = _add_pad_token(tokenizer)
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if decoder_start_token_id is None:
        decoder_start_token_id = pad_token_id

    fmt_fn = lambda x: _format_seq2seq_example(  # noqa: E731
        x,
        tokenizer,
        source_template,
        source_key,
        target_key,
        decoder_start_token_id,
        pad_token_id,
        seq_length,
        truncation,
    )

    return LazyMappedDataset(dataset, fmt_fn)
