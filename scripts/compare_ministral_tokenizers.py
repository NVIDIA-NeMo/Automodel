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

"""Compare tokenization behavior across related Ministral and Nemotron repositories."""

import argparse
import logging

import transformers
from transformers import AutoTokenizer

DEFAULT_REPOSITORIES = (
    "mistralai/Ministral-3-3B-Instruct-2512",
    # "mistralai/Ministral-3-3B-Instruct-2512-BF16",
    "/lustre/fsw/portfolios/nemotron/projects/nemotron_retriever_dev/retriever_models_research/nemotron_4_embed_vl/pixtral_nemotron_3_embed_1B",
    "nvidia/Nemotron-3-Embed-1B-BF16",
)
DEFAULT_TEXTS = (
    "Hello, world!",
    "  Leading spaces, repeated   spaces, and a trailing space ",
    "First line\nSecond line\r\nThird line",
    "Unicode: café naïve — Ελληνικά — 中文 — العربية 😀",
    "def fibonacci(n: int) -> int:\n    return n if n < 2 else fibonacci(n - 1) + fibonacci(n - 2)",
    "Special-token text: <s> [INST] hello [/INST] </s> [IMG] [THINK]",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-a", default=DEFAULT_REPOSITORIES[0])
    parser.add_argument("--repo-b", default=DEFAULT_REPOSITORIES[1])
    parser.add_argument("--repo-c", default=DEFAULT_REPOSITORIES[2])
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="Text to compare. Repeat this option for multiple texts; defaults to a built-in multilingual set.",
    )
    parser.add_argument("--revision", default="main", help="Hugging Face revision to load from all repositories.")
    return parser.parse_args()


def first_difference(left: list[int], right: list[int]) -> int | None:
    """Return the index of the first differing token ID, or ``None`` when equal."""
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right)) if len(left) != len(right) else None


def main() -> int:
    """Load all tokenizers and compare observable tokenization behavior."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("transformers=%s", transformers.__version__)
    logger.info("Loading %s with fix_mistral_regex=True", args.repo_a)
    tokenizer_a = AutoTokenizer.from_pretrained(
        args.repo_a,
        revision=args.revision,
        trust_remote_code=False,
        # fix_mistral_regex=True,
    )
    logger.info("Loading %s with fix_mistral_regex=True", args.repo_b)
    tokenizer_b = AutoTokenizer.from_pretrained(
        args.repo_b,
        revision=args.revision,
        trust_remote_code=False,
        # fix_mistral_regex=True,
    )
    logger.info("Loading %s without fix_mistral_regex", args.repo_c)
    tokenizer_c = AutoTokenizer.from_pretrained(
        args.repo_c,
        revision=args.revision,
        trust_remote_code=False,
    )

    tokenizers = (("A", tokenizer_a), ("B", tokenizer_b), ("C", tokenizer_c))
    pairs = ((tokenizers[0], tokenizers[1]), (tokenizers[0], tokenizers[2]), (tokenizers[1], tokenizers[2]))
    for label, tokenizer in tokenizers:
        logger.info("%s class: %s.%s", label, type(tokenizer).__module__, type(tokenizer).__name__)
        logger.info(
            "%s special IDs: bos=%s eos=%s pad=%s unk=%s",
            label,
            tokenizer.bos_token_id,
            tokenizer.eos_token_id,
            tokenizer.pad_token_id,
            tokenizer.unk_token_id,
        )

    texts = args.texts or list(DEFAULT_TEXTS)
    mismatch_count = 0
    for add_special_tokens in (False, True):
        for (label_a, current_a), (label_b, current_b) in pairs:
            logger.info("\n=== %s vs %s; add_special_tokens=%s ===", label_a, label_b, add_special_tokens)
            for index, sample in enumerate(texts):
                ids_a = current_a.encode(sample, add_special_tokens=add_special_tokens)
                ids_b = current_b.encode(sample, add_special_tokens=add_special_tokens)
                difference = first_difference(ids_a, ids_b)
                if difference is None:
                    logger.info("[%d] MATCH (%d tokens): %r", index, len(ids_a), sample)
                    continue

                mismatch_count += 1
                logger.info("[%d] MISMATCH at token %d: %r", index, difference, sample)
                if ids_b == [current_b.bos_token_id, *ids_a]:
                    logger.info("    Difference is only the BOS token inserted by %s.", label_b)
                elif ids_a == [current_a.bos_token_id, *ids_b]:
                    logger.info("    Difference is only the BOS token inserted by %s.", label_a)
                logger.info("    %s ids: %s", label_a, ids_a)
                logger.info("    %s ids: %s", label_b, ids_b)
                logger.info("    %s tokens: %s", label_a, current_a.convert_ids_to_tokens(ids_a))
                logger.info("    %s tokens: %s", label_b, current_b.convert_ids_to_tokens(ids_b))

    logger.info("\n=== padded batch ===")
    for (label_a, current_a), (label_b, current_b) in pairs:
        batch_a = current_a(texts, add_special_tokens=True, padding=True)
        batch_b = current_b(texts, add_special_tokens=True, padding=True)
        for field in ("input_ids", "attention_mask"):
            equal = batch_a[field] == batch_b[field]
            mismatch_count += int(not equal)
            logger.info("%s vs %s %s: %s", label_a, label_b, field, "MATCH" if equal else "MISMATCH")

    logger.info("\nRESULT: %d mismatch(es)", mismatch_count)
    logger.info("Re-run with --text 'your exact production text' to test application-specific inputs.")
    return int(mismatch_count != 0)


if __name__ == "__main__":
    raise SystemExit(main())
