#!/usr/bin/env python3
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

"""Compare baseline vs fine-tuned cross-encoder models on retrieval quality.

Evaluates two checks:
  1. Fine-tuning improves positive-pair scores: mean(ft_pos - base_pos) > 0.
  2. The fine-tuned model ranks the positive document first at least 50% of the time.
"""

import argparse
from pathlib import Path

import numpy as np
import torch

from nemo_automodel._transformers.auto_model import NeMoAutoModelCrossEncoder
from nemo_automodel.components.datasets.llm import retrieval_dataset_inline as rdi
from nemo_automodel.components.distributed.init_utils import initialize_distributed

PROMPT_TEMPLATE = "question:{query} \n \n passage:{passage}"


@torch.no_grad()
def _score_pairs(model, tokenizer, pairs: list[tuple[str, str]], device: torch.device, max_length: int = 512):
    """Score (query, passage) pairs through the cross-encoder. Returns a list of float scores."""
    texts = [PROMPT_TEMPLATE.format(query=q, passage=p) for q, p in pairs]
    # Tokenize without padding first (NeMoAutoTokenizer BOS/EOS insertion needs list input),
    # then pad and convert to tensors in a separate step.
    encodings = tokenizer(texts, max_length=max_length, padding=False, truncation=True)
    tok_features = [{k: encodings[k][i] for k in encodings} for i in range(len(texts))]
    batch_dict = tokenizer.pad(tok_features, padding=True, return_tensors="pt")
    batch_dict = {k: v.to(device) for k, v in batch_dict.items()}
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**batch_dict, return_dict=True)
    return outputs.logits.view(-1).float().cpu().tolist()


def _evaluate_model(model, tokenizer, ds, device, max_length, max_samples):
    """Run cross-encoder evaluation. Returns (pos_scores, pos_ranked_first) as numpy arrays."""
    n = min(len(ds), max_samples)
    pos_scores = []
    pos_ranked_first = []

    for i in range(n):
        ex = ds[i]
        query = ex["question"]
        pairs = [(query, d) for d in ex["doc_text"]]

        scores = _score_pairs(model, tokenizer, pairs, device, max_length=max_length)

        # Index 0 is the positive document.
        pos_scores.append(scores[0])
        pos_ranked_first.append(scores[0] >= max(scores))

    return np.array(pos_scores, dtype=np.float64), np.array(pos_ranked_first, dtype=bool)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline vs fine-tuned cross-encoder models (relevance scoring)."
    )
    parser.add_argument("base_model_path", type=str, help="Path to base/pretrained HF model")
    parser.add_argument("checkpoint_root", type=str, help="Path to fine-tuned checkpoint root directory")
    parser.add_argument("dataset_jsonl", type=str, help="Path to evaluation JSONL dataset")
    parser.add_argument("trust_remote_code", type=bool, help="Trust remote code")
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Max token length for (query, passage) inputs (default: 512)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Cap on number of evaluation samples (default: use all data)",
    )
    return parser.parse_args()


def _load_cross_encoder(path, device, dtype=torch.bfloat16):
    """Load a cross-encoder model onto device in eval mode."""
    return (
        NeMoAutoModelCrossEncoder.from_pretrained(
            str(path),
            pooling="avg",
            num_labels=1,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )


def main() -> int:
    args = _parse_args()

    ckpt_root = Path(args.checkpoint_root)
    max_length = args.max_length

    dist = initialize_distributed(backend="nccl", timeout_minutes=5)
    device = dist.device if dist.device is not None else torch.device("cpu")

    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

    tokenizer = NeMoAutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=args.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    # Use bi_encoder model_type to preserve per-query pos/neg grouping.
    ds = rdi.make_retrieval_dataset(
        data_dir_list=str(args.dataset_jsonl),
        model_type="bi_encoder",
        data_type="eval",
        eval_negative_size=1,
        do_shuffle=False,
        max_train_samples=args.max_samples,
    )

    max_samples = args.max_samples if args.max_samples is not None else len(ds)
    print(f"Config: max_length={max_length}, max_samples={max_samples}")

    # Baseline model
    print(f"Loading baseline model from {args.base_model_path}")
    base_model = _load_cross_encoder(args.base_model_path, device)
    base_pos_scores, base_pos_ranked_first = _evaluate_model(base_model, tokenizer, ds, device, max_length, max_samples)
    del base_model
    torch.cuda.empty_cache()

    # Fine-tuned model
    consolidated_dir = ckpt_root / "model" / "consolidated"
    if not consolidated_dir.exists():
        raise FileNotFoundError(f"Expected consolidated checkpoint at {consolidated_dir}")

    print(f"Loading fine-tuned model from {consolidated_dir}")
    ft_model = _load_cross_encoder(consolidated_dir, device)
    ft_pos_scores, ft_pos_ranked_first = _evaluate_model(ft_model, tokenizer, ds, device, max_length, max_samples)
    del ft_model
    torch.cuda.empty_cache()

    # Diagnostics
    mean_improvement = float(np.mean(ft_pos_scores - base_pos_scores))
    ft_ranking_accuracy = float(np.mean(ft_pos_ranked_first))

    print(f"Baseline  mean(pos_score):   {float(np.mean(base_pos_scores)):.6f}")
    print(f"Finetuned mean(pos_score):   {float(np.mean(ft_pos_scores)):.6f}")
    print(f"Mean pos-score improvement:  {mean_improvement:.6f}")
    print(f"Baseline  ranking accuracy:  {float(np.mean(base_pos_ranked_first)):.4f}")
    print(f"Finetuned ranking accuracy:  {ft_ranking_accuracy:.4f}")

    # Check 1: fine-tuning improves positive-pair scores
    assert mean_improvement > 0, (
        f"Fine-tuned model did not improve positive-pair scores: mean improvement = {mean_improvement:.6f}"
    )
    print("Check 1 (mean pos-score improvement > 0): PASS")

    # Check 2: positive document ranked first at least 75% of the time
    assert ft_ranking_accuracy >= 0.75, (
        f"Fine-tuned model does not rank positive document first often enough: "
        f"accuracy = {ft_ranking_accuracy:.4f} < 0.75"
    )
    print("Check 2 (finetuned pos ranked first >= 0.75): PASS")

    print("All checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
