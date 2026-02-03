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

import glob
import os
import sys
from pathlib import Path

import numpy as np
import torch

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.datasets.llm import retrieval_dataset_inline as rdi
from nemo_automodel.components.datasets.llm import RetrievalBiencoderCollator
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.models.biencoder import NeMoAutoModelBiencoder


def _resolve_latest_checkpoint_dir(ckpt_root: Path) -> Path:
    latest_link = ckpt_root / "LATEST"
    latest_txt = ckpt_root / "LATEST.txt"

    if latest_link.exists():
        # Symlink case: resolve() follows symlinks.
        return latest_link.resolve()

    if latest_txt.exists():
        rel = latest_txt.read_text().strip()
        if not rel:
            raise ValueError(f"Empty {latest_txt}")
        return (ckpt_root / rel).resolve()

    # Fallback: pick the lexicographically last epoch_*_step_* directory.
    matches = sorted(ckpt_root.glob("epoch_*_step_*"))
    if not matches:
        raise FileNotFoundError(f"No epoch_*_step_* checkpoints found under {ckpt_root}")
    return matches[-1].resolve()


def _iter_batches(ds, batch_size: int, max_samples: int):
    n = min(len(ds), max_samples)
    batch = []
    for i in range(n):
        batch.append(ds[i])
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _compute_pos_neg_diffs(
    *,
    model,
    collator: RetrievalBiencoderCollator,
    ds,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    use_bf16_autocast: bool,
) -> np.ndarray:
    model.eval()
    diffs: list[np.ndarray] = []

    autocast_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16) if (use_bf16_autocast and device.type == "cuda") else None
    )

    with torch.no_grad():
        for batch_examples in _iter_batches(ds, batch_size=batch_size, max_samples=max_samples):
            batch = collator(batch_examples)
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}

            q_prefix, d_prefix = "q_", "d_"
            query = {k[len(q_prefix) :]: v for k, v in batch.items() if k.startswith(q_prefix)}
            passage = {k[len(d_prefix) :]: v for k, v in batch.items() if k.startswith(d_prefix)}

            if autocast_ctx is None:
                out = model(query=query, passage=passage)
            else:
                with autocast_ctx:
                    out = model(query=query, passage=passage)

            scores = out.scores  # [batch, n_passages]
            if scores is None or scores.shape[-1] < 2:
                raise RuntimeError(f"Unexpected scores shape: {None if scores is None else tuple(scores.shape)}")

            diff = (scores[:, 0] - scores[:, 1]).float().detach().cpu().numpy()
            diffs.append(diff)

    out = np.concatenate(diffs, axis=0) if diffs else np.array([], dtype=np.float32)
    if out.size == 0:
        raise RuntimeError("No diffs computed (empty dataset?)")
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite diffs found")
    return out


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: compare_biencoder_models.py <base_model_path> <checkpoint_root> [dataset_jsonl]", file=sys.stderr)
        return 2

    base_model_path = sys.argv[1]
    ckpt_root = Path(sys.argv[2])
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else os.environ.get("TEST_DATASET", "/app/data/cust-1234/training.jsonl")

    max_samples = int(os.environ.get("MAX_COMPARE_SAMPLES", "32"))
    batch_size = int(os.environ.get("COMPARE_BATCH_SIZE", "8"))

    # Initialize torch.distributed (DCP load requires a process group even in single-process mode).
    dist = initialize_distributed(backend="nccl", timeout_minutes=5)
    device = dist.device if dist.device is not None else torch.device("cpu")

    ckpt_dir = _resolve_latest_checkpoint_dir(ckpt_root)
    ft_model_dir = ckpt_dir / "model"
    if not ft_model_dir.exists():
        raise FileNotFoundError(f"Expected finetuned model dir at {ft_model_dir}")

    # Find any saved safetensors as a quick integrity check.
    st = glob.glob(str(ft_model_dir / "**" / "*.safetensors"), recursive=True)
    if not st:
        raise FileNotFoundError(f"No .safetensors found under {ft_model_dir}")

    # Build tokenizer/collator/dataset.
    from nemo_automodel._transformers.auto_tokenizer import NeMoAutoTokenizer

    tokenizer = NeMoAutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

    ds = rdi.make_retrieval_dataset(
        data_dir_list=str(dataset_path),
        data_type="train",
        train_n_passages=2,
        do_shuffle=False,
        max_train_samples=max_samples,
    )

    collator = RetrievalBiencoderCollator(
        tokenizer=tokenizer,
        q_max_len=64,
        p_max_len=64,
        query_prefix="",
        passage_prefix="",
        padding="longest",
        pad_to_multiple_of=8,
    )

    # Build a single model instance, compute baseline diffs, then load finetuned weights and recompute.
    model = NeMoAutoModelBiencoder.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        share_encoder=True,
        add_linear_pooler=False,
        out_dimension=768,
        do_gradient_checkpointing=False,
        train_n_passages=2,
        eval_negative_size=1,
        pooling="avg",
        l2_normalize=True,
        t=0.02,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        torch_dtype=torch.bfloat16,
    ).to(device)

    base_diffs = _compute_pos_neg_diffs(
        model=model,
        collator=collator,
        ds=ds,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        use_bf16_autocast=True,
    )

    # Load finetuned weights into the same model.
    ckpt_cfg = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=str(ckpt_root),
        model_save_format="safetensors",
        model_cache_dir="/tmp",
        model_repo_id="__local__",
        save_consolidated=False,
        is_peft=False,
    )
    checkpointer = Checkpointer(config=ckpt_cfg, dp_rank=0, tp_rank=0, pp_rank=0, moe_mesh=None)
    checkpointer.load_model(model, model_path=str(ft_model_dir))
    checkpointer.close()

    ft_diffs = _compute_pos_neg_diffs(
        model=model,
        collator=collator,
        ds=ds,
        device=device,
        batch_size=batch_size,
        max_samples=max_samples,
        use_bf16_autocast=True,
    )

    # Compare baseline vs finetuned diffs (pos - neg); finetuned should not be degraded.
    import scipy.stats

    t_stat, p_value = scipy.stats.ttest_rel(base_diffs, ft_diffs)
    if not np.isfinite(p_value):
        # Can happen with degenerate inputs (e.g., identical diffs).
        p_value = 1.0
    delta = ft_diffs - base_diffs
    denom = float(np.std(delta, ddof=1))
    cohen_d = float(np.mean(delta) / denom) if denom > 0 else 0.0

    print(f"Baseline mean(diff): {base_diffs.mean():.6f}")
    print(f"Fine-tuned mean(diff): {ft_diffs.mean():.6f}")
    print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.6f}, CohenD={cohen_d:.4f}")

    model_not_degraded = (p_value > 0.05) or (p_value < 0.05 and cohen_d > 0)
    assert model_not_degraded, "Fine-tuned model appears degraded vs baseline"
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

