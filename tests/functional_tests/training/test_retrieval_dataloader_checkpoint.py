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

"""Verify that retrieval dataloaders (bi-encoder & cross-encoder) correctly
save and restore iteration state across checkpoint boundaries.

The test creates a small synthetic retrieval dataset, builds a
StatefulDataLoader via the retrieval recipe's ``build_dataloader``, advances
it a few batches, checkpoints the state, then creates a fresh dataloader,
loads the state, and asserts that the next batch matches the expected one.

Launch (single-GPU is sufficient):
    torchrun --nproc-per-node=1 -m pytest tests/functional_tests/training/test_retrieval_dataloader_checkpoint.py -vs
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.retrieval.train_bi_encoder import build_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _init_dist():
    """Ensure torch.distributed is initialized (torchrun sets env vars)."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")


def _write_test_data(path: Path, n_samples: int = 20) -> None:
    """Write a minimal JSONL file that ``make_retrieval_dataset`` can load."""
    with open(path, "w") as f:
        for i in range(n_samples):
            entry = {
                "query": f"What is question number {i}?",
                "pos_doc": f"This is the positive passage for question {i}.",
                "neg_doc": [
                    f"Negative passage {j} for question {i}." for j in range(4)
                ],
            }
            f.write(json.dumps(entry) + "\n")


def _make_local_tokenizer(tmp_dir: str):
    """Create a tiny GPT-2 tokenizer saved locally so we don't need network access."""
    from transformers import AutoTokenizer

    # Save to a local dir so subsequent from_pretrained is purely local.
    tok_dir = os.path.join(tmp_dir, "tokenizer")
    os.makedirs(tok_dir, exist_ok=True)

    # Build from vocab — GPT2Tokenizer ships with the library, so from_pretrained
    # will work even offline if files are cached. As a fallback, construct manually.
    try:
        tok = AutoTokenizer.from_pretrained("gpt2")
    except Exception:
        # If gpt2 is truly unreachable, create the smallest BPE tokenizer possible.
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        base = Tokenizer(models.BPE())
        base.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Train on a tiny corpus to get a working vocab
        from tokenizers.trainers import BpeTrainer

        trainer = BpeTrainer(vocab_size=256, special_tokens=["<pad>", "<eos>"])
        base.train_from_iterator(
            [f"This is sentence number {i} for tokenizer training." for i in range(50)],
            trainer=trainer,
        )
        tok = PreTrainedTokenizerFast(tokenizer_object=base)
        tok.add_special_tokens({"pad_token": "<pad>", "eos_token": "<eos>"})

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.save_pretrained(tok_dir)
    return AutoTokenizer.from_pretrained(tok_dir)


def _make_checkpointer(checkpoint_dir: str, dp_rank: int = 0) -> Checkpointer:
    config = CheckpointingConfig(
        enabled=True,
        checkpoint_dir=checkpoint_dir,
        model_save_format="safetensors",
        model_cache_dir="",
        model_repo_id="",
        save_consolidated=False,
        is_peft=False,
        model_state_dict_keys=[],
    )
    return Checkpointer(config=config, dp_rank=dp_rank, tp_rank=0, pp_rank=0)


def _build_retrieval_dataloader(data_file: str, model_type: str, tokenizer, batch_size: int = 2, seed: int = 42):
    """Build a retrieval StatefulDataLoader from a ConfigNode, mirroring the YAML config structure."""
    if model_type == "bi_encoder":
        collate_target = "nemo_automodel.components.datasets.llm.BiEncoderCollator"
        collate_kwargs = {
            "_target_": collate_target,
            "q_max_len": 64,
            "p_max_len": 64,
            "query_prefix": "query:",
            "passage_prefix": "passage:",
            "pad_to_multiple_of": 8,
        }
    else:
        collate_target = "nemo_automodel.components.datasets.llm.CrossEncoderCollator"
        collate_kwargs = {
            "_target_": collate_target,
            "rerank_max_length": 64,
            "prompt_template": "question:{query} \n \n passage:{passage}",
            "pad_to_multiple_of": 8,
        }

    cfg_dl = ConfigNode(
        {
            "_target_": "torchdata.stateful_dataloader.StatefulDataLoader",
            "dataset": {
                "_target_": "nemo_automodel.components.datasets.llm.retrieval_dataset_inline.make_retrieval_dataset",
                "model_type": model_type,
                "data_dir_list": [data_file],
                "data_type": "train",
                "n_passages": 5,
                "seed": seed,
                "do_shuffle": True,
            },
            "collate_fn": collate_kwargs,
            "shuffle": True,
            "num_workers": 0,
        }
    )

    return build_dataloader(
        cfg_dl,
        tokenizer=tokenizer,
        seed=seed,
        batch_size=batch_size,
        dp_rank=0,
        dp_world_size=1,
    )


def _tensors_equal(batch_a: dict, batch_b: dict) -> None:
    """Assert all tensor values in two batch dicts are identical."""
    assert set(batch_a.keys()) == set(batch_b.keys()), (
        f"Batch keys differ: {set(batch_a.keys())} vs {set(batch_b.keys())}"
    )
    for k in batch_a:
        if isinstance(batch_a[k], torch.Tensor):
            assert torch.all(batch_a[k] == batch_b[k]), (
                f"Tensor mismatch on key '{k}'"
            )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

SAVE_AFTER_BATCH = 2  # save state after consuming this many batches


def _run_checkpoint_test(model_type: str):
    """Core test logic shared by bi-encoder and cross-encoder variants."""
    _init_dist()

    tmp_dir = tempfile.mkdtemp(prefix=f"retrieval_dl_ckpt_{model_type}_")
    data_file = os.path.join(tmp_dir, "train.jsonl")
    ckpt_dir = os.path.join(tmp_dir, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    try:
        _write_test_data(Path(data_file), n_samples=20)

        # Build a minimal tokenizer on disk to avoid network downloads.
        # The retrieval collators only need .encode()/.pad() from a HF tokenizer.
        tokenizer = _make_local_tokenizer(tmp_dir)

        checkpointer = _make_checkpointer(ckpt_dir)

        # --- Phase 1: advance dataloader, save state, record expected batch ---
        dl = _build_retrieval_dataloader(data_file, model_type, tokenizer)
        expected_batch = None
        for i, batch in enumerate(dl):
            if i == SAVE_AFTER_BATCH:
                checkpointer.save_on_dp_ranks(dl, "dataloader", ckpt_dir)
            elif i == SAVE_AFTER_BATCH + 1:
                expected_batch = batch
                break

        assert expected_batch is not None, "Dataset too small to reach expected batch"

        # Verify checkpoint files exist
        ckpt_file = Path(ckpt_dir) / "dataloader" / "dataloader_dp_rank_0.pt"
        assert ckpt_file.exists(), f"Expected checkpoint file {ckpt_file}"
        assert ckpt_file.stat().st_size > 0, "Checkpoint file is empty"

        del dl

        # --- Phase 2: fresh dataloader without loading state ---
        dl_fresh = _build_retrieval_dataloader(data_file, model_type, tokenizer)
        _initial_batch = next(iter(dl_fresh))

        # The fresh dataloader should NOT start where we left off (it restarts)
        # (This is a sanity check — not strictly guaranteed if dataset is tiny and
        # the shuffle seed happens to align, so we just log rather than assert.)

        # --- Phase 3: fresh dataloader WITH loaded state ---
        dl_restored = _build_retrieval_dataloader(data_file, model_type, tokenizer)
        checkpointer.load_on_dp_ranks(dl_restored, "dataloader", ckpt_dir)

        restored_batch = next(iter(dl_restored))
        _tensors_equal(restored_batch, expected_batch)

        del dl_fresh, dl_restored

    finally:
        dist.barrier()
        if dist.get_rank() == 0:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        dist.barrier()


def test_bi_encoder_dataloader_checkpoint():
    """Bi-encoder dataloader saves and restores iteration state correctly."""
    _run_checkpoint_test("bi_encoder")


def test_cross_encoder_dataloader_checkpoint():
    """Cross-encoder dataloader saves and restores iteration state correctly."""
    _run_checkpoint_test("cross_encoder")
