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

"""Functional test: HFDSparkTargetModel + DSparkTrainerModule against a tiny Qwen3.

Exercises the recipe's per-step path end to end -- capture target supervision via
the target wrapper, then train the draft via the trainer module -- on the actual
component classes (not hand-rolled capture). Requires a GPU (FlexAttention).
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, Qwen3Config

from nemo_automodel.components.datasets.llm.dspark_cache import (
    build_cached_dspark_dataloader,
    compute_batch_cache,
    read_target_weight_modules,
    write_manifest,
    write_shard,
    write_target_weights,
)
from nemo_automodel.components.speculative.dspark import Qwen3DSparkModel, build_draft_config
from nemo_automodel.components.speculative.dspark.core import DSparkStepMetrics, DSparkTrainerModule
from nemo_automodel.components.speculative.dspark.registry import build_target_layer_ids
from nemo_automodel.components.speculative.dspark.target import HFDSparkTargetModel
from nemo_automodel.recipes.llm.train_dspark import TrainDSparkRecipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention backward requires a GPU")

VOCAB = 1024
HIDDEN = 128


class _Args(dict):
    def __getattr__(self, key):
        return self[key]


def test_trainer_module_with_target_wrapper_trains():
    device, dtype = "cuda", torch.float32
    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    target_layer_ids = build_target_layer_ids(target_config.num_hidden_layers, 3)

    target = AutoModelForCausalLM.from_config(target_config).to(device=device, dtype=dtype).eval()
    target.requires_grad_(False)
    target_wrapper = HFDSparkTargetModel(target, target_layer_ids=target_layer_ids)

    margs = _Args(
        num_draft_layers=2,
        target_layer_ids=target_layer_ids,
        block_size=4,
        mask_token_id=7,
        num_anchors=16,
        markov_rank=32,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    draft = Qwen3DSparkModel(build_draft_config(target_config, margs)).to(device=device, dtype=dtype)
    draft.initialize_embeddings_and_head(
        embed_tokens=target_wrapper.get_input_embeddings(),
        lm_head=target_wrapper.get_output_embeddings(),
        freeze=True,
    )
    trainer = DSparkTrainerModule(
        draft, loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0
    ).to(device)

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (1, 32), device=device)
    attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)
    loss_mask = torch.ones(1, 32, dtype=torch.uint8, device=device)

    batch = target_wrapper.generate_batch(input_ids, attention_mask, loss_mask)
    assert batch.target_hidden_states.shape == (1, 32, len(target_layer_ids) * HIDDEN)
    assert batch.target_last_hidden_states.shape == (1, 32, HIDDEN)

    optim = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=5e-3)
    trainer.train()
    losses = []
    for _ in range(20):
        optim.zero_grad()
        torch.manual_seed(123)
        out = trainer(
            input_ids=batch.input_ids,
            target_hidden_states=batch.target_hidden_states,
            loss_mask=batch.loss_mask,
            target_last_hidden_states=batch.target_last_hidden_states,
        )
        assert isinstance(out, DSparkStepMetrics)
        out.loss.backward()
        optim.step()
        losses.append(out.loss.item())

    assert losses[-1] < losses[0], f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_recipe_loop_flushes_partial_accumulation_window():
    """Exercise the live-target recipe loop and its trailing optimizer update."""
    device, dtype = torch.device("cuda"), torch.float32
    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    target_layer_ids = build_target_layer_ids(target_config.num_hidden_layers, 3)
    target = AutoModelForCausalLM.from_config(target_config).to(device=device, dtype=dtype).eval()
    target.requires_grad_(False)
    target_wrapper = HFDSparkTargetModel(target, target_layer_ids=target_layer_ids)
    model_args = _Args(
        num_draft_layers=2,
        target_layer_ids=target_layer_ids,
        block_size=4,
        mask_token_id=7,
        num_anchors=16,
        markov_rank=32,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    draft = Qwen3DSparkModel(build_draft_config(target_config, model_args)).to(device=device, dtype=dtype)
    draft.initialize_embeddings_and_head(
        embed_tokens=target_wrapper.get_input_embeddings(),
        lm_head=target_wrapper.get_output_embeddings(),
        freeze=True,
    )
    trainer = DSparkTrainerModule(
        draft,
        loss_decay_gamma=4.0,
        ce_loss_alpha=0.1,
        l1_loss_alpha=0.9,
        confidence_head_alpha=1.0,
    ).to(device)
    optimizer = torch.optim.AdamW([parameter for parameter in trainer.parameters() if parameter.requires_grad], lr=1e-3)

    torch.manual_seed(5)
    batches = []
    for _ in range(3):
        input_ids = torch.randint(0, VOCAB, (1, 16))
        batches.append(
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "loss_mask": torch.ones_like(input_ids, dtype=torch.uint8),
            }
        )

    recipe = TrainDSparkRecipe.__new__(TrainDSparkRecipe)
    recipe.trainer_module = trainer
    recipe.target_wrapper = target_wrapper
    recipe.train_dataloader = batches
    recipe.val_dataloader = None
    recipe.optimizer = optimizer
    recipe.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    recipe.runtime = SimpleNamespace(global_step=0)
    recipe.num_epochs = 1
    recipe._resume_epoch = 0
    recipe.total_optim_steps = 2
    recipe.grad_accumulation_steps = 2
    recipe.defer_fsdp_grad_sync = True
    recipe.max_grad_norm = 1.0
    recipe.block_size = 4
    recipe.device = device
    recipe.dist_env = SimpleNamespace(is_main=False)
    recipe.log_every_steps = 100
    recipe.metric_logger = None
    recipe.wandb_run = None
    recipe.save_checkpoint_every_epoch = False
    recipe.ckpt_every_steps = None
    recipe._make_progress_bar = lambda **kwargs: None
    recipe._maybe_precompute_fp8_scales = lambda: None
    recipe._maybe_save_step_checkpoint = lambda epoch: False
    recipe._maybe_save_final_checkpoint = lambda completed_epochs: False
    recipe._finalize_and_close_checkpointer = lambda: None

    before = next(parameter for parameter in trainer.parameters() if parameter.requires_grad).detach().clone()
    recipe.run_train_validation_loop()

    assert recipe.runtime.global_step == 2
    assert not torch.equal(next(parameter for parameter in trainer.parameters() if parameter.requires_grad), before)


def test_trainer_module_trains_from_offline_cache(tmp_path):
    device, dtype = "cuda", torch.float32
    target_config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=2 * HIDDEN,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    target_layer_ids = build_target_layer_ids(target_config.num_hidden_layers, 3)

    target = AutoModelForCausalLM.from_config(target_config).to(device=device, dtype=dtype).eval()
    target.requires_grad_(False)
    target_wrapper = HFDSparkTargetModel(target, target_layer_ids=target_layer_ids)

    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (1, 32), device=device)
    attention_mask = torch.ones(1, 32, dtype=torch.long, device=device)
    loss_mask = torch.ones(1, 32, dtype=torch.uint8, device=device)
    with torch.no_grad():
        target_batch = target_wrapper.generate_batch(input_ids, attention_mask, loss_mask)

    cache_dir = str(tmp_path / "dspark-cache")
    write_target_weights(
        cache_dir,
        target_wrapper.get_input_embeddings(),
        target_wrapper.get_output_embeddings(),
        dtype=torch.float32,
    )
    write_shard(cache_dir, 0, compute_batch_cache(target_batch, cache_dtype=torch.float32))
    write_manifest(
        cache_dir,
        {
            "target_model": "tiny-qwen3",
            "target_model_type": "qwen3",
            "target_vocab_size": VOCAB,
            "hidden_size": HIDDEN,
            "num_hidden_layers": target_config.num_hidden_layers,
            "seq_length": 32,
            "dtype": "fp32",
            "num_samples": 1,
            "shard_size": 1,
            "target_hidden_dim": HIDDEN * len(target_layer_ids),
            "target_last_hidden_dim": HIDDEN,
            "target_layer_ids": list(target_layer_ids),
        },
    )

    margs = _Args(
        num_draft_layers=2,
        target_layer_ids=target_layer_ids,
        block_size=4,
        mask_token_id=7,
        num_anchors=16,
        markov_rank=32,
        markov_head_type="vanilla",
        confidence_head_alpha=1.0,
        confidence_head_with_markov=True,
    )
    draft = Qwen3DSparkModel(build_draft_config(target_config, margs)).to(device=device, dtype=dtype)
    embed_src, head_src = read_target_weight_modules(cache_dir)
    draft.initialize_embeddings_and_head(embed_tokens=embed_src, lm_head=head_src, freeze=True)
    trainer = DSparkTrainerModule(
        draft, loss_decay_gamma=4.0, ce_loss_alpha=0.1, l1_loss_alpha=0.9, confidence_head_alpha=1.0
    ).to(device)

    (batch,) = list(build_cached_dspark_dataloader(cache_dir=cache_dir, batch_size=1, shuffle=False))
    batch = {k: v.to(device) for k, v in batch.items()}
    optim = torch.optim.AdamW([p for p in trainer.parameters() if p.requires_grad], lr=5e-3)
    losses = []
    for _ in range(20):
        optim.zero_grad()
        torch.manual_seed(123)
        out = trainer(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        out.loss.backward()
        optim.step()
        losses.append(out.loss.item())

    assert losses[-1] < losses[0], f"cached loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
