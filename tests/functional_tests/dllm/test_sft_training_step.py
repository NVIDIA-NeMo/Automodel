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

"""Real-CUDA functional coverage for an MDLM supervised-finetuning step."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from nemo_automodel.recipes.dllm.strategy import get_dllm_strategy
from nemo_automodel.recipes.dllm.train_ft import DiffusionLMSFTRecipe

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


class _TinyDenoiser(nn.Module):
    """Small bidirectional token denoiser used to exercise the production recipe."""

    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return per-token logits for token IDs shaped ``[batch, sequence]``."""
        return SimpleNamespace(logits=self.projection(self.embedding(input_ids)))


def test_mdlm_sft_step_corrupts_tokens_and_updates_model() -> None:
    """Run corruption, denoising loss, backward, clipping, and AdamW on CUDA."""
    torch.manual_seed(7)
    device = torch.device("cuda")
    model = _TinyDenoiser(vocab_size=32, hidden_size=16).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    strategy = get_dllm_strategy("mdlm")
    recipe = DiffusionLMSFTRecipe.__new__(DiffusionLMSFTRecipe)
    recipe.cfg = {}
    recipe.dllm_mode = "mdlm"
    recipe.dllm_strategy = strategy
    recipe.dllm_loss_fn = strategy.create_loss_fn({})
    recipe.dllm_eps = 1.0
    recipe.dllm_block_size = None
    recipe.dllm_half_life_ratio = None
    recipe.mask_token_id = 31
    recipe._self_cond_base_seed = 123
    recipe._dllm_loss_buffer = []
    recipe._dflash_correct_per_pos_buffer = []
    recipe._dflash_count_per_pos_buffer = []
    recipe.model_parts = [model]
    recipe.optimizer = [optimizer]
    recipe.lr_scheduler = None
    recipe.checkpointer = SimpleNamespace(maybe_wait_for_staging=lambda: None)
    recipe.dist_env = SimpleNamespace(device=device, world_size=1)
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True, autocast_dtype=None)
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.pp_enabled = False
    recipe.te_fp8 = None
    recipe.step_scheduler = SimpleNamespace(step=0, epoch=0, grad_acc_steps=1)
    recipe.timestamp = time.perf_counter()
    recipe.mfu_calculator = None
    recipe._get_dp_group_size = lambda include_cp=False: 1
    recipe._get_cp_group_size = lambda: 1
    recipe._dp_allreduce = lambda value, include_cp=False: value

    clean_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11, 12, 13, 14, 15, 16]],
        dtype=torch.long,
    )
    batches = [
        {"input_ids": clean_ids, "attention_mask": torch.ones_like(clean_ids), "loss_mask": torch.ones_like(clean_ids)}
    ]
    before = model.projection.weight.detach().clone()

    metrics = recipe._run_train_optim_step(batches, max_grad_norm=1.0)

    assert metrics.metrics["supervised_tokens"] == clean_ids.numel()
    assert metrics.metrics["tokens_per_step"] == clean_ids.numel()
    assert metrics.metrics["loss"] > 0
    assert torch.isfinite(torch.tensor(metrics.metrics["dllm_loss"]))
    assert torch.isfinite(torch.as_tensor(metrics.metrics["grad_norm"]))
    assert not torch.equal(model.projection.weight, before)
    assert torch.all(batches[0]["_noisy_input_ids"] == recipe.mask_token_id)
    assert torch.equal(batches[0]["input_ids"], clean_ids)
