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

"""CPU smoke test for the Uno dLLM training path.

The full ``DiffusionLMSFTRecipe`` loop is CUDA-only, so this drives the same per-step sequence the recipe
runs — ``UnoStrategy.pre_step`` (uniform noise, curriculum block size) then ``forward_backward`` (token-gated
LoRA over the ``[x_t | x_0]`` forward, total-variation loss, backward) — against a hermetically constructed
tiny Qwen3 with Automodel LoRA. The GPU counterpart is ``tests/functional_tests/dllm/L2_DLLM_Uno_Smoke.sh``.
"""

import types

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.recipes.dllm.strategy import get_dllm_strategy

VOCAB = 128
SEQ_LEN = 24
BATCH = 2
DLLM_CFG = {
    "mode": "uno",
    "mask_token_id": VOCAB - 1,
    "block_curriculum": {
        "tokens_per_step": BATCH * SEQ_LEN,
        "stages": [
            {"block_size": 2, "tokens": 3 * BATCH * SEQ_LEN},
            {"block_size": 4, "tokens": 3 * BATCH * SEQ_LEN},
        ],
    },
}


def _tiny_lora_model() -> Qwen3ForCausalLM:
    """Build a 2-layer randomly-initialised Qwen3 with LoRA on every projection, no network access."""
    config = Qwen3Config(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=2 * SEQ_LEN,
        attn_implementation="sdpa",
    )
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(config)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*_proj"], dim=4, alpha=64, use_triton=False))
    return model


def _recipe(model, loss_fn, step: int, seed: int):
    """A stand-in for ``DiffusionLMSFTRecipe`` exposing what ``UnoStrategy`` reads per step."""

    def apply_corruption(input_ids, loss_mask, microbatch_idx=0):
        return recipe.dllm_strategy.apply_corruption(
            input_ids,
            loss_mask,
            DLLM_CFG["mask_token_id"],
            eps=1e-3,
            block_size=None,
            half_life_ratio=None,
            generator=torch.Generator().manual_seed(seed),
        )

    recipe = types.SimpleNamespace(
        dist_env=types.SimpleNamespace(device=torch.device("cpu")),
        model_parts=[model],
        distributed_config=types.SimpleNamespace(defer_fsdp_grad_sync=True, autocast_dtype=None),
        te_fp8=None,
        device_mesh=None,
        dllm_loss_fn=loss_fn,
        _dllm_loss_buffer=[],
        _get_dp_group_size=lambda include_cp=True: 1.0,
        step_scheduler=types.SimpleNamespace(step=step),
        _apply_corruption=apply_corruption,
    )
    return recipe


def _batch() -> dict:
    """Return a batch with ``input_ids`` and ``loss_mask`` of shape ``[batch, sequence]``; the first third is prompt."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB - 1, (BATCH, SEQ_LEN))
    loss_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    loss_mask[:, SEQ_LEN // 3 :] = 1
    return {"input_ids": input_ids, "loss_mask": loss_mask}


def _train_step(strategy, loss_fn, model, step: int, seed: int) -> float:
    """Run one optimizer step's worth of the recipe: ``pre_step`` then ``forward_backward``; return the loss."""
    recipe = _recipe(model, loss_fn, step, seed)
    recipe.dllm_strategy = strategy
    batch = _batch()
    num_noise, _ = strategy.pre_step(recipe, [batch])
    loss_buffer = []
    strategy.forward_backward(recipe, 0, batch, loss_buffer=loss_buffer, num_diffusion_tokens=num_noise, num_batches=1)
    return loss_buffer[0].item()


def test_uno_step_trains_only_the_lora_adapter():
    """Noise -> gated forward -> TV loss -> backward must give finite gradients on LoRA weights only."""
    strategy = get_dllm_strategy("uno")
    loss_fn = strategy.create_loss_fn(DLLM_CFG)
    model = _tiny_lora_model()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.05)

    loss = _train_step(strategy, loss_fn, model, step=0, seed=0)
    assert torch.isfinite(torch.tensor(loss)) and loss > 0
    with_grad = {name for name, p in model.named_parameters() if p.grad is not None}
    assert with_grad and all("lora_" in name for name in with_grad)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_uno_loss_decreases_under_optimization():
    """With the noise held fixed, a few Adam steps on the adapter must drive the TV loss down."""
    strategy = get_dllm_strategy("uno")
    loss_fn = strategy.create_loss_fn(DLLM_CFG)
    model = _tiny_lora_model()
    base = {name: p.detach().clone() for name, p in model.named_parameters() if "lora_" not in name}
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)

    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        losses.append(_train_step(strategy, loss_fn, model, step=0, seed=0))
        optimizer.step()

    assert losses[-1] < losses[0], f"Uno TV loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    assert all(torch.equal(p, base[name]) for name, p in model.named_parameters() if "lora_" not in name)


def test_uno_curriculum_switches_block_size_across_steps():
    """``pre_step`` applies the curriculum: steps 0-2 use block size 2, steps from 3 on use block size 4."""
    strategy = get_dllm_strategy("uno")
    loss_fn = strategy.create_loss_fn(DLLM_CFG)
    model = _tiny_lora_model()
    seen = []
    for step in range(6):
        _train_step(strategy, loss_fn, model, step=step, seed=step)
        seen.append(strategy.block_size)
    assert seen == [2, 2, 2, 4, 4, 4]
