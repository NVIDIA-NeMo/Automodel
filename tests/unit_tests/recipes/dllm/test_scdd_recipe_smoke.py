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

"""CPU smoke test for the SCDD dLLM training path.

The full ``DiffusionLMSFTRecipe`` loop is CUDA-only, so this exercises the same
sequence the recipe drives per micro-batch — strategy corruption, batch
preparation, a real transformer forward, the SCDD loss, and backward — against a
hermetically constructed tiny model. It answers "would this actually train?"
without a GPU or a checkpoint download; the GPU counterpart is
``tests/functional_tests/dllm/L2_DLLM_SCDD_Smoke.sh``.
"""

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from nemo_automodel.recipes.dllm.strategy import get_dllm_strategy

VOCAB = 128
MASK_TOKEN_ID = VOCAB - 1
SEQ_LEN = 24
BATCH = 4

DLLM_CFG = {
    "mode": "scdd",
    "mask_token_id": MASK_TOKEN_ID,
    "vocab_size": VOCAB,
    "eps": 1e-3,
    "num_timesteps": 1000,
    "uniform_ratio": 0.1,
    "schedule_shape": 1.0,
    "schedule_peak": 0.5,
}


def _tiny_model() -> LlamaForCausalLM:
    """Build a 2-layer randomly-initialised causal LM with no network access."""
    config = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=SEQ_LEN,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(config)


def _batch() -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, loss_mask)``, both of shape ``[batch, sequence]``.

    The first third of each row stands in for a prompt (unsupervised); the rest
    is the supervised response.
    """
    torch.manual_seed(1)
    input_ids = torch.randint(0, VOCAB - 1, (BATCH, SEQ_LEN))
    loss_mask = torch.zeros(BATCH, SEQ_LEN, dtype=torch.long)
    loss_mask[:, SEQ_LEN // 3 :] = 1
    return input_ids, loss_mask


def _step(strategy, loss_fn, model, input_ids, loss_mask, seed):
    """Run one micro-batch exactly as ``_forward_backward_step`` does.

    Args:
        strategy: The ``SCDDStrategy`` under test.
        loss_fn: The ``SCDDLoss`` built by the strategy.
        model: The tiny causal LM.
        input_ids: Clean token IDs of shape ``[batch, sequence]``.
        loss_mask: Supervised-position mask of shape ``[batch, sequence]``.
        seed: Seed for the corruption generator; reuse it to hold the noise fixed.

    Returns:
        The scalar loss tensor for this micro-batch.
    """
    noisy_input_ids, noise_mask, p_mask = strategy.apply_corruption(
        input_ids,
        loss_mask,
        MASK_TOKEN_ID,
        eps=DLLM_CFG["eps"],
        block_size=None,
        half_life_ratio=None,
        generator=torch.Generator().manual_seed(seed),
    )
    batch = strategy.prepare_batch(
        {"input_ids": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)},
        noisy_input_ids,
        noise_mask,
        input_ids,
    )
    logits = model(**batch).logits
    return loss_fn(
        logits=logits,
        target_ids=input_ids,
        noise_mask=noise_mask,
        p_mask=p_mask,
        loss_mask=loss_mask,
        noisy_input_ids=noisy_input_ids,
        num_diffusion_tokens=int(loss_mask.sum()),
    ).total_loss


def test_scdd_micro_batch_runs_end_to_end():
    """Corruption -> forward -> loss -> backward must produce finite gradients on
    the parameters the optimizer would update."""
    strategy = get_dllm_strategy("scdd")
    loss_fn = strategy.create_loss_fn(DLLM_CFG)
    model = _tiny_model()
    input_ids, loss_mask = _batch()

    loss = _step(strategy, loss_fn, model, input_ids, loss_mask, seed=0)
    assert torch.isfinite(loss)
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


def test_scdd_loss_decreases_under_optimization():
    """With the corruption held fixed, a few Adam steps must drive the SCDD
    objective down — the signal that its gradients point the right way rather
    than merely being finite."""
    strategy = get_dllm_strategy("scdd")
    loss_fn = strategy.create_loss_fn(DLLM_CFG)
    model = _tiny_model()
    input_ids, loss_mask = _batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss = _step(strategy, loss_fn, model, input_ids, loss_mask, seed=0)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert all(torch.isfinite(torch.tensor(losses)))
    assert losses[-1] < losses[0], f"SCDD loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_scdd_batch_feeds_corrupted_tokens_to_the_model():
    """The model must see the corrupted sequence (and no attention mask, since
    it attends bidirectionally), while the loss scores the clean targets."""
    strategy = get_dllm_strategy("scdd")
    strategy.create_loss_fn(DLLM_CFG)
    input_ids, loss_mask = _batch()

    noisy_input_ids, noise_mask, _ = strategy.apply_corruption(
        input_ids,
        loss_mask,
        MASK_TOKEN_ID,
        eps=DLLM_CFG["eps"],
        block_size=None,
        half_life_ratio=None,
        generator=torch.Generator().manual_seed(3),
    )
    batch = strategy.prepare_batch(
        {"input_ids": input_ids.clone(), "attention_mask": torch.ones_like(input_ids)},
        noisy_input_ids,
        noise_mask,
        input_ids,
    )
    assert torch.equal(batch["input_ids"], noisy_input_ids)
    assert "attention_mask" not in batch
    assert noise_mask.any(), "corruption produced nothing to learn from"


@pytest.mark.parametrize("uniform_ratio", [0.0, 0.05, 0.4])
def test_scdd_runs_across_the_uniform_ratio_range(uniform_ratio):
    """Both ends of the schedule are reachable from config: 0 degenerates to
    MDLM and a large ratio makes most corrupted tokens visible-but-wrong."""
    strategy = get_dllm_strategy("scdd")
    loss_fn = strategy.create_loss_fn({**DLLM_CFG, "uniform_ratio": uniform_ratio})
    model = _tiny_model()
    input_ids, loss_mask = _batch()

    loss = _step(strategy, loss_fn, model, input_ids, loss_mask, seed=5)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
