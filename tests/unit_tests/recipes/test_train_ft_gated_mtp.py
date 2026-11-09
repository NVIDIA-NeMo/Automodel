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

"""Gated-LoRA MTP through the real non-PP ``_forward_backward_step`` of the fine-tuning recipe.

The LoRA gradients of a training step are compared with references built without the dense layout: one causal
forward per (anchor, depth), so that the layout, the mask, the loss alignment, and the soft-MTP gradient routing
are all checked against first principles.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import Qwen3Config, Qwen3ForCausalLM

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules, lora_token_gate
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.speculative.gated_mtp import GatedMTPConfig
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

_VOCAB = 64
_MASK_IDS = [62, 63]
_SCALING = 0.5


def _build_lora_model() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    config = Qwen3Config(
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    model = Qwen3ForCausalLM(config)
    apply_lora_to_linear_modules(model, PeftConfig(target_modules=["*_proj"], dim=4, alpha=8, use_triton=False))
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.1)
    return model


def _make_recipe(model, monkeypatch, *, distillation_mode: str):
    """Build a minimal harness around the real non-PP ``_forward_backward_step`` with gated MTP enabled."""
    recipe = TrainFinetuneRecipeForNextTokenPrediction.__new__(TrainFinetuneRecipeForNextTokenPrediction)
    object.__setattr__(
        recipe, "cfg", SimpleNamespace(mtp=SimpleNamespace(distillation_mode=distillation_mode, scaling_factor=None))
    )
    object.__setattr__(recipe, "dist_env", SimpleNamespace(device=torch.device("cpu")))
    object.__setattr__(recipe, "device_mesh", object())
    object.__setattr__(recipe, "pp_enabled", False)
    object.__setattr__(recipe, "tokenizer", SimpleNamespace(pad_token_id=0))
    object.__setattr__(recipe, "te_fp8", None)
    object.__setattr__(recipe, "model_parts", [model])
    object.__setattr__(recipe, "distributed_config", SimpleNamespace(defer_fsdp_grad_sync=True))
    object.__setattr__(recipe, "loss_fn", MaskedCrossEntropy())
    object.__setattr__(recipe, "domain_mixture", None)
    object.__setattr__(recipe, "_get_cp_group_size", lambda: 1)
    object.__setattr__(recipe, "_get_dp_group_size", lambda include_cp=False: 1)
    object.__setattr__(recipe, "gated_mtp", GatedMTPConfig(mask_token_ids=_MASK_IDS, scaling_factor=_SCALING).build())
    recipe._mtp_accept_metrics = []

    class _IdentitySharder:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def shard(self, batch):
            return nullcontext, batch

    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.ContextParallelSharder", _IdentitySharder)
    return recipe


def _batch():
    """Two rows of 6 tokens; ``labels[t]`` is token ``t + 1``, unsupervised at the end of each row."""
    torch.manual_seed(1)
    input_ids = torch.randint(0, 60, (2, 6))
    labels = torch.cat([input_ids[:, 1:], torch.full((2, 1), -100)], dim=1)
    labels[1, 3:] = -100
    return input_ids, labels


def _lora_grads(model):
    return {name: param.grad.clone() for name, param in model.named_parameters() if "lora_" in name}


def _sequential_mtp_logits(model, input_ids, anchor: int, depth: int) -> torch.Tensor:
    """Logits of the depth-``depth`` mask token appended after ``anchor``, from a plain causal forward.

    Args:
        input_ids: Tensor of shape [1, sequence], one row.

    Returns:
        Tensor of shape [1, vocab].
    """
    ids = torch.cat([input_ids[:, : anchor + 1], torch.tensor([_MASK_IDS[:depth]])], dim=1)
    position_ids = torch.cat([torch.arange(anchor + 1), anchor + torch.arange(1, depth + 1)]).unsqueeze(0)
    gate = torch.tensor([[False] * (anchor + 1) + [True] * depth])
    with lora_token_gate(model, gate):
        return model(input_ids=ids, position_ids=position_ids).logits[:, -1]


def _reference_lora_grads(input_ids, labels, *, soft: bool):
    """LoRA gradients of the MTP objective, built from one causal forward per (anchor, depth).

    Depth ``d`` at anchor ``t`` predicts ``labels[t + d]``. Hard: cross-entropy to that token. Soft: on top of the
    hard term, the soft-MTP KL-recovery turns the gradient into that of a cross-entropy to the base model's own
    next-token distribution at position ``t + d``.
    """
    model = _build_lora_model()
    model.train()
    num_label_tokens = int((labels != -100).sum())
    num_depths = len(_MASK_IDS)
    with torch.no_grad(), lora_token_gate(model, torch.zeros_like(input_ids, dtype=torch.bool)):
        base_probs = model(input_ids=input_ids).logits.softmax(-1)
    loss = 0.0
    for b in range(input_ids.shape[0]):
        for t in range(input_ids.shape[1]):
            for d in range(1, num_depths + 1):
                if t + d >= labels.shape[1] or labels[b, t + d] == -100:
                    continue
                logits = _sequential_mtp_logits(model, input_ids[b : b + 1], t, d)
                if soft:
                    term = -(base_probs[b : b + 1, t + d] * logits.float().log_softmax(-1)).sum()
                else:
                    term = F.cross_entropy(logits.float(), labels[b : b + 1, t + d], reduction="sum")
                loss = loss + _SCALING / num_depths * term / num_label_tokens
    loss.backward()
    return _lora_grads(model)


@pytest.mark.parametrize("distillation_mode", ["hard", "soft"])
def test_training_step_lora_gradients_match_sequential_reference(monkeypatch, distillation_mode):
    model = _build_lora_model()
    model.train()
    recipe = _make_recipe(model, monkeypatch, distillation_mode=distillation_mode)
    input_ids, labels = _batch()
    loss_buffer = []

    recipe._forward_backward_step(
        0,
        {"input_ids": input_ids, "labels": labels},
        loss_buffer=loss_buffer,
        num_label_tokens=int((labels != -100).sum()),
        num_batches=1,
        is_train=True,
    )

    expected = _reference_lora_grads(input_ids, labels, soft=distillation_mode == "soft")
    grads = _lora_grads(model)
    assert grads.keys() == expected.keys()
    for name, grad in grads.items():
        assert torch.count_nonzero(grad) > 0, name
        torch.testing.assert_close(grad, expected[name], rtol=1e-4, atol=1e-6, msg=lambda msg: f"{name}: {msg}")
    assert len(loss_buffer) == 1 and torch.isfinite(loss_buffer[0])
    # the gate is removed once the step, including backward, is done
    assert all(getattr(m, "_lora_token_gate", None) is None for m in model.modules())


def test_eval_step_scores_drafts_against_the_base_model(monkeypatch):
    model = _build_lora_model()
    model.eval()
    recipe = _make_recipe(model, monkeypatch, distillation_mode="hard")
    input_ids, labels = _batch()

    with torch.no_grad():
        recipe._forward_backward_step(
            0,
            {"input_ids": input_ids, "labels": labels},
            loss_buffer=[],
            num_label_tokens=int((labels != -100).sum()),
            num_batches=1,
            is_train=False,
        )
        with lora_token_gate(model, torch.zeros_like(input_ids, dtype=torch.bool)):
            verifier = model(input_ids=input_ids).logits.argmax(-1)
        drafts = {
            (b, t, d): _sequential_mtp_logits(model, input_ids[b : b + 1], t, d).argmax(-1).item()
            for b in range(2)
            for t in range(6)
            for d in (1, 2)
        }

    (metrics,) = recipe._mtp_accept_metrics
    # depth d at anchor t is supervised while labels[t + d] is; the draft is accepted when it equals the base
    # model's own greedy token at t + d, and a depth only counts while every shallower depth was accepted too.
    supervised = {(b, t, d) for (b, t, d) in drafts if t + d < 6 and labels[b, t + d] != -100}
    valid = [sum(1 for b, t, d in supervised if d == depth) for depth in (1, 2)]
    hits_1 = {(b, t) for b, t, d in supervised if d == 1 and drafts[b, t, 1] == verifier[b, t + 1]}
    hits_2 = {(b, t) for b, t in hits_1 if (b, t, 2) in supervised and drafts[b, t, 2] == verifier[b, t + 2]}
    assert valid == [6, 4]
    torch.testing.assert_close(metrics.valid, torch.tensor(valid))
    torch.testing.assert_close(metrics.prefix_hits, torch.tensor([len(hits_1), len(hits_2)]))


def test_validate_rejects_setups_that_break_losslessness(monkeypatch):
    model = _build_lora_model()
    recipe = _make_recipe(model, monkeypatch, distillation_mode="hard")
    object.__setattr__(recipe, "peft_config", None)
    object.__setattr__(recipe, "mesh_context", SimpleNamespace(cp_size=1, tp_size=1))
    model.supports = SimpleNamespace(mtp_enabled=False)
    with pytest.raises(ValueError, match="token-gated LoRA adapter"):
        recipe._validate_gated_mtp(is_packed=False)

    object.__setattr__(recipe, "peft_config", PeftConfig(target_modules=["*_proj"], dim=4))
    recipe._validate_gated_mtp(is_packed=False)
    with pytest.raises(NotImplementedError, match="packed sequences"):
        recipe._validate_gated_mtp(is_packed=True)
    object.__setattr__(recipe, "mesh_context", SimpleNamespace(cp_size=2, tp_size=1))
    with pytest.raises(NotImplementedError, match="context, or tensor parallelism"):
        recipe._validate_gated_mtp(is_packed=False)

    object.__setattr__(recipe, "mesh_context", SimpleNamespace(cp_size=1, tp_size=1))
    model.model.layers[0].self_attn.sliding_window = 4
    try:
        with pytest.raises(NotImplementedError, match="sliding-window"):
            recipe._validate_gated_mtp(is_packed=False)
    finally:
        model.model.layers[0].self_attn.sliding_window = None
