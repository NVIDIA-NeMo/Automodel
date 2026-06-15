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
"""End-to-end proof that the Engine serves verl's backend patterns on CPU.

verl produces a jagged/THD micro-batch (flat tokens + offsets), derives
next-token targets by ``roll(-1)``, and carries advantages / old logprobs as
sibling fields. Two integration doors are exercised:

1. **Datum door** — ``datums_from_verl`` (the adapter that would live in verl's
   ``AutomodelEngine``) un-packs the jagged batch into ``list[Datum]``; the
   Engine collates, normalizes, and runs a built-in RL loss.
2. **Pass-through door** — a ``PackedBatch`` hands verl's already-packed tensors
   straight to the Engine with a verl-style scalar-returning loss closure
   ``loss(model_output, data, dp_group) -> (loss, metrics)``; the caller owns
   normalization.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.datasets.datum import Datum, PackedBatch
from nemo_automodel.components.training.model_output import ModelOutput
from nemo_automodel.components.training.engine import Engine


class ToyLM(nn.Module):
    def __init__(self, vocab=16, dim=8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def _engine(vocab=16):
    torch.manual_seed(0)
    model = ToyLM(vocab)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    return Engine(model_parts=[model], optimizers=[opt]), model


def _fake_verl_microbatch(vocab=16):
    """Mimic verl's jagged micro-batch: flat values + offsets + sibling fields."""
    seqs = [torch.randint(0, vocab, (5,)), torch.randint(0, vocab, (3,))]
    flat = torch.cat(seqs)
    offsets = [0, 5, 8]
    total = flat.shape[0]
    return {
        "flat_input_ids": flat,
        "offsets": offsets,
        "advantages": torch.randn(total),
        "old_logprobs": -torch.rand(total),
        "loss_mask": torch.ones(total),
    }


# ── Door 1: the datums_from_verl adapter (lives in verl's AutomodelEngine) ───


def datums_from_verl(mb) -> list[Datum]:
    flat, offsets = mb["flat_input_ids"], mb["offsets"]
    datums = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        seq = flat[a:b]
        datums.append(
            Datum(
                input_ids=seq,
                loss_inputs={
                    "target_tokens": torch.roll(seq, -1),  # verl derives targets by roll(-1)
                    "weights": mb["loss_mask"][a:b],
                    "advantages": mb["advantages"][a:b],
                    "logprobs": mb["old_logprobs"][a:b],
                },
            )
        )
    return datums


def _datum_ppo_loss(model_output, datums, *, clip_eps=0.2, **kwargs):
    """Caller-supplied PPO LossFn (Datum-door signature). RL objectives are not
    Engine built-ins; the consumer provides them."""
    out = []
    for lp, d in zip(model_output.logprobs, datums):
        old = d.loss_inputs["logprobs"].to(lp)
        adv = d.loss_inputs["advantages"].to(lp)
        ratio = torch.exp(lp - old)
        out.append(-torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv))
    return out


def test_datum_door_runs_ppo_through_engine():
    engine, model = _engine()
    datums = datums_from_verl(_fake_verl_microbatch())
    out = engine.forward_backward(datums, loss_fn=_datum_ppo_loss)
    assert isinstance(out, ModelOutput)
    assert torch.isfinite(out.loss)
    assert len(out.logprobs) == 2  # per-datum, input order
    ok, grad_norm = engine.optimizer_step()
    assert ok and grad_norm >= 0.0


# ── Door 2: PackedBatch pass-through with a verl-style scalar loss closure ───


def verl_style_ppo_loss(model_output, *, data, dp_group, clip_eps=0.2):
    """verl's loss signature: (model_output, data, dp_group) -> (loss, metrics).

    Caller-owned normalization by data["num_tokens"] (verl all-reduces this).
    """
    losses = []
    for lp, adv, old in zip(model_output.logprobs, data["adv_split"], data["old_split"]):
        ratio = torch.exp(lp - old)
        losses.append((-torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)).sum())
    loss = torch.stack(losses).sum() / data["num_tokens"]
    return loss, {"pg_loss": float(loss.detach())}


def test_passthrough_door_runs_verl_closure():
    engine, model = _engine()
    mb = _fake_verl_microbatch()
    flat, offsets = mb["flat_input_ids"], mb["offsets"]
    seq_lens = [b - a for a, b in zip(offsets[:-1], offsets[1:])]

    packed = PackedBatch(
        model_inputs={"input_ids": flat.unsqueeze(0)},  # [1, total] THD
        seq_lens=seq_lens,
        targets=torch.roll(flat, -1),  # flat next-token targets
    )
    # verl carries advantages/old logprobs split per-datum + the global token count.
    data = {
        "adv_split": [mb["advantages"][a:b] for a, b in zip(offsets[:-1], offsets[1:])],
        "old_split": [mb["old_logprobs"][a:b] for a, b in zip(offsets[:-1], offsets[1:])],
        "num_tokens": float(mb["loss_mask"].sum()),
    }
    loss_fn = lambda mo: verl_style_ppo_loss(mo, data=data, dp_group=engine.dp_group)  # noqa: E731

    out = engine.forward_backward(packed, loss_fn=loss_fn)
    assert isinstance(out, ModelOutput)
    assert torch.isfinite(out.loss)
    assert "pg_loss" in out.metrics  # metrics from the verl closure propagate
    assert [lp.shape[0] for lp in out.logprobs] == seq_lens  # per-datum via seq_lens
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_passthrough_forward_only_no_grad():
    engine, model = _engine()
    mb = _fake_verl_microbatch()
    flat, offsets = mb["flat_input_ids"], mb["offsets"]
    packed = PackedBatch(
        model_inputs={"input_ids": flat.unsqueeze(0)},
        seq_lens=[b - a for a, b in zip(offsets[:-1], offsets[1:])],
        targets=torch.roll(flat, -1),
    )
    out = engine.forward_backward(packed, loss_fn=None, forward_only=True)
    assert out.loss is None
    assert [lp.shape[0] for lp in out.logprobs] == [5, 3]  # extraction still works
    assert all(p.grad is None for p in model.parameters())


def test_passthrough_multi_microbatch_accumulates():
    """verl hands the whole batch; the pass-through micro-batches with grad accum."""
    engine, model = _engine()

    def _packed_and_closure(mb):
        flat, offsets = mb["flat_input_ids"], mb["offsets"]
        seq_lens = [b - a for a, b in zip(offsets[:-1], offsets[1:])]
        packed = PackedBatch(
            model_inputs={"input_ids": flat.unsqueeze(0)}, seq_lens=seq_lens, targets=torch.roll(flat, -1)
        )
        adv = [mb["advantages"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        old = [mb["old_logprobs"][a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        ntok = float(mb["loss_mask"].sum())

        def closure(mo, adv=adv, old=old, ntok=ntok):
            losses = [(-(torch.exp(lp - o) * a)).sum() for lp, a, o in zip(mo.logprobs, adv, old)]
            loss = torch.stack(losses).sum() / ntok
            return loss, {"pg_loss": float(loss.detach())}

        return packed, closure

    m1, c1 = _packed_and_closure(_fake_verl_microbatch())
    m2, c2 = _packed_and_closure(_fake_verl_microbatch())
    out = engine.forward_backward([m1, m2], loss_fn=[c1, c2])  # list of microbatches + per-mb closures
    assert torch.isfinite(out.loss)
    assert "pg_loss" in out.metrics
    assert len(out.logprobs) == 4  # 2 datums x 2 microbatches, concatenated in order
    # grads accumulated across both microbatches before any optimizer step
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
