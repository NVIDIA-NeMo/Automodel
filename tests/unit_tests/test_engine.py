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
"""CPU unit tests for the Engine spine via the injection construction path."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components.datasets.datum import Datum
from nemo_automodel.components.training.engine import Engine
from nemo_automodel.components.training.model_output import ModelOutput


class ToyLM(nn.Module):
    """Tiny next-token model: embedding -> linear -> [B, T, V] logits."""

    def __init__(self, vocab: int = 16, dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.head = nn.Linear(dim, vocab)

    def forward(self, input_ids, position_ids=None):
        return SimpleNamespace(logits=self.head(self.embed(input_ids)))


def toy_loss(logits=None, labels=None, num_label_tokens=None):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)


def _engine(vocab=16, dim=8, lr=0.1):
    torch.manual_seed(0)
    model = ToyLM(vocab, dim)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    return Engine(model_parts=[model], optimizers=[opt]), model


def _datums(vocab=16):
    return [
        Datum(
            input_ids=torch.randint(0, vocab, (5,)),
            loss_fn_inputs={"target_tokens": torch.randint(0, vocab, (5,)), "weights": torch.ones(5)},
        ),
        Datum(
            input_ids=torch.randint(0, vocab, (3,)),
            loss_fn_inputs={"target_tokens": torch.randint(0, vocab, (3,)), "weights": torch.ones(3)},
        ),
    ]


# ── construction / introspection ────────────────────────────────────────────


def test_injection_construction():
    engine, model = _engine()
    assert engine.parts == [model]
    assert engine.model is model
    assert engine.pp_enabled is False
    assert engine.dp_size == 1
    assert engine.dp_rank == 0


# ── forward_backward ─────────────────────────────────────────────────────────


def test_forward_backward_dict_returns_dict_and_grads():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    out = engine.forward_backward(batch, loss_fn=toy_loss)
    assert isinstance(out, dict) and "loss" in out and "metrics" in out
    assert torch.isfinite(out["loss"])
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def _rl_datums(vocab=16):
    """Datums carrying old logprobs + advantages for a caller-supplied RL loss."""
    return [
        Datum(
            input_ids=torch.randint(0, vocab, (5,)),
            loss_fn_inputs={
                "target_tokens": torch.randint(0, vocab, (5,)),
                "weights": torch.ones(5),
                "logprobs": -torch.rand(5),
                "advantages": torch.randn(5),
            },
        ),
        Datum(
            input_ids=torch.randint(0, vocab, (3,)),
            loss_fn_inputs={
                "target_tokens": torch.randint(0, vocab, (3,)),
                "weights": torch.tensor([1.0, 1.0, 0.0]),
                "logprobs": -torch.rand(3),
                "advantages": torch.randn(3),
            },
        ),
    ]


def test_forward_backward_datums_default_cross_entropy():
    engine, model = _engine()
    out = engine.forward_backward(_datums())  # default loss_fn="cross_entropy"
    assert isinstance(out, ModelOutput)
    assert out.loss is not None and torch.isfinite(out.loss)
    assert out.metrics["loss"] == pytest.approx(float(out.loss))
    assert len(out.logprobs) == 2  # per-datum, aligned to input order
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_caller_supplied_rl_lossfn_runs_and_backprops():
    # RL objectives (PPO, importance sampling) are caller-supplied LossFns, not
    # Engine built-ins. Verify the Datum door runs an advantage-reading loss and
    # backprops through it.
    def importance_sampling(model_output, datums, **kwargs):
        out = []
        for lp, d in zip(model_output.logprobs, datums):
            old = d.loss_fn_inputs["logprobs"].to(lp)
            adv = d.loss_fn_inputs["advantages"].to(lp)
            out.append(-(torch.exp(lp - old) * adv))
        return out

    engine, model = _engine()
    out = engine.forward_backward(_rl_datums(), loss_fn=importance_sampling)
    assert torch.isfinite(out.loss)
    assert any(p.grad is not None for p in model.parameters())


def test_custom_lossfn_matches_cross_entropy():
    def my_ce(model_output, datums):
        return [-lp for lp in model_output.logprobs]

    torch.manual_seed(0)
    e1, _ = _engine()
    l1 = float(e1.forward_backward(_datums()).loss)  # default loss_fn = cross_entropy
    torch.manual_seed(0)
    e2, _ = _engine()
    l2 = float(e2.forward_backward(_datums(), loss_fn=my_ce).loss)
    assert l1 == pytest.approx(l2, rel=1e-5)


def test_datums_forward_only_no_grads():
    engine, model = _engine()
    out = engine.forward_backward(_datums(), forward_only=True)
    assert torch.isfinite(out.loss)
    assert all(p.grad is None for p in model.parameters())


def test_multiple_microbatches_accumulate_datums():
    engine, model = _engine()
    mbs = [_datums(), _datums()]  # list[list[Datum]]
    out = engine.forward_backward(mbs)  # default loss_fn = cross_entropy
    assert torch.isfinite(out.loss)
    assert len(out.logprobs) == 4  # 2 datums x 2 microbatches, in order


def test_reduce_token_level_normalization():
    # Two datums, token-level loss; weighted sum / global token count.
    d = [
        Datum(input_ids=torch.tensor([1, 2]), loss_fn_inputs={"weights": torch.tensor([1.0, 1.0])}),
        Datum(input_ids=torch.tensor([3]), loss_fn_inputs={"weights": torch.tensor([0.0])}),
    ]
    per = [torch.tensor([2.0, 4.0]), torch.tensor([8.0])]  # token-level
    # weighted sum = 2*1 + 4*1 + 8*0 = 6; global token count = 2 -> 3.0
    loss = Engine._reduce_datum_losses(per, d, token_denom=2.0, sample_denom=2.0)
    assert float(loss) == pytest.approx(3.0)


def test_reduce_sample_level_normalization():
    d = [Datum(input_ids=torch.tensor([1, 2])), Datum(input_ids=torch.tensor([3]))]
    per = [torch.tensor(2.0), torch.tensor(4.0)]  # scalar per datum -> sample-level
    loss = Engine._reduce_datum_losses(per, d, token_denom=99.0, sample_denom=2.0)
    assert float(loss) == pytest.approx(3.0)  # (2 + 4) / 2


def test_global_denominator_local():
    engine, _ = _engine()
    token, sample = engine._global_denominator(_datums())
    assert token == pytest.approx(8.0)  # weights: ones(5) + ones(3)
    assert sample == pytest.approx(2.0)


def test_forward_only_leaves_no_grads():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss, forward_only=True)
    assert all(p.grad is None for p in model.parameters())


def test_list_of_microbatches_accumulates():
    engine, model = _engine()
    mbs = [
        {"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))},
        {"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))},
    ]
    out = engine.forward_backward(mbs, loss_fn=toy_loss)
    assert torch.isfinite(out["loss"])


def test_pp_path_raises():
    engine, _ = _engine()
    engine.pp = object()  # simulate AutoPipeline present
    with pytest.raises(NotImplementedError, match="pipeline parallelism"):
        engine.forward_backward({"input_ids": torch.zeros(1, 1, dtype=torch.long)}, loss_fn=toy_loss)


# ── forward (per-datum logprobs / entropy) ───────────────────────────────────


def test_forward_returns_per_datum_logprobs_and_entropy():
    engine, _ = _engine()
    datums = _datums()
    out = engine.forward(datums)
    assert isinstance(out, ModelOutput)
    assert [lp.shape[0] for lp in out.logprobs] == [d.seq_len for d in datums]
    assert [e.shape[0] for e in out.entropy] == [d.seq_len for d in datums]
    # logprobs are log-probabilities -> non-positive.
    assert all((lp <= 1e-4).all() for lp in out.logprobs)


def test_forward_requires_datums():
    engine, _ = _engine()
    with pytest.raises(TypeError, match="Datum"):
        engine.forward([{"input_ids": torch.zeros(3, dtype=torch.long)}])


# ── optimizer ─────────────────────────────────────────────────────────────────


def test_optimizer_step_updates_params():
    engine, model = _engine(lr=0.5)
    before = model.head.weight.detach().clone()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    ok, grad_norm = engine.optimizer_step()
    assert ok is True
    assert grad_norm >= 0.0
    assert not torch.equal(before, model.head.weight)


def test_optim_step_sets_lr():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    metrics = engine.optim_step(lr=0.123)
    assert metrics["lr"] == 0.123
    assert metrics["update_succeeded"] is True
    assert all(g["lr"] == 0.123 for opt in engine.optimizers for g in opt.param_groups)


def test_zero_grad():
    engine, model = _engine()
    batch = {"input_ids": torch.randint(0, 16, (2, 6)), "labels": torch.randint(0, 16, (2, 6))}
    engine.forward_backward(batch, loss_fn=toy_loss)
    engine.zero_grad()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())


# ── modes / device / export ──────────────────────────────────────────────────


def test_train_eval_mode_restore():
    engine, model = _engine()
    model.eval()
    with engine.train_mode():
        assert model.training is True
    assert model.training is False  # restored


def test_to_cpu_is_safe():
    engine, model = _engine()
    engine.to("cpu")
    assert next(model.parameters()).device.type == "cpu"


def test_export_weights_yields_named_tensors():
    engine, _ = _engine()
    weights = dict(engine.export_weights())
    assert "head.weight" in weights and "embed.weight" in weights
    assert isinstance(weights["head.weight"], torch.Tensor)


# ── MoE aux-loss scale + DP grad compensation ────────────────────────────────


class _FakeCpMesh:
    """Minimal device-mesh stand-in exposing a ``cp`` dim of the given size."""

    def __init__(self, cp_size):
        self._cp = SimpleNamespace(size=lambda: cp_size)

    def __getitem__(self, name):
        assert name == "cp"
        return self._cp


def test_set_moe_scale_noop_without_moe_mesh(monkeypatch):
    from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

    sentinel = torch.tensor(-1.0)
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", sentinel)
    engine, _ = _engine()
    assert engine.moe_mesh is None
    engine._set_moe_scale(4)
    assert MoEAuxLossAutoScaler.main_loss_backward_scale is sentinel


def test_set_moe_scale_averages_microbatches_and_restores_cp(monkeypatch):
    from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(-1.0))
    engine, _ = _engine()
    engine.moe_mesh = object()

    engine._set_moe_scale(4)  # no device mesh -> cp=1
    assert float(MoEAuxLossAutoScaler.main_loss_backward_scale) == pytest.approx(0.25)

    engine.device_mesh = _FakeCpMesh(2)
    engine._set_moe_scale(4)
    assert float(MoEAuxLossAutoScaler.main_loss_backward_scale) == pytest.approx(0.5)


def test_forward_backward_sets_moe_scale_in_every_door(monkeypatch):
    from nemo_automodel.components.datasets.datum import PackedBatch
    from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler

    engine, _ = _engine()
    engine.moe_mesh = object()

    # dict door: 2 microbatches -> cp(1)/2
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(-1.0))
    mbs = [{"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))} for _ in range(2)]
    engine.forward_backward(mbs, loss_fn=toy_loss)
    assert float(MoEAuxLossAutoScaler.main_loss_backward_scale) == pytest.approx(0.5)

    # Datum door: 2 microbatch datum lists -> cp(1)/2
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(-1.0))
    engine.forward_backward([_datums(), _datums()])
    assert float(MoEAuxLossAutoScaler.main_loss_backward_scale) == pytest.approx(0.5)

    # packed door: 2 PackedBatch microbatches -> cp(1)/2
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(-1.0))
    flat = torch.randint(0, 16, (8,))
    packs = [
        PackedBatch(
            model_inputs={"input_ids": flat.unsqueeze(0)},
            seq_lens=[5, 3],
            targets=torch.roll(flat, -1),
        )
        for _ in range(2)
    ]
    engine.forward_backward(packs, loss_fn=lambda mo: torch.stack([lp.sum() for lp in mo.logprobs]).sum())
    assert float(MoEAuxLossAutoScaler.main_loss_backward_scale) == pytest.approx(0.5)


def test_datum_backward_cancels_fsdp_grad_average(monkeypatch):
    # The Datum-door backward scales by the DP-only size (CP is handled by the
    # output gather), so a 4x DP group yields 4x the pre-average gradient.
    grads = {}
    for dp in (1, 4):
        monkeypatch.setattr(Engine, "_dp_only_size", property(lambda self, _dp=dp: _dp))
        engine, model = _engine()
        torch.manual_seed(1)
        engine.forward_backward([_datums()])
        grads[dp] = model.head.weight.grad.clone()
    assert torch.allclose(grads[4], 4 * grads[1])


def test_dict_backward_scales_only_globally_normalized_losses(monkeypatch):
    grads = {}
    for dp, num_label_tokens in ((1, 16), (4, 16), (4, None)):
        monkeypatch.setattr(Engine, "dp_size", property(lambda self, _dp=dp: _dp))
        engine, model = _engine()
        torch.manual_seed(1)
        batch = {"input_ids": torch.randint(0, 16, (2, 4)), "labels": torch.randint(0, 16, (2, 4))}
        engine.forward_backward(batch, loss_fn=toy_loss, num_label_tokens=num_label_tokens)
        grads[(dp, num_label_tokens)] = model.head.weight.grad.clone()
    # Global-token normalization: grads are DP-size-invariant only with the x dp_size
    # compensation, so the pre-average grad must be 4x.
    assert torch.allclose(grads[(4, 16)], 4 * grads[(1, 16)])
    # Locally normalized loss (no global count): FSDP's average is already correct.
    assert torch.allclose(grads[(4, None)], grads[(1, 16)])


# ── pack_datums: packed Datum door (text-only, CP=1) ─────────────────────────


def _packing_engines(lr=0.1):
    """Two engines over identically-initialized models: padded vs packed."""
    torch.manual_seed(0)
    model_a = ToyLM()
    torch.manual_seed(0)
    model_b = ToyLM()
    eng_a = Engine(model_parts=[model_a], optimizers=[torch.optim.SGD(model_a.parameters(), lr=lr)])
    eng_b = Engine(model_parts=[model_b], optimizers=[torch.optim.SGD(model_b.parameters(), lr=lr)])
    eng_b.config.pack_datums = True
    return (eng_a, model_a), (eng_b, model_b)


def _parity_datums():
    torch.manual_seed(7)
    datums = _datums()
    # Exercise the alignment-sensitive cases: a masked position and a
    # per-sample advantage next to per-token weights.
    datums[0].loss_fn_inputs["weights"][-1] = 0.0
    datums[0].loss_fn_inputs["advantages"] = torch.tensor([0.3])
    datums[1].loss_fn_inputs["advantages"] = torch.tensor([0.7])
    return datums


def test_pack_datums_loss_parity_with_padded():
    (eng_a, model_a), (eng_b, model_b) = _packing_engines()
    torch.manual_seed(7)
    out_a = eng_a.forward_backward([_parity_datums()])
    torch.manual_seed(7)
    out_b = eng_b.forward_backward([_parity_datums()])
    # Packing is a pure layout change: loss, per-datum logprobs, and the
    # resulting gradients must be identical.
    assert torch.allclose(out_a.loss, out_b.loss)
    assert len(out_a.logprobs) == len(out_b.logprobs)
    for lp_a, lp_b in zip(out_a.logprobs, out_b.logprobs):
        assert torch.allclose(lp_a, lp_b)
    assert torch.allclose(model_a.head.weight.grad, model_b.head.weight.grad)
    assert torch.allclose(model_a.embed.weight.grad, model_b.embed.weight.grad)


def test_pack_datums_multi_microbatch_parity():
    (eng_a, model_a), (eng_b, model_b) = _packing_engines()
    torch.manual_seed(11)
    mbs_a = [_datums(), _datums()]
    torch.manual_seed(11)
    mbs_b = [_datums(), _datums()]
    out_a = eng_a.forward_backward(mbs_a)
    out_b = eng_b.forward_backward(mbs_b)
    assert torch.allclose(out_a.loss, out_b.loss)
    assert torch.allclose(model_a.head.weight.grad, model_b.head.weight.grad)


def test_pack_datums_forward_extraction_parity():
    (eng_a, _), (eng_b, _) = _packing_engines()
    torch.manual_seed(13)
    datums_a = _datums()
    torch.manual_seed(13)
    datums_b = _datums()
    mo_a = eng_a.forward(datums_a)
    mo_b = eng_b.forward(datums_b)
    for lp_a, lp_b in zip(mo_a.logprobs, mo_b.logprobs):
        assert torch.allclose(lp_a, lp_b)
    for en_a, en_b in zip(mo_a.entropy, mo_b.entropy):
        assert torch.allclose(en_a, en_b)


# CP correctness for the Datum door (padded and packed) is covered by the
# multi-GPU parity test in tests/functional_tests/ (padded/packed vs CP=1).


def test_pack_datums_implies_packed_sequence_model_build(monkeypatch):
    captured = {}

    def fake_build_model(model_cfg, peft_cfg, **kwargs):
        captured.update(kwargs)
        return ToyLM()

    # build_model is imported lazily inside the config-build path.
    monkeypatch.setattr("nemo_automodel.recipes.llm.train_ft.build_model", fake_build_model)
    cfg = Engine.Config(model=object(), pack_datums=True)
    try:
        # Inject a stub distributed_setup so the build path never initializes
        # torch.distributed (the session guard flags leaked process groups).
        Engine(config=cfg, distributed_setup=SimpleNamespace())
    except Exception:
        pass  # later build stages may fail; we only assert the model-build wiring
    assert captured.get("has_packed_sequence") is True
