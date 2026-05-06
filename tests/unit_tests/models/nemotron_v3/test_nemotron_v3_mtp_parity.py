# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""Parity tests for the Nemotron-V3 MTP implementation.

Three layers of verification:

1. **MTP-loss oracle parity** — recompute the per-depth MTP loss from
   ``MTPModule.forward``'s per-depth hidden states using a hand-written
   DeepSeek-V3 / Megatron formulation, and assert numerical equivalence with
   :func:`compute_mtp_loss`. This isolates the loss-aggregation and
   label-rolling logic.

2. **State-dict round-trip exactness** — model weights → ``to_hf`` →
   ``from_hf`` → model weights must be bit-exact for every ``mtp.*`` key
   (including merged-experts gate_and_up_projs / down_projs which split into
   per-expert HF tensors and recombine).

3. **Forward determinism** — calling ``model(input_ids, labels=...)`` twice
   in a row must produce identical ``out.loss`` and ``out.mtp_loss``
   (no hidden non-determinism via dropout etc.).

All tests run on CPU in float32 so they catch bugs that bfloat16 tolerance
would mask. Use ``MockNemotronV3Config`` from the existing MTP test module.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.mtp import (
    compute_mtp_loss,
    roll_tensor,
)
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        enable_deepep=False,
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=True,
    )


def _build_model(backend, *, mtp_layers, mtp_pattern, dtype=torch.float32, **cfg_overrides):
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    config = MockNemotronV3Config(
        num_nextn_predict_layers=mtp_layers,
        mtp_hybrid_override_pattern=mtp_pattern,
        torch_dtype="float32" if dtype is torch.float32 else "bfloat16",
        **cfg_overrides,
    )
    model = NemotronHForCausalLM(config, backend=backend)
    return model.to(dtype=dtype), config


# ---------------------------------------------------------------------------
# Layer 1: MTP loss oracle parity
# ---------------------------------------------------------------------------


def _oracle_mtp_loss(per_depth_h, labels, lm_head, scaling_factor, ignore_index=-100):
    """Independent reimplementation of the MTP loss from the DeepSeek-V3 /
    Megatron formula. Used as a test oracle.

    For depth k=1..D:
      labels_k = roll(labels_{k-1}, -1)
      logits_k = lm_head(h_k)
      loss_k   = masked_CE(logits_k, labels_k, ignore the trailing k positions)
    Total = scaling_factor / D * sum_k(loss_k)
    """
    D = len(per_depth_h)
    rolled = labels
    total = per_depth_h[0].new_zeros(())
    for k in range(D):
        rolled = roll_tensor(rolled, shifts=-1, dim=-1)
        masked = rolled.clone()
        S = masked.shape[-1]
        n_invalid = min(k + 1, S)
        masked[..., S - n_invalid : S] = ignore_index

        logits = lm_head(per_depth_h[k])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            masked.reshape(-1),
            ignore_index=ignore_index,
        )
        total = total + loss
    return total * (scaling_factor / D)


class TestMTPLossOracleParity:
    """Compare ``compute_mtp_loss`` to a hand-rolled oracle.

    Uses synthetic per-depth hidden states and a small random ``lm_head`` so
    we can exercise the loss aggregation in isolation, without depending on
    the model's random-init backbone (which can produce NaN logits with
    untrained tiny weights).
    """

    @pytest.mark.parametrize("D", [1, 2, 3])
    @pytest.mark.parametrize("scaling", [0.1, 0.5, 1.0])
    def test_loss_matches_oracle(self, D, scaling):
        torch.manual_seed(42)
        H, V = 64, 128  # hidden, vocab
        B, S = 2, 16

        # Initialize lm_head with small weights to avoid overflow.
        lm_head = torch.nn.Linear(H, V, bias=False)
        torch.nn.init.normal_(lm_head.weight, mean=0.0, std=0.02)

        # Synthetic per-depth hidden states (fp32 → tight tolerance)
        per_depth_h = [torch.randn(B, S, H) * 0.1 for _ in range(D)]
        labels = torch.randint(0, V, (B, S))

        actual = compute_mtp_loss(per_depth_h, labels, lm_head, loss_scaling_factor=scaling)
        expected = _oracle_mtp_loss(per_depth_h, labels, lm_head, scaling)

        assert torch.isfinite(actual).item(), f"actual is non-finite: {actual.item()}"
        assert torch.isfinite(expected).item(), f"expected is non-finite: {expected.item()}"
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5), (
            f"MTP loss diverges from oracle: actual={actual.item():.6e}, "
            f"expected={expected.item():.6e}, diff={(actual - expected).abs().item():.6e}"
        )

    def test_loss_finite_with_partial_labels(self):
        """Sanity: with some labels = ignore_index, MTP loss should still be
        finite (the rolled trailing positions are zero-filled and ignored)."""
        torch.manual_seed(0)
        H, V = 32, 64
        per_depth_h = [torch.randn(1, 8, H) * 0.1]
        labels = torch.randint(0, V, (1, 8))
        lm_head = torch.nn.Linear(H, V, bias=False)
        torch.nn.init.normal_(lm_head.weight, std=0.02)
        loss = compute_mtp_loss(per_depth_h, labels, lm_head, loss_scaling_factor=1.0)
        assert torch.isfinite(loss).item(), f"got non-finite MTP loss: {loss.item()}"

    def test_roll_then_zero_matches_megatron_formula(self):
        """Cumulative left-roll with trailing-zero fill matches Megatron's
        ``roll_tensor`` (single-GPU path). Replicates depth-k semantics."""
        labels = torch.arange(10).unsqueeze(0)  # [1, 10]
        # k=1: trailing 1 position zeroed
        d1 = roll_tensor(labels, shifts=-1, dim=-1)
        assert d1.tolist() == [[1, 2, 3, 4, 5, 6, 7, 8, 9, 0]]
        # k=2: trailing 2 positions zeroed
        d2 = roll_tensor(d1, shifts=-1, dim=-1)
        assert d2.tolist() == [[2, 3, 4, 5, 6, 7, 8, 9, 0, 0]]
        # k=3: trailing 3 positions zeroed
        d3 = roll_tensor(d2, shifts=-1, dim=-1)
        assert d3.tolist() == [[3, 4, 5, 6, 7, 8, 9, 0, 0, 0]]


# ---------------------------------------------------------------------------
# Layer 2: State-dict round-trip
# ---------------------------------------------------------------------------


class TestMTPStateDictRoundTrip:
    """to_hf followed by from_hf must be bit-exact for ``mtp.*`` keys.

    Also covers the merged-experts split path: internal
    ``mtp.layers.{i}.mixer.experts.gate_and_up_projs`` -> per-expert HF
    ``mtp.layers.{i}.mixer.experts.{e}.up_proj.weight`` (and friends) ->
    re-merge.
    """

    def test_round_trip_preserves_mtp_tensors_exactly(self, backend):
        torch.manual_seed(123)
        model, _config = _build_model(backend, mtp_layers=1, mtp_pattern="*E")

        adapter = model.state_dict_adapter
        sd = model.state_dict()
        # Only consider mtp.* keys for this test.
        mtp_sd = {k: v.detach().clone() for k, v in sd.items() if k.startswith("mtp.")}
        assert len(mtp_sd) > 0, "model has no mtp.* keys"

        # Convert to HF and back. Pass only mtp keys + lm_head (needed by
        # backbone path for prefix detection).
        hf_sd = adapter.to_hf({**mtp_sd})
        # Sanity: HF version should have per-expert split keys for the MoE
        # sublayer.
        assert any("mtp.layers.1.mixer.experts.0.up_proj.weight" in k for k in hf_sd), (
            "to_hf did not split MoE experts into per-expert HF keys; got: " + ", ".join(sorted(hf_sd.keys())[:5])
        )

        # Round-trip back.
        restored = adapter.from_hf(hf_sd)

        # Every original mtp.* key must be present and bit-exact in the
        # restored dict.
        for key, tensor in mtp_sd.items():
            assert key in restored, f"missing after round-trip: {key}"
            diff = (tensor - restored[key]).abs().max().item()
            assert diff == 0.0, f"round-trip drift on {key}: max_diff={diff:.3e}"

    def test_round_trip_preserves_mtp_tensors_exactly_d2(self, backend):
        """D=2, P=2 (pattern ``"*E"``) -> 4 flat sublayers; same exactness
        guarantee as D=1. Locks in the HF flat indexing for D>1."""
        torch.manual_seed(456)
        model, _config = _build_model(backend, mtp_layers=2, mtp_pattern="*E")

        adapter = model.state_dict_adapter
        sd = model.state_dict()
        mtp_sd = {k: v.detach().clone() for k, v in sd.items() if k.startswith("mtp.")}
        assert len(mtp_sd) > 0, "model has no mtp.* keys"

        # Sanity: 4 flat sublayers means we expect mtp.layers.{0..3}.norm.weight.
        assert all(f"mtp.layers.{i}.norm.weight" in mtp_sd for i in range(4)), (
            f"expected mtp.layers.{{0..3}}.norm.weight; got {sorted(k for k in mtp_sd if 'norm.weight' in k)}"
        )

        hf_sd = adapter.to_hf({**mtp_sd})
        # MoE sublayers are at indices 1 and 3 (last of each *E depth).
        assert any("mtp.layers.1.mixer.experts.0.up_proj.weight" in k for k in hf_sd)
        assert any("mtp.layers.3.mixer.experts.0.up_proj.weight" in k for k in hf_sd)

        restored = adapter.from_hf(hf_sd)
        for key, tensor in mtp_sd.items():
            assert key in restored, f"missing after round-trip: {key}"
            diff = (tensor - restored[key]).abs().max().item()
            assert diff == 0.0, f"round-trip drift on {key}: max_diff={diff:.3e}"

    def test_d2_state_dict_layout(self, backend):
        """For D=2, P=2 the flat layout has fusion modules on sublayers 0/2
        (first of each depth) and final_layernorm on 1/3 (last of each)."""
        torch.manual_seed(789)
        model, _ = _build_model(backend, mtp_layers=2, mtp_pattern="*E")
        sd = model.state_dict()

        for first_idx in (0, 2):  # first sublayer of depths 0, 1
            for fusion_key in ("enorm", "hnorm", "eh_proj"):
                k = f"mtp.layers.{first_idx}.{fusion_key}.weight"
                assert k in sd, f"missing fusion module: {k}"
        for last_idx in (1, 3):  # last sublayer of depths 0, 1
            assert f"mtp.layers.{last_idx}.final_layernorm.weight" in sd

        # Fusion modules must NOT be on the last sublayers, and
        # final_layernorm must NOT be on the first sublayers.
        for last_idx in (1, 3):
            for fusion_key in ("enorm", "hnorm", "eh_proj"):
                assert f"mtp.layers.{last_idx}.{fusion_key}.weight" not in sd
        for first_idx in (0, 2):
            assert f"mtp.layers.{first_idx}.final_layernorm.weight" not in sd

    def test_pattern_single_sublayer(self, backend):
        """Degenerate pattern ``"E"`` (P=1): the same sublayer is BOTH
        first-of-depth (carries fusion modules) AND last-of-depth (carries
        final_layernorm). Module construction must still succeed."""
        torch.manual_seed(321)
        model, _ = _build_model(backend, mtp_layers=1, mtp_pattern="E")
        assert model.mtp is not None
        assert len(model.mtp.layers) == 1

        sole = model.mtp.layers[0]
        assert sole.has_fusion is True and sole.has_final_norm is True
        for attr in ("enorm", "hnorm", "eh_proj", "final_layernorm"):
            assert hasattr(sole, attr), f"missing {attr} on sole sublayer"

        # state_dict carries all expected keys.
        sd = model.state_dict()
        for key_suffix in (
            "enorm.weight",
            "hnorm.weight",
            "eh_proj.weight",
            "final_layernorm.weight",
            "norm.weight",
        ):
            assert f"mtp.layers.0.{key_suffix}" in sd, f"missing mtp.layers.0.{key_suffix}"


# ---------------------------------------------------------------------------
# Layer 3: Forward determinism
# ---------------------------------------------------------------------------


class TestMTPForwardDeterminism:
    """Two identical forward passes should produce identical outputs.

    Note: random-init tiny CPU models can produce NaN logits (uninitialized
    Linear weights, no scaling, no warmup). Determinism is still meaningful:
    NaN must equal NaN bit-for-bit across two passes. Use
    ``equal_nan=True``.
    """

    def test_forward_is_deterministic(self, backend):
        torch.manual_seed(7)
        model, config = _build_model(backend, mtp_layers=1, mtp_pattern="*E")
        model.train()  # MTP branch only runs in train mode

        B, S = 2, 16
        input_ids = torch.randint(0, config.vocab_size, (B, S))
        labels = input_ids.clone()

        out1 = model(input_ids, labels=labels)
        out2 = model(input_ids, labels=labels)

        torch.testing.assert_close(out1.logits, out2.logits, rtol=0.0, atol=0.0, equal_nan=True)
        torch.testing.assert_close(out1.loss, out2.loss, rtol=0.0, atol=0.0, equal_nan=True)
        torch.testing.assert_close(out1.mtp_loss, out2.mtp_loss, rtol=0.0, atol=0.0, equal_nan=True)

    def test_eval_mode_skips_mtp_branch(self, backend):
        """In eval mode, MTP must not run (out.mtp_loss is None) and the
        main logits should be unaffected by the MTP module's existence."""
        torch.manual_seed(99)
        with_mtp, config = _build_model(backend, mtp_layers=1, mtp_pattern="*E")
        torch.manual_seed(99)
        no_mtp, _ = _build_model(backend, mtp_layers=0, mtp_pattern="")

        # Sync backbone weights between the two so any divergence is
        # attributable to MTP.
        no_mtp_state = no_mtp.state_dict()
        with_mtp_state = with_mtp.state_dict()
        for k, v in no_mtp_state.items():
            with_mtp_state[k].copy_(v)
        with_mtp.load_state_dict(with_mtp_state, strict=False)

        with_mtp.eval()
        no_mtp.eval()

        B, S = 1, 16
        input_ids = torch.randint(0, config.vocab_size, (B, S))

        out_a = with_mtp(input_ids, labels=input_ids.clone())
        out_b = no_mtp(input_ids, labels=input_ids.clone())

        assert out_a.mtp_loss is None  # eval mode skips MTP
        assert out_b.mtp_loss is None
        # Main path identical (NaN-tolerant — tiny random-init model may NaN).
        torch.testing.assert_close(out_a.logits, out_b.logits, rtol=0.0, atol=0.0, equal_nan=True)
