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

"""Tests for freeze_unused_kv_sharing_params (GitHub issue #1687).

Verifies that dead K/V parameters in KV-shared layers are frozen before
optimizer creation so that checkpoint save/resume stays consistent.

Uses a lightweight mock model so these tests run without the full Gemma4
transformers dependency.
"""

import os
import tempfile
from types import SimpleNamespace

import torch
import torch.nn as nn

from nemo_automodel.components.utils.model_utils import freeze_unused_kv_sharing_params


# ---------------------------------------------------------------------------
# Helpers: minimal model that mimics Gemma4 KV-sharing layer naming
# ---------------------------------------------------------------------------
class _FakeAttention(nn.Module):
    """Mimics Gemma4TextAttention with k_proj/v_proj/k_norm/v_norm + q_proj."""

    def __init__(self, dim: int):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.k_norm = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.o_proj = nn.Linear(dim, dim, bias=False)


class _FakeDecoderLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.self_attn = _FakeAttention(dim)
        self.mlp = nn.Linear(dim, dim)


class _FakeModel(nn.Module):
    """A minimal model with ``config.text_config.num_kv_shared_layers``."""

    def __init__(self, num_hidden_layers: int = 6, num_kv_shared_layers: int = 0, dim: int = 16):
        super().__init__()
        text_config = SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            num_kv_shared_layers=num_kv_shared_layers,
        )
        self.config = SimpleNamespace(text_config=text_config)
        self.layers = nn.ModuleList([_FakeDecoderLayer(dim) for _ in range(num_hidden_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer.mlp(layer.self_attn.q_proj(x))
        return x


def _dead_param_names(model, first_shared, num_hidden):
    """Return the set of parameter names that should be frozen."""
    dead = set()
    projs = ("k_proj", "v_proj", "k_norm", "v_norm")
    for name, _ in model.named_parameters():
        for idx in range(first_shared, num_hidden):
            if any(f"layers.{idx}.self_attn.{p}" in name for p in projs):
                dead.add(name)
    return dead


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFreezeUnusedKVSharingParams:
    """Unit tests for freeze_unused_kv_sharing_params."""

    def test_noop_when_no_kv_sharing(self):
        """Models with num_kv_shared_layers=0 are untouched."""
        model = _FakeModel(num_hidden_layers=4, num_kv_shared_layers=0)
        grad_before = {n: p.requires_grad for n, p in model.named_parameters()}

        freeze_unused_kv_sharing_params(model)

        grad_after = {n: p.requires_grad for n, p in model.named_parameters()}
        assert grad_before == grad_after

    def test_noop_when_config_missing(self):
        """Plain nn.Module without a config is silently skipped."""
        model = nn.Linear(4, 4)
        freeze_unused_kv_sharing_params(model)
        assert model.weight.requires_grad is True

    def test_shared_layers_frozen(self):
        """Dead K/V params in shared layers have requires_grad=False."""
        num_hidden, num_shared = 6, 3
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)
        first_shared = num_hidden - num_shared  # layers 3, 4, 5

        freeze_unused_kv_sharing_params(model)

        dead = _dead_param_names(model, first_shared, num_hidden)
        assert len(dead) > 0, "Sanity: there should be dead params"

        for name, param in model.named_parameters():
            if name in dead:
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should remain trainable"

    def test_non_shared_layers_untouched(self):
        """k_proj/v_proj in non-shared layers stay trainable."""
        num_hidden, num_shared = 6, 2
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)

        freeze_unused_kv_sharing_params(model)

        for name, param in model.named_parameters():
            if "layers.0.self_attn.k_proj" in name or "layers.3.self_attn.v_proj" in name:
                assert param.requires_grad, f"Non-shared {name} must stay trainable"

    def test_q_proj_and_o_proj_stay_trainable(self):
        """q_proj and o_proj in shared layers must remain trainable."""
        num_hidden, num_shared = 6, 3
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)

        freeze_unused_kv_sharing_params(model)

        for name, param in model.named_parameters():
            if "q_proj" in name or "o_proj" in name:
                assert param.requires_grad, f"{name} should remain trainable"


class TestCheckpointResumeConsistency:
    """Reproduces the bug from issue #1687 and verifies the fix."""

    def test_bug_reproduction_optimizer_tracks_dead_params(self):
        """WITHOUT fix: optimizer tracks dead params -> save/load key mismatch.

        This test demonstrates the root cause: when dead KV-sharing params
        keep requires_grad=True, the optimizer lazily creates state only for
        params that receive gradients. Params that never participate in the
        forward pass get no optimizer state, creating a mismatch between
        what is saved and what is expected on resume.
        """
        num_hidden, num_shared = 6, 3
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)
        first_shared = num_hidden - num_shared
        dead = _dead_param_names(model, first_shared, num_hidden)

        # Do NOT freeze -> all params go to optimizer
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(all_trainable, lr=1e-3)

        # Forward/backward only uses non-dead params (q_proj + mlp)
        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Optimizer state exists only for params that got gradients
        params_with_state = {id(p) for p in optimizer.state if len(optimizer.state[p]) > 0}
        dead_param_ids = set()
        for name, param in model.named_parameters():
            if name in dead:
                dead_param_ids.add(id(param))

        # Dead params have NO optimizer state (Adam state is lazily initialized)
        dead_with_state = dead_param_ids & params_with_state
        assert len(dead_with_state) == 0, (
            "Dead params should have no optimizer state (they got no gradients), "
            "but on checkpoint resume the loader would expect state for ALL optimizer params."
        )

        # The optimizer param_groups include dead params though
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert dead_param_ids.issubset(optimizer_param_ids), (
            "Without the fix, dead params ARE in the optimizer param groups"
        )

    def test_fix_dead_params_excluded_from_optimizer(self):
        """WITH fix: frozen params excluded from optimizer -> consistent save/load."""
        num_hidden, num_shared = 6, 3
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)
        first_shared = num_hidden - num_shared
        dead = _dead_param_names(model, first_shared, num_hidden)

        # Apply the fix
        freeze_unused_kv_sharing_params(model)

        # Build optimizer the same way as build_optimizer() in train_ft.py
        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

        # Verify dead params are NOT in the optimizer
        optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
        for name, param in model.named_parameters():
            if name in dead:
                assert id(param) not in optimizer_param_ids, f"Frozen param {name} should not be in optimizer"

        # Forward/backward/step works normally
        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # The optimizer has fewer params than the full model (dead ones excluded)
        total_model_params = sum(1 for _ in model.parameters())
        assert len(trainable_params) < total_model_params, (
            "Optimizer should track fewer params than the model (dead ones excluded)"
        )

    def test_checkpoint_save_load_roundtrip(self):
        """Full save/load roundtrip succeeds with the fix applied."""
        num_hidden, num_shared = 6, 3
        model = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)

        # Apply fix
        freeze_unused_kv_sharing_params(model)

        trainable_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(trainable_params, lr=1e-3)

        # Train one step
        x = torch.randn(2, 16, device="cpu")
        loss = model(x).sum()
        loss.backward()
        optimizer.step()

        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "checkpoint.pt")
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_path,
            )

            # Create a fresh model + optimizer (simulating resume)
            model2 = _FakeModel(num_hidden_layers=num_hidden, num_kv_shared_layers=num_shared)
            freeze_unused_kv_sharing_params(model2)
            trainable_params2 = list(filter(lambda p: p.requires_grad, model2.parameters()))
            optimizer2 = torch.optim.Adam(trainable_params2, lr=1e-3)

            # Need to do a dummy step so optimizer state structure exists for load
            x2 = torch.randn(2, 16, device="cpu")
            loss2 = model2(x2).sum()
            loss2.backward()
            optimizer2.step()

            # Load
            ckpt = torch.load(ckpt_path, weights_only=False)
            model2.load_state_dict(ckpt["model"])
            optimizer2.load_state_dict(ckpt["optimizer"])  # This would fail without the fix

        # Verify model weights match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert n1 == n2
            assert torch.allclose(p1.cpu(), p2.cpu()), f"Mismatch in {n1}"


class TestFreezeWithTextConfigDirectly:
    """Test models where text_config is the top-level config (no nesting)."""

    def test_flat_config(self):
        """Model whose config IS the text config (no text_config attr)."""
        model = _FakeModel(num_hidden_layers=4, num_kv_shared_layers=2, dim=8)
        # Replace nested config with flat config
        model.config = SimpleNamespace(
            num_hidden_layers=4,
            num_kv_shared_layers=2,
        )

        freeze_unused_kv_sharing_params(model)

        # Layers 2, 3 should have frozen K/V params
        for name, param in model.named_parameters():
            if any(
                f"layers.{i}.self_attn.{p}" in name for i in (2, 3) for p in ("k_proj", "v_proj", "k_norm", "v_norm")
            ):
                assert not param.requires_grad, f"{name} should be frozen"
            else:
                assert param.requires_grad, f"{name} should be trainable"
