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

"""Stub-based unit tests for :class:`Gemma4WithDrafter`.

These tests exercise ``Gemma4WithDrafter`` against ``torch.nn.Module`` stubs
that mimic the HF ``Gemma4ForConditionalGeneration`` and
``Gemma4AssistantForCausalLM`` API surface. They run *without* the optional
``transformers.models.gemma4_assistant`` package, so they execute in CI on
every PR (the real-HF tests in ``test_composite.py`` are skipped on the
default transformers version).

Surface covered here:
  * ``__init__`` validation (K < 1, share-embedding shape mismatch,
    ``base_activation_checkpointing`` with/without GC method).
  * ``__init__`` side-effects: post_projection freezing for K=1 vs K>1,
    masked_embedding.centroids freezing, freeze_base_for_drafter,
    share_embedding_with_base (copy + lm_head tie).
  * ``forward`` for K=1 and K=2 (teacher-forced shifted ids, recurrent
    feedback through ``prev_last_hidden_state``).
  * ``forward`` error paths (missing input_ids, missing hidden_states,
    missing shared_kv_states).
  * Property pass-throughs (config / vision_tower / audio_tower /
    language_model / get_input/output_embeddings / get_rope_index).
  * ``save_pretrained`` and ``load_pretrained`` happy + error paths.
  * ``from_pretrained`` guards (PP / PEFT / missing paths / bad torch_dtype /
    CP > 1) and the happy path with monkey-patched NeMoAutoModel loaders.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.gemma4_drafter.composite import (
    Gemma4JointOutput,
    Gemma4WithDrafter,
)


# ---------------------------------------------------------------------------
# Tiny stand-in models. The composite only uses a narrow slice of the HF API
# (``get_input_embeddings``, ``config.text_config.hidden_size``, ``forward``
# returning an object with ``logits`` / ``hidden_states`` / ``shared_kv_states``
# for the base and ``logits`` / ``last_hidden_state`` for the drafter). The
# stubs below are the smallest objects that satisfy that contract.
# ---------------------------------------------------------------------------
@dataclass
class _StubBaseOutput:
    logits: torch.Tensor
    hidden_states: Optional[tuple]
    shared_kv_states: Optional[dict]


@dataclass
class _StubDrafterOutput:
    logits: torch.Tensor
    last_hidden_state: torch.Tensor


class _StubTextConfig:
    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class _StubConfigWithText:
    def __init__(self, hidden_size: int):
        self.text_config = _StubTextConfig(hidden_size)


class _StubConfigFlat:
    """Config that exposes ``hidden_size`` directly (no nested text_config)."""

    def __init__(self, hidden_size: int):
        self.hidden_size = hidden_size


class _StubBase(nn.Module):
    """Stub Gemma 4 base. Owns an embed table and returns a deterministic
    ``Gemma4`` forward shape. Records the kwargs it was called with so tests
    can assert that the composite forwards (or pops) the right keys."""

    def __init__(
        self,
        vocab: int = 16,
        hidden: int = 8,
        *,
        include_vision_tower: bool = True,
        include_audio_tower: bool = True,
        include_language_model: bool = True,
        include_rope: bool = True,
        return_hidden_states: bool = True,
        return_shared_kv_states: bool = True,
        config_with_text: bool = True,
        no_config: bool = False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.vocab = vocab
        self.hidden = hidden
        self._return_hidden_states = return_hidden_states
        self._return_shared_kv_states = return_shared_kv_states
        if no_config:
            self.config = None
        elif config_with_text:
            self.config = _StubConfigWithText(hidden)
        else:
            self.config = _StubConfigFlat(hidden)

        if include_vision_tower:
            self.vision_tower = nn.Identity()
        if include_audio_tower:
            self.audio_tower = nn.Identity()
        if include_language_model:
            self.language_model = nn.Identity()
        if include_rope:
            # ``get_rope_index`` is invoked as a method, not a property; HF
            # multimodal models expose it on the conditional-generation class.
            def _rope(*args, **kwargs):
                return torch.tensor([0])

            self.get_rope_index = _rope  # type: ignore[assignment]

        self.last_forward_kwargs: dict = {}
        self.gc_enabled = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def gradient_checkpointing_enable(self):
        self.gc_enabled = True

    def forward(self, **kwargs):
        self.last_forward_kwargs = dict(kwargs)
        ids = kwargs.get("input_ids")
        # ``input_ids=None`` is tolerated so the composite can reach its own
        # post-base ``ValueError("input_ids")`` guard (composite.forward calls
        # ``self.base(input_ids=...)`` *before* that guard runs). We emit a
        # 1x1 dummy shape; the composite raises before any of these tensors
        # are dereferenced.
        if ids is None:
            B, S = 1, 1
            h = torch.zeros((B, S, self.hidden))
        else:
            B, S = ids.shape
            h = self.embed_tokens(ids)
        logits = self.lm_head(h)
        hidden = (h,) if self._return_hidden_states else None
        if self._return_shared_kv_states:
            # Two layer types, each with (K, V) of arbitrary shape; tests only
            # check identity, not values.
            k = torch.zeros((B, 1, S, self.hidden), requires_grad=True)
            v = torch.zeros((B, 1, S, self.hidden), requires_grad=True)
            shared = {"full_attention": (k, v), "sliding_attention": (k, v)}
        else:
            shared = None
        return _StubBaseOutput(logits=logits, hidden_states=hidden, shared_kv_states=shared)


class _StubDrafter(nn.Module):
    """Stub drafter exposing ``pre_projection``, ``post_projection``,
    ``masked_embedding`` (optional), and a forward that returns
    ``last_hidden_state`` at ``backbone_hidden_size`` so multi-step
    recurrence has a well-typed feedback signal."""

    def __init__(
        self,
        vocab: int = 16,
        hidden: int = 4,
        backbone_hidden: int = 8,
        *,
        with_masked_embedding: bool = False,
        with_backbone_attr: bool = True,
        backbone_mismatch: bool = False,
    ):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, hidden)
        # pre_projection consumes [embed | backbone] -> drafter hidden.
        self.pre_projection = nn.Linear(2 * backbone_hidden, hidden, bias=False)
        # post_projection lifts drafter hidden -> backbone hidden (feedback
        # path for multi-step recurrence).
        self.post_projection = nn.Linear(hidden, backbone_hidden, bias=False)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if with_masked_embedding:
            self.masked_embedding = nn.Module()
            self.masked_embedding.centroids = nn.Linear(hidden, 4)
        else:
            self.masked_embedding = None

        if with_backbone_attr:
            # The composite reads ``drafter.config.backbone_hidden_size`` to
            # cross-check shape compatibility. Use the mismatch flag to force
            # the assert.
            cfg_dim = backbone_hidden + (1 if backbone_mismatch else 0)
            self.config = types.SimpleNamespace(backbone_hidden_size=cfg_dim)
        else:
            self.config = types.SimpleNamespace()  # no backbone_hidden_size attr

        self.calls: list[dict] = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, **kwargs):
        self.calls.append(dict(kwargs))
        inputs_embeds = kwargs["inputs_embeds"]
        B, S, _ = inputs_embeds.shape
        h = self.pre_projection(inputs_embeds)
        logits = self.lm_head(h)
        last_hidden = self.post_projection(h)
        return _StubDrafterOutput(logits=logits, last_hidden_state=last_hidden)


# ---------------------------------------------------------------------------
# __init__ paths
# ---------------------------------------------------------------------------
class TestInit:
    def test_rejects_k_less_than_one(self):
        with pytest.raises(ValueError, match="drafter_num_steps"):
            Gemma4WithDrafter(_StubBase(), _StubDrafter(), drafter_num_steps=0)

    def test_k_one_freezes_post_projection(self):
        comp = Gemma4WithDrafter(_StubBase(), _StubDrafter(), drafter_num_steps=1)
        # K=1 path: post_projection output is unused, so we freeze it to keep
        # the optimizer state-dict consistent on resume.
        assert all(not p.requires_grad for p in comp.drafter.post_projection.parameters())

    def test_k_two_leaves_post_projection_trainable(self):
        comp = Gemma4WithDrafter(_StubBase(), _StubDrafter(), drafter_num_steps=2)
        # K>1: post_projection is on the recurrent feedback path; keep it
        # trainable so its gradient lands.
        assert all(p.requires_grad for p in comp.drafter.post_projection.parameters())

    def test_masked_embedding_centroids_always_frozen(self):
        d = _StubDrafter(with_masked_embedding=True)
        Gemma4WithDrafter(_StubBase(), d, drafter_num_steps=2)
        assert all(not p.requires_grad for p in d.masked_embedding.centroids.parameters())

    def test_freeze_base_for_drafter_freezes_all_base_params(self):
        b = _StubBase()
        before = [p.requires_grad for p in b.parameters()]
        assert all(before)
        Gemma4WithDrafter(b, _StubDrafter(), freeze_base_for_drafter=True)
        assert all(not p.requires_grad for p in b.parameters())

    def test_share_embedding_with_base_copies_when_shapes_match(self):
        # Stub base / drafter with matching vocab + hidden so the embed tables
        # are copy-compatible.
        b = _StubBase(vocab=16, hidden=8)
        d = _StubDrafter(vocab=16, hidden=8, backbone_hidden=8)
        # Make the two embed tables initially distinct.
        with torch.no_grad():
            d.embed_tokens.weight.fill_(0.0)
        Gemma4WithDrafter(b, d, share_embedding_with_base=True)
        torch.testing.assert_close(d.embed_tokens.weight, b.embed_tokens.weight)

    def test_share_embedding_with_base_raises_on_shape_mismatch(self):
        b = _StubBase(vocab=16, hidden=8)
        d = _StubDrafter(vocab=32, hidden=8, backbone_hidden=8)
        with pytest.raises(ValueError, match="share_embedding_with_base"):
            Gemma4WithDrafter(b, d, share_embedding_with_base=True)

    def test_base_activation_checkpointing_calls_enable(self):
        b = _StubBase()
        assert b.gc_enabled is False
        Gemma4WithDrafter(b, _StubDrafter(), base_activation_checkpointing=True)
        assert b.gc_enabled is True

    def test_base_activation_checkpointing_raises_when_method_missing(self):
        b = _StubBase()
        # Shadow the class-level method with ``None`` so the composite's
        # ``getattr(..., None)`` returns None and triggers the error path.
        # (Cannot ``del`` a class-level method via an instance.)
        b.gradient_checkpointing_enable = None  # type: ignore[assignment]
        with pytest.raises(RuntimeError, match="gradient_checkpointing_enable"):
            Gemma4WithDrafter(b, _StubDrafter(), base_activation_checkpointing=True)

    def test_backbone_hidden_size_mismatch_asserts(self):
        b = _StubBase(hidden=8)
        d = _StubDrafter(hidden=4, backbone_hidden=8, backbone_mismatch=True)
        with pytest.raises(AssertionError, match="backbone_hidden_size"):
            Gemma4WithDrafter(b, d)

    def test_drafter_missing_backbone_attr_is_tolerated(self):
        b = _StubBase(hidden=8)
        d = _StubDrafter(hidden=4, backbone_hidden=8, with_backbone_attr=False)
        # No assert when the drafter config has no ``backbone_hidden_size``.
        Gemma4WithDrafter(b, d)


# ---------------------------------------------------------------------------
# _get_base_text_config helper
# ---------------------------------------------------------------------------
class TestGetBaseTextConfig:
    def test_returns_text_config_when_nested(self):
        b = _StubBase(hidden=8)
        cfg = Gemma4WithDrafter._get_base_text_config(b)
        assert cfg.hidden_size == 8

    def test_returns_config_when_flat(self):
        b = _StubBase(hidden=8, config_with_text=False)
        cfg = Gemma4WithDrafter._get_base_text_config(b)
        assert cfg.hidden_size == 8

    def test_raises_when_config_is_none(self):
        b = _StubBase(no_config=True)
        with pytest.raises(ValueError, match="config"):
            Gemma4WithDrafter._get_base_text_config(b)


# ---------------------------------------------------------------------------
# forward
# ---------------------------------------------------------------------------
def _make_composite(K: int = 1, hidden: int = 8, vocab: int = 16):
    """Convenience constructor for the forward tests."""
    b = _StubBase(vocab=vocab, hidden=hidden)
    d = _StubDrafter(vocab=vocab, hidden=4, backbone_hidden=hidden)
    return Gemma4WithDrafter(b, d, drafter_num_steps=K), b, d


class TestForwardK1:
    def test_returns_gemma4_joint_output(self):
        comp, b, d = _make_composite(K=1, hidden=8, vocab=16)
        ids = torch.randint(0, 16, (2, 5))
        out = comp(input_ids=ids)
        assert isinstance(out, Gemma4JointOutput)
        assert out.logits.shape == (2, 5, 16)
        assert len(out.drafter_logits) == 1
        assert out.drafter_logits[0].shape == (2, 5, 16)
        # drafter_loss_weight is plumbed from the composite onto the output.
        assert out.drafter_loss_weight == comp.drafter_loss_weight

    def test_inputs_embeds_is_concat_of_embed_and_h_final(self):
        comp, b, d = _make_composite(K=1, hidden=8, vocab=16)
        ids = torch.randint(0, 16, (1, 4))
        comp(input_ids=ids)
        # The drafter sees exactly one call with inputs_embeds.shape[-1] = 2H.
        assert len(d.calls) == 1
        ie = d.calls[0]["inputs_embeds"]
        assert ie.shape == (1, 4, 2 * 8)

    def test_shared_kv_states_forwarded_verbatim(self):
        comp, b, d = _make_composite(K=1)
        ids = torch.randint(0, 16, (1, 3))
        comp(input_ids=ids)
        kv = d.calls[0]["shared_kv_states"]
        assert kv is not None
        assert set(kv.keys()) == {"full_attention", "sliding_attention"}

    def test_pops_labels_and_logits_to_keep_from_kwargs(self):
        """The composite must drop ``labels`` and ``logits_to_keep`` before
        forwarding to the base; otherwise the base computes its own internal
        loss and the recipe's loss math becomes inconsistent."""
        comp, b, _ = _make_composite(K=1)
        ids = torch.randint(0, 16, (1, 3))
        comp(input_ids=ids, labels=ids, logits_to_keep=2)
        assert "labels" not in b.last_forward_kwargs
        assert "logits_to_keep" not in b.last_forward_kwargs

    def test_forwards_attention_and_position_ids(self):
        """Stays a thin pass-through for ``attention_mask`` and ``position_ids``."""
        comp, b, d = _make_composite(K=1)
        ids = torch.randint(0, 16, (1, 3))
        mask = torch.ones_like(ids)
        pos = torch.arange(3).unsqueeze(0)
        comp(input_ids=ids, attention_mask=mask, position_ids=pos)
        assert b.last_forward_kwargs["attention_mask"] is mask
        assert b.last_forward_kwargs["position_ids"] is pos
        assert d.calls[0]["attention_mask"] is mask
        assert d.calls[0]["position_ids"] is pos


class TestForwardK2:
    def test_drafter_called_K_times(self):
        comp, _, d = _make_composite(K=2)
        ids = torch.randint(0, 16, (1, 4))
        out = comp(input_ids=ids)
        assert len(d.calls) == 2
        assert len(out.drafter_logits) == 2

    def test_round1_uses_post_projected_last_hidden_state(self):
        """Round k >= 1 feeds the previous round's ``last_hidden_state`` (the
        post-projected H_b activation) into the backbone slot of ``inputs_embeds``."""
        comp, _, d = _make_composite(K=2, hidden=8)
        ids = torch.randint(0, 16, (1, 4))
        comp(input_ids=ids)
        # Round 0 backbone slice = base.h_final (set during round 0).
        # Round 1 backbone slice = round-0 drafter.last_hidden_state.
        ie0_backbone = d.calls[0]["inputs_embeds"][..., 8:]
        ie1_backbone = d.calls[1]["inputs_embeds"][..., 8:]
        # The two are different objects (the round-1 backbone is a fresh
        # post_projection output), so we just assert they live in the graph.
        assert ie0_backbone.requires_grad
        assert ie1_backbone.requires_grad

    def test_round1_uses_teacher_forced_shifted_embed(self):
        """Round k token slice = embed(input_ids[t + k]); the trailing k
        positions are zero-filled (and the recipe's ``_shift_labels_left``
        already masks those positions with -100)."""
        comp, b, d = _make_composite(K=2, hidden=8, vocab=16)
        ids = torch.randint(1, 16, (1, 4))  # avoid 0s so we can detect the fill
        comp(input_ids=ids)
        # Round 1 backbone = post-projected hidden; token slice for round 1
        # should equal embed(shifted_ids) where shifted_ids = [ids[1], ids[2],
        # ids[3], 0]. Reconstruct the expected embed and compare to the
        # token slice of the round-1 ``inputs_embeds``.
        shifted = torch.zeros_like(ids)
        shifted[..., :3] = ids[..., 1:]
        with torch.no_grad():
            expected_embed = b.embed_tokens(shifted)
        ie1_tokens = d.calls[1]["inputs_embeds"][..., :8].detach()
        torch.testing.assert_close(ie1_tokens, expected_embed)


class TestForwardErrors:
    def test_missing_input_ids_raises(self):
        comp, _, _ = _make_composite(K=1)
        with pytest.raises(ValueError, match="input_ids"):
            comp(input_ids=None)

    def test_missing_hidden_states_raises(self):
        b = _StubBase(return_hidden_states=False)
        comp = Gemma4WithDrafter(b, _StubDrafter())
        with pytest.raises(RuntimeError, match="hidden_states"):
            comp(input_ids=torch.randint(0, 16, (1, 3)))

    def test_missing_shared_kv_states_raises(self):
        b = _StubBase(return_shared_kv_states=False)
        comp = Gemma4WithDrafter(b, _StubDrafter())
        with pytest.raises(RuntimeError, match="shared_kv_states"):
            comp(input_ids=torch.randint(0, 16, (1, 3)))


# ---------------------------------------------------------------------------
# Property pass-throughs
# ---------------------------------------------------------------------------
class TestPassThroughs:
    def test_config(self):
        comp, b, _ = _make_composite()
        assert comp.config is b.config

    def test_get_input_and_output_embeddings(self):
        comp, b, _ = _make_composite()
        assert comp.get_input_embeddings() is b.embed_tokens
        assert comp.get_output_embeddings() is b.lm_head

    def test_vision_audio_language_present(self):
        comp, b, _ = _make_composite()
        assert comp.vision_tower is b.vision_tower
        assert comp.audio_tower is b.audio_tower
        assert comp.language_model is b.language_model

    def test_vision_audio_language_absent_returns_none(self):
        """When the base does not own these attributes the pass-throughs
        must return ``None`` (not raise) so multimodal-aware recipe code
        keeps probing successfully."""
        b = _StubBase(
            include_vision_tower=False,
            include_audio_tower=False,
            include_language_model=False,
        )
        comp = Gemma4WithDrafter(b, _StubDrafter())
        assert comp.vision_tower is None
        assert comp.audio_tower is None
        assert comp.language_model is None

    def test_get_rope_index_dispatches(self):
        comp, _, _ = _make_composite()
        out = comp.get_rope_index(1, foo=2)
        torch.testing.assert_close(out, torch.tensor([0]))

    def test_get_rope_index_missing_raises_attribute_error(self):
        b = _StubBase(include_rope=False)
        comp = Gemma4WithDrafter(b, _StubDrafter())
        with pytest.raises(AttributeError, match="get_rope_index"):
            comp.get_rope_index()


# ---------------------------------------------------------------------------
# save_pretrained / load_pretrained
# ---------------------------------------------------------------------------
class TestSaveLoadPretrained:
    def test_save_requires_checkpointer(self, tmp_path):
        comp, _, _ = _make_composite()
        with pytest.raises(ValueError, match="checkpointer"):
            comp.save_pretrained(str(tmp_path))

    def test_save_writes_two_subdirs(self, tmp_path):
        comp, b, d = _make_composite()
        ckpt = MagicMock()
        comp.save_pretrained(str(tmp_path), checkpointer=ckpt)
        assert ckpt.save_model.call_count == 2
        seen_paths = sorted(call.kwargs["weights_path"] for call in ckpt.save_model.call_args_list)
        assert seen_paths == [os.path.join(str(tmp_path), "base"), os.path.join(str(tmp_path), "drafter")]
        models_passed = {call.kwargs["model"] for call in ckpt.save_model.call_args_list}
        assert b in models_passed and d in models_passed

    def test_save_forwards_tokenizer_and_peft_config(self, tmp_path):
        """``tokenizer`` goes to BOTH sides; ``peft_config`` only to the base
        (the drafter is always trained end-to-end without PEFT in this
        recipe)."""
        comp, _, _ = _make_composite()
        ckpt = MagicMock()
        tok = object()
        peft = {"r": 8}
        comp.save_pretrained(str(tmp_path), checkpointer=ckpt, tokenizer=tok, peft_config=peft)
        calls_by_path = {call.kwargs["weights_path"]: call for call in ckpt.save_model.call_args_list}
        base_call = calls_by_path[os.path.join(str(tmp_path), "base")]
        drafter_call = calls_by_path[os.path.join(str(tmp_path), "drafter")]
        assert base_call.kwargs["tokenizer"] is tok
        assert drafter_call.kwargs["tokenizer"] is tok
        assert base_call.kwargs["peft_config"] == peft
        assert drafter_call.kwargs["peft_config"] is None

    def test_load_requires_checkpointer(self, tmp_path):
        comp, _, _ = _make_composite()
        with pytest.raises(ValueError, match="checkpointer"):
            comp.load_pretrained(str(tmp_path))

    def test_load_dispatches_to_both_subdirs(self, tmp_path):
        comp, b, d = _make_composite()
        base_dir = tmp_path / "base" / "model"
        drafter_dir = tmp_path / "drafter" / "model"
        base_dir.mkdir(parents=True)
        drafter_dir.mkdir(parents=True)
        ckpt = MagicMock()
        comp.load_pretrained(str(tmp_path), checkpointer=ckpt)
        assert ckpt.load_model.call_count == 2
        seen = [(call.args[0], call.args[1]) for call in ckpt.load_model.call_args_list]
        # First call: base; second: drafter (composite preserves order).
        assert seen[0] == (b, str(base_dir))
        assert seen[1] == (d, str(drafter_dir))

    def test_load_missing_base_dir_raises(self, tmp_path):
        comp, _, _ = _make_composite()
        # Drafter dir exists but base/model does not.
        (tmp_path / "drafter" / "model").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="base"):
            comp.load_pretrained(str(tmp_path), checkpointer=MagicMock())

    def test_load_missing_drafter_dir_raises(self, tmp_path):
        comp, _, _ = _make_composite()
        (tmp_path / "base" / "model").mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="drafter"):
            comp.load_pretrained(str(tmp_path), checkpointer=MagicMock())


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------
class TestFromPretrainedGuards:
    """Pure-Python guard branches that don't touch the HF loader at all."""

    def test_pp_config_rejected(self):
        with pytest.raises(ValueError, match="Pipeline parallelism"):
            Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", pipeline_config=object())

    def test_peft_config_rejected(self):
        with pytest.raises(NotImplementedError, match="PEFT"):
            Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", peft_config={"r": 8})

    def test_missing_base_path_raises(self):
        with pytest.raises(ValueError, match="base_path"):
            Gemma4WithDrafter.from_pretrained(drafter_path="y")

    def test_missing_drafter_path_raises(self):
        with pytest.raises(ValueError, match="drafter_path"):
            Gemma4WithDrafter.from_pretrained(base_path="x")

    def test_bad_torch_dtype_rejected(self):
        with pytest.raises(ValueError, match="bfloat16"):
            Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", torch_dtype=torch.float32)
        with pytest.raises(ValueError, match="bfloat16"):
            Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", torch_dtype="float32")

    def test_cp_mesh_size_gt_one_rejected(self):
        """Context parallelism (cp axis > 1) is incompatible with the
        shared_kv_states path; the guard runs before any HF load."""
        mesh = MagicMock()
        mesh.mesh_dim_names = ("cp",)
        cp_axis = MagicMock()
        cp_axis.size.return_value = 2
        mesh.__getitem__.return_value = cp_axis
        with pytest.raises(ValueError, match="Context parallelism"):
            Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", device_mesh=mesh)

    def test_cp_mesh_size_one_accepted(self, monkeypatch):
        """``cp_size == 1`` is fine; the guard short-circuits on the size check."""
        mesh = MagicMock()
        mesh.mesh_dim_names = ("cp",)
        cp_axis = MagicMock()
        cp_axis.size.return_value = 1
        mesh.__getitem__.return_value = cp_axis

        b = _StubBase()
        d = _StubDrafter()
        from nemo_automodel._transformers import auto_model as _auto

        monkeypatch.setattr(
            _auto.NeMoAutoModelForImageTextToText,
            "from_pretrained",
            lambda path, **kw: b,
        )
        monkeypatch.setattr(
            _auto.NeMoAutoModelForCausalLM,
            "from_pretrained",
            lambda path, **kw: d,
        )
        comp = Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", device_mesh=mesh)
        assert isinstance(comp, Gemma4WithDrafter)


class TestFromPretrainedHappyPath:
    """Full ``from_pretrained`` path with the NeMoAuto loaders monkey-patched."""

    def _patch_loaders(self, monkeypatch, b, d, captured):
        from nemo_automodel._transformers import auto_model as _auto

        def _img(path, **kw):
            captured["image_path"] = path
            captured["image_kwargs"] = kw
            return b

        def _causal(path, **kw):
            captured["causal_path"] = path
            captured["causal_kwargs"] = kw
            return d

        monkeypatch.setattr(_auto.NeMoAutoModelForImageTextToText, "from_pretrained", _img)
        monkeypatch.setattr(_auto.NeMoAutoModelForCausalLM, "from_pretrained", _causal)

    def test_pretrained_alias_is_forwarded_as_base_path(self, monkeypatch):
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        Gemma4WithDrafter.from_pretrained(
            pretrained_model_name_or_path="alias-base",
            drafter_path="drafter-x",
        )
        assert captured["image_path"] == "alias-base"
        assert captured["causal_path"] == "drafter-x"

    def test_dtype_string_bfloat16_accepted(self, monkeypatch):
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", torch_dtype="bfloat16")
        # Both sub-loaders receive the dtype string verbatim.
        assert captured["image_kwargs"]["torch_dtype"] == "bfloat16"
        assert captured["causal_kwargs"]["torch_dtype"] == "bfloat16"

    def test_optional_kwargs_thread_to_both_loaders(self, monkeypatch):
        """``attn_implementation``, ``use_liger_kernel``, ``use_sdpa_patching``,
        and ``cache_dir`` are routed to BOTH sub-loaders."""
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        Gemma4WithDrafter.from_pretrained(
            base_path="x",
            drafter_path="y",
            attn_implementation="eager",
            use_liger_kernel=True,
            use_sdpa_patching=False,
            cache_dir="/tmp/hfcache",
        )
        for side in ("image_kwargs", "causal_kwargs"):
            kw = captured[side]
            assert kw["attn_implementation"] == "eager"
            assert kw["use_liger_kernel"] is True
            assert kw["use_sdpa_patching"] is False
            assert kw["cache_dir"] == "/tmp/hfcache"

    def test_text_config_only_threads_to_base(self, monkeypatch):
        """``text_config`` is a base-only override; the drafter takes its own."""
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", text_config={"use_cache": True})
        assert captured["image_kwargs"]["text_config"] == {"use_cache": True}
        assert "text_config" not in captured["causal_kwargs"]

    def test_pipeline_config_forced_none_for_both_sides(self, monkeypatch):
        """Even without an explicit ``pipeline_config``, both loaders must be
        called with ``pipeline_config=None`` so PP is disabled per-side."""
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y")
        assert captured["image_kwargs"]["pipeline_config"] is None
        assert captured["causal_kwargs"]["pipeline_config"] is None

    def test_freeze_config_only_passes_to_base(self, monkeypatch):
        """``freeze_config`` is recipe-level and applies to the base. The
        drafter is always passed ``freeze_config=None`` so it stays
        trainable end-to-end."""
        captured: dict = {}
        self._patch_loaders(monkeypatch, _StubBase(), _StubDrafter(), captured)
        fc = object()
        Gemma4WithDrafter.from_pretrained(base_path="x", drafter_path="y", freeze_config=fc)
        assert captured["image_kwargs"]["freeze_config"] is fc
        assert captured["causal_kwargs"]["freeze_config"] is None
