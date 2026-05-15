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

"""Unit tests for ``Gemma4WithDrafter`` against real HF transformers 5.8.

These tests exercise the composite end-to-end against the real HuggingFace
``Gemma4ForConditionalGeneration`` and ``Gemma4AssistantForCausalLM`` on a tiny
config. They cover:

  * forward shapes and the ``Gemma4JointOutput`` contract;
  * the inputs_embeds wiring (``cat(base.embed(input_ids), base.h_final)``);
  * forwarding of ``shared_kv_states`` from base into drafter;
  * gradient flow on a joint loss
    (``L_base + lambda * L_drafter``) reaches both branches and the base
    KV-store layers / embedding via the drafter pathway;
  * ``Gemma4WithDrafter.from_pretrained`` construction guards
    (PP / PEFT / missing paths, ``pretrained_model_name_or_path`` alias);
  * ``save_pretrained`` dispatches into ``base/`` and ``drafter/`` sub-dirs.

The whole module skips when ``transformers.models.gemma4_assistant`` is not
importable (i.e. transformers < 5.8.0.dev). When CUDA is available the tests
run on GPU in bf16, matching the production recipe's dtype; otherwise they run
on CPU in fp32.
"""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.models.gemma4_drafter.composite import (
    Gemma4JointOutput,
    Gemma4WithDrafter,
)


def _gemma4_assistant_available() -> bool:
    try:
        importlib.import_module("transformers.models.gemma4_assistant")
        importlib.import_module("transformers.models.gemma4")
        return True
    except (ModuleNotFoundError, ImportError):
        return False


_HAS_GEMMA4_ASSISTANT = _gemma4_assistant_available()
_SKIP_REASON = (
    "transformers.models.gemma4_assistant not available "
    "(requires transformers>=5.8.0.dev with the gemma4_assistant module)."
)

pytestmark = pytest.mark.skipif(not _HAS_GEMMA4_ASSISTANT, reason=_SKIP_REASON)


# ---------------------------------------------------------------------------
# Tiny HF Gemma4 base + drafter builders
# ---------------------------------------------------------------------------
def _device_and_dtype():
    """Pick the device/dtype used by all real-model tests in this module.

    GPU + bf16 mirrors what the production recipe uses and exercises the
    bf16 autograd path. Falls back to CPU + fp32 when CUDA is absent.
    """
    if torch.cuda.is_available():
        return torch.device(f"cuda:{torch.cuda.current_device()}"), torch.bfloat16
    return torch.device("cpu"), torch.float32


def _dtype_str(dtype: torch.dtype) -> str:
    """Map ``torch.bfloat16`` -> ``"bfloat16"``, etc.; HF Gemma4Config wants this."""
    return str(dtype).rsplit(".", 1)[-1]


def _build_tiny_base(device: torch.device, dtype: torch.dtype):
    """Tiny Gemma4 base with 4 layers, 2 KV-shared (so layers 1-2 are 'store'
    layers for the drafter to consume). Layer-types alternate so both
    ``full_attention`` and ``sliding_attention`` end up in ``shared_kv_states``."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4Config, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    text_cfg = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        intermediate_size=64,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention", "sliding_attention", "full_attention"],
        sliding_window=8,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        torch_dtype=_dtype_str(dtype),
    )
    cfg = Gemma4Config(text_config=text_cfg)
    model = Gemma4ForConditionalGeneration(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model, text_cfg


def _build_tiny_drafter(base_text_cfg, device: torch.device, dtype: torch.dtype):
    """Tiny drafter (all KV-shared, set automatically by Gemma4AssistantConfig
    when ``num_kv_shared_layers`` is 0). ``backbone_hidden_size`` matches the
    base's text hidden size."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
        Gemma4AssistantConfig,
    )
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )

    draft_text = Gemma4TextConfig(
        vocab_size=base_text_cfg.vocab_size,
        hidden_size=24,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=12,
        num_hidden_layers=2,
        intermediate_size=48,
        rms_norm_eps=1e-6,
        max_position_embeddings=64,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window=8,
        hidden_size_per_layer_input=0,
        vocab_size_per_layer_input=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        torch_dtype=_dtype_str(dtype),
    )
    cfg = Gemma4AssistantConfig(
        text_config=draft_text,
        backbone_hidden_size=base_text_cfg.hidden_size,
    )
    model = Gemma4AssistantForCausalLM(cfg).to(device=device, dtype=dtype)
    model.eval()
    return model, draft_text


@pytest.fixture(scope="module")
def device_and_dtype():
    return _device_and_dtype()


@pytest.fixture
def composite(device_and_dtype):
    """Fresh ``Gemma4WithDrafter`` over tiny real HF sub-models, on the
    selected device + dtype. Re-instantiated each test so gradients don't leak."""
    device, dtype = device_and_dtype
    torch.manual_seed(0)
    base, base_text_cfg = _build_tiny_base(device, dtype)
    drafter, draft_text_cfg = _build_tiny_drafter(base_text_cfg, device, dtype)
    composite = Gemma4WithDrafter(
        base,
        drafter,
        drafter_loss_weight=0.5,
        drafter_num_steps=1,
    )
    return composite, base_text_cfg, draft_text_cfg, device, dtype


def _input_ids(base_text_cfg, device: torch.device, batch=1, seq=4):
    return torch.randint(0, base_text_cfg.vocab_size, (batch, seq), device=device)


# ---------------------------------------------------------------------------
# Forward + output dataclass
# ---------------------------------------------------------------------------
class TestCompositeForward:
    def test_output_is_gemma4_joint_output(self, composite):
        comp, base_text_cfg, draft_text_cfg, device, dtype = composite
        ids = _input_ids(base_text_cfg, device, batch=1, seq=4)
        out = comp(input_ids=ids)

        assert isinstance(out, Gemma4JointOutput)
        assert out.logits.shape == (1, 4, base_text_cfg.vocab_size)
        assert out.logits.dtype == dtype
        assert isinstance(out.drafter_logits, list)
        assert len(out.drafter_logits) == 1
        assert out.drafter_logits[0].shape == (1, 4, draft_text_cfg.vocab_size)
        assert out.drafter_logits[0].dtype == dtype
        assert out.drafter_loss_weight == 0.5
        assert out.hidden_states is not None
        assert out.hidden_states[-1].shape == (1, 4, base_text_cfg.hidden_size)

    def test_drafter_pre_projection_input_is_concat_of_embed_and_h_final(self, composite):
        """The composite must concatenate the base's input embeddings (via
        ``get_input_embeddings()``, which already applies the sqrt(H) scale in
        ``Gemma4TextScaledWordEmbedding``) with the base's final hidden state
        along ``dim=-1``."""
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)

        captured = {}

        def hook(_module, args):
            captured["x"] = args[0].detach().clone()

        h = comp.drafter.pre_projection.register_forward_pre_hook(hook)
        try:
            out = comp(input_ids=ids)
        finally:
            h.remove()

        assert "x" in captured
        x = captured["x"]
        H_b = base_text_cfg.hidden_size
        assert x.shape == (1, 4, 2 * H_b)

        with torch.no_grad():
            expected_embed = comp.base.get_input_embeddings()(ids)
        h_final = out.hidden_states[-1].detach()
        torch.testing.assert_close(x[..., :H_b], expected_embed)
        torch.testing.assert_close(x[..., H_b:], h_final)

    def test_drafter_receives_base_shared_kv_states(self, composite):
        """The composite must forward the base's ``shared_kv_states`` dict
        verbatim to the drafter (not detach, not flip keys, not drop a side)."""
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)

        captured = {}
        original_forward = comp.drafter.forward

        def wrapped(**kwargs):
            captured["shared_kv_states"] = kwargs.get("shared_kv_states")
            return original_forward(**kwargs)

        comp.drafter.forward = wrapped  # type: ignore[assignment]
        try:
            comp(input_ids=ids)
        finally:
            comp.drafter.forward = original_forward  # type: ignore[assignment]

        shared_kv = captured["shared_kv_states"]
        assert shared_kv is not None
        # Both layer types are present because the base has both layer kinds.
        assert set(shared_kv.keys()) == {"full_attention", "sliding_attention"}
        for layer_type in ("full_attention", "sliding_attention"):
            k, v = shared_kv[layer_type]
            assert k.requires_grad, f"{layer_type} K must keep requires_grad"
            assert v.requires_grad, f"{layer_type} V must keep requires_grad"

    def test_raises_when_input_ids_missing(self, composite):
        comp, *_ = composite
        with pytest.raises(ValueError, match="input_ids"):
            comp(input_ids=None)

    def test_drafter_num_steps_other_than_one_rejected(self, device_and_dtype):
        device, dtype = device_and_dtype
        torch.manual_seed(0)
        base, base_text_cfg = _build_tiny_base(device, dtype)
        drafter, _ = _build_tiny_drafter(base_text_cfg, device, dtype)
        with pytest.raises(NotImplementedError, match="drafter_num_steps"):
            Gemma4WithDrafter(base, drafter, drafter_num_steps=2)


# ---------------------------------------------------------------------------
# Gradient flow on a joint loss — the most failure-prone path. Both losses
# must reach the parameters the plan promises will be trained.
# ---------------------------------------------------------------------------
class TestCompositeGradientFlow:
    def test_drafter_loss_reaches_drafter_params(self, composite):
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)
        out = comp(input_ids=ids)
        out.drafter_logits[0].float().sum().backward()

        # pre_projection / post_projection are the drafter's signature layers
        # for joint training (they consume base's embed+h_final and project
        # back to backbone-hidden).
        assert comp.drafter.pre_projection.weight.grad is not None
        assert torch.any(comp.drafter.pre_projection.weight.grad != 0)
        assert comp.drafter.post_projection.weight.grad is not None
        assert torch.any(comp.drafter.post_projection.weight.grad != 0)

    def test_drafter_loss_reaches_base_kv_store_layers(self, composite):
        """The drafter loss must propagate through ``shared_kv_states`` back
        into at least one of the base's K/V projection weights for *each*
        ``layer_type`` (the 'store' layer index is picked by HF's KV-share
        logic, so we search across all base k_proj/v_proj weights)."""
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)
        out = comp(input_ids=ids)
        out.drafter_logits[0].float().sum().backward()

        seen_k = False
        seen_v = False
        for name, p in comp.base.named_parameters():
            if p.grad is None or not torch.any(p.grad != 0):
                continue
            if ".k_proj." in name:
                seen_k = True
            if ".v_proj." in name:
                seen_v = True
        assert seen_k, "Drafter loss did not reach any base k_proj weight"
        assert seen_v, "Drafter loss did not reach any base v_proj weight"

    def test_drafter_loss_reaches_base_embed_tokens(self, composite):
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)
        out = comp(input_ids=ids)
        out.drafter_logits[0].float().sum().backward()

        embed_grad = comp.base.get_input_embeddings().weight.grad
        assert embed_grad is not None, "base embed_tokens has no gradient from drafter loss"
        assert torch.any(embed_grad != 0), "base embed_tokens gradient is all zeros from drafter loss"

    def test_base_loss_reaches_base_params(self, composite):
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)
        out = comp(input_ids=ids)
        out.logits.float().sum().backward()

        # Find any base body weight that received a non-zero gradient. We
        # look at q_proj across layers since every layer has one.
        seen = False
        for name, p in comp.base.named_parameters():
            if ".q_proj." in name and p.grad is not None and torch.any(p.grad != 0):
                seen = True
                break
        assert seen, "Base loss did not reach any base q_proj weight"

    def test_joint_loss_routes_to_both_branches(self, composite):
        """A single backward pass on ``L_base + lambda*L_drafter`` must populate
        gradients on both base body params and drafter projection params."""
        comp, base_text_cfg, _, device, _ = composite
        ids = _input_ids(base_text_cfg, device)
        out = comp(input_ids=ids)
        loss = out.logits.float().sum() + out.drafter_loss_weight * out.drafter_logits[0].float().sum()
        loss.backward()

        # Drafter pre_projection is the only place the drafter loss can land
        # without help from the base, so it's a clean drafter-side signal.
        assert comp.drafter.pre_projection.weight.grad is not None
        assert torch.any(comp.drafter.pre_projection.weight.grad != 0)

        # Base side: any q_proj weight grad is enough.
        seen_base = False
        for name, p in comp.base.named_parameters():
            if ".q_proj." in name and p.grad is not None and torch.any(p.grad != 0):
                seen_base = True
                break
        assert seen_base


# ---------------------------------------------------------------------------
# Construction guards — exercise ``Gemma4WithDrafter.from_pretrained`` without
# loading any real weights from disk by monkey-patching the NeMoAuto loaders.
# ---------------------------------------------------------------------------
class TestCompositeGuards:
    def test_pp_config_rejected(self):
        with pytest.raises(ValueError, match="Pipeline parallelism"):
            Gemma4WithDrafter.from_pretrained(
                base_path="ignored",
                drafter_path="ignored",
                pipeline_config=object(),
            )

    def test_peft_config_rejected(self):
        with pytest.raises(NotImplementedError, match="PEFT"):
            Gemma4WithDrafter.from_pretrained(
                base_path="ignored",
                drafter_path="ignored",
                peft_config={"foo": "bar"},
            )

    def test_missing_paths_rejected(self):
        with pytest.raises(ValueError, match="base_path"):
            Gemma4WithDrafter.from_pretrained(drafter_path="x")
        with pytest.raises(ValueError, match="drafter_path"):
            Gemma4WithDrafter.from_pretrained(base_path="x")

    def test_pretrained_alias_forwarded_as_base_path(self, monkeypatch, device_and_dtype):
        """``pretrained_model_name_or_path`` is accepted as an alias for
        ``base_path`` so the recipe's ``_get_model_name`` (which reads
        ``cfg.model.pretrained_model_name_or_path``) keeps finding the
        processor without splitting the joint config into two paths."""
        device, dtype = device_and_dtype
        captured: dict = {}

        torch.manual_seed(0)
        base, base_text_cfg = _build_tiny_base(device, dtype)
        drafter, _ = _build_tiny_drafter(base_text_cfg, device, dtype)

        def _fake_image_load(path, **kwargs):
            captured["base_path"] = path
            return base

        def _fake_causal_load(path, **kwargs):
            captured["drafter_path"] = path
            return drafter

        from nemo_automodel._transformers import auto_model as _auto

        monkeypatch.setattr(_auto.NeMoAutoModelForImageTextToText, "from_pretrained", _fake_image_load)
        monkeypatch.setattr(_auto.NeMoAutoModelForCausalLM, "from_pretrained", _fake_causal_load)

        composite = Gemma4WithDrafter.from_pretrained(
            pretrained_model_name_or_path="alias-path",
            drafter_path="drafter-x",
        )
        assert captured == {"base_path": "alias-path", "drafter_path": "drafter-x"}
        assert isinstance(composite, Gemma4WithDrafter)


# ---------------------------------------------------------------------------
# save_pretrained dispatch
# ---------------------------------------------------------------------------
class TestSavePretrained:
    def test_save_pretrained_writes_two_subdirs(self, composite, tmp_path):
        comp, *_ = composite
        ckpt = MagicMock()
        comp.save_pretrained(str(tmp_path), checkpointer=ckpt)

        # Two save_model calls -- one per sub-module under the right sub-path.
        assert ckpt.save_model.call_count == 2
        seen_paths = sorted(call.kwargs["weights_path"] for call in ckpt.save_model.call_args_list)
        assert seen_paths == [
            os.path.join(str(tmp_path), "base"),
            os.path.join(str(tmp_path), "drafter"),
        ]

        # The base receives ``self.base``; the drafter receives ``self.drafter``.
        models_passed = {call.kwargs["model"] for call in ckpt.save_model.call_args_list}
        assert comp.base in models_passed
        assert comp.drafter in models_passed

    def test_save_pretrained_requires_checkpointer(self, composite, tmp_path):
        comp, *_ = composite
        with pytest.raises(ValueError, match="checkpointer"):
            comp.save_pretrained(str(tmp_path))
