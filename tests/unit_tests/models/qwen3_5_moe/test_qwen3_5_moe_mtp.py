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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("transformers.models.qwen3_5_moe")

from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
from tests.unit_tests.models.passthrough_layers import passthrough_layers


def _tiny_vl_config(**text_kwargs):
    text_config = Qwen3_5MoeTextConfig(
        vocab_size=64,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=16,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"],
        **text_kwargs,
    )
    vision_config = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=16,
        num_position_embeddings=8,
    )
    return Qwen3_5MoeConfig(text_config=text_config.to_dict(), vision_config=vision_config)


def _backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


class TestQwen3_5MoeMTP:
    def test_vlm_declares_mtp_cp_and_prepares_multi_axis_inputs(self):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())
        batch = {
            "input_ids": torch.tensor([[10, 11, 20, 21]]),
            "labels": torch.tensor([[11, 12, 21, 22]]),
            "position_ids": torch.tensor(
                [
                    [[0, 1, 0, 1]],
                    [[10, 11, 20, 21]],
                    [[30, 31, 40, 41]],
                ]
            ),
            "_packed_seq_ids": torch.tensor([[0, 0, 1, 1]]),
        }

        prepared = model.prepare_mtp_inputs_for_cp(batch, ignore_index=-100)

        assert model.ModelCapabilities().supports_mtp_cp
        assert prepared.position_ids_seq_dim == 2
        torch.testing.assert_close(prepared.input_ids[0], torch.tensor([[11, 0, 21, 0]]))
        torch.testing.assert_close(
            prepared.position_ids[0],
            torch.tensor(
                [
                    [[1, 0, 1, 0]],
                    [[11, 0, 21, 0]],
                    [[31, 0, 41, 0]],
                ]
            ),
        )
        torch.testing.assert_close(prepared.targets[0], torch.tensor([[12, -100, 22, -100]]))
        torch.testing.assert_close(prepared.valid_masks[0], torch.tensor([[True, False, True, False]]))

    def test_moe_mtp_builds_full_attention_moe_block(self):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())

        assert model.mtp is not None
        mtp_layer = model.mtp.layers[0]
        assert mtp_layer.layer_type == "full_attention"
        assert hasattr(mtp_layer, "self_attn")
        assert hasattr(mtp_layer, "eh_proj")
        assert hasattr(mtp_layer, "final_layernorm")

        mtp_keys = [key for key in model.state_dict() if key.startswith("mtp.")]
        assert "mtp.layers.0.mlp.gate.weight" in mtp_keys
        assert "mtp.layers.0.mlp.experts.gate_and_up_projs" in mtp_keys
        assert "mtp.layers.0.mlp.shared_experts.gate_proj.weight" in mtp_keys

    def test_moe_mtp_emits_with_media_token_ids(self, monkeypatch):
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        cfg.image_token_id = 1
        cfg.vision_start_token_id = 2
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.train()

        hidden_states = torch.randn(1, 4, cfg.text_config.hidden_size, dtype=model.lm_head.weight.dtype)

        def fake_model_forward(**kwargs):
            assert "pixel_values" in kwargs
            return SimpleNamespace(last_hidden_state=hidden_states)

        def fake_mtp_forward(hidden_states, **kwargs):
            assert "pixel_values" not in kwargs
            assert "image_grid_thw" not in kwargs
            return [hidden_states]

        monkeypatch.setattr(model.model, "forward", fake_model_forward)
        monkeypatch.setattr(model.mtp, "forward", fake_mtp_forward)

        out = model(
            input_ids=torch.tensor([[3, 2, 1, 4]], dtype=torch.long),
            pixel_values=torch.empty(0),
            image_grid_thw=torch.empty((0, 3), dtype=torch.long),
        )

        assert out.logits.shape == (1, 4, cfg.text_config.vocab_size)
        assert out.mtp_per_depth_h is not None
        assert len(out.mtp_per_depth_h) == 1
        assert out.mtp_per_depth_h[0].shape == (1, 4, cfg.text_config.hidden_size)

    @pytest.mark.parametrize("vision_tower_trainable", [True, False], ids=["trainable_tower", "frozen_tower"])
    def test_mtp_training_enters_modules_in_one_order_for_text_and_media(self, vision_tower_trainable):
        """Text-only and media batches call ``embed_tokens`` once and enter the wrapped modules in one order.

        FSDP2 gathers a module's parameters when it is entered and prefetches in backward by forward call
        order, so ranks that enter these modules in different orders issue different collectives.
        """
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        cfg.image_token_id, cfg.video_token_id = 5, 6
        cfg.vision_start_token_id, cfg.vision_end_token_id = 7, 8
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.initialize_weights()
        if not vision_tower_trainable:
            # With a frozen tower only the MTP half of the forward condition sends text-only batches through it.
            model.model.visual.requires_grad_(False)
        model.train()
        language_model = model.model.language_model

        calls: list[str] = []
        watched = {
            "embed_tokens": language_model.embed_tokens,
            "language_model": language_model,
            "lm_head": model.lm_head,
            "mtp": model.mtp,
        }
        handles = [
            module.register_forward_pre_hook(lambda _module, _args, name=name: calls.append(name))
            for name, module in watched.items()
        ]
        try:
            model(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]))
            text_calls = list(calls)
            calls.clear()

            # Four image placeholders for one 2x2-patch image; pixel_values has shape [patches, patch_features].
            model(
                input_ids=torch.tensor([[7, 5, 5, 5, 5, 8, 10, 11]]),
                pixel_values=torch.randn(4, 12, dtype=model.lm_head.weight.dtype),
                image_grid_thw=torch.tensor([[1, 2, 2]]),
            )
            media_calls = list(calls)
        finally:
            for handle in handles:
                handle.remove()

        assert text_calls == ["embed_tokens", "language_model", "lm_head", "mtp"]
        assert media_calls == text_calls

    def test_text_only_mtp_matches_the_token_id_reference(self):
        """Rolling the backbone embeddings gives MTP the inputs that re-embedding rolled token IDs would.

        The last position is excluded: it has no future token, the two forms fill it differently, and the
        MTP loss ignores it.
        """
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(mtp_num_hidden_layers=1), backend=_backend())
        model.initialize_weights()
        model.train()
        language_model = model.model.language_model
        batch, sequence = 2, 8
        input_ids = torch.randint(1, 64, (batch, sequence), generator=torch.Generator().manual_seed(0))

        backbone_output = []
        handle = model.model.register_forward_hook(
            lambda _module, _args, output: backbone_output.append(output.last_hidden_state)
        )
        try:
            out = model(input_ids=input_ids)
        finally:
            handle.remove()

        # Text positions on all three mRoPE axes: the layout the model derives when none is supplied.
        position_ids = torch.arange(sequence).view(1, 1, -1).expand(3, batch, -1)
        reference = model.mtp(
            backbone_output[0],
            input_ids=input_ids,
            embed_fn=language_model.embed_tokens,
            position_ids=position_ids,
            attention_mask=None,
            padding_mask=None,
            rotary_emb=language_model.rotary_emb,
        )

        assert len(out.mtp_per_depth_h) == len(reference) == 1
        assert torch.isfinite(reference[0]).all()
        torch.testing.assert_close(out.mtp_per_depth_h[0][:, :-1], reference[0][:, :-1], rtol=0, atol=0)


def _media_config(mtp_num_hidden_layers):
    cfg = _tiny_vl_config(mtp_num_hidden_layers=mtp_num_hidden_layers)
    cfg.image_token_id, cfg.video_token_id = 5, 6
    cfg.vision_start_token_id, cfg.vision_end_token_id = 7, 8
    return cfg


def _record_module_entries(model):
    """Record the order in which the modules that FSDP wraps separately are entered."""
    language_model = model.model.language_model
    watched = {
        "embed_tokens": language_model.embed_tokens,
        "visual": model.model.visual,
        "language_model": language_model,
        "lm_head": model.lm_head,
    }
    calls: list[str] = []
    handles = [
        module.register_forward_pre_hook(lambda _module, _args, name=name: calls.append(name))
        for name, module in watched.items()
    ]
    return calls, handles


class TestQwen3_5MoeTextOnlyVisionTower:
    @pytest.mark.parametrize(
        ("mtp_num_hidden_layers", "training"), [(0, True), (1, True), (1, False)], ids=["no-mtp", "mtp", "mtp-eval"]
    )
    def test_trainable_tower_is_entered_in_one_order_for_text_and_media(self, mtp_num_hidden_layers, training):
        """With a trainable vision tower, text-only and media batches enter the wrapped modules in one order.

        A trainable tower is its own FSDP unit, so a rank that skipped it would skip its collectives.
        """
        model = Qwen3_5MoeForConditionalGeneration(_media_config(mtp_num_hidden_layers), backend=_backend())
        model.initialize_weights()
        model.train(training)
        calls, handles = _record_module_entries(model)
        try:
            with torch.set_grad_enabled(training):
                model(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]))
                text_calls = list(calls)
                calls.clear()
                # Four image placeholders for one 2x2-patch image; pixel_values has shape [patches, patch_features].
                model(
                    input_ids=torch.tensor([[7, 5, 5, 5, 5, 8, 10, 11]]),
                    pixel_values=torch.randn(4, 12, dtype=model.lm_head.weight.dtype),
                    image_grid_thw=torch.tensor([[1, 2, 2]]),
                    mm_token_type_ids=torch.tensor([[0, 1, 1, 1, 1, 0, 0, 0]]),
                )
                media_calls = list(calls)
        finally:
            for handle in handles:
                handle.remove()

        assert text_calls == ["embed_tokens", "visual", "language_model", "lm_head"]
        assert media_calls == text_calls

    def test_frozen_tower_is_not_entered_for_text(self):
        """A frozen tower stays in the root FSDP unit, so a text-only batch does not run it."""
        model = Qwen3_5MoeForConditionalGeneration(_media_config(0), backend=_backend())
        model.initialize_weights()
        model.model.visual.requires_grad_(False)
        model.train()
        calls, handles = _record_module_entries(model)
        try:
            model(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]))
        finally:
            for handle in handles:
                handle.remove()

        assert calls == ["embed_tokens", "language_model", "lm_head"]

    def test_text_only_step_gives_the_tower_zero_gradients_and_unchanged_logits(self):
        """Entering the trainable tower changes neither the logits nor the language-model gradients."""
        input_ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(0))

        def step(tower_trainable):
            torch.manual_seed(0)
            model = Qwen3_5MoeForConditionalGeneration(_media_config(0), backend=_backend())
            model.initialize_weights()
            model.model.visual.requires_grad_(tower_trainable)
            model.train()
            logits = model(input_ids=input_ids).logits
            logits.float().square().mean().backward()
            return model, logits.detach()

        trainable, trainable_logits = step(True)
        frozen, frozen_logits = step(False)

        assert torch.isfinite(trainable_logits).all()
        torch.testing.assert_close(trainable_logits, frozen_logits, rtol=0, atol=0)
        tower_grads = [param.grad for param in trainable.model.visual.parameters()]
        assert all(grad is not None and not grad.any() for grad in tower_grads)
        frozen_grads = dict(frozen.model.language_model.named_parameters())
        for name, param in trainable.model.language_model.named_parameters():
            torch.testing.assert_close(param.grad, frozen_grads[name].grad, rtol=0, atol=0)

    def test_moe_vlm_packed_forward_keeps_a_batch_axis(self):
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.train()
        tokens = 6
        # Two packed documents of 3 tokens; M-RoPE positions are [axes, batch, tokens].
        positions = torch.arange(3).repeat(2).expand(3, 1, tokens)

        layers = [*model.model.language_model.layers.values(), *model.mtp.layers]

        with passthrough_layers(layers) as layer_inputs:
            out = model(
                input_ids=torch.randint(1, 50, (1, tokens)),
                position_ids=positions,
                qkv_format="thd",
                cu_seqlens=torch.tensor([[0, 3, 6]], dtype=torch.int32),
                max_seqlen=torch.tensor([3]),
            )

        hidden_size = cfg.text_config.hidden_size
        # The decoder and the MTP blocks run on packed states; the MTP blocks embed the packed ids.
        assert layer_inputs == [(tokens, hidden_size)] * len(layers)
        assert out.logits.shape == (1, tokens, cfg.text_config.vocab_size)
        assert [tuple(h.shape) for h in out.mtp_per_depth_h] == [(tokens, hidden_size)]

    @pytest.mark.parametrize("logits_dtype", [None, torch.float32])
    def test_moe_vlm_stage_metas_declare_the_head_logits_dtype(self, logits_dtype):
        model = Qwen3_5MoeForConditionalGeneration(_tiny_vl_config(), backend=_backend(), logits_dtype=logits_dtype)

        _, outputs = model.get_pipeline_stage_metas(is_first=False, microbatch_size=1, seq_len=6, dtype=torch.bfloat16)

        assert [output.dtype for output in outputs] == [logits_dtype or torch.bfloat16]

    @pytest.mark.parametrize("vision_tower_trainable", [True, False], ids=["trainable_tower", "frozen_tower"])
    def test_text_only_splice_enters_vision_tower_only_when_trainable(self, vision_tower_trainable):
        """A text-only ``_embed_and_splice_for_cp`` enters a trainable vision tower once and keeps the embeddings."""
        cfg = _tiny_vl_config()
        cfg.image_token_id, cfg.video_token_id = 5, 6
        cfg.vision_start_token_id, cfg.vision_end_token_id = 7, 8
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.initialize_weights()
        if not vision_tower_trainable:
            model.model.visual.requires_grad_(False)
        model.train()
        input_ids = torch.tensor([[10, 11, 12, 13]])

        visual_entries: list[str] = []
        handle = model.model.visual.register_forward_pre_hook(lambda _module, _args: visual_entries.append("visual"))
        try:
            embeds = model._embed_and_splice_for_cp(
                input_ids,
                pixel_values=None,
                pixel_values_videos=None,
                image_grid_thw=None,
                video_grid_thw=None,
            )
        finally:
            handle.remove()

        assert len(visual_entries) == (1 if vision_tower_trainable else 0)
        expected = model.model.language_model.embed_tokens(input_ids)
        torch.testing.assert_close(embeds, expected)
