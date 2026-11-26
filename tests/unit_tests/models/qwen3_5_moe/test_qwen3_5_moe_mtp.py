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

import copy
from collections.abc import Iterator
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import fully_shard

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


@pytest.fixture
def cpu_mesh() -> Iterator[DeviceMesh]:
    """One-rank CPU mesh: ``fully_shard`` on it makes a module a real FSDP unit inside this process."""
    dist.init_process_group("gloo", rank=0, world_size=1, store=dist.HashStore())
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


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

    @pytest.mark.parametrize("tower", ["fsdp_unit_trainable_tower", "plain_trainable_tower", "frozen_tower"])
    def test_mtp_training_enters_modules_in_one_order_for_text_and_media(self, tower, cpu_mesh):
        """Text-only and media batches call ``embed_tokens`` once and enter the wrapped modules in one order.

        FSDP2 gathers a module's parameters when it is entered and prefetches in backward by forward call
        order, so ranks that enter these modules in different orders issue different collectives. A text-only
        batch enters the vision tower only when the tower is a trainable FSDP unit.
        """
        cfg = _tiny_vl_config(mtp_num_hidden_layers=1)
        cfg.image_token_id, cfg.video_token_id = 5, 6
        cfg.vision_start_token_id, cfg.vision_end_token_id = 7, 8
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.initialize_weights()
        if tower == "frozen_tower":
            model.model.visual.requires_grad_(False)
        elif tower == "fsdp_unit_trainable_tower":
            fully_shard(model.model.visual, mesh=cpu_mesh)
        # Unless the tower is a trainable FSDP unit, only the MTP half of the forward condition embeds
        # text-only batches before the language model.
        model.train()
        language_model = model.model.language_model

        visual_entries: list[str] = []
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
        handles.append(
            model.model.visual.register_forward_pre_hook(lambda _module, _args: visual_entries.append("visual"))
        )
        try:
            model(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]]))
            text_calls = list(calls)
            text_visual_entries = len(visual_entries)
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
        assert text_visual_entries == (1 if tower == "fsdp_unit_trainable_tower" else 0)

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
    def test_trainable_tower_is_entered_in_one_order_for_text_and_media(
        self, mtp_num_hidden_layers, training, cpu_mesh
    ):
        """With a trainable vision tower under FSDP, text-only and media batches enter the wrapped modules in one order.

        A trainable tower is its own FSDP unit, so a rank that skipped it would skip its collectives.
        """
        model = Qwen3_5MoeForConditionalGeneration(_media_config(mtp_num_hidden_layers), backend=_backend())
        model.initialize_weights()
        fully_shard(model.model.visual, mesh=cpu_mesh)
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

    @pytest.mark.parametrize("training", [True, False], ids=["train", "eval"])
    def test_trainable_tower_outside_fsdp_is_not_entered_for_text(self, training):
        """Without FSDP nothing is a separate unit, so a text-only batch leaves a trainable tower alone."""
        model = Qwen3_5MoeForConditionalGeneration(_media_config(0), backend=_backend())
        model.initialize_weights()
        model.train(training)
        assert all(param.requires_grad for param in model.model.visual.parameters())
        calls, handles = _record_module_entries(model)
        try:
            with torch.set_grad_enabled(training):
                logits = model(input_ids=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])).logits
                if training:
                    logits.float().square().mean().backward()
        finally:
            for handle in handles:
                handle.remove()

        assert calls == ["embed_tokens", "language_model", "lm_head"]
        assert all(param.grad is None for param in model.model.visual.parameters())

    def test_text_only_step_gives_the_tower_zero_gradients_and_unchanged_logits(self, cpu_mesh):
        """Entering the trainable tower changes neither the logits nor the language-model gradients."""
        input_ids = torch.randint(1, 64, (2, 8), generator=torch.Generator().manual_seed(0))

        def step(tower_trainable):
            torch.manual_seed(0)
            model = Qwen3_5MoeForConditionalGeneration(_media_config(0), backend=_backend())
            model.initialize_weights()
            model.model.visual.requires_grad_(tower_trainable)
            if tower_trainable:
                fully_shard(model.model.visual, mesh=cpu_mesh)
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

    @pytest.mark.parametrize(
        ("fsdp_unit", "vision_tower_trainable"),
        [(True, True), (False, True), (True, False), (False, False)],
        ids=["fsdp_unit_trainable_tower", "plain_trainable_tower", "fsdp_unit_frozen_tower", "plain_frozen_tower"],
    )
    def test_text_only_splice_enters_vision_tower_only_as_a_trainable_fsdp_unit(
        self, fsdp_unit, vision_tower_trainable, cpu_mesh
    ):
        """A text-only ``_embed_and_splice_for_cp`` enters a trainable FSDP-unit tower once and keeps the embeddings."""
        cfg = _tiny_vl_config()
        cfg.image_token_id, cfg.video_token_id = 5, 6
        cfg.vision_start_token_id, cfg.vision_end_token_id = 7, 8
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.initialize_weights()
        if not vision_tower_trainable:
            model.model.visual.requires_grad_(False)
        if fsdp_unit:
            fully_shard(model.model.visual, mesh=cpu_mesh)
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

        assert len(visual_entries) == (1 if fsdp_unit and vision_tower_trainable else 0)
        expected = model.model.language_model.embed_tokens(input_ids)
        torch.testing.assert_close(embeds, expected)


_ON_CPU_AND_CUDA = pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
    ],
)

# Token ids of ``_media_config``: image placeholder, vision start, vision end.
_IMAGE, _START, _END = 5, 7, 8


class TestQwen3_5MoeMediaPositions:
    @_ON_CPU_AND_CUDA
    def test_first_pipeline_stage_decodes_a_media_batch_with_the_later_stages_positions(self, device):
        """Every layer of a media batch split into two pipeline stages gets the same positions.

        The last stage receives hidden states only and builds sequential positions, so the first stage
        (no ``lm_head``) decodes with them as well: the staged logits equal the unsplit model run on
        sequential positions.
        """
        cfg = _media_config(0)
        cfg.text_config.num_hidden_layers = 2
        cfg.text_config.layer_types = ["full_attention", "full_attention"]
        model = Qwen3_5MoeForConditionalGeneration(cfg, backend=_backend())
        model.initialize_weights()
        model.to(device).eval()

        # Split as the pipeline splitter does: copy the model, drop the other stage's layers, and set the
        # modules a stage does not own to None. Layer 0 goes with the embeddings, layer 1 with the head.
        first, last = copy.deepcopy(model), copy.deepcopy(model)
        del first.model.language_model.layers["1"]
        first.model.language_model.norm = None
        first.lm_head = None
        del last.model.language_model.layers["0"]
        last.model.language_model.embed_tokens = None

        # One 2 x 2 image: its M-RoPE positions (1, 1..2, 1..2) are not sequential.
        input_ids = torch.tensor([[_START, _IMAGE, _IMAGE, _IMAGE, _IMAGE, _END, 10, 11]], device=device)
        media = dict(
            pixel_values=torch.randn(4, 12, device=device, dtype=model.lm_head.weight.dtype),
            image_grid_thw=torch.tensor([[1, 2, 2]], device=device),
            mm_token_type_ids=(input_ids == _IMAGE).long(),
            attention_mask=torch.ones_like(input_ids),
        )
        # position_ids has shape [axes, batch, sequence].
        sequential = torch.arange(8, device=device).expand(3, 1, 8)

        layer_positions = []
        layers = [*first.model.language_model.layers.values(), *last.model.language_model.layers.values()]
        handles = [
            layer.register_forward_pre_hook(
                lambda _module, _args, kwargs: layer_positions.append(kwargs["position_ids"]), with_kwargs=True
            )
            for layer in layers
        ]
        try:
            with torch.no_grad():
                stage_hidden = first(input_ids=input_ids, **media).logits
                staged_logits = last(stage_hidden).logits
                unsplit_logits = model(input_ids=input_ids, position_ids=sequential, **media).logits
        finally:
            for handle in handles:
                handle.remove()

        assert len(layer_positions) == 2
        for positions in layer_positions:
            assert positions.tolist() == sequential.tolist()
        assert torch.isfinite(staged_logits).all()
        torch.testing.assert_close(staged_logits, unsplit_logits, rtol=0, atol=0)
