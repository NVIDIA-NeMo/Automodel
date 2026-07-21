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

"""Pipeline-parallel + MTP wiring for NemotronH (nemotron_v3).

Mirrors ``tests/unit_tests/models/deepseek_v4/test_deepseek_v4_mtp.py``'s
``TestPipelineHooks`` and PP-forward cases, adapted for the nemotron-h
hybrid Mamba/Attention/MoE layout with plain ``[B, S, H]`` inter-stage
tensors (no HC stream) and ``mtp_layers_block_type`` list-form MTP pattern.
"""

import types

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig

# Reuse the CPU-friendly Mock config from the existing MTP test module.
from tests.unit_tests.models.nemotron_v3.test_nemotron_v3_mtp import MockNemotronV3Config


@pytest.fixture
def backend():
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        dispatcher="torch",
        fake_balanced_gate=True,
        enable_hf_state_dict_adapter=False,
    )


def _make_model(backend, *, mtp_layers=0, mtp_pattern="", mtp_layers_block_type=None, **cfg_overrides):
    from nemo_automodel.components.models.nemotron_v3.model import NemotronHForCausalLM

    cfg = MockNemotronV3Config(
        num_nextn_predict_layers=mtp_layers,
        mtp_hybrid_override_pattern=mtp_pattern,
        **cfg_overrides,
    )
    if mtp_layers_block_type is not None:
        cfg.mtp_layers_block_type = mtp_layers_block_type
    model = NemotronHForCausalLM(cfg, backend=backend)
    return model.to(torch.bfloat16), cfg


# ---------------------------------------------------------------------------
# Stage-detection + helpers
# ---------------------------------------------------------------------------


class TestIsPipelineParallelStage:
    def test_full_model_is_not_pp_stage(self, backend):
        model, _ = _make_model(backend)
        assert model._is_pipeline_parallel_stage() is False

    def test_missing_lm_head_marks_pp(self, backend):
        model, _ = _make_model(backend)
        model.lm_head = None
        assert model._is_pipeline_parallel_stage() is True

    def test_missing_embed_tokens_marks_pp(self, backend):
        model, _ = _make_model(backend)
        model.model.embed_tokens = None
        assert model._is_pipeline_parallel_stage() is True

    def test_trimmed_layer_count_marks_pp(self, backend):
        model, _ = _make_model(backend)
        # Pop one layer to simulate the splitter trimming.
        keys = list(model.model.layers.keys())
        del model.model.layers[keys[-1]]
        assert model._is_pipeline_parallel_stage() is True


class TestBuildMTPEmbedInputsForPP:
    def test_rolls_input_ids_and_embeds_per_depth(self, backend):
        model, cfg = _make_model(
            backend,
            mtp_layers=2,
            mtp_layers_block_type=["attention", "moe"],
        )
        # Deterministic embedding for assertion.
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                torch.arange(cfg.vocab_size * cfg.hidden_size, dtype=torch.float32).view(
                    cfg.vocab_size, cfg.hidden_size
                )
            )

        input_ids = torch.tensor([[10, 11, 12, 13]])
        out = model._build_mtp_embed_inputs_for_pp(input_ids)
        assert len(out) == 2
        expected_d0_ids = torch.tensor([[11, 12, 13, 0]])
        expected_d1_ids = torch.tensor([[12, 13, 0, 0]])
        torch.testing.assert_close(out[0], model.model.embed_tokens(expected_d0_ids))
        torch.testing.assert_close(out[1], model.model.embed_tokens(expected_d1_ids))


# ---------------------------------------------------------------------------
# customize_pipeline_stage_modules
# ---------------------------------------------------------------------------


class TestCustomizePipelineStageModules:
    def test_appends_mtp_to_last_stage_only(self, backend):
        model, _ = _make_model(
            backend,
            mtp_layers=1,
            mtp_layers_block_type=["attention", "moe"],
        )
        stages = [
            ["model.embed_tokens", "model.layers.0", "model.layers.1"],
            ["model.layers.2", "model.layers.3", "model.norm", "lm_head"],
        ]
        out = model.customize_pipeline_stage_modules(stages, layers_prefix="model.", text_model=model.model)
        assert "mtp" not in out[0]
        assert "mtp" in out[-1]

    def test_no_mtp_when_disabled(self, backend):
        model, _ = _make_model(backend)
        stages = [["model.embed_tokens", "model.layers.0"], ["model.norm", "lm_head"]]
        out = model.customize_pipeline_stage_modules(stages, layers_prefix="model.", text_model=model.model)
        assert all("mtp" not in s for s in out)


# ---------------------------------------------------------------------------
# get_pipeline_stage_metas
# ---------------------------------------------------------------------------


class TestPipelineStageMetas:
    def test_first_middle_final_arity_and_shapes_with_mtp(self, backend):
        D = 2
        first, cfg = _make_model(
            backend,
            mtp_layers=D,
            mtp_layers_block_type=["attention", "moe"],
        )
        first.lm_head = None
        first.model.norm = None
        first.mtp = None
        f_in, f_out = first.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=16, dtype=torch.bfloat16)
        assert f_in[0].shape == (2, 16) and f_in[0].dtype == torch.long
        assert len(f_out) == 1 + D
        assert f_out[0].shape == (2, 16, cfg.hidden_size)
        for h in f_out[1:]:
            assert h.shape == (2, 16, cfg.hidden_size)

        middle, _ = _make_model(
            backend,
            mtp_layers=D,
            mtp_layers_block_type=["attention", "moe"],
        )
        middle.model.embed_tokens = None
        middle.lm_head = None
        middle.model.norm = None
        middle.mtp = None
        m_in, m_out = middle.get_pipeline_stage_metas(
            is_first=False, microbatch_size=2, seq_len=16, dtype=torch.bfloat16
        )
        assert len(m_in) == 1 + D and len(m_out) == 1 + D
        assert m_in[0].shape == (2, 16, cfg.hidden_size)

        final, _ = _make_model(
            backend,
            mtp_layers=D,
            mtp_layers_block_type=["attention", "moe"],
        )
        final.model.embed_tokens = None
        l_in, l_out = final.get_pipeline_stage_metas(
            is_first=False, microbatch_size=2, seq_len=16, dtype=torch.bfloat16
        )
        assert len(l_in) == 1 + D
        # Final stage appends an int32 [B, S] seq_idx tail: (logits, *mtp_h, seq_idx).
        assert len(l_out) == 1 + D + 1
        assert l_out[0].shape == (2, 16, cfg.vocab_size)
        for h in l_out[1 : 1 + D]:
            assert h.shape == (2, 16, cfg.hidden_size)
        assert l_out[-1].shape == (2, 16) and l_out[-1].dtype == torch.int32

    def test_no_mtp_arity_is_one(self, backend):
        model, cfg = _make_model(backend)
        f_in, f_out = model.get_pipeline_stage_metas(is_first=True, microbatch_size=1, seq_len=8, dtype=torch.bfloat16)
        assert len(f_in) == 1 and f_in[0].dtype == torch.long
        assert len(f_out) == 1


# ---------------------------------------------------------------------------
# PP forward variants (stubbed backbone)
# ---------------------------------------------------------------------------


class TestPPForward:
    def test_first_stage_propagates_shifted_mtp_embeddings(self, backend):
        model, cfg = _make_model(
            backend,
            mtp_layers=1,
            mtp_layers_block_type=["attention", "moe"],
        )
        model.train()
        # Simulate a first-stage trim: lm_head + norm + mtp absent, embed_tokens kept.
        model.lm_head = None
        model.model.norm = None
        model.mtp = None
        with torch.no_grad():
            model.model.embed_tokens.weight.copy_(
                torch.arange(cfg.vocab_size * cfg.hidden_size, dtype=torch.float32).view(
                    cfg.vocab_size, cfg.hidden_size
                )
            )

        def fake_inner(self, input_ids, **kwargs):
            del self, kwargs
            return torch.ones(input_ids.shape[0], input_ids.shape[1], cfg.hidden_size, dtype=torch.bfloat16)

        model.model.forward = types.MethodType(fake_inner, model.model)

        input_ids = torch.tensor([[10, 11, 12, 13]])
        out = model(input_ids)

        assert isinstance(out, tuple)
        assert len(out) == 2  # 1 + D
        assert out[0].shape == (1, 4, cfg.hidden_size)
        expected_ids = torch.tensor([[11, 12, 13, 0]])
        torch.testing.assert_close(out[1], model.model.embed_tokens(expected_ids))

    def test_middle_stage_passes_through_mtp_embeds(self, backend):
        model, cfg = _make_model(
            backend,
            mtp_layers=1,
            mtp_layers_block_type=["attention", "moe"],
        )
        model.train()
        # Simulate a middle-stage trim: nothing owned except some backbone layers.
        model.lm_head = None
        model.model.embed_tokens = None
        model.model.norm = None
        model.mtp = None

        captured = {}

        def fake_inner(self, input_ids, **kwargs):
            captured["received_as_input_ids"] = input_ids
            return input_ids  # passthrough; same shape

        model.model.forward = types.MethodType(fake_inner, model.model)

        activation = torch.zeros(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        mtp_embed = torch.randn(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        out = model(activation, mtp_embed)

        assert isinstance(out, tuple)
        assert len(out) == 2
        assert out[0].shape == (1, 4, cfg.hidden_size)
        # mtp_embed flows through unchanged
        torch.testing.assert_close(out[1], mtp_embed)
        # Backbone received the inter-stage tensor through the input_ids slot.
        assert captured["received_as_input_ids"] is activation

    def test_final_stage_uses_propagated_mtp_embeddings(self, backend):
        model, cfg = _make_model(
            backend,
            mtp_layers=1,
            mtp_layers_block_type=["attention", "moe"],
        )
        model.train()
        # Simulate a final-stage trim: embed_tokens absent, lm_head + mtp owned.
        model.model.embed_tokens = None

        captured = {}

        def fake_inner(self, input_ids, **kwargs):
            del self, kwargs
            return torch.ones(input_ids.shape[0], input_ids.shape[1], cfg.hidden_size, dtype=torch.bfloat16)

        def fake_mtp_forward(self, **kwargs):
            captured.update(kwargs)
            return [kwargs["hidden_states"].clone()]

        model.model.forward = types.MethodType(fake_inner, model.model)
        model.mtp.forward = types.MethodType(fake_mtp_forward, model.mtp)

        activation = torch.zeros(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        mtp_embed = torch.randn(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        out = model(activation, mtp_embed)

        assert isinstance(out, tuple)
        assert len(out) == 3  # (logits, mtp_per_depth_h[0], seq_idx)
        assert out[0].shape == (1, 4, cfg.vocab_size)
        assert out[-1].shape == (1, 4) and out[-1].dtype == torch.int32
        # The MTP head was given the upstream embedding via embed_inputs and
        # NOT via the input_ids/embed_fn path.
        assert "embed_inputs" in captured
        torch.testing.assert_close(captured["embed_inputs"][0], mtp_embed)
        assert captured.get("input_ids") is None
        assert captured.get("embed_fn") is None

    def test_final_stage_eval_emits_placeholders(self, backend):
        """In eval mode, the last stage keeps the (1 + D + seq_idx) tuple arity."""
        D = 1
        model, cfg = _make_model(
            backend,
            mtp_layers=D,
            mtp_layers_block_type=["attention", "moe"],
        )
        model.eval()
        model.model.embed_tokens = None

        def fake_inner(self, input_ids, **kwargs):
            del self, kwargs
            return torch.ones(input_ids.shape[0], input_ids.shape[1], cfg.hidden_size, dtype=torch.bfloat16)

        model.model.forward = types.MethodType(fake_inner, model.model)

        activation = torch.zeros(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        mtp_embed = torch.randn(1, 4, cfg.hidden_size, dtype=torch.bfloat16)
        with torch.no_grad():
            out = model(activation, mtp_embed)

        assert isinstance(out, tuple)
        assert len(out) == 1 + D + 1
        # Logits live on out[0]; placeholders match the activation's hidden shape;
        # the int32 [B, S] seq_idx tail is last.
        assert out[0].shape == (1, 4, cfg.vocab_size)
        for ph in out[1 : 1 + D]:
            assert ph.shape == (1, 4, cfg.hidden_size)
        assert out[-1].shape == (1, 4) and out[-1].dtype == torch.int32


# ---------------------------------------------------------------------------
# Initialize_weights on a trimmed stage
# ---------------------------------------------------------------------------


class TestInitializeWeightsOnTrimmedStage:
    def test_middle_stage_init_no_attribute_error(self, backend):
        """Stage with embed_tokens=None, norm=None, lm_head=None, mtp=None must init cleanly."""
        model, _ = _make_model(backend, mtp_layers=1, mtp_layers_block_type=["attention", "moe"])
        model.lm_head = None
        model.model.embed_tokens = None
        model.model.norm = None
        model.mtp = None
        # Should not raise AttributeError on any of the trimmed attrs.
        model.initialize_weights(buffer_device=torch.device("cpu"))

    def test_first_stage_init_no_attribute_error(self, backend):
        model, _ = _make_model(backend, mtp_layers=1, mtp_layers_block_type=["attention", "moe"])
        model.lm_head = None
        model.model.norm = None
        model.mtp = None
        model.initialize_weights(buffer_device=torch.device("cpu"))


# ---------------------------------------------------------------------------
# MoE FSDP iterator on a trimmed stage
# ---------------------------------------------------------------------------


class TestMoEIterOnTrimmedStage:
    def test_iter_skips_absent_mtp(self, backend):
        from nemo_automodel.components.moe.parallelizer import _iter_transformer_and_mtp_blocks

        model, _ = _make_model(backend, mtp_layers=1, mtp_layers_block_type=["attention", "moe"])
        # Mimic a middle stage that holds backbone layers but no mtp.
        model.lm_head = None
        model.model.embed_tokens = None
        model.model.norm = None
        model.mtp = None

        yielded = list(_iter_transformer_and_mtp_blocks(model))
        # Only backbone layers should be iterated; no MTP-side blocks.
        assert len(yielded) == len(model.model.layers)
        for parent_layers, layer_id, _block in yielded:
            assert parent_layers is model.model.layers
            assert layer_id in model.model.layers


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-vv"]))
