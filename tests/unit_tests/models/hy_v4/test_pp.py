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

import copy
import types

import pytest
import torch
from torch import nn


def _make_two_pipeline_stages(model: nn.Module) -> tuple[nn.Module, nn.Module]:
    """Create lightweight first/final HY V4 stage shells for contract tests."""
    first_stage = copy.deepcopy(model)
    final_stage = copy.deepcopy(model)

    first_stage.lm_head = None
    first_stage.model.norm = None
    first_stage.model.hc_head = None
    first_stage.model.mtp_layers = nn.ModuleList()
    final_stage.model.embed_tokens = None
    return first_stage, final_stage


def test_pipeline_stage_module_ownership_pins_hc_and_mtp_to_final_stage(tiny_hy_v4_model):
    original = [
        ["model.embed_tokens", "model.layers.0", "model.layers.1"],
        ["model.layers.2", "model.norm", "lm_head"],
    ]

    stages = tiny_hy_v4_model.customize_pipeline_stage_modules(
        original,
        layers_prefix="model.",
        text_model=tiny_hy_v4_model.model,
    )

    assert stages is not original
    assert stages[0] == original[0]
    assert stages[1] == [*original[1], "model.hc_head", "model.mtp_layers"]
    assert original[1] == ["model.layers.2", "model.norm", "lm_head"]


def test_pipeline_stage_metas_cover_ihc_indexshare_mtp_and_fp32_logits(tiny_hy_v4_model):
    first_stage, final_stage = _make_two_pipeline_stages(tiny_hy_v4_model)

    first_inputs, first_outputs = first_stage.get_pipeline_stage_metas(
        is_first=True,
        microbatch_size=1,
        seq_len=16,
        dtype=torch.bfloat16,
    )
    final_inputs, final_outputs = final_stage.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=1,
        seq_len=16,
        dtype=torch.bfloat16,
    )

    assert [(tuple(t.shape), t.dtype) for t in first_inputs] == [((1, 16), torch.int64)]
    boundary_metas = [
        ((16, 2, 8), torch.bfloat16),
        ((16, 1, 3), torch.float32),
        ((16, 8), torch.bfloat16),
    ]
    assert [(tuple(t.shape), t.dtype) for t in first_outputs] == boundary_metas
    assert [(tuple(t.shape), t.dtype) for t in final_inputs] == boundary_metas
    assert [(tuple(t.shape), t.dtype) for t in final_outputs] == [
        ((1, 16, 32), torch.float32),
        ((1, 16, 8), torch.bfloat16),
        ((1, 16), torch.int32),
        ((1, 16), torch.int64),
    ]

    final_stage._pp_return_hidden_states = True
    _, fused_outputs = final_stage.get_pipeline_stage_metas(
        is_first=False,
        microbatch_size=1,
        seq_len=16,
        dtype=torch.bfloat16,
    )
    assert [(tuple(t.shape), t.dtype) for t in fused_outputs] == [
        ((1, 16, 8), torch.bfloat16),
        ((1, 16, 8), torch.bfloat16),
        ((1, 16), torch.int32),
        ((1, 16), torch.int64),
    ]

    with pytest.raises(ValueError, match="pp_microbatch_size=1"):
        first_stage.get_pipeline_stage_metas(
            is_first=True,
            microbatch_size=2,
            seq_len=16,
            dtype=torch.bfloat16,
        )


def test_pipeline_forward_propagates_indexshare_and_boundary_safe_mtp(monkeypatch, tiny_hy_v4_model):
    first_stage, final_stage = _make_two_pipeline_stages(tiny_hy_v4_model.train())
    captured: dict[str, torch.Tensor] = {}

    def first_backbone_forward(
        backbone: nn.Module,
        input_ids: torch.Tensor,
        *,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return differentiable iHC states and deterministic top-k coordinates.

        Args:
            backbone: First-stage HY V4 backbone.
            input_ids: Packed token IDs with shape ``[tokens]``.
            prev_topk_indices: Optional prior top-k tensor; must be absent on stage zero.
            **kwargs: Unused packed attention metadata.

        Returns:
            iHC states with shape ``[tokens, hc_mult, hidden]`` and int32
            top-k coordinates with shape ``[tokens, 1, index_topk]``.
        """
        del kwargs
        assert prev_topk_indices is None
        embedded = backbone.embed_tokens(input_ids)
        hidden = embedded.unsqueeze(1).expand(-1, backbone.config.hc_mult, -1)
        positions = torch.arange(input_ids.shape[0], device=input_ids.device, dtype=torch.int32)
        topk = positions[:, None, None].expand(-1, 1, backbone.config.index_topk).contiguous()
        return hidden, topk

    def final_backbone_forward(
        backbone: nn.Module,
        hidden_states: torch.Tensor,
        *,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Collapse iHC states while recording the transported IndexShare tensor.

        Args:
            backbone: Final-stage HY V4 backbone.
            hidden_states: Upstream iHC states with shape
                ``[tokens, hc_mult, hidden]``.
            prev_topk_indices: Transported int32 top-k coordinates with shape
                ``[tokens, 1, index_topk]``.
            **kwargs: Unused packed attention metadata.

        Returns:
            Collapsed states with shape ``[tokens, hidden]`` and the unchanged
            int32 top-k coordinates.
        """
        del backbone, kwargs
        assert prev_topk_indices is not None
        captured["topk"] = prev_topk_indices
        return hidden_states.mean(dim=1), prev_topk_indices

    def final_mtp_forward(
        model: nn.Module,
        hidden_states: torch.Tensor,
        *,
        mtp_embed_inputs: tuple[torch.Tensor, ...] | None,
        **kwargs,
    ) -> list[torch.Tensor]:
        """Consume the propagated MTP embedding in a differentiable stand-in.

        Args:
            model: Final-stage HY V4 causal LM.
            hidden_states: Final backbone states with shape ``[tokens, hidden]``.
            mtp_embed_inputs: One future-token embedding tensor with shape
                ``[tokens, hidden]``.
            **kwargs: Unused native MTP inputs and packed metadata.

        Returns:
            One MTP state tensor with shape ``[tokens, hidden]``.
        """
        del model, kwargs
        assert mtp_embed_inputs is not None and len(mtp_embed_inputs) == 1
        captured["mtp_embed"] = mtp_embed_inputs[0]
        return [hidden_states + mtp_embed_inputs[0]]

    monkeypatch.setattr(first_stage.model, "forward", types.MethodType(first_backbone_forward, first_stage.model))
    monkeypatch.setattr(final_stage.model, "forward", types.MethodType(final_backbone_forward, final_stage.model))
    monkeypatch.setattr(final_stage, "_run_mtp", types.MethodType(final_mtp_forward, final_stage))

    input_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
    labels = torch.tensor([[2, 3, -100, 5, 6, -100]])
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2]])
    cu_seqlens = torch.tensor([[0, 3, 6]], dtype=torch.int32)
    mtp_inputs = first_stage.prepare_mtp_inputs_for_cp(
        {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "seq_lens_padded": torch.tensor([[3, 3]]),
        }
    )
    assert mtp_inputs is not None
    stage0_output = first_stage(
        input_ids,
        position_ids=position_ids,
        mtp_per_depth_input_ids=mtp_inputs.input_ids,
        mtp_per_depth_position_ids=mtp_inputs.position_ids,
        mtp_per_depth_targets=mtp_inputs.targets,
        qkv_format="thd",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(stage0_output, tuple) and len(stage0_output) == 3
    stage0_output[1].retain_grad()

    final_output = final_stage(
        *stage0_output,
        position_ids=position_ids,
        mtp_per_depth_input_ids=mtp_inputs.input_ids,
        mtp_per_depth_position_ids=mtp_inputs.position_ids,
        mtp_per_depth_targets=mtp_inputs.targets,
        qkv_format="thd",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(final_output, tuple) and len(final_output) == 4
    logits, mtp_hidden, seq_idx, mtp_targets = final_output

    expected_ids = torch.tensor([2, 3, 0, 5, 6, 0])
    expected_positions = torch.tensor([1, 2, 0, 1, 2, 0])
    expected_embed = first_stage.model.embed_tokens(expected_ids)
    expected_embed = torch.where((expected_positions == 0).unsqueeze(-1), 0, expected_embed)
    torch.testing.assert_close(captured["mtp_embed"], expected_embed)
    torch.testing.assert_close(captured["topk"], stage0_output[1].to(torch.int32))
    torch.testing.assert_close(seq_idx, torch.tensor([[0, 0, 0, 1, 1, 1]], dtype=torch.int32))
    torch.testing.assert_close(mtp_targets, torch.tensor([[3, -100, -100, 6, -100, -100]]))
    assert logits.shape == (1, 6, tiny_hy_v4_model.config.vocab_size)
    assert logits.dtype is torch.float32
    assert mtp_hidden.shape == (1, 6, tiny_hy_v4_model.config.hidden_size)

    (logits.square().mean() + mtp_hidden.square().mean()).backward()
    assert first_stage.model.embed_tokens.weight.grad is not None
    assert torch.isfinite(first_stage.model.embed_tokens.weight.grad).all()
    assert stage0_output[1].grad is not None
    torch.testing.assert_close(stage0_output[1].grad, torch.zeros_like(stage0_output[1]))

    def fail_lm_head(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fused PP loss must not materialize HY V4 logits")

    final_stage._pp_return_hidden_states = True
    monkeypatch.setattr(final_stage.lm_head, "forward", fail_lm_head)
    fused_output = final_stage(
        *stage0_output,
        position_ids=position_ids,
        mtp_per_depth_input_ids=mtp_inputs.input_ids,
        mtp_per_depth_position_ids=mtp_inputs.position_ids,
        mtp_per_depth_targets=mtp_inputs.targets,
        qkv_format="thd",
        cu_seqlens=cu_seqlens,
    )
    assert isinstance(fused_output, tuple) and len(fused_output) == 4
    main_hidden, fused_mtp_hidden, fused_seq_idx, fused_mtp_targets = fused_output
    assert main_hidden.shape == (1, 6, tiny_hy_v4_model.config.hidden_size)
    assert main_hidden.dtype is torch.bfloat16
    assert fused_mtp_hidden.shape == main_hidden.shape
    torch.testing.assert_close(fused_seq_idx, seq_idx)
    torch.testing.assert_close(fused_mtp_targets, mtp_targets)


def test_pipeline_mtp_embedding_uses_global_shift_across_cp_rank_boundary(tiny_hy_v4_model):
    """The last local query embeds a future token owned by the next CP rank."""
    local_ids = torch.tensor([1, 2, 3])
    local_positions = torch.tensor([0, 1, 2])
    precomputed_ids = (torch.tensor([[2, 3, 4]]),)
    precomputed_positions = (torch.tensor([[1, 2, 3]]),)
    attn_kwargs = {"cu_seqlens": torch.tensor([0, 6], dtype=torch.int32)}

    actual = tiny_hy_v4_model._build_mtp_embed_inputs_for_pp(
        local_ids,
        position_ids=local_positions,
        attn_kwargs=attn_kwargs,
        mtp_per_depth_input_ids=precomputed_ids,
        mtp_per_depth_position_ids=precomputed_positions,
    )[0]
    expected = tiny_hy_v4_model.model.embed_tokens(precomputed_ids[0].squeeze(0))
    torch.testing.assert_close(actual, expected)

    local_only = tiny_hy_v4_model._build_mtp_embed_inputs_for_pp(
        local_ids,
        position_ids=local_positions,
        attn_kwargs=attn_kwargs,
    )[0]
    torch.testing.assert_close(local_only[-1], torch.zeros_like(local_only[-1]))
    assert not torch.equal(actual[-1], local_only[-1])


def test_pipeline_mtp_final_stage_uses_precomputed_cp_positions(monkeypatch, tiny_hy_v4_model):
    """The final PP stage must not locally roll positions at a CP boundary."""
    captured = {}

    def fake_prepare(backbone, hidden_states, padding_mask, attn_kwargs):
        del backbone, hidden_states, padding_mask, attn_kwargs
        return {}

    def fake_freqs(position_ids, freqs):
        del freqs
        captured["position_ids"] = position_ids.detach().clone()
        return torch.empty(0)

    def fake_mtp_layer(
        layer,
        hidden_states,
        *,
        embed_input,
        freqs_cis,
        attention_mask,
        padding_mask,
        **kwargs,
    ):
        del layer, freqs_cis, attention_mask, padding_mask, kwargs
        return hidden_states + embed_input

    monkeypatch.setattr(
        tiny_hy_v4_model.model,
        "prepare_packed_dsa_kwargs",
        types.MethodType(fake_prepare, tiny_hy_v4_model.model),
    )
    monkeypatch.setattr("nemo_automodel.components.models.hy_v4.model.freqs_cis_from_position_ids", fake_freqs)
    monkeypatch.setattr(
        tiny_hy_v4_model.model.mtp_layers[0],
        "forward",
        types.MethodType(fake_mtp_layer, tiny_hy_v4_model.model.mtp_layers[0]),
    )

    hidden = torch.zeros(3, tiny_hy_v4_model.config.hidden_size, dtype=torch.bfloat16)
    embed_input = tiny_hy_v4_model.model.embed_tokens(torch.tensor([2, 3, 4]))
    outputs = tiny_hy_v4_model._run_mtp(
        hidden,
        input_ids=None,
        position_ids=torch.tensor([0, 1, 2]),
        attention_mask=None,
        padding_mask=None,
        mtp_embed_inputs=(embed_input,),
        mtp_per_depth_input_ids=None,
        mtp_per_depth_position_ids=(torch.tensor([[1, 2, 3]]),),
        attn_kwargs={"qkv_format": "thd", "cu_seqlens": torch.tensor([0, 6], dtype=torch.int32)},
    )

    torch.testing.assert_close(captured["position_ids"], torch.tensor([1, 2, 3]))
    torch.testing.assert_close(outputs[0], embed_input)
