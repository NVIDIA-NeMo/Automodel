# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""CPU parity tests for the Qwen image-edit flow-matching adapter."""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.models.qwen_image_edit.adapter import QwenImageEditAdapter


def _make_context(
    noisy_latents: torch.Tensor,
    context_latents: list[torch.Tensor],
    text_embeddings: torch.Tensor,
    text_attention_mask: torch.Tensor,
) -> FlowMatchingContext:
    """Build a CPU image-edit flow-matching context.

    Args:
        noisy_latents: Target image latents of shape ``[batch, channels, height,
            width]``.
        context_latents: Context image latents, each of shape ``[batch,
            channels, height, width]``.
        text_embeddings: Prompt embeddings of shape ``[batch, sequence,
            hidden]``.
        text_attention_mask: Prompt mask of shape ``[batch, sequence]``.

    Returns:
        Context whose target and context tensors retain their supplied shapes
        and whose timestep and sigma tensors have shape ``[batch]``.
    """
    batch_size = noisy_latents.shape[0]
    return FlowMatchingContext(
        noisy_latents=noisy_latents,
        latents=torch.randn_like(noisy_latents),
        timesteps=torch.full((batch_size,), 500.0),
        sigma=torch.full((batch_size,), 0.5),
        task_type="image-edit",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch={
            "context_latents": context_latents,
            "text_embeddings": text_embeddings,
            "text_attention_mask": text_attention_mask,
        },
    )


def _tiny_transformer(*, num_layers: int):
    """Build a deterministic tiny upstream Diffusers Qwen transformer."""
    diffusers = pytest.importorskip("diffusers")
    return diffusers.QwenImageTransformer2DModel(
        patch_size=2,
        in_channels=16,
        out_channels=4,
        num_layers=num_layers,
        attention_head_dim=8,
        num_attention_heads=1,
        joint_attention_dim=12,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    )


def test_pack_latents_matches_upstream_diffusers_and_round_trips() -> None:
    """Require exact upstream patch ordering and lossless BCHW restoration."""
    diffusers = pytest.importorskip("diffusers")
    latents = torch.arange(2 * 4 * 6 * 8, dtype=torch.float32).reshape(2, 4, 6, 8)

    actual = QwenImageEditAdapter._pack_latents(latents)
    expected = diffusers.QwenImageEditPlusPipeline._pack_latents(latents, 2, 4, 6, 8)

    torch.testing.assert_close(actual, expected)
    restored = QwenImageEditAdapter._unpack_latents(actual, height=6, width=8)
    torch.testing.assert_close(restored, latents)


def test_prepare_inputs_keeps_target_first_and_context_order() -> None:
    """Check concatenation order, Qwen shape tuples, mask, and timestep scaling."""
    adapter = QwenImageEditAdapter()
    target = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(1, 4, 4, 4)
    first_context = torch.full((1, 4, 4, 6), 11.0)
    second_context = torch.full((1, 4, 6, 4), 22.0)
    text_embeddings = torch.randn(1, 5, 12)
    text_mask = torch.tensor([[1, 1, 1, 0, 0]])

    inputs = adapter.prepare_inputs(_make_context(target, [first_context, second_context], text_embeddings, text_mask))

    packed_parts = [
        adapter._pack_latents(target),
        adapter._pack_latents(first_context),
        adapter._pack_latents(second_context),
    ]
    torch.testing.assert_close(inputs["hidden_states"], torch.cat(packed_parts, dim=1))
    assert inputs["img_shapes"] == [[(1, 2, 2), (1, 2, 3), (1, 3, 2)]]
    torch.testing.assert_close(inputs["encoder_hidden_states"], text_embeddings)
    torch.testing.assert_close(inputs["encoder_hidden_states_mask"], text_mask)
    torch.testing.assert_close(inputs["timestep"], torch.tensor([0.5]))
    assert inputs["_target_token_count"] == 4
    assert inputs["_target_latent_shape"] == (4, 4, 4)


def test_tiny_upstream_transformer_prediction_loss_and_branch_gradients() -> None:
    """Compare slicing to Diffusers and reach all four dual-stream branches."""
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(17)
    model = _tiny_transformer(num_layers=2).eval()
    adapter = QwenImageEditAdapter()
    target = torch.randn(1, 4, 4, 4)
    context = _make_context(
        target,
        [torch.randn(1, 4, 4, 4)],
        torch.randn(1, 5, 12),
        torch.tensor([[1, 1, 1, 1, 0]]),
    )
    inputs = adapter.prepare_inputs(context)

    with torch.no_grad():
        direct_output = model(
            hidden_states=inputs["hidden_states"],
            encoder_hidden_states=inputs["encoder_hidden_states"],
            encoder_hidden_states_mask=inputs["encoder_hidden_states_mask"],
            timestep=inputs["timestep"],
            img_shapes=inputs["img_shapes"],
            guidance=None,
            attention_kwargs=None,
            return_dict=False,
        )[0]
        expected = diffusers.QwenImageEditPlusPipeline._unpack_latents(
            direct_output[:, :4],
            height=32,
            width=32,
            vae_scale_factor=8,
        )[:, :, 0]
        actual = adapter.forward(model, inputs)
    torch.testing.assert_close(actual, expected)

    prediction = adapter.forward(model, inputs)
    loss = adapter.compute_loss(prediction, torch.randn_like(prediction)).mean()
    loss.backward()
    assert torch.isfinite(loss)

    parameter_names = (
        "transformer_blocks.0.attn.to_q.weight",
        "transformer_blocks.0.attn.add_q_proj.weight",
        "transformer_blocks.0.img_mlp.net.0.proj.weight",
        "transformer_blocks.0.txt_mlp.net.0.proj.weight",
    )
    parameters = dict(model.named_parameters())
    for name in parameter_names:
        gradient = parameters[name].grad
        assert gradient is not None, f"missing gradient for {name}"
        assert torch.isfinite(gradient).all(), f"non-finite gradient for {name}"
        assert torch.count_nonzero(gradient), f"zero gradient for {name}"


def test_loss_rejects_prediction_shape_drift() -> None:
    """Reject accidental supervision of context tokens or channel drift."""
    adapter = QwenImageEditAdapter()
    with pytest.raises(ValueError, match="identical shapes"):
        adapter.compute_loss(torch.zeros(1, 4, 4, 4), torch.zeros(1, 4, 4, 6))
