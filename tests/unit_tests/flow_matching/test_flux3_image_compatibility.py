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

"""Exercise the image-only upstream model and its conditioning contract."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_automodel.components.flow_matching.adapters.base import FlowMatchingContext
from nemo_automodel.components.flow_matching.adapters.flux3 import Flux3Adapter
from nemo_automodel.components.models.flux3.text_encoding import encode_text_qwen3vl


@pytest.fixture
def flux3_transformer_cls() -> type[torch.nn.Module]:
    """Load the upstream image model only for tests that require it."""
    diffusers = pytest.importorskip("diffusers")
    # The supported Diffusers 0.39 floor does not ship FLUX3. Remove this
    # guard when the minimum dependency includes Flux3Transformer2DModel.
    # Inspect the lazy export list so failures importing an available model
    # still fail the test instead of being treated as a missing dependency.
    if "Flux3Transformer2DModel" not in dir(diffusers):
        pytest.skip("Requires a Diffusers build exporting Flux3Transformer2DModel.")
    return diffusers.Flux3Transformer2DModel


@pytest.mark.parametrize("with_reference", [False, True])
def test_image_transformer_forward_backward(with_reference: bool, flux3_transformer_cls: type[torch.nn.Module]) -> None:
    torch.manual_seed(7)
    model = flux3_transformer_cls(
        in_channels=4,
        cond_in_channels=4,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=8,
        timestep_channels=8,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
    )
    noisy = torch.randn(2, 4, 2, 3, requires_grad=True)
    text = torch.randn(2, 5, 8, requires_grad=True)
    reference = torch.randn(2, 4, 2, 2, requires_grad=True)
    batch = {"text_embeddings": text}
    if with_reference:
        batch["context_latents"] = [reference]
    context = FlowMatchingContext(
        noisy_latents=noisy,
        latents=torch.zeros_like(noisy),
        timesteps=torch.tensor([250.0, 750.0]),
        sigma=torch.tensor([0.25, 0.75]),
        task_type="i2i" if with_reference else "t2i",
        data_type="image",
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch=batch,
        cfg_dropout_prob=0.0,
    )
    adapter = Flux3Adapter()
    prediction = adapter.forward(model, adapter.prepare_inputs(context))
    assert prediction.shape == noisy.shape
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    parameters = [noisy, text, model.proj_out.weight, model.single_transformer_blocks[0].attn.to_q.weight]
    if with_reference:
        parameters.extend([reference, model.x_embedders["image_cond"].weight])
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("real_length,expected_length", [(1, 80), (17, 80), (80, 80), (81, 160), (8192, 8192)])
def test_text_padding_preserves_training_positions(real_length: int, expected_length: int) -> None:
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "templated caption"

    def tokenize(
        text: str,
        *,
        return_tensors: str,
        truncation: bool,
        max_length: int,
        padding: str | None = None,
        padding_side: str | None = None,
    ) -> dict[str, torch.Tensor]:
        length = max_length if padding == "max_length" else real_length
        mask = (torch.arange(length)[None] < real_length).long()
        return {"input_ids": torch.ones(1, length, dtype=torch.long), "attention_mask": mask}

    tokenizer.tokenizer.side_effect = tokenize
    encoder = MagicMock()
    encoder.dtype = torch.float32
    encoder.device = torch.device("cpu")

    def encode(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool,
        use_cache: bool,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        """Return deterministic layer outputs.

        Args:
            input_ids: Token IDs of shape [1, sequence].
            attention_mask: Real-token mask of shape [1, sequence].
            output_hidden_states: Whether intermediate states were requested.
            use_cache: Whether generation caching was requested.
            position_ids: Optional token positions of shape [1, sequence].

        Returns:
            An output with three hidden states, each of shape [1, sequence, 4].
        """
        length = input_ids.shape[1]
        return SimpleNamespace(hidden_states=tuple(torch.full((1, length, 4), float(i)) for i in range(3)))

    encoder.side_effect = encode
    stack = {"text_encoder": encoder, "tokenizer": tokenizer, "text_encoder_out_layers": (1, 2)}
    result = encode_text_qwen3vl("a fox", stack, "cpu")
    assert result.shape == (expected_length, 8)
    positions = encoder.call_args.kwargs["position_ids"]
    assert torch.equal(positions[0, :real_length], torch.arange(real_length))
    assert torch.equal(positions[0, real_length:], torch.ones(expected_length - real_length, dtype=torch.long))
    assert (result[:, :4] == 1).all()
    assert (result[:, 4:] == 2).all()


def test_t2i_recipe_lora_targets_only_active_streams(flux3_transformer_cls: type[torch.nn.Module]) -> None:
    """The T2I optimizer must not contain never-used image-conditioning adapters."""
    from pathlib import Path

    import yaml

    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules

    recipe_path = Path(__file__).resolve().parents[3] / "examples/diffusion/finetune/flux3_t2i_flow_lora.yaml"
    peft_values = yaml.safe_load(recipe_path.read_text())["peft"]
    peft_values.pop("_target_")
    config = PeftConfig(**peft_values)
    config.lora_dtype = torch.float32
    model = flux3_transformer_cls(
        in_channels=4,
        cond_in_channels=4,
        num_layers=1,
        num_single_layers=1,
        num_attention_heads=2,
        attention_head_dim=8,
        joint_attention_dim=8,
        timestep_channels=8,
        mlp_ratio=2.0,
        axes_dims_rope=(2, 2, 2, 2),
    )
    count = apply_lora_to_linear_modules(model, config)
    assert count == 21
    trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable_names
    assert all("image_cond" not in name for name in trainable_names)
    assert any(name.startswith("mode_blocks.image.0.attn.to_q.") for name in trainable_names)
    assert any(name.startswith("txt_mode_blocks.0.ff.linear_in.") for name in trainable_names)
    assert any(name.startswith("single_transformer_blocks.0.attn.to_q.") for name in trainable_names)
    assert any(name.startswith("proj_out.") for name in trainable_names)
    adapter = Flux3Adapter()
    result = model(
        hidden_states=torch.randn(1, 4, 4),
        img_ids=adapter._prepare_latent_ids(2, 2, torch.device("cpu")),
        encoder_hidden_states=torch.randn(1, 3, 8),
        txt_ids=adapter._prepare_text_ids(3, torch.device("cpu")),
        timestep=torch.tensor([0.5]),
        return_dict=False,
    )[0]
    result.square().mean().backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
