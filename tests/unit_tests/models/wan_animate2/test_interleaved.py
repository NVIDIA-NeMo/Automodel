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

"""Numerical tests against the released Diffusers two-pass implementation."""

from copy import deepcopy

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.flow_matching.pipeline import FlowMatchingPipeline
from nemo_automodel.components.models.wan_animate2.adapter import WanAnimate2Adapter
from nemo_automodel.components.models.wan_animate2.interleaved import install_forward_origin


def _upstream_forward(model, inputs):
    """Run unmodified Diffusers with gradient through the reference stream.

    Args:
        model: The reference upstream transformer.
        inputs: Adapter tensors with layouts documented by prepare_inputs.

    Returns:
        Predictions [batch, 16, target_frames + 1, height, width], including the reference slot.
    """
    from diffusers.models.transformers.transformer_wan_animate_2 import WanAnimate2KVCache

    cache = WanAnimate2KVCache(len(model.blocks))
    model(
        hidden_states=inputs["x_ref"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["context_ref"],
        condition_latents=inputs["condition_y"],
        encoder_hidden_states_image=inputs["clip_fea_ref"],
        offset_grid_sizes=inputs["grid_sizes_ref"],
        seq_len=inputs["seq_len_ref"],
        kv_cache=cache,
        kv_cache_mode="extract",
    )
    result = model(
        hidden_states=inputs["x"],
        timestep=inputs["timestep"],
        encoder_hidden_states=inputs["context"],
        condition_latents=inputs["y"],
        encoder_hidden_states_image=inputs["clip_fea"],
        reference_grid_sizes=inputs["grid_sizes_ref"],
        seq_len=inputs["seq_len"],
        kv_cache=cache,
        kv_cache_mode="cached",
        origin_len=inputs["origin_len"],
        origin_area=inputs["origin_area"],
    ).sample
    return torch.stack(result)


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.runtime_budget(30, reason="Real upstream flex-attention forward/backward includes cold CPU compilation")
def test_forward_gradients_and_update_match_upstream(tiny_model, training_inputs, checkpointed, lora):
    """Both trained streams, checkpoint replay, and LoRA match real Diffusers."""
    if lora:
        apply_lora_to_linear_modules(
            tiny_model,
            PeftConfig(
                dim=4,
                alpha=4,
                dropout=0.0,
                lora_dtype=torch.float32,
                use_memory_efficient_lora=False,
                target_modules=["*.self_attn.to_q", "*.self_attn.to_k", "*.self_attn.to_v"],
            ),
        )
    reference = deepcopy(tiny_model)
    expected_inputs = deepcopy(training_inputs)
    training_inputs["x_ref"][0].requires_grad_()
    expected_inputs["x_ref"][0].requires_grad_()
    install_forward_origin(tiny_model)
    if checkpointed:
        for i, block in enumerate(tiny_model.blocks):
            tiny_model.blocks[i] = checkpoint_wrapper(block, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    actual = WanAnimate2Adapter().forward(tiny_model, training_inputs)
    expected = _upstream_forward(reference, expected_inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    gradient = torch.randn_like(actual)
    actual.backward(gradient)
    expected.backward(gradient)
    driving_grad = training_inputs["x_ref"][0].grad
    assert driving_grad is not None and driving_grad.abs().max() > 0
    torch.testing.assert_close(driving_grad, expected_inputs["x_ref"][0].grad, rtol=2e-5, atol=2e-6)
    actual_params = dict(tiny_model.named_parameters())
    expected_params = dict(reference.named_parameters())
    # Checkpoint wrappers preserve state-dict keys; named_parameters exposes
    # their wrapper path, so compare through the wrapper's state-dict traversal.
    actual_params = {k.replace("._checkpoint_wrapped_module", ""): v for k, v in actual_params.items()}
    assert actual_params.keys() == expected_params.keys()
    for name, parameter in actual_params.items():
        expected_grad = expected_params[name].grad
        assert (parameter.grad is None) == (expected_grad is None), name
        if expected_grad is not None:
            torch.testing.assert_close(parameter.grad, expected_grad, rtol=3e-5, atol=3e-6, msg=name)
    for model in (tiny_model, reference):
        torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False).step()
    for name, value in tiny_model.state_dict().items():
        torch.testing.assert_close(value, reference.state_dict()[name], rtol=1e-5, atol=2e-6, msg=name)


def test_installation_is_instance_local_and_preserves_checkpoint(tiny_model, training_inputs, tmp_path):
    """Training setup leaves other models, names, and Diffusers reload intact."""
    reference = deepcopy(tiny_model)
    keys = set(tiny_model.state_dict())
    original_forward = type(tiny_model).forward
    block_forward = type(tiny_model.blocks[0]).forward
    install_forward_origin(tiny_model)
    install_forward_origin(tiny_model)
    assert type(tiny_model).forward is original_forward
    assert type(tiny_model.blocks[0]).forward is block_forward
    assert reference.forward.__func__ is original_forward
    assert set(tiny_model.state_dict()) == keys
    tiny_model.save_pretrained(tmp_path)
    restored = type(reference).from_pretrained(tmp_path)
    for name, tensor in tiny_model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[name], rtol=0, atol=0)
    expected = _upstream_forward(restored, training_inputs)
    actual = WanAnimate2Adapter().forward(tiny_model, training_inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_repeated_steps_do_not_reuse_reference_cache(tiny_model, training_inputs):
    """A new driving video changes the prediction and matches a fresh reference."""
    reference = deepcopy(tiny_model)
    adapter = WanAnimate2Adapter()
    first = adapter.forward(tiny_model, training_inputs)
    second_inputs = deepcopy(training_inputs)
    second_inputs["x_ref"][0] = torch.randn_like(second_inputs["x_ref"][0])
    second = adapter.forward(tiny_model, second_inputs)
    expected = _upstream_forward(reference, second_inputs)
    assert not torch.allclose(first, second)
    torch.testing.assert_close(second, expected, rtol=1e-5, atol=1e-6)


def test_unsupported_model_reports_required_diffusers_class():
    with pytest.raises(TypeError, match="diffusers>=0.40.0"):
        install_forward_origin(torch.nn.Linear(2, 2))


@pytest.mark.runtime_budget(30, reason="Real upstream flex-attention model exercises complete recipe loss")
def test_recipe_loss_supervises_reference_slot(tiny_model, training_batch, monkeypatch):
    """A nonzero reference frame contributes loss even when all video targets are zero."""
    training_batch["video_latents"].zero_()
    training_batch["reference_latents"].fill_(3.0)
    with torch.no_grad():
        tiny_model.head.head.weight.zero_()
        tiny_model.head.head.bias.zero_()
    pipeline = FlowMatchingPipeline(
        model_adapter=WanAnimate2Adapter(),
        timestep_sampling="uniform_discrete",
        loss_weighting_scheme="bsmntw_shifted",
        flow_shift=5.0,
        i2v_prob=0.0,
        cfg_dropout_prob=0.0,
        device=torch.device("cpu"),
    )
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor([500]))
    monkeypatch.setattr(torch, "randn_like", torch.zeros_like)
    per_element, loss, _, _ = pipeline.step(
        tiny_model, training_batch, torch.device("cpu"), torch.float32, collect_metrics=False, check_loss=False
    )
    assert per_element.shape == (1, 16, 3, 4, 4)
    # One reference frame with velocity -3 and two zero target frames. The
    # frozen weight is from DiffSynth c458cb42's scheduler at grid index 500.
    torch.testing.assert_close(loss, torch.tensor(3.0 * 1.0248619318), rtol=1e-6, atol=1e-6)
    assert per_element[:, :, 1:].count_nonzero() == 0
    loss.backward()
    assert tiny_model.head.head.bias.grad.abs().max() > 0


@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.runtime_budget(30, reason="Real attention backward and optimizer checkpoint round trip")
def test_optimizer_resume_reproduces_next_update(tiny_model, training_inputs, tmp_path, lora):
    """Saving after step one reproduces step two exactly for SFT and LoRA."""
    if lora:
        apply_lora_to_linear_modules(
            tiny_model,
            PeftConfig(
                dim=4,
                alpha=4,
                dropout=0.0,
                lora_dtype=torch.float32,
                use_memory_efficient_lora=False,
                target_modules=["*.self_attn.to_q", "*.self_attn.to_k", "*.self_attn.to_v"],
            ),
        )
    restored = deepcopy(tiny_model)
    adapter = WanAnimate2Adapter()
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3, foreach=False)
    target = torch.randn(1, 16, 3, 4, 4)

    def step(model, opt):
        """Update from prediction/target tensors [1, 16, 3, 4, 4]."""
        opt.zero_grad(set_to_none=True)
        loss = (adapter.forward(model, training_inputs) - target).square().mean()
        loss.backward()
        opt.step()
        return loss.detach()

    step(tiny_model, optimizer)
    checkpoint = tmp_path / "resume.pt"
    torch.save({"model": tiny_model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
    expected_loss = step(tiny_model, optimizer)
    saved = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(saved["model"])
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3, foreach=False)
    restored_optimizer.load_state_dict(saved["optimizer"])
    actual_loss = step(restored, restored_optimizer)
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    for name, value in tiny_model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0, msg=name)
