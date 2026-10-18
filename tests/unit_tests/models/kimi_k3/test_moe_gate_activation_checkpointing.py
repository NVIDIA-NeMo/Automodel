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

"""CPU regression coverage for KimiK3Gate routing replay under activation checkpointing.

Activation checkpointing re-runs the gate during the backward recompute. The replayed
selection must survive that second gate call in every checkpointing mode, must not leak
between microbatches, and must leave the no-replay and record paths untouched.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import set_checkpoint_early_stop

from nemo_automodel.components.distributed.activation_checkpointing import unwrap_checkpoint_wrapper
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.kimi_k3.config import KimiK3TextConfig
from nemo_automodel.components.models.kimi_k3.model import KimiK3Gate, KimiK3MoE, _build_moe_config
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.parallelizer import apply_ac
from nemo_automodel.components.moe.router_replay import RouterReplay

HIDDEN_SIZE = 32
NUM_EXPERTS = 4
TOPK = 2

CHECKPOINT_MODES = [
    pytest.param({}, id="router-preserving"),
    pytest.param({"selective": True}, id="operator-selective"),
    pytest.param({"ignore_router": False}, id="full"),
]


def _tiny_config() -> KimiK3TextConfig:
    return KimiK3TextConfig(
        vocab_size=64,
        hidden_size=HIDDEN_SIZE,
        head_dim=8,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        torch_dtype="float32",
        num_experts=NUM_EXPERTS,
        num_experts_per_token=TOPK,
        num_shared_experts=0,
        first_k_dense_replace=0,
        moe_intermediate_size=16,
        routed_expert_hidden_size=16,
        q_lora_rank=16,
        kv_lora_rank=16,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [],
            "full_attn_layers": [1],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
        attn_res_block_size=1,
    )


class _TinyK3Block(nn.Module):
    def __init__(self, config: KimiK3TextConfig, moe_config: MoEConfig) -> None:
        super().__init__()
        backend = BackendConfig(
            attn="eager",
            linear="torch",
            experts="torch",
            dispatcher="torch",
            enable_hf_state_dict_adapter=False,
        )
        self.mlp = KimiK3MoE(config, moe_config, backend)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the K3 MoE sub-block.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.mlp(hidden_states)


class _TinyK3Decoder(nn.Module):
    def __init__(self, config: KimiK3TextConfig, moe_config: MoEConfig) -> None:
        super().__init__()
        self.layers = nn.ModuleDict({"0": _TinyK3Block(config, moe_config)})
        self.moe_config = moe_config

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the single decoder block.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.layers["0"](hidden_states)


class _TinyK3Model(nn.Module):
    def __init__(self, config: KimiK3TextConfig, moe_config: MoEConfig) -> None:
        super().__init__()
        self.model = _TinyK3Decoder(config, moe_config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Run the tiny K3 model.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        return self.model(hidden_states)


@pytest.fixture(autouse=True)
def clean_router_replay_registry():
    """Isolate the process-global RouterReplay registry around every test.

    Yields:
        None.
    """
    RouterReplay.clear_registry()
    yield
    RouterReplay.clear_registry()


def _gate(model: _TinyK3Model) -> KimiK3Gate:
    block = unwrap_checkpoint_wrapper(model.model.layers["0"])
    assert isinstance(block.mlp.gate, KimiK3Gate)
    return block.mlp.gate


def _build_model(enable_routing_replay: bool, seed: int = 0) -> _TinyK3Model:
    """Build a tiny K3 MoE model with deterministic, non-trivial router and expert weights.

    Args:
        enable_routing_replay: Whether the gate owns a ``RouterReplay`` handle.
        seed: Seed for the parameter initialization.

    Returns:
        The model in training mode with random fp32 weights. The correction bias is
        randomized so the biased selection differs from the unbiased weights.
    """
    config = _tiny_config()
    moe_overrides = {"enable_routing_replay": True} if enable_routing_replay else None
    moe_config = _build_moe_config(config, torch.float32, moe_overrides)
    model = _TinyK3Model(config, moe_config).train()
    torch.manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.02)
        gate = _gate(model)
        gate.weight.normal_(0.0, 1.0)
        gate.e_score_correction_bias.normal_(0.0, 0.1)
    return model


def _build_pair(enable_routing_replay: bool, checkpoint_kwargs: dict[str, bool]) -> tuple[_TinyK3Model, _TinyK3Model]:
    """Build an uncheckpointed reference and an identically initialized checkpointed model.

    Each model registers its own ``RouterReplay`` handle (``copy.deepcopy`` would skip the
    registry), so the registry holds exactly two instances in (reference, checkpointed) order.

    Args:
        enable_routing_replay: Whether the gates own ``RouterReplay`` handles.
        checkpoint_kwargs: Keyword arguments forwarded to the production ``apply_ac``.

    Returns:
        Tuple of (reference, checkpointed) models.
    """
    reference = _build_model(enable_routing_replay)
    checkpointed = _build_model(enable_routing_replay)
    for (name, parameter), (other_name, other) in zip(reference.named_parameters(), checkpointed.named_parameters()):
        assert name == other_name
        assert torch.equal(parameter, other), name
    assert torch.equal(_gate(reference).e_score_correction_bias, _gate(checkpointed).e_score_correction_bias)
    apply_ac(checkpointed, hidden_size=HIDDEN_SIZE, num_experts=NUM_EXPERTS, **checkpoint_kwargs)
    if enable_routing_replay:
        assert RouterReplay.instances() == [_gate(reference).router_replay, _gate(checkpointed).router_replay]
    return reference, checkpointed


class _IndexCapture:
    """Record the expert indices every gate call returns, including backward recomputes."""

    def __init__(self, gate: KimiK3Gate) -> None:
        self.calls: list[torch.Tensor] = []
        self._handle = gate.register_forward_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple, output: tuple) -> None:
        """Store the indices from one gate call.

        Args:
            module: The gate.
            inputs: Gate inputs; ``inputs[0]`` is the [tokens, hidden] state tensor (unused).
            output: Gate outputs ``(weights, indices, aux_loss)``; ``indices`` has shape [tokens, topk].
        """
        del module, inputs
        self.calls.append(output[1].detach().clone())

    def remove(self) -> None:
        self._handle.remove()

    def assert_all_equal(self, expected: torch.Tensor) -> None:
        """Assert every captured gate call returned ``expected``.

        Args:
            expected: Tensor of shape [tokens, topk].
        """
        assert self.calls, "the gate was never called"
        for indices in self.calls:
            assert torch.equal(indices, expected)


def _live_indices(model: _TinyK3Model, inputs: torch.Tensor) -> torch.Tensor:
    """Return the live (non-replayed) selection of ``model`` on ``inputs``.

    Args:
        model: Model whose gate is not in any replay mode.
        inputs: Tensor of shape [batch, sequence, hidden].

    Returns:
        Tensor of shape [batch * sequence, topk].
    """
    capture = _IndexCapture(_gate(model))
    with torch.no_grad():
        model(inputs)
    capture.remove()
    assert len(capture.calls) == 1
    return capture.calls[0]


def _unpicked(live_indices: torch.Tensor) -> torch.Tensor:
    """Return, per token, the experts the live router did not select.

    With four experts and top-2 this is exactly one alternative selection per token,
    so replaying it moves every token to different experts than the checkpointed
    forward would route to on its own.

    Args:
        live_indices: Tensor of shape [tokens, topk].

    Returns:
        Tensor of shape [tokens, topk] disjoint from ``live_indices`` on every row.
    """
    tokens = live_indices.shape[0]
    unpicked = torch.ones(tokens, NUM_EXPERTS, dtype=torch.bool).scatter_(1, live_indices, False)
    target = unpicked.nonzero()[:, 1].view(tokens, TOPK)
    assert not (target.unsqueeze(-1) == live_indices.unsqueeze(1)).any()
    return target


def _forward_backward(
    model: _TinyK3Model, inputs: torch.Tensor, upstream: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run one forward/backward step with a nonuniform upstream gradient.

    Args:
        model: Model under test.
        inputs: Leaf tensor of shape [batch, sequence, hidden] with ``requires_grad``.
        upstream: Tensor of the same shape used as the upstream gradient.

    Returns:
        Detached output, the input gradient, and the (accumulated) parameter gradients
        keyed by the unwrapped parameter name.
    """
    with set_checkpoint_early_stop(False):
        output = model(inputs)
    (output * upstream).sum().backward()
    assert inputs.grad is not None
    grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), inputs.grad.detach().clone(), grads


def _assert_parity(
    reference: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
    checkpointed: tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]],
) -> None:
    """Assert the checkpointed step reproduces the reference step exactly.

    Args:
        reference: ``_forward_backward`` result of the uncheckpointed model: output and
            input gradient of shape [batch, sequence, hidden] plus parameter gradients.
        checkpointed: The same for the checkpointed model.
    """
    ref_output, ref_input_grad, ref_grads = reference
    ckpt_output, ckpt_input_grad, ckpt_grads = checkpointed
    torch.testing.assert_close(ckpt_output, ref_output)
    torch.testing.assert_close(ckpt_input_grad, ref_input_grad)
    assert ckpt_grads.keys() == ref_grads.keys()
    # Router, routed experts and the latent projections must all receive gradients.
    for required in (
        "model.layers.0.mlp.gate.weight",
        "model.layers.0.mlp.experts.gate_and_up_projs",
        "model.layers.0.mlp.experts.down_projs",
        "model.layers.0.mlp.routed_expert_down_proj.weight",
        "model.layers.0.mlp.routed_expert_up_proj.weight",
    ):
        assert required in ref_grads, required
        assert ref_grads[required].abs().max() > 0, required
    for name, ref_grad in ref_grads.items():
        torch.testing.assert_close(ckpt_grads[name], ref_grad, msg=name)


def _assert_replay_inactive() -> None:
    for instance in RouterReplay.instances():
        assert instance.mode is None
        assert instance.target_indices is None


@pytest.mark.parametrize("checkpoint_kwargs", CHECKPOINT_MODES)
def test_replayed_selection_survives_checkpoint_recompute(checkpoint_kwargs: dict[str, bool]) -> None:
    reference, checkpointed = _build_pair(True, checkpoint_kwargs)
    torch.manual_seed(1)
    inputs = torch.randn(2, 3, HIDDEN_SIZE)
    upstream = torch.randn(2, 3, HIDDEN_SIZE)
    live = _live_indices(reference, inputs)
    assert torch.equal(_live_indices(checkpointed, inputs), live)
    target = _unpicked(live)

    ref_capture = _IndexCapture(_gate(reference))
    ckpt_capture = _IndexCapture(_gate(checkpointed))
    with RouterReplay.replay([target, target]):
        ref_result = _forward_backward(reference, inputs.clone().requires_grad_(True), upstream)
        ckpt_result = _forward_backward(checkpointed, inputs.clone().requires_grad_(True), upstream)
    ref_capture.remove()
    ckpt_capture.remove()

    ref_capture.assert_all_equal(target)
    ckpt_capture.assert_all_equal(target)
    _assert_parity(ref_result, ckpt_result)
    _assert_replay_inactive()


@pytest.mark.parametrize("checkpoint_kwargs", CHECKPOINT_MODES)
def test_consecutive_microbatches_replay_their_own_targets(checkpoint_kwargs: dict[str, bool]) -> None:
    reference, checkpointed = _build_pair(True, checkpoint_kwargs)
    torch.manual_seed(2)
    microbatches = [
        (torch.randn(2, 3, HIDDEN_SIZE), torch.randn(2, 3, HIDDEN_SIZE)),
        (torch.randn(1, 4, HIDDEN_SIZE), torch.randn(1, 4, HIDDEN_SIZE)),
    ]
    targets = [_unpicked(_live_indices(reference, inputs)) for inputs, _ in microbatches]
    assert targets[0].shape != targets[1].shape

    for (inputs, upstream), target in zip(microbatches, targets):
        ref_capture = _IndexCapture(_gate(reference))
        ckpt_capture = _IndexCapture(_gate(checkpointed))
        with RouterReplay.replay([target, target]):
            ref_result = _forward_backward(reference, inputs.clone().requires_grad_(True), upstream)
            ckpt_result = _forward_backward(checkpointed, inputs.clone().requires_grad_(True), upstream)
        ref_capture.remove()
        ckpt_capture.remove()
        ref_capture.assert_all_equal(target)
        ckpt_capture.assert_all_equal(target)
        torch.testing.assert_close(ckpt_result[0], ref_result[0])
        torch.testing.assert_close(ckpt_result[1], ref_result[1])
        _assert_replay_inactive()

    # Gradients accumulated over both microbatches match the uncheckpointed reference.
    _assert_parity(ref_result, ckpt_result)

    # After the contexts exit, a plain forward uses the live selection again.
    inputs, _ = microbatches[1]
    live = _live_indices(reference, inputs)
    assert torch.equal(_live_indices(checkpointed, inputs), live)
    assert not torch.equal(live, targets[1])


@pytest.mark.parametrize("checkpoint_kwargs", CHECKPOINT_MODES)
def test_checkpointing_without_routing_replay_is_unchanged(checkpoint_kwargs: dict[str, bool]) -> None:
    reference, checkpointed = _build_pair(False, checkpoint_kwargs)
    assert _gate(reference).router_replay is None
    assert _gate(checkpointed).router_replay is None
    assert RouterReplay.instances() == []
    torch.manual_seed(3)
    inputs = torch.randn(2, 3, HIDDEN_SIZE)
    upstream = torch.randn(2, 3, HIDDEN_SIZE)

    ref_result = _forward_backward(reference, inputs.clone().requires_grad_(True), upstream)
    ckpt_result = _forward_backward(checkpointed, inputs.clone().requires_grad_(True), upstream)
    _assert_parity(ref_result, ckpt_result)


@pytest.mark.parametrize("checkpoint_kwargs", CHECKPOINT_MODES)
def test_inactive_routing_replay_routes_live_under_checkpointing(checkpoint_kwargs: dict[str, bool]) -> None:
    reference, checkpointed = _build_pair(True, checkpoint_kwargs)
    torch.manual_seed(4)
    inputs = torch.randn(2, 3, HIDDEN_SIZE)
    upstream = torch.randn(2, 3, HIDDEN_SIZE)
    live = _live_indices(reference, inputs)

    _assert_replay_inactive()
    ref_capture = _IndexCapture(_gate(reference))
    ckpt_capture = _IndexCapture(_gate(checkpointed))
    ref_result = _forward_backward(reference, inputs.clone().requires_grad_(True), upstream)
    ckpt_result = _forward_backward(checkpointed, inputs.clone().requires_grad_(True), upstream)
    ref_capture.remove()
    ckpt_capture.remove()

    ref_capture.assert_all_equal(live)
    ckpt_capture.assert_all_equal(live)
    _assert_parity(ref_result, ckpt_result)


@pytest.mark.parametrize("checkpoint_kwargs", CHECKPOINT_MODES)
def test_record_mode_captures_live_selection_through_checkpoint_recompute(
    checkpoint_kwargs: dict[str, bool],
) -> None:
    reference, checkpointed = _build_pair(True, checkpoint_kwargs)
    torch.manual_seed(5)
    inputs = torch.randn(2, 3, HIDDEN_SIZE)
    upstream = torch.randn(2, 3, HIDDEN_SIZE)
    live = _live_indices(reference, inputs)

    with RouterReplay.record():
        ref_result = _forward_backward(reference, inputs.clone().requires_grad_(True), upstream)
        with set_checkpoint_early_stop(False):
            ckpt_output = checkpointed(inputs.clone().requires_grad_(True))
        recorded_after_forward = [instance.recorded_indices.clone() for instance in RouterReplay.instances()]
        (ckpt_output * upstream).sum().backward()

    collected = RouterReplay.collect()
    assert len(collected) == 2
    for recorded, before_backward in zip(collected, recorded_after_forward):
        assert torch.equal(recorded, live)
        assert torch.equal(recorded, before_backward)
    _assert_replay_inactive()

    torch.testing.assert_close(ckpt_output.detach(), ref_result[0])
    ckpt_grads = {
        name.replace("._checkpoint_wrapped_module", ""): parameter.grad.detach().clone()
        for name, parameter in checkpointed.named_parameters()
        if parameter.grad is not None
    }
    for name, ref_grad in ref_result[2].items():
        torch.testing.assert_close(ckpt_grads[name], ref_grad, msg=name)
