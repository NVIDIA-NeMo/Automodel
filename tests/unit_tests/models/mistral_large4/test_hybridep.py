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

"""Grouped Mistral expert layout, FP8 import and numerical training parity."""

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.models.mistral_large4.state_dict_adapter import MistralLarge4StateDictAdapter
from nemo_automodel.components.moe.experts import GroupedExperts
from tests.unit_tests.models.mistral_large4.test_model import assert_bf16_error, make_model, tensor_error


def grouped_backend(dispatcher: str = "torch") -> BackendConfig:
    """Return the supported grouped backend; CPU tests exercise torch dispatch."""
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch_mm",
        dispatcher=dispatcher,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("seed", [312, 47, 99])
def test_grouped_logits_gradients_and_update(dtype, topk, seed):
    reference = make_model(seed=seed, dtype=dtype, num_experts_per_tok=topk, moe_intermediate_size=32)
    model = MistralLarge4ForCausalLM(reference.config, grouped_backend())
    source = reference.state_dict_adapter.to_hf(reference.state_dict())
    model.load_state_dict(model.state_dict_adapter.from_hf(source), strict=True)
    exported = model.state_dict_adapter.to_hf(model.state_dict())
    assert set(exported) == set(source)
    for key in source:
        torch.testing.assert_close(exported[key], source[key], atol=0, rtol=0)
    ids = torch.tensor([[1, 5, 7, 2, 9, 3, 8]])
    expected, actual = reference(ids).logits, model(ids).logits
    atol, rtol = (3e-6, 3e-5) if dtype == torch.float32 else (0.002, 0.04)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    grads = model.state_dict_adapter.to_hf({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
    ref_grads = reference.state_dict_adapter.to_hf(
        {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    )
    assert set(grads) == set(ref_grads)
    errors = []
    for key in grads:
        assert torch.isfinite(grads[key]).all(), key
        if dtype == torch.float32:
            torch.testing.assert_close(grads[key], ref_grads[key], atol=atol * 8, rtol=rtol * 2, msg=key)
        else:
            error = tensor_error(key, grads[key], ref_grads[key])
            assert_bf16_error(error)
            errors.append(error)
    if errors:
        relative_l2 = (sum(e.squared_error for e in errors) / sum(e.reference_squared_norm for e in errors)) ** 0.5
        assert relative_l2 < 0.005, relative_l2
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    torch.optim.SGD(reference.parameters(), lr=0.001).step()
    updated = model.state_dict_adapter.to_hf(model.state_dict())
    expected = reference.state_dict_adapter.to_hf(reference.state_dict())
    for key in updated:
        torch.testing.assert_close(updated[key], expected[key], atol=atol, rtol=rtol, msg=key)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_fp8_import_aliases_and_peak_staging(dtype):
    adapter = MistralLarge4StateDictAdapter(dtype, grouped_experts=True, n_routed_experts=4)
    # Deliberately asymmetric matrices and dimensions crossing a 128-element scale block.
    generator = torch.Generator().manual_seed(15)
    source = {}
    for expert in range(4):
        for projection, shape in (("w1", (132, 260)), ("w3", (132, 260)), ("w2", (260, 132))):
            source[f"layers.1.experts.{expert}.{projection}.weight"] = torch.randn(shape, generator=generator).to(dtype)
    individual = MistralLarge4StateDictAdapter(dtype)
    quantized = {}
    for name, value in source.items():
        quantized[name] = value.to(torch.float8_e4m3fn)
        rows, cols = (value.shape[0] + 127) // 128, (value.shape[1] + 127) // 128
        quantized[name.removesuffix(".weight") + ".qscale_weight"] = (
            torch.arange(1, rows * cols + 1, dtype=torch.float32).reshape(rows, cols) / 8
        )
    decoded = individual.from_hf(quantized)
    expected = adapter.from_hf(quantized)
    destinations = {key: torch.empty_like(value) for key, value in expected.items()}
    pointers = {key: value.data_ptr() for key, value in destinations.items()}
    completed = set()
    for part in adapter.iter_checkpoint_load_parts(destinations):
        assert completed.isdisjoint(part.model_keys)
        completed.update(part.model_keys)
        for key, value in part.checkpoint_tensors.items():
            value.copy_(quantized[key])
        # At most one FP8 local grouped parameter plus its scale grids is staged.
        temporary_bytes = sum(
            part.checkpoint_tensors[key].numel() * part.checkpoint_tensors[key].element_size()
            for key in part.temporary_checkpoint_keys
        )
        parameter = destinations[next(iter(part.model_keys))]
        assert temporary_bytes < parameter.numel() + 1024
        part.finish()
    assert completed == set(destinations)
    for key in destinations:
        assert destinations[key].data_ptr() == pointers[key]
        torch.testing.assert_close(destinations[key], expected[key], atol=0, rtol=0)
    exported = adapter.to_hf(destinations)
    expected_export = individual.to_hf(decoded)
    for key in exported:
        torch.testing.assert_close(exported[key], expected_export[key], atol=0, rtol=0)
    # Export views write through to the final grouped parameter, including transpose/split.
    exported["layers.1.experts.2.w3.weight"].fill_(0.375)
    assert destinations["model.layers.1.mlp.experts.gate_and_up_projs"][2, :, 132:].eq(0.375).all()


def test_grouped_meta_initialization_and_native_resume(tmp_path):
    reference = make_model(dtype=torch.bfloat16, moe_intermediate_size=32)
    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(reference.config, grouped_backend())
    model.to_empty(device="cpu")
    model.initialize_weights(dtype=torch.bfloat16)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    assert all(p.dtype == torch.float32 for name, p in model.named_parameters() if "mlp.gate.weight" in name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids = torch.tensor([[1, 2, 3, 4]])
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, tmp_path / "native.pt")
    restored = MistralLarge4ForCausalLM(reference.config, grouped_backend())
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001)
    state = torch.load(tmp_path / "native.pt", weights_only=True)
    restored.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    for candidate, opt in ((model, optimizer), (restored, restored_optimizer)):
        opt.zero_grad(set_to_none=True)
        candidate(ids, labels=ids).loss.backward()
        opt.step()
    for actual, expected in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_grouped_dense_only_prefix_remains_valid():
    reference = make_model(first_k_dense_replace=100)
    model = MistralLarge4ForCausalLM(reference.config, grouped_backend())
    model.load_state_dict(reference.state_dict())
    ids = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(model(ids).logits, reference(ids).logits, atol=0, rtol=0)


def test_hybridep_rejects_fp32_storage():
    reference = make_model(dtype=torch.float32)
    with pytest.raises(ValueError, match="requires BF16"):
        MistralLarge4ForCausalLM(reference.config, grouped_backend("hybridep"))


class TransportRoundedExperts(GroupedExperts):
    """Single-process oracle for BF16 weighted-output transport before combine.

    Shared Torch experts otherwise retain FP32 weighted outputs until summation.
    This reference inserts the cast performed by HybridEP communication; it does
    not communicate, change routing choices, or modify the original oracle.
    """

    def _forward_grouped_mm(
        self,
        x,
        token_mask,
        weights,
        indices,
        gate_and_up_projs,
        down_projs,
        gate_up_proj_bias,
        down_proj_bias,
        n_local_experts,
        experts_start_idx,
        post_expert_output,
    ) -> torch.Tensor:
        """Round separate weighted expert outputs before the parent's reduction.

        Args:
            x: BF16 local activations [tokens, hidden].
            token_mask: Boolean valid-token mask [tokens].
            weights: FP32 routing probabilities [tokens, topk].
            indices: Integer expert IDs [tokens, topk].
            gate_and_up_projs: BF16 fused matrices [experts, hidden, 2*intermediate].
            down_projs: BF16 matrices [experts, intermediate, hidden].
            gate_up_proj_bias: None for this model.
            down_proj_bias: None for this model.
            n_local_experts: All experts in the single-process reference.
            experts_start_idx: Zero for the reference.
            post_expert_output: Optional inherited output transform.

        Returns:
            Independent FP32 [tokens, topk, hidden] values rounded to BF16,
            ready for the parent's FP32 combine and final BF16 cast. No input
            or parameter is modified; autograd traverses the cast normally.
        """
        output = super()._forward_grouped_mm(
            x,
            token_mask,
            weights,
            indices,
            gate_and_up_projs,
            down_projs,
            gate_up_proj_bias,
            down_proj_bias,
            n_local_experts,
            experts_start_idx,
            post_expert_output,
        )
        assert output.ndim == 3 and self.config.apply_router_weight_after_down
        return output.to(x.dtype).float()


def use_transport_rounding_reference(model: MistralLarge4ForCausalLM) -> None:
    """Install test-only expert arithmetic on an unsharded model before optimizer creation."""
    for block in model.model.layers:
        if isinstance(block.mlp.experts if hasattr(block.mlp, "experts") else None, GroupedExperts):
            source = block.mlp.experts
            replacement = TransportRoundedExperts(source.config, backend=grouped_backend()).to(
                source.gate_and_up_projs.device
            )
            replacement.expert_activation_grouped = source.expert_activation_grouped
            replacement.load_state_dict(source.state_dict())
            block.mlp.experts = replacement


def test_transport_reference_matches_individual_weighted_bf16_outputs():
    torch.manual_seed(61)
    model = MistralLarge4ForCausalLM(
        make_model(dtype=torch.bfloat16, moe_intermediate_size=32).config, grouped_backend()
    )
    model.initialize_weights()
    use_transport_rounding_reference(model)
    expert = model.model.layers[1].mlp.experts
    hidden = torch.randn(13, 16, dtype=torch.bfloat16, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_(True)
    scores = torch.randn(13, 3, dtype=torch.float32)
    probabilities, indices = scores.softmax(-1).topk(2, dim=-1)
    probabilities = (probabilities / probabilities.sum(-1, keepdim=True)).requires_grad_(True)
    reference_probabilities = probabilities.detach().clone().requires_grad_(True)
    gate_up = expert.gate_and_up_projs.detach().clone().requires_grad_(True)
    down = expert.down_projs.detach().clone().requires_grad_(True)
    expected = torch.zeros_like(hidden, dtype=torch.float32)
    for identity in range(3):
        rows, slots = torch.where(indices == identity)
        gate, up = gate_up[identity].chunk(2, dim=-1)
        value = (torch.nn.functional.silu(reference_hidden[rows] @ gate) * (reference_hidden[rows] @ up)) @ down[
            identity
        ]
        weighted = (value.float() * reference_probabilities[rows, slots, None]).bfloat16()
        expected = expected.index_add(0, rows, weighted.float())
    expected = expected.bfloat16()
    actual = expert(hidden, torch.ones(13, dtype=torch.bool), probabilities, indices)
    assert_bf16_error(tensor_error("output", actual, expected))
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for name, observed, oracle in (
        ("input", hidden.grad, reference_hidden.grad),
        ("probabilities", probabilities.grad, reference_probabilities.grad),
        ("gate_up", expert.gate_and_up_projs.grad, gate_up.grad),
        ("down", expert.down_projs.grad, down.grad),
    ):
        assert_bf16_error(tensor_error(name, observed, oracle))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_resume_returns_original_banks_and_clears_load_state(dtype):
    reference = make_model(dtype=dtype, moe_intermediate_size=32)
    adapter = MistralLarge4StateDictAdapter(dtype, grouped_experts=True, n_routed_experts=3)
    source = reference.state_dict_adapter.to_hf(reference.state_dict())
    expected = adapter.from_hf(source)
    state = {key: torch.zeros_like(value) for key, value in expected.items()}
    views = adapter.to_hf(state, for_checkpoint_load=True)
    # Exercise both in-place DCP fill and a loader returning equivalent new tensors.
    for key, destination in views.items():
        destination.copy_(source[key])
    loaded = adapter.from_hf(dict(views))
    for key in expected:
        assert loaded[key] is state[key]
        torch.testing.assert_close(loaded[key], expected[key], atol=0, rtol=0)
    views = adapter.to_hf(state, for_checkpoint_load=True)
    loaded = adapter.from_hf({key: value.clone() for key, value in views.items()})
    for key in expected:
        if key.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
            assert loaded[key] is state[key]
        torch.testing.assert_close(loaded[key], expected[key], atol=0, rtol=0)
    assert not adapter._grouped_load_views
    views = adapter.to_hf(state, for_checkpoint_load=True)
    key = next(key for key in views if ".experts." in key)
    del views[key]
    with pytest.raises(ValueError, match="Incomplete Mistral grouped checkpoint destination"):
        adapter.from_hf(views)
    assert not adapter._grouped_load_views
    unrelated = adapter.from_hf(source)
    for key in expected:
        torch.testing.assert_close(unrelated[key], expected[key], atol=0, rtol=0)


def test_grouped_resume_handles_distributed_export_copies(monkeypatch):
    # A CPU-offloaded bank on a CUDA mesh is copied by DTensor.from_local.
    # Emulate that loss of aliasing without requiring a GPU in the regression.
    from nemo_automodel.components.moe import state_dict_utils

    reference = make_model(dtype=torch.bfloat16, moe_intermediate_size=32)
    adapter = MistralLarge4StateDictAdapter(grouped_experts=True, n_routed_experts=3)
    source = reference.state_dict_adapter.to_hf(reference.state_dict())
    expected = adapter.from_hf(source)
    state = {key: torch.zeros_like(value) for key, value in expected.items()}
    split = state_dict_utils.split_experts_weights_dtensor_aware

    def copied_split(value, count):
        matrices, identities = split(value, count)
        return [matrix.clone() for matrix in matrices], identities

    monkeypatch.setattr(state_dict_utils, "split_experts_weights_dtensor_aware", copied_split)
    destinations = adapter.to_hf(state, for_checkpoint_load=True)
    for key, destination in destinations.items():
        destination.copy_(source[key])
    for key, value in state.items():
        if key.endswith((".experts.gate_and_up_projs", ".experts.down_projs")):
            assert value.count_nonzero() == 0
    restored = adapter.from_hf(destinations)
    for key in expected:
        assert restored[key] is state[key]
        torch.testing.assert_close(restored[key], expected[key], atol=0, rtol=0)
