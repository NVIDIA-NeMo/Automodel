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

"""Text equations, precision, packed attention and distributed capability contracts."""

import copy
import math
from dataclasses import dataclass
from typing import Literal

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.nn import functional as F

from nemo_automodel._transformers.model_capabilities import query_capabilities
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.distributed.parallelizer import get_model_layer_groups
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.mistral_large4.attention import MistralLarge4Indexer, SequenceLayout
from nemo_automodel.components.models.mistral_large4.config import ConfigValue
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from nemo_automodel.components.models.mistral_large4.rope_utils import MistralLarge4Rotary
from tests.unit_tests.models.mistral_large4.math_reference import reference_logits, tiny_config


def make_model(*, seed: int = 312, **overrides: ConfigValue | torch.dtype) -> MistralLarge4ForCausalLM:
    torch.manual_seed(seed)
    model = MistralLarge4ForCausalLM(tiny_config(**overrides))
    model.initialize_weights(torch.device("cpu"))
    # Exercise learned frequencies/SSMax, including gradients into every shared alpha.
    with torch.no_grad():
        model.model.rotary_emb.alpha.copy_(torch.tensor([0.1234567, -0.217891]))
        for name, p in model.named_parameters():
            if name.endswith("ssmax.s"):
                p.fill_(0.12)
    return model


def test_forward_backward_update_against_independent_oracle() -> None:
    dtype = torch.float32
    model = make_model(dtype=dtype)
    source = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    oracle = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in source.items()}
    ids = torch.tensor([1, 5, 7, 2, 9, 3, 8])
    expected = reference_logits(oracle, ids, model.config)
    actual = model(ids[None]).logits[0]
    # FP32 permits only floating reduction-order differences. BF16 backward
    # accumulation against this token-wise oracle is a separately reported gate.
    atol, rtol = 2e-6, 2e-5
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    gradients = model.state_dict_adapter.to_hf({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    for key, parameter in source.items():
        if not parameter.requires_grad:
            assert key not in gradients and oracle[key].grad is None, key
        else:
            reference_grad = oracle[key].grad
            # Token dispatch does not visit an unused expert; vector baseline yields zero.
            if reference_grad is None:
                reference_grad = torch.zeros_like(parameter)
            torch.testing.assert_close(gradients[key], reference_grad, atol=atol * 8, rtol=rtol * 2, msg=key)
    torch.optim.SGD(model.parameters(), lr=0.001).step()
    torch.optim.SGD(oracle.values(), lr=0.001).step()
    source = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    for key, parameter in source.items():
        torch.testing.assert_close(parameter, oracle[key], atol=atol, rtol=rtol, msg=key)


def test_registry_meta_initialization_and_precision() -> None:
    c = tiny_config()
    assert ModelRegistry.get_model_cls_from_model_arch(c.architectures[0]) is MistralLarge4ForCausalLM
    caps = query_capabilities(c)
    assert caps.supports_thd and caps.supports_cp and caps.supports_ep
    assert not caps.supports_tp and caps.supports_pp
    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(c)
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with torch.no_grad():
        model.model.rotary_emb.alpha.copy_(torch.tensor([0.1234567, -0.217891]))
    original = model.model.rotary_emb.alpha.detach().clone()
    cast_model_to_dtype(model, torch.bfloat16)
    torch.testing.assert_close(model.model.rotary_emb.alpha, original, rtol=0, atol=0)
    assert all(
        p.dtype == torch.float32
        for name, p in model.named_parameters()
        if "mlp.gate." in name or "indexer.k_norm." in name
    )
    loss = model(torch.tensor([[1, 2, 3, 4]]), labels=torch.tensor([[1, 2, 3, 4]])).loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr()


def test_packed_isolation_padding_and_document_edge_loss() -> None:
    model = make_model()
    a, b = torch.tensor([1, 5, 2, 7]), torch.tensor([3, 6, 8, 9, 2])
    packed = torch.cat((a, b))
    args = dict(qkv_format="thd", cu_seqlens=torch.tensor([0, 4, 9]))
    output = model(packed, labels=packed, **args)
    torch.testing.assert_close(output.logits[0, :4], model(a[None]).logits[0], atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(output.logits[0, 4:], model(b[None]).logits[0], atol=2e-6, rtol=2e-5)
    perturbed = packed.clone()
    perturbed[:4] = 10
    torch.testing.assert_close(output.logits[0, 4:], model(perturbed, **args).logits[0, 4:], atol=0, rtol=0)
    expected_loss = (model(a[None], labels=a[None]).loss * 3 + model(b[None], labels=b[None]).loss * 4) / 7
    torch.testing.assert_close(output.loss, expected_loss)
    padded = torch.tensor([[0, 0, 3, 6, 8, 9, 2, 0]])
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 1, 0]])
    torch.testing.assert_close(
        model(padded, attention_mask=mask).logits[:, 2:7], model(b[None]).logits, atol=2e-6, rtol=2e-5
    )
    assert model(padded, attention_mask=torch.zeros_like(mask), labels=padded).loss.item() == 0


def test_late_kv_sharing_35_39_recompute_gradients() -> None:
    depth = 40
    model = make_model(num_hidden_layers=depth, nope_layer_flags=None, kv_source_layers=None, index_sharing=None)
    checked = copy.deepcopy(model)
    for i, layer in enumerate(checked.model.layers):
        checked.model.layers[i] = checkpoint_wrapper(layer)
    ids = torch.tensor([[1, 4, 7, 2, 9]])
    positions = torch.tensor([[2047, 2048, 2049, 4095, 4096]])
    out, out_ac = model(ids, position_ids=positions).logits, checked(ids, position_ids=positions).logits
    source = model.state_dict_adapter.to_hf(dict(model.named_parameters()))
    oracle = {k: v.detach().clone().requires_grad_(v.requires_grad) for k, v in source.items()}
    expected = reference_logits(oracle, ids[0], model.config, positions[0])
    torch.testing.assert_close(out[0], expected, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(out, out_ac, atol=0, rtol=0)
    upstream = torch.randn_like(out)
    out.backward(upstream)
    out_ac.backward(upstream)
    expected.backward(upstream[0])
    gradients = model.state_dict_adapter.to_hf({n: p.grad for n, p in model.named_parameters() if p.grad is not None})
    for key, p in source.items():
        if p.requires_grad:
            reference = oracle[key].grad
            if reference is None:
                reference = torch.zeros_like(p)
            torch.testing.assert_close(gradients[key], reference, atol=2e-5, rtol=4e-5, msg=key)
    for (name, p), (other_name, q) in zip(model.named_parameters(), checked.named_parameters()):
        assert name == other_name.replace("._checkpoint_wrapped_module", "")
        if p.grad is None:
            assert q.grad is None
        else:
            torch.testing.assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
    assert model.model.layers[39].self_attn.shares_kv
    assert model.model.layers[35].self_attn.kv_a_proj_with_mqa.weight.grad.abs().sum() > 0
    assert not hasattr(model.model.layers[39].self_attn, "kv_a_proj_with_mqa")
    # A second forward may not inherit graph-bearing caches from the first.
    model.zero_grad(set_to_none=True)
    model(ids).logits.square().mean().backward()


@pytest.mark.parametrize("length", [2047, 2048, 2049, 4096])
def test_sparse_boundary_causal_indices(length: int) -> None:
    c = tiny_config(index_topk=2048)
    indexer = MistralLarge4Indexer(c, layer_idx=0, dtype=torch.float32)
    # Zero queries intentionally generate exact ties; earlier valid keys win.
    with torch.no_grad():
        indexer.wq_b.weight.zero_()
    x = torch.randn(length, c.hidden_size)
    positions = torch.arange(length)
    layout = SequenceLayout(positions, torch.zeros(length, dtype=torch.long), torch.ones(length, dtype=torch.bool))
    selected = indexer(x, MistralLarge4Rotary(c)(positions), layout)
    assert selected.shape == (length, min(2048, length))
    assert (selected[0] >= 0).sum() == 1
    assert (selected[-1] >= 0).sum() == min(length, 2048)
    torch.testing.assert_close(selected[-1], torch.arange(min(length, 2048)))
    assert ((selected < 0) | (selected <= positions[:, None])).all()


def test_softcap_all_positions_tail_and_empty_supervision() -> None:
    model = make_model()
    with torch.no_grad():
        model.lm_head.weight.mul_(1000)
    ids = torch.tensor([[1, 3, 8, 4]])
    out = model(ids, output_hidden_states=True, labels=ids)
    raw = model.lm_head(out.hidden_states).float()
    expected = 30 * torch.tanh(raw / 30)
    assert raw.abs().max() > 30 and out.logits.abs().max() <= 30
    torch.testing.assert_close(out.logits, expected)
    torch.testing.assert_close(out.loss, F.cross_entropy(expected[:, :-1].reshape(-1, 31), ids[:, 1:].reshape(-1)))
    torch.testing.assert_close(model(ids, logits_to_keep=2).logits, expected[:, -2:])
    torch.testing.assert_close(model(ids, logits_to_keep=torch.tensor([0, 2])).logits, expected[:, [0, 2]])
    empty = model(ids, labels=torch.full_like(ids, -100)).loss
    assert empty.item() == 0
    empty.backward()
    assert torch.count_nonzero(model.lm_head.weight.grad) == 0


def test_reject_unsupported_training_paths() -> None:
    with pytest.raises(ValueError, match="fp8_indexer=False"):
        MistralLarge4ForCausalLM(tiny_config(fp8_indexer=True))
    model = make_model()
    ids = torch.tensor([[1, 2, 3]])
    with pytest.raises(ValueError, match="without a KV cache"):
        model(ids, use_cache=True)
    model.model.layers[0].self_attn.indexer.wq_b.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="must remain frozen"):
        model(ids)


def test_bf16_forward_against_token_oracle() -> None:
    model = make_model(dtype=torch.bfloat16)
    source = model.state_dict_adapter.to_hf(model.state_dict())
    ids = torch.tensor([1, 5, 7, 2, 9, 3, 8])
    expected = reference_logits(source, ids, model.config)
    torch.testing.assert_close(model(ids[None]).logits[0], expected, rtol=0, atol=0)


def test_sparse_attention_excludes_unselected_keys_beyond_2048() -> None:
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.mistral_large4.attention import MistralLarge4Attention

    c = tiny_config(index_topk=2048)
    attention = MistralLarge4Attention(c, BackendConfig(), layer_idx=0, dtype=torch.float32)
    with torch.no_grad():
        attention.q_proj.weight.zero_()
        attention.indexer.wq_b.weight.zero_()
    positions = torch.arange(2049)
    layout = SequenceLayout(positions, torch.zeros_like(positions), torch.ones(2049, dtype=torch.bool))
    tables = MistralLarge4Rotary(c)(positions)
    x = torch.randn(2049, c.hidden_size)
    expected, _, selected = attention(x, tables, layout, None, None)
    assert 2048 not in selected[-1].tolist()
    altered = x.clone()
    altered[2048] += 10
    output, _, _ = attention(altered, tables, layout, None, None)
    torch.testing.assert_close(output[-1], expected[-1], atol=0, rtol=0)
    altered = x.clone()
    altered[0] *= -1
    output, _, _ = attention(altered, tables, layout, None, None)
    assert (output[-1] - expected[-1]).abs().max() > 1e-7


def test_existing_loss_guard_and_padding_mask() -> None:
    from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy

    model = make_model()
    with pytest.raises(ValueError, match="softcapping"):
        ChunkedCrossEntropy.validate_lm_head(model.lm_head, model_config=model.config)
    ids = torch.tensor([[0, 1, 3, 8, 0]])
    padding = torch.tensor([[True, False, False, False, True]])
    expected = model(ids, attention_mask=(~padding).long()).logits
    torch.testing.assert_close(model(ids, padding_mask=padding).logits, expected, rtol=0, atol=0)


def test_indexer_lora_rejected_after_framework_initialization() -> None:
    from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    model = make_model()
    peft = PeftConfig(target_modules=["*.indexer.wq_b"], dim=2, alpha=4, use_memory_efficient_lora=False)
    assert apply_lora_to_linear_modules(model, peft) > 0
    Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier")
    with pytest.raises(ValueError, match="must remain frozen"):
        model(torch.tensor([[1, 2, 3]]))


@dataclass(frozen=True)
class TensorError:
    """Scalar error statistics; no tensor storage is retained."""

    name: str
    elements: int
    max_absolute: float
    squared_error: float
    reference_squared_norm: float

    @property
    def relative_l2(self) -> float:
        return math.sqrt(self.squared_error / max(self.reference_squared_norm, 1e-30))

    @property
    def rms_error(self) -> float:
        return math.sqrt(self.squared_error / self.elements)

    @property
    def reference_rms(self) -> float:
        return math.sqrt(self.reference_squared_norm / self.elements)


@dataclass(frozen=True)
class BF16Comparison:
    """Numerical evidence for one independently initialized comparison."""

    seed: int
    tokens: int
    layers: int
    objective: str
    fp32_master: bool
    logits: TensorError
    gradients: tuple[TensorError, ...]
    updates: tuple[TensorError, ...]
    loss_absolute_error: float
    global_gradient_relative_l2: float
    global_update_relative_l2: float
    strict_fp32_gradient_failures: int
    frozen_parameters: int


def tensor_error(name: str, actual: torch.Tensor, expected: torch.Tensor) -> TensorError:
    """Compare a corresponding logit, gradient, or update tensor.

    Args:
        name: Diagnostic tensor identity.
        actual: Finite tensor of shape [...], with arbitrary leading dimensions.
        expected: Finite tensor with the same shape and semantic axis order.

    Returns:
        Scalar errors accumulated in FP64; inputs are not modified or aliased.
    """
    assert actual.shape == expected.shape, name
    assert actual.numel() > 0 and torch.isfinite(actual).all() and torch.isfinite(expected).all(), name
    a, b = actual.detach().double(), expected.detach().double()
    difference = a - b
    return TensorError(
        name, a.numel(), difference.abs().max().item(), difference.square().sum().item(), b.square().sum().item()
    )


def assert_bf16_error(error: TensorError, *, absolute: float = 1.6e-5) -> None:
    """Apply a per-tensor RMS budget with a small-gradient absolute floor.

    Relative allowance is two BF16 spacings at unity (2/128). The absolute
    floor retains the original FP32 gradient budget instead of dividing by
    nearly zero SSMax/LeRoPE gradients. Aggregate gradients have a tighter 0.5%
    budget below, so large tensors cannot consume this allowance collectively.
    """
    assert error.rms_error <= absolute + (1 / 64) * error.reference_rms, error


def compare_bf16_gradients(
    *,
    seed: int,
    tokens: int,
    objective: Literal["random", "ce"],
    fp32_master: bool,
    device: torch.device,
    layers: int = 4,
) -> BF16Comparison:
    """Check BF16 compute against independent equations and SGD updates.

    Constructs identical independent parameters. The master mode differentiates
    once-cast BF16 views of FP32 leaves via functional_call, preserving the
    model's FP32 router/LeRoPE/indexer-norm policy. This is a single-device
    numerical check, not a substitute for distributed precision/lifecycle tests.
    """
    overrides = {}
    if layers != 4:
        overrides = dict(num_hidden_layers=layers, nope_layer_flags=None, kv_source_layers=None, index_sharing=None)
    model = make_model(dtype=torch.bfloat16, seed=seed, **overrides).to(device=device)
    native = dict(model.named_parameters())
    parameters = (
        {
            name: torch.nn.Parameter(value.detach().float().clone(), requires_grad=value.requires_grad)
            for name, value in native.items()
        }
        if fp32_master
        else native
    )
    source = model.state_dict_adapter.to_hf(parameters)
    original_types = model.state_dict_adapter.to_hf(native)
    reference = {name: value.detach().clone().requires_grad_(value.requires_grad) for name, value in source.items()}
    for name, value in source.items():
        assert torch.equal(value, reference[name]), name
        assert value.untyped_storage().data_ptr() != reference[name].untyped_storage().data_ptr(), name
    generator = torch.Generator().manual_seed(seed + 10000)
    ids = torch.randint(model.config.vocab_size, (tokens,), generator=generator).to(device)
    # Repeated IDs exercise embedding reduction; large positions exercise LeRoPE.
    ids[-1] = ids[0]
    positions = (
        (torch.arange(tokens, device=device) * 293 + 2047) if layers > 4 else torch.arange(tokens, device=device)
    )
    if fp32_master:
        cast = {name: value.to(native[name].dtype) for name, value in parameters.items()}
        actual = torch.func.functional_call(model, cast, (ids[None],), {"position_ids": positions[None]}).logits[0]
    else:
        actual = model(ids[None], position_ids=positions[None]).logits[0]
    reference_cast = {name: value.to(original_types[name].dtype) for name, value in reference.items()}
    expected = reference_logits(reference_cast, ids, model.config, positions, expert_batching="grouped")
    logits_error = tensor_error("logits", actual, expected)
    assert_bf16_error(logits_error, absolute=2e-6)
    if objective == "random":
        upstream = torch.randn(actual.shape, generator=generator, dtype=torch.float32).to(device)
        actual_loss, reference_loss = (actual * upstream).sum(), (expected * upstream).sum()
    else:
        actual_loss = F.cross_entropy(actual[:-1], ids[1:])
        reference_loss = F.cross_entropy(expected[:-1], ids[1:])
    actual_loss.backward()
    reference_loss.backward()
    gradients = model.state_dict_adapter.to_hf({n: p.grad for n, p in parameters.items() if p.grad is not None})
    errors, frozen, strict_failures = [], 0, 0
    for name, value in source.items():
        other = reference[name]
        if not value.requires_grad:
            assert name not in gradients and other.grad is None, name
            frozen += 1
            continue
        assert name in gradients, name
        # The selected-expert oracle can omit an unused expert completely.
        reference_gradient = torch.zeros_like(other) if other.grad is None else other.grad
        error = tensor_error(name, gradients[name], reference_gradient)
        assert_bf16_error(error)
        errors.append(error)
        strict_failures += not torch.allclose(
            gradients[name].float(), reference_gradient.float(), atol=1.6e-5, rtol=4e-5
        )
    global_relative = math.sqrt(
        sum(e.squared_error for e in errors) / max(sum(e.reference_squared_norm for e in errors), 1e-30)
    )
    assert global_relative <= 0.005, global_relative

    # Update independent FP32 master copies with the observed gradients. This
    # measures optimizer deltas, not total weights that could hide a bad update.
    left, right, before = {}, {}, {}
    for name, value in source.items():
        if not value.requires_grad:
            continue
        before[name] = value.detach().float().clone()
        left[name] = torch.nn.Parameter(before[name].clone())
        right[name] = torch.nn.Parameter(before[name].clone())
        left[name].grad = gradients[name].float().clone()
        gradient = reference[name].grad
        right[name].grad = torch.zeros_like(right[name]) if gradient is None else gradient.float().clone()
    torch.optim.SGD(left.values(), lr=0.01).step()
    torch.optim.SGD(right.values(), lr=0.01).step()
    updates = tuple(tensor_error(name, left[name] - value, right[name] - value) for name, value in before.items())
    for error in updates:
        assert_bf16_error(error, absolute=1.6e-7)
    update_relative = math.sqrt(
        sum(e.squared_error for e in updates) / max(sum(e.reference_squared_norm for e in updates), 1e-30)
    )
    assert update_relative <= 0.005, update_relative
    return BF16Comparison(
        seed,
        tokens,
        layers,
        objective,
        fp32_master,
        logits_error,
        tuple(errors),
        updates,
        abs(actual_loss.item() - reference_loss.item()),
        global_relative,
        update_relative,
        strict_failures,
        frozen,
    )


@pytest.mark.parametrize("seed,tokens", [(101, 7), (211, 17), (307, 31)])
@pytest.mark.parametrize("objective,fp32_master", [("random", False), ("ce", True)])
def test_bf16_held_out_gradients_and_master_updates(
    seed: int, tokens: int, objective: Literal["random", "ce"], fp32_master: bool
) -> None:
    compare_bf16_gradients(
        seed=seed, tokens=tokens, objective=objective, fp32_master=fp32_master, device=torch.device("cpu")
    )


def test_layer_discovery_uses_decoder_stack_when_experts_outnumber_layers() -> None:
    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(tiny_config(n_routed_experts=128))
    assert get_model_layer_groups(model) == {"language": list(model.model.layers)}
    assert model.model.layers[1].mlp.experts.gate_and_up_projs.shape[0] == 128


def test_unsharded_thd_selects_model_owned_batch_preparation() -> None:
    with torch.device("meta"):
        model = MistralLarge4ForCausalLM(tiny_config())
    assert "cp_sharder" in model.prepare_model_inputs_for_cp({"qkv_format": "thd"})


def test_cp_rejects_invalid_format_and_chunk_layouts() -> None:
    from nemo_automodel.components.models.mistral_large4.context_parallel import prepare_cp_inputs

    with pytest.raises(ValueError, match="BSHD"):
        prepare_cp_inputs({"qkv_format": "unsupported"})
    with pytest.raises(ValueError, match="num_chunks >= 1"):
        prepare_cp_inputs({}, num_chunks=0)
