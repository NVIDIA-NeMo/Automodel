# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import pytest
import torch
import torch.nn.functional as F

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.loss import ChunkedCEConfig, build_loss_module
from nemo_automodel.components.loss.utils import calculate_loss


@pytest.mark.parametrize("chunk_len", [1, 5, 64])
@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
@pytest.mark.parametrize("train_hidden,train_weight", [(True, True), (True, False), (False, True)])
def test_loss_and_gradients(chunk_len, reduction, train_hidden, train_weight):
    torch.manual_seed(7)
    # Noncontiguous hidden states and an uneven final chunk.
    hidden = torch.randn(2, 9, 7).transpose(0, 1).detach().requires_grad_(train_hidden)
    weight = torch.randn(13, 7, requires_grad=train_weight)
    labels = torch.randint(0, 13, hidden.shape[:-1])
    labels[2] = -100
    mask = torch.ones_like(labels)
    mask[4] = 0
    originals = [x.detach().clone() for x in (hidden, weight, labels)]
    ref_hidden = hidden.detach().clone().requires_grad_(train_hidden)
    ref_weight = weight.detach().clone().requires_grad_(train_weight)
    actual = ChunkedCrossEntropy(chunk_len, compile=False, reduction=reduction)(hidden, labels, weight, mask=mask)
    expected = F.cross_entropy(
        F.linear(ref_hidden, ref_weight).flatten(0, 1),
        labels.masked_fill(mask == 0, -100).flatten(),
        reduction=reduction,
    )
    if reduction == "none":
        expected = expected.reshape(labels.shape)
    torch.testing.assert_close(actual, expected)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for tensor, reference, original in zip((hidden, weight), (ref_hidden, ref_weight), originals):
        if tensor.requires_grad:
            torch.testing.assert_close(tensor.grad, reference.grad, rtol=2e-5, atol=2e-6)
        else:
            assert tensor.grad is None
        torch.testing.assert_close(tensor, original, rtol=0, atol=0)
    torch.testing.assert_close(labels, originals[2], rtol=0, atol=0)


@pytest.mark.parametrize("empty", [False, True])
@pytest.mark.parametrize("normalizer", [0, 9])
def test_weighted_and_empty_loss(empty, normalizer):
    torch.manual_seed(9)
    hidden = torch.randn(11, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.full((11,), -7) if empty else torch.arange(11)
    labels[1:5] = -7  # A completely ignored chunk.
    loss_weights = torch.linspace(0.25, 2, 11)
    actual = ChunkedCrossEntropy(3, compile=False, ignore_index=-7)(
        hidden,
        labels,
        weight,
        num_label_tokens=normalizer,
        loss_weights=loss_weights,
    )
    expected = (
        F.cross_entropy(F.linear(hidden, weight), labels, ignore_index=-7, reduction="none") * loss_weights
    ).sum()
    expected = expected / normalizer if normalizer else expected * 0
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual * 2.3, (hidden, weight))
    expected_grads = torch.autograd.grad(expected * 2.3, (hidden, weight))
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("chunk_len", [-3, 0])
def test_invalid_chunk_len(chunk_len):
    with pytest.raises(ValueError, match="chunk_len must be greater than zero"):
        ChunkedCrossEntropy(chunk_len)


def test_routing_and_config():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(7, 13, bias=False)
    hidden = torch.randn(2, 5, 7, requires_grad=True)
    labels = torch.randint(0, 13, (2, 5))
    loss_fn = build_loss_module(ChunkedCEConfig(chunk_len=3, compile=False))
    # No logits are supplied: this must take the hidden-state route.
    actual = calculate_loss(loss_fn, hidden_states=hidden, labels=labels, model=model)
    expected = F.cross_entropy(model.lm_head(hidden).flatten(0, 1), labels.flatten(), reduction="sum")
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert model.lm_head.weight.grad is not None


def test_saves_no_vocabulary_activations():
    hidden = torch.randn(17, 7, requires_grad=True)
    weight = torch.randn(31, 7, requires_grad=True)
    labels = torch.randint(0, 31, (17,))
    saved = []

    def pack(tensor):
        """Record a saved tensor.

        Args:
            tensor: Autograd-saved tensor of arbitrary shape.

        Returns:
            The same tensor without mutation.
        """
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = ChunkedCrossEntropy(4, compile=False)(hidden, labels, weight)
    assert sum(x.numel() * x.element_size() for x in saved) <= sum(
        x.numel() * x.element_size() for x in (hidden, weight, labels)
    )
    loss.backward()


def test_double_backward_rejected():
    hidden = torch.randn(4, 7, requires_grad=True)
    weight = torch.randn(13, 7, requires_grad=True)
    labels = torch.arange(4)
    upstream = torch.ones((), requires_grad=True)
    grad = torch.autograd.grad(
        ChunkedCrossEntropy(2, compile=False)(hidden, labels, weight) * upstream,
        hidden,
        create_graph=True,
    )[0]
    with pytest.raises(RuntimeError, match="not have been used|does not require grad|differentiate twice"):
        torch.autograd.grad(grad.sum(), hidden)


def test_profiler_has_no_full_vocabulary_allocation():
    from torch.profiler import ProfilerActivity, profile

    # CPU profiler makes the allocation-shape contract runnable without a GPU.
    tokens, vocab, hidden_dim = 128, 1024, 16
    hidden = torch.randn(tokens, hidden_dim, requires_grad=True)
    weight = torch.randn(vocab, hidden_dim, requires_grad=True)
    labels = torch.randint(vocab, (tokens,))
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
        ChunkedCrossEntropy(8, compile=False)(hidden, labels, weight).backward()
    assert max(event.cpu_memory_usage for event in prof.events()) < tokens * vocab * 4


def test_pipeline_hidden_states_loss():
    from nemo_automodel.components.loss.mtp import PipelineCausalLMLoss

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(7, 13, bias=False)
    hidden = torch.randn(2, 11, 7, requires_grad=True)
    labels = torch.randint(13, (2, 11))
    loss_fn = ChunkedCrossEntropy(4, compile=False)
    actual = PipelineCausalLMLoss(loss_fn, model)(hidden, labels)
    expected = F.cross_entropy(model.lm_head(hidden).flatten(0, 1), labels.flatten(), reduction="sum")
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (hidden, model.lm_head.weight))
    expected_grads = torch.autograd.grad(expected, (hidden, model.lm_head.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("reduction", ["sum", "mean", "none"])
def test_glm_packing_to_loss(reduction):
    from types import SimpleNamespace

    from nemo_automodel.components.datasets.utils import packed_sequence_thd_collater
    from nemo_automodel.components.models.common.utils import compute_lm_head_logits
    from nemo_automodel.components.models.glm_moe_dsa.cp import make_glm_dsa_packed_cp_batch_and_ctx

    batch = packed_sequence_thd_collater(
        [
            {
                "input_ids": [1, 2, 3, 4],
                "labels": [2, -100, 4, 5],
                "position_ids": [0, 1, 0, 1],
                "seq_lens": [2, 2],
                "seq_lens_padded": [2, 2],
            },
        ]
    )
    mesh = SimpleNamespace(size=lambda: 1, get_group=lambda: None, get_local_rank=lambda: 0)
    _, packed = make_glm_dsa_packed_cp_batch_and_ctx(mesh, None, batch)
    labels = packed["labels"]
    assert labels.tolist() == [2, -100, 4, 5]
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(7, 13, bias=False)
    hidden = torch.randn(4, 7, requires_grad=True)
    out = compute_lm_head_logits(model.lm_head, hidden, is_thd=True, output_hidden_states=True)
    assert out.hidden_states.shape == (1, 4, 7)
    actual = calculate_loss(
        ChunkedCrossEntropy(3, compile=False, reduction=reduction),
        model=model,
        hidden_states=out.hidden_states,
        logits=out.logits,
        labels=labels,
    )
    expected = F.cross_entropy(out.logits.squeeze(0), labels, reduction=reduction)
    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual.sum(), (hidden, model.lm_head.weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, model.lm_head.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize(
    "compiled",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.runtime_budget(
                60, hard_timeout=120, reason="Cold Inductor compilation of mixed-dtype projection and backward"
            ),
        ),
    ],
)
@pytest.mark.parametrize("cast_logits", [False, True])
def test_fp32_head_bf16_hidden_matches_model_projection(cast_logits, compiled):
    from nemo_automodel.components.models.common.utils import compute_lm_head_logits

    torch.manual_seed(67)
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(19, 37, bias=False)
    hidden = torch.randn(2, 9, 19, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(37, (2, 9))
    out = compute_lm_head_logits(model.lm_head, hidden, fp32_lm_head=cast_logits, output_hidden_states=True)
    actual = calculate_loss(
        ChunkedCrossEntropy(4, compile=compiled),
        model=model,
        hidden_states=hidden,
        logits=out.logits,
        labels=labels,
    )
    expected = F.cross_entropy(out.logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
    torch.testing.assert_close(actual, expected)
    actual_grads = torch.autograd.grad(actual, (hidden, model.lm_head.weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, model.lm_head.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_main_and_mtp_retain_one_shared_mixed_dtype_head():
    from nemo_automodel.components.loss.mtp import calculate_mtp_loss
    from nemo_automodel.components.loss.utils import prepare_lm_weight

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(7, 13, bias=False)
    hidden = [torch.randn(1, 9, 7, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    labels = torch.randint(13, (1, 9))
    loss_fn = ChunkedCrossEntropy(4, compile=False)
    weight = prepare_lm_weight(loss_fn, model)
    saved_weights = []

    def pack(tensor):
        """Record the projection tensors retained by autograd.

        Args:
            tensor: Saved tensor of arbitrary shape; [vocab, hidden] weights are recorded.

        Returns:
            The input tensor unchanged, preserving shape, dtype and storage.
        """
        if tensor.shape == weight.shape:
            saved_weights.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        loss = calculate_loss(
            loss_fn, model=model, hidden_states=hidden[0], labels=labels, lm_weight=weight, logits_dtype=torch.bfloat16
        )
        loss = loss + calculate_mtp_loss(
            loss_fn,
            model=model,
            mtp_per_depth_h=hidden[1:],
            labels=labels,
            lm_weight=weight,
            logits_dtype=torch.bfloat16,
        )
    assert len(saved_weights) == 3
    assert {tensor.data_ptr() for tensor in saved_weights} == {weight.data_ptr()}
    assert weight.data_ptr() == model.lm_head.weight.data_ptr()
    # Dense reference includes the model-owned BF16 output cast at each depth.
    reference = F.cross_entropy(
        F.linear(hidden[0].float(), weight).bfloat16().float().flatten(0, 1), labels.flatten(), reduction="sum"
    )
    for depth, states in enumerate(hidden[1:], 1):
        targets = labels.roll(-depth, -1)
        targets[..., -depth:] = -100
        reference = reference + 0.05 * F.cross_entropy(
            F.linear(states.float(), weight).bfloat16().float().flatten(0, 1), targets.flatten(), reduction="sum"
        )
    torch.testing.assert_close(loss, reference)
    actual_grads = torch.autograd.grad(loss, (*hidden, weight), retain_graph=True)
    expected_grads = torch.autograd.grad(reference, (*hidden, weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)


def test_transformed_head_rejected_before_training():
    from nemo_automodel.components.models.baichuan.model import NormHead
    from nemo_automodel.recipes.llm.train_ft import _maybe_downgrade_loss_fn

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = NormHead(7, 13)

        def forward(self, input_ids, logits_to_keep=0):
            """Fail if setup incorrectly allows this transformed head.

            Args:
                input_ids: Token IDs of shape [batch, sequence].
                logits_to_keep: Number of token positions to project.
            """
            raise AssertionError("must fail during setup")

    model = Model().train()
    original_weight = model.lm_head.weight.detach().clone()
    with pytest.raises(ValueError, match="plain bias-free nn.Linear"):
        _maybe_downgrade_loss_fn(ChunkedCrossEntropy(compile=False), model, False)
    torch.testing.assert_close(model.lm_head.weight, original_weight)
    assert model.lm_head.weight.grad is None
