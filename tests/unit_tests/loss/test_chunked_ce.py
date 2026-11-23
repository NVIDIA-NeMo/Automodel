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
    batched = hidden.unsqueeze(0)  # a packed forward returns [1, tokens, hidden]
    logits = model.lm_head(batched)
    actual = calculate_loss(
        ChunkedCrossEntropy(3, compile=False, reduction=reduction),
        model=model,
        hidden_states=batched,
        logits=logits,
        labels=labels,
    )
    expected = F.cross_entropy(logits.squeeze(0), labels, reduction=reduction)
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
                30, hard_timeout=120, reason="Cold Inductor compilation of mixed-dtype projection and backward"
            ),
        ),
    ],
)
@pytest.mark.parametrize("rounding", [None, "model_output", "head"])
def test_fp32_head_bf16_hidden_matches_model_projection(rounding, compiled):
    from nemo_automodel.components.models.common.lm_head import LMHead

    torch.manual_seed(67)
    model = torch.nn.Module()
    # An FP32 head may round its own logits to bf16 (logits_dtype) ...
    model.lm_head = LMHead(19, 37, logits_dtype=torch.bfloat16 if rounding == "head" else None)
    hidden = torch.randn(2, 9, 19, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(37, (2, 9))
    logits = model.lm_head(hidden)
    # ... or the model may round them at its output (for example an FSDP output_dtype).
    logits = logits.to(hidden.dtype) if rounding == "model_output" else logits
    actual = calculate_loss(
        ChunkedCrossEntropy(4, compile=compiled),
        model=model,
        hidden_states=hidden,
        logits=logits,
        labels=labels,
    )
    expected = F.cross_entropy(logits.float().flatten(0, 1), labels.flatten(), reduction="sum")
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


@pytest.mark.parametrize("chunk_len", [4, 64])
def test_fp32_projection_reproduces_the_fp32_logits_head(chunk_len):
    from nemo_automodel.components.models.common.lm_head import LMHead

    torch.manual_seed(71)
    model = torch.nn.Module()
    model.lm_head = LMHead(24, 53, logits_dtype=torch.float32, dtype=torch.bfloat16)
    hidden = torch.randn(2, 9, 24, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(53, (2, 9))
    labels[0, 3] = -100
    logits = model.lm_head(hidden)
    expected = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum")

    actual = calculate_loss(
        ChunkedCrossEntropy(chunk_len, compile=False),
        model=model,
        hidden_states=hidden,
        logits=logits,
        labels=labels,
    )

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-5)
    actual_grads = torch.autograd.grad(actual, (hidden, model.lm_head.weight), retain_graph=True)
    expected_grads = torch.autograd.grad(expected, (hidden, model.lm_head.weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        assert actual_grad.dtype == torch.bfloat16
        # Chunks round their BF16 weight gradients before the FP32 cross-chunk sum.
        assert (actual_grad.float() - expected_grad.float()).norm() <= 1e-2 * expected_grad.float().norm()
    reference = F.cross_entropy(
        (hidden.double() @ model.lm_head.weight.double().t()).flatten(0, 1), labels.flatten(), reduction="sum"
    )
    bf16_projection = ChunkedCrossEntropy(chunk_len, compile=False)(hidden, labels, model.lm_head.weight)
    assert (actual.double() - reference).abs() * 10 < (bf16_projection.double() - reference).abs()
    # A backward that projected in BF16 would put the input gradient as far off as the control's.
    (bf16_hidden_grad,) = torch.autograd.grad(bf16_projection, (hidden,))
    hidden_error = (actual_grads[0].float() - expected_grads[0].float()).norm()
    assert hidden_error * 10 < (bf16_hidden_grad.float() - expected_grads[0].float()).norm()


def test_compiled_chunks_keep_the_fp32_projection():
    from nemo_automodel.components.models.common.lm_head import LMHead

    # ChunkedCrossEntropy compiles its per-chunk loss by default.
    torch.manual_seed(71)
    weight = (torch.randn(53, 24) * 0.1).bfloat16().requires_grad_()
    hidden = torch.randn(2, 9, 24, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(53, (2, 9))
    labels[0, 3] = -100
    # Chunks project with the gathered weight, so the head's own parameters stay on meta.
    head = LMHead(24, 53, logits_dtype=torch.float32, device="meta")
    results = {}
    for use_compile in (False, True):
        loss = ChunkedCrossEntropy(4, compile=use_compile)(hidden, labels, weight, lm_head=head)
        results[use_compile] = (loss, *torch.autograd.grad(loss, (hidden, weight)))

    for compiled, eager in zip(results[True], results[False]):
        assert compiled.dtype == eager.dtype
        torch.testing.assert_close(compiled, eager)
    reference = F.cross_entropy(
        (hidden.double() @ weight.double().t()).flatten(0, 1), labels.flatten(), reduction="sum"
    )
    bf16_projection = ChunkedCrossEntropy(4, compile=False)(hidden, labels, weight)
    assert (results[True][0].double() - reference).abs() * 10 < (bf16_projection.double() - reference).abs()


def test_compiled_chunks_round_a_softcapped_head_once():
    from nemo_automodel.components.models.common.lm_head import LMHead

    torch.manual_seed(72)
    weight = (torch.randn(53, 24) * 0.1).bfloat16()
    hidden = torch.randn(2, 9, 24, dtype=torch.bfloat16)
    labels = torch.randint(53, (2, 9))
    head = LMHead(24, 53, logits_dtype=torch.bfloat16, logit_softcapping=2.0, device="meta")
    fp32_logits = LMHead.linear(hidden, weight, logits_dtype=torch.float32)
    once = (2.0 * torch.tanh(fp32_logits / 2.0)).bfloat16()
    expected = F.cross_entropy(once.float().flatten(0, 1), labels.flatten(), reduction="sum")

    for use_compile in (False, True):
        loss = ChunkedCrossEntropy(4, compile=use_compile)(hidden, labels, weight, lm_head=head)
        torch.testing.assert_close(loss, expected, msg=f"compile={use_compile}")


def test_validate_lm_head_accepts_plain_and_fp32_logits_heads():
    from nemo_automodel.components.models.common.lm_head import LMHead

    class TransformedHead(torch.nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Scale the projection, which chunks cannot reproduce.

            Args:
                input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

            Returns:
                Tensor of shape [..., vocab], twice the plain projection.
            """
            return 2 * super().forward(input)

    class InputOnlyHead(LMHead):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            """Project like an LMHead through an overridden forward, which chunks cannot see.

            Args:
                input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

            Returns:
                Tensor of shape [..., vocab].
            """
            return super().forward(input)

    plain = torch.nn.Linear(4, 8, bias=False)
    fp32_head = LMHead(4, 8, logits_dtype=torch.float32)
    loss_fn = ChunkedCrossEntropy(4, compile=False)

    assert loss_fn.validate_lm_head(plain) is plain
    assert loss_fn.validate_lm_head(fp32_head) is fp32_head
    for head in (TransformedHead(4, 8, bias=False), InputOnlyHead(4, 8), LMHead(4, 8, bias=True)):
        with pytest.raises(ValueError, match="plain bias-free nn.Linear"):
            loss_fn.validate_lm_head(head)


def test_validate_lm_head_accepts_only_a_softcap_the_head_applies():
    from transformers import PretrainedConfig

    from nemo_automodel.components.models.common.lm_head import LMHead

    config = PretrainedConfig(final_logit_softcapping=30.0)
    loss_fn = ChunkedCrossEntropy(4, compile=False)
    owned = LMHead(4, 8, logit_softcapping=30.0)

    assert loss_fn.validate_lm_head(owned, model_config=config) is owned
    # A model that softcaps after a head that does not would make every chunk wrong.
    for head in (torch.nn.Linear(4, 8, bias=False), LMHead(4, 8), LMHead(4, 8, logit_softcapping=20.0)):
        with pytest.raises(ValueError, match="final_logit_softcapping applied outside the LM head"):
            loss_fn.validate_lm_head(head, model_config=config)


def test_mtp_depths_project_like_the_fp32_logits_head():
    from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
    from nemo_automodel.components.loss.mtp import calculate_mtp_loss
    from nemo_automodel.components.models.common.lm_head import LMHead

    torch.manual_seed(73)
    bf16_head = torch.nn.Linear(24, 53, bias=False, dtype=torch.bfloat16)
    model, plain = torch.nn.Module(), torch.nn.Module()
    plain.lm_head = torch.nn.Linear(24, 53, bias=False, dtype=torch.bfloat16)
    plain.lm_head.weight = bf16_head.weight
    model.lm_head = LMHead.from_linear(bf16_head, logits_dtype=torch.float32)
    depths = [torch.randn(2, 9, 24, dtype=torch.bfloat16) for _ in range(2)]
    labels = torch.randint(53, (2, 9))

    def mtp_loss(loss_fn, owner):
        return calculate_mtp_loss(
            loss_fn, mtp_per_depth_h=depths, labels=labels, model=owner, scaling_factor=1.0, ignore_index=-100
        )

    # MaskedCrossEntropy consumes the head's own fp32 logits for every depth.
    expected = mtp_loss(MaskedCrossEntropy(), model)
    torch.testing.assert_close(mtp_loss(ChunkedCrossEntropy(4, compile=False), model), expected, rtol=1e-6, atol=1e-5)
    bf16_loss = mtp_loss(ChunkedCrossEntropy(4, compile=False), plain)
    assert (bf16_loss - expected).abs() > 1e-4


@pytest.mark.parametrize("recipe", ["llm.train_ft", "vlm.finetune"])
def test_recipe_setup_accepts_an_fp32_logits_head_for_chunked_ce(recipe):
    import importlib

    from nemo_automodel.components.models.common.lm_head import LMHead

    _maybe_downgrade_loss_fn = importlib.import_module(f"nemo_automodel.recipes.{recipe}")._maybe_downgrade_loss_fn

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = LMHead(7, 13, logits_dtype=torch.float32, dtype=torch.bfloat16)

        def forward(self, input_ids, logits_to_keep=0):
            """Fail if setup runs the model.

            Args:
                input_ids: Token IDs of shape [batch, sequence].
                logits_to_keep: Number of token positions to project.
            """
            raise AssertionError("setup must not run the model")

    loss_fn = ChunkedCrossEntropy(compile=False)
    assert _maybe_downgrade_loss_fn(loss_fn, Model(), False) is loss_fn
