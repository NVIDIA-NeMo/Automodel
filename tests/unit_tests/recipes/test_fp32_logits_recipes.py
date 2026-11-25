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

"""Recipes training a ``model.logits_dtype: float32`` model with linear losses.

Linear losses recompute the LM-head projection from hidden states, so each recipe
must project the way the FP32-logits head does. Each case compares the recipe's
cross-entropy with the cross-entropy of the model's own FP32 logits.
"""

import copy
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import LlamaConfig

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.kd_loss import KDLoss
from nemo_automodel.components.models.common.lm_head import LMHead
from nemo_automodel.recipes.llm import kd as llm_kd
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

pytestmark = pytest.mark.cpu_only

_VOCAB = 128


@pytest.fixture(autouse=True)
def _cpu_current_device(monkeypatch):
    # Build on the CPU on GPU machines too: setup checks the CUDA device when CUDA is available.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))


def _fp32_logits_llama() -> nn.Module:
    """Build a one-layer BF16 Llama whose LM head writes FP32 logits."""
    config = LlamaConfig(
        architectures=["LlamaForCausalLM"],
        vocab_size=_VOCAB,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    # Eager RMSNorm avoids a cold Inductor compile that the head does not depend on.
    model = NeMoAutoModelForCausalLM.from_config(
        config, backend={"attn": "sdpa", "rms_norm": "torch"}, torch_dtype=torch.bfloat16, logits_dtype="float32"
    )
    assert isinstance(model.lm_head, LMHead) and model.lm_head.logits_dtype == torch.float32
    return model


def _batch() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Build one microbatch with partly unsupervised labels.

    Returns:
        Token ids and targets, each of shape [batch, sequence] with -100 marking
        unsupervised targets, and the count of supervised targets.
    """
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(0, _VOCAB, (2, 12), generator=generator)
    labels = torch.randint(0, _VOCAB, (2, 12), generator=generator)
    labels[0, :3] = -100
    return input_ids, labels, int((labels != -100).sum())


def _assert_matches_fp32_logits_ce(
    loss: torch.Tensor, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, num_label_tokens: int
) -> torch.Tensor:
    """Check a recipe's cross-entropy against the model's own FP32 logits.

    Args:
        loss: FP32 scalar summed over supervised tokens and divided by ``num_label_tokens``.
        model: Unchanged copy of the trained model, with an FP32-logits head.
        input_ids: Token ids of shape [batch, sequence].
        labels: Targets of shape [batch, sequence]; -100 marks unsupervised tokens.
        num_label_tokens: Count of supervised targets in ``labels``.

    Returns:
        The reference FP32 scalar loss, which retains its autograd graph.
    """
    output = model(input_ids=input_ids, output_hidden_states=True)
    assert output.logits.dtype == torch.float32
    expected = F.cross_entropy(output.logits.flatten(0, 1), labels.flatten(), reduction="sum") / num_label_tokens
    torch.testing.assert_close(loss, expected, rtol=1e-6, atol=1e-6)
    # Projecting in the BF16 weight dtype, as a plain head would, rounds every logit.
    with torch.no_grad():
        bf16_logits = F.linear(output.hidden_states[-1], model.lm_head.weight).float()
        bf16_loss = F.cross_entropy(bf16_logits.flatten(0, 1), labels.flatten(), reduction="sum") / num_label_tokens
    assert (loss - expected).abs() * 10 < (bf16_loss - expected).abs()
    return expected


def _bf16_projection_gradients(
    model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor, num_label_tokens: int
) -> dict[str, torch.Tensor]:
    """Backpropagate the cross-entropy of logits projected in the BF16 weight dtype, as a plain head would.

    Args:
        model: Unchanged copy of the trained model, whose parameters receive the gradients.
        input_ids: Token ids of shape [batch, sequence].
        labels: Targets of shape [batch, sequence]; -100 marks unsupervised tokens.
        num_label_tokens: Count of supervised targets in ``labels``.

    Returns:
        Each parameter's gradient by name, in the parameter's shape and dtype.
    """
    hidden = model(input_ids=input_ids, output_hidden_states=True).hidden_states[-1]
    logits = F.linear(hidden, model.lm_head.weight).float()
    (F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum") / num_label_tokens).backward()
    return {name: parameter.grad for name, parameter in model.named_parameters()}


def test_train_ft_step_projects_chunked_ce_like_the_fp32_logits_head():
    model = _fp32_logits_llama()
    reference = copy.deepcopy(model)
    control = copy.deepcopy(model)
    recipe = SimpleNamespace(
        model_parts=[model],
        loss_fn=ChunkedCrossEntropy(8, compile=False),
        dist_env=SimpleNamespace(device=torch.device("cpu")),
        pp_enabled=False,
        device_mesh=None,
        tokenizer=None,
        domain_mixture=None,
        te_quantization=None,
        distributed_config=SimpleNamespace(defer_fsdp_grad_sync=True),
        _get_cp_group_size=lambda: 1,
        _get_dp_group_size=lambda **kw: 1,
        _get_dp_group=lambda **kw: None,
    )
    input_ids, labels, num_label_tokens = _batch()
    losses = []
    head_calls = []
    model.lm_head.register_forward_pre_hook(lambda module, args: head_calls.append(module))

    TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step(
        recipe,
        0,
        {"input_ids": input_ids, "labels": labels},
        loss_buffer=losses,
        num_label_tokens=num_label_tokens,
        num_batches=1,
    )

    # The model can skip its logits, so the step never runs the head; the loss projects the hidden states.
    assert head_calls == []
    expected = _assert_matches_fp32_logits_ce(losses[0], reference, input_ids, labels, num_label_tokens)
    expected.backward()
    control_grads = _bf16_projection_gradients(control, input_ids, labels, num_label_tokens)
    for (name, parameter), reference_parameter in zip(model.named_parameters(), reference.parameters()):
        reference_grad = reference_parameter.grad.float()
        error = (parameter.grad.float() - reference_grad).norm()
        if name == "lm_head.weight":
            # Chunks round their BF16 head-weight gradients before the FP32 cross-chunk sum.
            assert error <= 1e-2 * reference_grad.norm(), name
        else:
            # A backward that projected in BF16 would be as far off as the control.
            assert error * 10 < (control_grads[name].float() - reference_grad).norm(), name


def test_vlm_validation_projects_chunked_ce_like_the_fp32_logits_head(monkeypatch):
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device=None: 0)
    model = _fp32_logits_llama()
    reference = copy.deepcopy(model)
    recipe = SimpleNamespace(
        model_parts=[model],
        loss_fn=ChunkedCrossEntropy(8, compile=False),
        dist_env=SimpleNamespace(device=torch.device("cpu")),
        device_mesh=None,
        pp_enabled=False,
        step_scheduler=SimpleNamespace(step=3, epoch=1),
        optimizer=[SimpleNamespace(param_groups=[{"lr": 1e-3}])],
        _cp_vision_frame_sharding_context=nullcontext,
        _maybe_add_drafter_loss=lambda *, out, base_loss, labels, model, num_label_tokens: base_loss,
        _dp_allreduce=lambda tensor, include_cp=False: tensor,
    )
    input_ids, labels, num_label_tokens = _batch()
    head_calls = []
    model.lm_head.register_forward_pre_hook(lambda module, args: head_calls.append(module))

    metrics = FinetuneRecipeForVLM._run_validation_epoch(recipe, [{"input_ids": input_ids, "labels": labels}]).metrics

    # The model skips its logits, so validation never runs the head; the loss projects the hidden states.
    assert head_calls == []
    assert metrics["num_label_tokens"] == num_label_tokens
    _assert_matches_fp32_logits_ce(torch.tensor(metrics["val_loss"]), reference, input_ids, labels, num_label_tokens)


class _OneHotTeacher(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        """Return one-hot teacher logits.

        Args:
            input_ids: Token ids of shape [batch, sequence].

        Returns:
            Namespace whose ``logits`` is an FP32 tensor of shape [batch, sequence, vocab].
        """
        return SimpleNamespace(logits=F.one_hot(input_ids, num_classes=_VOCAB).float())


def test_kd_step_projects_chunked_ce_like_the_fp32_logits_head(monkeypatch):
    def no_op_sharder(model, mesh, batch, **kwargs):
        """Return a context-parallel sharder that leaves the batch unchanged.

        Args:
            model: Student model under test.
            mesh: Unused device mesh.
            batch: Mapping of tensors of shape [batch, sequence].
            **kwargs: Unused context-parallel options.

        Returns:
            Object whose ``shard`` returns a null context and the unchanged batch.
        """
        del model, mesh, kwargs
        return SimpleNamespace(shard=lambda actual_batch: (nullcontext, actual_batch))

    monkeypatch.setattr(llm_kd, "ContextParallelSharder", no_op_sharder)
    student = _fp32_logits_llama()
    reference = copy.deepcopy(student)
    recipe = object.__new__(llm_kd.KnowledgeDistillationRecipeForNextTokenPrediction)
    recipe.dist_env = SimpleNamespace(device="cpu")
    recipe.device_mesh = None
    recipe.pp_enabled = False
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.model_parts = [student]
    recipe.teacher_model = _OneHotTeacher()
    recipe.loss_fn = ChunkedCrossEntropy(8, compile=False)
    recipe.kd_loss_fn = KDLoss()
    recipe.kd_ratio = 0.5
    recipe._offload_teacher_model = False
    recipe.te_quantization = None
    recipe.separate_meshes = False
    input_ids, labels, num_label_tokens = _batch()

    _, _, ce_loss = recipe._forward_backward_step(
        0,
        {"input_ids": input_ids, "labels": labels},
        num_label_tokens=num_label_tokens,
        num_batches=1,
        is_train=False,
    )

    _assert_matches_fp32_logits_ce(ce_loss, reference, input_ids, labels, num_label_tokens)
