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

from copy import deepcopy
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, MistralConfig, MistralForCausalLM

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.utils import prepare_lm_weight
from nemo_automodel.recipes.llm.train_ft import (
    TrainFinetuneRecipeForNextTokenPrediction,
    _maybe_downgrade_loss_fn,
)


@pytest.fixture
def ddp_group(tmp_path):
    dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _mistral():
    return MistralForCausalLM(
        MistralConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            use_cache=False,
            attention_dropout=0.0,
        )
    )


def _gpt_neox():
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            use_cache=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
    )


@pytest.mark.parametrize("model_factory", [_mistral, _gpt_neox])
def test_ddp_preserves_chunked_ce_capability(ddp_group, model_factory):
    model = model_factory()
    loss = ChunkedCrossEntropy(4, compile=False)
    assert _maybe_downgrade_loss_fn(loss, model, False) is loss
    wrapped = DDP(model)
    assert _maybe_downgrade_loss_fn(loss, wrapped, False) is loss
    assert prepare_lm_weight(loss, wrapped) is model.get_output_embeddings().weight


class _UnsupportedModel(nn.Linear):
    def forward(self, input_ids, **kwargs):
        """Accept generic kwargs without implementing selective logits.

        Args:
            input_ids: Tensor of shape [batch, sequence, hidden].
            **kwargs: Unused keyword arguments.

        Returns:
            Tensor of shape [batch, sequence, vocab].
        """
        return super().forward(input_ids)


def test_ddp_does_not_invent_logits_to_keep_support(ddp_group):
    model = _UnsupportedModel(16, 32, bias=False)
    # An arbitrary .module child is not a transparent wrapper contract.
    model.module = _mistral()
    for probe in (model, DDP(model)):
        with pytest.raises(ValueError, match="supporting logits_to_keep"):
            _maybe_downgrade_loss_fn(ChunkedCrossEntropy(compile=False), probe, False)


@pytest.mark.parametrize("invalid", ["bias", "softcap"])
def test_ddp_preserves_chunked_ce_head_validation(ddp_group, invalid):
    model = _mistral()
    if invalid == "bias":
        model.lm_head = nn.Linear(16, 32, bias=True)
        message = "plain bias-free"
    else:
        # Use a real config with a declared softcapping field.
        from transformers import Gemma2Config

        model.config = Gemma2Config(final_logit_softcapping=30.0)
        message = "final_logit_softcapping"
    wrapped = DDP(model)
    loss = ChunkedCrossEntropy(compile=False)
    with pytest.raises(ValueError, match=message):
        _maybe_downgrade_loss_fn(loss, wrapped, False)
    with pytest.raises(ValueError, match=message):
        prepare_lm_weight(loss, wrapped)


def _check_ddp_chunked_ce_training(rank, world_size, model_factory):
    torch.manual_seed(17)
    model = model_factory()
    reference = deepcopy(model)
    wrapped = DDP(model)
    loss = ChunkedCrossEntropy(4, compile=False)
    assert _maybe_downgrade_loss_fn(loss, wrapped, False) is loss
    recipe = SimpleNamespace(
        model_parts=[wrapped],
        loss_fn=loss,
        dist_env=SimpleNamespace(device=torch.device("cpu")),
        pp_enabled=False,
        device_mesh=None,
        tokenizer=None,
        domain_mixture=None,
        te_fp8=None,
        distributed_config=SimpleNamespace(defer_fsdp_grad_sync=True),
        _get_cp_group_size=lambda: 1,
        _get_dp_group_size=lambda **kw: world_size,
        _get_dp_group=lambda **kw: dist.group.WORLD,
    )
    forwarded = []

    def record_forward(module, args, kwargs):
        """Record DDP invocation without changing tensors.

        Args:
            module: DDP wrapper.
            args: Empty positional arguments.
            kwargs: Forward arguments including input_ids of shape [batch, sequence].
        """
        forwarded.append(kwargs["logits_to_keep"])

    handle = wrapped.register_forward_pre_hook(record_forward, with_kwargs=True)
    optimizer = torch.optim.SGD(wrapped.parameters(), lr=0.1)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    try:
        # A second iteration also exercises DDP's reducer completion checks.
        for _ in range(2):
            input_ids = torch.randint(32, (world_size, 2, 8))
            labels = torch.randint(32, (world_size, 2, 8))
            # Unequal valid-token counts catch per-rank normalization mistakes.
            for batch_index in range(world_size):
                labels[batch_index, :, : batch_index + 1] = -100
            losses = []
            for microbatch in range(2):
                TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step(
                    recipe,
                    microbatch,
                    {
                        "input_ids": input_ids[rank, microbatch : microbatch + 1],
                        "labels": labels[rank, microbatch : microbatch + 1],
                    },
                    loss_buffer=losses,
                    num_label_tokens=int(labels.ne(-100).sum()),
                    num_batches=2,
                )
            ref_loss = nn.functional.cross_entropy(
                reference(input_ids.flatten(0, 1)).logits.flatten(0, 1), labels.flatten()
            )
            ref_loss.backward()
            total_loss = torch.stack(losses).sum()
            dist.all_reduce(total_loss)
            torch.testing.assert_close(total_loss, ref_loss)
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter.grad, ref_parameter.grad, rtol=1e-4, atol=2e-6)
            optimizer.step()
            ref_optimizer.step()
            for parameter, ref_parameter in zip(model.parameters(), reference.parameters()):
                torch.testing.assert_close(parameter, ref_parameter, rtol=1e-4, atol=2e-6)
            optimizer.zero_grad()
            ref_optimizer.zero_grad()
    finally:
        handle.remove()
    assert forwarded == [1, 1, 1, 1]


@pytest.mark.parametrize("model_factory", [_mistral, _gpt_neox])
def test_ddp_chunked_ce_trains_through_wrapper(ddp_group, model_factory):
    _check_ddp_chunked_ce_training(rank=0, world_size=1, model_factory=model_factory)


def _run_ddp_chunked_ce_worker(rank, init_method):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        _check_ddp_chunked_ce_training(rank=rank, world_size=2, model_factory=_mistral)
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="Two spawned CPU/Gloo workers import the training stack and verify real DDP gradient reductions.",
)
def test_ddp_chunked_ce_two_rank_parity(tmp_path):
    torch.multiprocessing.spawn(
        _run_ddp_chunked_ce_worker,
        args=((tmp_path / "two_rank_rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize("supported", [False, True])
def test_ddp_fused_loss_capability(ddp_group, supported):
    model = _mistral() if supported else _UnsupportedModel(16, 32, bias=False)
    loss = FusedLinearCrossEntropy(ignore_index=0)
    result = _maybe_downgrade_loss_fn(loss, DDP(model), False)
    if supported:
        assert result is loss
    else:
        assert isinstance(result, MaskedCrossEntropy)
        assert result.ignore_index == 0
