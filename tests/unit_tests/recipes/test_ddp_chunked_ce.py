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
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import MistralConfig, MistralForCausalLM

from nemo_automodel.components.loss.chunked_ce import ChunkedCrossEntropy
from nemo_automodel.components.utils.model_utils import _supports_logits_to_keep
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


def test_ddp_preserves_chunked_ce_capability(ddp_group):
    model = _mistral()
    loss = ChunkedCrossEntropy(4, compile=False)
    assert _supports_logits_to_keep(model)
    assert _maybe_downgrade_loss_fn(loss, model, False) is loss
    wrapped = DDP(model)
    assert _supports_logits_to_keep(wrapped)
    assert _maybe_downgrade_loss_fn(loss, wrapped, False) is loss


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
        assert not _supports_logits_to_keep(probe)
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
    with pytest.raises(ValueError, match=message):
        _maybe_downgrade_loss_fn(ChunkedCrossEntropy(compile=False), DDP(model), False)


def test_ddp_chunked_ce_trains_through_wrapper(ddp_group):
    torch.manual_seed(17)
    model = _mistral()
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
        _get_dp_group_size=lambda **kw: 1,
        _get_dp_group=lambda **kw: None,
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
            input_ids = torch.randint(32, (1, 8))
            labels = torch.randint(32, (1, 8))
            losses = []
            TrainFinetuneRecipeForNextTokenPrediction._forward_backward_step(
                recipe,
                0,
                {"input_ids": input_ids, "labels": labels},
                loss_buffer=losses,
                num_label_tokens=8,
                num_batches=1,
            )
            ref_loss = nn.functional.cross_entropy(reference(input_ids).logits.flatten(0, 1), labels.flatten())
            ref_loss.backward()
            torch.testing.assert_close(losses[0], ref_loss)
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
    assert forwarded == [1, 1]
