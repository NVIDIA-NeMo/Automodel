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


from collections.abc import Callable

import pytest
import torch

from nemo_automodel.components.models.hy_v4_flash.attention import HYV4FlashRope, HYV4FlashSparsePlan
from nemo_automodel.components.models.hy_v4_flash.model import HYV4FlashForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_ac
from tests.unit_tests.models.hy_v4_flash.conftest import CPU_BACKEND, TOKEN_RANGE, tiny_config


def test_tiny_forward_backward_is_finite() -> None:
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
    model.initialize_weights(dtype=torch.float32)
    ids = torch.randint(*TOKEN_RANGE, (2, 24))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def _train_step(mode: str | None) -> tuple[torch.Tensor, dict[str, torch.Tensor], HYV4FlashForCausalLM]:
    """Loss and gradients of one training step, plain (``None``) or under a checkpointing mode; selection runs."""
    ids = torch.randint(*TOKEN_RANGE, (2, 40), generator=torch.Generator().manual_seed(1))
    torch.manual_seed(0)
    model = HYV4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
    assert ids.shape[1] > model.config.text_config.index_topk
    chunked_calls: list[bool] = []
    for layer in model.model.layers:
        indexer = layer.self_attn.compressed_indexer
        if indexer is None:
            continue
        indexer.score_block_bytes = 1  # one query per score chunk
        original = indexer.select

        def select(
            q_resid: torch.Tensor,
            hidden_states: torch.Tensor,
            query_rope: HYV4FlashRope,
            group_keys: torch.Tensor,
            group_rope: HYV4FlashRope,
            first_query_slot: int,
            original: Callable[..., HYV4FlashSparsePlan] = original,
        ) -> HYV4FlashSparsePlan:
            chunked_calls.append(q_resid.shape[0] > 1)
            return original(q_resid, hidden_states, query_rope, group_keys, group_rope, first_query_slot)

        indexer.select = select
    if mode == "moe":
        apply_ac(model, ignore_router=True, moe_only=True)
    elif mode == "selective":
        apply_ac(model, selective=True)
    torch.manual_seed(0)
    model.initialize_weights(dtype=torch.float32)
    model.train()
    out = model(ids, labels=ids)
    out.loss.backward()
    assert chunked_calls and all(chunked_calls)
    # A checkpoint wrapper inserts its own level into parameter names; compare under the plain names.
    grads = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return out.loss.detach(), grads, model


@pytest.mark.parametrize("mode", ["moe", "selective"])
def test_activation_checkpointing_modes_train_like_the_plain_model(mode: str) -> None:
    """Both partial checkpointing modes train like the plain model.

    ``moe`` wraps each MoE (not a ``MoE`` instance any more); ``selective`` caches matmul / bmm results, which the
    model must not mutate in place.
    """
    plain_loss, plain_grads, _ = _train_step(None)
    loss, grads, model = _train_step(mode)
    if mode == "moe":
        wrapped = [type(layer.mlp).__name__ for layer in model.model.layers]
        assert wrapped.count("CheckpointWrapper") == model.config.text_config.mlp_layer_types.count("sparse") > 0
    assert torch.equal(plain_loss, loss)
    assert plain_grads.keys() == grads.keys() and len(plain_grads) > 0
    for name, grad in plain_grads.items():
        torch.testing.assert_close(grads[name], grad, rtol=1e-5, atol=1e-7, msg=lambda m, name=name: f"{name}: {m}")
