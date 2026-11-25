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


import torch

from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM

from .tiny import CPU_BACKEND, tiny_config


def test_tiny_forward_backward_is_finite():
    torch.manual_seed(0)
    model = HY4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
    model.initialize_weights(dtype=torch.float32)
    ids = torch.randint(3, 280, (2, 24))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_moe_only_activation_checkpointing_wraps_the_moe_and_trains():
    """``activation_checkpointing: moe`` replaces each block's MoE with a checkpoint wrapper before initialisation.

    The wrapper forwards attribute access and calls but is not a ``MoE`` instance, so a block must not classify its
    MLP by type: the wrapped model has to initialise, run and differentiate exactly like the unwrapped one.
    """
    from nemo_automodel.components.moe.parallelizer import apply_ac

    ids = torch.randint(3, 280, (2, 24), generator=torch.Generator().manual_seed(1))
    losses = []
    for wrap in (False, True):
        torch.manual_seed(0)
        model = HY4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
        if wrap:
            apply_ac(model, ignore_router=True, moe_only=True)
            wrapped = [type(layer.mlp).__name__ for layer in model.model.layers]
            assert wrapped.count("CheckpointWrapper") == model.config.text_config.mlp_layer_types.count("sparse") > 0
        torch.manual_seed(0)
        model.initialize_weights(dtype=torch.float32)
        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        losses.append(out.loss.detach())
    assert torch.equal(losses[0], losses[1])


def test_selective_activation_checkpointing_trains_like_the_plain_model():
    """Selective checkpointing caches matmul / bmm results, so the model must not mutate one in place.

    The indexer's score block is such a result. The documents are longer than ``index_topk`` and the scores are
    computed in several chunks, so the sparse selection runs under the wrapper.
    """
    from nemo_automodel.components.moe.parallelizer import apply_ac

    ids = torch.randint(3, 280, (2, 40), generator=torch.Generator().manual_seed(1))
    results = []
    for wrap in (False, True):
        torch.manual_seed(0)
        model = HY4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
        indexers = [m.self_attn.compressed_indexer for m in model.model.layers if m.self_attn.compressed_indexer]
        chunked_calls = []
        for indexer in indexers:
            indexer.query_chunk = 5
            original = indexer.select

            def select(q_resid, *args, original=original):
                chunked_calls.append(q_resid.shape[0] > 5)
                return original(q_resid, *args)

            indexer.select = select
        assert ids.shape[1] > model.config.text_config.index_topk
        if wrap:
            apply_ac(model, selective=True)
        torch.manual_seed(0)
        model.initialize_weights(dtype=torch.float32)
        model.train()
        out = model(ids, labels=ids)
        out.loss.backward()
        assert chunked_calls and all(chunked_calls)
        results.append(
            (out.loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None})
        )
    (plain_loss, plain_grads), (wrapped_loss, wrapped_grads) = results
    assert torch.equal(plain_loss, wrapped_loss)
    assert plain_grads.keys() == wrapped_grads.keys() and len(plain_grads) > 0
    for name, grad in plain_grads.items():
        torch.testing.assert_close(
            wrapped_grads[name], grad, rtol=1e-5, atol=1e-7, msg=lambda m, name=name: f"{name}: {m}"
        )
