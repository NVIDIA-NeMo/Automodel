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
