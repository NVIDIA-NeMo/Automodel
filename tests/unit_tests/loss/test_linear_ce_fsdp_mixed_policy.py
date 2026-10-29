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
"""Two-rank CPU/Gloo regression: the fused loss consumes the FSDP head's master weight after resharding.

The reviewer's scenario on #4288: a native model with FP32 parameters under
``MixedPrecisionPolicy(param_dtype=bf16, reduce_dtype=fp32, output_dtype=fp32, cast_forward_inputs=True)``,
``reshard_after_forward=True`` and ``wrap_outer_model=False``. The recipes gather the head's weight with
``prepare_lm_weight`` *after* the forward, when the head has been resharded back to its FP32 master, so the
hidden states the model hands out must be projected in that weight's dtype; the ``torch_compile`` projection
``e @ c.T`` inside cut-cross-entropy aligns nothing itself. The CPU stand-in for the reviewer's Qwen3MoE is a
tiny dense DeepSeek-V3 (Qwen3MoE builds its rotary cache on the current CUDA device): the same
``compute_lm_head_logits`` head path, the same standalone FSDP ``lm_head`` unit under the policy.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.utils import prepare_lm_weight
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v3.model import DeepseekV3ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_fsdp
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states

_WORLD_SIZE = 2
_VOCAB = 100


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _config() -> DeepseekV3Config:
    # first_k_dense_replace >= num_hidden_layers keeps every layer dense: the MoE experts are not what the
    # scenario is about, and the dense tiny model builds and runs on CPU.
    return DeepseekV3Config(
        vocab_size=_VOCAB,
        hidden_size=64,
        num_attention_heads=4,
        num_hidden_layers=2,
        first_k_dense_replace=2,
        intermediate_size=128,
        qk_rope_head_dim=16,
        v_head_dim=16,
        qk_nope_head_dim=16,
    )


def _torch_compile_stand_in(hidden, weight, targets=None, *, ignore_index=-100, reduction="sum", **_):
    """cut-cross-entropy's ``torch_compile`` projection, ``e @ c.T``; mixed operands would raise there."""
    assert hidden.dtype == weight.dtype, f"mixed operands reached the kernel: {hidden.dtype} vs {weight.dtype}"
    logits = (hidden @ weight.T).float()
    return F.cross_entropy(
        logits.view(-1, logits.shape[-1]), targets.view(-1), ignore_index=ignore_index, reduction=reduction
    )


def _worker(rank: int, port: int) -> None:
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        from nemo_automodel.components.loss import linear_ce as linear_ce_mod

        linear_ce_mod.HAVE_CUT_CROSS_ENTROPY = True
        linear_ce_mod.linear_cross_entropy = _torch_compile_stand_in

        torch.manual_seed(3)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
        model = DeepseekV3ForCausalLM(_config(), backend=backend)
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.float32,
            cast_forward_inputs=True,
        )
        apply_fsdp(
            model,
            mesh,
            ep_enabled=False,
            ep_shard_enabled=False,
            mp_policy=policy,
            reshard_after_forward=True,
            wrap_outer_model=False,
        )
        assert isinstance(model.lm_head, FSDPModule) and not isinstance(model, FSDPModule)

        loss_fn = FusedLinearCrossEntropy(impl="torch_compile")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        input_ids = torch.randint(0, _VOCAB, (2, 8), generator=torch.Generator().manual_seed(100 + rank))
        labels = input_ids.roll(-1, dims=1)
        for _ in range(2):  # the second step runs the gather on a resharded head again
            out = model(input_ids, logits_to_keep=1, output_hidden_states=True)
            hidden = get_final_hidden_states(out)
            assert hidden.shape == (2, 8, 64)
            # The recipes prepare the weight after the forward: the head is resharded, its master dtype is fp32.
            lm_weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=dist.group.WORLD)
            assert lm_weight.dtype == torch.float32 and lm_weight.shape == (_VOCAB, 64)
            loss = loss_fn(
                hidden, labels, lm_weight, num_label_tokens=labels.numel(), grad_reduce_group=dist.group.WORLD
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert model.lm_head.weight.grad is not None
            assert sum(parameter.grad is not None for parameter in model.parameters()) > 1  # the backbone too
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="spawns two Gloo FSDP2 workers for a tiny Qwen3MoE forward/backward/step twice",
)
def test_fused_loss_consumes_the_resharded_fp32_head_under_a_bf16_policy() -> None:
    mp.spawn(_worker, args=(_free_port(),), nprocs=_WORLD_SIZE, join=True)
