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
``reshard_after_forward=True`` and ``wrap_outer_model=False``, and the same with a wrapped root. The recipes call
the model with ``compute_logits=False``, so the head never runs and stays sharded in its master dtype;
``prepare_lm_weight`` applies the head's policy instead, as FSDP would have. The loss streams the master in
vocabulary chunks that it projects in BF16, with the hidden states the model hands out cast to match (the
``torch_compile`` projection ``e @ c.T`` aligns nothing itself), and averages the head's gradient in FP32. The
CPU stand-in for the reviewer's Qwen3MoE is a
tiny dense DeepSeek-V3 (Qwen3MoE builds its rotary cache on the current CUDA device): the same
``LMHead`` projection path, the same standalone FSDP ``lm_head`` unit under the policy.
"""

from __future__ import annotations

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.loss.linear_ce_base import ShardedLMHeadWeight
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


def _eager_chunk_projection(hidden, weight, targets, softcap, logits_dtype, lm_head=None):
    """The streamed ``torch_compile`` kernel's chunk math, run eagerly; mixed operands would raise here.

    Args:
        hidden: Tensor of shape [tokens, hidden].
        weight: Tensor of shape [chunk_vocab, hidden], one streamed vocabulary chunk.
        targets: Tensor of shape [tokens] with chunk-local target rows, or -1.
        softcap: Optional logit softcapping value.
        logits_dtype: Optional rounding dtype for the projection output.
        lm_head: Optional ``LMHead`` whose ``project`` gives the chunk's logits.

    Returns:
        FP32 tensors of shape [tokens]: the chunk log-sum-exp and target logits.
    """
    from nemo_automodel.components.loss import linear_ce_base

    assert hidden.dtype == weight.dtype, f"mixed operands reached the kernel: {hidden.dtype} vs {weight.dtype}"
    return linear_ce_base._chunk_lse_and_target(hidden, weight, targets, softcap, logits_dtype, lm_head)


def _worker(
    rank: int, port: int, master_dtype: torch.dtype, wrap_outer_model: bool, reduce_dtype: torch.dtype | None
) -> None:
    torch.set_num_threads(1)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("gloo", rank=rank, world_size=_WORLD_SIZE)
    try:
        from nemo_automodel.components.loss import linear_ce as linear_ce_mod
        from nemo_automodel.components.loss import linear_ce_base

        linear_ce_mod.HAVE_CUT_CROSS_ENTROPY = True
        # Inductor on CPU is not under test; run the streamed kernel eagerly behind a dtype check.
        linear_ce_base._compiled_chunk_lse_and_target = _eager_chunk_projection
        linear_ce_base._compiled_chunk_logits = linear_ce_base._chunk_logits

        torch.manual_seed(3)
        backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
        model = DeepseekV3ForCausalLM(_config(), backend=backend)
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=master_dtype)
        mesh = init_device_mesh("cpu", (_WORLD_SIZE,), mesh_dim_names=("dp",))
        policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=reduce_dtype,
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
            wrap_outer_model=wrap_outer_model,
        )
        assert isinstance(model.lm_head, FSDPModule) and isinstance(model, FSDPModule) == wrap_outer_model

        loss_fn = FusedLinearCrossEntropy(impl="torch_compile")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        input_ids = torch.randint(0, _VOCAB, (2, 8), generator=torch.Generator().manual_seed(100 + rank))
        labels = input_ids.roll(-1, dims=1)
        reduce_scatter = dist.reduce_scatter_tensor
        reduced_dtypes = []

        def record_reduce_scatter(output: torch.Tensor, input: torch.Tensor, *args, **kwargs):
            """Record the dtype of each streamed head-gradient reduce-scatter, then run it.

            Args:
                output: Tensor of shape [chunk_rows, hidden] receiving this rank's rows.
                input: Tensor of shape [ranks * chunk_rows, hidden] holding every rank's rows.
                *args: Remaining ``reduce_scatter_tensor`` arguments.
                **kwargs: Remaining ``reduce_scatter_tensor`` options.

            Returns:
                The collective's work handle.
            """
            # The streamed head sends [ranks * chunk_rows, hidden] chunks; FSDP's own reduce-scatters are flat.
            if input.ndim == 2:
                reduced_dtypes.append(input.dtype)
            return reduce_scatter(output, input, *args, **kwargs)

        dist.reduce_scatter_tensor = record_reduce_scatter
        for _ in range(2):  # the second step streams the head again
            out = model(input_ids, compute_logits=False, output_hidden_states=True)
            hidden = get_final_hidden_states(out)
            assert hidden.shape == (2, 8, 64)
            # The forward skipped the head, which stays sharded in its master dtype; the loss streams it under the
            # head's policy instead of gathering it.
            assert isinstance(model.lm_head.weight, DTensor) and model.lm_head.weight.dtype == master_dtype
            lm_weight = prepare_lm_weight(loss_fn, model, grad_reduce_group=dist.group.WORLD)
            assert isinstance(lm_weight, ShardedLMHeadWeight)
            assert lm_weight.dtype == torch.bfloat16 and lm_weight.shape == (_VOCAB, 64)
            reduced_dtypes.clear()
            loss = loss_fn(
                hidden, labels, lm_weight, num_label_tokens=labels.numel(), grad_reduce_group=dist.group.WORLD
            )
            assert torch.isfinite(loss)
            loss.backward()
            assert model.lm_head.weight.grad is not None and model.lm_head.weight.grad.dtype == master_dtype
            # The head's gradient is averaged in the policy's reduce_dtype, or its param_dtype when that is unset,
            # as FSDP reduces an unsharded head's.
            assert reduced_dtypes and set(reduced_dtypes) == {reduce_dtype or torch.bfloat16}
            assert sum(parameter.grad is not None for parameter in model.parameters()) > 1  # the backbone too
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        # A wrapped root hooks every gradient-requiring tensor of the forward's output for backward; a hook on a
        # parameter is never removed, so the output must not carry the head's weight.
        assert not model.lm_head.weight._backward_hooks
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(
    30,
    hard_timeout=60,
    reason="spawns two Gloo FSDP2 workers for a tiny Qwen3MoE forward/backward/step twice",
)
@pytest.mark.parametrize(
    ("master_dtype", "wrap_outer_model", "reduce_dtype"),
    [
        (torch.float32, False, torch.float32),
        (torch.bfloat16, False, torch.float32),
        (torch.float32, True, torch.float32),
        (torch.float32, False, None),
    ],
    ids=["fp32_master", "bf16_master", "fp32_master_wrapped_root", "fp32_master_unset_reduce_dtype"],
)
def test_fused_loss_streams_the_head_under_its_fsdp_policy(
    master_dtype: torch.dtype, wrap_outer_model: bool, reduce_dtype: torch.dtype | None
) -> None:
    mp.spawn(_worker, args=(_free_port(), master_dtype, wrap_outer_model, reduce_dtype), nprocs=_WORLD_SIZE, join=True)
