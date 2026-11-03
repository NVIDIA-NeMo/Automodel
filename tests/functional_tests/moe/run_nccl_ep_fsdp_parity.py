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

"""Run with torchrun on two SM90+ GPUs in an NCCL-EP-enabled container.

Checks one training step of a tiny ERNIE-4.5-MoE model whose experts use NCCL-EP, composed by
the production MoE parallelizer with FSDP2 over the same two ranks (dp=2, ep=2), against a
single-process FP32 copy of the model. Expert gradients come from both ranks' tokens through
NCCL-EP, while FSDP reduces the dense, attention, router, and embedding gradients.
"""

import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.tensor import DTensor
from transformers.models.ernie4_5_moe.configuration_ernie4_5_moe import Ernie4_5_MoeConfig

from nemo_automodel.components.distributed.config import FSDP2Config
from nemo_automodel.components.distributed.mesh import MeshContext, ParallelismSizes
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie4_5.model import Ernie4_5_MoeForCausalLM
from nemo_automodel.components.moe.nccl_ep import finalize_nccl_ep
from nemo_automodel.components.moe.parallelizer import parallelize_model
from nemo_automodel.components.training.utils import (
    prepare_for_final_backward,
    prepare_for_grad_accumulation,
    scale_grads_and_clip_grad_norm,
)

logger = logging.getLogger(__name__)

_BATCH = 2
_SEQUENCE = 16
_VOCAB = 128
_LEARNING_RATE = 0.1
# BF16 rounds every activation and weight to 8 significant bits (unit roundoff 2**-8), and
# the composed step chains about ten rounded GEMM and normalization stages. Allow a 5%
# relative L2 error per tensor against the FP32 model, the bound the transport parity test
# uses for its BF16 expert gradients.
_RELATIVE_TOLERANCE = 0.05


def _config() -> Ernie4_5_MoeConfig:
    """Return a dense first layer and a top-2 MoE layer with four experts, two per EP rank."""
    return Ernie4_5_MoeConfig(
        vocab_size=_VOCAB,
        hidden_size=512,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        max_position_embeddings=64,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        use_bias=False,
        tie_word_embeddings=True,
        pad_token_id=0,
        moe_intermediate_size=128,
        moe_k=2,
        moe_num_experts=4,
        moe_num_shared_experts=0,
        moe_layer_start_index=1,
        moe_layer_end_index=1,
        moe_layer_interval=1,
        router_aux_loss_coef=0.0,
        # FP32 master weights; FSDP computes in BF16, which NCCL-EP requires.
        torch_dtype=torch.float32,
    )


def _backend(dispatcher: str, experts: str) -> BackendConfig:
    """Return the backend shared by the NCCL-EP model and its FP32 reference."""
    return BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts=experts,
        dispatcher=dispatcher,
        nccl_ep={"max_tokens_per_rank": _BATCH * _SEQUENCE},
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _batch(rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return rank-distinct token IDs and next-token labels.

    Args:
        rank: Data-parallel rank whose batch to build.

    Returns:
        Input IDs and labels, each an int64 CUDA tensor of shape [batch, sequence].
    """
    generator = torch.Generator().manual_seed(100 + rank)
    tokens = torch.randint(1, _VOCAB, (_BATCH, _SEQUENCE + 1), generator=generator)
    return tokens[:, :-1].cuda(), tokens[:, 1:].cuda()


def _loss(model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Return the mean next-token cross-entropy of one batch.

    Args:
        model: Causal language model returning logits of shape [batch, sequence, vocab].
        input_ids: Token IDs of shape [batch, sequence].
        labels: Target token IDs of shape [batch, sequence].

    Returns:
        FP32 scalar loss.
    """
    logits = model(input_ids).logits
    return F.cross_entropy(logits.float().flatten(0, 1), labels.flatten())


def _full(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a parameter or gradient to its global shape.

    Args:
        tensor: DTensor sharded by FSDP or EP, or a regular tensor, of arbitrary shape.

    Returns:
        FP32 tensor with the global shape. Gathering a DTensor is a collective over its mesh.
    """
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).float()


def _assert_close(name: str, actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Bound the relative L2 error of a BF16-computed tensor against the FP32 reference.

    Args:
        name: Tensor name for the failure message.
        actual: FP32 tensor of arbitrary shape from the composed NCCL-EP model.
        expected: FP32 tensor of the same shape from the reference model.

    Returns:
        The relative L2 error, or 0 when the reference is exactly zero.
    """
    error = torch.linalg.vector_norm(actual - expected).item()
    scale = torch.linalg.vector_norm(expected).item()
    # The floor covers tensors whose exact value is zero, such as unrouted experts' gradients.
    bound = _RELATIVE_TOLERANCE * scale + 1e-7
    assert error <= bound, f"{name}: L2 error {error:.3e} exceeds {bound:.3e}"
    return error / scale if scale else 0.0


def _check_step(mesh: MeshContext, *, rank: int, world_size: int, empty_receiver: bool, checkpointed: bool) -> None:
    """Compare gradients, their global norm, and one SGD update with the FP32 model.

    Args:
        mesh: Two-rank mesh with dp=2 and ep=2.
        rank: Global rank, also the data-parallel and EP rank.
        world_size: Number of ranks.
        empty_receiver: Route every token to rank 0's experts, so rank 1 receives no rows.
        checkpointed: Whether the parallelizer applies activation checkpointing.
    """
    torch.manual_seed(1234)
    model = Ernie4_5_MoeForCausalLM(_config(), backend=_backend("nccl_ep", "torch_mm")).cuda()
    moe = model.model.layers["1"].mlp
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.normal_(mean=0.0, std=0.02)
        if empty_receiver:
            # Selection adds this bias to the router probabilities, while the routing weights
            # come from the probabilities alone, so the router still receives gradients.
            moe.gate.e_score_correction_bias.copy_(torch.tensor([0.0, 0.0, -10.0, -10.0]))
    reference = Ernie4_5_MoeForCausalLM(_config(), backend=_backend("torch", "torch")).cuda()
    reference.load_state_dict(model.state_dict())
    initial = {name: parameter.detach().clone() for name, parameter in reference.named_parameters()}

    parallelize_model(
        model,
        mesh.device_mesh,
        mesh.moe_mesh,
        activation_checkpointing=checkpointed,
        **mesh.parallelize_axis_kwargs(),
    )
    # The recipe's step for a single microbatch.
    prepare_for_grad_accumulation([model])
    prepare_for_final_backward([model])
    loss = _loss(model, *_batch(rank))
    loss.backward()
    grad_norm = scale_grads_and_clip_grad_norm(
        # Large enough to leave the gradients unclipped, so that they stay comparable.
        1e6,
        [model],
        norm_type=2.0,
        device_mesh=mesh.device_mesh,
        moe_mesh=mesh.moe_mesh,
        ep_axis_name="ep",
        foreach=True,
        dp_group_size=world_size,
    )
    # Dense and expert parameters live on different meshes, so update them one at a time.
    torch.optim.SGD(model.parameters(), lr=_LEARNING_RATE, foreach=False).step()

    # FSDP averages the replicated gradients over ranks, so the reference loss is the rank mean.
    reference_losses = [_loss(reference, *_batch(source)) for source in range(world_size)]
    (sum(reference_losses) / world_size).backward()
    torch.testing.assert_close(loss.detach(), reference_losses[rank].detach(), rtol=0.02, atol=0)

    local_expert_grad = moe.experts.gate_and_up_projs.grad.to_local()
    if empty_receiver and rank == 1:
        # No routes reach this rank, yet its experts take part in EP backward with zero gradients.
        assert torch.count_nonzero(local_expert_grad) == 0
    else:
        assert torch.count_nonzero(local_expert_grad) > 0

    reference_parameters = dict(reference.named_parameters())
    names = set()
    reference_norm_sq = 0.0
    worst = 0.0
    for name, parameter in model.named_parameters():
        name = name.replace("_checkpoint_wrapped_module.", "")
        names.add(name)
        expected_grad = reference_parameters[name].grad
        worst = max(worst, _assert_close(f"{name}.grad", _full(parameter.grad), expected_grad))
        _assert_close(f"{name} update", _full(parameter) - initial[name], -_LEARNING_RATE * expected_grad)
        reference_norm_sq += expected_grad.square().sum().item()
    assert names == set(reference_parameters), names ^ set(reference_parameters)
    assert torch.isfinite(grad_norm)
    torch.testing.assert_close(grad_norm.item(), reference_norm_sq**0.5, rtol=_RELATIVE_TOLERANCE, atol=0)
    logger.info(
        "rank=%d case=fsdp_ep empty_receiver=%s checkpoint=%s: gradients, norm and update passed "
        "(largest relative gradient error %.2e)",
        rank,
        empty_receiver,
        checkpointed,
        worst,
    )


def main() -> None:
    """Compose NCCL-EP with FSDP2 through the production parallelizer and check one step."""
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        mesh = MeshContext.build(
            FSDP2Config(), ParallelismSizes(dp_size=world_size, ep_size=world_size), world_size=world_size
        )
        for empty_receiver in (False, True):
            for checkpointed in (False, True):
                _check_step(
                    mesh, rank=rank, world_size=world_size, empty_receiver=empty_receiver, checkpointed=checkpointed
                )
    finally:
        torch.cuda.synchronize()
        finalize_nccl_ep()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
