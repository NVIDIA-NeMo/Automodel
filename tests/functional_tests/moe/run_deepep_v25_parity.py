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

"""DeepEP v2.5 expert-parallel forward/backward parity against a dense FP32 oracle."""

import argparse
import logging
import os
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Shard, distribute_tensor
from torch.utils.checkpoint import checkpoint

from nemo_automodel.components.models.common.utils import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.layers import MoE
from nemo_automodel.shared.deepep_v25 import DeepEpV25Config, finalize_deepep_v25

logger = logging.getLogger(__name__)

HIDDEN = 512
MAX_TOKENS = 5
ROUTINGS = ("spread", "uneven", "silent_senders", "one_receiver", "all_masked")


def _dense_moe(hidden, probs, indices, mask, up, down):
    """Run every routed token through every global expert in FP32.

    Args:
        hidden: Tokens [tokens, hidden].
        probs: Routing weights [tokens, top_k].
        indices: Global expert IDs [tokens, top_k].
        mask: Valid-token mask [tokens].
        up: Gate/up weights [experts, hidden, 2 * intermediate].
        down: Down weights [experts, intermediate, hidden].

    Returns:
        Weighted expert sum [tokens, hidden].
    """
    gate_weight = (F.one_hot(indices, up.shape[0]) * probs[..., None]).sum(1) * mask[:, None]
    gate, value = torch.einsum("th,ehi->eti", hidden, up).chunk(2, dim=-1)
    expert_out = torch.einsum("eti,eih->eth", F.silu(gate) * value, down)
    return torch.einsum("eth,te->th", expert_out, gate_weight)


def _fp8_transport_reference(hidden, payload, restored):
    """Validate FP8 encoding mathematically, then construct its straight-through oracle.

    Args:
        hidden: FP32 leaf [tokens, hidden] containing the original BF16 values.
        payload: Actual source FP8 bytes [capacity, hidden] and FP32 block-128 scales.
        restored: Actual BF16 restoration [tokens, hidden] before transport.

    Returns:
        Independently restored values with an identity derivative. E4M3 nearest
        neighbors are derived from the format, without production conversion
        helpers. Either neighbor is valid only at a boundary within FP32 scaling
        roundoff; the observed, validated code determines the expert's input.
    """
    if hidden.shape[0] == 0:
        return hidden
    tokens, width = hidden.shape
    blocks = hidden.detach().double().reshape(tokens, width // 128, 128)
    maximum = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    expected_scales = maximum / 448.0
    codes = payload[0][:tokens].float().reshape_as(blocks)
    scales = payload[1][:tokens].float().unsqueeze(-1)
    torch.testing.assert_close(scales.double(), expected_scales, rtol=2 * torch.finfo(torch.float32).eps, atol=0)
    # E4M3: exponent bias 7, three mantissa bits; exponent 15 / mantissa 7 is NaN.
    positive_values = [0.0] + [mantissa * 2.0**-9 for mantissa in range(1, 8)]
    positive_values += [
        (1 + mantissa / 8) * 2.0 ** (exponent - 7)
        for exponent in range(1, 16)
        for mantissa in range(7 if exponent == 15 else 8)
    ]
    values = torch.tensor(positive_values, dtype=torch.float64, device=hidden.device)
    normalized = blocks.abs() * 448.0 / maximum
    upper_index = torch.searchsorted(values, normalized).clamp_max(values.numel() - 1)
    lower = values[(upper_index - 1).clamp_min(0)]
    upper = values[upper_index]
    nearest_error = torch.minimum((normalized - lower).abs(), (upper - normalized).abs())
    scaling_roundoff = normalized * (4 * torch.finfo(torch.float32).eps)
    actual_error = (codes.double() - blocks.sign() * normalized).abs()
    assert torch.all(actual_error <= nearest_error + scaling_roundoff), "FP8 code is not a nearest E4M3 value"
    expected = (codes * scales).reshape_as(hidden).bfloat16().float()
    torch.testing.assert_close(restored.float(), expected, rtol=0, atol=0)
    return hidden + (expected - hidden).detach()


def _routing(routing, rank, num_experts, generator):
    """Build one microbatch's routes for this rank.

    Args:
        routing: One of ``ROUTINGS``.
        rank: EP rank.
        num_experts: Global expert count.
        generator: CUDA generator seeded per rank and microbatch.

    Returns:
        Token count, expert indices [tokens, 2], and valid-token mask [tokens].
    """
    tokens = MAX_TOKENS
    if routing == "uneven":
        tokens = MAX_TOKENS - 2 * (rank % 2)
    elif routing == "silent_senders" and rank:
        tokens = 0
    if routing == "one_receiver":
        # Every route lands on rank 0's experts; the other ranks receive nothing.
        indices = torch.tensor([[0, 1]], device="cuda").expand(tokens, 2).contiguous()
    else:
        scores = torch.rand(tokens, num_experts, device="cuda", generator=generator)
        indices = scores.topk(2, dim=-1).indices
    mask = torch.rand(tokens, device="cuda", generator=generator) > 0.2
    if routing == "all_masked":
        mask.zero_()
    return tokens, indices, mask


def _check(experts, up, down, *, rank, world_size, routing, checkpointed, fp8_dispatch):
    """Run two accumulated microbatches and compare outputs and gradients with the oracle.

    Args:
        experts: Production expert module wired to DeepEP v2.5.
        up: Global gate/up weights [experts, hidden, 2 * intermediate], BF16.
        down: Global down weights [experts, intermediate, hidden], BF16.
        rank: EP rank.
        world_size: EP size.
        routing: One of ``ROUTINGS``.
        checkpointed: Recompute the expert forward during backward.
        fp8_dispatch: Feed the oracle the independently validated FP8 transport values.
    """
    local = up.shape[0] // world_size
    owned = slice(rank * local, (rank + 1) * local)
    with torch.no_grad():
        experts.gate_and_up_projs.to_local().copy_(up[owned])
        experts.down_projs.to_local().copy_(down[owned])
    experts.zero_grad(set_to_none=True)
    oracle_up = up.float().requires_grad_()
    oracle_down = down.float().requires_grad_()
    losses, oracle_losses, leaves = [], [], []
    for microbatch in range(2):
        generator = torch.Generator("cuda").manual_seed(1000 * microbatch + rank)
        tokens, indices, mask = _routing(routing, rank, up.shape[0], generator)
        hidden = (0.2 * torch.randn(tokens, HIDDEN, device="cuda", generator=generator)).bfloat16().requires_grad_()
        probs = torch.rand(tokens, 2, device="cuda", generator=generator).softmax(-1).requires_grad_()
        oracle_hidden = hidden.detach().float().requires_grad_()
        oracle_probs = probs.detach().clone().requires_grad_()
        oracle_input = oracle_hidden
        if fp8_dispatch:
            manager = experts.token_dispatcher._comm_manager
            with torch.no_grad():
                padded = F.pad(hidden.detach(), (0, 0, 0, manager.capacity - tokens))
                payload = manager.fp8.quantize(padded)
                restored = manager.fp8.dequantize(*payload)[:tokens]
            oracle_input = _fp8_transport_reference(oracle_hidden, payload, restored)
        if checkpointed:
            output = checkpoint(experts, hidden, mask, probs, indices, use_reentrant=False)
        else:
            output = experts(hidden, mask, probs, indices)
        expected = _dense_moe(oracle_input, oracle_probs, indices, mask, oracle_up, oracle_down)
        # BF16 expert GEMMs against an FP32 oracle.
        torch.testing.assert_close(output.float(), expected, rtol=0.05, atol=2e-4)
        upstream = torch.randn(expected.shape, device="cuda", generator=generator)
        losses.append((output.float() * upstream).sum())
        oracle_losses.append((expected * upstream).sum())
        leaves.append((hidden, probs, oracle_hidden, oracle_probs))
    sum(losses).backward()
    sum(oracle_losses).backward()
    for hidden, probs, oracle_hidden, oracle_probs in leaves:
        torch.testing.assert_close(hidden.grad.float(), oracle_hidden.grad, rtol=0.05, atol=2e-4)
        torch.testing.assert_close(probs.grad, oracle_probs.grad, rtol=0.05, atol=2e-4)
    squared = torch.zeros(2, device="cuda")
    for parameter, oracle in ((experts.gate_and_up_projs, oracle_up), (experts.down_projs, oracle_down)):
        # Every rank's loss reaches every expert; sum the oracle gradients over the group.
        dist.all_reduce(oracle.grad)
        actual, reference = parameter.grad.to_local().float(), oracle.grad[owned]
        # BF16 accumulation error grows with the number of contributing ranks.
        torch.testing.assert_close(actual, reference, rtol=0.05, atol=1.5e-4 * world_size)
        squared += torch.stack((actual.square().sum(), reference.square().sum()))
    dist.all_reduce(squared)
    torch.testing.assert_close(squared[0].sqrt(), squared[1].sqrt(), rtol=0.025, atol=2e-4)
    logger.info("rank=%d routing=%s checkpointed=%s passed", rank, routing, checkpointed)


def main() -> None:
    """Check every routing pattern eagerly and under activation checkpointing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp8-dispatch", action="store_true")
    parser.add_argument("--async-dispatch", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    try:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        config = MoEConfig(
            n_routed_experts=2 * world_size,
            n_shared_experts=0,
            n_activated_experts=2,
            n_expert_groups=1,
            n_limited_groups=1,
            train_gate=True,
            gate_bias_update_factor=0.0,
            aux_loss_coeff=0.0,
            score_func="softmax",
            route_scale=1.0,
            dim=HIDDEN,
            inter_dim=64,
            moe_inter_dim=64,
            norm_topk_prob=False,
        )
        backend = BackendConfig(
            dispatcher="deepep_v25",
            experts="torch_mm",
            deepep_v25=DeepEpV25Config(max_tokens_per_rank=MAX_TOKENS, use_fp8_dispatch=args.fp8_dispatch),
            dispatcher_async_dispatch=args.async_dispatch,
            attn="sdpa",
            linear="torch",
        )
        experts = MoE(config, backend).experts.cuda().bfloat16()
        generator = torch.Generator("cuda").manual_seed(17)
        up = (0.02 * torch.randn(experts.gate_and_up_projs.shape, device="cuda", generator=generator)).bfloat16()
        down = (0.02 * torch.randn(experts.down_projs.shape, device="cuda", generator=generator)).bfloat16()
        mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("ep",))
        experts.gate_and_up_projs = torch.nn.Parameter(distribute_tensor(up, mesh, [Shard(0)]))
        experts.down_projs = torch.nn.Parameter(distribute_tensor(down, mesh, [Shard(0)]))
        experts.init_token_dispatcher(mesh)
        for checkpointed in (False, True):
            for routing in ROUTINGS:
                _check(
                    experts,
                    up,
                    down,
                    rank=rank,
                    world_size=world_size,
                    routing=routing,
                    checkpointed=checkpointed,
                    fp8_dispatch=args.fp8_dispatch,
                )
    except Exception:
        logger.exception("DeepEP v2.5 parity failed")
        raise
    else:
        torch.cuda.synchronize()
        finalize_deepep_v25()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
