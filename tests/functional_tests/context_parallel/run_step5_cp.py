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

"""CP1/CP2 logits, loss and all-parameter-gradient parity via native CP hooks.

Launch with torchrun --standalone --nproc_per_node=2. The small four-layer model
covers dense FFN, routed/shared experts, SWA, CSA, padding and packed documents.
The default is the original short sweep. Opt in to --scenario long-packed,
--scenario long-crossing, or --scenario long-single to exercise 8K native-varlen
and truly sparse shards. Every launch first checks SWA halo exchange against an
exact forward/backward oracle. This random-weight component test is not a
checkpoint loading validation.
"""

import argparse
import gc
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import TypedDict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn import functional as F
from typing_extensions import NotRequired

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5.configuration import Step5BackendConfig, Step5Config
from nemo_automodel.components.models.step5.cp import gather_halo
from nemo_automodel.components.models.step5.layers import Step5Attention
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.models.step5.sparse import RegionSelection, Step5SparseIndexer
from nemo_automodel.components.moe.layers import Gate
from nemo_automodel.components.moe.parallelizer import apply_cp


class GradientMetrics(TypedDict):
    relative_rms: float
    max_abs: float


class RouteBucket(TypedDict):
    queries: int
    exact_set_queries: int
    selected_id_agreement: float


class RouteMetrics(TypedDict):
    cp1_native_varlen: bool
    cp2_native_varlen: bool
    cp1_all_padding_key: int | None
    cp2_all_padding_key: int | None
    prefix: RouteBucket
    sparse: RouteBucket


@dataclass
class RouteSnapshot:
    """Selected IDs [batch, provider_groups, local_queries, slots] and dispatch metadata."""

    ids: torch.Tensor
    native_varlen: bool
    all_padding_key: int | None


class LogitBucket(TypedDict):
    queries: int
    max_abs: float
    rms: float
    kl: float


class CPReport(TypedDict):
    rank: int
    length: int
    cp1_loss: float
    cp2_loss: float
    logit_max_abs: float
    logit_rms: float
    logit_kl: float
    cp1_peak_bytes: int
    cp2_peak_bytes: int
    gradients: dict[str, GradientMetrics]
    packed: NotRequired[bool]
    scenario: NotRequired[str]
    swa_varlen: NotRequired[bool]
    csa_forward_full_blocks: NotRequired[bool]
    csa_backward_full_blocks: NotRequired[bool]
    position_buckets: NotRequired[dict[str, LogitBucket]]
    padding_queries: NotRequired[int]
    routes: NotRequired[dict[str, RouteMetrics]]


def compare_cp(
    model: Step5ForCausalLM,
    tokens: torch.Tensor,
    positions: torch.Tensor,
    padding: torch.Tensor | None = None,
    *,
    stage_output: Path | None = None,
) -> CPReport:
    """Compare a replicated model before/after native CP setup on identical inputs.

    Args:
        model: Small random-weight model; no checkpoint-loading validation is claimed.
        tokens: Global token IDs of shape [batch, sequence].
        positions: Global document-local positions of shape [batch, sequence].
        padding: Optional global Boolean mask [batch, sequence], true for padding.
        stage_output: Optional directory for downstream attention and routing diagnostics.

    Returns:
        Logit, loss, route, gradient and memory measurements.
    """
    rank = dist.get_rank()
    route_records: list[dict[str, RouteSnapshot]] = [{}, {}]
    phase = 0

    def route_hook(
        name: str,
    ) -> Callable[[Step5SparseIndexer, tuple[torch.Tensor | None, ...], RegionSelection], None]:
        def record(
            _module: Step5SparseIndexer, _inputs: tuple[torch.Tensor | None, ...], output: RegionSelection
        ) -> None:
            """Retain selected routes independently of CUDA execution counters.

            Args:
                _module: Indexer whose output is recorded.
                _inputs: Indexer inputs: local hidden [batch, local_queries, hidden],
                    local positions [batch, local_queries], optional global projected
                    K/Z [batch, global_keys, groups, proxy_dim] and global position/
                    padding metadata [batch, global_keys].
                output: RegionSelection with IDs [batch, provider_groups, local_queries,
                    slots] and global key-region/document metadata.
            """
            route_records[phase][name] = RouteSnapshot(
                output.region_ids.detach().cpu(),
                output.causal_layout is not None,
                output.all_padding_key,
            )

        return record

    handles = [
        module.register_forward_hook(route_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, Step5SparseIndexer)
    ]
    stage_records: list[dict[str, torch.Tensor]] = [{}, {}]

    def attention_hook(name: str) -> Callable[[Step5Attention, tuple[torch.Tensor, ...], torch.Tensor], None]:
        def record(module: Step5Attention, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            stage_records[phase][name + ".input"] = inputs[0].detach().cpu()
            stage_records[phase][name + ".output"] = output.detach().cpu()

        return record

    def gate_hook(
        name: str,
    ) -> Callable[
        [
            Gate,
            tuple[torch.Tensor, torch.Tensor, DeviceMesh | None],
            tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        ],
        None,
    ]:
        def record(
            module: Gate,
            inputs: tuple[torch.Tensor, torch.Tensor, DeviceMesh | None],
            output: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        ) -> None:
            stage_records[phase][name + ".input"] = inputs[0].detach().cpu()
            stage_records[phase][name + ".weights"] = output[0].detach().cpu()
            stage_records[phase][name + ".indices"] = output[1].detach().cpu()
            with torch.no_grad():
                scores = F.linear(inputs[0].float(), module.weight.float(), module.bias).sigmoid()
                scores = scores + module.e_score_correction_bias
            stage_records[phase][name + ".choice_scores"] = scores.cpu()

        return record

    if stage_output is not None:
        for name, module in model.named_modules():
            if isinstance(module, Step5Attention):
                handles.append(module.register_forward_hook(attention_hook(name)))
            elif isinstance(module, Gate):
                handles.append(module.register_forward_hook(gate_hook(name)))
    length = tokens.shape[1]
    labels = tokens.roll(-1, dims=1)
    labels[:, -1] = -100
    labels[:, :-1].masked_fill_(positions[:, 1:] == 0, -100)
    if padding is not None:
        labels.masked_fill_(padding, -100)
    count = (labels != -100).sum()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    baseline_logits = model(tokens, position_ids=positions, padding_mask=padding).logits
    baseline_loss = F.cross_entropy(baseline_logits.float().flatten(0, 1), labels.flatten(), reduction="sum") / count
    baseline_loss.backward()
    baseline_grads = {name: p.grad.detach().cpu() for name, p in model.named_parameters() if p.grad is not None}
    baseline_cpu = baseline_logits.detach().cpu()
    reference_loss = baseline_loss.detach().item()
    cp1_peak = torch.cuda.max_memory_allocated()
    del baseline_logits, baseline_loss
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()

    phase = 1
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    batch = {"input_ids": tokens.clone(), "labels": labels.clone(), "position_ids": positions.clone()}
    if padding is not None:
        batch["padding_mask"] = padding.clone()
    sharder = ContextParallelSharder(model, mesh, batch)
    context, local = sharder.shard(batch)
    torch.cuda.reset_peak_memory_stats()
    with context():
        logits = model(
            local["input_ids"], position_ids=local["position_ids"], padding_mask=local["padding_mask"]
        ).logits
        local_loss = F.cross_entropy(logits.float().flatten(0, 1), local["labels"].flatten(), reduction="sum") / count
        local_loss.backward()
    cp2_peak = torch.cuda.max_memory_allocated()
    loss = local_loss.detach().clone()
    dist.all_reduce(loss)
    local_length = logits.shape[1]
    start = rank * local_length
    valid = min(local_length, length - start)
    reference = baseline_cpu[:, start : start + valid].cuda().float()
    actual = logits[:, :valid].detach().float()
    report: CPReport = {
        "rank": rank,
        "length": length,
        "cp1_loss": reference_loss,
        "cp2_loss": loss.item(),
        "logit_max_abs": (reference - actual).abs().max().item(),
        "logit_rms": (reference - actual).square().mean().sqrt().item(),
        "logit_kl": (reference.log_softmax(-1).exp() * (reference.log_softmax(-1) - actual.log_softmax(-1)))
        .sum(-1)
        .mean()
        .item(),
        "cp1_peak_bytes": cp1_peak,
        "cp2_peak_bytes": cp2_peak,
        "gradients": {},
    }
    route_metrics: dict[str, RouteMetrics] = {}
    local_positions = positions[:, start : start + valid].cpu()
    boundary = (model.config.sparse_config.topk + 1) * model.config.sparse_config.region_block_size
    position_buckets: dict[str, LogitBucket] = {}
    for bucket, cpu_mask in (("prefix", local_positions < boundary), ("sparse", local_positions >= boundary)):
        mask = cpu_mask.to(actual.device)
        queries = int(cpu_mask.sum())
        if queries:
            bucket_reference, bucket_actual = reference[mask], actual[mask]
            position_buckets[bucket] = {
                "queries": queries,
                "max_abs": (bucket_reference - bucket_actual).abs().max().item(),
                "rms": (bucket_reference - bucket_actual).square().mean().sqrt().item(),
                "kl": (
                    bucket_reference.log_softmax(-1).exp()
                    * (bucket_reference.log_softmax(-1) - bucket_actual.log_softmax(-1))
                )
                .sum(-1)
                .mean()
                .item(),
            }
        else:
            position_buckets[bucket] = {"queries": 0, "max_abs": 0.0, "rms": 0.0, "kl": 0.0}
    report["position_buckets"] = position_buckets
    for name, cp2_routes in route_records[1].items():
        cp1_routes = route_records[0][name]
        expected_ids = cp1_routes.ids[:, :, start : start + valid]
        actual_ids = cp2_routes.ids[:, :, :valid]
        agreement = expected_ids == actual_ids
        per_query = agreement.all(dim=-1).all(dim=1)
        buckets: dict[str, RouteBucket] = {}
        for bucket, mask in (("prefix", local_positions < boundary), ("sparse", local_positions >= boundary)):
            queries = int(mask.sum())
            buckets[bucket] = {
                "queries": queries,
                "exact_set_queries": int((per_query & mask).sum()),
                "selected_id_agreement": float(agreement.permute(0, 2, 1, 3)[mask].float().mean()) if queries else 1.0,
            }
        route_metrics[name] = {
            "cp1_native_varlen": cp1_routes.native_varlen,
            "cp2_native_varlen": cp2_routes.native_varlen,
            "cp1_all_padding_key": cp1_routes.all_padding_key,
            "cp2_all_padding_key": cp2_routes.all_padding_key,
            "prefix": buckets["prefix"],
            "sparse": buckets["sparse"],
        }
    report["routes"] = route_metrics
    for handle in handles:
        handle.remove()
    if stage_output is not None:
        stage_output.mkdir(parents=True, exist_ok=True)
        torch.save({"cp1": stage_records[0], "cp2": stage_records[1]}, stage_output / f"rank{rank}.pt")
    assert abs(reference_loss - loss.item()) < 0.003, report
    assert report["logit_kl"] < 1e-4, report
    assert {name for name, p in model.named_parameters() if p.grad is not None} == set(baseline_grads)
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        dist.all_reduce(parameter.grad)
        reference_grad = baseline_grads.pop(name).flatten()
        actual_grad = parameter.grad.detach().flatten()
        error, energy, maximum = 0.0, 0.0, 0.0
        for offset in range(0, actual_grad.numel(), 2**22):
            a = actual_grad[offset : offset + 2**22].float()
            b = reference_grad[offset : offset + 2**22].cuda().float()
            error += (a - b).square().sum().item()
            energy += b.square().sum().item()
            maximum = max(maximum, (a - b).abs().max().item())
        relative = (error / max(energy, 1e-30)) ** 0.5
        report["gradients"][name] = {"relative_rms": relative, "max_abs": maximum}
        assert relative < 0.025, (name, relative, maximum, {k: v for k, v in report.items() if k != "gradients"})
    model.zero_grad(set_to_none=True)
    return report


def check_halo() -> None:
    """Compare SWA halo exchange with an exact forward/backward oracle on every rank."""
    rank, world, length = dist.get_rank(), dist.get_world_size(), 7
    global_input = torch.arange(2 * world * length * 3, device="cuda", dtype=torch.float32).view(2, world * length, 3)
    # Zero, short, exact-shard, and multi-shard halos exercise different peers.
    for halo, padding in product((0, 3, 7, 19, world * length), (0, rank + 1)):
        local = global_input[:, rank * length : (rank + 1) * length].clone().requires_grad_()
        actual = gather_halo(local, dist.group.WORLD, halo, left_padding=padding)
        first = max(0, rank * length - halo)
        torch.testing.assert_close(
            actual, F.pad(global_input[:, first : (rank + 1) * length], (0, 0, padding, 0)), atol=0, rtol=0
        )
        ((rank + 1) * actual.square().sum()).backward()
        expected = torch.zeros_like(global_input)
        for consumer in range(world):
            start, stop = max(0, consumer * length - halo), (consumer + 1) * length
            expected[:, start:stop] += 2 * (consumer + 1) * global_input[:, start:stop]
        torch.testing.assert_close(local.grad, expected[:, rank * length : (rank + 1) * length], atol=0, rtol=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("flex", "fa4"), default="fa4")
    parser.add_argument("--swa-varlen", action="store_true", help="Exercise opt-in native SWA window kernels")
    parser.add_argument(
        "--csa-backward-full-blocks", action="store_true", help="Exercise opt-in CSA backward mask skipping"
    )
    parser.add_argument(
        "--csa-forward-full-blocks", action="store_true", help="Exercise opt-in CSA forward mask skipping"
    )
    parser.add_argument(
        "--stage-output", type=Path, help="Optional attention/router diagnostic tensors, before parity gates"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        choices=(
            "short",
            "packed4k",
            "long-packed",
            "long-crossing",
            "long-single",
            "trailing4k",
            "trailing8k",
            "empty-tail8k",
            "sparse-tail8k",
        ),
        help="Repeat to select scenarios; default: original short single/packed sweep.",
    )
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    check_halo()
    scenarios: list[tuple[str, tuple[int, ...], int, int, int]] = []
    for scenario in args.scenario or ["short"]:
        if scenario == "short":
            scenarios.extend((("short-single", (263,), 4, 65, 0), ("short-packed", (19, 244), 4, 65, 0)))
        else:
            lengths = {
                "packed4k": (1031, 3065),
                "long-packed": (4096, 4096),
                "long-crossing": (2051, 4096, 2045),
                "long-single": (8192,),
                "trailing4k": (1031, 3058),
                "trailing8k": (2051, 4096, 2038),
                "empty-tail8k": (3000,),
                "sparse-tail8k": (8185,),
            }[scenario]
            padded = {"trailing4k": 7, "trailing8k": 7, "empty-tail8k": 5192, "sparse-tail8k": 7}.get(scenario, 0)
            scenarios.append((scenario, lengths, 512, 512, padded))
    for scenario, lengths, topk, window, padded in scenarios:
        torch.manual_seed(73)
        config = Step5Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_attention_groups=2,
            head_dim=192,
            vocab_size=128,
            moe_num_experts=4,
            moe_top_k=2,
            moe_intermediate_size=32,
            share_expert_dim=32,
            moe_layer_list=[3],
            dtype="bfloat16",
            num_nextn_predict_layers=0,
            sliding_window=window,
            sparse_config=dict(
                proxy_dim=32,
                sparse_indexer_rope_dim=8,
                sparse_indexer_num_heads=4,
                num_provider_groups=2,
                topk=topk,
                region_block_size=8,
            ),
        )
        backend = (
            Step5BackendConfig(
                attn=args.backend,
                linear="torch",
                dispatcher="torch",
                experts="torch_mm",
                swa_varlen=args.swa_varlen,
                csa_forward_full_blocks=args.csa_forward_full_blocks,
                csa_backward_full_blocks=args.csa_backward_full_blocks,
            )
            if args.swa_varlen or args.csa_forward_full_blocks or args.csa_backward_full_blocks
            else BackendConfig(attn=args.backend, linear="torch", dispatcher="torch", experts="torch_mm")
        )
        model = Step5ForCausalLM(config, backend=backend).cuda()
        tokens = torch.randint(1, 127, (1, sum(lengths) + padded), device="cuda")
        positions = torch.cat(
            [torch.arange(length) for length in lengths] + [torch.zeros(padded, dtype=torch.long)]
        ).cuda()[None]
        padding = torch.arange(tokens.shape[1], device="cuda")[None] >= sum(lengths) if padded else None
        report = compare_cp(
            model,
            tokens,
            positions,
            padding,
            stage_output=None if args.stage_output is None else args.stage_output / scenario,
        )
        report["padding_queries"] = padded
        report["packed"] = len(lengths) > 1
        report["scenario"] = scenario
        report["swa_varlen"] = args.swa_varlen
        report["csa_forward_full_blocks"] = args.csa_forward_full_blocks
        report["csa_backward_full_blocks"] = args.csa_backward_full_blocks
        print(json.dumps(report), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
