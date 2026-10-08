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
"""

import argparse
import gc
import json
import os
from typing import TypedDict

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn import functional as F
from typing_extensions import NotRequired

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.step5.configuration import Step5Config
from nemo_automodel.components.models.step5.model import Step5ForCausalLM
from nemo_automodel.components.moe.parallelizer import apply_cp


class GradientMetrics(TypedDict):
    relative_rms: float
    max_abs: float


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


def compare_cp(model: Step5ForCausalLM, tokens: torch.Tensor, positions: torch.Tensor) -> CPReport:
    """Compare a replicated model before/after native CP setup on identical inputs."""
    rank = dist.get_rank()
    length = tokens.shape[1]
    labels = tokens.roll(-1, dims=1)
    labels[:, -1] = -100
    labels[:, :-1].masked_fill_(positions[:, 1:] == 0, -100)
    count = (labels != -100).sum()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    baseline_logits = model(tokens, position_ids=positions).logits
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

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    batch = {"input_ids": tokens.clone(), "labels": labels.clone(), "position_ids": positions.clone()}
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("flex", "fa4"), default="fa4")
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    assert dist.get_world_size() == 2
    for packed in (False, True):
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
            sliding_window=65,
            sparse_config=dict(
                proxy_dim=32,
                sparse_indexer_rope_dim=8,
                sparse_indexer_num_heads=4,
                num_provider_groups=2,
                topk=4,
                region_block_size=8,
            ),
        )
        backend = BackendConfig(attn=args.backend, linear="torch", dispatcher="torch", experts="torch_mm")
        model = Step5ForCausalLM(config, backend=backend).cuda()
        tokens = torch.randint(1, 127, (1, 263), device="cuda")
        positions = (torch.cat((torch.arange(19), torch.arange(244))) if packed else torch.arange(263)).cuda()[None]
        report = compare_cp(model, tokens, positions)
        report["packed"] = packed
        print(json.dumps(report), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
