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

"""Two-GPU forward and backward parity for four-layer Inkling Ulysses CP.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_inkling_ulysses_cp.py
"""

import os
import sys
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def _build_config() -> Any:
    """Build a tiny Inkling config containing full and sliding attention."""
    from transformers.models.inkling.configuration_inkling import InklingConfig

    config = InklingConfig(
        text_config={
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "swa_num_attention_heads": 4,
            "swa_num_key_value_heads": 2,
            "swa_head_dim": 16,
            "sliding_window_size": 16,
            "d_rel": 4,
            "rel_extent": 32,
            "vocab_size": 128,
            "moe_intermediate_size": 32,
            "n_routed_experts": 8,
            "num_experts_per_tok": 2,
            "n_shared_experts": 2,
            "route_scale": 8.0,
            "dense_intermediate_size": 96,
            "dense_mlp_idx": 2,
            "conv_kernel_size": 4,
            "max_position_embeddings": 256,
            "log_scaling_n_floor": 32,
            "logits_mup_width_multiplier": 4.0,
            "use_cache": False,
        },
        vision_config={"patch_size": 8, "temporal_patch_size": 2, "num_channels": 3, "n_layers": 2},
        audio_config={"n_mel_bins": 8, "mel_vocab_size": 16},
        image_token_id=126,
        audio_token_id=127,
        torch_dtype="bfloat16",
        _attn_implementation="sdpa",
    )
    config.text_config.layer_types = ["hybrid_sliding", "hybrid", "hybrid_sliding", "hybrid"]
    return config


def _gather_sequence(local: torch.Tensor) -> torch.Tensor:
    """Gather ``[batch, local_sequence, hidden]`` tensors in CP-rank order."""
    parts = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, local.contiguous())
    return torch.cat(parts, dim=1)


def main() -> None:
    """Compare CP2 FlexAttention with CP1 SDPA on a four-layer model."""
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from torch.distributed.device_mesh import init_device_mesh
    from transformers.models.inkling.modeling_inkling import (
        InklingForConditionalGeneration as HFInklingForConditionalGeneration,
    )

    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.inkling.model import InklingForConditionalGeneration

    torch.manual_seed(1234)
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )
    reference = InklingForConditionalGeneration.from_config(_build_config(), backend=backend).to(device).eval()
    candidate = InklingForConditionalGeneration.from_config(_build_config(), backend=backend).to(device).eval()
    hf_reference = HFInklingForConditionalGeneration(_build_config()).to(device).eval()
    reference.load_state_dict(reference.state_dict_adapter.from_hf(hf_reference.state_dict()))
    del hf_reference
    candidate.load_state_dict(reference.state_dict())

    for parameter in reference.parameters():
        dist.broadcast(parameter.data, src=0)
    candidate.load_state_dict(reference.state_dict())

    sequence_length = 64
    valid_length = 57
    input_ids = torch.randint(2, 120, (1, sequence_length), device=device)
    labels = torch.randint(0, 120, (1, sequence_length), device=device)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[:, valid_length:] = 0
    labels[:, valid_length:] = -100
    dist.broadcast(input_ids, src=0)
    dist.broadcast(labels, src=0)

    reference_logits = reference(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    ).logits

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    batch = {
        "input_ids": input_ids.clone(),
        "attention_mask": attention_mask.clone(),
        "labels": labels.clone(),
    }
    sharder = ContextParallelSharder(candidate, mesh, batch)
    train_context, batch = sharder.shard(batch)
    local_labels = batch.pop("labels")
    with train_context():
        local_logits = candidate(**batch, use_cache=False).logits
    candidate_logits = _gather_sequence(local_logits)[:, :sequence_length]

    forward_diff = (candidate_logits[:, :valid_length].float() - reference_logits[:, :valid_length].float()).abs()
    reference_loss = torch.nn.functional.cross_entropy(
        reference_logits.float().flatten(0, 1),
        labels.flatten(),
        reduction="sum",
    )
    local_loss = torch.nn.functional.cross_entropy(
        local_logits.float().flatten(0, 1),
        local_labels.flatten(),
        reduction="sum",
    )
    reference_loss.backward()
    local_loss.backward()

    max_grad_diff = 0.0
    for (reference_name, reference_parameter), (candidate_name, candidate_parameter) in zip(
        reference.named_parameters(),
        candidate.named_parameters(),
        strict=True,
    ):
        if reference_parameter.grad is None:
            if candidate_parameter.grad is not None:
                raise AssertionError(f"Unexpected CP gradient for {candidate_name}")
            continue
        if candidate_parameter.grad is None:
            raise AssertionError(f"Missing CP gradient for {candidate_name}")
        dist.all_reduce(candidate_parameter.grad)
        grad_diff = (candidate_parameter.grad.float() - reference_parameter.grad.float()).abs().max().item()
        max_grad_diff = max(max_grad_diff, grad_diff)
        if reference_name != candidate_name:
            raise AssertionError(f"Parameter order mismatch: {reference_name} != {candidate_name}")

    loss = local_loss.detach()
    dist.all_reduce(loss)
    loss_diff = abs(loss.item() - reference_loss.item())
    passed = forward_diff.max().item() < 1e-2 and loss_diff < 1e-2 and max_grad_diff < 2e-2
    if rank == 0:
        print(
            f"Inkling Ulysses: mean_logit_diff={forward_diff.mean().item():.4e} "
            f"max_logit_diff={forward_diff.max().item():.4e} "
            f"loss_diff={loss_diff:.4e} max_grad_diff={max_grad_diff:.4e}"
        )
        print("RESULT:", "PASS" if passed else "FAIL")

    status = torch.tensor(int(passed), device=device)
    dist.all_reduce(status, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    sys.exit(0 if status.item() else 1)


if __name__ == "__main__":
    main()
