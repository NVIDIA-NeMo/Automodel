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

"""Two-GPU CP1/CP2 loss parity for Qwen3.5-MoE VLM with MTP.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_qwen3_5_moe_mtp_cp_parity.py
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig, Qwen3_5MoeTextConfig

from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForConditionalGeneration
from nemo_automodel.components.moe.parallelizer import apply_cp


def _config() -> Qwen3_5MoeConfig:
    """Build a four-backbone-layer, one-MTP-depth Qwen VLM config."""
    text = Qwen3_5MoeTextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention", "linear_attention"] * 2,
        mtp_num_hidden_layers=1,
        torch_dtype="bfloat16",
    )
    vision = {
        "depth": 1,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_heads": 4,
        "in_channels": 3,
        "patch_size": 2,
        "spatial_merge_size": 1,
        "temporal_patch_size": 1,
        "out_hidden_size": 64,
        "num_position_embeddings": 16,
    }
    config = Qwen3_5MoeConfig(text_config=text.to_dict(), vision_config=vision)
    config.image_token_id = 120
    config.video_token_id = 121
    config.vision_start_token_id = 122
    config.vision_end_token_id = 123
    return config


def _backend() -> BackendConfig:
    """Use torch modules and SDPA so the test exercises PyTorch CP transport."""
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )


def _losses(
    model: Qwen3_5MoeForConditionalGeneration,
    logits: torch.Tensor,
    mtp_hidden: list[torch.Tensor],
    labels: torch.Tensor,
    mtp_targets: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base, scaled-MTP, and total losses over global-order tensors."""
    valid_tokens = int((labels != -100).sum().item())
    base_loss = (
        F.cross_entropy(
            logits.float().reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )
        / valid_tokens
    )
    mtp_loss = calculate_mtp_loss(
        MaskedCrossEntropy(ignore_index=-100),
        mtp_per_depth_h=mtp_hidden,
        mtp_per_depth_targets=mtp_targets,
        labels=labels,
        model=model,
        scaling_factor=model.mtp_config.loss_scaling_factor,
        num_label_tokens=valid_tokens,
        ignore_index=-100,
    )
    return base_loss, mtp_loss, base_loss + mtp_loss


def main() -> None:
    """Compare recipe-shaped CP2 execution with the same model at CP1."""
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    if world != 2:
        raise RuntimeError(f"This parity test requires exactly 2 GPUs, got {world}")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    torch.manual_seed(1234)
    model = Qwen3_5MoeForConditionalGeneration(_config(), backend=_backend()).to(device)
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    model.train()
    for parameter in model.parameters():
        dist.broadcast(parameter.data, src=0)
    for buffer in model.buffers():
        dist.broadcast(buffer.data, src=0)

    sequence = 32
    torch.manual_seed(5678)
    input_ids = torch.randint(2, 100, (1, sequence), device=device)
    input_ids[0, 4] = model.config.vision_start_token_id
    input_ids[0, 5:9] = model.config.image_token_id
    input_ids[0, 9] = model.config.vision_end_token_id
    pixel_values = torch.randn(4, 12, device=device, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor([[1, 2, 2]], device=device, dtype=torch.long)
    for tensor in (input_ids, pixel_values, image_grid_thw):
        dist.broadcast(tensor, src=0)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = -100

    reference_batch = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "pixel_values": pixel_values.clone(),
        "image_grid_thw": image_grid_thw.clone(),
    }
    reference_prepared = model.prepare_model_inputs_for_cp(reference_batch)
    reference_batch["position_ids"] = reference_prepared["position_ids"]
    reference_mtp = model.prepare_mtp_inputs_for_cp(reference_batch, ignore_index=-100)
    with torch.no_grad():
        reference_output = model(
            input_ids=reference_batch["input_ids"],
            position_ids=reference_batch["position_ids"],
            pixel_values=reference_batch["pixel_values"],
            image_grid_thw=reference_batch["image_grid_thw"],
        )
        reference_losses = _losses(
            model,
            reference_output.logits,
            reference_output.mtp_per_depth_h,
            labels,
            reference_mtp.targets,
        )

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    apply_cp(model, mesh["cp"])
    cp_batch = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "pixel_values": pixel_values.clone(),
        "image_grid_thw": image_grid_thw.clone(),
    }
    sharder = ContextParallelSharder(model, mesh, cp_batch)
    cp_mtp = model.prepare_mtp_inputs_for_cp(cp_batch, ignore_index=-100)
    train_ctx, cp_batch = sharder.shard(cp_batch)
    local_mtp_input_ids = tuple(sharder.shard_token_tensor(ids, seq_dim=1, fill=0) for ids in cp_mtp.input_ids)
    local_mtp_position_ids = tuple(
        sharder.shard_token_tensor(ids, seq_dim=cp_mtp.position_ids_seq_dim, fill=0) for ids in cp_mtp.position_ids
    )
    local_mtp_valid_masks = tuple(
        sharder.shard_token_tensor(mask, seq_dim=1, fill=False) for mask in cp_mtp.valid_masks
    )
    local_mtp_targets = tuple(sharder.shard_token_tensor(targets, seq_dim=1, fill=-100) for targets in cp_mtp.targets)
    cp_batch.update(
        {
            "mtp_per_depth_input_ids": local_mtp_input_ids,
            "mtp_per_depth_position_ids": local_mtp_position_ids,
            "mtp_per_depth_valid_masks": local_mtp_valid_masks,
        }
    )
    cp_batch.pop("labels")
    with torch.no_grad(), train_ctx():
        cp_output = model(**cp_batch)

    gathered_logits = sharder.gather_token_tensor(cp_output.logits, seq_dim=1, trim=True)
    gathered_mtp_hidden = [
        sharder.gather_token_tensor(hidden, seq_dim=1, trim=True) for hidden in cp_output.mtp_per_depth_h
    ]
    gathered_mtp_targets = tuple(
        sharder.gather_token_tensor(targets, seq_dim=1, trim=True) for targets in local_mtp_targets
    )
    gathered_mtp_input_ids = tuple(
        sharder.gather_token_tensor(ids, seq_dim=1, trim=True) for ids in local_mtp_input_ids
    )
    gathered_mtp_position_ids = tuple(
        sharder.gather_token_tensor(ids, seq_dim=cp_mtp.position_ids_seq_dim, trim=True)
        for ids in local_mtp_position_ids
    )

    for actual, expected in zip(gathered_mtp_input_ids, reference_mtp.input_ids):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(gathered_mtp_position_ids, reference_mtp.position_ids):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual, expected in zip(gathered_mtp_targets, reference_mtp.targets):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    cp_losses = _losses(model, gathered_logits, gathered_mtp_hidden, labels, gathered_mtp_targets)
    names = ("base", "mtp", "total")
    differences = [
        abs(cp.detach().item() - reference.detach().item()) for cp, reference in zip(cp_losses, reference_losses)
    ]
    tolerance = 5e-2
    for name, reference, cp, difference in zip(names, reference_losses, cp_losses, differences):
        if rank == 0:
            print(f"{name}: cp1={reference.detach().item():.8f} cp2={cp.detach().item():.8f} abs_diff={difference:.3e}")
        if difference >= tolerance:
            raise AssertionError(f"{name} loss parity failed: abs_diff={difference:.3e} >= {tolerance:.3e}")

    dist.barrier()
    if rank == 0:
        print("RESULT: PASS (Qwen3.5-MoE MTP CP1/CP2 input and loss parity)")
    dist.destroy_process_group()
    sys.exit(0)


if __name__ == "__main__":
    main()
