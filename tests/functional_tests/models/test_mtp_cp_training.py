# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Two-rank forward/backward smoke test for model-owned MTP+CP preparation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.loss.mtp import calculate_mtp_loss
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.parallelizer import apply_cp

_RESULT_PREFIX = "MTP_CP_TRAINING_RESULT "
_IGNORE_INDEX = -100


def _backend(*, attn: str = "sdpa") -> BackendConfig:
    return BackendConfig(
        attn=attn,
        linear="torch",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        experts="torch_mm",
        enable_hf_state_dict_adapter=False,
    )


def _deepseek_model():
    from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
    from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

    config = DeepseekV4Config(
        vocab_size=128,
        hidden_size=64,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=16,
        qk_rope_head_dim=8,
        q_lora_rank=32,
        o_lora_rank=32,
        o_groups=2,
        n_routed_experts=4,
        n_shared_experts=1,
        num_experts_per_tok=2,
        routed_scaling_factor=1.5,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        topk_method="noaux_tc",
        max_position_embeddings=128,
        rope_theta=10000.0,
        rope_scaling=None,
        hc_mult=4,
        num_hash_layers=0,
        compress_ratios=[0, 0, 0, 0],
        sliding_window=32,
        num_nextn_predict_layers=1,
        rms_norm_eps=1e-6,
        torch_dtype="float32",
    )
    return DeepseekV4ForCausalLM(config, backend=_backend())


def _step_model():
    from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
    from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration

    config = Step3p7Config(
        vision_config={
            "width": 16,
            "layers": 0,
            "heads": 2,
            "num_channels": 3,
            "image_size": 8,
            "patch_size": 2,
            "mlp_ratio": 2.0,
            "hidden_act": "gelu",
            "use_ln_pre": False,
            "use_ln_post": False,
            "use_abs_posemb": False,
            "use_rope2d": False,
        },
        text_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_attention_groups": 2,
            "num_hidden_layers": 4,
            "num_nextn_predict_layers": 1,
            "mtp_base_layer_idx": 4,
            "vocab_size": 128,
            "moe_num_experts": 4,
            "moe_top_k": 2,
            "moe_intermediate_size": 32,
            "share_expert_dims": 32,
            "head_dim": 16,
            "torch_dtype": "float32",
            "moe_layers_enum": (1, 2, 3),
            "layer_types": ["full_attention"] * 5,
        },
        image_token_id=127,
    )
    return Step3p7ForConditionalGeneration(config, backend=_backend(attn="te"))


def _minimax_model():
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration

    config = MiniMaxM3VLConfig(
        vision_config={
            "hidden_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 0,
            "intermediate_size": 64,
            "patch_size": 2,
            "num_channels": 3,
            "img_token_compression_config": {"spatial_merge_size": 2, "temporal_patch_size": 2},
        },
        text_config={
            "hidden_size": 256,
            "intermediate_size": 64,
            "dense_intermediate_size": 128,
            "shared_intermediate_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "rotary_dim": 32,
            "partial_rotary_factor": 0.5,
            "vocab_size": 128,
            "max_position_embeddings": 128,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "moe_layer_freq": [0, 1, 1, 1],
            "num_mtp_modules": 1,
            "sparse_attention_config": {
                "use_sparse_attention": True,
                "sparse_index_dim": 64,
                "sparse_num_index_heads": 2,
                "sparse_topk_blocks": 2,
                "sparse_block_size": 128,
                "sparse_score_type": "max",
                "sparse_init_block": 0,
                "sparse_local_block": 1,
                "sparse_attention_freq": [0, 1, 1, 1],
                "sparse_disable_index_value": [0, 1, 1, 1],
            },
            "torch_dtype": "float32",
        },
        image_token_index=126,
        video_token_index=127,
        projector_hidden_size=256,
    )
    return MiniMaxM3SparseForConditionalGeneration(config, backend=_backend())


def _model_from_name(name: str):
    builders = {
        "deepseek_v4": _deepseek_model,
        "step3p7": _step_model,
        "minimax_m3": _minimax_model,
    }
    return builders[name]()


def _cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=_IGNORE_INDEX,
    )


def _run_model(name: str, device: torch.device) -> float:
    model = _model_from_name(name).to(device).train()
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]
    apply_cp(model, cp_mesh)

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    last_loss = 0.0
    for step in range(2):
        generator = torch.Generator(device=device).manual_seed(1234 + step)
        seq_len = 256 if name == "minimax_m3" else 64
        input_ids = torch.randint(2, 120, (1, seq_len), generator=generator, device=device)
        labels = torch.roll(input_ids, shifts=-1, dims=1)
        labels[:, -1] = _IGNORE_INDEX
        batch = {"input_ids": input_ids, "labels": labels}

        sharder = ContextParallelSharder(model, mesh, batch, padding_token_id=0, invoke_pre_embed=True)
        mtp_inputs = model.prepare_mtp_inputs_for_cp(batch, ignore_index=_IGNORE_INDEX)
        train_ctx, batch = sharder.shard(batch)
        batch["mtp_per_depth_input_ids"] = tuple(
            sharder.shard_token_tensor(ids, seq_dim=1, fill=0) for ids in mtp_inputs.input_ids
        )
        batch["mtp_per_depth_position_ids"] = tuple(
            sharder.shard_token_tensor(ids, seq_dim=mtp_inputs.position_ids_seq_dim, fill=0)
            for ids in mtp_inputs.position_ids
        )
        batch["mtp_per_depth_valid_masks"] = tuple(
            sharder.shard_token_tensor(mask, seq_dim=1, fill=False) for mask in mtp_inputs.valid_masks
        )
        mtp_targets = tuple(
            sharder.shard_token_tensor(target, seq_dim=1, fill=_IGNORE_INDEX) for target in mtp_inputs.targets
        )
        local_labels = batch.pop("labels")

        optimizer.zero_grad(set_to_none=True)
        with train_ctx():
            output = model(**batch)
            loss = _cross_entropy(output.logits, local_labels)
            mtp_logits = getattr(output, "mtp_per_depth_logits", None)
            loss = loss + calculate_mtp_loss(
                MaskedCrossEntropy(),
                mtp_per_depth_h=output.mtp_per_depth_h if mtp_logits is None else None,
                mtp_per_depth_logits=mtp_logits,
                mtp_per_depth_targets=mtp_targets,
                labels=local_labels,
                model=model,
                scaling_factor=0.1,
                ignore_index=_IGNORE_INDEX,
            )
            loss.backward()

        for parameter in model.parameters():
            if parameter.grad is not None:
                dist.all_reduce(parameter.grad, group=cp_mesh.get_group())
                parameter.grad.div_(dist.get_world_size())
        mtp_grad = next(parameter.grad for parameter in model.mtp.parameters() if parameter.grad is not None)
        assert torch.isfinite(loss)
        assert torch.isfinite(mtp_grad).all()
        assert torch.count_nonzero(mtp_grad) > 0
        optimizer.step()
        last_loss = loss.detach().item()
    return last_loss


def _run_worker(model_names: list[str]) -> None:
    dist.init_process_group("nccl")
    try:
        if dist.get_world_size() != 2:
            raise RuntimeError(f"This smoke test requires exactly two ranks, got {dist.get_world_size()}")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        results = {}
        for name in model_names:
            results[name] = _run_model(name, device)
            dist.barrier()
        if dist.get_rank() == 0:
            print(_RESULT_PREFIX + repr(results), flush=True)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires at least two CUDA devices")
def test_mtp_cp_two_rank_forward_backward() -> None:
    """All newly enabled model families complete two MTP+CP optimizer steps."""
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        str(Path(__file__).resolve()),
        "--worker",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=600, check=False)
    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert _RESULT_PREFIX in output, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["deepseek_v4", "step3p7", "minimax_m3"],
        choices=["deepseek_v4", "step3p7", "minimax_m3"],
    )
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    _run_worker(args.models)
