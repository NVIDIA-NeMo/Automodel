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

"""Two-rank CP1/CP2 parity test for model-owned MTP+CP preparation."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
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


def _qwen3_5_model():
    from transformers.models.qwen3_5.configuration_qwen3_5 import (
        Qwen3_5Config,
        Qwen3_5TextConfig,
        Qwen3_5VisionConfig,
    )

    from nemo_automodel.components.models.qwen3_5.model import Qwen3_5ForConditionalGeneration

    text_config = Qwen3_5TextConfig(
        vocab_size=128,
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        layer_types=["full_attention"] * 4,
        mtp_num_hidden_layers=1,
        torch_dtype="float32",
    )
    vision_config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
        out_hidden_size=64,
    )
    config = Qwen3_5Config(
        architectures=["Qwen3_5ForConditionalGeneration"],
        text_config=text_config.to_dict(),
        vision_config=vision_config.to_dict(),
        image_token_id=125,
        video_token_id=126,
        vision_start_token_id=127,
    )
    return Qwen3_5ForConditionalGeneration(config, backend=_backend())


def _model_from_name(name: str):
    builders = {
        "deepseek_v4": _deepseek_model,
        "step3p7": _step_model,
        "qwen3_5": _qwen3_5_model,
    }
    return builders[name]()


def _initialize_model(name: str, device: torch.device):
    torch.manual_seed(2026)
    torch.cuda.manual_seed(2026)
    model = _model_from_name(name).to(device).train()
    model.initialize_weights(buffer_device=device, dtype=torch.bfloat16)
    return model


def _make_batch(name: str, device: torch.device) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(1234)
    seq_len = 64
    input_ids = torch.randint(2, 120, (1, seq_len), generator=generator, device=device)
    labels = torch.roll(input_ids, shifts=-1, dims=1)
    labels[:, -1] = _IGNORE_INDEX
    return {"input_ids": input_ids, "labels": labels}


def _losses(
    model,
    output,
    labels: torch.Tensor,
    *,
    mtp_targets: tuple[torch.Tensor, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    num_label_tokens = int((labels != _IGNORE_INDEX).sum().item())
    loss_fn = MaskedCrossEntropy()
    base_loss = loss_fn(output.logits, labels, num_label_tokens=num_label_tokens)
    mtp_logits = getattr(output, "mtp_per_depth_logits", None)
    mtp_loss = calculate_mtp_loss(
        loss_fn,
        mtp_per_depth_h=output.mtp_per_depth_h if mtp_logits is None else None,
        mtp_per_depth_logits=mtp_logits,
        mtp_per_depth_targets=mtp_targets,
        labels=labels,
        model=model,
        scaling_factor=0.1,
        num_label_tokens=num_label_tokens,
        ignore_index=_IGNORE_INDEX,
        return_per_depth=True,
    )
    return base_loss, mtp_loss.loss, base_loss + mtp_loss.loss, mtp_loss.per_depth_losses


def _first_mtp_grad(model) -> torch.Tensor:
    return next(parameter.grad for parameter in model.mtp.parameters() if parameter.grad is not None)


def _run_cp1(name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model = _initialize_model(name, device)
    batch = _make_batch(name, device)
    output = model(**batch)
    losses = _losses(model, output, batch["labels"])
    losses[2].backward()
    grad = _first_mtp_grad(model).detach().clone()
    values = torch.stack([losses[0], losses[1], losses[2], *losses[3]]).detach()
    del model, output, losses
    torch.cuda.empty_cache()
    return values, grad


def _run_cp2(name: str, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    model = _initialize_model(name, device)
    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    cp_mesh = mesh["cp"]
    apply_cp(model, cp_mesh)
    batch = _make_batch(name, device)
    num_label_tokens = int((batch["labels"] != _IGNORE_INDEX).sum().item())
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

    with train_ctx():
        output = model(**batch)
        loss_fn = MaskedCrossEntropy()
        base_loss = loss_fn(output.logits, local_labels, num_label_tokens=num_label_tokens)
        mtp_logits = getattr(output, "mtp_per_depth_logits", None)
        mtp_loss = calculate_mtp_loss(
            loss_fn,
            mtp_per_depth_h=output.mtp_per_depth_h if mtp_logits is None else None,
            mtp_per_depth_logits=mtp_logits,
            mtp_per_depth_targets=mtp_targets,
            labels=local_labels,
            model=model,
            scaling_factor=0.1,
            num_label_tokens=num_label_tokens,
            ignore_index=_IGNORE_INDEX,
            return_per_depth=True,
        )
        total_loss = base_loss + mtp_loss.loss
        (total_loss * dist.get_world_size()).backward()

    grad = _first_mtp_grad(model)
    dist.all_reduce(grad, group=cp_mesh.get_group())
    grad.div_(dist.get_world_size())
    local_values = torch.stack([base_loss, mtp_loss.loss, total_loss, *mtp_loss.per_depth_losses]).detach()
    dist.all_reduce(local_values, group=cp_mesh.get_group())
    return local_values, grad.detach()


def _run_model(name: str, device: torch.device) -> dict[str, list[float]]:
    cp1_values, cp1_grad = _run_cp1(name, device)
    cp2_values, cp2_grad = _run_cp2(name, device)

    torch.testing.assert_close(cp2_values, cp1_values, rtol=5e-3, atol=5e-3)
    torch.testing.assert_close(cp2_grad, cp1_grad, rtol=1e-2, atol=1e-2)
    assert torch.isfinite(cp2_grad).all()
    assert torch.count_nonzero(cp2_grad) > 0
    return {
        "cp1": cp1_values.float().cpu().tolist(),
        "cp2": cp2_values.float().cpu().tolist(),
    }


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
def test_mtp_cp_two_rank_loss_and_gradient_parity() -> None:
    """All newly enabled model families match CP1 base/MTP losses and MTP gradients."""
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
        default=["deepseek_v4", "step3p7", "qwen3_5"],
        choices=["deepseek_v4", "step3p7", "qwen3_5"],
    )
    args = parser.parse_args()
    if not args.worker:
        parser.error("run under torch.distributed.run with --worker")
    _run_worker(args.models)
