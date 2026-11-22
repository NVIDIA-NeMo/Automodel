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

"""Exercise split MTP checkpoint loading through distributed Checkpointer destinations."""

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_5_moe.model import Qwen3_5MoeForCausalLM

pytestmark = pytest.mark.cpu_only


def _model(path: Path) -> Qwen3_5MoeForCausalLM:
    config = Qwen3_5MoeTextConfig(
        vocab_size=32,
        hidden_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=256,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=32,
        layer_types=["full_attention"],
        mtp_num_hidden_layers=1,
        dtype=torch.bfloat16,
        _name_or_path=str(path),
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]},
    )
    return Qwen3_5MoeForCausalLM(
        config,
        backend=BackendConfig(
            linear="torch",
            attn="sdpa",
            rms_norm="torch",
            experts="torch_mm",
            dispatcher="torch",
            enable_hf_state_dict_adapter=True,
        ),
    )


def _write_checkpoint(path: Path, mtp_format: str, base_fp8: bool) -> dict[str, torch.Tensor]:
    model = _model(path)
    torch.manual_seed(42)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.02, 0.02)
    expected = {key: value.clone() for key, value in model.state_dict().items()}
    # Only ordinary key conversion comes from the adapter. Expert values and their
    # numerical oracle below are constructed independently of its FP8 helpers.
    checkpoint = model.state_dict_adapter.to_hf(model.state_dict())
    for prefix, quantized in (("model.layers.0", base_fp8), ("mtp.layers.0", mtp_format == "fp8")):
        if (prefix.startswith("model.") and not base_fp8) or (prefix.startswith("mtp.") and mtp_format == "grouped"):
            continue
        checkpoint.pop(f"{prefix}.mlp.experts.gate_up_proj")
        checkpoint.pop(f"{prefix}.mlp.experts.down_proj")
        gate_up, down = [], []
        for expert in range(2):
            values = {}
            for index, projection in enumerate(("gate_proj", "up_proj", "down_proj")):
                weight = ((torch.arange(256 * 256).reshape(256, 256) % 11) - 5).to(torch.float8_e4m3fn)
                # FP32 scales also detect accidentally rounding load destinations to BF16.
                scale = torch.tensor([[0.2509, 0.503], [1.007, 2.009]]) * (expert + 1) * (index + 1)
                if prefix.startswith("model."):
                    scale = scale.bfloat16()
                block_ids = torch.arange(256) // 128
                dense = (weight.float() * scale.float()[block_ids[:, None], block_ids[None, :]]).bfloat16()
                key = f"{prefix}.mlp.experts.{expert}.{projection}.weight"
                checkpoint[key] = weight if quantized else dense
                if quantized:
                    checkpoint[f"{key}_scale_inv"] = scale
                values[projection] = dense
            gate_up.append(torch.cat((values["gate_proj"].T, values["up_proj"].T), dim=1))
            down.append(values["down_proj"].T)
        expected[f"{prefix}.mlp.experts.gate_and_up_projs"] = torch.stack(gate_up)
        expected[f"{prefix}.mlp.experts.down_projs"] = torch.stack(down)
    save_file({key: value.contiguous() for key, value in checkpoint.items()}, path / "model.safetensors")
    if base_fp8:
        (path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {key: "model.safetensors" for key in checkpoint}})
        )
    model.config.save_pretrained(path)
    return expected


def _load_worker(rank: int, path: str, ep_size: int, missing_scales: bool) -> None:
    root = Path(path)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root / 'rendezvous'}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=45),
    )
    try:
        mesh = init_device_mesh("cpu", (2 // ep_size, ep_size), mesh_dim_names=("ep_shard", "ep"))
        model = _model(root)
        placements = [Shard(1), Shard(0)] if ep_size == 1 else [Shard(0)]
        expert_mesh = mesh if ep_size == 1 else mesh["ep"]
        for name, parameter in list(model.named_parameters()):
            if name.endswith(("experts.gate_and_up_projs", "experts.down_projs")):
                parent_name, parameter_name = name.rsplit(".", 1)
                model.get_submodule(parent_name).register_parameter(
                    parameter_name,
                    torch.nn.Parameter(distribute_tensor(torch.zeros_like(parameter), expert_mesh, placements)),
                )
        config = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(root / "output"),
            model_save_format="safetensors",
            model_cache_dir=str(root),
            model_repo_id=str(root),
            save_consolidated=False,
        )
        checkpointer = Checkpointer(config, dp_rank=rank, tp_rank=0, pp_rank=0, moe_mesh=mesh)
        try:
            if missing_scales:
                with pytest.raises(CheckpointException, match="weight_scale_inv"):
                    checkpointer.load_model(model, model_path=str(root), is_init_step=True)
                return
            checkpointer.load_model(model, model_path=str(root), is_init_step=True)
            expected = torch.load(root / "expected.pt", weights_only=True)
            for key, actual in model.state_dict().items():
                reference = expected[key]
                if isinstance(actual, DTensor):
                    if ep_size == 2:
                        reference = reference[rank : rank + 1]
                    else:
                        reference = reference[:, rank * 128 : (rank + 1) * 128]
                    actual = actual.to_local()
                torch.testing.assert_close(actual, reference, rtol=0, atol=0, msg=lambda message: f"{key}: {message}")
        finally:
            checkpointer.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.runtime_budget(30, hard_timeout=90, reason="Two Gloo workers load a real Qwen MTP checkpoint through DCP.")
@pytest.mark.parametrize(
    "ep_size,mtp_format,base_fp8,missing_scales",
    [
        (2, "fp8", False, False),
        (1, "fp8", False, False),
        (2, "fp8", True, False),
        (2, "bf16", True, False),
        (2, "grouped", True, False),
        (2, "fp8", False, True),
    ],
)
def test_distributed_mtp_checkpoint_load(
    tmp_path: Path,
    ep_size: int,
    mtp_format: str,
    base_fp8: bool,
    missing_scales: bool,
) -> None:
    expected = _write_checkpoint(tmp_path, mtp_format, base_fp8)
    torch.save(expected, tmp_path / "expected.pt")
    if missing_scales:
        checkpoint = load_file(tmp_path / "model.safetensors")
        checkpoint = {key: value for key, value in checkpoint.items() if not key.endswith("_scale_inv")}
        save_file(checkpoint, tmp_path / "model.safetensors")
    context = mp.spawn(
        _load_worker,
        args=(str(tmp_path), ep_size, missing_scales),
        nprocs=2,
        join=False,
    )
    try:
        deadline = time.monotonic() + 75
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Distributed MTP checkpoint loading timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
