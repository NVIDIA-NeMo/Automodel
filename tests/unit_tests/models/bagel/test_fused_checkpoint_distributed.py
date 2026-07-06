# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Two-rank FSDP2 checkpoint regression for BAGEL fused projections."""

from __future__ import annotations

import glob
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _checkpoint_worker(
    rank: int,
    world_size: int,
    port: int,
    checkpoint_root: str,
    device_type: str,
    model_save_format: str,
) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world_size)

    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard
        from transformers import Qwen2Config

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig
        from nemo_automodel.components.models.bagel.modeling_qwen2_packed import Qwen2ForCausalLM
        from nemo_automodel.components.models.bagel.state_dict_adapter import BagelStateDictAdapter

        config = Qwen2Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            attention_dropout=0.0,
        )
        config.qk_norm = True
        config.layer_module = "Qwen2DecoderLayer"
        config.fused_projections = True

        torch.manual_seed(1234)
        device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
        model = Qwen2ForCausalLM(config).to(device=device)
        model.state_dict_adapter = BagelStateDictAdapter(
            config=SimpleNamespace(text_config=config),
            stage="stage1",
        )
        fully_shard(model, mesh=init_device_mesh(device_type, (world_size,)))

        qkv_param = dict(model.named_parameters())["model.layers.0.self_attn.qkv_proj.weight"]
        expected_local = qkv_param.to_local().detach().clone()

        checkpoint_config = CheckpointingConfig(
            checkpoint_dir=checkpoint_root,
            model_save_format=model_save_format,
            save_consolidated=False,
            model_repo_id="test/bagel-fused",
        )
        checkpointer = Checkpointer(
            checkpoint_config,
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )
        weights_path = os.path.join(checkpoint_root, "step_1")
        checkpointer.save_model(model, weights_path)
        dist.barrier()

        if rank == 0:
            model_path = os.path.join(weights_path, "model")
            if model_save_format == "safetensors":
                from safetensors import safe_open

                checkpoint_keys = set()
                for shard_path in glob.glob(os.path.join(model_path, "**", "*.safetensors"), recursive=True):
                    with safe_open(shard_path, framework="pt", device="cpu") as shard:
                        checkpoint_keys.update(shard.keys())
                assert not os.path.exists(os.path.join(model_path, ".hf_metadata"))
                assert not os.path.exists(os.path.join(model_path, "consolidate.sh"))
            else:
                metadata = dcp.FileSystemReader(model_path).read_metadata()
                checkpoint_keys = set(metadata.state_dict_metadata)
            assert "model.layers.0.self_attn.qkv_proj.weight" in checkpoint_keys
            assert "model.layers.0.self_attn.q_proj.weight" not in checkpoint_keys

        with torch.no_grad():
            qkv_param.zero_()
        checkpointer.load_model(model, os.path.join(weights_path, "model"), is_init_step=False)
        torch.testing.assert_close(qkv_param.to_local(), expected_local)
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("model_save_format", ["torch_save", "safetensors"])
def test_fused_projection_dcp_resume_preserves_sharded_layout(tmp_path, model_save_format) -> None:
    device_type = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "cpu"
    mp.spawn(
        _checkpoint_worker,
        args=(2, _free_port(), str(tmp_path), device_type, model_save_format),
        nprocs=2,
        join=True,
    )


def _hf_init_worker(rank: int, world_size: int, port: int, checkpoint_root: str, device_type: str) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    if device_type == "cuda":
        torch.cuda.set_device(rank)
    dist.init_process_group("nccl" if device_type == "cuda" else "gloo", rank=rank, world_size=world_size)

    try:
        from safetensors.torch import save_file
        from torch import nn
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
        from nemo_automodel.components.checkpoint.config import CheckpointingConfig
        from nemo_automodel.components.models.bagel.state_dict_adapter import BagelStateDictAdapter

        source_dir = os.path.join(checkpoint_root, "hf_source")
        q = torch.arange(16, dtype=torch.float32).reshape(4, 4)
        k = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 100
        v = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 200
        if rank == 0:
            os.makedirs(source_dir, exist_ok=True)
            prefix = "language_model.model.layers.0.self_attn"
            save_file(
                {
                    f"{prefix}.q_proj.weight": q,
                    f"{prefix}.k_proj.weight": k,
                    f"{prefix}.v_proj.weight": v,
                },
                os.path.join(source_dir, "model.safetensors"),
            )
        dist.barrier()

        class TinyBagel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.language_model = nn.Module()
                self.model.language_model.model = nn.Module()
                layer = nn.Module()
                layer.self_attn = nn.Module()
                layer.self_attn.qkv_proj = nn.Linear(4, 8, bias=False)
                self.model.language_model.model.layers = nn.ModuleList([layer])

        text_config = SimpleNamespace(
            hidden_size=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            fused_projections=True,
        )
        device = torch.device(f"cuda:{rank}" if device_type == "cuda" else "cpu")
        model = TinyBagel().to(device=device)
        model.state_dict_adapter = BagelStateDictAdapter(
            config=SimpleNamespace(text_config=text_config),
            stage="stage1",
        )
        fully_shard(model, mesh=init_device_mesh(device_type, (world_size,)))

        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=checkpoint_root,
                model_save_format="safetensors",
                model_cache_dir=source_dir,
                model_repo_id=source_dir,
                save_consolidated=False,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=None,
        )
        checkpointer.load_model(model, source_dir, is_init_step=True)

        qkv_param = dict(model.named_parameters())["model.language_model.model.layers.0.self_attn.qkv_proj.weight"]
        expected_local = torch.cat([q, k, v], dim=0).chunk(world_size, dim=0)[rank].to(device)
        torch.testing.assert_close(qkv_param.to_local(), expected_local)
        dist.barrier()
    finally:
        dist.destroy_process_group()


def test_fused_projection_hf_init_after_sharding(tmp_path) -> None:
    device_type = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "cpu"
    mp.spawn(
        _hf_init_worker,
        args=(2, _free_port(), str(tmp_path), device_type),
        nprocs=2,
        join=True,
    )
