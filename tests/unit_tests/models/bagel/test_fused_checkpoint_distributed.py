# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Two-rank FSDP2 checkpoint regression for BAGEL fused projections."""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _checkpoint_worker(rank: int, world_size: int, port: int, checkpoint_root: str, device_type: str) -> None:
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
            model_save_format="torch_save",
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
            metadata = dcp.FileSystemReader(os.path.join(weights_path, "model")).read_metadata()
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


def test_fused_projection_dcp_resume_preserves_sharded_layout(tmp_path) -> None:
    device_type = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() >= 2 else "cpu"
    mp.spawn(
        _checkpoint_worker,
        args=(2, _free_port(), str(tmp_path), device_type),
        nprocs=2,
        join=True,
    )
