# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real DCP conversion with expert shards and attention heads split across ranks."""

import socket
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.muse_spark.configuration_muse_spark import MuseSparkConfig
from nemo_automodel.components.models.muse_spark.model import MuseSpark1xOmniForCausalLM
from tests.unit_tests.models.muse_spark.conftest import tiny_backend
from tests.unit_tests.models.muse_spark.test_hf_release import tiny_hf_config


def _worker(rank: int, port: int, root: str) -> None:
    """Check loading, gradients and export against an unsharded native model."""
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2, timeout=timedelta(seconds=180)
    )
    try:
        data = tiny_hf_config().to_dict()
        data["text_config"].update(num_attention_heads=2, num_key_value_heads=1, head_dim=8)
        data["vision_config"]["num_attention_heads"] = 1
        config = MuseSparkConfig(**data)
        torch.manual_seed(42)
        reference = MuseSpark1xOmniForCausalLM(config, backend=tiny_backend()).cuda()
        reference.initialize_weights(torch.device("cuda", rank), dtype=torch.float32)
        checkpoint = Path(root) / "source"
        raw = reference.state_dict_adapter.to_hf(reference.state_dict())
        if rank == 0:
            checkpoint.mkdir()
            save_file(
                {key: value.detach().cpu().contiguous() for key, value in raw.items()}, checkpoint / "model.safetensors"
            )
            config.save_pretrained(checkpoint)
        dist.barrier()
        torch.manual_seed(71 + rank)
        actual = MuseSpark1xOmniForCausalLM(config, backend=tiny_backend()).cuda()
        actual.initialize_weights(torch.device("cuda", rank), dtype=torch.float32)
        fully_shard(actual, mesh=init_device_mesh("cuda", (2,)))
        state = actual.state_dict()
        key = "model.layers.0.self_attn.k_proj.weight"
        assert isinstance(state[key], DTensor)
        assert state[key].to_local().shape[0] == config.head_dim // 2
        expert_key = "model.layers.0.feed_forward.experts.gate_and_up_projs"
        destinations = actual.state_dict_adapter.to_hf(state, for_checkpoint_load=True)
        assert (
            destinations["model.language_model.layers.0.mlp.experts.gate_up_proj"]
            .to_local()
            .untyped_storage()
            .data_ptr()
            == state[expert_key].to_local().untyped_storage().data_ptr()
        )
        checkpointer = Checkpointer(
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir=str(Path(root) / "saved"),
                model_save_format="safetensors",
                save_consolidated=False,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
        )
        with patch(
            "nemo_automodel.components.checkpoint.checkpointing._load_hf_checkpoint_preserving_dtype",
            side_effect=AssertionError("Expected DCP direct loading"),
        ):
            checkpointer.load_model(actual, str(checkpoint), is_init_step=True)
        for name, value in actual.state_dict().items():
            full = value.full_tensor() if isinstance(value, DTensor) else value
            torch.testing.assert_close(full, reference.state_dict()[name], rtol=0, atol=0)
        exported = actual.state_dict_adapter.to_hf(actual.state_dict())
        assert set(exported) == set(raw)
        for name, value in exported.items():
            full = value.full_tensor() if isinstance(value, DTensor) else value
            torch.testing.assert_close(full, raw[name], rtol=0, atol=0)

        torch.manual_seed(88)
        inputs = dict(
            input_ids=torch.tensor([[1, 61, 2, 63, 63, 3]], device=rank),
            pixel_values=torch.randn(4, 24, device=rank),
            image_grid_thw=torch.tensor([[1, 2, 2]], device=rank),
            audio_values=[torch.randn(8, device=rank)],
        )
        expected, observed = reference(**inputs).logits, actual(**inputs).logits
        torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
        upstream = torch.randn_like(expected)
        expected.backward(upstream)
        observed.backward(upstream)
        parameters = dict(reference.named_parameters())
        for name, parameter in actual.named_parameters():
            assert parameter.grad is not None, name
            grad = parameter.grad.full_tensor() if isinstance(parameter.grad, DTensor) else parameter.grad
            torch.testing.assert_close(grad, parameters[name].grad, rtol=1e-5, atol=1e-6)
        torch.optim.SGD(actual.parameters(), lr=0.01).step()
        torch.optim.SGD(reference.parameters(), lr=0.01).step()
        for name, parameter in actual.named_parameters():
            full = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            torch.testing.assert_close(full, parameters[name], rtol=1e-5, atol=1e-6)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA GPUs")
def test_v2_dcp_with_fragmented_rope_heads(tmp_path: Path) -> None:
    """The DP64 K-projection case must work when each rank owns only part of a head."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_worker, args=(port, str(tmp_path)), nprocs=2, join=True)
