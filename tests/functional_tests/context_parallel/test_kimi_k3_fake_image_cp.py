# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Kimi K3 KDA context parallelism across masked fake-image spans on 2 GPUs."""

import copy
import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from tests.unit_tests.models.kimi_k3.test_processing import _fake_and_real_batch, processor  # noqa: F401


def _free_port() -> int:
    """Return an unused local TCP port for the process group."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _fake_image_cp_rank(rank: int, port: int, batch: dict[str, torch.Tensor], image_token_id: int, pad_id: int) -> None:
    """Compare CP2 against the native unsharded path on a fake-image batch.

    Args:
        rank: CP rank.
        port: Rendezvous port.
        batch: Native fake/real batch; input_ids and attention_mask have shape
            [2, sequence], and row 0 holds a masked fake-image span.
        image_token_id: Media placeholder token ID.
        pad_id: Padding token ID.
    """
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.kimi_k3.multimodal import KimiK3ForConditionalGeneration
    from tests.unit_tests.models.kimi_k3.test_multimodal import _tiny_vlm

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group("nccl", rank=rank, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (2,), mesh_dim_names=("cp",))
        config = copy.deepcopy(
            _tiny_vlm(
                vocab_size=272, media_placeholder_token_id=image_token_id, pad_token_id=pad_id, patch_size=14
            ).config
        )
        # One KDA layer followed by one MLA layer; CP flex attention needs head dims of at least 16.
        config.text_config.num_hidden_layers = 2
        config.text_config.qk_nope_head_dim = 8
        config.text_config.qk_rope_head_dim = 8
        config.text_config.v_head_dim = 16
        config.text_config.linear_attn_config = {
            "head_dim": 64,
            "num_heads": 2,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        }
        config.text_config.kda_use_fused_gate = False
        backend = BackendConfig(
            attn="eager", linear="torch", experts="torch", dispatcher="torch", enable_hf_state_dict_adapter=False
        )
        torch.manual_seed(4511)
        model = KimiK3ForConditionalGeneration(config, backend=backend).cuda()
        model.initialize_weights(torch.device("cuda"), dtype=torch.float32)
        model.eval()
        batch = {key: value.cuda() for key, value in batch.items()}
        valid = batch["attention_mask"] > 0
        kda_outputs = []
        hook = model.model.layers["0"].self_attn.register_forward_hook(
            lambda module, inputs, output: kda_outputs.append(output.detach())
        )

        with torch.no_grad():
            cp1_logits = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                image_grid_thw=batch["image_grid_thw"],
            ).logits
            model.cp_mesh = mesh["cp"]
            sharder = ContextParallelSharder(model, mesh, batch, padding_token_id=pad_id, invoke_pre_embed=True)
            context, local = sharder.shard(dict(batch))
            local.pop("labels")
            order = local["kimi_vlm_token_order"]
            with context():
                cp2_logits = model(**local).logits
        hook.remove()

        def source_order(local_tensor: torch.Tensor) -> torch.Tensor:
            """Gather [batch, local, features] shards and restore source order [batch, sequence, features]."""
            parts = [torch.empty_like(local_tensor) for _ in range(2)]
            dist.all_gather(parts, local_tensor.contiguous())
            sharded = torch.cat(parts, dim=1)
            restored = torch.empty_like(sharded).scatter_(1, order.unsqueeze(-1).expand_as(sharded), sharded)
            return restored[:, : valid.shape[1]]

        # Remaining CP differences match a batch row without masked spans (KDA output
        # ~3e-5, logits ~1e-3); an interrupted KDA state shifts them by ~1e-2 and ~1.
        torch.testing.assert_close(source_order(kda_outputs[1])[valid], kda_outputs[0][valid], atol=2e-4, rtol=0)
        torch.testing.assert_close(source_order(cp2_logits)[valid], cp1_logits[valid], atol=1e-2, rtol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="K3 KDA context parallelism needs 2 GPUs")
@pytest.mark.parametrize("path", ["pretokenized", "default_collate"])
def test_injected_fake_image_kda_cp2_matches_unsharded(processor, path):
    """CP must keep KDA state continuous across a masked fake-image span, as the unsharded path does."""
    pytest.importorskip("fla")
    batch = _fake_and_real_batch(processor, path, "white")
    mp.spawn(
        _fake_image_cp_rank,
        args=(_free_port(), batch, processor.image_token_id, processor.tokenizer.pad_token_id),
        nprocs=2,
        join=True,
    )
