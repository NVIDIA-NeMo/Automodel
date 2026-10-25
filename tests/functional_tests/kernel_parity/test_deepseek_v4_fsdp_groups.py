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
"""CUDA component parity for cross-block FP32 HC and sink FSDP units."""

import copy
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from nemo_automodel.components.distributed.parallelizer_utils import fully_shard_across_blocks
from nemo_automodel.components.models.deepseek_v4.fsdp import fully_shard_deepseek_v4
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4FP32Parameter, DeepseekV4HyperConnection

_ROLES = ("attn_hc", "ffn_hc", "self_attn.sinks_param")


class _Block(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn_hc = DeepseekV4HyperConnection(4, 32, 5, 1e-6, 1e-6, "torch")
        self.ffn_hc = DeepseekV4HyperConnection(4, 32, 5, 1e-6, 1e-6, "torch")
        self.attn_hc.init_weights(0.02)
        self.ffn_hc.init_weights(0.02)
        self.self_attn = torch.nn.Module()
        self.self_attn.sinks_param = DeepseekV4FP32Parameter(torch.randn(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix four residual streams through both HC sites and add the sinks.

        Args:
            x: Tensor of shape [batch, sequence, streams, hidden] with four streams.

        Returns:
            Tensor of shape [batch, sequence, streams, hidden].
        """
        for site in (self.attn_hc, self.ffn_hc):
            pre, post, comb = site(x)
            x = site.expand(site.collapse(x, pre), x, post, comb)
        return x + self.self_attn.sinks_param(x).to(x.dtype).view(1, 1, 4, 1)


class _Model(torch.nn.Module):
    def __init__(self, checkpointed: bool) -> None:
        super().__init__()
        blocks = [_Block(), _Block()]
        self.layers = torch.nn.ModuleList([checkpoint_wrapper(b) for b in blocks] if checkpointed else blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply both blocks.

        Args:
            x: Tensor of shape [batch, sequence, streams, hidden] with four streams.

        Returns:
            Tensor of shape [batch, sequence, streams, hidden].
        """
        for block in self.layers:
            x = block(x)
        return x


def _worker(rank: int, init_file: str, checkpointed: bool) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(13)
        model = _Model(checkpointed).cuda()
        grouped = copy.deepcopy(model)
        mesh = init_device_mesh("cuda", (2,))
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32, cast_forward_inputs=False)
        for block in model.layers:
            for role in _ROLES:
                fully_shard(
                    block.get_submodule(role),
                    mesh=mesh,
                    mp_policy=policy,
                    reshard_after_forward=not role.endswith("sinks_param"),
                )
        fully_shard_across_blocks(
            list(grouped.layers),
            _ROLES,
            mesh=mesh,
            mp_policy=policy,
            offload_policy=None,
            shard_module=fully_shard_deepseek_v4,
        )
        for candidate in (model, grouped):
            fully_shard(candidate, mesh=mesh, mp_policy=policy)
        torch.manual_seed(71 + rank)
        inputs = [torch.randn(1, 16, 4, 32, device="cuda", dtype=torch.bfloat16) for _ in range(2)]
        upstream = [torch.randn_like(x) for x in inputs]
        runs = []
        for candidate in (model, grouped):
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01, weight_decay=0.1)
            records = []
            for step, (x_base, grad) in enumerate(zip(inputs, upstream, strict=True)):
                x = x_base.clone().requires_grad_()
                output = candidate(x)
                output.backward(grad)
                records.append(
                    [output.detach(), x.grad.clone(), *[p.grad.full_tensor() for p in candidate.parameters()]]
                )
                assert all(p.dtype == torch.float32 for p in candidate.parameters())
                optimizer.step()
                records[-1].extend(p.full_tensor() for p in candidate.parameters())
                optimizer.zero_grad(set_to_none=True)
            runs.append(records)
        for expected_step, actual_step in zip(*runs, strict=True):
            for expected, actual in zip(expected_step, actual_step, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_grouped_hc_and_sinks_preserve_gradients_and_optimizer_updates(tmp_path: Path, checkpointed: bool) -> None:
    mp.spawn(_worker, args=(str(tmp_path / "init"), checkpointed), nprocs=2, join=True)


_COMPRESSOR_ROLES = (
    "self_attn.compressor.wkv",
    "self_attn.compressor.wgate",
    "self_attn.compressor.ape_param",
    "self_attn.compressor.indexer.wkv",
    "self_attn.compressor.indexer.wgate",
    "self_attn.compressor.indexer.ape_param",
)


class _AttentionBlock(torch.nn.Module):
    def __init__(self, ratio: int) -> None:
        super().__init__()
        self.complete_compressor_windows = False
        from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
        from nemo_automodel.components.models.deepseek_v4.layers import (
            DeepseekV4Attention,
            DeepseekV4RotaryEmbedding,
        )

        config = DeepseekV4Config(
            hidden_size=16,
            head_dim=8,
            qk_rope_head_dim=4,
            q_lora_rank=8,
            o_lora_rank=8,
            o_groups=1,
            num_attention_heads=2,
            num_hidden_layers=1,
            num_nextn_predict_layers=0,
            compress_ratios=[ratio],
            index_head_dim=8,
            index_n_heads=2,
            index_topk=2,
            sliding_window=8,
            torch_dtype="float32",
        )
        self.self_attn = DeepseekV4Attention(config, 0)
        self.self_attn.init_weights(torch.device("cpu"))
        if ratio == 4:
            self.self_attn.compressor.indexer.requires_grad_(False)
        self.rotary = DeepseekV4RotaryEmbedding(
            rope_theta=160000.0,
            head_dim=8,
            partial_rotary_factor=0.5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to complete, canonical local sequences.

        Args:
            x: Tensor of shape [batch, sequence, hidden].

        Returns:
            Tensor of shape [batch, sequence, hidden].
        """
        positions = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        rotary = self.rotary(x, positions)
        return (
            self.self_attn(
                x,
                position_embeddings=rotary,
                position_embeddings_compress=rotary,
                rotary_compress=self.rotary,
                position_ids=positions,
                _dsv4_all_queries_nonempty=True,
                _dsv4_complete_compressor_windows=self.complete_compressor_windows,
            )[0]
            + x
        )


def _compressor_worker(rank: int, init_file: str, checkpointed: bool) -> None:
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        torch.manual_seed(37)
        blocks = [_AttentionBlock(ratio) for ratio in (4, 128, 4, 128)]
        model = torch.nn.Sequential(*[checkpoint_wrapper(b) if checkpointed else b for b in blocks]).cuda().float()
        grouped = copy.deepcopy(model)
        mesh = init_device_mesh("cuda", (2,))
        for current in (model, grouped):
            for block in current:
                block.self_attn.compressor._set_hca_param_sync_group(mesh.get_group())
                attention_block = block._checkpoint_wrapped_module if checkpointed else block
                attention_block.complete_compressor_windows = current is grouped
        policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
        for block in model:
            modules = dict(block.named_modules())
            for role in _COMPRESSOR_ROLES:
                matches = [m for n, m in modules.items() if n == role or n.endswith("." + role)]
                for module in matches:
                    fully_shard(
                        module,
                        mesh=mesh,
                        mp_policy=policy,
                        reshard_after_forward=not role.endswith("ape_param"),
                    )
        fully_shard_across_blocks(
            list(grouped),
            _COMPRESSOR_ROLES,
            mesh=mesh,
            mp_policy=policy,
            offload_policy=None,
            shard_module=fully_shard_deepseek_v4,
        )
        for candidate in (model, grouped):
            fully_shard(candidate, mesh=mesh, mp_policy=policy)
        torch.manual_seed(91 + rank)
        inputs = [torch.randn(1, 132, 16, device="cuda") for _ in range(2)]
        upstream = [torch.randn_like(x) for x in inputs]
        runs = []
        for candidate in (model, grouped):
            optimizer = torch.optim.SGD(candidate.parameters(), lr=0.01, weight_decay=0.1)
            records = []
            for step, (x_base, grad) in enumerate(zip(inputs, upstream, strict=True)):
                x = x_base.clone().requires_grad_()
                output = candidate(x)
                output.backward(grad)
                records.append([output.detach(), x.grad.clone()])
                for parameter in candidate.parameters():
                    assert parameter.dtype == torch.float32
                    if parameter.requires_grad:
                        assert parameter.grad is not None
                        records[-1].append(parameter.grad.full_tensor())
                    else:
                        assert parameter.grad is None
                optimizer.step()
                records[-1].extend(p.full_tensor() for p in candidate.parameters())
                optimizer.zero_grad(set_to_none=True)
            runs.append(records)
        for expected_step, actual_step in zip(*runs, strict=True):
            for expected, actual in zip(expected_step, actual_step, strict=True):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires two CUDA devices")
@pytest.mark.parametrize("checkpointed", [False, True])
def test_grouped_active_compressors_preserve_gradients_and_updates(
    tmp_path: Path,
    checkpointed: bool,
) -> None:
    mp.spawn(_compressor_worker, args=(str(tmp_path / "init"), checkpointed), nprocs=2, join=True)
