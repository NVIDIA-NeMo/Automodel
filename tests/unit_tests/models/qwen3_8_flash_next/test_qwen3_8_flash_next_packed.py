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

"""CPU parity tests for the packed (THD) Qwen3.8-Flash-Next QSA path.

The gold reference for a packed row is the set of independent per-document
runs of the exact same modules: routes must match after adding the document
start offsets, and layer outputs and input gradients must match the
concatenated per-document results.
"""

import importlib.util

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.qwen3_8_flash_next.config import Qwen3_8_FlashNextTextConfig
from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextQSAAttention
from nemo_automodel.components.models.qwen3_8_flash_next.qsa import Qwen3_8_FlashNextQSAIndexer

_CU_SEQLENS = (0, 5, 12, 20)


def _can_run_h100_tilelang() -> bool:
    """Return whether this process has TileLang and an exact H100/SM90 target."""
    if importlib.util.find_spec("tilelang") is None or not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability() == (9, 0) and "H100" in torch.cuda.get_device_name()


requires_h100_tilelang = pytest.mark.skipif(
    not _can_run_h100_tilelang(),
    reason="requires TileLang on an H100/SM90 CUDA device",
)


def _config(*, token_budget: int = 8, compress_ratio: int = 2) -> Qwen3_8_FlashNextTextConfig:
    return Qwen3_8_FlashNextTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        layer_types=["full_attention"],
        moe_intermediate_size=4,
        shared_expert_intermediate_size=4,
        num_experts=2,
        num_experts_per_tok=1,
        hc_count=2,
        hc_lowrank=2,
        ple_layer_ids=[],
        indexer_budget=token_budget,
        indexer_compress_ratio=compress_ratio,
        indexer_n_heads=2,
        indexer_kv_heads=1,
        indexer_head_dim=4,
        max_position_embeddings=4096,
        rope_parameters={
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
        partial_rotary_factor=1.0,
        dtype="float32",
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=1,
    )


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )


def _document_relative_freqs(cu_seqlens: tuple[int, ...], rotary_width: int = 4) -> torch.Tensor:
    """Build packed rotary values whose positions restart at every document."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, rotary_width, 2).float() / rotary_width))
    segments = []
    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        positions = torch.arange(end - start, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        segments.append(torch.cat((angles.cos(), angles.sin()), dim=-1))
    return torch.cat(segments, dim=0).unsqueeze(0)


def test_packed_indexer_routes_match_per_document_runs() -> None:
    torch.manual_seed(21)
    config = _config(token_budget=4, compress_ratio=2)
    indexer = Qwen3_8_FlashNextQSAIndexer(config, _backend())
    indexer.init_weights()

    total_tokens = _CU_SEQLENS[-1]
    hidden = torch.randn(1, total_tokens, config.hidden_size)
    packed_freqs = _document_relative_freqs(_CU_SEQLENS)
    packed_routes = indexer(
        hidden,
        freqs_cis=packed_freqs,
        cu_seqlens=torch.tensor(_CU_SEQLENS, dtype=torch.int32),
    )
    assert packed_routes.shape == (1, total_tokens, config.indexer_budget + config.indexer_compress_ratio - 1)
    assert packed_routes.dtype == torch.int32

    for start, end in zip(_CU_SEQLENS, _CU_SEQLENS[1:]):
        document_routes = indexer(
            hidden[:, start:end],
            freqs_cis=packed_freqs[:, start:end],
        )
        expected = torch.where(
            document_routes >= 0,
            document_routes + start,
            document_routes.new_full((), -1),
        )
        torch.testing.assert_close(packed_routes[:, start:end], expected, rtol=0, atol=0)
        # Every valid route stays inside its own document.
        valid = packed_routes[0, start:end]
        valid = valid[valid >= 0]
        assert bool((valid >= start).all()) and bool((valid < end).all())


def test_packed_qsa_layer_matches_per_document_forward_backward() -> None:
    torch.manual_seed(33)
    config = _config(token_budget=4, compress_ratio=2)
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    attention.init_weights(torch.device("cpu"))
    attention.train()

    total_tokens = _CU_SEQLENS[-1]
    base_hidden = torch.randn(1, total_tokens, config.hidden_size)
    packed_freqs = _document_relative_freqs(_CU_SEQLENS)
    grad_output = torch.randn(1, total_tokens, config.hidden_size)

    packed_hidden = base_hidden.clone().requires_grad_(True)
    packed_output = attention(
        packed_hidden,
        freqs_cis=packed_freqs,
        cu_seqlens=torch.tensor(_CU_SEQLENS, dtype=torch.int32),
    )
    packed_output.backward(grad_output)

    for start, end in zip(_CU_SEQLENS, _CU_SEQLENS[1:]):
        document_hidden = base_hidden[:, start:end].clone().requires_grad_(True)
        document_output = attention(
            document_hidden,
            freqs_cis=packed_freqs[:, start:end],
        )
        document_output.backward(grad_output[:, start:end])
        torch.testing.assert_close(packed_output[:, start:end], document_output)
        torch.testing.assert_close(packed_hidden.grad[:, start:end], document_hidden.grad)


def test_packed_indexer_rejects_invalid_boundaries() -> None:
    config = _config(token_budget=4, compress_ratio=2)
    indexer = Qwen3_8_FlashNextQSAIndexer(config, _backend())
    indexer.init_weights()
    hidden = torch.randn(1, 6, config.hidden_size)
    freqs = _document_relative_freqs((0, 6))

    with pytest.raises(ValueError, match="strictly increasing"):
        indexer(hidden, freqs_cis=freqs, cu_seqlens=torch.tensor([0, 4, 4, 6]))
    with pytest.raises(ValueError, match="strictly increasing"):
        indexer(hidden, freqs_cis=freqs, cu_seqlens=torch.tensor([0, 4]))
    with pytest.raises(ValueError, match="batch of one row"):
        indexer(
            hidden.expand(2, -1, -1),
            freqs_cis=freqs.expand(2, -1, -1),
            cu_seqlens=torch.tensor([0, 6]),
        )
    with pytest.raises(ValueError, match="not attention_mask"):
        indexer(
            hidden,
            freqs_cis=freqs,
            attention_mask=torch.ones(1, 6),
            cu_seqlens=torch.tensor([0, 6]),
        )


def test_packed_qsa_layer_rejects_cp_and_multi_row_batches() -> None:
    config = _config(token_budget=4, compress_ratio=2)
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
    attention.init_weights(torch.device("cpu"))
    hidden = torch.randn(2, 6, config.hidden_size)
    freqs = _document_relative_freqs((0, 6)).expand(2, -1, -1)

    with pytest.raises(ValueError, match="batch of one row"):
        attention(hidden, freqs_cis=freqs, cu_seqlens=torch.tensor([0, 6]))
    with pytest.raises(NotImplementedError, match="global document boundaries in the CP context"):
        attention(
            hidden[:1],
            freqs_cis=freqs[:1],
            cp_context=object(),
            cu_seqlens=torch.tensor([0, 6]),
        )


def _full_size_config() -> Qwen3_8_FlashNextTextConfig:
    """Kernel-shaped heads (24 query, 2 KV, dim 256) on a small hidden width."""
    config = _config(token_budget=8, compress_ratio=4)
    config.hidden_size = 512
    config.num_attention_heads = 24
    config.num_key_value_heads = 2
    config.head_dim = 256
    config.indexer_head_dim = 256
    return config


@requires_h100_tilelang
def test_packed_qsa_layer_cuda_matches_per_document_dense_kernel() -> None:
    """The fused THD dispatch must match per-document dense-kernel runs."""
    torch.manual_seed(55)
    config = _full_size_config()
    backend = BackendConfig(
        linear="torch",
        attn="tilelang",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=backend)
    attention.init_weights(torch.device("cpu"))
    attention = attention.to(device="cuda", dtype=torch.bfloat16)
    attention.train()

    cu_seqlens = (0, 7, 18, 40)
    total_tokens = cu_seqlens[-1]
    base_hidden = torch.randn(1, total_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16).mul_(0.25)
    packed_freqs = _document_relative_freqs(cu_seqlens, rotary_width=config.head_dim).to(
        device="cuda", dtype=torch.bfloat16
    )
    grad_output = torch.randn(1, total_tokens, config.hidden_size, device="cuda", dtype=torch.bfloat16).mul_(0.25)

    packed_hidden = base_hidden.clone().requires_grad_(True)
    packed_output = attention(
        packed_hidden,
        freqs_cis=packed_freqs,
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device="cuda"),
    )
    packed_output.backward(grad_output)
    torch.cuda.synchronize()

    for start, end in zip(cu_seqlens, cu_seqlens[1:]):
        document_hidden = base_hidden[:, start:end].clone().requires_grad_(True)
        document_output = attention(
            document_hidden,
            freqs_cis=packed_freqs[:, start:end],
        )
        document_output.backward(grad_output[:, start:end])
        torch.cuda.synchronize()
        for label, actual, expected in (
            ("output", packed_output[:, start:end], document_output),
            ("dhidden", packed_hidden.grad[:, start:end], document_hidden.grad),
        ):
            actual_float = actual.float().flatten()
            expected_float = expected.float().flatten()
            assert torch.isfinite(actual_float).all(), label
            expected_norm = torch.linalg.vector_norm(expected_float)
            relative_l2 = torch.linalg.vector_norm(actual_float - expected_float) / expected_norm
            assert relative_l2 <= 0.02, f"{label} relative_l2={relative_l2.item()}"


class _FakeMesh:
    """Minimal CP mesh stand-in exposing size and process group."""

    def __init__(self, size: int, group: object | None = None) -> None:
        self._size = size
        self._group = group

    def size(self) -> int:
        return self._size

    def get_group(self) -> object:
        return self._group


def test_gdn_wrapper_builds_packed_segments_with_padding_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stashed global boundaries reach the blockdiag core with a pad segment."""
    from nemo_automodel.components.distributed.blockdiag_cp import BlockdiagCpModelState
    from nemo_automodel.components.models.qwen3_5_moe.cp_linear_attn import CPAwareGatedDeltaNet
    from nemo_automodel.components.models.qwen3_8_flash_next.layers import Qwen3_8_FlashNextGatedDeltaNet

    captured: list[BlockdiagCpModelState] = []

    def _capture_base_cp(self, hidden_states, *, position_ids, seq_index, blockdiag_state=None):
        del self, position_ids, seq_index
        assert blockdiag_state is not None
        captured.append(blockdiag_state)
        return hidden_states

    monkeypatch.setattr(CPAwareGatedDeltaNet, "_forward_with_cp", _capture_base_cp)
    gdn_config = _config()
    gdn_config.layer_types = ["linear_attention"]
    gdn_config.linear_conv_kernel_dim = 4
    gdn_config.linear_key_head_dim = 4
    gdn_config.linear_value_head_dim = 4
    gdn_config.linear_num_key_heads = 1
    gdn_config.linear_num_value_heads = 2
    layer = Qwen3_8_FlashNextGatedDeltaNet(gdn_config, layer_idx=0)
    layer._cp_mesh = _FakeMesh(2, object())

    layer._packed_global_cu_seqlens = torch.tensor([0, 3, 10])
    layer._forward_with_cp(
        torch.randn(1, 8, gdn_config.hidden_size),
        position_ids=torch.arange(8).view(1, -1),
        seq_index=None,
    )
    torch.testing.assert_close(captured[0].packed_cu_seqlens, torch.tensor([0, 3, 10, 16]))

    layer._packed_global_cu_seqlens = None
    layer._forward_with_cp(
        torch.randn(1, 8, gdn_config.hidden_size),
        position_ids=torch.arange(8).view(1, -1),
        seq_index=None,
    )
    torch.testing.assert_close(captured[1].packed_cu_seqlens, torch.tensor([0, 16]))


def _packed_cp_parity_worker(rank: int, world_size: int, store_path: str) -> None:
    import torch.distributed as dist

    from nemo_automodel.components.models.qwen3_8_flash_next.cp import (
        Qwen3_8_FlashNextCPContext,
        shard_batch_for_qwen3_8_flash_next_cp,
    )

    try:
        torch.set_num_threads(1)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{store_path}",
            rank=rank,
            world_size=world_size,
        )
        from torch.distributed.device_mesh import init_device_mesh

        cp_mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("cp",))["cp"]

        # ---- packed sharder: boundaries survive as replicated global context ----
        boundaries = (0, 3, 10)
        padded_boundaries = (*boundaries, 16)
        total_tokens = boundaries[-1]
        input_ids = torch.arange(1, total_tokens + 1, dtype=torch.long).view(1, -1)
        document_positions = torch.cat(
            [torch.arange(end - start) for start, end in zip(boundaries, boundaries[1:])]
        ).view(1, -1)
        _, local_batch, layout = shard_batch_for_qwen3_8_flash_next_cp(
            cp_mesh,
            None,
            {
                "input_ids": input_ids.clone(),
                "labels": input_ids.clone(),
                "position_ids": document_positions.clone(),
                "cu_seqlens": torch.tensor(boundaries, dtype=torch.int32),
                "qkv_format": "thd",
            },
            padding_token_id=0,
            pad_multiple=4,
        )
        assert layout.padded_seq_len == 16
        local_length = 16 // world_size
        local_start = rank * local_length
        batch_context = local_batch["_qwen3_8_flash_next_cp_context"]
        assert isinstance(batch_context, Qwen3_8_FlashNextCPContext)
        assert batch_context.global_cu_seqlens is not None
        torch.testing.assert_close(
            batch_context.global_cu_seqlens,
            torch.tensor(boundaries, dtype=torch.long),
        )
        assert "cu_seqlens" not in local_batch

        # ---- QSA packed CP2 vs packed CP1 ----
        torch.manual_seed(321)
        config = _config(token_budget=4, compress_ratio=2)
        reference = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
        reference.init_weights(torch.device("cpu"))
        cp_attention = Qwen3_8_FlashNextQSAAttention(config, layer_idx=0, backend=_backend())
        cp_attention.load_state_dict(reference.state_dict())

        full_freqs = _document_relative_freqs(padded_boundaries)
        torch.manual_seed(99)
        full_hidden_values = torch.randn(1, 16, config.hidden_size)
        grad_output = torch.randn(1, 16, config.hidden_size)

        reference_hidden = full_hidden_values[:, :total_tokens].clone().requires_grad_(True)
        reference_routes: list[torch.Tensor] = []
        handle = reference.indexer.register_forward_hook(lambda _module, _args, output: reference_routes.append(output))
        reference_output = reference(
            reference_hidden,
            freqs_cis=full_freqs[:, :total_tokens],
            cu_seqlens=torch.tensor(boundaries, dtype=torch.int32),
        )
        handle.remove()
        reference_output.backward(grad_output[:, :total_tokens])

        context = Qwen3_8_FlashNextCPContext(
            group=dist.group.WORLD,
            rank=rank,
            size=world_size,
            global_input_ids=torch.nn.functional.pad(input_ids, (0, 16 - total_tokens)),
            global_padding_mask=torch.arange(16).view(1, -1) >= total_tokens,
            local_sequence_start=local_start,
            local_sequence_length=local_length,
            global_cu_seqlens=torch.tensor(boundaries, dtype=torch.long),
        )
        local_hidden = full_hidden_values[:, local_start : local_start + local_length].clone().requires_grad_(True)
        cp_routes: list[torch.Tensor] = []
        handle = cp_attention.indexer.register_forward_hook(lambda _module, _args, output: cp_routes.append(output))
        cp_output = cp_attention(
            local_hidden,
            freqs_cis=full_freqs[:, local_start : local_start + local_length],
            cp_context=context,
        )
        handle.remove()
        cp_output.backward(grad_output[:, local_start : local_start + local_length])

        # Routes: bitwise-global IDs on real tokens, all -1 on the CP pad tail.
        for local_idx in range(local_length):
            global_idx = local_start + local_idx
            if global_idx < total_tokens:
                torch.testing.assert_close(
                    cp_routes[0][0, local_idx],
                    reference_routes[0][0, global_idx],
                    rtol=0,
                    atol=0,
                )
            else:
                assert bool((cp_routes[0][0, local_idx] == -1).all())

        real_length = max(0, min(total_tokens - local_start, local_length))
        if real_length:
            torch.testing.assert_close(
                cp_output[:, :real_length],
                reference_output[:, local_start : local_start + real_length],
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                local_hidden.grad[:, :real_length],
                reference_hidden.grad[:, local_start : local_start + real_length],
                rtol=2e-5,
                atol=2e-6,
            )
        if real_length < local_length:
            assert bool((cp_output[:, real_length:] == 0).all())

        # Parameter gradients: rank-local contributions must sum to CP1.
        reference_parameters = dict(reference.named_parameters())
        for name, parameter in cp_attention.named_parameters():
            reference_gradient = reference_parameters[name].grad
            if parameter.grad is None:
                assert reference_gradient is None
                continue
            dist.all_reduce(parameter.grad, group=dist.group.WORLD)
            assert reference_gradient is not None
            torch.testing.assert_close(parameter.grad, reference_gradient, rtol=2e-5, atol=2e-6)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def test_two_rank_packed_qsa_and_sharder_match_packed_cp1(tmp_path) -> None:
    torch.multiprocessing.spawn(
        _packed_cp_parity_worker,
        args=(2, str(tmp_path / "qwen3-8-flash-next-packed-cp-gloo")),
        nprocs=2,
        join=True,
    )
