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

"""Bounded released-weight loading into live CPU model and owner storage.

Expected FP8/FP4 values use explicit released-layout arithmetic, independently
of the adapter. Gloo cases exercise EP2 and EP2 with a second inner shard axis.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.checkpoint.checkpointing import Checkpointer, CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM
from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import (
    DeepSeekV41StateDictAdapter,
    dequantize_checkpoint_weight,
)
from nemo_automodel.components.moe.config import MoEConfig
from tests.unit_tests.models.deepseek_v41.conftest import tiny_config


def _checkpoint_model(tensors: dict[str, torch.Tensor]) -> torch.nn.Module:
    model = torch.nn.Module()
    for fqn, tensor in tensors.items():
        owner = model
        parts = fqn.split(".")
        for part in parts[:-1]:
            if part not in owner._modules:
                owner.add_module(part, torch.nn.Module())
            owner = owner._modules[part]
        owner.register_buffer(parts[-1], tensor)
    return model


class _CheckpointOnlyV41(DeepseekV41ForCausalLM):
    """Keep the real custom-model checkpoint contract with tiny tensor fixtures."""

    def __init__(self, config: DeepseekV41Config, tensors: dict[str, torch.Tensor]) -> None:
        torch.nn.Module.__init__(self)
        self.config = config
        for name, module in _checkpoint_model(tensors).named_children():
            self.add_module(name, module)
        for module in self.modules():
            for name, value in list(module._buffers.items()):
                del module._buffers[name]
                module.register_parameter(name, torch.nn.Parameter(value))


def _adapter(
    engram_rows: int | None = None,
    *,
    second_engram_rows: int | None = None,
    dtype: torch.dtype = torch.float32,
    experts: str = "torch_mm",
    dim: int = 32,
) -> DeepSeekV41StateDictAdapter:
    engram_layers = [] if engram_rows is None else [1]
    table_rows = [] if engram_rows is None else [engram_rows]
    if second_engram_rows is not None:
        engram_layers.append(4)
        table_rows.append(second_engram_rows)
    config = tiny_config(
        vocab_size=64,
        hidden_size=dim,
        moe_intermediate_size=dim,
        n_routed_experts=2,
        num_experts_per_tok=1,
        engram_layer_ids=engram_layers,
        engram_num_embeddings=table_rows,
        torch_dtype=str(dtype).removeprefix("torch."),
    )
    moe = MoEConfig(
        dim=dim,
        inter_dim=dim,
        moe_inter_dim=dim,
        n_routed_experts=2,
        n_shared_experts=1,
        n_activated_experts=1,
        n_expert_groups=0,
        n_limited_groups=0,
        train_gate=True,
        gate_bias_update_factor=0,
        aux_loss_coeff=0,
        score_func="sqrtsoftplus",
        route_scale=1.5,
        norm_topk_prob=True,
        dtype=dtype,
    )
    return DeepSeekV41StateDictAdapter(
        config,
        moe,
        BackendConfig(linear="torch", attn="sdpa", dispatcher="torch", experts=experts),
        dtype=dtype,
    )


def test_dense_fp8_partial_blocks_use_32_by_32_scales() -> None:
    weight = torch.ones(33, 35).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1, 2], [4, 8]], dtype=torch.float32).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(weight, scales, dtype=torch.float32)
    assert decoded.shape == (33, 35)
    assert (decoded[:32, :32] == 1).all()
    assert (decoded[:32, 32:] == 2).all()
    assert (decoded[32:, :32] == 4).all()
    assert (decoded[32:, 32:] == 8).all()


def test_fp4_decoding_preserves_nibble_order_sign_and_row_scales() -> None:
    packed = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2], dtype=torch.uint8).view(torch.int8)
    scale = torch.tensor([[2.0]]).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(packed, scale, dtype=torch.float32)
    expected = torch.tensor([[0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12] * 2]).float()
    torch.testing.assert_close(decoded, expected, rtol=0, atol=0)


def test_engram_fp8_uses_per_row_scales() -> None:
    weight = torch.ones(2, 64).to(torch.float8_e4m3fn)
    scales = torch.tensor([[1, 2], [4, 8]], dtype=torch.float32).to(torch.float8_e8m0fnu)
    decoded = dequantize_checkpoint_weight(weight, scales, dtype=torch.float32, rowwise=True)
    assert (decoded[0, :32] == 1).all()
    assert (decoded[0, 32:] == 2).all()
    assert (decoded[1, :32] == 4).all()
    assert (decoded[1, 32:] == 8).all()
    with pytest.raises(ValueError, match="Scale shape"):
        dequantize_checkpoint_weight(weight, scales)


def test_global_key_audit_accepts_owner_local_meta_tables() -> None:
    state = {
        "model.layers.1.engram.embed.weight": torch.empty(9, 64, device="meta"),
        "model.layers.0.mlp.experts.gate_and_up_projs": torch.empty(1, 32, 64, device="meta"),
        "model.layers.0.mlp.experts.down_projs": torch.empty(1, 32, 32, device="meta"),
    }
    keys = _adapter(engram_rows=17).get_hf_state_dict_keys(state)
    assert "layers.1.engram.embed.weight" in keys
    assert len(keys) == 7
    for index in range(2):
        for projection in (1, 2, 3):
            assert f"layers.0.ffn.experts.{index}.w{projection}.weight" in keys


def test_stream_load_copies_quantized_sources_into_grouped_views(tmp_path: Path) -> None:
    adapter = _adapter()
    state = {
        "model.layers.0.self_attn.wq_a.weight": torch.zeros(33, 64),
        "model.layers.0.mlp.experts.gate_and_up_projs": torch.zeros(2, 32, 64),
        "model.layers.0.mlp.experts.down_projs": torch.zeros(2, 32, 32),
        "model.norm.weight": torch.zeros(32),
    }
    source = adapter.to_hf(state, quantization=True, for_checkpoint_load=True)
    for key, value in source.items():
        if value.dtype == torch.int8:
            value.fill_(0x44)
        else:
            value.copy_(torch.full(value.shape, 2.0 if key.endswith(".scale") else 1.0))
    save_file(source, tmp_path / "model.safetensors")
    model = _checkpoint_model(state)
    audit = adapter.load_from_checkpoint(model, tmp_path)
    assert set(audit.loaded_keys) == set(source)
    assert audit.loaded_bytes == sum(t.numel() * t.element_size() for t in state.values())
    assert audit.source_bytes == sum(t.numel() * t.element_size() for t in source.values())
    assert audit.max_chunk_source_bytes < 4 * 1024 * 1024
    assert audit.max_chunk_output_bytes < 16 * 1024 * 1024
    assert (state["model.layers.0.self_attn.wq_a.weight"] == 2).all()
    assert (state["model.layers.0.mlp.experts.gate_and_up_projs"] == 4).all()
    assert (state["model.layers.0.mlp.experts.down_projs"] == 4).all()
    assert (state["model.norm.weight"] == 1).all()
    assert not getattr(adapter, "_inplace_loaded_native_keys", None)


def test_streaming_checkpoint_rejects_reduced_precision_strict_storage(tmp_path) -> None:
    adapter = _adapter()
    original = torch.tensor([1.001, -2.003, 0.3333, 17.125])
    save_file({"layers.0.attn.attn_sink": original}, tmp_path / "model.safetensors")
    fqn = "model.layers.0.self_attn.sinks_param.weight"
    model = _checkpoint_model({fqn: torch.zeros_like(original, dtype=torch.bfloat16)})
    model._keep_in_fp32_modules_strict = ["self_attn.sinks_param"]
    with pytest.raises(ValueError, match="Strict FP32 checkpoint destination.*local storage dtype torch.bfloat16"):
        adapter.load_from_checkpoint(model, tmp_path)
    model = _checkpoint_model({fqn: torch.zeros_like(original)})
    model._keep_in_fp32_modules_strict = ["self_attn.sinks_param"]
    adapter.load_from_checkpoint(model, tmp_path)
    assert torch.equal(model.state_dict()[fqn], original)


def test_streaming_checkpoint_rejects_detached_expert_destinations(tmp_path, monkeypatch) -> None:
    adapter = _adapter()
    state = {"model.layers.0.mlp.experts.gate_and_up_projs": torch.full((2, 32, 64), -101.0)}
    save_file(
        {name: value.contiguous() for name, value in adapter.to_hf(state).items()}, tmp_path / "model.safetensors"
    )
    convert = adapter.convert_single_tensor_to_hf

    def detached_destinations(fqn, tensor, **kwargs):
        """Simulate projection copies that cannot update grouped model storage.

        Args:
            fqn: Native expert parameter name.
            tensor: Grouped expert tensor [experts, hidden, 2 * intermediate].
            **kwargs: Forwarded checkpoint conversion options.

        Returns:
            Released names and detached-storage projection matrices [output, input], preserving the source
            dtype and values. Each matrix has storage distinct from the model parameter.
        """
        return [(key, value.clone()) for key, value in convert(fqn, tensor, **kwargs)]

    monkeypatch.setattr(adapter, "convert_single_tensor_to_hf", detached_destinations)
    with pytest.raises(ValueError, match="does not alias model storage"):
        adapter.load_from_checkpoint(_checkpoint_model(state), tmp_path)
    assert (state["model.layers.0.mlp.experts.gate_and_up_projs"] == -101).all()


def _quantized_checkpointer_worker(rank: int, rendezvous: str, expert_shard_size: int = 1) -> None:
    """Read a real quantized HF dump through streaming and DCP, then round-trip SafeTensors."""
    torch.set_num_threads(1)
    world_size = 2 * expert_shard_size
    dim = 32 * expert_shard_size
    dist.init_process_group(
        "gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size, timeout=timedelta(seconds=90)
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        expert_mesh = (
            init_device_mesh("cpu", (expert_shard_size, 2), mesh_dim_names=("ep_shard", "ep"))
            if expert_shard_size > 1
            else init_device_mesh("cpu", (2,), mesh_dim_names=("ep",))
        )
        adapter = _adapter(17, second_engram_rows=19, dtype=torch.bfloat16, experts="torch_mm", dim=dim)
        source = {}
        expected = {}
        for layer, rows in ((1, 17), (4, 19)):
            raw = (torch.arange(rows * 64).reshape(rows, 64) % 7 - 3).to(torch.float8_e4m3fn)
            scales = (2.0 ** (torch.arange(rows * 2).reshape(rows, 2) % 5 - 2)).to(torch.float8_e8m0fnu)
            source[f"layers.{layer}.engram.embed.weight"] = raw
            source[f"layers.{layer}.engram.embed.scale"] = scales
            logical = (raw.float() * scales.float().repeat_interleave(32, dim=1)).bfloat16()
            padded_rows = (rows + world_size - 1) // world_size * world_size
            expected[f"model.layers.{layer}.engram.embed.weight"] = torch.nn.functional.pad(
                logical, (0, 0, 0, padded_rows - rows)
            )

        raw = (torch.arange(65 * 64).reshape(65, 64) % 5 - 2).to(torch.float8_e4m3fn)
        scales = torch.tensor([[0.5, 1.0], [2.0, 4.0], [8.0, 16.0]]).to(torch.float8_e8m0fnu)
        source["layers.0.attn.wq_a.weight"] = raw
        source["layers.0.attn.wq_a.scale"] = scales
        expected["model.layers.0.self_attn.wq_a.weight"] = (
            raw.float() * scales.float().repeat_interleave(32, dim=0)[:65].repeat_interleave(32, dim=1)
        ).bfloat16()

        # The fixture constructs all E2M1 code values explicitly, with distinct
        # low/high nibbles and per-row powers of two. Expected expert matrices
        # are independent of the adapter's decoder and use released layouts.
        fp4_values = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])
        projections = {}
        for expert in range(2):
            for projection in (1, 2, 3):
                row, column = torch.meshgrid(torch.arange(dim), torch.arange(dim // 2), indexing="ij")
                low = (row + column + expert * 3 + projection * 5) % 16
                high = (row * 3 + column * 5 + expert + projection) % 16
                raw = (low | (high << 4)).to(torch.uint8).view(torch.int8)
                scales = (2.0 ** ((torch.arange(dim)[:, None] + 2 * torch.arange(dim // 32)[None, :]) % 3 - 1)).to(
                    torch.float8_e8m0fnu
                )
                key = f"layers.0.ffn.experts.{expert}.w{projection}"
                source[f"{key}.weight"] = raw
                source[f"{key}.scale"] = scales
                decoded = torch.stack((fp4_values[low], fp4_values[high]), dim=-1).flatten(1)
                projections[expert, projection] = (decoded * scales.float().repeat_interleave(32, dim=1)).bfloat16()
        expected["model.layers.0.mlp.experts.gate_and_up_projs"] = torch.stack(
            [torch.cat((projections[expert, 1].T, projections[expert, 3].T), dim=-1) for expert in range(2)]
        )
        expected["model.layers.0.mlp.experts.down_projs"] = torch.stack(
            [projections[expert, 2].T for expert in range(2)]
        )
        for released, native, value in (
            (
                "layers.0.attn.attn_sink",
                "model.layers.0.self_attn.sinks_param.weight",
                torch.tensor([1.00123, -0.33337, 2.00456, -3.00091]),
            ),
            (
                "layers.0.hc_attn_fn",
                "model.layers.0.attn_hc.fn",
                torch.arange(8 * 128).reshape(8, 128).float() / 10000 + 0.00123,
            ),
            ("layers.0.hc_ffn_scale", "model.layers.0.ffn_hc.scale", torch.tensor([1.00123, 0.33337, 0.00456])),
        ):
            source[released] = value
            expected[native] = value

        root = Path(rendezvous).parent
        checkpoint = root / "quantized_hf"
        if rank == 0:
            checkpoint.mkdir()
            save_file(source, checkpoint / "model.safetensors")
            (checkpoint / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": dict.fromkeys(source, "model.safetensors")})
            )
            adapter.config.save_pretrained(checkpoint)
        dist.barrier()
        local_expected = {}
        tensors = {}
        for key, value in expected.items():
            expert = ".mlp.experts." in key
            if expert:
                local = value[rank % 2 : rank % 2 + 1]
                if expert_shard_size > 1:
                    local = local[:, (rank // 2) * 32 : (rank // 2 + 1) * 32]
                placements = (Shard(1), Shard(0)) if expert_shard_size > 1 else (Shard(0),)
            else:
                local_rows = (value.shape[0] + world_size - 1) // world_size
                local = value[rank * local_rows : (rank + 1) * local_rows]
                placements = (Shard(0),)
            local_expected[key] = local
            tensors[key] = DTensor.from_local(
                torch.full_like(local, -101),
                expert_mesh if expert else mesh,
                placements,
                shape=value.shape,
                stride=value.stride(),
            )
        model = _CheckpointOnlyV41(adapter.config, tensors)
        model.state_dict_adapter = adapter
        if expert_shard_size > 1:
            for key, tensor in model.state_dict().items():
                if ".mlp.experts." not in key:
                    continue
                pointer = tensor.to_local().untyped_storage().data_ptr()
                # Explicit direct loading must preserve live expert storage;
                # raw quantized destinations must have independent storage.
                for loading, quantized, preserve, aliases in (
                    (True, False, True, True),
                    (True, True, True, False),
                ):
                    converted = adapter.convert_single_tensor_to_hf(
                        key,
                        tensor,
                        for_checkpoint_load=loading,
                        quantization=quantized,
                        preserve_dtensor_load_views=preserve,
                    )
                    for _, destination in converted:
                        assert isinstance(destination, DTensor)
                        assert (destination.to_local().untyped_storage().data_ptr() == pointer) == aliases
        adapter.load_from_checkpoint(model, checkpoint, device_mesh=expert_mesh)
        for key, tensor in model.state_dict().items():
            reference = local_expected[key]
            assert tensor.dtype == reference.dtype and tensor.to_local().dtype == reference.dtype
            torch.testing.assert_close(tensor.to_local(), reference, rtol=0, atol=0)
            tensor.to_local().fill_(-101)
        checkpointer = Checkpointer(
            CheckpointingConfig(
                checkpoint_dir=str(root),
                model_save_format="safetensors",
                save_consolidated=False,
                model_cache_dir=str(root / "cache"),
                model_repo_id="test/deepseek-v41",
                dequantize_base_checkpoint=True,
            ),
            dp_rank=rank,
            tp_rank=0,
            pp_rank=0,
            process_group=dist.group.WORLD,
            moe_mesh=expert_mesh,
        )
        try:
            checkpointer.load_model(model, str(checkpoint), is_init_step=True)
            for key, tensor in model.state_dict().items():
                reference = local_expected[key]
                assert tensor.dtype == reference.dtype and tensor.to_local().dtype == reference.dtype
                torch.testing.assert_close(tensor.to_local(), reference, rtol=0, atol=0)
            checkpointer.save_model(model, str(root / "trained_safetensors"))
            for tensor in model.state_dict().values():
                tensor.to_local().fill_(-99)
            checkpointer.load_model(model, str(root / "trained_safetensors" / "model"))
            for key, tensor in model.state_dict().items():
                reference = local_expected[key]
                assert tensor.dtype == reference.dtype and tensor.to_local().dtype == reference.dtype
                torch.testing.assert_close(tensor.to_local(), reference, rtol=0, atol=0)
        finally:
            checkpointer.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("expert_shard_size", [1, 2], ids=["ep2", "ep2_fsdp2"])
def test_real_quantized_hf_initialization_and_safetensors_resume(tmp_path: Path, expert_shard_size: int) -> None:
    mp.spawn(
        _quantized_checkpointer_worker,
        args=(str(tmp_path / "quantized_rendezvous"), expert_shard_size),
        nprocs=2 * expert_shard_size,
        join=True,
    )


@pytest.mark.parametrize("failure", ["missing_key", "missing_scale", "wrong_shape"])
def test_stream_load_rejects_incomplete_or_incompatible_source(tmp_path: Path, failure: str) -> None:
    """Malformed source weights fail before the sentinel destination is copied."""
    key = "layers.0.attn.wq_a.weight"
    source = {key: torch.ones(32, 32, dtype=torch.float32)}
    message = "missing model tensor"
    if failure == "missing_key":
        source = {"norm.weight": torch.ones(32)}
    elif failure == "missing_scale":
        source[key] = source[key].to(torch.float8_e4m3fn)
        message = "missing its scale"
    else:
        source[key] = torch.ones(31, 32)
        message = "decoded shape"
    save_file(source, tmp_path / "model.safetensors")
    target = torch.full((32, 32), -101.0)
    with pytest.raises(ValueError, match=message):
        _adapter().load_from_checkpoint(_checkpoint_model({"model.layers.0.self_attn.wq_a.weight": target}), tmp_path)
    assert torch.equal(target, torch.full_like(target, -101.0))


def test_stream_load_chunks_source_larger_than_decode_budget(tmp_path: Path) -> None:
    """A matrix beyond 4M elements is loaded in bounded, exact BF16 chunks."""
    shape = (4112, 1024)
    raw = torch.full(shape, 1.5).to(torch.float8_e4m3fn)
    scales = torch.full((129, 32), 2.0).to(torch.float8_e8m0fnu)
    save_file({"layers.0.attn.wq_a.weight": raw, "layers.0.attn.wq_a.scale": scales}, tmp_path / "model.safetensors")
    target = torch.full(shape, -101.0, dtype=torch.bfloat16)
    audit = _adapter(dtype=torch.bfloat16).load_from_checkpoint(
        _checkpoint_model({"model.layers.0.self_attn.wq_a.weight": target}), tmp_path
    )
    assert torch.equal(target, torch.full_like(target, 3.0))
    assert audit.source_bytes == raw.numel() + scales.numel()
    assert audit.loaded_bytes == target.numel() * target.element_size()
    assert audit.max_chunk_source_bytes < audit.source_bytes
    assert audit.max_chunk_output_bytes == 4 * 1024 * 1024 * target.element_size()
    assert audit.max_chunk_output_bytes < audit.loaded_bytes


def test_e8m0_smallest_exponent_byte_is_not_zero() -> None:
    """E8M0 byte zero is 2**-127, which must survive FP32 checkpoint decoding."""
    weight = torch.ones(32, 32).to(torch.float8_e4m3fn)
    scale = torch.zeros((1, 1), dtype=torch.uint8).view(torch.float8_e8m0fnu)
    actual = dequantize_checkpoint_weight(weight, scale, dtype=torch.float32)
    assert torch.equal(actual, torch.full((32, 32), 2.0**-127, dtype=torch.float32))
