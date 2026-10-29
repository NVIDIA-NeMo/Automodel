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

"""CP contracts: complete images, global masks, packed documents, PP metadata and sparse MLA."""

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from nemo_automodel._transformers.capabilities import ModelSupports, validate_for_mesh
from nemo_automodel.components._peft.lora import apply_lora_to_linear_modules
from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder
from nemo_automodel.components.models.mistral_large4 import attention
from nemo_automodel.components.models.mistral_large4 import context_parallel as cp
from nemo_automodel.components.models.mistral_large4.context_parallel import prepare_cp_inputs
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4ForCausalLM
from tests.unit_tests.models.mistral_large4.math_reference import tiny_config
from tests.unit_tests.models.mistral_large4.test_checkpoint import peft_config
from tests.unit_tests.models.mistral_large4.test_multimodal import image_batch
from tests.unit_tests.models.mistral_large4.test_pipeline import make_pipeline_model, partition_model, run_partition


class Mesh:
    """CP coordinate fixture for pure shard-shape checks."""

    def __init__(self, rank: int = 0, size: int = 2) -> None:
        self.rank, self.count = rank, size

    def size(self) -> int:
        return self.count

    def get_local_rank(self) -> int:
        return self.rank


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("primary", [False, True])
def test_cp_preserves_images_global_positions_and_local_targets(rank, primary):
    ids, pixels = image_batch()
    ids = ids.expand(2, -1).clone()
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[0, 0] = False
    padding = torch.zeros_like(mask)
    padding[1, -2:] = True
    labels = ids.clone()
    batch = dict(input_ids=ids, labels=labels, attention_mask=mask, padding_mask=padding, pixel_values=pixels)
    before = {k: v.clone() for k, v in batch.items() if isinstance(v, torch.Tensor)}
    prepared = prepare_cp_inputs(batch, num_chunks=2, shard_primary=primary)
    context, actual, layout = prepared["cp_sharder"].shard_batch(Mesh(rank), None, batch)
    assert layout.original_seq_len == 11 and layout.padded_seq_len == 12
    assert actual["pixel_values"] is pixels
    assert actual["labels"].shape == (2, 6)
    start, stop = rank * 6, (rank + 1) * 6
    expected_targets = torch.cat((labels, torch.full((2, 1), -100)), 1)[:, start:stop]
    torch.testing.assert_close(actual["labels"], expected_targets)
    expected_keep = mask & ~padding
    expected_positions = (expected_keep.long().cumsum(1) - 1).clamp_min(0)
    # The shared padder assigns arbitrary positions only to excluded CP pad slots.
    length = min(6, 11 - start)
    torch.testing.assert_close(actual["position_ids"][:, :length], expected_positions[:, start : start + length])
    expected_padding = torch.cat((~expected_keep, torch.ones(2, 1, dtype=torch.bool)), 1)[:, start:stop]
    torch.testing.assert_close(actual["padding_mask"], expected_padding)
    if primary:
        assert actual["input_ids"].shape == (2, 6)
        assert "attention_mask" not in actual
    else:
        assert actual["input_ids"] is ids
        torch.testing.assert_close(actual["attention_mask"], expected_keep)
    for key, value in before.items():
        original = dict(input_ids=ids, labels=labels, attention_mask=mask, padding_mask=padding)[key]
        torch.testing.assert_close(original, value)


@pytest.mark.parametrize("multimodal", [False, True])
def test_pipeline_cp_metadata_uses_global_key_width_and_local_activations(multimodal):
    model = make_pipeline_model(multimodal=multimodal)
    model.config.index_topk = 9
    parts = partition_model(model, (0, 2, 4))
    for part in parts:
        part.cp_mesh = Mesh(0)
    first_in, first_out = parts[0].get_pipeline_stage_metas(
        is_first=True, microbatch_size=1, seq_len=11, dtype=torch.bfloat16
    )
    last_in, last_out = parts[1].get_pipeline_stage_metas(
        is_first=False, microbatch_size=1, seq_len=11, dtype=torch.bfloat16
    )
    assert first_in[0].shape == (1, 11)
    assert first_out[0].shape == (1, 6, model.config.hidden_size)
    assert first_out[-1].shape == (1, 6, 9)  # index_topk exceeds the local sequence
    assert first_out[1].dtype == torch.float32
    assert [(v.shape, v.dtype) for v in first_out] == [(v.shape, v.dtype) for v in last_in]
    assert last_out[0].shape == (1, 6, model.config.vocab_size)
    ids = torch.ones(2, 11, dtype=torch.long)
    for part in parts:
        batch = {"input_ids": ids, "labels": ids}
        sharder = part.prepare_model_inputs_for_cp(batch, num_chunks=2)["cp_sharder"]
        _, sharded, _ = sharder.shard_batch(Mesh(0), None, batch)
        assert sharded["input_ids"].shape == (2, 11)
        assert sharded["labels"].shape == (2, 6)


@pytest.mark.parametrize("mask", [torch.tensor([[1, 2, 2, 0]]), torch.tensor([[1.0, 0.5, 1.0, 0.0]])])
def test_cp_rejects_nonbinary_masks_instead_of_joining_documents(mask):
    batch = {
        "input_ids": torch.ones(1, 4, dtype=torch.long),
        "labels": torch.ones(1, 4, dtype=torch.long),
        "attention_mask": mask,
    }
    sharder = prepare_cp_inputs(batch)["cp_sharder"]
    with pytest.raises(ValueError, match="binary"):
        sharder.shard_batch(Mesh(0), None, batch)


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("backend", ["sdpa", "tilelang"])
def test_cp_capability_passes_the_shared_recipe_validator(multimodal: bool, backend: str) -> None:
    model = (
        make_sparse_model(multimodal=multimodal)
        if backend == "tilelang"
        else make_pipeline_model(multimodal=multimodal)
    )
    mesh = SimpleNamespace(cp_size=2, pp_size=2, ep_size=2, tp_size=1)
    supports = ModelSupports(model, mesh)
    assert model.ModelCapabilities().supports_cp
    assert supports.supports_cp
    assert supports.supports_cp_with_sequence_packing == (backend == "tilelang")
    validate_for_mesh(model, mesh)


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("chunks", [1, 2])
def test_packed_sharding_preserves_documents_padding_targets_and_microbatches(
    rank: int, primary: bool, chunks: int
) -> None:
    ids = torch.arange(chunks * 10).reshape(chunks, 10)
    batch = dict(
        input_ids=ids,
        labels=ids + 1,
        seq_lens=torch.tensor([[3, 5], [4, 4]])[:chunks],
        seq_lens_padded=torch.tensor([[4, 6], [5, 5]])[:chunks],
        qkv_format="thd",
    )
    mask = ids.float() / 20
    batch["loss_mask"] = torch.zeros_like(mask)
    sharder = cp.prepare_packed_cp_inputs(batch, num_chunks=chunks, shard_primary=primary)["cp_sharder"]
    _, local, layout = sharder.shard_batch(Mesh(rank), None, batch, loss_mask=mask)
    assert layout.original_seq_len == 10 and layout.padded_seq_len == 12
    start = rank * 6
    expected_docs = torch.tensor([[0] * 4 + [1] * 6 + [-1] * 2, [0] * 5 + [1] * 5 + [-1] * 2])[
        :chunks, start : start + 6
    ]
    torch.testing.assert_close(local["_mistral_packed_document_ids"], expected_docs)
    expected_pad = torch.tensor(
        [[False] * 3 + [True] + [False] * 5 + [True] * 3, [False] * 4 + [True] + [False] * 4 + [True] * 3]
    )[:chunks, start : start + 6]
    torch.testing.assert_close(local["padding_mask"], expected_pad)
    torch.testing.assert_close(local["loss_mask"], torch.nn.functional.pad(mask, (0, 2))[:, start : start + 6])
    assert local["cu_seqlens"].shape == (chunks, 3)
    assert local["input_ids"].shape == (chunks, 6 if primary else 10)
    if not primary:
        assert local["attention_mask"].shape == ids.shape
    assert batch["input_ids"].shape == (chunks, 10)


def test_packed_capabilities_and_constructor_rejections() -> None:
    model = MistralLarge4ForCausalLM(tiny_config())
    supports = ModelSupports(model, SimpleNamespace(cp_size=2, tp_size=1, pp_size=2, ep_size=1))
    assert supports.supports_cp and not supports.supports_cp_with_sequence_packing
    with pytest.raises(ValueError, match="packed THD"):
        cp.prepare_packed_cp_inputs({"input_ids": torch.ones(1, 8, dtype=torch.long)}, num_chunks=1, shard_primary=True)


def test_public_sharder_selects_the_model_owned_packed_path() -> None:
    config = tiny_config(dtype="bfloat16", qk_nope_head_dim=60, qk_rope_head_dim=4, v_head_dim=64)
    model = MistralLarge4ForCausalLM(config)
    mesh = Mesh()
    model.cp_mesh = mesh
    group = SimpleNamespace(size=lambda: 2)
    mesh.get_group = lambda: group

    class DeviceMesh:
        mesh_dim_names = ("cp",)

        def __getitem__(self, name: str) -> Mesh:
            return mesh

    batch = dict(
        input_ids=torch.arange(8)[None],
        labels=torch.arange(8)[None],
        seq_lens=torch.tensor([[3, 5]]),
        seq_lens_padded=torch.tensor([[3, 5]]),
        qkv_format="thd",
    )
    sharder = ContextParallelSharder(model, DeviceMesh(), batch)
    _, local = sharder.shard(dict(batch))
    assert local["_mistral_packed_document_ids"].tolist() == [[0, 0, 0, 1]]


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("ac", [False, True])
def test_prepared_packed_model_matches_unsharded_thd_with_pp_and_shared_gradients(
    pipeline: bool, multimodal: bool, ac: bool
) -> None:
    """Compare real SDPA packed forwards and gradients with native THD execution."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn
    from tests.unit_tests.models.mistral_large4.test_multimodal import image_batch
    from tests.unit_tests.models.mistral_large4.test_pipeline import make_pipeline_model, partition_model

    original = make_pipeline_model(dtype=torch.bfloat16, multimodal=multimodal)
    config = original.config
    reference = type(original)(config)
    reference.initialize_weights()
    model = type(original)(config)
    model.load_state_dict(reference.state_dict())
    parts = partition_model(model, (0, 2, 4), ac=ac) if pipeline else [model]
    if ac and not pipeline:
        for i, block in enumerate(model.model.layers):
            model.model.layers[i] = checkpoint_wrapper(block)
    mesh = Mesh(size=1)
    mesh.get_group = lambda: None
    for part in parts:
        part.cp_mesh = mesh
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    media = {}
    lengths = [3, 5]
    if multimodal:
        ids, pixels = image_batch()
        media = {"pixel_values": [p.bfloat16() for p in pixels]}
        lengths = [ids.shape[1]]
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    positions = torch.cat([torch.arange(n) for n in lengths])[None]
    batch = dict(
        input_ids=ids,
        labels=ids.clone(),
        position_ids=positions,
        seq_lens=torch.tensor([lengths]),
        seq_lens_padded=torch.tensor([lengths]),
        qkv_format="thd",
        **media,
    )
    prepared = parts[0].prepare_model_inputs_for_cp(batch)["cp_sharder"]
    _, local, _ = prepared.shard_batch(mesh, None, batch)
    local.pop("labels")
    actual = parts[0](**local)
    for part in parts[1:]:
        kwargs = {k: v for k, v in local.items() if k not in ("input_ids", "pixel_values")}
        actual = part(*actual, **kwargs)
    actual = actual if pipeline else actual.logits
    expected = reference(ids, qkv_format="thd", cu_seqlens=boundaries, position_ids=positions, **media).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    ref_parameters = dict(reference.named_parameters())
    diff2, norm2 = 0.0, 0.0
    for part in parts:
        for name, parameter in part.named_parameters():
            ref = ref_parameters[canonical_parameter_fqn(name)]
            if not parameter.requires_grad:
                assert parameter.grad is None
                continue
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
            diff2 += float(((parameter.grad.float() - ref.grad.float()) ** 2).sum())
            norm2 += float((ref.grad.float() ** 2).sum())
    assert (diff2 / norm2) ** 0.5 <= 0.005


def test_packed_cp1_without_device_mesh_preserves_token_verbs() -> None:
    model = MistralLarge4ForCausalLM(tiny_config())
    model.initialize_weights()
    batch = dict(
        input_ids=torch.tensor([[1, 2, 3, 4, 5]]),
        labels=torch.tensor([[2, -100, 4, 5, -100]]),
        seq_lens=torch.tensor([[2, 3]]),
        seq_lens_padded=torch.tensor([[2, 3]]),
        qkv_format="thd",
    )
    sharder = ContextParallelSharder(model, None, batch)
    _, local = sharder.shard(batch)
    labels = local.pop("labels")
    actual = model(**local).logits
    torch.testing.assert_close(sharder.gather_token_tensor(actual, trim=True), actual)
    torch.testing.assert_close(sharder.shard_token_tensor(labels), labels)
    expected = model(batch["input_ids"], qkv_format="thd", cu_seqlens=torch.tensor([0, 2, 5])).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("rank", [0, 1])
def test_packed_cp_preserves_real_eos_when_eos_is_the_padding_token(rank: int) -> None:
    """Pack lengths must distinguish real EOS from identical-valued padding."""
    batch = dict(
        input_ids=torch.tensor([[4, 5, 0, 0, 6, 7, 0, 0]]),
        labels=torch.tensor([[5, 0, -100, -100, 7, 0, -100, -100]]),
        seq_lens=torch.tensor([[3, 3]]),
        seq_lens_padded=torch.tensor([[4, 4]]),
        position_ids=torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3]]),
        qkv_format="thd",
    )
    sharder = cp.prepare_packed_cp_inputs(batch, num_chunks=1, shard_primary=True)["cp_sharder"]
    _, local, _ = sharder.shard_batch(Mesh(rank), None, batch, padding_token_id=0)
    assert local["input_ids"][0, 2:].tolist() == [0, 0]
    assert local["padding_mask"].tolist() == [[False, False, False, True]]
    assert local["labels"].tolist() == [[5 if rank == 0 else 7, 0, -100, -100]]
    assert local["position_ids"].tolist() == [[0, 1, 2, 3]]
    assert local["_mistral_packed_document_ids"].tolist() == [[rank] * 4]


def make_sparse_model(
    *, dtype: torch.dtype = torch.bfloat16, multimodal: bool = False, backend: str = "tilelang"
) -> MistralLarge4ForCausalLM:
    """Build a small four-layer model with the released 512+64 MLA dimensions."""
    template = make_pipeline_model(dtype=dtype, multimodal=multimodal)
    config = copy.deepcopy(template.config)
    config.kv_lora_rank = 512
    config.qk_rope_head_dim = 64
    config.qk_nope_head_dim = 64
    config.head_dim = 128
    config.v_head_dim = 128
    config.index_head_dim = 128
    model = type(template)(config, replace(template.backend, attn=backend))
    model.initialize_weights()
    with torch.no_grad():
        model.model.rotary_emb.alpha.copy_(torch.linspace(-0.07, 0.03, 32))
    return model


def dense_latent_oracle(
    q: torch.Tensor, kv: torch.Tensor, indices: torch.Tensor, value_weight: torch.Tensor, *, scale: float
) -> torch.Tensor:
    """CPU FP32 oracle for the kernel boundary, independent of the adapter.

    Args:
        q: Local [tokens, heads, 576] absorbed queries.
        kv: Global [key_tokens, 576] latent/rotary KV.
        indices: Integer [tokens, selected_keys], with -1 padding.
        value_weight: Differentiable [heads, value_dim, 512] output projection.
        scale: Attention score scale.

    Returns:
        Attention output [tokens, heads, value_dim], with zero empty rows.
        This replaces only the CUDA kernel for CPU algebra checks.
    """
    mask = (torch.arange(kv.shape[0], device=q.device)[None, None] == indices[..., None]).any(1)
    values = torch.einsum("sc,hdc->shd", kv[:, :512], value_weight)
    keys = kv[:, None].expand(-1, q.shape[1], -1)
    return F.scaled_dot_product_attention(
        q.transpose(0, 1),
        keys.transpose(0, 1),
        values.transpose(0, 1),
        attn_mask=mask[None],
        scale=scale,
    ).transpose(0, 1)


@pytest.mark.parametrize("pipeline", [False, True])
@pytest.mark.parametrize("lora", [False, True])
def test_absorbed_sparse_algebra_matches_expanded_attention_all_gradients(
    monkeypatch: pytest.MonkeyPatch, pipeline: bool, lora: bool
) -> None:
    torch.manual_seed(907)
    reference = make_sparse_model(dtype=torch.float32, backend="sdpa")
    if lora:
        peft = peft_config()
        peft.dropout = 0.0
        apply_lora_to_linear_modules(reference, peft)
        with torch.no_grad():
            for name, parameter in reference.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.01)
    model = copy.deepcopy(reference)
    for block in model.model.layers:
        block.self_attn.attn_backend = "tilelang"
    monkeypatch.setattr(attention, "sparse_mla", dense_latent_oracle)
    parts = partition_model(model, (0, 2, 4), ac=True) if pipeline else [model]
    ids = torch.tensor([[1, 2, 3, 0, 4, 5, 6, 0]])
    docs = torch.tensor([[1, 1, 1, 0, 2, 2, 2, 0]])
    kwargs = dict(attention_mask=docs)
    expected = reference(ids, **kwargs).logits
    actual = run_partition(parts, ids, **kwargs) if pipeline else model(ids, **kwargs).logits
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    upstream = torch.randn_like(expected)
    (actual * upstream).sum().backward()
    (expected * upstream).sum().backward()
    from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

    parameters = dict(reference.named_parameters())
    nonzero_kvb_lora = set()
    for part in parts:
        for name, parameter in part.named_parameters():
            name = canonical_parameter_fqn(name)
            ref = parameters[name]
            if not parameter.requires_grad:
                continue
            assert parameter.grad is not None and ref.grad is not None, name
            torch.testing.assert_close(parameter.grad, ref.grad, atol=2e-5, rtol=1e-4, msg=name)
            if ".kv_b_proj.lora_" in name and parameter.grad.abs().sum() > 0:
                nonzero_kvb_lora.add(name)
    if lora:
        assert len(nonzero_kvb_lora) == 8  # A and B in all four source/consumer layers.


def test_sparse_backend_capability_and_dimension_rejection() -> None:
    from types import SimpleNamespace

    model = make_sparse_model()
    topology = SimpleNamespace(cp_size=2, pp_size=2, tp_size=1, ep_size=1)
    assert ModelSupports(model, topology).supports_cp_with_sequence_packing
    config = copy.deepcopy(model.config)
    config.kv_lora_rank = 256
    with pytest.raises(ValueError, match="kv_lora_rank=512"):
        type(model)(config, model.backend)


def test_sparse_absorption_rejects_active_kvb_lora_dropout(monkeypatch: pytest.MonkeyPatch) -> None:
    model = make_sparse_model(dtype=torch.float32, backend="sdpa")
    peft = peft_config()
    peft.dropout = 0.1
    apply_lora_to_linear_modules(model, peft)
    for block in model.model.layers:
        block.self_attn.attn_backend = "tilelang"
    monkeypatch.setattr(attention, "sparse_mla", dense_latent_oracle)
    with pytest.raises(RuntimeError, match="active LoRA training dropout"):
        model(torch.tensor([[1, 2, 3]]))
