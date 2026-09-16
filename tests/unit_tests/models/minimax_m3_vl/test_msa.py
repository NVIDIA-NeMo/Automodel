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

"""CPU contracts of the MSA microbatch, the construction-time gates and the packed-loader declaration."""

import copy
import subprocess
import sys
import weakref

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import TEFp8Config
from nemo_automodel.components.models.minimax_m3_vl import msa
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3MSAAttention
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
    MiniMaxM3TextModel,
)
from tests.unit_tests.models.minimax_m3_vl.conftest import VISION_CONFIG

_FORCED = (0, 1)
_SPARSE = {
    "use_sparse_attention": True,
    "sparse_num_index_heads": 4,
    "sparse_index_dim": 128,
    "sparse_block_size": 128,
    "sparse_topk_blocks": 16,
    "sparse_init_block": 0,
    "sparse_local_block": 1,
    "sparse_score_type": "max",
}


def _build(hidden: torch.Tensor, **sources: torch.Tensor | None) -> msa.MSAMicrobatch:
    """Build a microbatch from hidden[batch, sequence, hidden] and the given document-map sources; the rest are None."""
    sources = {"packed_seq_ids": None, "attention_mask": None, "padding_mask": None} | sources
    return msa.MSAMicrobatch.build(hidden, attn_kwargs={}, forced_blocks=_FORCED, **sources)


def test_pack_and_unpack_round_trip_real_tokens_and_their_gradients() -> None:
    torch.manual_seed(42)
    doc_ids = torch.zeros(3, 262, dtype=torch.int64)
    doc_ids[0, 1:128] = 42
    doc_ids[0, 130:259] = 7
    doc_ids[1, :128] = 7
    doc_ids[1, 128] = 9
    microbatch = msa.MSAMicrobatch.from_document_map(doc_ids, forced_blocks=_FORCED)
    external = torch.randn(3, 262, 3, requires_grad=True)
    upstream = torch.randn_like(external)
    packed = microbatch.pack(external)
    restored = microbatch.unpack(packed)
    keep = doc_ids > 0
    assert packed.shape == (385, 3)
    torch.testing.assert_close(packed, external[keep], rtol=0, atol=0)
    torch.testing.assert_close(restored[keep], external[keep], rtol=0, atol=0)
    assert torch.count_nonzero(restored[~keep]) == 0
    assert torch.equal(microbatch.padding_mask, ~keep)
    restored.backward(upstream)
    torch.testing.assert_close(external.grad[keep], upstream[keep], rtol=0, atol=0)
    assert torch.count_nonzero(external.grad[~keep]) == 0


def test_the_microbatch_exposes_the_packed_document_geometry() -> None:
    doc_ids = torch.zeros(2, 8, dtype=torch.int64)
    doc_ids[0, :3] = 1
    doc_ids[0, 3:5] = 2
    doc_ids[1, 2:6] = 1
    microbatch = msa.MSAMicrobatch.from_document_map(doc_ids, forced_blocks=_FORCED)

    cu_seqlens = microbatch.cu_seqlens
    assert cu_seqlens.dtype == torch.int32 and cu_seqlens.is_contiguous()
    assert cu_seqlens.tolist() == [0, 3, 5, 9]
    assert microbatch.max_seqlen == 4
    # Every cu_seqlens slice covers exactly one document, in pack row order.
    packed_ids = microbatch.pack(doc_ids.unsqueeze(-1)).squeeze(-1)
    assert int(cu_seqlens[-1]) == packed_ids.shape[0]
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist(), strict=True):
        assert packed_ids[start:end].unique().numel() == 1
    # Each document starts on a 128-aligned workspace row and its tokens count from that row.
    assert microbatch.document_workspace_starts.tolist() == [0, 128, 256] and microbatch.workspace_size == 384
    assert microbatch.document_positions.tolist() == [0, 1, 2, 0, 1, 0, 1, 2, 3]
    assert torch.equal(
        microbatch.workspace_positions, microbatch.document_positions + torch.tensor([0] * 3 + [128] * 2 + [256] * 4)
    )


def test_document_map_sources_and_precedence() -> None:
    documents = torch.tensor([[9, 9, 0, 4, 4]])
    keep = documents > 0
    real = torch.tensor([[True, True, True, True, False]])
    dense = (documents.unsqueeze(-1) == documents.unsqueeze(-2)) & keep.unsqueeze(-1) & keep.unsqueeze(-2)
    dense = (dense & torch.ones(5, 5, dtype=torch.bool).tril()).unsqueeze(1)
    # (sources, cu_seqlens, padding): the packed ids, a 2-D document map or the loader's dense block-causal
    # mask name two documents; a bool mask or a padding mask names one document per row ahead of its
    # padding; nothing at all names one full document per row.
    cases = [
        (
            {"packed_seq_ids": documents, "attention_mask": torch.ones_like(keep), "padding_mask": ~keep},
            [0, 2, 4],
            ~keep,
        ),
        ({"attention_mask": documents}, [0, 2, 4], ~keep),
        ({"attention_mask": dense}, [0, 2, 4], ~keep),
        ({"attention_mask": real}, [0, 4], ~real),
        ({"padding_mask": ~real}, [0, 4], ~real),
        ({}, [0, 5], torch.zeros_like(keep)),
    ]
    for sources, cu_seqlens, padding in cases:
        microbatch = _build(torch.empty(1, 5, 8), **sources)
        assert microbatch.cu_seqlens.tolist() == cu_seqlens and torch.equal(microbatch.padding_mask, padding)


def test_a_dense_mask_that_is_not_block_causal_is_rejected() -> None:
    with pytest.raises(ValueError, match="standard bool block-causal"):
        _build(torch.empty(1, 4, 8), attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool))


@pytest.mark.parametrize(
    "documents,match",
    [
        (torch.ones(4, dtype=torch.int64), r"\[batch, sequence\]"),
        (torch.ones(1, 4), "integer tensor"),
        (torch.tensor([[1, -1, 1]]), "non-negative"),
        (torch.zeros(1, 4, dtype=torch.int64), "at least one real token"),
        (torch.tensor([[1, 0, 1]]), "contiguous run"),
    ],
)
def test_invalid_documents(documents: torch.Tensor, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        msa.MSAMicrobatch.from_document_map(documents, forced_blocks=_FORCED)


def test_one_microbatch_per_batch_tensor() -> None:
    # Every virtual pipeline stage of one microbatch resolves the same batch tensor and must share one
    # state; a rewritten tensor, another tensor, or no tensor at all each get their own.
    packed_seq_ids = torch.tensor([[1, 1, 2, 2, 0]])
    hidden = torch.empty(1, 5, 8)
    first = _build(hidden, packed_seq_ids=packed_seq_ids)
    assert all(_build(hidden, packed_seq_ids=packed_seq_ids) is first for _ in range(6))
    packed_seq_ids.fill_(1)
    rewritten = _build(hidden, packed_seq_ids=packed_seq_ids)
    assert rewritten is not first and rewritten.cu_seqlens.tolist() == [0, 5]
    assert _build(hidden, packed_seq_ids=packed_seq_ids.clone()) is not rewritten
    assert _build(hidden, attention_mask=packed_seq_ids) is not _build(hidden, attention_mask=packed_seq_ids)
    # A second model sharing the batch (a KD teacher) must not inherit this model's forced blocks.
    other_rule = msa.MSAMicrobatch.build(
        hidden,
        packed_seq_ids=packed_seq_ids,
        attention_mask=None,
        padding_mask=None,
        attn_kwargs={},
        forced_blocks=(1, 1),
    )
    assert other_rule is not rewritten and other_rule.forced_blocks == (1, 1)
    # The memo holds the batch weakly: dropping the batch drops its state.
    alive = weakref.ref(rewritten)
    del first, rewritten, packed_seq_ids
    assert alive() is None


def _config(**overrides: object) -> MiniMaxM3VLTextConfig:
    """The smallest text config at MSA's fixed topology: 1 dense + 1 sparse layer, hidden 32."""
    sparse = _SPARSE | {"sparse_attention_freq": [0, 1], "sparse_disable_index_value": [0, 1]}
    sparse |= {k: overrides.pop(k) for k in list(overrides) if k.startswith("sparse_")}
    settings = dict(
        hidden_size=32,
        intermediate_size=32,
        dense_intermediate_size=48,
        shared_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        vocab_size=64,
        rotary_dim=64,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=0,
        moe_layer_freq=[0, 0],
        num_mtp_modules=0,
        sparse_attention_config=sparse,
    )
    return MiniMaxM3VLTextConfig(**(settings | overrides))


def _backend(sparse_attn: str = "msa", **overrides: object) -> BackendConfig:
    settings = dict(attn="te", sparse_attn=sparse_attn, linear="torch", rms_norm="torch", rope_fusion=False)
    return BackendConfig(**(settings | overrides))


def _msa_attention(
    config: MiniMaxM3VLTextConfig | None = None, backend: BackendConfig | None = None
) -> MiniMaxM3MSAAttention:
    with torch.device("meta"):
        return MiniMaxM3MSAAttention(config or _config(), backend or _backend())


@pytest.mark.parametrize(
    "field,value",
    # score_type: the fused scorer reports unscaled QK maxima, so "lse" would rank differently.
    # index dim: the fused scorer's QK tile fixes the channel extent at 128 and checks nothing.
    [("num_attention_heads", 32), ("head_dim", 64), ("sparse_score_type", "lse"), ("sparse_index_dim", 192)],
)
def test_fixed_topology(field: str, value: int | str) -> None:
    with pytest.raises(ValueError, match="requires"):
        _msa_attention(config=_config(**{field: value}))


def test_unsupported_backend() -> None:
    # Set after construction: BackendConfig.__post_init__ would normalize either against the other fields.
    fused_rope, fp8 = _backend(), _backend()
    fused_rope.rope_fusion = True
    fp8.te_fp8 = TEFp8Config()
    for backend in (fused_rope, fp8):
        with pytest.raises(NotImplementedError):
            _msa_attention(backend=backend)


@pytest.mark.parametrize(
    "runtime,match",
    [
        ({"qkv_format": "thd"}, "BSHD"),
        ({"use_cache": True}, "cache-free prefill"),
        ({"is_causal": False}, "causal self-attention"),
    ],
)
def test_unsupported_runtime(runtime: dict[str, object], match: str) -> None:
    with pytest.raises(NotImplementedError, match=match):
        msa.MSAMicrobatch.build(
            torch.empty(1, 4, 8),
            packed_seq_ids=None,
            attention_mask=None,
            padding_mask=None,
            attn_kwargs=runtime,
            forced_blocks=_FORCED,
        )


def test_context_parallelism_is_rejected_at_setup() -> None:
    # moe/parallelizer.py's apply_cp dispatches on hasattr(self_attn, "setup_cp_attention"); without
    # this method CP would be skipped with only a log warning and the run would silently be wrong.
    with pytest.raises(NotImplementedError, match="cp_size=1"):
        _msa_attention().setup_cp_attention(cp_mesh=None)


def test_the_generic_attention_backend_is_not_installed() -> None:
    # A leftover TE DotProductAttention would also route apply_cp into TE's CP branch instead of
    # setup_cp_attention.
    attention = _msa_attention()
    assert attention.attn_module is None and attention.attn_func is None


def test_deterministic_algorithms_are_rejected_before_any_kernel_runs() -> None:
    # The backward accumulates dK/dV with fp32 atomics and dQ with packed bf16 atomics, so a run that
    # asked for determinism must be told, not silently given a non-reproducible result.
    microbatch = msa.MSAMicrobatch.from_document_map(torch.ones(1, 8, dtype=torch.int64), forced_blocks=_FORCED)
    empty = torch.empty(0)
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(NotImplementedError, match="not bitwise deterministic"):
            msa.sparse_attention(empty, empty, empty, empty, microbatch)
    finally:
        torch.use_deterministic_algorithms(False)


@pytest.mark.timeout(130)
def test_optional_dependencies_are_lazy() -> None:
    # Importing the model package and building a microbatch must not touch the msa extra;
    # test_msa_import_guard covers the error a host without the extra gets at the first kernel call.
    # A fresh child must import torch and Automodel; revisit this exception when that cold start
    # reliably fits the default 5s unit-test timeout in the CI container. The pytest budget must
    # exceed subprocess.run's 120s limit so it can terminate the child first.
    script = """
import sys
import torch
class RejectGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fmha_sm100", "cutlass", "quack"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectGpuImports())
from nemo_automodel.components.models.minimax_m3_vl import model, msa
microbatch = msa.MSAMicrobatch.from_document_map(torch.ones(1, 8, dtype=torch.int64), forced_blocks=(0, 1))
assert microbatch.cu_seqlens.tolist() == [0, 8]
kernel_prefix = "nemo_automodel.components.models.minimax_m3_vl.kernels."
private_kernels = {
    kernel_prefix + name
    for name in (
        "msa_backward_sm100",
        "msa_task_build_sm100",
        "msa_backward_preprocess_sm100",
        "msa_backward_postprocess_sm100",
    )
}
loaded_private_kernels = private_kernels.intersection(sys.modules)
assert not loaded_private_kernels, loaded_private_kernels
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=120)


@pytest.mark.parametrize("sparse_attn,declared", [("msa", True), ("generic", False)])
def test_the_model_declares_whether_it_consumes_packed_seq_ids(sparse_attn: str, declared: bool) -> None:
    # PR #3831: the packed loader asks the first model part once and then hands it the compact document
    # map for every pack, including single-document ones, instead of a dense mask. The answer is
    # decided on the whole model at construction, so a pipeline stage's deep copy answers the same.
    # All-sparse so no dense layer is built: a TE-backed one cannot be constructed on the CPU job.
    text = _config(sparse_attention_freq=[1, 1], sparse_disable_index_value=[1, 1])
    backend = _backend(sparse_attn, attn="sdpa")
    with torch.device("meta"):
        model = MiniMaxM3SparseForCausalLM(text, backend=backend)
        vlm = MiniMaxM3SparseForConditionalGeneration(
            MiniMaxM3VLConfig(vision_config=dict(VISION_CONFIG), text_config=text), backend=backend
        )
    assert model.consumes_packed_seq_ids is vlm.consumes_packed_seq_ids is declared
    stage = copy.deepcopy(model)
    stage.model.layers["1"] = None
    assert stage.consumes_packed_seq_ids is declared


def test_msa_with_dense_layers_requires_a_varlen_attention_backend() -> None:
    # Dense layers are packed to [tokens, hidden] once MSA is on, so they isolate documents with
    # cu_seqlens; sdpa drops cu_seqlens on the floor (attention/utils.py:207-212).
    with torch.device("meta"):
        with pytest.raises(NotImplementedError, match="backend.attn='te'"):
            MiniMaxM3TextModel(_config(), _backend(attn="sdpa"))
        MiniMaxM3TextModel(_config(), _backend("generic", attn="sdpa"))
