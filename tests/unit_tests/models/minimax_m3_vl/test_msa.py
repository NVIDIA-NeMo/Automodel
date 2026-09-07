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

"""CPU contracts for the model-owned MSA interface."""

import subprocess
import sys

import pytest
import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import TEFp8Config
from nemo_automodel.components.models.minimax_m3_vl import _msa as msa
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.msa_attn import MiniMaxM3MSAAttention
from nemo_automodel.shared.import_utils import UnavailableError


def test_packed_layout_roundtrip_and_gradients() -> None:
    torch.manual_seed(42)
    doc_ids = torch.zeros(3, 262, dtype=torch.int64)
    doc_ids[0, 1:128] = 42
    doc_ids[0, 130:259] = 7
    doc_ids[1, :128] = 7
    doc_ids[1, 128] = 9
    layout = msa._MSAPackedLayout.build(doc_ids)
    external = torch.randn(3, 262, 3, requires_grad=True)
    upstream = torch.randn_like(external)
    packed = layout.pack(external)
    restored = layout.unpack(packed)
    keep = doc_ids > 0
    assert packed.shape == (385, 3)
    torch.testing.assert_close(packed, external[keep], rtol=0, atol=0)
    torch.testing.assert_close(restored[keep], external[keep], rtol=0, atol=0)
    assert torch.count_nonzero(restored[~keep]) == 0
    restored.backward(upstream)
    torch.testing.assert_close(external.grad[keep], upstream[keep], rtol=0, atol=0)
    assert torch.count_nonzero(external.grad[~keep]) == 0


def test_document_map_sources_and_precedence() -> None:
    documents = torch.tensor([[9, 9, 0, 4, 4]])
    keep = documents > 0
    mask = (documents.unsqueeze(-1) == documents.unsqueeze(-2)) & keep.unsqueeze(-1) & keep.unsqueeze(-2)
    mask = (mask & torch.ones(5, 5, dtype=torch.bool).tril()).unsqueeze(1)
    cases = [
        (documents, torch.ones_like(keep), ~keep, documents),
        (None, documents, None, documents),
        (None, mask, None, torch.tensor([[1, 1, 0, 2, 2]])),
        (None, keep, None, keep.long()),
        (None, None, ~keep, keep.long()),
        (None, None, None, torch.ones_like(documents)),
    ]
    for packed_ids, attention_mask, padding_mask, expected in cases:
        actual = msa._resolve_canonical_document_map(
            torch.empty(1, 5, 8), packed_seq_ids=packed_ids, attention_mask=attention_mask, padding_mask=padding_mask
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert actual.is_contiguous()


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
    """Reject invalid integer document ids, expected shape [batch, sequence]."""
    with pytest.raises(ValueError, match=match):
        msa._MSAPackedLayout.build(documents)


def test_noncausal_mask_rejected() -> None:
    with pytest.raises(ValueError, match="standard bool block-causal"):
        msa._resolve_canonical_document_map(
            torch.empty(1, 4, 8),
            packed_seq_ids=None,
            attention_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
            padding_mask=None,
        )


@pytest.mark.parametrize("field,value", [("num_heads", 32), ("head_dim", 64), ("attention_dropout", 0.1)])
def test_fixed_topology(field: str, value: int | float) -> None:
    topology = dict(
        num_heads=64,
        num_kv_heads=4,
        head_dim=128,
        num_index_heads=4,
        block_size=128,
        topk_blocks=16,
        attention_dropout=0.0,
    )
    msa._validate_msa_topology(**topology)
    topology[field] = value
    with pytest.raises(ValueError, match="requires|supports"):
        msa._validate_msa_topology(**topology)


@pytest.mark.parametrize("field", ["rope_fusion", "te_fp8"])
def test_unsupported_backend(field: str) -> None:
    backend = BackendConfig(rope_fusion=False)
    if field == "rope_fusion":
        backend.rope_fusion = True
    else:
        backend.te_fp8 = TEFp8Config()
    with pytest.raises(NotImplementedError):
        msa._reject_unsupported_msa_configuration(backend)


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
        msa._reject_unsupported_msa_runtime(runtime)


def _msa_attention() -> MiniMaxM3MSAAttention:
    """Build one MSA attention layer on the meta device, at MSA's fixed 64/4/128 topology."""
    config = MiniMaxM3VLTextConfig(
        hidden_size=32,
        num_attention_heads=64,
        num_key_value_heads=4,
        head_dim=128,
        rotary_dim=64,
        attention_dropout=0.0,
        sparse_attention_config={
            "use_sparse_attention": True,
            "sparse_num_index_heads": 4,
            "sparse_index_dim": 128,
            "sparse_block_size": 128,
            "sparse_topk_blocks": 16,
            "sparse_init_block": 0,
            "sparse_local_block": 1,
            "sparse_score_type": "max",
        },
    )
    backend = BackendConfig(attn="te", sparse_attn="msa", linear="torch", rms_norm="torch", rope_fusion=False)
    with torch.device("meta"):
        return MiniMaxM3MSAAttention(config, backend)


def test_context_parallelism_is_rejected_at_setup() -> None:
    # moe/parallelizer.py:991 dispatches on hasattr(self_attn, "setup_cp_attention"); without this
    # method CP would be skipped with only a log warning and the run would silently be wrong.
    with pytest.raises(NotImplementedError, match="cp_size=1"):
        _msa_attention().setup_cp_attention(cp_mesh=None)


def test_the_generic_attention_backend_is_not_installed() -> None:
    # A leftover TE DotProductAttention would also route apply_cp into TE's CP branch instead of
    # setup_cp_attention (moe/parallelizer.py:983).
    attention = _msa_attention()
    assert attention.attn_module is None and attention.attn_func is None


def test_deterministic_algorithms_are_rejected_before_any_kernel_runs() -> None:
    # The backward accumulates dK/dV with fp32 atomics and dQ with packed 16-bit atomics, so a run
    # that asked for determinism must be told, not silently given a non-reproducible result.
    layout = msa._MSAPackedLayout.build(torch.ones(1, 8, dtype=torch.int64))
    empty = torch.empty(0)
    torch.use_deterministic_algorithms(True)
    try:
        with pytest.raises(NotImplementedError, match="not bitwise deterministic"):
            msa._MSAFlatAttention(0.125)(empty, empty, empty, empty, layout=layout)
    finally:
        torch.use_deterministic_algorithms(False)


def test_optional_dependencies_are_lazy_and_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    script = """
import sys
class RejectGpuImports:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"fmha_sm100", "cutlass", "quack"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, RejectGpuImports())
from nemo_automodel.components.models.minimax_m3_vl._msa import _MSAFlatAttention
_MSAFlatAttention(0.125)
"""
    subprocess.run([sys.executable, "-c", script], check=True, timeout=60)
    monkeypatch.setattr(msa, "_resolve_msa_forward", lambda: None)
    monkeypatch.setattr(msa, "_resolve_msa_backward", lambda: None)
    for require in (msa._require_msa, msa._require_msa_backward):
        with pytest.raises(UnavailableError, match=r"uv sync --extra msa"):
            require()
