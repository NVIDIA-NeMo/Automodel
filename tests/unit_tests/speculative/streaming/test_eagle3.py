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

"""Unit tests for the EAGLE-3 streaming schema.

Three contracts are checked here:

1. :func:`eagle3_logits_tensors` packs the colocated-path encoder's
   tensors under the documented key set, in insertion order, so a
   :class:`SampleRef` derived from it has a stable feature ordering.
2. :func:`eagle3_logits_feature_specs` mirrors the tensor dict's
   ``shape`` / ``dtype`` so :meth:`FeatureStore.get` can detect a
   producer-side drift.
3. :func:`validate_eagle3_ref` enforces the "core features present" and
   "exactly one supervision encoding" rules from RFC §"Feature schema
   per algorithm", including the negative case where neither encoding
   is present.
"""

from __future__ import annotations

import pytest
import torch

from nemo_automodel.components.speculative.streaming.eagle3 import (
    EAGLE3_CORE_FEATURES,
    EAGLE3_DRAFT_VOCAB_SUPERVISION,
    EAGLE3_LOGITS_SUPERVISION,
    EAGLE3_SCHEMA_VERSION,
    EAGLE3_SUPERVISION_ENCODINGS,
    eagle3_logits_feature_specs,
    eagle3_logits_tensors,
    validate_eagle3_ref,
)
from nemo_automodel.components.speculative.streaming.refs import (
    FeatureAlgorithm,
    FeatureSpec,
    SampleRef,
)


def _ref(
    *,
    include_logits: bool = True,
    include_draft: bool = False,
    omit_core: tuple[str, ...] = (),
    algorithm: FeatureAlgorithm = FeatureAlgorithm.EAGLE3,
) -> SampleRef:
    """Build a :class:`SampleRef` for the schema tests.

    Bypasses :meth:`SampleRef.__post_init__` so the schema validator
    under test is the only check the ref runs through; this is what
    lets each test exercise one specific failure mode without
    being pre-filtered by the constructor.
    """
    feature_keys = {
        "aux_hidden_states": "s/aux_hidden_states",
        "input_ids": "s/input_ids",
        "attention_mask": "s/attention_mask",
        "loss_mask": "s/loss_mask",
    }
    feature_specs = {
        "aux_hidden_states": FeatureSpec(shape=(2, 8), dtype=torch.float32),
        "input_ids": FeatureSpec(shape=(2, 8), dtype=torch.long),
        "attention_mask": FeatureSpec(shape=(2, 8), dtype=torch.long),
        "loss_mask": FeatureSpec(shape=(2, 8), dtype=torch.long),
    }
    if include_logits:
        feature_keys["logits"] = "s/logits"
        feature_specs["logits"] = FeatureSpec(shape=(2, 8, 32), dtype=torch.float32)
    if include_draft:
        feature_keys["target_probs"] = "s/target_probs"
        feature_keys["position_mask"] = "s/position_mask"
        feature_specs["target_probs"] = FeatureSpec(shape=(2, 8, 16), dtype=torch.float32)
        feature_specs["position_mask"] = FeatureSpec(shape=(2, 8), dtype=torch.bool)
    for name in omit_core:
        feature_keys.pop(name, None)
        feature_specs.pop(name, None)
    ref = object.__new__(SampleRef)
    object.__setattr__(ref, "sample_id", "s")
    object.__setattr__(ref, "run_id", "r")
    object.__setattr__(ref, "store_uri", "mem://test")
    object.__setattr__(ref, "feature_keys", feature_keys)
    object.__setattr__(ref, "feature_specs", feature_specs)
    object.__setattr__(ref, "algorithm", algorithm)
    object.__setattr__(ref, "schema_version", 1)
    object.__setattr__(ref, "num_tokens", 16)
    object.__setattr__(ref, "estimated_bytes", 64)
    object.__setattr__(ref, "target_model_version", "0")
    object.__setattr__(ref, "draft_weight_version", "0")
    return ref


# --- 1. Tensor-dict packing + insertion order -----------------------------


def test_eagle3_logits_tensors_includes_packing_metadata_when_provided() -> None:
    aux = torch.zeros(2, 8, 96)
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    logits = torch.zeros(2, 8, 32)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    seq_lens = torch.tensor([[4, 4], [8, 0]], dtype=torch.long)
    doc_remaining = torch.ones(2, 8, dtype=torch.long)
    out = eagle3_logits_tensors(
        aux_hidden_states=aux,
        input_ids=ids,
        attention_mask=mask,
        loss_mask=loss,
        logits=logits,
        position_ids=position_ids,
        seq_lens=seq_lens,
        doc_remaining=doc_remaining,
    )
    assert set(out) == set(EAGLE3_SUPERVISION_ENCODINGS) | {"position_ids", "seq_lens", "doc_remaining"}
    specs = eagle3_logits_feature_specs(aux, ids, mask, loss, logits, position_ids, seq_lens, doc_remaining)
    assert specs["seq_lens"].shape == (2, 2)


def test_eagle3_logits_tensors_packs_documented_key_set() -> None:
    aux = torch.zeros(2, 8, 96)
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    logits = torch.zeros(2, 8, 32)
    out = eagle3_logits_tensors(
        aux_hidden_states=aux, input_ids=ids, attention_mask=mask, loss_mask=loss, logits=logits
    )
    assert set(out) == set(EAGLE3_SUPERVISION_ENCODINGS)
    # Insertion order matches the colocated Eagle3TargetBatch field set so
    # a SampleRef derived from it shares the colocated ordering.
    assert tuple(out) == EAGLE3_SUPERVISION_ENCODINGS


def test_eagle3_logits_feature_specs_match_tensor_shapes_and_dtypes() -> None:
    aux = torch.zeros(2, 8, 96)
    ids = torch.zeros(2, 8, dtype=torch.long)
    mask = torch.ones(2, 8, dtype=torch.long)
    loss = torch.ones(2, 8, dtype=torch.long)
    logits = torch.zeros(2, 8, 32)
    specs = eagle3_logits_feature_specs(aux, ids, mask, loss, logits)
    assert specs["aux_hidden_states"].shape == (2, 8, 96)
    assert specs["aux_hidden_states"].dtype is torch.float32
    assert specs["input_ids"].dtype is torch.long
    assert specs["logits"].dtype is torch.float32
    assert specs["logits"].shape == (2, 8, 32)


# --- 2. validate_eagle3_ref ----------------------------------------------


def test_validate_accepts_logits_supervision() -> None:
    validate_eagle3_ref(_ref(include_logits=True))


def test_validate_accepts_draft_vocab_supervision() -> None:
    validate_eagle3_ref(_ref(include_logits=False, include_draft=True))


def test_validate_rejects_non_eagle3_algorithm() -> None:
    ref = _ref(algorithm=FeatureAlgorithm.DFLASH)
    with pytest.raises(ValueError, match=r"must be .*EAGLE3.* for an EAGLE-3 consumer"):
        validate_eagle3_ref(ref)


def test_validate_rejects_wrong_schema_version() -> None:
    ref = _ref(include_logits=True)
    object.__setattr__(ref, "schema_version", 0)
    with pytest.raises(ValueError, match=f"schema_version must be {EAGLE3_SCHEMA_VERSION}"):
        validate_eagle3_ref(ref)


def test_validate_rejects_partial_packing_features() -> None:
    ref = _ref(include_logits=True)
    object.__setattr__(
        ref,
        "feature_specs",
        {**ref.feature_specs, "seq_lens": FeatureSpec(shape=(2, 2), dtype=torch.long)},
    )
    object.__setattr__(ref, "feature_keys", {**ref.feature_keys, "seq_lens": "s/seq_lens"})
    with pytest.raises(ValueError, match="position_ids, seq_lens, and doc_remaining together"):
        validate_eagle3_ref(ref)


@pytest.mark.parametrize("missing", list(EAGLE3_CORE_FEATURES))
def test_validate_rejects_missing_core_feature(missing) -> None:
    with pytest.raises(ValueError, match="missing required core features"):
        validate_eagle3_ref(_ref(omit_core=(missing,)))


def test_validate_rejects_neither_supervision() -> None:
    with pytest.raises(ValueError, match="exactly one supervision encoding"):
        validate_eagle3_ref(_ref(include_logits=False, include_draft=False))


def test_validate_rejects_both_supervisions() -> None:
    with pytest.raises(ValueError, match="exactly one supervision encoding"):
        validate_eagle3_ref(_ref(include_logits=True, include_draft=True))


def test_validate_constant_draft_supervision_lists_target_probs_and_position_mask() -> None:
    # A regression in the supervision-key constants would invalidate the
    # validator; pin them so the change must be deliberate.
    assert set(EAGLE3_LOGITS_SUPERVISION) == {"logits"}
    assert set(EAGLE3_DRAFT_VOCAB_SUPERVISION) == {"target_probs", "position_mask"}
