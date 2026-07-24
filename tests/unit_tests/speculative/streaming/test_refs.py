# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Unit tests for the streaming package's tensor-free control-plane contracts.

Three contracts are under test here:

1. :class:`FeatureSpec` -- shape + dtype round-trip on tuple / list inputs;
   rejects non-int / negative entries.
2. :class:`SampleRef` -- frozen, validates the field invariants, and is
   tensor-free at every depth (a tensor in any nested dict / list at
   construction time trips :func:`assert_no_tensors`).
3. :func:`assert_no_tensors` -- accepts the documented primitive set,
   rejects torch tensors, numpy arrays, and duck-typed tensor-likes
   anywhere in the structure, and accepts the empty case.

These tests are CPU-only (no GPU required) and cover the control-plane
contract surface that producers and consumers must honor.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from nemo_automodel.components.speculative.streaming.refs import (
    FeatureAlgorithm,
    FeatureSpec,
    SampleRef,
    assert_no_tensors,
)

# --- 1. FeatureSpec -------------------------------------------------------


def test_feature_spec_accepts_tuple_shape() -> None:
    spec = FeatureSpec(shape=(2, 8, 96), dtype=torch.bfloat16)
    assert spec.shape == (2, 8, 96)
    assert spec.dtype is torch.bfloat16


def test_feature_spec_normalizes_list_shape_to_tuple() -> None:
    spec = FeatureSpec(shape=[2, 8, 96], dtype=torch.bfloat16)  # type: ignore[arg-type]
    assert isinstance(spec.shape, tuple)
    assert spec.shape == (2, 8, 96)


def test_feature_spec_equality_across_equivalent_inputs() -> None:
    # Two FeatureSpec objects built from equivalent sequences compare
    # equal -- otherwise the consumer-side dict-key-by-spec comparison
    # would miss-cache.
    a = FeatureSpec(shape=(2, 8), dtype=torch.float32)
    b = FeatureSpec(shape=[2, 8], dtype=torch.float32)  # type: ignore[arg-type]
    assert a == b


@pytest.mark.parametrize(
    "bad_shape",
    [
        (2, -1, 8),
        ("a", "b"),
        (2, "8"),
    ],
)
def test_feature_spec_rejects_invalid_shapes(bad_shape) -> None:
    with pytest.raises(ValueError, match="tuple of non-negative ints"):
        # The tuple-of-floats case verifies the int check fires before any
        # shape normalization; the (int, str) cases verify non-int entries
        # are rejected.
        FeatureSpec(shape=bad_shape, dtype=torch.float32)  # type: ignore[arg-type]


def test_feature_spec_rejects_float_shape() -> None:
    with pytest.raises(ValueError, match="tuple of non-negative ints"):
        FeatureSpec(shape=(2.0, 8), dtype=torch.float32)  # type: ignore[arg-type]  # test parametrized bad shapes


# --- 2. SampleRef ---------------------------------------------------------


def _good_ref(**overrides) -> SampleRef:
    """Return a minimal valid EAGLE-3 :class:`SampleRef` (4 features)."""
    fields = {
        "sample_id": "s1",
        "run_id": "r1",
        "store_uri": "mem://test",
        "feature_keys": {
            "aux_hidden_states": "s1/aux_hidden_states",
            "input_ids": "s1/input_ids",
            "attention_mask": "s1/attention_mask",
            "loss_mask": "s1/loss_mask",
        },
        "feature_specs": {
            name: FeatureSpec(shape=(2, 8), dtype=torch.float32)
            for name in ("aux_hidden_states", "input_ids", "attention_mask", "loss_mask")
        },
        "algorithm": FeatureAlgorithm.EAGLE3,
        "schema_version": 1,
        "num_tokens": 16,
        "estimated_bytes": 256,
        "target_model_version": "0",
        "draft_weight_version": "0",
    }
    fields.update(overrides)
    return SampleRef(**fields)


def test_sample_ref_is_frozen() -> None:
    ref = _good_ref()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.sample_id = "different"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param({"sample_id": ""}, id="empty-sample_id"),
        pytest.param({"sample_id": 1}, id="non-str-sample_id"),
        pytest.param({"run_id": ""}, id="empty-run_id"),
        pytest.param({"store_uri": ""}, id="empty-store_uri"),
        pytest.param({"store_uri": "no-scheme"}, id="no-scheme"),
        pytest.param({"num_tokens": -1}, id="negative-num_tokens"),
        pytest.param({"estimated_bytes": -4}, id="negative-estimated_bytes"),
    ],
)
def test_sample_ref_rejects_invalid_invariants(bad) -> None:
    with pytest.raises(ValueError):
        _good_ref(**bad)


def test_sample_ref_rejects_mismatched_feature_keys_and_specs() -> None:
    with pytest.raises(ValueError, match="must name the same feature set"):
        _good_ref(
            feature_specs={
                "aux_hidden_states": FeatureSpec(shape=(2, 8), dtype=torch.float32),
                "input_ids": FeatureSpec(shape=(2, 8), dtype=torch.float32),
            },
        )


def test_sample_ref_requires_features_for_algorithm() -> None:
    # A DFLASH ref MUST name ``hidden_states`` + the standard three. Drop
    # ``loss_mask`` deliberately to trip the required-set check.
    feature_keys = {
        "hidden_states": "s/hidden_states",
        "input_ids": "s/input_ids",
        "attention_mask": "s/attention_mask",
        # "loss_mask" intentionally omitted
    }
    feature_specs = {name: FeatureSpec(shape=(2, 8), dtype=torch.float32) for name in feature_keys}
    with pytest.raises(ValueError, match="missing required features"):
        _good_ref(
            feature_keys=feature_keys,
            feature_specs=feature_specs,
            algorithm=FeatureAlgorithm.DFLASH,
        )


@pytest.mark.parametrize(
    "algo",
    [FeatureAlgorithm.EAGLE3, FeatureAlgorithm.DFLASH, FeatureAlgorithm.DSPARK],
)
def test_sample_ref_accepts_each_algorithm_with_its_required_features(algo) -> None:
    if algo is FeatureAlgorithm.EAGLE3:
        keys = ("aux_hidden_states", "input_ids", "attention_mask", "loss_mask")
    elif algo is FeatureAlgorithm.DFLASH:
        keys = ("hidden_states", "input_ids", "attention_mask", "loss_mask")
    elif algo is FeatureAlgorithm.DSPARK:
        keys = ("target_hidden_states", "target_last_hidden_states", "input_ids", "loss_mask")
    else:
        raise AssertionError(f"unhandled {algo}")
    feature_keys = {k: f"s/{k}" for k in keys}
    feature_specs = {k: FeatureSpec(shape=(2, 8), dtype=torch.float32) for k in keys}
    ref = _good_ref(algorithm=algo, feature_keys=feature_keys, feature_specs=feature_specs)
    assert ref.algorithm is algo
    assert ref.feature_names() == keys


def test_sample_ref_feature_names_preserves_insertion_order() -> None:
    # The order matters: some draft ``fc`` projections dimension on
    # *order* of the aux hidden states, not just count. Insertion order
    # in a Python 3.7+ dict is stable; the ref must preserve it.
    ordered_keys = ("loss_mask", "input_ids", "attention_mask", "aux_hidden_states")
    feature_keys = {k: f"s/{k}" for k in ordered_keys}
    feature_specs = {k: FeatureSpec(shape=(2, 8), dtype=torch.float32) for k in ordered_keys}
    ref = _good_ref(feature_keys=feature_keys, feature_specs=feature_specs)
    assert ref.feature_names() == ordered_keys


def test_sample_ref_rejects_tensor_in_nested_dict_at_construction() -> None:
    # Putting a tensor inside a feature_specs dict MUST fail at
    # SampleRef construction -- the ref itself, validated at __post_init__,
    # is the right place to enforce "the ref is tensor-free" rather than
    # relying on a producer remembering to call assert_no_tensors itself.
    feature_specs = {
        "aux_hidden_states": torch.zeros(2, 8),  # tensor where a FeatureSpec belongs
        "input_ids": FeatureSpec(shape=(2, 8), dtype=torch.float32),
        "attention_mask": FeatureSpec(shape=(2, 8), dtype=torch.float32),
        "loss_mask": FeatureSpec(shape=(2, 8), dtype=torch.float32),
    }
    with pytest.raises(ValueError, match="tensor-like object is not allowed"):
        _good_ref(feature_specs=feature_specs)


# --- 3. assert_no_tensors -------------------------------------------------


def test_assert_no_tensors_accepts_primitives() -> None:
    for value in [None, "", "hi", 0, 1, -1, 0.5, True, False, b"bytes"]:
        assert_no_tensors(value)


def test_assert_no_tensors_accepts_nested_dict_with_str_keys() -> None:
    payload = {"a": 1, "b": {"c": "x"}, "d": [1, 2, 3]}
    assert_no_tensors(payload)


def test_assert_no_tensors_rejects_torch_tensor() -> None:
    with pytest.raises(ValueError, match="tensor-like"):
        assert_no_tensors(torch.zeros(3))


def test_assert_no_tensors_rejects_tensor_in_list() -> None:
    with pytest.raises(ValueError, match=r"\[1\]"):
        assert_no_tensors([1, torch.zeros(2), 3])


def test_assert_no_tensors_rejects_tensor_in_dict_value() -> None:
    with pytest.raises(ValueError, match=r"\['bad'\]"):
        assert_no_tensors({"good": 1, "bad": torch.zeros(2)})


def test_assert_no_tensors_rejects_non_str_dict_key() -> None:
    with pytest.raises(ValueError, match="control-plane dicts must be str-keyed"):
        assert_no_tensors({1: "x"})


def test_assert_no_tensors_rejects_duck_typed_tensor_like() -> None:
    class _FakeTensor:
        data_ptr = lambda self: None  # noqa: E731
        is_cuda = False
        numel = lambda self: 0  # noqa: E731

    with pytest.raises(ValueError, match="tensor-like"):
        assert_no_tensors(_FakeTensor())


def test_assert_no_tensors_walks_into_nested_dataclass() -> None:
    inner = _good_ref()
    outer = {"ref": inner, "meta": "ok"}
    # inner already validated; reassembling at the outer level must still pass.
    assert_no_tensors(outer)


def test_assert_no_tensors_rejects_nested_tensor_in_dataclass() -> None:
    inner = _good_ref()
    payload = {"ref": inner, "injected": torch.zeros(2)}
    with pytest.raises(ValueError, match=r"\['injected'\]"):
        assert_no_tensors(payload)


def test_assert_no_tensors_path_argument_is_used_for_error_messages() -> None:
    with pytest.raises(ValueError, match=r"my\.path"):
        assert_no_tensors(torch.zeros(2), path="my.path")
