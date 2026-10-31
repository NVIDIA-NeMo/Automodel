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

"""Exercise strict released-checkpoint coverage without loading a publisher model."""

import copy
import runpy
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

VERIFIER = runpy.run_path(
    str(Path(__file__).resolve().parents[3] / "functional_tests/decision_models/d1/verify_released_checkpoint.py")
)


def test_release_precision_follows_publisher_guidance() -> None:
    assert VERIFIER["RELEASE_DTYPES"] == {"d1-3b": torch.bfloat16, "d1-omni": torch.float32}


def test_native_class_guard_accepts_real_capability_attachment() -> None:
    from nemo_automodel._transformers.capabilities import attach_capabilities_and_validate

    class Native(nn.Module):
        pass

    model = Native()
    attached = attach_capabilities_and_validate(model, None)
    assert type(attached) is not Native
    assert type(attached).__name__ == Native.__name__
    VERIFIER["_require_native_class"](attached, Native)


def test_native_class_guard_rejects_same_name_fallback() -> None:
    class Native(nn.Module):
        pass

    fallback = type("Native", (nn.Module,), {})()
    with pytest.raises(ValueError, match="Actual AutoModel loader selected") as error:
        VERIFIER["_require_native_class"](fallback, Native)
    assert f"{type(fallback).__module__}.{type(fallback).__qualname__}" in str(error.value)


def test_only_documented_tied_pair_expands() -> None:
    expand = VERIFIER["_canonical_checkpoint_keys"]
    keys = {"model.language_model.embed_tokens.weight", "other.weight"}
    assert expand(keys, tied=True) == keys | {"lm_head.weight"}
    assert expand(keys, tied=False) == keys
    assert expand({"unrelated.lm_head.weight"}, tied=True) == {"unrelated.lm_head.weight"}


@pytest.mark.parametrize("change", ["shape", "dtype", "value", "meta"])
def test_exact_tensor_check_rejects_incomplete_match(change: str) -> None:
    expected = torch.ones(2, 3)
    actual = {
        "shape": torch.ones(3, 2),
        "dtype": expected.bfloat16(),
        "value": torch.zeros_like(expected),
        "meta": torch.empty(2, 3, device="meta"),
    }[change]
    with pytest.raises(ValueError):
        VERIFIER["_require_tensor_equal"](actual, expected, name="probe.weight")


def _tied_model() -> nn.Module:
    model = nn.Module()
    model.model = nn.Module()
    model.model.language_model = nn.Module()
    model.model.language_model.embed_tokens = nn.Embedding(3, 4, dtype=torch.bfloat16)
    model.lm_head = nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    model.lm_head.weight = model.model.language_model.embed_tokens.weight
    model.extra = nn.Linear(4, 2, bias=False, dtype=torch.bfloat16)
    return model


def test_complete_checkpoint_allows_omitted_tied_alias(tmp_path: Path) -> None:
    reference = _tied_model()
    state = {name: value.clone() for name, value in reference.state_dict().items() if name != "lm_head.weight"}
    save_file(state, tmp_path / "model.safetensors")
    report = VERIFIER["_check_weights"](copy.deepcopy(reference), reference, tmp_path, tied=True)
    assert report["model_state_keys"] == 3
    assert report["serialized_weight_keys"] == 2
    assert report["complete_serialized_weight_coverage"]
    assert report["parameter_identity_ties"] == [["lm_head.weight", "model.language_model.embed_tokens.weight"]]


def test_missing_unrelated_checkpoint_weight_fails(tmp_path: Path) -> None:
    reference = _tied_model()
    save_file(
        {
            "model.language_model.embed_tokens.weight": reference.get_submodule(
                "model.language_model.embed_tokens"
            ).weight
        },
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="does not cover"):
        VERIFIER["_check_weights"](copy.deepcopy(reference), reference, tmp_path, tied=True)


def test_equal_values_without_parameter_tie_fail(tmp_path: Path) -> None:
    reference = _tied_model()
    native = copy.deepcopy(reference)
    native.lm_head.weight = nn.Parameter(native.lm_head.weight.detach().clone())
    with pytest.raises(ValueError, match="different ties"):
        VERIFIER["_check_weights"](native, reference, tmp_path, tied=True)


def _vision_model(prefix: str) -> nn.Module:
    model = nn.Module()
    parent = model
    for name in prefix.rstrip(".").split("."):
        child = nn.Module()
        parent.add_module(name, child)
        parent = child
    parent.add_module("projection", nn.Linear(3, 2, bias=False, dtype=torch.bfloat16))
    return model


@pytest.mark.parametrize("prefix", ["model.vision_tower.", "vision.tower."])
def test_verified_vision_wrapper_preserves_full_serialized_coverage(tmp_path: Path, prefix: str) -> None:
    reference = _vision_model(prefix)
    loaded_name = prefix + "projection.weight"
    serialized_name = prefix + "vision_model.projection.weight"
    save_file({serialized_name: reference.state_dict()[loaded_name]}, tmp_path / "model.safetensors")
    report = VERIFIER["_check_weights"](copy.deepcopy(reference), reference, tmp_path, tied=False)
    assert report["complete_serialized_weight_coverage"]
    assert report["model_state_keys"] == report["serialized_weight_keys"] == 1
    assert report["serialized_key_renames"] == [{"serialized_key": serialized_name, "loaded_key": loaded_name}]


@pytest.mark.parametrize("prefix", ["model.vision_tower.", "vision.tower."])
def test_vision_wrapper_mapping_collision_fails(tmp_path: Path, prefix: str) -> None:
    reference = _vision_model(prefix)
    loaded_name = prefix + "projection.weight"
    value = reference.state_dict()[loaded_name]
    save_file(
        {loaded_name: value.clone(), prefix + "vision_model.projection.weight": value.clone()},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(ValueError, match="mapping collision"):
        VERIFIER["_check_weights"](copy.deepcopy(reference), reference, tmp_path, tied=False)


def test_vision_wrapper_is_not_removed_from_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        VERIFIER["_checkpoint_state_key"]("other.vision_model.projection.weight", {"other.projection.weight"})


def test_vision_wrapper_requires_existing_target_and_preserves_exact_names() -> None:
    name = "vision.tower.vision_model.projection.weight"
    assert VERIFIER["_checkpoint_state_key"](name, {name}) == name
    with pytest.raises(ValueError, match="Unexpected checkpoint tensor"):
        VERIFIER["_checkpoint_state_key"](name, {"vision.tower.unrelated.weight"})
