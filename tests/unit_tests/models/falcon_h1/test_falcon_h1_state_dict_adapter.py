# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

"""State-dict adapter tests for Falcon-H1.

The adapter's contract: translate the *complete* HF checkpoint key set to the
local module's key set and back, losing nothing. The headline structural risk
for this port is that the repo's StateDictAdapter base declares three abstract
methods (to_hf, from_hf, convert_single_tensor_to_hf) while the submitted
subclass implements only two — making it impossible to instantiate. The first
test below isolates that as a single actionable finding; the rest exercise the
round-trip once the adapter is concrete.
"""

import pytest

torch = pytest.importorskip("torch")

from .conftest import requires_falcon_h1, make_adapter

from nemo_automodel.components.models.falcon_h1.model import FalconH1ForCausalLM  # noqa: E402


# --------------------------------------------------------------------------- #
# Structural contract
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_adapter_is_concrete():
    """The adapter must implement every abstract method of the base class.

    Regression guard for the submitted adapter omitting
    ``convert_single_tensor_to_hf`` (required by StateDictAdapter), which makes
    the class abstract and uninstantiable.
    """
    from nemo_automodel.components.models.falcon_h1.state_dict_adapter import (
        FalconH1StateDictAdapter,
    )

    missing = sorted(getattr(FalconH1StateDictAdapter, "__abstractmethods__", frozenset()))
    assert missing == [], f"adapter is abstract; unimplemented methods: {missing}"


@requires_falcon_h1
def test_convert_single_tensor_to_hf_final_norm(tiny_config):
    """The per-tensor hook must apply the final-norm rename and return 1 tuple."""
    adapter = make_adapter(tiny_config)
    out = adapter.convert_single_tensor_to_hf("model.norm.weight", torch.zeros(4))
    assert isinstance(out, list) and len(out) == 1
    (fqn, _t), = out
    assert fqn == "model.final_layernorm.weight"


@requires_falcon_h1
def test_convert_single_tensor_to_hf_passthrough(tiny_config):
    """Non-renamed tensors pass through fqn-unchanged as a single tuple."""
    adapter = make_adapter(tiny_config)
    out = adapter.convert_single_tensor_to_hf("model.layers.0.mamba.in_proj.weight", torch.zeros(2, 2))
    assert isinstance(out, list) and len(out) == 1
    assert out[0][0] == "model.layers.0.mamba.in_proj.weight"


# --------------------------------------------------------------------------- #
# Round-trip
# --------------------------------------------------------------------------- #
@requires_falcon_h1
def test_from_hf_to_hf_inverse(tiny_config):
    """from_hf then to_hf is the identity on keys and values."""
    model = FalconH1ForCausalLM(tiny_config)
    adapter = make_adapter(tiny_config)

    hf = adapter.to_hf(model.state_dict())
    local = adapter.from_hf(hf)
    hf2 = adapter.to_hf(local)

    assert set(hf2.keys()) == set(hf.keys())
    for k in hf:
        assert torch.equal(hf2[k], hf[k]), f"value drift at {k}"


@requires_falcon_h1
def test_local_keys_match_module(tiny_config):
    """from_hf output keys must exactly equal the module's own state_dict keys."""
    model = FalconH1ForCausalLM(tiny_config)
    adapter = make_adapter(tiny_config)

    module_keys = set(model.state_dict().keys())
    hf = adapter.to_hf(model.state_dict())
    local = adapter.from_hf(hf)

    assert set(local.keys()) == module_keys, (
        f"missing={module_keys - set(local)} orphan={set(local) - module_keys}"
    )


@requires_falcon_h1
def test_strict_load_succeeds(tiny_config):
    """The from_hf output loads cleanly (the real integration point)."""
    model = FalconH1ForCausalLM(tiny_config)
    adapter = make_adapter(tiny_config)
    hf = adapter.to_hf(model.state_dict())

    fresh = FalconH1ForCausalLM(tiny_config)
    missing, unexpected = fresh.load_state_dict(adapter.from_hf(hf), strict=False)
    assert missing == [], f"missing keys on load: {missing}"
    assert unexpected == [], f"unexpected keys on load: {unexpected}"


@requires_falcon_h1
def test_no_final_layernorm_leaks_into_local(tiny_config):
    """The HF-only name must never appear in local keys after from_hf."""
    model = FalconH1ForCausalLM(tiny_config)
    adapter = make_adapter(tiny_config)
    hf = adapter.to_hf(model.state_dict())
    local = adapter.from_hf(hf)
    assert not any("final_layernorm" in k for k in local)
    assert any(k == "model.norm.weight" for k in local)


@requires_falcon_h1
def test_gated_norm_keys_roundtrip(tiny_config):
    """When mamba_rms_norm=True, mamba.norm.weight survives the round-trip.

    xfail(strict): the submitted mixer drops the gated RMSNorm, so the module
    has no mamba.norm.weight to round-trip. Flips to a pass once implemented.
    """
    if not tiny_config.mamba_rms_norm:
        pytest.skip("only relevant for the gated-norm config variant")

    model = FalconH1ForCausalLM(tiny_config)
    if not any(k.endswith("mamba.norm.weight") for k in model.state_dict()):
        pytest.xfail("gated RMSNorm not implemented (mamba_rms_norm path missing)")

    adapter = make_adapter(tiny_config)
    local = model.state_dict()
    restored = adapter.from_hf(adapter.to_hf(local))
    n_local = sum(k.endswith("mamba.norm.weight") for k in local)
    n_restored = sum(k.endswith("mamba.norm.weight") for k in restored)
    assert n_local == n_restored == tiny_config.num_hidden_layers


@requires_falcon_h1
def test_tied_embedding_adapter(tied_config):
    """Tied config: no standalone lm_head.weight in HF export.

    xfail(strict): the submitted model never ties lm_head to embeddings, so it
    emits an independent lm_head.weight. Flips to a pass once tying is honored.
    """
    model = FalconH1ForCausalLM(tied_config)
    adapter = make_adapter(tied_config)
    hf = adapter.to_hf(model.state_dict())
    if "lm_head.weight" in hf:
        pytest.xfail("tie_word_embeddings not honored; lm_head.weight present in export")
    fresh = FalconH1ForCausalLM(tied_config)
    _missing, unexpected = fresh.load_state_dict(adapter.from_hf(hf), strict=False)
    assert unexpected == [], f"unexpected keys: {unexpected}"


@requires_falcon_h1
def test_in_proj_weight_preserved(tiny_config):
    """The fused mamba in_proj weight passes through untouched (shape+value)."""
    model = FalconH1ForCausalLM(tiny_config)
    adapter = make_adapter(tiny_config)
    local = model.state_dict()
    key = "model.layers.0.mamba.in_proj.weight"
    assert key in local
    restored = adapter.from_hf(adapter.to_hf(local))
    assert restored[key].shape == local[key].shape
    assert torch.equal(restored[key], local[key])
