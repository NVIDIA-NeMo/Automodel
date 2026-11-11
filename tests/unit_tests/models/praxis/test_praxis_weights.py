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

"""CPU checks of the Praxis checkpoint files around the model: eager loading, the meta build and the metadata exporter."""

import json

import torch
from safetensors.torch import save_file

from nemo_automodel.components.models.praxis.backbone import MoeTrainingSettings
from nemo_automodel.components.models.praxis.config import PraxisConfig
from nemo_automodel.components.models.praxis.hub import load_checkpoint_tensors, resolve_repo
from nemo_automodel.components.models.praxis.model import PraxisPolicy
from tests.unit_tests.models.praxis.test_praxis_model import TINY_CONFIG, cpu_backend


def _write_base_checkpoint(root):
    """Write a one-shard checkpoint of a randomly initialized tiny model under ``root / "base"``; return the module state."""
    torch.manual_seed(0)
    model = PraxisPolicy(PraxisConfig.from_json(TINY_CONFIG), cpu_backend())
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_()
    base = root / "base"
    base.mkdir(parents=True)
    (base / "config.json").write_text(json.dumps(TINY_CONFIG))
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    shard = "model-00001-of-00001.safetensors"
    checkpoint_state = {k: v.contiguous() for k, v in model.state_dict_adapter.to_hf(state).items()}
    save_file(checkpoint_state, str(base / shard), metadata={"format": "pt"})
    index = {"metadata": {}, "weight_map": {k: shard for k in checkpoint_state}}
    (base / "model.safetensors.index.json").write_text(json.dumps(index))
    return state


def _materialized(root, moe_training=None):
    policy = PraxisPolicy.from_pretrained(
        root / "base", backend=cpu_backend(), device="meta", moe_training=moe_training
    )
    assert all(p.device.type == "meta" for p in policy.parameters())
    assert policy.pretrained_path == root / "base"
    policy.to_empty(device="cpu")
    return policy


def _assert_state_equal(policy, state):
    actual = policy.state_dict()
    for key, value in state.items():
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0, msg=key)


def test_eager_load_assigns_every_checkpoint_tensor(tmp_path):
    state = _write_base_checkpoint(tmp_path)
    eager = PraxisPolicy.from_pretrained(tmp_path / "base", backend=cpu_backend())
    _assert_state_equal(eager, state)
    assert eager.pretrained_path == tmp_path / "base"
    assert eager.name_or_path == str(tmp_path / "base")
    assert set(load_checkpoint_tensors(tmp_path / "base")) == set(eager.state_dict_adapter.to_hf(state))


def test_meta_build_leaves_the_weights_to_the_checkpointer(tmp_path):
    _write_base_checkpoint(tmp_path)
    policy = _materialized(tmp_path)
    assert all(p.device.type == "cpu" for p in policy.parameters())
    assert policy.name_or_path == str(tmp_path / "base")


def test_aux_loss_weight_is_shared_over_the_moe_layers(tmp_path):
    _write_base_checkpoint(tmp_path)
    policy = _materialized(tmp_path, MoeTrainingSettings(gate_bias_update_factor=1e-3, aux_loss_coeff=3e-4))
    layers = policy.moe_layers()
    assert len(layers) == 3  # encoder depth 3 with one dense block, decoder depth 2 with one dense block
    assert all(abs(layer.gate.aux_loss_coeff - 1e-4) < 1e-12 for layer in layers)


def test_metadata_exporter_copies_the_checkpoint_config(tmp_path):
    _write_base_checkpoint(tmp_path)
    policy = _materialized(tmp_path)
    exporter = policy._get_consolidated_hf_metadata_exporter(tokenizer=None, original_model_path=None)
    exporter.validate(tokenizer=None, original_model_path=None)
    (tmp_path / "meta").mkdir()
    exporter.save(hf_metadata_dir=str(tmp_path / "meta"), tokenizer=None, original_model_path=None, v4_compatible=False)
    assert (tmp_path / "meta" / "config.json").read_bytes() == (tmp_path / "base" / "config.json").read_bytes()


def test_resolve_repo_keeps_local_paths(tmp_path):
    _write_base_checkpoint(tmp_path)
    assert resolve_repo(tmp_path / "base") == tmp_path / "base"
    config = tmp_path / "base" / "config.json"
    assert resolve_repo(str(config), filename="config.json") == config
