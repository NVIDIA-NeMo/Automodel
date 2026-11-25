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

"""Config routing and portable checkpoint metadata for the native MTP entry."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.model_capabilities import query_capabilities
from nemo_automodel._transformers.model_init import _init_model, get_hf_config
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.ernie5_5.model import Ernie5_5_MoeForCausalLM
from nemo_automodel.components.models.ernie5_5.mtp_training import Ernie5_5MTPForCausalLM
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import backend
from tests.unit_tests.models.ernie5_5.test_mtp_training_local import config as tiny_config

pytestmark = pytest.mark.cpu_only


def _construct(config, **kwargs) -> tuple[bool, nn.Module]:
    with torch.device("meta"):
        return _init_model(
            NeMoAutoModelForCausalLM, config, "eager", torch.bfloat16, None, False, backend=backend(), **kwargs
        )


@pytest.mark.parametrize(
    "architecture,depth,mtp_enabled",
    [
        ("Ernie5_5_MoeForCausalLM", 0, False),
        ("Ernie5_5MTPForCausalLM", 1, True),
    ],
)
def test_native_config_dispatch_and_constructor_coefficient(architecture: str, depth: int, mtp_enabled: bool) -> None:
    """The existing builder's class resolution constructs the requested native variant."""
    cfg = tiny_config("bfloat16")
    cfg.architectures = [architecture]
    cfg.num_nextn_predict_layers = depth
    custom, result = _construct(cfg, **({"mtp_loss_scaling_factor": 0.25} if mtp_enabled else {}))
    assert custom and isinstance(result, Ernie5_5_MoeForCausalLM)
    assert isinstance(result, Ernie5_5MTPForCausalLM) == mtp_enabled
    assert cfg.architectures == [architecture] and cfg.num_nextn_predict_layers == depth
    assert result.config.architectures == ["Ernie5_5_MoeForCausalLM"]
    assert result.model.config.architectures == ["Ernie5_5_MoeForCausalLM"]
    owned = [name for name, _ in result.named_parameters() if name.startswith("mtp.")]
    assert len(owned) == (8 if mtp_enabled else 0)
    caps = query_capabilities(result)
    assert caps.supports_ep
    if mtp_enabled:
        assert result.mtp_config.loss_scaling_factor == 0.25
        assert not caps.supports_cp and not caps.supports_thd


@pytest.mark.parametrize("depth,shared", [(0, True), (2, True), (1, False)])
def test_native_mtp_entry_refuses_unsupported_depth_or_sharing(depth: int, shared: bool) -> None:
    """Explicit MTP entry selection still enforces the one-depth shared-body contract."""
    cfg = tiny_config()
    cfg.architectures = ["Ernie5_5MTPForCausalLM"]
    cfg.num_nextn_predict_layers = depth
    cfg.mtp_shared_last_layer = shared
    with pytest.raises(ValueError, match="one shared-last-layer MTP depth"):
        _construct(cfg)


def test_canonical_export_config_and_explicit_recipe_reload(tmp_path: Path) -> None:
    """Persist released-family metadata and resolve the MTP recipe override before construction."""
    cfg = tiny_config("bfloat16")
    cfg.architectures = ["Ernie5_5MTPForCausalLM"]
    _, result = _construct(cfg)
    result.config.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["architectures"] == ["Ernie5_5_MoeForCausalLM"]
    assert saved["num_nextn_predict_layers"] == 1 and saved["mtp_shared_last_layer"]
    restored = get_hf_config(
        str(tmp_path),
        "eager",
        trust_remote_code=False,
        local_files_only=True,
        config={"architectures": ["Ernie5_5MTPForCausalLM"]},
    )
    resolved = ModelRegistry.resolve_custom_model_cls(restored.architectures[0], restored)
    assert issubclass(resolved, Ernie5_5MTPForCausalLM)
    _, fresh = _construct(restored)
    assert isinstance(fresh, Ernie5_5MTPForCausalLM)
    assert fresh.config.architectures == ["Ernie5_5_MoeForCausalLM"]


def test_decoder_entry_error_names_decoder_and_mtp_choices() -> None:
    """An unchanged release config explains both supported loading choices."""
    cfg = tiny_config()
    cfg.architectures = ["Ernie5_5_MoeForCausalLM"]
    with pytest.raises(
        ValueError,
        match=r"MTP depth 0.*model\.num_nextn_predict_layers: 0.*model\.config\.architectures: \[Ernie5_5MTPForCausalLM\]",
    ):
        _construct(cfg)


def test_checkpointer_writes_canonical_config_for_shards_and_consolidation(tmp_path: Path) -> None:
    """Exercise the native writer's metadata rather than calling config.save_pretrained directly."""
    cfg = tiny_config("bfloat16")
    cfg.architectures = ["Ernie5_5MTPForCausalLM"]
    _, result = _construct(cfg)
    Checkpointer.initialize_model_weights(result, torch.device("cpu"))
    checkpointer = CheckpointingConfig(
        checkpoint_dir=str(tmp_path), model_save_format="safetensors", save_consolidated=True, is_async=False
    ).build(0, 0, 0)
    try:
        checkpointer.save_model(result, str(tmp_path), is_final_checkpoint=True)
    finally:
        checkpointer.close()
    for directory in [tmp_path / "model/.hf_metadata", tmp_path / "model/consolidated"]:
        text = (directory / "config.json").read_text()
        saved = json.loads(text)
        assert saved["architectures"] == ["Ernie5_5_MoeForCausalLM"]
        assert saved["num_nextn_predict_layers"] == 1 and saved["mtp_shared_last_layer"]
        assert "Ernie5_5MTPForCausalLM" not in text
    assert cfg.architectures == ["Ernie5_5MTPForCausalLM"]


@pytest.mark.parametrize("logits_dtype", [None, torch.float32])
def test_shared_main_and_mtp_head_precision_backward(logits_dtype: torch.dtype | None) -> None:
    """Both head uses retain the requested output dtype and propagate finite gradients."""
    candidate = Ernie5_5MTPForCausalLM(tiny_config("bfloat16"), backend(), logits_dtype=logits_dtype)
    candidate.initialize_weights()
    result = candidate(torch.arange(1, 9)[None])
    expected = torch.bfloat16 if logits_dtype is None else logits_dtype
    assert result.logits.dtype == result.mtp_per_depth_logits[0].dtype == expected
    loss = result.logits.float().square().mean() + result.mtp_per_depth_logits[0].float().square().mean()
    loss.backward()
    for parameter in [candidate.model.lm_head.weight, *candidate.mtp.parameters()]:
        assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())


def test_public_mtp_alias_load_preserves_owned_checkpoint_values(tmp_path: Path) -> None:
    """Native base loading must read the MTP weights rather than retain new-head initialization."""
    source = Ernie5_5MTPForCausalLM(tiny_config("bfloat16"), backend())
    source.initialize_weights()
    source.config.save_pretrained(tmp_path)
    exported = source.state_dict_adapter.to_hf(source.state_dict())
    save_file({name: value.contiguous().clone() for name, value in exported.items()}, tmp_path / "model.safetensors")
    loaded_config = get_hf_config(
        str(tmp_path),
        "eager",
        trust_remote_code=False,
        local_files_only=True,
        config={"architectures": ["Ernie5_5MTPForCausalLM"]},
    )
    original_load = Checkpointer.load_base_model
    fresh_owned = {}

    def load_after_fresh_initialization(
        owner: Checkpointer,
        model: nn.Module,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        load_base_model: bool = True,
    ) -> None:
        assert isinstance(model, Ernie5_5MTPForCausalLM)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(94017)
            model.initialize_weights()
        fresh_owned.update({name: value.detach().clone() for name, value in model.mtp.state_dict().items()})
        assert fresh_owned and any(
            not torch.equal(value, source.mtp.state_dict()[name]) for name, value in fresh_owned.items()
        ), "The control must differ before the native loader reads the checkpoint"
        original_load(owner, model, device, root_dir, model_name, load_base_model)

    with (
        patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
        patch.object(Checkpointer, "load_base_model", new=load_after_fresh_initialization),
    ):
        actual = NeMoAutoModelForCausalLM.from_config(
            loaded_config,
            load_base_model=True,
            backend=backend(),
            attn_implementation="eager",
            torch_dtype=torch.bfloat16,
            force_hf=False,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            cache_dir=str(tmp_path / "cache"),
        )
    assert fresh_owned, "The public path must call the native base loader"
    assert isinstance(actual, Ernie5_5MTPForCausalLM)
    actual_state, expected_state = actual.state_dict(), source.state_dict()
    assert actual_state.keys() == expected_state.keys()
    for name, expected in expected_state.items():
        torch.testing.assert_close(actual_state[name], expected, rtol=0, atol=0)
