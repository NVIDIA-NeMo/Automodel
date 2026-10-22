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

"""M3 initialization through the registered VL factory and real Checkpointer.

AutoModel's unconditional CUDA device query is adapted to CPU, following
the DeepSeek-V4.1 loading tests. Text allocations are poisoned to expose a
skipped initializer. Construction, initialization and checkpoint loading stay
real. The unregistered standalone text wrapper uses its meta constructor and
the same infrastructure. VL forward coverage is text-only.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel._transformers.infrastructure import apply_model_infrastructure
from nemo_automodel.components.checkpoint import checkpointing
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig, MiniMaxM3VLTextConfig
from nemo_automodel.components.models.minimax_m3_vl.layers import MiniMaxM3RMSNorm
from nemo_automodel.components.models.minimax_m3_vl.model import (
    MiniMaxM3SparseForCausalLM,
    MiniMaxM3SparseForConditionalGeneration,
)
from nemo_automodel.components.moe.layers import Gate

from .conftest import IMAGE_TOKEN_INDEX, SPARSE_ATTENTION_CONFIG, TINY_CFG, VIDEO_TOKEN_INDEX, VISION_CONFIG

M3Model = MiniMaxM3SparseForCausalLM | MiniMaxM3SparseForConditionalGeneration


def _config(vlm: bool, mtp: bool = False) -> MiniMaxM3VLTextConfig | MiniMaxM3VLConfig:
    text = {
        **TINY_CFG,
        "torch_dtype": "float32",
        "num_mtp_modules": int(mtp),
        "sparse_attention_config": dict(SPARSE_ATTENTION_CONFIG),
    }
    if vlm:
        return MiniMaxM3VLConfig(
            text_config=text,
            architectures=["MiniMaxM3SparseForConditionalGeneration"],
            vision_config=dict(VISION_CONFIG),
            image_token_index=IMAGE_TOKEN_INDEX,
            video_token_index=VIDEO_TOKEN_INDEX,
            projector_hidden_size=TINY_CFG["hidden_size"],
        )
    return MiniMaxM3VLTextConfig(**text)


def _backend() -> BackendConfig:
    return BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=True,
    )


def _build_model(
    config: MiniMaxM3VLTextConfig | MiniMaxM3VLConfig, cache_dir: Path, checkpoint: Path | None = None
) -> M3Model:
    # Only the VL wrapper is registered with AutoModel. Test that real public
    # route, and use the standalone text constructor with the same infrastructure.
    if isinstance(config, MiniMaxM3VLConfig):
        return NeMoAutoModelForImageTextToText.from_config(
            str(checkpoint) if checkpoint is not None else config,
            load_base_model=checkpoint is not None,
            backend=_backend(),
            dtype=torch.float32,
            attn_implementation="sdpa",
            force_hf=False,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            local_files_only=True,
            cache_dir=str(cache_dir),
        )
    with torch.device("meta"):
        model = MiniMaxM3SparseForCausalLM(config, backend=_backend())
    return apply_model_infrastructure(
        model=model,
        is_meta_device=True,
        device=torch.device("cpu"),
        load_base_model=checkpoint is not None,
        pretrained_model_name_or_path=str(checkpoint) if checkpoint is not None else "",
        cache_dir=str(cache_dir),
    )


def _assert_text_initialized(model: M3Model) -> None:
    for name, parameter in model.named_parameters():
        if name.startswith("vision_tower."):
            continue
        assert not parameter.is_meta, name
        assert torch.isfinite(parameter).all(), name
        if parameter.ndim >= 2:
            assert parameter.float().std() > 0, name

    norms = [module for module in model.model.modules() if isinstance(module, MiniMaxM3RMSNorm)]
    assert norms
    for norm in norms:
        torch.testing.assert_close(norm.weight, torch.zeros_like(norm.weight), rtol=0, atol=0)

    gates = [module for module in model.model.modules() if isinstance(module, Gate)]
    assert gates
    for gate in gates:
        assert gate.weight.dtype == torch.float32
        assert gate.e_score_correction_bias.dtype == torch.float32
        torch.testing.assert_close(
            gate.e_score_correction_bias, torch.zeros_like(gate.e_score_correction_bias), rtol=0, atol=0
        )
    _, inv_freq = model.model.rotary_emb._compute_concentration_and_inv_freq()
    assert inv_freq.dtype == torch.float32
    assert torch.isfinite(inv_freq).all()


@pytest.mark.runtime_budget(30, reason="First real CPU model forward/backward incurs TorchInductor compilation.")
@pytest.mark.parametrize("vlm,mtp", [(False, False), (True, False), (False, True)], ids=["text", "vl_text", "mtp"])
def test_automodel_random_initialization(tmp_path: Path, vlm: bool, mtp: bool) -> None:
    """Materialized text parameters must be overwritten before the first training step."""
    materialize = checkpointing.to_empty_parameters_only

    def materialize_and_poison_text(model: M3Model, *, device: torch.device) -> None:
        materialize(model, device=device)
        # Deterministically expose a skipped initializer instead of depending on
        # whatever values the empty allocator happens to return.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not name.startswith("vision_tower."):
                    parameter.fill_(float("nan"))

    with (
        # Build on CPU even on GPU hosts: the model formats ``cuda:{current_device()}`` only
        # when CUDA is available, while from_config uses current_device() as the device.
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
        patch.object(
            checkpointing, "to_empty_parameters_only", side_effect=materialize_and_poison_text
        ) as materialized,
    ):
        model = _build_model(_config(vlm, mtp), tmp_path)
    assert isinstance(model, MiniMaxM3SparseForConditionalGeneration if vlm else MiniMaxM3SparseForCausalLM)
    materialized.assert_called_once()
    _assert_text_initialized(model)

    model.train()
    # One sequence spans four blocks with top-k two; selection is genuinely sparse.
    input_ids = torch.arange(16).unsqueeze(0)
    output = model(input_ids)
    logits = output.logits if mtp else output
    assert torch.isfinite(logits).all()
    loss = F.cross_entropy(logits.flatten(0, 1), input_ids.roll(-1, dims=1).flatten())
    if mtp:
        assert len(output.mtp_per_depth_logits) == 1
        for prediction in output.mtp_per_depth_logits:
            assert torch.isfinite(prediction).all()
            loss = loss + prediction.square().mean()
    loss.backward()
    assert model.model.embed_tokens.weight.grad is not None
    assert model.lm_head.weight.grad is not None
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all(), name


def test_checkpoint_overwrites_random_initialization(tmp_path: Path) -> None:
    """A standalone-text SafeTensors checkpoint replaces all initialized state."""
    config = _config(False)
    source = MiniMaxM3SparseForCausalLM(config, backend=_backend()).to(torch.float32)
    # Independent, non-random native values make post-load reinitialization visible.
    expected = {
        name: ((torch.arange(value.numel()).reshape(value.shape) % 17).float() / 1000 + 0.0123).to(value.dtype)
        for name, value in source.state_dict().items()
    }
    exported = source.state_dict_adapter.to_hf({name: value.clone() for name, value in expected.items()})
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config.save_pretrained(checkpoint)
    save_file({name: value.contiguous().clone() for name, value in exported.items()}, checkpoint / "model.safetensors")

    events: list[str] = []
    initialized_embeddings: list[torch.Tensor] = []
    initialize = Checkpointer.initialize_model_weights
    load = Checkpointer.load_base_model

    def record_initialize(model: M3Model, device: torch.device, peft_init_method: str | None = None) -> None:
        events.append("initialize")
        initialize(model, device, peft_init_method)
        _assert_text_initialized(model)
        initialized_embeddings.append(model.model.embed_tokens.weight.detach().clone())

    def record_load(
        checkpointer: Checkpointer,
        model: M3Model,
        device: torch.device,
        root_dir: str,
        model_name: str | None,
        load_base_model: bool = True,
    ) -> None:
        events.append("load")
        load(checkpointer, model, device, root_dir, model_name, load_base_model)

    with (
        # Build on CPU even on GPU hosts: the model formats ``cuda:{current_device()}`` only
        # when CUDA is available, while from_config uses current_device() as the device.
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda, "current_device", return_value=torch.device("cpu")),
        patch.object(Checkpointer, "initialize_model_weights", side_effect=record_initialize),
        patch.object(Checkpointer, "load_base_model", autospec=True, side_effect=record_load),
    ):
        restored = _build_model(config, tmp_path / "cache", checkpoint)

    assert events == ["initialize", "load"]
    assert len(initialized_embeddings) == 1
    assert not torch.equal(initialized_embeddings[0], expected["model.embed_tokens.weight"])
    actual = restored.state_dict()
    assert actual.keys() == expected.keys()
    for name, reference in expected.items():
        assert actual[name].dtype == reference.dtype, name
        torch.testing.assert_close(actual[name], reference, rtol=0, atol=0, msg=name)
