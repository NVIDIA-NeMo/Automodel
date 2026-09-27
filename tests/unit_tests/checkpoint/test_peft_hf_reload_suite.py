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

"""Every supported family's PEFT export must load into the real Hugging Face consumer.

Export failures keep recurring because the outer ``base_model.model.`` prefix, the
model-owned renames, the fused expert tensor layout, and the ``target_modules``
metadata are produced in separate places. A save/reload round trip inside AutoModel
can agree with itself while the exported artifact fails in PEFT, so these tests save
through the production checkpoint path and reload with ``PeftModel.from_pretrained``.

Each family is one entry in ``_FAMILIES``: a tiny Transformers model plus the
AutoModel state-dict adapter that owns its naming. The model comes from Transformers
rather than the native AutoModel class because several native MoE classes build their
rope buffers on ``torch.cuda.current_device()`` and cannot be constructed on CPU; the
adapter is the component whose conversion these tests cover either way.

Families with a known, filed export defect stay in the list marked ``xfail`` so the
gap is visible and closing the bug turns the test green instead of leaving it
uncovered.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from torch import nn

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig

# Over the default 5s budget on purpose: each case builds two models, saves through the
# real checkpoint path, and reloads with PEFT. Trim the family list before raising this.
pytestmark = pytest.mark.timeout(120)

_BACKEND = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", rope_fusion=False)
_INPUT_IDS = torch.tensor([[1, 2, 3, 4]])


def _moe_config(*, dim: int, moe_inter_dim: int, n_routed_experts: int, gated: bool) -> MoEConfig:
    """Minimal MoE description for the adapters that convert fused expert tensors."""
    return MoEConfig(
        dim=dim,
        inter_dim=moe_inter_dim * 2,
        moe_inter_dim=moe_inter_dim,
        n_routed_experts=n_routed_experts,
        n_shared_experts=0,
        n_activated_experts=1,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=False,
        gate_bias_update_factor=0.0,
        score_func="softmax",
        route_scale=1.0,
        aux_loss_coeff=0.0,
        norm_topk_prob=False,
        expert_activation="swiglu" if gated else "relu2",
        dtype=torch.float32,
    )


# --------------------------------------------------------------------------- families


def _build_llama():
    """Dense control: no state-dict adapter, so only the shared boundary names the tensors."""
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    config = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return LlamaForCausalLM(config), None


def _build_nemotron_v3():
    """Mamba + MoE: renames the backbone namespace and owns a fused expert layout."""
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHForCausalLM

    from nemo_automodel.components.models.nemotron_v3.state_dict_adapter import NemotronV3StateDictAdapter

    config = NemotronHConfig(
        vocab_size=32,
        hidden_size=16,
        layers_block_type=["moe"],
        num_hidden_layers=1,
        n_routed_experts=2,
        moe_intermediate_size=12,
        moe_shared_expert_intermediate_size=12,
        moe_latent_size=None,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        use_mamba_kernels=False,
    )
    config._attn_implementation = "sdpa"
    moe_config = _moe_config(dim=16, moe_inter_dim=12, n_routed_experts=2, gated=False)
    return NemotronHForCausalLM(config), NemotronV3StateDictAdapter(config, moe_config, _BACKEND)


def _build_qwen3_moe():
    """Fused gated experts under ``mlp.experts``."""
    from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeForCausalLM

    from nemo_automodel.components.models.qwen3_moe.state_dict_adapter import Qwen3MoeStateDictAdapter

    config = Qwen3MoeConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        norm_topk_prob=False,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    moe_config = _moe_config(dim=16, moe_inter_dim=8, n_routed_experts=2, gated=True)
    return Qwen3MoeForCausalLM(config), Qwen3MoeStateDictAdapter(config, moe_config, _BACKEND)


def _build_minimax_m2():
    """Fused gated experts whose hidden size equals the fused input width."""
    from transformers.models.minimax_m2.configuration_minimax_m2 import MiniMaxM2Config
    from transformers.models.minimax_m2.modeling_minimax_m2 import MiniMaxM2ForCausalLM

    from nemo_automodel.components.models.minimax_m2.state_dict_adapter import MiniMaxM2StateDictAdapter

    config = MiniMaxM2Config(
        vocab_size=32,
        num_local_experts=2,
        hidden_size=16,
        intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        num_experts_per_tok=1,
        hidden_act="silu",
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    moe_config = _moe_config(dim=16, moe_inter_dim=8, n_routed_experts=2, gated=True)
    return MiniMaxM2ForCausalLM(config), MiniMaxM2StateDictAdapter(config, moe_config, _BACKEND)


def _build_qwen3_omni_moe():
    """Multimodal: the adapter names tensors for the whole Omni model.

    The thinker builds on its own, but its modules are then named without the
    ``thinker.`` segment the adapter emits, so the reload target has to be the full
    model. ``spatial_merge_size`` and ``shared_expert_intermediate_size`` have no
    defaults on the talker configs and are read during construction.
    """
    from transformers.models.qwen3_omni_moe import Qwen3OmniMoeConfig
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
        Qwen3OmniMoeThinkerForConditionalGeneration,
    )

    from nemo_automodel.components.models.qwen3_omni_moe.state_dict_adapter import Qwen3OmniMoeStateDictAdapter

    thinker_config = dict(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            moe_intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_experts=2,
            num_experts_per_tok=1,
            max_position_embeddings=64,
        ),
        vision_config=dict(
            depth=1,
            hidden_size=32,
            num_heads=4,
            out_hidden_size=32,
            intermediate_size=64,
            patch_size=14,
            spatial_merge_size=2,
            temporal_patch_size=2,
        ),
        audio_config=dict(
            d_model=32,
            encoder_layers=1,
            encoder_attention_heads=4,
            encoder_ffn_dim=64,
            num_mel_bins=128,
            output_dim=32,
            max_source_positions=64,
        ),
    )
    config = Qwen3OmniMoeConfig(
        thinker_config=thinker_config,
        talker_config=dict(
            spatial_merge_size=2,
            text_config=dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
                shared_expert_intermediate_size=32,
            ),
            code_predictor_config=dict(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
            ),
        ),
        code2wav_config=dict(hidden_size=32, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64),
    )
    config._attn_implementation = "sdpa"
    moe_config = _moe_config(dim=32, moe_inter_dim=16, n_routed_experts=2, gated=True)
    thinker = Qwen3OmniMoeThinkerForConditionalGeneration(config.thinker_config)
    return thinker, Qwen3OmniMoeStateDictAdapter(config, moe_config, _BACKEND)


def _build_qwen3_omni_moe_reference():
    """The reload target for the Omni adapter: the full model the exported names describe."""
    from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration

    _, adapter = _build_qwen3_omni_moe()
    return Qwen3OmniMoeForConditionalGeneration(adapter.config)


def _build_qwen3_omni_moe_standalone_thinker():
    """The same adapter against a thinker-only base, which carries no ``thinker.`` segment.

    ``from_hf`` records the base checkpoint's layout, and both tensor exporters drop the
    namespace when it is absent, so target_modules has to drop it too. Getting that wrong
    is invisible to a self-consistency check and only shows up on a real PEFT reload.
    """
    thinker, adapter = _build_qwen3_omni_moe()

    # Production reaches the adapter through the checkpoint load, which is where the
    # layout is recorded. Checkpoints store experts split per index; the model fuses
    # them on load, so the detector only ever sees the split form.
    thinkerless = {}
    for expert in range(2):
        prefix = f"model.layers.0.mlp.experts.{expert}"
        thinkerless[f"{prefix}.gate_proj.weight"] = torch.zeros(16, 32)
        thinkerless[f"{prefix}.up_proj.weight"] = torch.zeros(16, 32)
        thinkerless[f"{prefix}.down_proj.weight"] = torch.zeros(32, 16)
    adapter.from_hf(thinkerless)
    assert adapter._uses_thinker_prefix is False, "the thinker-only layout was not detected"

    return thinker, adapter


@dataclass(frozen=True)
class _Family:
    """One covered model family.

    Attributes:
        id: pytest parameter id.
        build: Returns ``(tiny Transformers model, adapter or None)``. The adapter is the
            AutoModel component that owns this family's HF naming.
        peft_kwargs: Overrides for the ``PeftConfig`` the export is driven with.
        build_reference: The model the export is meant to load into, when that is not the
            same class as the source. AutoModel trains Omni's thinker on its own and the
            adapter adds the ``thinker.`` segment, so the artifact targets the full model.
        reference_path: Attribute on the reference that corresponds to the source model.
        xfail: Issue reference when this family has a known, filed export defect.
    """

    id: str
    build: Callable[[], tuple[nn.Module, object | None]]
    peft_kwargs: dict = field(default_factory=dict)
    build_reference: Callable[[], nn.Module] | None = None
    reference_path: str | None = None
    xfail: str | None = None


_FAMILIES = (
    _Family("llama_dense", _build_llama, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    _Family("nemotron_v3", _build_nemotron_v3, {"exclude_modules": ["*.out_proj"]}),
    _Family("qwen3_moe", _build_qwen3_moe, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    _Family("minimax_m2", _build_minimax_m2, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    # Scoped to the thinker: the talker and code2wav towers add adapters the thinker's
    # naming rules do not describe, which is a separate contract from the one under test.
    _Family(
        "qwen3_omni_moe",
        _build_qwen3_omni_moe,
        {"target_modules": ["*.q_proj", "*.v_proj"]},
        build_reference=_build_qwen3_omni_moe_reference,
        reference_path="thinker",
    ),
    # Same adapter, thinker-only base: the namespace the full layout requires is exactly
    # the one this layout must not have, so one hook has to serve both.
    _Family(
        "qwen3_omni_moe_standalone_thinker",
        _build_qwen3_omni_moe_standalone_thinker,
        {"target_modules": ["*.q_proj", "*.v_proj"]},
    ),
)


def _params():
    """Declarative xfail, so a family whose defect gets fixed reports XPASS and fails.

    An imperative ``pytest.xfail()`` aborts before the body runs, which would let a
    fixed export silently keep its marker and go uncovered.
    """
    return [
        pytest.param(family, id=family.id, marks=pytest.mark.xfail(reason=family.xfail, strict=True))
        if family.xfail
        else pytest.param(family, id=family.id)
        for family in _FAMILIES
    ]


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def peft_process_group(tmp_path: Path):
    """The checkpoint save path reduces across ranks, so it needs an initialized group."""
    torch.distributed.init_process_group("gloo", init_method=f"file://{tmp_path}/rendezvous", rank=0, world_size=1)
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()


def _adapted_model(family: _Family, source: Path):
    """Build the family's model, attach its adapter, and give it nonzero adapter weights.

    Zero-initialized adapters would make a dropped or mis-shaped tensor invisible in a
    forward comparison, so every LoRA weight is randomized before the export.
    """
    torch.manual_seed(1234)
    source_model, adapter = family.build()
    source_model = source_model.eval()
    source_model.save_pretrained(source)

    if family.build_reference is None:
        reference = family.build()[0].eval()
        reference.load_state_dict(source_model.state_dict())
    else:
        # The artifact targets a larger model that embeds this one; give the matching
        # submodule the same base weights so only the adapter can explain a difference.
        reference = family.build_reference().eval()
        getattr(reference, family.reference_path).load_state_dict(source_model.state_dict())

    model, _ = family.build()
    model = model.eval()
    model.load_state_dict(source_model.state_dict())
    if adapter is not None:
        model.state_dict_adapter = adapter

    peft_config = PeftConfig(dim=2, alpha=4, use_triton=False, **family.peft_kwargs)
    applied = apply_lora_to_linear_modules(model, peft_config)
    assert applied > 0, f"{family.id}: no LoRA modules were applied"
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.normal_(std=0.05)
    return model, reference, peft_config


def _save_through_checkpointer(model: nn.Module, peft_config: PeftConfig, tmp_path: Path) -> Path:
    """Write the adapter with the production checkpoint path, not a direct adapter call."""
    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_cache_dir=str(tmp_path),
            model_repo_id="source",
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.save_model(model, str(tmp_path / "peft"), peft_config=peft_config)
    return tmp_path / "peft" / "model"


# --------------------------------------------------------------------------- tests


@pytest.mark.parametrize("family", _params())
def test_exported_adapter_loads_into_hf_peft(family: _Family, tmp_path: Path, peft_process_group):
    """Save through the real path, reload with real PEFT, and require identical behavior."""
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    model, reference, peft_config = _adapted_model(family, tmp_path / "source")
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert exported, f"{family.id}: the export wrote no adapter tensors"
    assert len(set(exported)) == len(exported), f"{family.id}: duplicate keys in the export"

    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), key_mapping={}, autocast_adapter_dtype=False).eval()
    loaded_state = get_peft_model_state_dict(loaded, save_embedding_layers=False)
    assert set(loaded_state) == set(exported), (
        f"{family.id}: PEFT loaded a different tensor set than was exported; "
        f"missing={sorted(set(exported) - set(loaded_state))} extra={sorted(set(loaded_state) - set(exported))}"
    )
    for name, value in exported.items():
        torch.testing.assert_close(loaded_state[name], value, rtol=0, atol=0)

    # Compare the adapted submodule, not the wrapper: a larger reference model has its
    # own forward signature, and the contract under test is this model's behavior.
    adapted = loaded.base_model.model
    if family.reference_path is not None:
        adapted = getattr(adapted, family.reference_path)
    with torch.no_grad():
        expected = model(_INPUT_IDS, use_cache=False).logits
        actual = adapted(_INPUT_IDS, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", _params())
def test_bulk_and_per_tensor_exports_agree(family: _Family, tmp_path: Path, peft_process_group):
    """``to_hf`` and ``convert_single_tensor_to_hf`` feed different consumers; they must match.

    Streaming consumers convert one tensor at a time, which cannot see the sibling
    tensors a fused conversion needs, so the two paths drift apart silently.
    """
    model, _, _ = _adapted_model(family, tmp_path / "source")
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        pytest.skip(f"{family.id} has no state-dict adapter; the boundary names its tensors directly")

    native = ModelState(model, is_peft=True).state_dict()
    bulk = adapter.to_hf(dict(native))
    streamed = dict(
        item for name, tensor in native.items() for item in adapter.convert_single_tensor_to_hf(name, tensor)
    )

    assert set(streamed) == set(bulk), (
        f"{family.id}: per-tensor export disagrees on keys; "
        f"bulk_only={sorted(set(bulk) - set(streamed))} streamed_only={sorted(set(streamed) - set(bulk))}"
    )
    for name, value in bulk.items():
        torch.testing.assert_close(streamed[name], value, rtol=0, atol=0)
