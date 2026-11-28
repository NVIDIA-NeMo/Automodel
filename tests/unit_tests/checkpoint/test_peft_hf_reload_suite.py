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

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.checkpoint.stateful_wrappers import ModelState
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.utils.model_utils import apply_parameter_freezing

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


def _build_qwen2_5_omni():
    """Dense multimodal sibling of qwen3_omni_moe: same ``thinker.`` namespacing, no experts."""
    from transformers.models.qwen2_5_omni import (
        Qwen2_5OmniBigVGANConfig,
        Qwen2_5OmniConfig,
        Qwen2_5OmniDiTConfig,
        Qwen2_5OmniTalkerConfig,
        Qwen2_5OmniTextConfig,
        Qwen2_5OmniToken2WavConfig,
    )
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    from nemo_automodel.components.models.qwen2_5_omni.state_dict_adapter import Qwen2_5OmniStateDictAdapter

    thinker_config = dict(
        text_config=dict(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            # Multimodal rope splits the head dim three ways and the sections have to
            # sum to half of it; head_dim is 32/4 here.
            rope_scaling={"rope_type": "default", "mrope_section": [2, 1, 1]},
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
    config = Qwen2_5OmniConfig(
        thinker_config=thinker_config,
        # The talker's nested text_config is not coerced from a dict the way the
        # thinker's is, and a plain dict there fails when the full model builds its
        # generation config. It is only here so the reload target exists.
        talker_config=Qwen2_5OmniTalkerConfig(
            # The talker carries its own sizes as well as a nested text_config, and its
            # defaults are a 28-layer 3584-wide stack: 6.6B parameters if left alone.
            vocab_size=64,
            embedding_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=64,
            text_config=Qwen2_5OmniTextConfig(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            ),
        ),
        # Same coercion gap as the talker, and the defaults here are a 22-layer DiT and
        # a 1536-channel vocoder, which is far too much to build for a naming test.
        token2wav_config=Qwen2_5OmniToken2WavConfig(
            dit_config=Qwen2_5OmniDiTConfig(
                hidden_size=32,
                num_hidden_layers=1,
                num_attention_heads=4,
                emb_dim=32,
                head_dim=8,
                num_embeds=64,
                max_position_embeddings=64,
                mel_dim=8,
                enc_emb_dim=16,
                enc_dim=16,
                enc_attention_channels=8,
                enc_se_channels=8,
            ),
            bigvgan_config=Qwen2_5OmniBigVGANConfig(mel_dim=8, upsample_initial_channel=32),
        ),
    )
    config._attn_implementation = "sdpa"
    thinker = Qwen2_5OmniThinkerForConditionalGeneration(config.thinker_config)
    return thinker, Qwen2_5OmniStateDictAdapter(config, _BACKEND, dtype=torch.float32)


def _build_qwen2_5_omni_reference():
    """The reload target: the full model whose layout the exported names describe."""
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

    _, adapter = _build_qwen2_5_omni()
    return Qwen2_5OmniForConditionalGeneration(adapter.config)


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
        legacy_namespace: Module prefix the ``v4_compatible`` export targets, when that is
            a Transformers v4 layout this environment cannot build. The legacy artifact is
            then checked for internal consistency instead of reloaded -- see
            ``test_legacy_export_names_one_namespace``.
        full_modules: Modules trained in full next to LoRA (``freeze_config.unfreeze_modules``)
            in the tests that cover ``modules_to_save``: the input embedding, the final norm,
            and the LM head, named as the Transformers class names them.
        layout_is_native: Whether the Transformers class names ``full_modules`` the way the
            native AutoModel model does. Loading an export back runs the family adapter's
            ``from_hf``, which lands on native names, so a resume into the Transformers class
            is only meaningful when the two agree.
        xfail: Issue reference when this family has a known, filed export defect.
    """

    id: str
    build: Callable[[], tuple[nn.Module, object | None]]
    peft_kwargs: dict = field(default_factory=dict)
    build_reference: Callable[[], nn.Module] | None = None
    reference_path: str | None = None
    legacy_namespace: str | None = None
    full_modules: tuple[str, ...] = ("model.embed_tokens", "model.norm", "lm_head")
    layout_is_native: bool = True
    xfail: str | None = None


_FAMILIES = (
    _Family("llama_dense", _build_llama, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    # v4 keeps the checkpoint's ``backbone.`` namespace, which is the Transformers v4
    # NemotronH layout; v5 renames it to ``model.``. Only the v5 target is buildable here.
    _Family(
        "nemotron_v3",
        _build_nemotron_v3,
        {"exclude_modules": ["*.out_proj"]},
        legacy_namespace="backbone.",
        # lm_head is a LoRA target here (every linear but out_proj). The native model
        # calls these model.embed_tokens and model.norm; the adapter renames them on
        # export, which tests/unit_tests/models/nemotron_v3 covers.
        full_modules=("model.embeddings", "model.norm_f"),
        layout_is_native=False,
    ),
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
        full_modules=("model.language_model.embed_tokens", "model.language_model.norm", "lm_head"),
    ),
    _Family(
        "qwen2_5_omni",
        _build_qwen2_5_omni,
        {"target_modules": ["*.q_proj", "*.v_proj"]},
        build_reference=_build_qwen2_5_omni_reference,
        reference_path="thinker",
        full_modules=("model.language_model.embed_tokens", "model.language_model.norm", "lm_head"),
    ),
    # Same adapter, thinker-only base: the namespace the full layout requires is exactly
    # the one this layout must not have, so one hook has to serve both.
    _Family(
        "qwen3_omni_moe_standalone_thinker",
        _build_qwen3_omni_moe_standalone_thinker,
        {"target_modules": ["*.q_proj", "*.v_proj"]},
        full_modules=("model.language_model.embed_tokens", "model.language_model.norm", "lm_head"),
    ),
)


def _params(families: tuple[_Family, ...] = _FAMILIES):
    """Declarative xfail, so a family whose defect gets fixed reports XPASS and fails.

    An imperative ``pytest.xfail()`` aborts before the body runs, which would let a
    fixed export silently keep its marker and go uncovered.
    """
    return [
        pytest.param(family, id=family.id, marks=pytest.mark.xfail(reason=family.xfail, strict=True))
        if family.xfail
        else pytest.param(family, id=family.id)
        for family in families
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


def _save_through_checkpointer(
    model: nn.Module, peft_config: PeftConfig, tmp_path: Path, *, v4_compatible: bool = False
) -> Path:
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
            v4_compatible=v4_compatible,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.save_model(model, str(tmp_path / "peft"), peft_config=peft_config)
    return tmp_path / "peft" / "model"


def _fmt(v4_compatible: bool) -> str:
    """Name the export format for assertion messages."""
    return "v4" if v4_compatible else "v5"


def _module_of(tensor_key: str) -> str:
    """The module an exported adapter tensor belongs to, as target_modules names it."""
    return tensor_key.removeprefix("base_model.model.").rsplit(".lora_", 1)[0]


# --------------------------------------------------------------------------- tests


# Both export formats are supported, so both have to be checked against the real
# consumer. They are the same artifact for most families; nemotron_v3 is the one that
# genuinely differs, because v4 keeps the checkpoint's ``backbone.`` namespace where v5
# uses ``model.`` -- in the tensor keys and in target_modules alike.
_FORMATS = [pytest.param(False, id="v5"), pytest.param(True, id="v4")]


@pytest.mark.parametrize("adapter_dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("use_dora", [False, True], ids=["lora", "dora"])
def test_adapter_storage_restoration_and_bf16_inference(
    adapter_dtype: torch.dtype, use_dora: bool, tmp_path: Path, peft_process_group
):
    """Preserve saved FP32/BF16 LoRA and DoRA values across training and inference reloads."""
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    family = _Family(
        "llama",
        _build_llama,
        {"target_modules": ["*.q_proj", "*.v_proj"], "use_dora": use_dora, "lora_dtype": adapter_dtype},
    )
    model, reference, peft_config = _adapted_model(family, tmp_path / "source")
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)
    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert exported and all(value.dtype == adapter_dtype for value in exported.values())
    source_state = ModelState(model, is_peft=True).state_dict()
    assert set(exported) == set(source_state)
    for name, value in exported.items():
        torch.testing.assert_close(value, source_state[name], rtol=0, atol=0)

    restored = PeftModel.from_pretrained(
        reference.bfloat16(), adapter_dir, is_trainable=True, autocast_adapter_dtype=True
    )
    restored_state = get_peft_model_state_dict(restored, save_embedding_layers=False)
    assert set(restored_state) == set(exported)
    for name, value in exported.items():
        torch.testing.assert_close(restored_state[name], value.float(), rtol=0, atol=0)
    restored(_INPUT_IDS).logits.float().square().mean().backward()
    for parameter in restored.parameters():
        if parameter.requires_grad:
            assert parameter.dtype == parameter.grad.dtype == torch.float32
            assert torch.isfinite(parameter.grad).all()

    inference_base, _ = _build_llama()
    inference_model = PeftModel.from_pretrained(
        inference_base.bfloat16(), adapter_dir, autocast_adapter_dtype=False
    ).eval()
    inference_state = get_peft_model_state_dict(inference_model, save_embedding_layers=False)
    assert set(inference_state) == set(exported)
    for name, value in exported.items():
        torch.testing.assert_close(inference_state[name], value.bfloat16(), rtol=0, atol=0)
    with torch.inference_mode():
        logits = inference_model(_INPUT_IDS).logits
    assert logits.dtype == torch.bfloat16 and torch.isfinite(logits).all()


@pytest.mark.parametrize("v4_compatible", _FORMATS)
@pytest.mark.parametrize("family", _params())
def test_exported_adapter_loads_into_hf_peft(family: _Family, v4_compatible: bool, tmp_path: Path, peft_process_group):
    """Save through the real path, reload with real PEFT, and require identical behavior."""
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file

    if v4_compatible and family.legacy_namespace is not None:
        pytest.skip(
            f"{family.id} v4 targets the {family.legacy_namespace!r} layout, which this "
            f"Transformers cannot build; test_legacy_export_names_one_namespace covers it"
        )

    model, reference, peft_config = _adapted_model(family, tmp_path / "source")
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path, v4_compatible=v4_compatible)

    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert exported, f"{family.id}/{_fmt(v4_compatible)}: the export wrote no adapter tensors"
    assert len(set(exported)) == len(exported), f"{family.id}/{_fmt(v4_compatible)}: duplicate keys in the export"

    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), key_mapping={}, autocast_adapter_dtype=False).eval()
    loaded_state = get_peft_model_state_dict(loaded, save_embedding_layers=False)
    assert set(loaded_state) == set(exported), (
        f"{family.id}/{_fmt(v4_compatible)}: PEFT loaded a different tensor set than was exported; "
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


@pytest.mark.parametrize("v4_compatible", _FORMATS)
@pytest.mark.parametrize("family", _params())
def test_bulk_and_per_tensor_exports_agree(family: _Family, v4_compatible: bool, tmp_path: Path, peft_process_group):
    """``to_hf`` and ``convert_single_tensor_to_hf`` feed different consumers; they must match.

    Streaming consumers convert one tensor at a time, which cannot see the sibling
    tensors a fused conversion needs, so the two paths drift apart silently.
    """
    model, _, _ = _adapted_model(family, tmp_path / "source")
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        pytest.skip(f"{family.id} has no state-dict adapter; the boundary names its tensors directly")

    native = ModelState(model, is_peft=True).state_dict()
    bulk = adapter.to_hf(dict(native), v4_compatible=v4_compatible)
    streamed = dict(
        item
        for name, tensor in native.items()
        for item in adapter.convert_single_tensor_to_hf(name, tensor, v4_compatible=v4_compatible)
    )

    assert set(streamed) == set(bulk), (
        f"{family.id}/{_fmt(v4_compatible)}: per-tensor export disagrees on keys; "
        f"bulk_only={sorted(set(bulk) - set(streamed))} streamed_only={sorted(set(streamed) - set(bulk))}"
    )
    for name, value in bulk.items():
        torch.testing.assert_close(streamed[name], value, rtol=0, atol=0)


@pytest.mark.parametrize("family", _params())
def test_legacy_export_names_one_namespace(family: _Family, tmp_path: Path, peft_process_group):
    """The legacy artifact has to name the same modules in its tensors and its metadata.

    PEFT resolves target_modules by suffix against the receiving model, then looks for
    tensors under the names that matched. An entry with no tensor leaves a randomly
    initialized adapter; a tensor with no entry is never loaded. Neither raises, which is
    what made #3944 silent.

    The v5 format gets this checked by the reload test above. The legacy format cannot:
    its target is a Transformers v4 model that this environment cannot build, so a reload
    would only prove the two versions disagree. The agreement between the two halves of
    the artifact is the part that is still ours to keep correct, and it is checkable here.
    """
    from safetensors.torch import load_file

    model, _, peft_config = _adapted_model(family, tmp_path / "source")
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path, v4_compatible=True)

    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert exported, f"{family.id}/v4: the export wrote no adapter tensors"

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    targets = set(config.get("target_modules") or [])
    assert targets, f"{family.id}/v4: the export declared no target_modules"

    modules = {_module_of(key) for key in exported}
    assert modules == targets, (
        f"{family.id}/v4: tensors and target_modules name different modules; "
        f"tensors_only={sorted(modules - targets)} targets_only={sorted(targets - modules)}"
    )


@pytest.mark.parametrize("family", _params())
def test_adapter_written_by_peft_loads_through_our_path(family: _Family, tmp_path: Path, peft_process_group):
    """An adapter this repo did not write has to load through the same boundary.

    Every other test here starts from our own export, so it can only prove the two
    directions agree with each other. A checkpoint produced by the PEFT trainer is the
    case that actually turns up on resume, and ``from_hf`` is what has to accept it.
    If our naming drifts from the ecosystem's, this is the test that notices.
    """
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file

    model, _, peft_config = _adapted_model(family, tmp_path / "source")
    adapter = getattr(model, "state_dict_adapter", None)
    if adapter is None:
        pytest.skip(f"{family.id} has no state-dict adapter; the boundary names its tensors directly")

    ours = ModelState(model, is_peft=True).state_dict()
    leaves = sorted({_module_of(key).rsplit(".", 1)[-1] for key in ours})

    # Same modules, adapted by PEFT itself and written by PEFT itself.
    theirs, _ = family.build()
    wrapped = get_peft_model(theirs.eval(), LoraConfig(r=2, lora_alpha=4, target_modules=leaves, lora_dropout=0.0))
    written_dir = tmp_path / "written-by-peft"
    wrapped.save_pretrained(str(written_dir))
    written_dir = written_dir / "default" if (written_dir / "default").exists() else written_dir
    written = load_file(str(written_dir / "adapter_model.safetensors"))

    # PEFT also stores the frozen base weight for modules it has to replace wholesale.
    # That is its own bookkeeping, not an adapter tensor, and no export produces it.
    written = {key: value for key, value in written.items() if ".base_layer." not in key}
    assert written, f"{family.id}: PEFT wrote no adapter tensors to load back"

    loaded = adapter.from_hf(dict(written))

    assert set(loaded) == set(ours), (
        f"{family.id}: loading a PEFT-written adapter lands on different names than our own "
        f"export uses; theirs_only={sorted(set(loaded) - set(ours))} "
        f"ours_only={sorted(set(ours) - set(loaded))}"
    )
    for name, value in written.items():
        torch.testing.assert_close(loaded[name], value, rtol=0, atol=0)

    if family.legacy_namespace is None:
        return

    # The harder half of "existing checkpoint": one saved while the model still used the
    # legacy namespace. from_hf carries the normalization for exactly this, and nothing
    # else here feeds it a legacy-namespaced adapter.
    legacy = {
        key.replace("base_model.model.model.", f"base_model.model.{family.legacy_namespace}", 1): value
        for key, value in written.items()
    }
    assert legacy != written, f"{family.id}: the legacy rewrite changed nothing, so this case is not being exercised"

    normalized = adapter.from_hf(dict(legacy))
    assert set(normalized) == set(ours), (
        f"{family.id}: a {family.legacy_namespace!r} adapter did not normalize to the native "
        f"namespace; legacy_only={sorted(set(normalized) - set(ours))} "
        f"ours_only={sorted(set(ours) - set(normalized))}"
    )


@pytest.mark.parametrize("family", _params())
def test_merged_weights_match_the_native_lora_math(family: _Family, tmp_path: Path, peft_process_group):
    """Merging through real PEFT has to reproduce the delta our own tensors describe.

    The reload test reads the delta through the logits of two models we both built.
    This reads it per weight instead: each adapted weight has to equal its base plus
    ``(B @ A) * alpha / r``, computed from the native tensors. A delta that lands on
    the wrong module, at the wrong magnitude, or is compensated for elsewhere has
    nowhere to hide in that comparison.

    This is the merge half of the coverage #3867 asks for. The fused-expert merge in
    tests/unit_tests/models/test_moe_peft_v5_state_dict_adapters.py stays where it is:
    it drives ``target_parameters`` on grouped expert LoRA, which this suite cannot
    build, because applying LoRA to a Transformers model only wraps ``nn.Linear``.
    """
    from peft import PeftModel

    model, reference, peft_config = _adapted_model(family, tmp_path / "source")
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    native = ModelState(model, is_peft=True).state_dict()
    target = reference if family.reference_path is None else getattr(reference, family.reference_path)
    before = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}

    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), key_mapping={}, autocast_adapter_dtype=False)
    merged = loaded.merge_and_unload()
    merged_root = merged if family.reference_path is None else getattr(merged, family.reference_path)
    after = dict(merged_root.named_parameters())

    factors: dict[str, dict[str, torch.Tensor]] = {}
    for key, tensor in native.items():
        side = "A" if ".lora_A" in key else "B" if ".lora_B" in key else None
        if side is not None:
            factors.setdefault(_module_of(key), {})[side] = tensor
    assert factors, f"{family.id}: no lora_A/lora_B pairs to check the merge against"

    scale = peft_config.alpha / peft_config.dim
    for module, pair in sorted(factors.items()):
        assert set(pair) == {"A", "B"}, f"{family.id}: {module} exported {sorted(pair)}, not both factors"
        name = f"{module}.weight"
        assert name in after, f"{family.id}: the merged model has no {name}"
        expected = before[name] + scale * (pair["B"] @ pair["A"])
        torch.testing.assert_close(
            after[name],
            expected,
            rtol=1e-5,
            atol=1e-6,
            msg=lambda formatted, module=module: (
                f"{family.id}: merged {module} is not base + scaled B@A -- {formatted}"
            ),
        )


# --------------------------------------------------------------- fully trained modules


def _train_full_modules(model: nn.Module, family: _Family) -> None:
    """Train ``family.full_modules`` in full next to LoRA, as ``unfreeze_modules`` does.

    The perturbation stands in for training: an export or reload that falls back to the
    base weights for these modules cannot pass a comparison against this model.
    """
    apply_parameter_freezing(model, {"unfreeze_modules": [{"path": path} for path in family.full_modules]})
    with torch.no_grad():
        for path in family.full_modules:
            for parameter in model.get_submodule(path).parameters():
                assert parameter.requires_grad, f"{family.id}: unfreeze_modules left {path} frozen"
                parameter.add_(torch.randn_like(parameter), alpha=0.05)


def _modules_to_save_entry(family: _Family, modules_to_save: list[str], path: str, *, v4_compatible: bool) -> str:
    """The single modules_to_save entry that names ``path``, with any export namespace before it."""
    if v4_compatible and family.legacy_namespace is not None:
        path = family.legacy_namespace + path.removeprefix("model.")
    matches = [entry for entry in modules_to_save if entry == path or entry.endswith(f".{path}")]
    assert len(matches) == 1, f"{family.id}: modules_to_save names {path} {len(matches)} times: {modules_to_save}"
    return matches[0]


@pytest.mark.parametrize("v4_compatible", _FORMATS)
@pytest.mark.parametrize("family", _params())
def test_fully_trained_modules_reload_in_hf_peft(
    family: _Family, v4_compatible: bool, tmp_path: Path, peft_process_group
):
    """Modules trained in full next to LoRA have to come back from the adapter, not the base model.

    The export used to keep only ``lora_`` tensors and wrote no ``modules_to_save``, so
    PEFT reloaded and merged the base weights of these modules without any error. The
    adapter now stores them whole and lists them; this checks the artifact names each
    module once, carries its tensors, and that the real PEFT reload and merge reproduce
    the trained model.
    """
    from peft import PeftModel
    from safetensors.torch import load_file

    model, reference, peft_config = _adapted_model(family, tmp_path / "source")
    _train_full_modules(model, family)
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path, v4_compatible=v4_compatible)

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    modules_to_save = config.get("modules_to_save") or []
    assert len(modules_to_save) == len(family.full_modules), (
        f"{family.id}/{_fmt(v4_compatible)}: expected one modules_to_save entry per trained module, "
        f"got {modules_to_save}"
    )
    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    for path in family.full_modules:
        entry = _modules_to_save_entry(family, modules_to_save, path, v4_compatible=v4_compatible)
        assert any(key.startswith(f"base_model.model.{entry}.") for key in exported), (
            f"{family.id}/{_fmt(v4_compatible)}: modules_to_save lists {entry} but the export has no tensor for it"
        )

    if v4_compatible and family.legacy_namespace is not None:
        # The legacy target is a Transformers v4 model this environment cannot build; the
        # save path itself rejects tensors and modules_to_save that disagree.
        return

    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), key_mapping={}, autocast_adapter_dtype=False).eval()
    adapted = loaded.base_model.model
    if family.reference_path is not None:
        adapted = getattr(adapted, family.reference_path)
    with torch.no_grad():
        expected = model(_INPUT_IDS, use_cache=False).logits
        actual = adapted(_INPUT_IDS, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    merged = loaded.merge_and_unload()
    merged_root = merged if family.reference_path is None else getattr(merged, family.reference_path)
    trained = dict(model.named_parameters())
    for path in family.full_modules:
        for name, parameter in merged_root.get_submodule(path).named_parameters():
            torch.testing.assert_close(
                parameter,
                trained[f"{path}.{name}"],
                rtol=0,
                atol=0,
                msg=lambda formatted, path=path, name=name: (
                    f"{family.id}: merged {path}.{name} is not the trained weight -- {formatted}"
                ),
            )


@pytest.mark.parametrize(
    ("unfreeze_modules", "modules_to_save"),
    [
        pytest.param(["model.layers"], ["model.layers.0"], id="decoder-layer-list"),
        pytest.param(["model"], ["model"], id="whole-backbone"),
    ],
)
def test_trained_module_lists_are_saved_as_modules_hf_peft_can_wrap(
    unfreeze_modules: list[str], modules_to_save: list[str], tmp_path: Path, peft_process_group
):
    """A trained ``ModuleList`` is listed through its blocks, and an enclosing trained module still wins.

    HF PEFT rejects a ``ModuleList`` in ``modules_to_save``, so training ``model.layers`` next to
    LoRA used to save and resume fine but give an adapter that HF PEFT could not load. The layers
    are listed one by one instead, unless the whole backbone trains in full, which PEFT wraps
    as one module.
    """
    from peft import PeftModel

    model, reference, peft_config = _adapted_model(
        _Family("llama_lm_head_lora", _build_llama, {"target_modules": ["lm_head"]}), tmp_path / "source"
    )
    apply_parameter_freezing(model, {"unfreeze_modules": [{"path": path} for path in unfreeze_modules]})
    trained = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    with torch.no_grad():
        for name, parameter in trained.items():
            if ".lora_" not in name:
                parameter.add_(torch.randn_like(parameter), alpha=0.05)
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert config["modules_to_save"] == modules_to_save

    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), autocast_adapter_dtype=False).eval()
    with torch.no_grad():
        expected = model(_INPUT_IDS, use_cache=False).logits
        torch.testing.assert_close(loaded(_INPUT_IDS, use_cache=False).logits, expected, rtol=1e-5, atol=1e-6)
    merged = dict(loaded.merge_and_unload().named_parameters())
    for name, parameter in trained.items():
        if ".lora_" not in name:
            torch.testing.assert_close(merged[name], parameter, rtol=0, atol=0, msg=f"merged {name} is not trained")


@pytest.mark.parametrize("family", _params(tuple(family for family in _FAMILIES if family.layout_is_native)))
def test_fully_trained_modules_resume_through_our_loader(family: _Family, tmp_path: Path, peft_process_group):
    """Resuming from the adapter checkpoint restores the fully trained modules with the LoRA weights.

    The training checkpoint of a PEFT run is the same adapter file. With the trained
    modules missing from it, a resumed run silently restarted them from the base weights
    while their optimizer state was restored.
    """
    model, _, peft_config = _adapted_model(family, tmp_path / "source")
    _train_full_modules(model, family)
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    # Same base weights as the trained model (_adapted_model seeds before building),
    # fresh LoRA weights, and the same trainability policy.
    torch.manual_seed(1234)
    source_model, _ = family.build()
    restored, adapter = family.build()
    restored.load_state_dict(source_model.state_dict())
    if adapter is not None:
        restored.state_dict_adapter = adapter
    apply_lora_to_linear_modules(restored, peft_config)
    apply_parameter_freezing(restored, {"unfreeze_modules": [{"path": path} for path in family.full_modules]})

    checkpointer = Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    )
    checkpointer.load_model(restored, str(adapter_dir))

    trained = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    resumed = {name: parameter for name, parameter in restored.named_parameters() if parameter.requires_grad}
    assert set(resumed) == set(trained), f"{family.id}: the resumed model trains different parameters"
    for name, parameter in trained.items():
        torch.testing.assert_close(
            resumed[name],
            parameter,
            rtol=0,
            atol=0,
            msg=lambda formatted, name=name: f"{family.id}: resume did not restore {name} -- {formatted}",
        )


def test_sequence_classifier_head_round_trips_through_hf_peft_and_resume(tmp_path: Path, peft_process_group):
    """The LoRA sequence-classification recipe's adapter restores its trained classifier.

    The recipe trains the classifier in full next to LoRA (``unfreeze_modules: *classifier``)
    and fine-tunes from a base checkpoint whose ``architectures`` names another task, here
    a masked LM. The adapter has to be typed SEQ_CLS, or PEFT cannot load it onto a
    classifier, and has to store the head, which PEFT's SEQ_CLS wrapper loads from it.
    """
    from peft import PeftModel, PeftModelForSequenceClassification
    from safetensors.torch import load_file
    from transformers import BertConfig, BertForSequenceClassification

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        num_labels=2,
        architectures=["BertForMaskedLM"],
    )
    torch.manual_seed(1234)
    source = BertForSequenceClassification(config).eval()
    peft_config = PeftConfig(target_modules=["*.query", "*.value"], dim=2, alpha=4, use_triton=False)
    freeze_config = {"unfreeze_modules": [{"glob": "*classifier"}]}

    def adapted_classifier() -> nn.Module:
        model = BertForSequenceClassification(config).eval()
        model.load_state_dict(source.state_dict())
        apply_lora_to_linear_modules(model, peft_config)
        apply_parameter_freezing(model, freeze_config)
        return model

    model = adapted_classifier()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter), alpha=0.05)
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["task_type"] == "SEQ_CLS"
    assert adapter_config["modules_to_save"] == ["classifier"]
    assert {"base_model.model.classifier.weight", "base_model.model.classifier.bias"} <= set(
        load_file(str(adapter_dir / "adapter_model.safetensors"))
    )

    reference = BertForSequenceClassification(config).eval()
    reference.load_state_dict(source.state_dict())
    loaded = PeftModel.from_pretrained(reference, str(adapter_dir)).eval()
    assert isinstance(loaded, PeftModelForSequenceClassification)
    with torch.no_grad():
        expected = model(_INPUT_IDS).logits
        torch.testing.assert_close(loaded(_INPUT_IDS).logits, expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(loaded.merge_and_unload()(_INPUT_IDS).logits, expected, rtol=1e-5, atol=1e-6)

    resumed = adapted_classifier()
    Checkpointer(
        CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(tmp_path / "checkpoints"),
            model_save_format="safetensors",
            save_consolidated=False,
            is_peft=True,
        ),
        dp_rank=0,
        tp_rank=0,
        pp_rank=0,
        moe_mesh=None,
    ).load_model(resumed, str(adapter_dir))
    trained = dict(model.named_parameters())
    for name, parameter in resumed.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(parameter, trained[name], rtol=0, atol=0)


def test_encoder_wrapped_classifier_head_reloads_in_hf_peft(tmp_path: Path, peft_process_group):
    """A retrieval-style wrapper exports its trained head under the backbone's own name.

    The wrapper holds the backbone under ``model.`` and its state-dict adapter drops that
    prefix from every exported tensor, so ``modules_to_save`` has to drop it too. The save
    used to stop with an inconsistent-export error naming ``model.classifier``.
    """
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import BertConfig, BertForSequenceClassification

    from nemo_automodel.components.models.common.bidirectional import EncoderStateDictAdapter

    config = BertConfig(
        vocab_size=32,
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=16,
        num_labels=2,
        architectures=["BertForSequenceClassification"],
    )
    torch.manual_seed(1234)
    source = BertForSequenceClassification(config).eval()
    wrapper = nn.Module()
    wrapper.model = BertForSequenceClassification(config).eval()
    wrapper.model.load_state_dict(source.state_dict())
    wrapper.config = config
    wrapper.state_dict_adapter = EncoderStateDictAdapter()
    peft_config = PeftConfig(target_modules=["*.query", "*.value"], dim=2, alpha=4, use_triton=False)
    apply_lora_to_linear_modules(wrapper, peft_config)
    apply_parameter_freezing(wrapper, {"unfreeze_modules": [{"path": "model.classifier"}]})
    with torch.no_grad():
        for parameter in wrapper.parameters():
            if parameter.requires_grad:
                parameter.add_(torch.randn_like(parameter), alpha=0.05)
    adapter_dir = _save_through_checkpointer(wrapper, peft_config, tmp_path)

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["modules_to_save"] == ["classifier"]
    assert all(not name.startswith("model.") for name in adapter_config["target_modules"])
    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    for name in ("weight", "bias"):
        torch.testing.assert_close(
            exported[f"base_model.model.classifier.{name}"],
            wrapper.model.classifier.get_parameter(name),
            rtol=0,
            atol=0,
        )

    reference = BertForSequenceClassification(config).eval()
    reference.load_state_dict(source.state_dict())
    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), autocast_adapter_dtype=False).eval()
    with torch.no_grad():
        expected = wrapper.model(_INPUT_IDS).logits
        torch.testing.assert_close(loaded(_INPUT_IDS).logits, expected, rtol=1e-5, atol=1e-6)
        merged = loaded.merge_and_unload()
        torch.testing.assert_close(merged(_INPUT_IDS).logits, expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(merged.classifier.weight, wrapper.model.classifier.weight, rtol=0, atol=0)


def test_router_trained_next_to_lora_round_trips_with_its_bias_buffer(tmp_path: Path, peft_process_group):
    """A module stored whole carries its persistent buffers, which PEFT restores with its parameters.

    Training an MoE router next to LoRA is a common case. DeepSeek-V3's router keeps its
    ``e_score_correction_bias`` in a persistent buffer, and PEFT's ``modules_to_save`` copy loads
    every state-dict entry of the module, so an export without the buffer failed to load. Resuming
    restores the buffer as well, including under expert parallelism, whose load bypasses DCP.
    """
    from peft import PeftModel
    from safetensors.torch import load_file
    from transformers import DeepseekV3Config, DeepseekV3ForCausalLM

    config = DeepseekV3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_group=1,
        topk_group=1,
        first_k_dense_replace=0,
        q_lora_rank=8,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
        max_position_embeddings=32,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1234)
    source = DeepseekV3ForCausalLM(config).eval()
    peft_config = PeftConfig(target_modules=["*.o_proj"], dim=2, alpha=4, use_triton=False)
    router = "model.layers.0.mlp.gate"

    def adapted_model() -> nn.Module:
        model = DeepseekV3ForCausalLM(config).eval()
        model.load_state_dict(source.state_dict())
        apply_lora_to_linear_modules(model, peft_config)
        apply_parameter_freezing(model, {"unfreeze_modules": [{"path": router}]})
        return model

    model = adapted_model()
    with torch.no_grad():
        gate = model.get_submodule(router)
        gate.weight.add_(torch.randn_like(gate.weight), alpha=0.05)
        # Stands in for load-balancing updates: the trained router routes with a different bias.
        gate.e_score_correction_bias.copy_(torch.randn_like(gate.e_score_correction_bias))
        for name, parameter in model.named_parameters():
            if ".lora_" in name:
                parameter.normal_(std=0.05)
    adapter_dir = _save_through_checkpointer(model, peft_config, tmp_path)

    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    assert adapter_config["modules_to_save"] == [router]
    exported = load_file(str(adapter_dir / "adapter_model.safetensors"))
    assert {f"base_model.model.{router}.weight", f"base_model.model.{router}.e_score_correction_bias"} <= set(exported)

    reference = DeepseekV3ForCausalLM(config).eval()
    reference.load_state_dict(source.state_dict())
    loaded = PeftModel.from_pretrained(reference, str(adapter_dir), autocast_adapter_dtype=False).eval()
    with torch.no_grad():
        expected = model(_INPUT_IDS).logits
        torch.testing.assert_close(loaded(_INPUT_IDS).logits, expected, rtol=1e-5, atol=1e-6)
        merged_router = loaded.merge_and_unload().get_submodule(router)
    torch.testing.assert_close(merged_router.e_score_correction_bias, gate.e_score_correction_bias, rtol=0, atol=0)

    expected_state = model.state_dict()
    # With expert parallelism the checkpointer loads PEFT tensors without DCP; both paths restore the buffer.
    for moe_mesh in (None, init_device_mesh("cpu", (1,), mesh_dim_names=("ep",))):
        resumed = adapted_model()
        Checkpointer(
            CheckpointingConfig(
                enabled=True,
                checkpoint_dir=str(tmp_path / "checkpoints"),
                model_save_format="safetensors",
                save_consolidated=False,
                is_peft=True,
            ),
            dp_rank=0,
            tp_rank=0,
            pp_rank=0,
            moe_mesh=moe_mesh,
        ).load_model(resumed, str(adapter_dir))
        for key in (f"{router}.weight", f"{router}.e_score_correction_bias"):
            torch.testing.assert_close(
                resumed.state_dict()[key],
                expected_state[key],
                rtol=0,
                atol=0,
                msg=lambda formatted, key=key, moe_mesh=moe_mesh: (
                    f"resume with moe_mesh={moe_mesh} did not restore {key} -- {formatted}"
                ),
            )
