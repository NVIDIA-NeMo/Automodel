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


@dataclass(frozen=True)
class _Family:
    """One covered model family.

    Attributes:
        id: pytest parameter id.
        build: Returns ``(tiny Transformers model, adapter or None)``. The adapter is the
            AutoModel component that owns this family's HF naming.
        peft_kwargs: Overrides for the ``PeftConfig`` the export is driven with.
        xfail: Issue reference when this family has a known, filed export defect.
    """

    id: str
    build: Callable[[], tuple[nn.Module, object | None]]
    peft_kwargs: dict = field(default_factory=dict)
    xfail: str | None = None


_FAMILIES = (
    _Family("llama_dense", _build_llama, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    _Family("nemotron_v3", _build_nemotron_v3, {"exclude_modules": ["*.out_proj"]}),
    _Family("qwen3_moe", _build_qwen3_moe, {"target_modules": ["*.q_proj", "*.v_proj"]}),
    _Family("minimax_m2", _build_minimax_m2, {"target_modules": ["*.q_proj", "*.v_proj"]}),
)


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
    reference, adapter = family.build()
    reference = reference.eval()
    reference.save_pretrained(source)

    model, _ = family.build()
    model = model.eval()
    model.load_state_dict(reference.state_dict())
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


@pytest.mark.parametrize("family", _FAMILIES, ids=lambda family: family.id)
def test_exported_adapter_loads_into_hf_peft(family: _Family, tmp_path: Path, peft_process_group):
    """Save through the real path, reload with real PEFT, and require identical behavior."""
    if family.xfail:
        pytest.xfail(family.xfail)
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

    with torch.no_grad():
        expected = model(_INPUT_IDS, use_cache=False).logits
        actual = loaded(_INPUT_IDS, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", _FAMILIES, ids=lambda family: family.id)
def test_bulk_and_per_tensor_exports_agree(family: _Family, tmp_path: Path, peft_process_group):
    """``to_hf`` and ``convert_single_tensor_to_hf`` feed different consumers; they must match.

    Streaming consumers convert one tensor at a time, which cannot see the sibling
    tensors a fused conversion needs, so the two paths drift apart silently.
    """
    if family.xfail:
        pytest.xfail(family.xfail)
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
