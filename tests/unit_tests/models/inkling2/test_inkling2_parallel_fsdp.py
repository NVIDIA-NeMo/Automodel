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

"""FSDP2 unit layout and expert-activation wiring of Inkling 2 on the expert-parallel path."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy

from nemo_automodel._transformers.model_capabilities import query_capabilities
from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2 import parallelization as parallelization_module
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.layers import Inkling2MoE, inkling2_softcapped_swiglu
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration
from nemo_automodel.components.moe import layers as moe_layers
from nemo_automodel.components.moe import parallelizer as moe_parallelizer
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP

pytestmark = pytest.mark.cpu_only

# Parameters the reference computes in FP32 (Transformers ``_keep_in_fp32_modules_strict``).
_FP32_ROLES = (
    "mlp.gate.weight",
    "attention_input.query",
    "mlp_input.query",
    "final_input.query",
    "post_attention_convolution.conv1d.weight",
    "post_mlp_convolution.conv1d.weight",
    "k_sconv.conv1d.weight",
    "v_sconv.conv1d.weight",
)


def _tiny_config() -> Inkling2Config:
    text = {
        "hidden_size": 32,
        "num_hidden_layers": 7,
        "local_layer_ids": [0, 1, 2, 3, 4, 6],
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 16,
        "swa_head_dim": 16,
        "swa_num_attention_heads": 2,
        "swa_num_key_value_heads": 1,
        "sliding_window_size": 8,
        "rel_extent": 8,
        "d_rel": 4,
        "vocab_size": 64,
        "unpadded_vocab_size": 60,
        "n_routed_experts": 4,
        "num_experts_per_tok": 2,
        "intermediate_size": 16,
        "n_layers_per_attn_res_block": 6,
        "sconv_kernel_size": 4,
        "eos_token_id": 1,
    }
    return Inkling2Config(text_config=text, vision_config=None, eos_token_id=1, torch_dtype=torch.bfloat16)


def _make_model(**backend_kwargs) -> Inkling2ForConditionalGeneration:
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch_fp32", **backend_kwargs)
    return Inkling2ForConditionalGeneration.from_config(_tiny_config(), backend=backend)


def _record_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[nn.Module, set[nn.Parameter], MixedPrecisionPolicy]]:
    """Replace ``fully_shard`` with a recorder of each unit's newly owned parameters and policy."""
    calls = []

    def shard(module, *, mesh, mp_policy, ignored_params=None, **kwargs):
        modules = module if isinstance(module, list) else [module]
        owned = {p for m in modules for p in m.parameters()} - set(ignored_params or ())
        for m in modules:
            assert not isinstance(m, FSDPModule), "An existing FSDP unit was wrapped twice"
            for child in m.modules():
                if child is not m and isinstance(child, FSDPModule):
                    owned.difference_update(child.parameters())
            m.__class__ = type(f"TestFSDP{type(m).__name__}", (FSDPModule, type(m)), {})
        calls.append((module, owned, mp_policy))
        return module

    for target in (parallelizer_utils, moe_parallelizer, parallelization_module):
        monkeypatch.setattr(target, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_uniform_reduce_dtype", lambda: None)
    # The recorder has no real mesh; keep the router correction-bias buffers local.
    monkeypatch.setattr(moe_parallelizer, "distribute_tensor", lambda tensor, **kwargs: tensor)
    return calls


def _owner_policies(model: nn.Module, calls) -> dict[str, MixedPrecisionPolicy]:
    owners = {}
    for name, param in model.named_parameters():
        units = [policy for _, owned, policy in calls if param in owned]
        assert len(units) <= 1, f"{name} is owned by {len(units)} FSDP units"
        if units:
            owners[name] = units[0]
    return owners


def test_inkling2_uses_model_owned_parallelizer():
    assert Inkling2ForConditionalGeneration.parallelizer is parallelization_module.PARALLELIZER
    assert get_model_parallelizer(_make_model()) is parallelization_module.PARALLELIZER


def test_inkling2_declares_expert_but_not_pipeline_parallelism():
    capabilities = query_capabilities(Inkling2ForConditionalGeneration)
    assert capabilities.supports_ep
    assert not capabilities.supports_pp


@pytest.mark.parametrize("model_owned", [False, True])
def test_fp32_parameters_get_fp32_units_on_the_expert_parallel_path(monkeypatch, model_owned):
    """Every reference-FP32 parameter computes in FP32 and every unit holds one storage dtype."""
    model = _make_model(experts="torch_mm", dispatcher="torch")
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    calls = _record_sharding(monkeypatch)
    moe_parallelizer.apply_fsdp(
        model,
        fsdp_mesh=SimpleNamespace(ndim=1),
        ep_enabled=True,
        ep_shard_enabled=False,
        mp_policy=policy,
        reshard_after_forward=True,
        model_parallelizer=parallelization_module.PARALLELIZER if model_owned else None,
    )
    owners = _owner_policies(model, calls)
    params = dict(model.named_parameters())
    fp32_names = [name for name in params if any(role in name for role in _FP32_ROLES)]
    assert any("final_input" in name for name in fp32_names)
    mixed_units = [module for module, owned, _ in calls if len({p.dtype for p in owned if p.requires_grad}) > 1]
    if not model_owned:
        # Without the sidecar the text root holds the BF16 final norm next to the FP32 final query, which
        # FSDP2 rejects at lazy init ("FSDP expects uniform original parameter dtype").
        assert mixed_units == [model.model.language_model]
        return
    assert not mixed_units
    for name in fp32_names:
        assert params[name].dtype is torch.float32, name
        assert owners[name].param_dtype is torch.float32, f"{name} computes in {owners[name].param_dtype}"
    experts = [p for n, p in model.named_parameters() if ".experts." in n]
    assert experts and not any(p in owned for p in experts for _, owned, _ in calls), "EP owns the experts"


@pytest.mark.parametrize(
    ("dispatcher", "experts"),
    [("torch", "torch"), ("torch", "torch_mm"), ("hybridep", "torch_mm"), ("hybridep", "te")],
)
def test_expert_activation_is_softcapped_on_single_rank_backends(dispatcher, experts):
    """HybridEP falls back to GroupedExperts on one rank; the Inkling 2 activation must still be plugged in."""
    model = _make_model(experts=experts, dispatcher=dispatcher)
    moes = [m for m in model.modules() if isinstance(m, Inkling2MoE)]
    assert len(moes) == 7
    for moe in moes:
        assert isinstance(moe.experts, GroupedExperts)
        activation = moe.experts.expert_activation_grouped
        gate_up = torch.tensor([[2.0, 30.0]])
        expected = (
            torch.tanh(torch.tensor(2.0 / 4.0))
            * 4.0
            * torch.sigmoid(torch.tensor(2.0))
            * 25.0
            * torch.tanh(torch.tensor(30.0 / 25.0))
        )
        torch.testing.assert_close(activation(gate_up, torch.ones(1, 1)), expected.view(1, 1))


def _softcapped_reference(gate: float, up: float) -> torch.Tensor:
    gate_t, up_t = torch.tensor(gate), torch.tensor(up)
    return (torch.tanh(gate_t / 4.0) * 4.0 * torch.sigmoid(gate_t) * torch.tanh(up_t / 25.0) * 25.0).view(1, 1)


@pytest.mark.parametrize("dispatcher", ["deepep", "hybridep", "uccl_ep"])
def test_expert_parallel_grouped_mm_experts_get_the_softcapped_activation(monkeypatch, dispatcher):
    """With more than one rank the flex dispatchers build GroupedExpertsDeepEP; its hook must be Inkling 2's."""
    monkeypatch.setattr(moe_layers, "get_world_size_safe", lambda: 2)
    config = _tiny_config().text_config
    moe = Inkling2MoE(config, BackendConfig(attn="sdpa", experts="torch_mm", dispatcher=dispatcher))
    assert isinstance(moe.experts, GroupedExpertsDeepEP)
    assert moe.experts.expert_activation.func is inkling2_softcapped_swiglu
    torch.testing.assert_close(
        moe.experts.expert_activation(torch.tensor([[2.0, 30.0]]), torch.ones(1, 1)), _softcapped_reference(2.0, 30.0)
    )


@pytest.mark.parametrize("world_size", [1, 4])
def test_mok_dispatcher_is_rejected(monkeypatch, world_size):
    """MoK fuses plain SwiGLU with no activation hook; construction must fail instead of training it."""
    monkeypatch.setattr(moe_layers, "get_world_size_safe", lambda: world_size)
    with pytest.raises(ValueError, match="mok"):
        Inkling2MoE(_tiny_config().text_config, BackendConfig(attn="sdpa", experts="torch_mm", dispatcher="mok"))


def test_checkpoint_free_construction_initializes_every_weight():
    """``from_config`` random init (meta construction, then the checkpointer's initializer) must not leave
    the ``to_empty`` memory in place: an all-zero model trains at a constant loss of ln(vocab)."""
    from nemo_automodel.components.checkpoint.checkpointing import Checkpointer

    torch.manual_seed(0)
    with torch.device("meta"):
        model = _make_model(experts="torch_mm", dispatcher="torch")
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    params = dict(model.named_parameters())
    for name in (
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.5.mlp.gate.weight",
        "model.language_model.layers.5.mlp.experts.gate_and_up_projs",
        "model.language_model.layers.6.mlp.experts.down_projs",
    ):
        assert torch.isfinite(params[name]).all(), name
        assert params[name].float().std() > 0, f"{name} was not initialized"
    for name, param in params.items():
        assert torch.isfinite(param).all(), name
