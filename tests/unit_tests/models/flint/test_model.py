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

"""Public loader, initialization, LoRA, padding-mask and tiny checkpoint gates for the Flint model."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, GenerationConfig
from transformers.modeling_utils import PreTrainedModel

from nemo_automodel import NeMoAutoModelForCausalLM, query_capabilities
from nemo_automodel._transformers.model_init import get_hf_config
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.checkpoint.addons import _save_generated_hf_assets
from nemo_automodel.components.checkpoint.checkpointing import Checkpointer
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.tie_word_embeddings import TieSupport
from nemo_automodel.components.models.common.utils import restore_pretrained_generation_config
from nemo_automodel.components.models.flint import FlintConfig, FlintForCausalLM
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from tests.unit_tests.models.flint.conftest import checkpoint_style_config, cpu_loader

pytestmark = pytest.mark.cpu_only


def _config(**overrides: object) -> FlintConfig:
    """Cover dense/full, MoE/sliding and MoE/full in a very small architecture."""
    settings = {
        "vocab_size": 31,
        "hidden_size": 12,
        "intermediate_size": 20,
        "num_hidden_layers": 3,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "moe_intermediate_size": 5,
        "shared_expert_intermediate_size": 7,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "mlp_only_layers": [0],
        "layer_types": ["full_attention", "sliding_attention", "full_attention"],
        "no_rope_layer_indices": [0, 2],
        "sliding_window": 3,
        "max_position_embeddings": 32,
        "pad_token_id": 2,
        "eos_token_id": [2, 3, 4],
        "dtype": torch.float32,
    }
    settings.update(overrides)
    return FlintConfig(**settings)


def _backend() -> BackendConfig:
    """Use the actual torch components and source-specific gate precision."""
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
    )


def _load(source: str | Path | FlintConfig) -> FlintForCausalLM:
    """Use adapter-aware public NeMo loading, with no CUDA or model-class mocks."""
    options = {
        "dtype": torch.float32,
        "backend": _backend(),
        "attn_implementation": "eager",
        "use_liger_kernel": False,
        "use_sdpa_patching": False,
    }
    with cpu_loader():
        if isinstance(source, FlintConfig):
            return NeMoAutoModelForCausalLM.from_config(source, **options).eval()
        return NeMoAutoModelForCausalLM.from_config(
            str(source), load_base_model=True, local_files_only=True, **options
        ).eval()


def test_checkpoint_config_resolves_local_class_without_remote_module(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(checkpoint_style_config()))
    config = get_hf_config(str(tmp_path), "eager", trust_remote_code=False, local_files_only=True)
    assert type(config) is FlintConfig
    assert type(AutoConfig.from_pretrained(tmp_path, trust_remote_code=False)) is FlintConfig
    assert config.architectures == ["FlintForCausalLM"]
    assert (config.hidden_size, config.head_dim * config.num_attention_heads) == (2048, 4096)
    assert config.router_aux_loss_coef == 0.0005
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": 10000.0}
    assert not hasattr(config, "rope_theta")
    assert len(config.no_rope_layer_indices) == 14


def test_public_random_init_is_finite_trainable_and_untied() -> None:
    model = _load(_config())
    assert isinstance(model, PreTrainedModel)
    assert model.tie_word_embeddings_support is TieSupport.UNTIED_ONLY
    assert model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr()
    assert model.get_input_embeddings().padding_idx is None
    assert all(parameter.device.type == "cpu" and torch.isfinite(parameter).all() for parameter in model.parameters())
    assert all(torch.isfinite(buffer).all() for buffer in model.buffers() if buffer.is_floating_point())
    tokens = torch.tensor([[1, 2, 6, 7, 8], [5, 4, 2, 2, 2]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    result = model(tokens, attention_mask=mask, output_hidden_states=True)
    assert result.logits.shape == (2, 5, 31) and result.logits.dtype == torch.float32
    assert len(result.hidden_states) == model.config.num_hidden_layers + 1
    (result.logits * torch.linspace(-1, 1, result.logits.numel()).reshape_as(result.logits)).sum().backward()
    for layer in model.model.layers:
        sink = layer.self_attn._fp32_params.softmax_offset
        assert sink.dtype == torch.float32 and sink.grad is not None and torch.isfinite(sink.grad).all()
        assert torch.count_nonzero(sink.grad) > 0
    assert torch.count_nonzero(model.get_input_embeddings().weight.grad[2]) > 0
    caps = query_capabilities(model)
    assert not caps.supports_tp
    assert all((caps.supports_cp, caps.supports_pp, caps.supports_ep))


def test_production_initializer_materializes_meta_without_checkpoint() -> None:
    with torch.device("meta"):
        model = _config().build(backend=_backend())
    assert all(parameter.is_meta for parameter in model.parameters())
    Checkpointer.initialize_model_weights(model, torch.device("cpu"))
    assert all(not parameter.is_meta and torch.isfinite(parameter).all() for parameter in model.parameters())
    output = model(torch.tensor([[1, 4, 8, 2]])).logits
    output.square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_tied_and_unresolved_auxiliary_configurations_fail_before_allocation() -> None:
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"):
        _config(tie_word_embeddings=True).build()
    with pytest.raises(ValueError, match="auxiliary-loss formula"):
        _config(router_aux_loss_coef=0.0005).build()


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"num_key_value_heads": 3}, "divisible"),
        ({"head_dim": 3}, "even"),
        ({"no_rope_layer_indices": [0]}, "no_rope_layer_indices"),
        ({"layer_types": ["full_attention"]}, "layer_types"),
        ({"sliding_window": 0}, "sliding_window"),
        ({"num_experts_per_tok": 1}, "num_experts_per_tok"),
        ({"mlp_only_layers": [0, 0]}, "mlp_only_layers"),
        ({"attention_bias": True}, "bias-free"),
    ],
)
def test_unsupported_architecture_settings_fail_explicitly(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides)


def test_cache_and_raw_hf_loader_are_rejected() -> None:
    model = _load(_config())
    with pytest.raises(ValueError, match="KV caching"):
        model(torch.tensor([[1, 2]]), use_cache=True)
    with pytest.raises(ValueError, match="NeMoAutoModelForCausalLM"):
        FlintForCausalLM.from_pretrained("unused")


def test_embedding_setters_reject_distinct_parameters_sharing_storage() -> None:
    model = _load(_config())
    aliased_input = torch.nn.Embedding.from_pretrained(model.lm_head.weight, freeze=False)
    assert aliased_input.weight is not model.lm_head.weight
    with pytest.raises(ValueError, match="untied storage"):
        model.set_input_embeddings(aliased_input)
    aliased_output = torch.nn.Linear(12, 31, bias=False)
    aliased_output.weight = torch.nn.Parameter(model.model.embed_tokens.weight.detach())
    assert aliased_output.weight is not model.model.embed_tokens.weight
    with pytest.raises(ValueError, match="untied"):
        model.set_output_embeddings(aliased_output)


@pytest.mark.parametrize("mask", [[0, 0, 1, 1], [1, 0, 1, 1], [0, 0, 0, 0]])
def test_hf_labels_supervise_only_adjacent_valid_positions(mask: list[int]) -> None:
    model = _load(_config())
    tokens = torch.tensor([[2, 2, 7, 8]])
    attention_mask = torch.tensor([mask])
    output = model(tokens, labels=tokens, attention_mask=attention_mask)
    valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
    target = tokens[:, 1:].clone().masked_fill(~valid, -100)
    expected = torch.nn.functional.cross_entropy(
        output.logits[:, :-1].reshape(-1, 31), target.reshape(-1), ignore_index=-100, reduction="sum"
    ) / valid.sum().clamp_min(1)
    torch.testing.assert_close(output.loss, expected, atol=0, rtol=0)
    if not valid.any():
        assert output.loss.item() == 0
    output.loss.backward()
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_reinitialization_keeps_adapter_dtype_and_fixed_vocabulary_consistent() -> None:
    model = _load(_config())
    model.initialize_weights(dtype=torch.bfloat16)
    assert model.state_dict_adapter.dtype == torch.bfloat16
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.float32
    assert all(layer.self_attn._fp32_params.softmax_offset.dtype == torch.float32 for layer in model.model.layers)
    native = model.state_dict()
    restored = model.state_dict_adapter.from_hf(model.state_dict_adapter.to_hf(native))
    assert all(value.dtype == native[name].dtype for name, value in restored.items())
    with pytest.raises(ValueError, match="only float32 and bfloat16"):
        model.initialize_weights(dtype=torch.float16)
    assert model.state_dict_adapter.dtype == torch.bfloat16
    assert model.resize_token_embeddings() is model.model.embed_tokens
    with pytest.raises(ValueError, match="fixed checkpoint vocabulary"):
        model.resize_token_embeddings(32)
    with pytest.raises(ValueError, match="fixed checkpoint vocabulary"):
        model.resize_token_embeddings(pad_to_multiple_of=16)
    with pytest.raises(ValueError, match="input embeddings"):
        model.set_input_embeddings(torch.nn.Embedding(32, 12))
    with pytest.raises(ValueError, match="output embeddings"):
        model.set_output_embeddings(torch.nn.Linear(12, 32, bias=False))


def _write_tiny_checkpoint(directory: Path, model: FlintForCausalLM) -> None:
    """Save the config and BF16 HF-layout weights of ``model`` as an indexed checkpoint."""
    directory.mkdir()
    model.config.save_pretrained(directory)
    hf_weights = {
        name: value.detach().to(torch.bfloat16).contiguous()
        for name, value in model.state_dict_adapter.to_hf(model.state_dict()).items()
    }
    save_file(hf_weights, directory / "model-00001-of-00001.safetensors")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.numel() * value.element_size() for value in hf_weights.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_weights},
            }
        )
    )


def test_generation_config_takes_stop_tokens_from_config() -> None:
    """A training-only class still seeds generation_config like PreTrainedModel does."""
    model = _config().build(backend=_backend())
    assert isinstance(model.generation_config, GenerationConfig)
    assert model.generation_config.eos_token_id == [2, 3, 4]
    assert model.generation_config.pad_token_id == 2


def test_public_load_restores_and_exports_checkpoint_generation_config(tmp_path: Path) -> None:
    """The checkpoint's sampling defaults and stop tokens survive the load and the export."""
    source = tmp_path / "source"
    _write_tiny_checkpoint(source, _config().build(backend=_backend()).eval())
    GenerationConfig(
        eos_token_id=[2, 3, 4, 7], pad_token_id=2, do_sample=True, temperature=0.7, top_k=20
    ).save_pretrained(source)
    loaded = _load(source)
    # NeMoAutoModelForCausalLM.from_pretrained restores the checkpoint's generation
    # config through this helper right after construction; the CPU from_config route
    # used here does not, so apply the same step explicitly.
    restore_pretrained_generation_config(loaded, str(source))
    assert loaded.generation_config.eos_token_id == [2, 3, 4, 7]
    assert loaded.generation_config.do_sample is True
    assert loaded.generation_config.temperature == 0.7
    export = tmp_path / "export"
    export.mkdir()
    _save_generated_hf_assets(loaded, None, str(export), tokenizer=None, v4_compatible=False)
    exported = GenerationConfig.from_pretrained(export)
    assert exported.eos_token_id == [2, 3, 4, 7] and exported.top_k == 20


def test_public_bf16_base_load_consolidated_export_and_fresh_reload(tmp_path: Path) -> None:
    """Real tiny files preserve source dtype, untied storage and exact loaded logits."""
    initial = _config().build(backend=_backend()).eval()
    source = tmp_path / "source"
    source.mkdir()
    initial.config.save_pretrained(source)
    hf_weights = {
        name: value.detach().to(torch.bfloat16).contiguous()
        for name, value in initial.state_dict_adapter.to_hf(initial.state_dict()).items()
    }
    save_file(hf_weights, source / "model-00001-of-00001.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(value.numel() * value.element_size() for value in hf_weights.values())},
                "weight_map": {name: "model-00001-of-00001.safetensors" for name in hf_weights},
            }
        )
    )
    loaded = _load(source)
    expected = _config().build(backend=_backend()).eval()
    expected.load_state_dict(expected.state_dict_adapter.from_hf(dict(hf_weights)), strict=True)
    assert all(parameter.dtype == torch.float32 for parameter in loaded.parameters())
    for name, parameter in loaded.state_dict().items():
        torch.testing.assert_close(parameter, expected.state_dict()[name], atol=0, rtol=0)
    tokens = torch.tensor([[1, 2, 5, 9]])
    torch.testing.assert_close(loaded(tokens).logits, expected(tokens).logits, atol=0, rtol=0)
    checkpointer = CheckpointingConfig(
        model_save_format="safetensors",
        save_consolidated="final",
        is_async=False,
        model_repo_id=str(source),
        original_model_root_dir=str(source),
        model_cache_dir=str(tmp_path),
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    try:
        loaded.save_pretrained(str(tmp_path / "saved"), checkpointer=checkpointer, is_final_checkpoint=True)
    finally:
        checkpointer.close()
    consolidated = tmp_path / "saved/model/consolidated"
    index = json.loads((consolidated / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(hf_weights)
    for filename in set(index.values()):
        with safe_open(consolidated / filename, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                exported = handle.get_tensor(name)
                assert exported.dtype == torch.bfloat16
                torch.testing.assert_close(exported, hf_weights[name], atol=0, rtol=0)
    restored = _load(consolidated)
    assert restored.get_input_embeddings().weight.data_ptr() != restored.lm_head.weight.data_ptr()
    torch.testing.assert_close(restored(tokens).logits, loaded(tokens).logits, atol=0, rtol=0)
    with pytest.raises(NotImplementedError, match="tie_word_embeddings"), cpu_loader():
        NeMoAutoModelForCausalLM.from_config(
            str(source),
            load_base_model=True,
            tie_word_embeddings=True,
            dtype=torch.float32,
            use_liger_kernel=False,
            use_sdpa_patching=False,
            attn_implementation="eager",
            backend=_backend(),
            local_files_only=True,
        )


def _expert_config() -> MoEConfig:
    return MoEConfig(
        dim=8,
        inter_dim=12,
        moe_inter_dim=8,
        n_routed_experts=4,
        n_shared_experts=0,
        n_activated_experts=2,
        n_expert_groups=1,
        n_limited_groups=1,
        train_gate=True,
        gate_bias_update_factor=0.0,
        aux_loss_coeff=0.0,
        score_func="sigmoid",
        route_scale=1.0,
        norm_topk_prob=True,
        expert_activation="swiglu",
        apply_router_weight_after_down=True,
        dtype=torch.float32,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_zero_expert_lora_preserves_after_down_routing_exactly(dtype: torch.dtype) -> None:
    """An inactive adapter must preserve the actual rounded BF16 base computation."""
    torch.manual_seed(812)
    base = GroupedExperts(_expert_config())
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(std=0.3)
    adapted = GroupedExpertsLoRA(base, lora_dim=4, alpha=8)
    x = torch.randn(11, 8, dtype=dtype)
    mask = torch.tensor([1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1], dtype=torch.bool)
    routes = torch.tensor([[0, 2], [3, 1], [2, 3], [1, 0], [2, 1], [0, 1], [3, 2], [1, 2], [3, 0], [1, 3], [2, 0]])
    probabilities = torch.rand(11, 2)
    actual = adapted(x, mask, probabilities, routes)
    expected = base(x, mask, probabilities, routes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[~mask]) == 0


@pytest.mark.parametrize("mode", ["loop", "grouped_mm", "dispatched"])
def test_nonzero_expert_lora_matches_independent_effective_weights_and_gradients(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch.manual_seed(813)
    base = GroupedExperts(_expert_config())
    with torch.no_grad():
        for parameter in base.parameters():
            parameter.normal_(std=0.2)
    if mode == "dispatched":
        base = GroupedExpertsDeepEP(_expert_config())
        with torch.no_grad():
            for parameter in base.parameters():
                parameter.normal_(std=0.2)
        module = GroupedExpertsDeepEPLoRA(base, lora_dim=4, alpha=8)
        module.ep_size = 1
    else:
        base.use_torch_mm = mode == "grouped_mm"
        module = GroupedExpertsLoRA(base, lora_dim=4, alpha=8)

    def grouped_mm(value: torch.Tensor, weight: torch.Tensor, *, offs: torch.Tensor) -> torch.Tensor:
        """Multiply sorted [rows,input] by per-expert [experts,input,output] weights.

        Args:
            value: Expert-sorted tensor [rows, input].
            weight: Projection tensors [experts, input, output].
            offs: Cumulative row endpoints [experts].

        Returns:
            Independent concatenated results [rows, output].
        """
        bounds = [0, *offs.tolist()]
        return torch.cat([value[bounds[e] : bounds[e + 1]] @ weight[e] for e in range(weight.shape[0])])

    if mode != "loop":
        monkeypatch.setattr(torch, "_grouped_mm", grouped_mm)
    names = ("lora_gate_and_up_A", "lora_gate_and_up_B", "lora_down_A", "lora_down_B")
    with torch.no_grad():
        module.lora_gate_and_up_B.normal_(std=0.1)
        module.lora_down_B.normal_(std=0.1)
    x = torch.randn(5, 8, requires_grad=True)
    probabilities = torch.rand(5, 2, requires_grad=True)
    ids = torch.tensor([[0, 2], [3, 1], [2, 3], [1, 0], [2, 1]])
    mask = torch.tensor([1, 1, 0, 1, 1], dtype=torch.bool)
    if mode == "dispatched":
        pairs = [(t, slot) for e in range(4) for t in range(5) for slot in range(2) if mask[t] and ids[t, slot] == e]
        token_rows = torch.tensor([t for t, _ in pairs])
        slot_rows = torch.tensor([slot for _, slot in pairs])

        def dispatch(
            value: torch.Tensor, valid: torch.Tensor, probs: torch.Tensor, routes: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """Provide deterministic EP-size1 sorted tensors, without simulating collectives.

            Args:
                value: Activations [tokens, hidden].
                valid: Token validity [tokens].
                probs: Routing probabilities [tokens, topk].
                routes: Expert IDs [tokens, topk].

            Returns:
                Sorted activations [routed_tokens, hidden], counts [experts],
                and sorted probabilities [routed_tokens, 1].
            """
            counts = torch.tensor([sum(int(routes[t, slot]) == e for t, slot in pairs) for e in range(4)])
            return value[token_rows], counts, probs[token_rows, slot_rows, None]

        def combine(value: torch.Tensor) -> torch.Tensor:
            """Combine expert-sorted [routed_tokens,hidden] into fresh [tokens,hidden]."""
            return value.new_zeros(5, 8).index_add(0, token_rows, value)

        monkeypatch.setattr(module, "_dispatch_tokens", dispatch)
        module.token_dispatcher = SimpleNamespace(token_unpermutation=combine)
    actual = module(x, mask, probabilities, ids)
    x_ref = x.detach().clone().requires_grad_()
    p_ref = probabilities.detach().clone().requires_grad_()
    adapters = [getattr(module, name).detach().clone().requires_grad_() for name in names]
    a1, b1, a2, b2 = adapters
    rows = []
    for token in range(5):
        slots = []
        for slot in range(2):
            expert = ids[token, slot]
            up = module.gate_and_up_projs[expert].detach() + module.scale * (a1[expert] @ b1[expert])
            down = module.down_projs[expert].detach() + module.scale * (a2[expert] @ b2[expert])
            gate, value = (x_ref[token] @ up).chunk(2, -1)
            slots.append(((F.silu(gate) * value) @ down) * p_ref[token, slot] * mask[token])
        rows.append(sum(slots))
    expected = torch.stack(rows)
    cotangent = torch.randn_like(actual)
    actual_grad = torch.autograd.grad(actual, [x, probabilities, *[getattr(module, name) for name in names]], cotangent)
    expected_grad = torch.autograd.grad(expected, [x_ref, p_ref, *adapters], cotangent)
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, reference, rtol=3e-5, atol=3e-6)
    assert not module.gate_and_up_projs.requires_grad
    assert not module.down_projs.requires_grad


def test_public_peft_targets_attention_and_experts_only_and_updates_adapters() -> None:
    torch.manual_seed(814)
    config = _model_config()
    model = FlintForCausalLM(config)
    baseline = copy.deepcopy(model)
    count = apply_lora_to_linear_modules(
        model,
        PeftConfig(
            target_modules=["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.experts"],
            dim=4,
            alpha=8,
            lora_dtype=torch.float32,
            use_memory_efficient_lora=False,
        ),
    )
    assert count == 9
    assert isinstance(model.model.layers[1].mlp.experts, GroupedExpertsLoRA)
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    torch.testing.assert_close(model(ids).logits, baseline(ids).logits, rtol=0, atol=0)
    frozen = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters() if not parameter.requires_grad
    }
    trainable = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable and all("lora_" in name for name in trainable)
    assert not model.lm_head.weight.requires_grad
    optimizer = torch.optim.AdamW(trainable.values(), lr=1e-3)
    model(ids, labels=ids).loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable.values())
    assert any(torch.count_nonzero(parameter.grad) for parameter in trainable.values())
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name in frozen:
            torch.testing.assert_close(parameter, frozen[name], rtol=0, atol=0)


def _model_config() -> FlintConfig:
    return FlintConfig(
        vocab_size=19,
        hidden_size=8,
        intermediate_size=12,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        layer_types=["full_attention", "sliding_attention"],
        no_rope_layer_indices=[0],
        mlp_only_layers=[0],
        moe_intermediate_size=8,
        shared_expert_intermediate_size=12,
        num_experts=4,
        num_experts_per_tok=2,
        sliding_window=3,
    )


@pytest.mark.parametrize("bulk_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("adapter_dtype", [torch.float32, torch.bfloat16])
def test_meta_materialization_preserves_explicit_linear_and_expert_adapter_dtype(
    bulk_dtype: torch.dtype, adapter_dtype: torch.dtype
) -> None:
    """The actual post-PEFT initializer must retain both adapter tensor layouts."""
    config = FlintConfig(**{**_model_config().to_dict(), "dtype": bulk_dtype})
    with torch.device("meta"):
        model = FlintForCausalLM(config)
        apply_lora_to_linear_modules(
            model,
            PeftConfig(
                target_modules=["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.experts"],
                dim=4,
                alpha=8,
                lora_dtype=adapter_dtype,
                use_memory_efficient_lora=False,
            ),
        )
    adapter_names = {name for name, _ in model.named_parameters() if "lora_" in name}
    assert any("self_attn" in name for name in adapter_names)
    assert any("mlp.experts.lora_" in name for name in adapter_names)
    Checkpointer.initialize_model_weights(model, torch.device("cpu"), peft_init_method="xavier")
    for name, parameter in model.named_parameters():
        assert parameter.device.type == "cpu" and torch.isfinite(parameter).all(), name
        if name in adapter_names:
            assert parameter.dtype == adapter_dtype, name
        elif "_fp32_params" in name or name.startswith("lm_head."):
            assert parameter.dtype == torch.float32, name
        else:
            assert parameter.dtype == bulk_dtype, name


def test_initializer_rejects_bulk_dtype_change_after_peft_installation() -> None:
    """Fail before mutating foreign adapter or FSDP policy-bearing parameters."""
    model = FlintForCausalLM(_model_config())
    apply_lora_to_linear_modules(
        model,
        PeftConfig(target_modules=["model.layers.*.self_attn.*_proj"], lora_dtype=torch.float32),
    )
    before = {name: parameter.dtype for name, parameter in model.named_parameters()}
    with pytest.raises(ValueError, match="bulk dtype before installing PEFT"):
        model.initialize_weights(dtype=torch.bfloat16)
    assert {name: parameter.dtype for name, parameter in model.named_parameters()} == before


def test_public_peft_load_train_real_checkpoint_reload(tmp_path: Path) -> None:
    """Exercise public base loading and actual adapter-file persistence on CPU."""
    torch.manual_seed(815)
    source_model = FlintForCausalLM(_model_config())
    source = tmp_path / "source"
    source.mkdir()
    source_model.config.save_pretrained(source)
    save_file(source_model.state_dict_adapter.to_hf(source_model.state_dict()), source / "model.safetensors")
    peft = PeftConfig(
        target_modules=["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.experts"],
        dim=4,
        alpha=8,
        lora_dtype=torch.float32,
        use_memory_efficient_lora=False,
    )
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
        gate_precision=None,
    )

    def load() -> FlintForCausalLM:
        with cpu_loader():
            return NeMoAutoModelForCausalLM.from_config(
                str(source),
                load_base_model=True,
                peft_config=peft,
                backend=backend,
                dtype=torch.float32,
                attn_implementation="eager",
                use_liger_kernel=False,
                use_sdpa_patching=False,
                local_files_only=True,
            )

    model = load()
    ids = torch.tensor([[1, 3, 5, 7, 9]])
    torch.testing.assert_close(model(ids).logits, source_model(ids).logits, rtol=0, atol=0)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    model(ids, labels=ids).loss.backward()
    optimizer.step()
    expected = model(ids).logits.detach()
    checkpointer = CheckpointingConfig(
        checkpoint_dir=tmp_path / "checkpoints",
        model_cache_dir=str(tmp_path),
        model_repo_id=str(source),
        model_save_format="safetensors",
        save_consolidated=False,
        is_peft=True,
        v4_compatible=False,
    ).build(dp_rank=0, tp_rank=0, pp_rank=0)
    try:
        checkpointer.save_model(model, str(tmp_path / "adapter"), peft_config=peft)
        restored = load()
        checkpointer.load_model(restored, str(tmp_path / "adapter/model"))
    finally:
        checkpointer.close()
    torch.testing.assert_close(restored(ids).logits, expected, rtol=0, atol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


def test_right_padding_mask_fix_preserves_valid_ce_and_every_gradient(tmp_path):
    """Masked future queries do not change CE on real causal-prefix targets.

    This is an actual small FP32 eager Flint dense/MoE execution with zero
    auxiliary loss and unlimited Torch dispatch. It does not certify BF16/TE
    kernel equality or the historical full49-model trajectory after padding
    work and grouped GEMM row counts change.
    """
    torch.manual_seed(437578)
    config = FlintConfig(
        vocab_size=16,
        hidden_size=8,
        intermediate_size=12,
        moe_intermediate_size=4,
        shared_expert_intermediate_size=6,
        num_hidden_layers=3,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_experts=4,
        num_experts_per_tok=2,
        mlp_only_layers=[0],
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        no_rope_layer_indices=[0, 2],
        sliding_window=3,
        max_position_embeddings=16,
        router_aux_loss_coef=0.0,
        residual_dtype="fp32",
        pad_token_id=15,
        eos_token_id=2,
        tie_word_embeddings=False,
        dtype=torch.float32,
    )
    model = config.build(
        backend=BackendConfig(
            attn="eager",
            linear="torch",
            rms_norm="torch_fp32",
            rope="torch",
            rope_fusion=False,
            experts="torch",
            dispatcher="torch",
            gate_precision=None,
            enable_fsdp_optimizations=False,
        )
    )
    model.train()
    labels = torch.tensor([[-100, 7, 2, -100, -100, -100, -100, -100]])
    cases = {
        "historical": (
            torch.tensor([[4, 0, 7, 2, 0, 0, 0, 2]]),
            torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]]),
        ),
        "corrected": (
            torch.tensor([[4, 0, 7, 2, 0, 0, 0, 0]]),
            torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]]),
        ),
    }
    results = {}
    for name, (ids, mask) in cases.items():
        model.zero_grad(set_to_none=True)
        logits = model(ids, attention_mask=mask, padding_mask=~mask.bool()).logits
        loss = F.cross_entropy(logits.flatten(0, 1), labels.flatten())
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        loss.backward()
        gradients = {}
        for fqn, parameter in model.named_parameters():
            assert parameter.grad is not None, fqn
            assert torch.isfinite(parameter.grad).all(), fqn
            gradients[fqn] = parameter.grad.detach().clone()
        results[name] = dict(logits=logits.detach(), loss=loss.detach(), gradients=gradients)
    old, new = results["historical"], results["corrected"]
    torch.testing.assert_close(new["logits"][:, :4], old["logits"][:, :4], atol=1e-7, rtol=1e-6)
    torch.testing.assert_close(new["loss"], old["loss"], atol=1e-7, rtol=1e-6)
    assert new["gradients"].keys() == old["gradients"].keys()
    for fqn, gradient in new["gradients"].items():
        torch.testing.assert_close(gradient, old["gradients"][fqn], atol=2e-6, rtol=1e-5, msg=fqn)
    receipt = {
        "status": "passed",
        "scope": "Small actual FP32 eager Flint dense/MoE causal CE, aux0, unlimited Torch dispatch",
        "valid_prefix_logit_max_abs": float((new["logits"][:, :4] - old["logits"][:, :4]).abs().max()),
        "losses": {name: float(result["loss"]) for name, result in results.items()},
        "every_parameter_gradient_max_abs": max(
            float((gradient - old["gradients"][fqn]).abs().max()) for fqn, gradient in new["gradients"].items()
        ),
        "parameter_gradient_count": len(new["gradients"]),
        "supervised_targets": int((labels != -100).sum()),
        "historical_vs_corrected_attention_valid_counts": [7, 4],
        "limitations": "Does not establish BF16/TE full49 trajectory equality; padding/expert GEMM workloads change",
    }
    path = Path(os.environ.get("FLINT_PADDING_IMPACT_EVIDENCE", tmp_path / "padding-impact.json"))
    path.write_text(json.dumps(receipt, indent=2) + "\n")
