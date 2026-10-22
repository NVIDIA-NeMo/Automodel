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

import copy
import gc
import math
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from torch.utils.checkpoint import noop_context_fn
from transformers import AutoConfig

from nemo_automodel._transformers.registry import resolve_custom_config_cls
from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater
from nemo_automodel.components.datasets.vlm.neat_packing_vlm import _build_packed_vlm_sample, _shift_sample
from nemo_automodel.components.distributed import parallelizer_utils
from nemo_automodel.components.distributed.activation_checkpointing import make_selective_checkpoint_context_fn
from nemo_automodel.components.distributed.model_parallelizer import get_model_parallelizer
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.packing import PackingCapabilities, configure_packing_for_models
from nemo_automodel.components.models.common.utils import cast_frozen_modules_to_compute_dtype, cast_model_to_dtype
from nemo_automodel.components.models.step5 import parallelization as parallelization_module
from nemo_automodel.components.models.step5.configuration import ConfigValue, Step5Config, Step5VLConfig
from nemo_automodel.components.models.step5.flops import step5_flops
from nemo_automodel.components.models.step5.layers import region_replay
from nemo_automodel.components.models.step5.model import Step5ForCausalLM, Step5ForConditionalGeneration
from nemo_automodel.components.models.step5.rope import Step5RotaryEmbedding
from nemo_automodel.components.moe import parallelizer as moe_parallelizer
from nemo_automodel.components.moe.parallelizer import _with_model_checkpoint_context
from nemo_automodel.components.optim.optimizer import _trainable_params_or_groups
from nemo_automodel.components.utils.flops_utils import get_flops_formula_for_hf_config


def tiny_config(**overrides: ConfigValue) -> Step5Config:
    values = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        dtype="float32",
        num_nextn_predict_layers=0,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=2,
        ),
    )
    values.update(overrides)
    return Step5Config(**values)


def make_model(**overrides: ConfigValue) -> Step5ForCausalLM:
    torch.manual_seed(137)
    return Step5ForCausalLM(
        tiny_config(**overrides),
        backend=BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm"),
    )


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the shared grouped-expert activation backward")
def test_tiny_forward_backward():
    model = make_model()
    inputs = torch.arange(16).unsqueeze(0)
    output = model(inputs, labels=inputs)
    assert output.logits.shape == (1, 16, 64)
    assert torch.isfinite(output.logits).all()
    output.loss.backward()
    for name, parameter in model.named_parameters():
        if ".indexer." in name:
            assert parameter.grad is None  # Discrete top-k has no LM-loss derivative.
        else:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name
    assert model.model.layers["3"].moe.gate.weight.grad.abs().sum() > 0


def test_rope_independent_oracle_and_wrong_layout():
    cfg = tiny_config()
    rope = Step5RotaryEmbedding(cfg, 0, span=4)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]]])
    actual = rope(x, torch.tensor([[3]]))
    a, b = 3.0, 3.0 / math.sqrt(cfg.rope_theta[0])
    expected = torch.tensor(
        [
            [
                [
                    [
                        math.cos(a) - 3 * math.sin(a),
                        2 * math.cos(b) - 4 * math.sin(b),
                        math.sin(a) + 3 * math.cos(a),
                        2 * math.sin(b) + 4 * math.cos(b),
                        5.0,
                        6.0,
                    ]
                ]
            ]
        ]
    )
    torch.testing.assert_close(actual, expected)
    interleaved = torch.tensor(
        [
            [
                [
                    [
                        math.cos(a) - 2 * math.sin(a),
                        math.sin(a) + 2 * math.cos(a),
                        3 * math.cos(b) - 4 * math.sin(b),
                        3 * math.sin(b) + 4 * math.cos(b),
                        5.0,
                        6.0,
                    ]
                ]
            ]
        ]
    )
    assert not torch.allclose(actual, interleaved)


def test_cast_preserves_original_precision():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    with torch.no_grad():
        indexer.w_proj.weight.fill_(1.0031)
        model.model.layers["3"].moe.gate.e_score_correction_bias.fill_(34.03125)
    original = indexer.w_proj.weight.detach().clone()
    cast_model_to_dtype(model, torch.bfloat16)
    assert indexer.w_proj.weight.dtype == torch.float32
    torch.testing.assert_close(indexer.w_proj.weight, original, rtol=0, atol=0)
    bias = model.model.layers["3"].moe.gate.e_score_correction_bias
    assert bias.dtype == torch.float32
    assert torch.equal(bias, torch.full_like(bias, 34.03125))


def test_fp32_storage_under_bf16_default_dtype() -> None:
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        model = make_model(dtype="bfloat16")
    finally:
        torch.set_default_dtype(previous)
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in model._keep_in_fp32_modules_strict):
            assert parameter.dtype == torch.float32, name
    assert model.model.embed_tokens.weight.dtype == torch.bfloat16


def test_adapter_roundtrip_and_directly_loaded_model():
    """Component conversion check; native distributed loading uses the recipe CLI."""
    model = make_model()
    adapter = model.state_dict_adapter
    native = model.state_dict()
    checkpoint = adapter.to_hf(native)
    restored = adapter.from_hf(checkpoint)
    assert restored.keys() == native.keys()
    for name in native:
        torch.testing.assert_close(restored[name], native[name], rtol=0, atol=0)
    other = make_model()
    other.load_state_dict(restored, strict=True)
    inputs = torch.arange(16).unsqueeze(0)
    torch.testing.assert_close(model(inputs).logits, other(inputs).logits, rtol=0, atol=0)


def test_sparse_selection_drops_history_and_breaks_ties():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    for parameter in indexer.parameters():
        torch.nn.init.zeros_(parameter)
    positions = torch.arange(12).unsqueeze(0)
    hidden = torch.zeros(1, 12, 32)
    keys, logits = indexer.project_keys(hidden, positions)
    selected = indexer(hidden, positions, keys, logits, positions, torch.zeros_like(positions, dtype=torch.bool))
    # At q=6 there are three complete history blocks, but only two history slots.
    # Equal scores retain blocks 0 and 1; current block 3 is appended, and 2 drops.
    assert selected.region_ids[0, 0, 6].tolist() == [0, 1, 3]
    assert selected.region_ids[0, 0, 4].tolist() == [0, 1, 2]
    assert selected.region_ids[0, 0, 0].tolist() == [0, 6, 6]


def test_packed_documents_do_not_leak():
    model = make_model().eval()
    ids = torch.arange(16).unsqueeze(0)
    positions = torch.arange(8).repeat(2).unsqueeze(0)
    packed = model(ids, position_ids=positions).logits
    separate = torch.cat([model(ids[:, :8]).logits, model(ids[:, 8:]).logits], dim=1)
    torch.testing.assert_close(packed, separate, rtol=1e-5, atol=1e-6)
    changed = ids.clone()
    changed[:, :8] = 63
    # Changing the first document alters MoE expert token counts, so GEMM shapes and FP32 rounding differ.
    torch.testing.assert_close(
        model(changed, position_ids=positions).logits[:, 8:], packed[:, 8:], rtol=1e-5, atol=1e-6
    )


def test_local_auto_config(tmp_path: Path) -> None:
    assert resolve_custom_config_cls("step5") is Step5Config
    cfg = tiny_config()
    cfg.save_pretrained(tmp_path)
    loaded = AutoConfig.from_pretrained(tmp_path)
    assert isinstance(loaded, Step5Config)
    assert loaded.sparse_config.topk == 2


def test_unsupported_sparse_semantics_fail():
    with pytest.raises(ValueError, match="Unsupported Step5 sparse semantics"):
        tiny_config(sparse_config={"compression_method": "mean"})


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the VLM shared-expert activation and backward")
def test_vlm_image_path():
    text = tiny_config(num_hidden_layers=1, moe_layer_list=[])
    config = Step5VLConfig(
        text_config=text,
        vision_config=dict(width=16, layers=1, heads=2, image_size=28, patch_size=14, mlp_ratio=2),
        image_token_id=63,
    )
    model = Step5ForConditionalGeneration(
        config, backend=BackendConfig(attn="sdpa", linear="torch", dispatcher="torch", experts="torch_mm")
    )
    tokens = torch.tensor([[0, 63, 1, 2]])
    pixels = torch.randn(1, 3, 28, 28)
    output = model(tokens, pixel_values=pixels)
    assert output.logits.shape == (1, 4, 64)
    output.logits.square().mean().backward()
    assert model.vision_model.conv1.weight.grad.abs().sum() > 0


def test_post_silu_clamp_is_not_pre_silu_clamp():
    from nemo_automodel.components.moe.experts import swiglu_clamped_deepep

    gate = torch.tensor([[6.5, 7.5, 9.0, -1.0]], requires_grad=True)
    up = torch.tensor([[8.0, -9.0, 2.0, 3.0]], requires_grad=True)
    packed = torch.cat((gate, up), -1)
    actual = swiglu_clamped_deepep(packed, torch.ones(1, 1), 7.0, clamp_after_activation=True)
    expected = torch.nn.functional.silu(gate).clamp(max=7) * up.clamp(-7, 7)
    wrong = swiglu_clamped_deepep(packed, torch.ones(1, 1), 7.0, clamp_after_activation=False)
    torch.testing.assert_close(actual, expected)
    assert not torch.allclose(actual, wrong, rtol=1e-5, atol=1e-5)
    actual_grad = torch.autograd.grad(actual.sum(), (gate, up), retain_graph=True)
    expected_grad = torch.autograd.grad(expected.sum(), (gate, up))
    for a, e in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, e)


@pytest.mark.parametrize(
    "excluded",
    [
        ["model.layers.3.moe.experts.w13_weight", "model.layers.3.moe.experts.w2_weight"],
        ["model.layers.3.moe.gate_proj", "model.layers.3.moe.up_proj", "model.layers.3.moe.down_proj"],
        ["model.layers.3.moe"],
    ],
)
def test_checkpoint_parts_preserve_unquantized_expert_weights(excluded):
    model = make_model(
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": excluded,
        }
    )
    adapter = model.state_dict_adapter
    native = {
        "model.layers.3.moe.experts.gate_and_up_projs": torch.zeros(4, 32, 32, dtype=torch.bfloat16),
        "model.layers.3.moe.experts.down_projs": torch.zeros(4, 16, 32, dtype=torch.bfloat16),
    }
    source = {
        "model.layers.3.moe.gate_proj.weight": (torch.arange(4 * 16 * 32).reshape(4, 16, 32) / 256).bfloat16(),
        "model.layers.3.moe.up_proj.weight": torch.full((4, 16, 32), 0.375, dtype=torch.bfloat16),
        "model.layers.3.moe.down_proj.weight": torch.full((4, 32, 16), -0.625, dtype=torch.bfloat16),
    }
    for part in adapter.iter_checkpoint_load_parts(native):
        assert set(part.checkpoint_tensors) <= source.keys()
        assert not part.temporary_checkpoint_keys
        for name, target in part.checkpoint_tensors.items():
            assert target.dtype == torch.bfloat16
            target.copy_(source[name])
        part.finish()
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.gate_and_up_projs"][..., :16],
        source["model.layers.3.moe.gate_proj.weight"].transpose(1, 2),
    )
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.gate_and_up_projs"][..., 16:],
        source["model.layers.3.moe.up_proj.weight"].transpose(1, 2),
    )
    torch.testing.assert_close(
        native["model.layers.3.moe.experts.down_projs"], source["model.layers.3.moe.down_proj.weight"].transpose(1, 2)
    )


@pytest.mark.parametrize("bf16_gate", [False, True])
def test_checkpoint_parts_decode_fp8_with_optional_bf16_gate(bf16_gate):
    excluded = ["model.layers.3.moe.gate_proj"] if bf16_gate else []
    adapter = make_model(
        quantization_config={
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": excluded,
        }
    ).state_dict_adapter
    name = "model.layers.3.moe.experts.gate_and_up_projs"
    destination = torch.zeros(2, 128, 256, dtype=torch.bfloat16)
    parts = list(adapter.iter_checkpoint_load_parts({name: destination}))
    assert len(parts) == 1
    part = parts[0]
    for source_name, tensor in part.checkpoint_tensors.items():
        if source_name.endswith("_scale_inv"):
            tensor.fill_(0.25)
        elif bf16_gate and source_name.endswith("gate_proj.weight"):
            assert tensor.dtype == torch.bfloat16
            tensor.fill_(0.375)
            assert source_name not in part.temporary_checkpoint_keys
        else:
            assert tensor.dtype == torch.float8_e4m3fn
            tensor.copy_(torch.full(tensor.shape, 2.0).to(tensor.dtype))
    part.finish()
    torch.testing.assert_close(
        destination[..., :128], torch.full((2, 128, 128), 0.375 if bf16_gate else 0.5, dtype=torch.bfloat16)
    )
    torch.testing.assert_close(destination[..., 128:], torch.full((2, 128, 128), 0.5, dtype=torch.bfloat16))


def test_checkpoint_parts_without_quantization_do_not_request_scales():
    adapter = make_model().state_dict_adapter
    destination = torch.zeros(4, 16, 32, dtype=torch.bfloat16)
    part = next(adapter.iter_checkpoint_load_parts({"model.layers.3.moe.experts.down_projs": destination}))
    assert set(part.checkpoint_tensors) == {"model.layers.3.moe.down_proj.weight"}
    assert not part.temporary_checkpoint_keys
    part.checkpoint_tensors["model.layers.3.moe.down_proj.weight"].fill_(0.375)
    part.finish()
    torch.testing.assert_close(destination, torch.full_like(destination, 0.375))


@pytest.mark.parametrize(
    "weight_name",
    [
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
        "model.layers.3.moe.experts.down_projs",
    ],
)
def test_checkpoint_parts_preserve_backend_extra_state(weight_name: str) -> None:
    """Component load-plan check; distributed loading is validated through the native CLI."""
    model = make_model()
    adapter = model.state_dict_adapter
    expected = model.state_dict()[weight_name].clone()
    source = adapter.to_hf({weight_name: expected})
    extra_state = torch.tensor([3, 1, 4], dtype=torch.uint8)
    native = {
        "model.layers.0.self_attn.qkv_proj._extra_state": extra_state.clone(),
        weight_name: torch.zeros_like(expected),
        "lm_head._extra_state": extra_state.clone(),
    }
    completed = set()
    for part in adapter.iter_checkpoint_load_parts(native):
        assert part.checkpoint_tensors
        assert set(part.checkpoint_tensors) <= source.keys()
        assert not completed.intersection(part.model_keys)
        completed.update(part.model_keys)
        for name, target in part.checkpoint_tensors.items():
            target.copy_(source[name])
        part.finish()
    assert completed == native.keys()
    torch.testing.assert_close(native[weight_name], expected, rtol=0, atol=0)
    for name in ("model.layers.0.self_attn.qkv_proj._extra_state", "lm_head._extra_state"):
        torch.testing.assert_close(native[name], extra_state, rtol=0, atol=0)


@pytest.mark.parametrize("first_length,padding", [(1, 0), (3, 0), (3, 2)])
@pytest.mark.parametrize("check", ["loss", "gradient"])
def test_packed_shifted_loss_does_not_cross_documents(first_length: int, padding: int, check: str) -> None:
    """A later document's first label must not train the preceding document."""
    model = make_model(num_hidden_layers=1, moe_layer_list=[])
    total = first_length + padding + 4
    start = first_length + padding
    inputs = torch.arange(total)[None]
    positions = torch.cat((torch.arange(first_length), torch.zeros(padding, dtype=torch.long), torch.arange(4)))[None]
    documents = torch.cat(
        (torch.ones(first_length, dtype=torch.long), torch.zeros(padding, dtype=torch.long), torch.full((4,), 2))
    )[None]
    labels = inputs.clone()
    labels[:, :start] = -100
    embeddings = model.get_input_embeddings()(inputs).detach().requires_grad_()
    actual = model(
        inputs_embeds=embeddings,
        labels=labels,
        position_ids=positions,
        attention_mask=documents,
        _packed_seq_ids=documents,
    )
    if check == "loss":
        expected = model(inputs[:, start:], labels=inputs[:, start:]).loss
        torch.testing.assert_close(actual.loss, expected, rtol=1e-6, atol=1e-6)
    else:
        actual.loss.backward()
        assert torch.count_nonzero(embeddings.grad[:, :start]) == 0
        assert embeddings.grad[:, start:].abs().sum() > 0


@pytest.mark.parametrize("pipeline_stage", [False, True])
def test_distributed_pipeline_stage_uses_ordinary_checkpoint_load(monkeypatch, pipeline_stage):
    adapter = make_model().state_dict_adapter
    layers = range(adapter.config.num_hidden_layers)
    owned = list(layers)[1:] if pipeline_stage else list(layers)
    native = {f"model.layers.{index}.input_layernorm.weight": torch.ones(64) for index in owned}
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 2)
    # Each part is a collective load; stages owning different layers would issue different counts.
    assert (adapter.iter_checkpoint_load_parts(native) is None) is pipeline_stage


def test_native_vlm_packing_contract() -> None:
    text = Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=192,
        vocab_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        num_nextn_predict_layers=0,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=8,
        ),
    )
    with torch.device("meta"):
        model = Step5ForConditionalGeneration(
            Step5VLConfig(
                text_config=text,
                vision_config=dict(width=16, layers=1, heads=2, image_size=112, patch_size=14, mlp_ratio=2),
                image_token_id=127,
            ),
            backend=BackendConfig(attn="fa4", linear="torch", dispatcher="torch", experts="torch_mm"),
        )
    packing = configure_packing_for_models([model])
    assert packing.packed_mask_type == "document_ids"
    assert not packing.requires_packed_sequence_metadata  # Model uses reset positions directly.


def test_native_vlm_packing_preserves_crops() -> None:
    samples = []
    for index, count in enumerate((2, 0, 1)):
        ids = torch.arange(13 + index)
        sample = dict(
            input_ids=ids,
            labels=ids.clone(),
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.full((1, 3, 112, 112), float(index)),
            num_patches=torch.tensor([count]),
        )
        if count:
            sample["patch_pixel_values"] = torch.full((count, 3, 56, 56), float(index + 10))
        samples.append(sample)
    shifted = [_shift_sample(sample) for sample in samples]
    packed = _build_packed_vlm_sample(shifted, 64, 0)
    batch = neat_packed_vlm_collater([packed], packing=PackingCapabilities(packed_mask_type="document_ids"))
    assert "patch_pixel_values" in batch
    assert "num_patches" in batch
    assert batch["num_patches"].tolist() == [2, 0, 1]
    torch.testing.assert_close(batch["pixel_values"][:, 0, 0, 0].float(), torch.tensor([0.0, 1.0, 2.0]))
    torch.testing.assert_close(batch["patch_pixel_values"][:, 0, 0, 0].float(), torch.tensor([10.0, 10.0, 12.0]))
    offset = 0
    for sample in samples:
        length = sample["input_ids"].numel() - 1
        assert torch.equal(batch["labels"][0, offset : offset + length], sample["labels"][1:])
        assert torch.equal(batch["position_ids"][0, offset : offset + length], torch.arange(length))
        offset += length
    assert batch["attention_mask"].ndim == 2


@pytest.mark.runtime_budget(20, reason="Four-layer vision/text eager oracle and independent image-gradient isolation")
def test_vlm_packed_fp32_independent_documents_and_gradients() -> None:
    """Packing must match independent documents and isolate image gradients."""
    torch.manual_seed(197)
    text = Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=128,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[],
        dtype="float32",
        num_nextn_predict_layers=0,
        sliding_window=13,
        sparse_config=dict(
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=8,
        ),
    )
    model = Step5ForConditionalGeneration(
        Step5VLConfig(
            text_config=text,
            vision_config=dict(width=16, layers=1, heads=2, image_size=112, patch_size=14, mlp_ratio=2),
            image_token_id=127,
        ),
        backend=BackendConfig(attn="eager", linear="torch", dispatcher="torch", experts="torch_mm"),
    )
    samples = []
    for length, crops in ((19, 1), (83, 0)):
        ids = torch.randint(1, 126, (length + 1,))
        ids[3 : 7 + crops] = 127
        labels = ids.clone()
        labels[labels == 127] = -100
        item = dict(
            input_ids=ids,
            labels=labels,
            attention_mask=torch.ones_like(ids),
            pixel_values=torch.randn(1, 3, 112, 112),
            num_patches=torch.tensor([crops]),
        )
        if crops:
            item["patch_pixel_values"] = torch.randn(crops, 3, 56, 56)
        samples.append(item)
    packing = configure_packing_for_models([model])

    def batch_for(documents):
        return neat_packed_vlm_collater(
            [_build_packed_vlm_sample([_shift_sample(s) for s in documents], 128, 0)], packing=packing
        )

    def evaluate(batch):
        return model(**{k: v for k, v in batch.items() if k != "labels"}).logits

    batch = batch_for(samples)
    batch["pixel_values"] = batch["pixel_values"].float().requires_grad_()
    batch["patch_pixel_values"] = batch["patch_pixel_values"].float().requires_grad_()
    packed = evaluate(batch)
    separate = torch.cat([evaluate(batch_for([s])) for s in samples], dim=1)
    torch.testing.assert_close(packed, separate, rtol=2e-5, atol=2e-6)
    packed[:, 19:].square().sum().backward()
    assert torch.count_nonzero(batch["pixel_values"].grad[0]) == 0
    assert torch.count_nonzero(batch["patch_pixel_values"].grad) == 0
    assert batch["pixel_values"].grad[1].abs().sum() > 0
    with torch.no_grad():
        changed = dict(batch)
        changed["pixel_values"] = batch["pixel_values"].detach().clone()
        changed["pixel_values"][0].mul_(-3)
        changed["patch_pixel_values"] = -batch["patch_pixel_values"].detach()
        torch.testing.assert_close(evaluate(changed)[:, 19:], packed[:, 19:], rtol=0, atol=0)
        missing = dict(batch)
        missing.pop("position_ids")
        with pytest.raises(ValueError, match="packed"):
            evaluate(missing)
        wrong = dict(batch)
        wrong["position_ids"] = torch.arange(102)[None]
        with pytest.raises(ValueError, match="document"):
            evaluate(wrong)
        duplicate = dict(batch)
        duplicate["position_ids"] = batch["position_ids"].clone()
        duplicate["position_ids"][0, 21] = duplicate["position_ids"][0, 20]
        with pytest.raises(ValueError, match="document"):
            evaluate(duplicate)
        # Removing both independent boundary sources simulates an upstream bug.
        wrong.pop("_packed_seq_ids")
        wrong["attention_mask"] = wrong["attention_mask"].bool()
        assert (evaluate(wrong)[:, 19:] - packed[:, 19:]).abs().max() > 1e-4


@pytest.mark.runtime_budget(30, reason="Cold CPU compilation of the shared grouped-expert activation backward")
@pytest.mark.parametrize("split", [1, 2, 3])
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("packed", [False, True])
def test_pipeline_stage_forward_backward_parity(split: int, dtype: str, packed: bool) -> None:
    """Splitting stages must preserve logits, every gradient, and an SGD update."""
    reference = make_model(dtype=dtype, moe_layer_list=[1, 3])
    first, last = copy.deepcopy(reference), copy.deepcopy(reference)
    first.model.layers = nn.ModuleDict({k: v for k, v in first.model.layers.items() if int(k) < split})
    first.model.norm = None
    first.lm_head = None
    last.model.layers = nn.ModuleDict({k: v for k, v in last.model.layers.items() if int(k) >= split})
    last.model.embed_tokens = None

    tokens = torch.arange(24).reshape(2, 12)
    positions = torch.arange(12).expand(2, -1)
    documents = None
    if packed:
        positions = torch.tensor([[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]]).expand(2, -1)
        documents = torch.tensor([[1] * 5 + [2] * 7]).expand(2, -1)
    kwargs = {"position_ids": positions, "_packed_seq_ids": documents}
    expected = reference(tokens, **kwargs).logits
    residual = first(tokens, **kwargs).logits
    assert residual.dtype == torch.float32
    actual = last(residual, **kwargs).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(expected)
    (expected * upstream).float().sum().backward()
    (actual * upstream).float().sum().backward()
    reference_parameters = dict(reference.named_parameters())
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            expected_parameter = reference_parameters[name]
            if expected_parameter.grad is None:
                assert parameter.grad is None, name
            else:
                torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0, msg=name)
    for model in (reference, first, last):
        torch.optim.SGD(model.parameters(), lr=0.01).step()
    for stage in (first, last):
        for name, parameter in stage.named_parameters():
            torch.testing.assert_close(parameter, reference_parameters[name], rtol=0, atol=0, msg=name)


def test_pipeline_metadata_and_fused_loss() -> None:
    """Intermediate buffers are FP32 while final fused-loss output stays BF16."""
    model = make_model(dtype="bfloat16")
    first, last = copy.deepcopy(model), copy.deepcopy(model)
    first.lm_head = None
    first.model.norm = None
    last.model.embed_tokens = None
    inputs, outputs = first.get_pipeline_stage_metas(is_first=True, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    assert inputs[0].shape == (2, 12) and inputs[0].dtype == torch.long
    assert outputs[0].shape == (2, 12, 32) and outputs[0].dtype == torch.float32
    inputs, outputs = last.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    assert inputs[0].shape == (2, 12, 32) and inputs[0].dtype == torch.float32
    assert outputs[0].shape == (2, 12, 64) and outputs[0].dtype == torch.bfloat16
    last._pp_return_hidden_states = True
    _, outputs = last.get_pipeline_stage_metas(is_first=False, microbatch_size=2, seq_len=12, dtype=torch.bfloat16)
    hidden = last(torch.randn(2, 12, 32, dtype=torch.float32))
    assert outputs[0].shape == hidden.shape == (2, 12, 32)
    assert outputs[0].dtype == hidden.dtype == torch.bfloat16
    assert first._is_pipeline_parallel_stage()
    assert last._is_pipeline_parallel_stage()
    assert not model._is_pipeline_parallel_stage()
    assert model.ModelCapabilities.supports_pp
    assert not Step5ForConditionalGeneration.ModelCapabilities.supports_pp


def test_pipeline_rejects_rounded_residual() -> None:
    """A BF16 carry must fail instead of silently losing residual precision."""
    stage = make_model(dtype="bfloat16")
    stage.model.embed_tokens = None
    with pytest.raises(ValueError, match="require FP32 residuals"):
        stage(torch.randn(1, 12, 32, dtype=torch.bfloat16))


def _record_sharding(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[nn.Module, set[nn.Parameter], MixedPrecisionPolicy]]:
    """Replace fully_shard with a recorder of each unit's newly owned parameters."""
    calls = []

    def shard(module, *, mesh, mp_policy, ignored_params=None, **kwargs):
        assert not isinstance(module, FSDPModule), "An existing FSDP unit was wrapped twice"
        owned = set(module.parameters()) - set(ignored_params or ())
        for child in module.modules():
            if child is not module and isinstance(child, FSDPModule):
                owned.difference_update(child.parameters())
        calls.append((module, owned, mp_policy))
        module.__class__ = type(f"TestFSDP{type(module).__name__}", (FSDPModule, type(module)), {})
        return module

    for target in (parallelizer_utils, moe_parallelizer, parallelization_module):
        monkeypatch.setattr(target, "fully_shard", shard)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_accumulated_grad_guard", lambda: None)
    monkeypatch.setattr(moe_parallelizer, "_patch_fsdp_uniform_reduce_dtype", lambda: None)
    # The recorder has no real mesh; keep router correction-bias buffers local.
    monkeypatch.setattr(moe_parallelizer, "distribute_tensor", lambda tensor, **kwargs: tensor)
    return calls


def test_step5_uses_model_owned_parallelizer():
    assert Step5ForCausalLM.parallelizer is parallelization_module.PARALLELIZER
    assert get_model_parallelizer(make_model()) is parallelization_module.PARALLELIZER


@pytest.mark.parametrize("model_owned", [False, True])
def test_final_norm_computes_in_fp32_only_with_step5_parallelizer(monkeypatch, model_owned):
    model = make_model()
    norm_weight = model.model.norm.weight
    assert norm_weight.dtype == torch.float32
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False)
    calls = _record_sharding(monkeypatch)
    moe_parallelizer.apply_fsdp(
        model,
        fsdp_mesh=SimpleNamespace(ndim=1),
        ep_enabled=False,
        ep_shard_enabled=False,
        mp_policy=policy,
        reshard_after_forward=True,
        model_parallelizer=parallelization_module.PARALLELIZER if model_owned else None,
    )
    owners = [(module, mp) for module, params, mp in calls if norm_weight in params]
    assert len(owners) == 1
    owner, owner_policy = owners[0]
    if model_owned:
        assert owner is model.model.norm
        assert owner_policy.param_dtype is torch.float32 and owner_policy.cast_forward_inputs is False
    else:
        # Upstream's root unit computes the strict-FP32 final norm in BF16.
        assert owner is model.model and owner_policy.param_dtype is torch.bfloat16
    assert sum(module is model.model for module, _, _ in calls) == 1


def test_pipeline_stage_without_final_norm_wraps_root_once(monkeypatch):
    model = make_model()
    model.model.norm = None
    calls = _record_sharding(monkeypatch)
    parallelization_module.PARALLELIZER._fully_shard_module(
        model.model, mesh=object(), mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    )
    assert [module for module, _, _ in calls] == [model.model]


def test_outer_root_wraps_contained_final_norm_once(monkeypatch):
    model = make_model()
    calls = _record_sharding(monkeypatch)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    parallelization_module.PARALLELIZER._fully_shard_module(model, mesh=object(), mp_policy=policy)
    assert [module for module, _, _ in calls] == [model.model.norm, model]
    assert calls[0][2].param_dtype is torch.float32
    assert model.model.norm.weight not in calls[1][1]


def _config(*, sparse_enabled: bool = True) -> Step5Config:
    return Step5Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_attention_groups=2,
        head_dim=12,
        vocab_size=64,
        moe_num_experts=4,
        moe_top_k=2,
        moe_intermediate_size=16,
        share_expert_dim=16,
        moe_layer_list=[3],
        sliding_window=3,
        num_nextn_predict_layers=0,
        sparse_config=dict(
            enabled=sparse_enabled,
            proxy_dim=16,
            sparse_indexer_rope_dim=4,
            sparse_indexer_num_heads=4,
            num_provider_groups=2,
            topk=2,
            region_block_size=2,
        ),
    )


def _enumerated_flops(config: Step5Config, length: int, supervised: int) -> int:
    """Enumerate matrix products and actual selected keys, without production count helpers."""
    total = 6 * supervised * config.hidden_size * config.vocab_size
    sparse = config.sparse_config
    for layer in range(config.num_hidden_layers):
        shapes = [
            (config.hidden_size, config.num_attention_heads * config.head_dim),
            (config.hidden_size, config.num_attention_groups * config.head_dim),
            (config.hidden_size, config.num_attention_groups * config.head_dim),
            (config.num_attention_heads * config.head_dim, config.hidden_size),
            (config.hidden_size, config.num_attention_heads),
        ]
        widths = (
            [config.moe_intermediate_size] * config.moe_top_k + [config.share_expert_dim]
            if layer in config.moe_layer_list
            else [config.intermediate_size]
        )
        if layer in config.moe_layer_list:
            shapes.append((config.hidden_size, config.moe_num_experts))
        for width in widths:
            shapes.extend([(config.hidden_size, width), (config.hidden_size, width), (width, config.hidden_size)])
        total += sum(6 * length * rows * columns for rows, columns in shapes)
        sparse_layer = sparse.enabled and config.layer_types[layer] == "full_attention"
        if sparse_layer:
            index_shapes = [
                (config.hidden_size, sparse.sparse_indexer_num_heads * sparse.proxy_dim),
                (config.hidden_size, sparse.proxy_dim),
                (config.hidden_size, sparse.proxy_dim),
                (config.hidden_size, sparse.sparse_indexer_num_heads),
            ]
            total += sum(2 * length * rows * columns for rows, columns in index_shapes)
        for query in range(length):
            causal = list(range(query + 1))
            if config.layer_types[layer] == "sliding_attention":
                keys = [key for key in causal if query - key < config.sliding_window]
            elif sparse_layer:
                historical_regions = sorted(
                    {
                        key // sparse.region_block_size
                        for key in causal
                        if key // sparse.region_block_size < query // sparse.region_block_size
                    }
                )
                selected = historical_regions[-sparse.topk :] + [query // sparse.region_block_size]
                keys = [key for key in causal if key // sparse.region_block_size in selected]
                total += 2 * len(historical_regions) * sparse.sparse_indexer_num_heads * sparse.proxy_dim
            else:
                keys = causal
            total += 2 * 6 * len(keys) * config.num_attention_heads * config.head_dim
    return total


@pytest.mark.parametrize("sparse_enabled", [False, True])
@pytest.mark.parametrize("length", [0, 1, 2, 3, 4, 5, 6, 7, 11, 17])
def test_shape_formula_matches_independently_enumerated_operations(length: int, sparse_enabled: bool) -> None:
    cfg = _config(sparse_enabled=sparse_enabled)
    assert step5_flops(cfg, gbs=3, seq_len=length) == 3 * _enumerated_flops(cfg, length, length)


def test_registry_uses_the_model_owned_formula() -> None:
    assert get_flops_formula_for_hf_config(_config()) is step5_flops


def test_full_checkpoint_geometry_long_single_document_count() -> None:
    cfg = Step5Config(num_nextn_predict_layers=0)
    assert cfg.hidden_size == 4096 and cfg.num_attention_heads == 64
    assert step5_flops(cfg, gbs=8, seq_len=131072) == 186839187784204288.0


def test_indexer_constructor_is_frozen_without_changing_registered_state():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    assert set(dict(indexer.named_parameters())) == {
        "q_proj.weight",
        "k_proj.weight",
        "z_proj.weight",
        "w_proj.weight",
        "q_norm.weight",
        "k_norm.weight",
        "k_norm.bias",
    }
    assert all(not parameter.requires_grad for parameter in indexer.parameters())
    assert all(parameter.requires_grad for name, parameter in model.named_parameters() if ".indexer." not in name)
    assert set(dict(indexer.named_parameters())).issubset(indexer.state_dict())
    assert indexer.ssmax_s.dtype == torch.float32
    assert all(parameter.dtype == torch.float32 for parameter in indexer.parameters())


def test_optimizer_exclusion_preserves_indexer_even_with_materialized_zero_gradients():
    model = make_model()
    initial = {name: value.detach().clone() for name, value in model.named_parameters()}
    parameters = _trainable_params_or_groups(model, [])
    optimizer = torch.optim.AdamW(parameters, lr=0.01, weight_decay=0.1)
    indexer = model.model.layers["3"].self_attn.indexer
    assert not set(indexer.parameters()).intersection(parameters)
    for _ in range(2):
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            if ".indexer." in name:
                assert torch.equal(parameter, initial[name]), name
                assert parameter not in optimizer.state
    assert not torch.equal(model.model.layers["3"].moe.gate.weight, initial["model.layers.3.moe.gate.weight"])


def test_frozen_cast_keeps_strict_indexer_fp32_values_and_rotary_contract():
    model = make_model()
    indexer = model.model.layers["3"].self_attn.indexer
    with torch.no_grad():
        indexer.w_proj.weight.fill_(0.1234567)
        indexer.q_norm.weight.fill_(0.2345678)
        indexer.k_norm.weight.fill_(1.1234567)
        indexer.k_norm.bias.fill_(0.3456789)
        indexer.ssmax_s.fill_(0.456789)
    before = {name: value.detach().clone() for name, value in indexer.state_dict().items()}
    cast_frozen_modules_to_compute_dtype(model, torch.bfloat16)
    for name, value in indexer.state_dict().items():
        if name in ("q_proj.weight", "k_proj.weight", "z_proj.weight"):
            assert value.dtype == torch.bfloat16
        else:
            assert value.dtype == before[name].dtype, name
            assert torch.equal(value, before[name]), name
    assert all(not parameter.requires_grad for parameter in indexer.parameters())


def _step(model: Step5ForCausalLM, inputs: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Run tiny native model loss/backward on integer [batch, sequence] inputs."""
    model.zero_grad(set_to_none=True)
    loss = model(inputs, labels=inputs).loss
    loss.backward()
    gradients = {
        name.replace("_checkpoint_wrapped_module.", ""): parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return loss.detach(), gradients


@pytest.mark.runtime_budget(20, reason="Three tiny-model backward passes with cold CPU compilation and native AC")
@pytest.mark.parametrize("selective", [False, True])
def test_native_checkpoint_replays_selector_once_with_exact_loss_and_gradients(selective: bool) -> None:
    reference = make_model()
    candidate = copy.deepcopy(reference)
    no_replay = copy.deepcopy(reference)
    inputs = torch.arange(16)[None]
    expected_loss, expected_gradients = _step(reference, inputs)
    for model, replay_enabled in [(no_replay, False), (candidate, True)]:
        calls = []
        module = model.model.layers["3"]
        module.self_attn.indexer.register_forward_hook(lambda *args: calls.append(1))
        base = make_selective_checkpoint_context_fn() if selective else None
        context_fn = _with_model_checkpoint_context(module, base) if replay_enabled else base
        model.model.layers["3"] = checkpoint_wrapper(
            module, preserve_rng_state=True, context_fn=context_fn or noop_context_fn
        )
        loss, gradients = _step(model, inputs)
        assert len(calls) == (1 if replay_enabled else 2)
        torch.testing.assert_close(loss, expected_loss, rtol=0, atol=0)
        assert gradients.keys() == expected_gradients.keys()
        for name, gradient in gradients.items():
            torch.testing.assert_close(gradient, expected_gradients[name], rtol=0, atol=0, msg=name)
        assert region_replay.current() is None


def test_live_microbatches_have_distinct_routes_and_release_records_after_backward() -> None:
    reference = make_model()
    candidate = copy.deepcopy(reference)
    module = candidate.model.layers["3"]
    routes = []
    module.self_attn.indexer.register_forward_hook(lambda module, args, result: routes.append(weakref.ref(result)))
    candidate.model.layers["3"] = checkpoint_wrapper(module, context_fn=_with_model_checkpoint_context(module, None))
    examples = [
        (torch.arange(16)[None], torch.arange(16)[None]),
        ((torch.arange(16) + 17)[None], torch.arange(8).repeat(2)[None]),
    ]
    expected = [reference(ids, labels=ids, position_ids=positions).loss for ids, positions in examples]
    actual = [candidate(ids, labels=ids, position_ids=positions).loss for ids, positions in examples]
    assert len(routes) == 2 and routes[0]() is not routes[1]()
    for result in reversed(actual):
        result.backward()
    for result in reversed(expected):
        result.backward()
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
    ref_grads = {name: p.grad for name, p in reference.named_parameters() if p.grad is not None}
    for name, parameter in candidate.named_parameters():
        if parameter.grad is not None:
            torch.testing.assert_close(
                parameter.grad, ref_grads[name.replace("_checkpoint_wrapped_module.", "")], rtol=0, atol=0
            )
    assert len(routes) == 2
    del actual, result, got
    gc.collect()
    assert all(route() is None for route in routes)
    assert region_replay.current() is None
