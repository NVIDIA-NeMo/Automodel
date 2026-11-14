# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Checkpoint-free, single-rank real-model coverage of reference LoRA QAT.

The public round-trip assertion is also used by the CUDA functional tests; no
private helpers from other model tests are imported. Missing production model
dependencies deliberately fail their individual cases, rather than skip them.
"""

import copy
import json
import os
from pathlib import Path
from typing import Literal

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import LinearLoRA, PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components._peft.lora_experts import GroupedExpertsLoRA
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightFakeQuantizer, WeightQuantizationConfig

# Full-model initialization, two backwards and serialization exceed the 5s unit
# fallback on shared CPU runners; these are still sub-2M-parameter CPU tests.
pytestmark = pytest.mark.timeout(60)


def _tiny_model(architecture: str, dtype: torch.dtype, *, glm_hybrid: bool = False) -> nn.Module:
    assert not glm_hybrid or architecture == "glm5_next"
    backend = BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        rope_fusion=False,
        enable_hf_state_dict_adapter=False,
    )
    dtype_name = str(dtype).removeprefix("torch.")
    if architecture == "glm5_next":
        from nemo_automodel.components.models.glm5_next.config import (
            Glm5NextConfig,
            Glm5NextTextConfig,
            Glm5NextVisionConfig,
        )
        from nemo_automodel.components.models.glm5_next.model import Glm5NextForConditionalGeneration

        config = Glm5NextConfig(
            text_config=Glm5NextTextConfig(
                vocab_size=64,
                hidden_size=128,
                # Both canonical expert matrices must tile 128x128 FP8 blocks;
                # doubling gate/up alone would leave the down matrix too small.
                intermediate_size=128,
                moe_intermediate_size=128,
                num_hidden_layers=2 if glm_hybrid else 1,
                num_attention_heads=2,
                num_key_value_heads=2,
                n_shared_experts=1,
                n_routed_experts=2,
                num_experts_per_tok=2,
                kv_lora_rank=32,
                q_lora_rank=128,
                qk_rope_head_dim=0,
                qk_nope_head_dim=32,
                v_head_dim=32,
                index_topk=4,
                index_head_dim=8,
                index_n_heads=2,
                index_kpool=2,
                hc_mult=2,
                hc_sinkhorn_iters=3,
                # DSA -> KDA is an adjacent attention pair in GLM-5.3-Flash.
                # Keep one routed MLP and a tiny dense second MLP, not the full
                # 45-layer checkpoint's dense/MoE schedule or 3:1 KDA/DSA ratio.
                mlp_layer_types=["sparse", "dense"] if glm_hybrid else ["sparse"],
                layer_types=(
                    ["deepseek_sparse_attention", "linear_attention"] if glm_hybrid else ["deepseek_sparse_attention"]
                ),
                linear_num_heads=2,
                linear_head_dim=32,
                linear_conv_kernel_dim=4,
                num_nextn_predict_layers=0,
                pad_token_id=0,
                dtype=dtype_name,
            ),
            vision_config=Glm5NextVisionConfig(
                depth=1,
                hidden_size=8,
                num_heads=2,
                patch_size=2,
                temporal_patch_size=2,
                spatial_merge_size=2,
                out_hidden_size=128,
                intermediate_size=16,
                projection_intermediate_size=32,
                dtype=dtype_name,
            ),
            image_token_id=63,
            pad_token_id=0,
        )
        model = Glm5NextForConditionalGeneration(config, backend=backend)
        assert config.model_type == "glm5_next"
        if glm_hybrid:
            from nemo_automodel.components.models.glm5_next.layers import (
                Glm5NextLinearAttention,
                Glm5NextSparseAttention,
            )

            assert isinstance(model.get_submodule("model.language_model.layers.0.self_attn"), Glm5NextSparseAttention)
            assert isinstance(model.get_submodule("model.language_model.layers.1.self_attn"), Glm5NextLinearAttention)
    elif architecture == "deepseek_v4":
        from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
        from nemo_automodel.components.models.deepseek_v4.model import DeepseekV4ForCausalLM

        config = DeepseekV4Config(
            vocab_size=64,
            hidden_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=32,
            qk_rope_head_dim=16,
            q_lora_rank=128,
            o_lora_rank=32,
            o_groups=1,
            n_routed_experts=2,
            n_shared_experts=1,
            num_experts_per_tok=2,
            num_hash_layers=0,
            compress_ratios=[0],
            sliding_window=16,
            max_position_embeddings=128,
            hc_mult=2,
            hc_sinkhorn_iters=3,
            num_nextn_predict_layers=0,
            dtype=dtype_name,
        )
        model = DeepseekV4ForCausalLM(config, backend=backend)
        if dtype == torch.float32:
            assert all(p.dtype == dtype for p in model.parameters() if p.is_floating_point())
    else:
        raise ValueError(f"Unknown test architecture: {architecture}")
    # Use each model's production initializer (which owns its init_weights /
    # reset_parameters traversal), not arbitrary parameter fills or checkpoints.
    model.initialize_weights(torch.device("cpu"), dtype=dtype)
    assert sum(p.numel() for p in model.parameters()) < 2_000_000
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert not value.is_meta, name
        assert torch.isfinite(value).all(), name
    if architecture == "deepseek_v4":
        for name in (
            "model.layers.0.self_attn.wq_a.weight",
            "model.layers.0.self_attn.q_norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.mlp.experts.gate_and_up_projs",
            "model.layers.0.mlp.experts.down_projs",
        ):
            assert model.get_parameter(name).dtype == dtype, name
        # V4 deliberately keeps the language head and HC mixers in fp32.
        for name in ("lm_head.weight", "model.layers.0.attn_hc.fn", "model.layers.0.ffn_hc.fn"):
            assert model.get_parameter(name).dtype == torch.float32, name
    return model


@torch.no_grad()
def _evaluate(
    model: nn.Module, input_ids: torch.Tensor, targets: tuple[str, ...]
) -> tuple[torch.Tensor, dict[str, list[torch.Tensor]]]:
    """Capture actual dense and routed-expert outputs alongside logits.

    Args:
        model: Eval model on the same device as input_ids.
        input_ids: Integer tensor of shape [batch, sequence].
        targets: Dense projection, grouped expert and optional hybrid attention
            module names. Attention modules return [batch, sequence, hidden].

    Returns:
        Independent logits [batch, sequence, vocab] and per-call intermediates:
        dense [batch, sequence, query_rank], experts [tokens, hidden], where
        query_rank is the attention query bottleneck width; optional hybrid
        attention outputs are [batch, sequence, hidden]. Tensors retain their
        source dtype/device; no input storage is modified or aliased.
    """
    captured = {name: [] for name in targets}
    handles = []
    for name in targets:

        def capture(
            module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor, *, key: str = name
        ) -> None:
            """Copy one projection output without changing its computation.

            Args:
                module: Dense or grouped expert projection being observed.
                inputs: Dense input [batch, sequence, hidden], or expert inputs
                    x [tokens, hidden], mask [tokens], routing probabilities
                    [tokens, top_k], and expert indices [tokens, top_k].
                output: Dense [batch, sequence, query_rank] or expert
                    [tokens, hidden] or attention [batch, sequence, hidden]
                    tensor, in the model compute dtype/device.
                key: Name under which to store an independent output clone.
            """
            captured[key].append(output.detach().clone())

        handles.append(model.get_submodule(name).register_forward_hook(capture))
    try:
        # GLM's reference sparse attention uses SDPA internally even with eager
        # selected. Pin math SDPA so CUDA does not silently select a fused kernel.
        with sdpa_kernel(SDPBackend.MATH):
            logits = model(input_ids).logits.detach().clone()
    finally:
        for handle in handles:
            handle.remove()
    assert all(captured.values()), "Every observed module must execute during the real-model forward"
    assert torch.isfinite(logits).all()
    return logits, captured


@torch.compiler.set_stance("force_eager")
def run_tiny_model_roundtrip(
    architecture: str,
    *,
    dtype: torch.dtype,
    device: torch.device,
    fp8_block_size: int,
    directory: Path,
    target_experts: bool = True,
    qat_experts_only: bool = False,
    expert_format: Literal["mxfp4", "fp8"] = "mxfp4",
    glm_hybrid: bool = False,
) -> None:
    """Check adapter training and packed reference reload on one local model.

    CPU reload is compared exactly against source evaluation on CPU, not against
    CUDA arithmetic. CUDA cases additionally compare source and reload on CUDA.
    This exercises dequantized inference, never a native FP4 serving backend.
    Force eager execution because the shared expert activation is explicitly
    decorated with fullgraph torch.compile even under the eager backend.
    Dense-only cases isolate expert-path failures without skipping those cases.
    Expert-only QAT retains attention LoRA, exercising independent selectors;
    only that case allows roundoff from exporting additive LoRA as merged floats.
    FP8 experts always use GLM-5.3-Flash's 128x128 weight blocks, independently
    of the dense block argument; legacy cases retain MXFP4 1x32 experts.
    The optional GLM hybrid adds production KDA after DSA, including gradients
    through KDA to the first layer's adapters. The default is sparse-only.
    """
    assert not qat_experts_only or target_experts
    assert int(os.environ.get("WORLD_SIZE", "1")) == 1, "LoRA QAT integration supports single rank only"
    assert not dist.is_initialized() or dist.get_world_size() == 1, "LoRA QAT integration supports single rank only"
    torch.manual_seed(19)
    model = _tiny_model(architecture, dtype, glm_hybrid=glm_hybrid).to(device=device)
    model_config = copy.deepcopy(model.config.to_dict())
    if architecture == "glm5_next":
        dense_name = "model.language_model.layers.0.self_attn.q_a_proj"
        experts_name = "model.language_model.layers.0.mlp.experts"
        experts_glob = "model.language_model.layers.*.mlp.experts"
    else:
        dense_name = "model.layers.0.self_attn.wq_a"
        experts_name = "model.layers.0.mlp.experts"
        experts_glob = "model.layers.*.mlp.experts"
    targets = (dense_name, experts_name)
    peft_targets = targets if target_experts else (dense_name,)
    qat_targets = (experts_name,) if qat_experts_only else peft_targets
    if glm_hybrid:
        targets += ("model.language_model.layers.0.self_attn", "model.language_model.layers.1.self_attn")
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    patched = apply_lora_to_linear_modules(
        model,
        PeftConfig(target_modules=list(peft_targets), dim=4, alpha=8, use_memory_efficient_lora=False),
    )
    assert patched == len(peft_targets)
    assert isinstance(model.get_submodule(dense_name), LinearLoRA)
    if target_experts:
        assert isinstance(model.get_submodule(experts_name), GroupedExpertsLoRA)
    for name, value in original.items():
        assert torch.equal(model.get_parameter(name), value), name

    dense_config = WeightQuantizationConfig("fp8", (fp8_block_size, fp8_block_size))
    expert_config = WeightQuantizationConfig(expert_format, (128, 128) if expert_format == "fp8" else (1, 32))
    if target_experts:
        experts = model.get_submodule(experts_name)
        inter = 128 if architecture == "glm5_next" else 64
        # Native grouped-MM storage is [experts, in, out]; QAT packs the
        # transpose as [experts, out, in], including fused gate/up.
        assert experts.gate_and_up_projs.shape == (2, 128, 2 * inter)
        assert experts.down_projs.shape == (2, inter, 128)
        rows, cols = expert_config.block_size
        for weight in (experts.gate_and_up_projs, experts.down_projs):
            assert weight.shape[-1] % rows == weight.shape[-2] % cols == 0
    rules = () if qat_experts_only else (QATRule((dense_name,), dense_config),)
    if target_experts:
        rules += (QATRule((experts_glob,), expert_config),)
    config = QATConfig(rules=rules)
    settings = copy.deepcopy(config)
    controller = QAT(config)
    assert controller is not QAT(config)
    identities = {name: id(p) for name, p in model.named_parameters()}
    assert controller.prepare(model) is model
    assert config == settings
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    for target in qat_targets:
        quantizer = model.get_submodule(target).weight_fake_quantizer
        assert isinstance(quantizer, WeightFakeQuantizer)
        assert quantizer.config == (expert_config if target == experts_name else dense_config)
    assert {
        name.removesuffix(".weight_fake_quantizer")
        for name, module in model.named_modules()
        if isinstance(module, WeightFakeQuantizer)
    } == set(qat_targets)
    if qat_experts_only:
        assert model.get_submodule(dense_name).weight_fake_quantizer is None
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    assert len(trainable) == (6 if target_experts else 2)
    assert all("lora_" in name and name not in original for name in trainable)
    assert all(any(name.startswith(target + ".lora_") for target in peft_targets) for name in trainable)
    if not qat_experts_only:
        assert set(qat_targets) == set(peft_targets), "Exact reload requires QAT on every trained target"
    initial_adapters = {name: p.detach().clone() for name, p in trainable.items()}
    optimizer = torch.optim.AdamW(trainable.values(), lr=0.01, weight_decay=0)
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21], [2, 6, 10, 14, 18, 22]], device=device)
    model.train()
    hybrid_gradients = []

    def capture_hybrid_backward(
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        """Check that training actually differentiates through production KDA.

        Args:
            module: KDA attention module, with frozen parameters.
            grad_input: One gradient [batch, sequence, hidden] for the KDA
                input; it must propagate to the upstream trainable adapters.
            grad_output: One gradient [batch, sequence, hidden] for the KDA
                output. Both gradients retain the model compute dtype/device.
        """
        for gradient in (*grad_input, *grad_output):
            assert gradient is not None and gradient.shape == (2, 6, 128)
            assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient) > 0
        hybrid_gradients.append(True)

    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        handle = (
            model.get_submodule(targets[-1]).register_full_backward_hook(capture_hybrid_backward)
            if glm_hybrid
            else None
        )
        try:
            with sdpa_kernel(SDPBackend.MATH):
                logits = model(input_ids).logits
                assert logits.shape == (2, 6, 64)
                assert torch.isfinite(logits).all()
                loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, 64), input_ids[:, 1:].reshape(-1))
                assert torch.isfinite(loss)
                loss.backward()
        finally:
            if handle is not None:
                handle.remove()
        if glm_hybrid:
            assert len(hybrid_gradients) == step + 1
        for name, p in model.named_parameters():
            if name in trainable:
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                # B starts at zero: A first gets a nonzero gradient on step 2.
                if step == 1 or "B" in name:
                    assert torch.count_nonzero(p.grad) > 0, name
            else:
                assert p.grad is None, name
        optimizer.step()
        for name, p in model.named_parameters():
            assert torch.isfinite(p).all(), name
            if name in original:
                assert torch.equal(p, original[name]), name
    assert all(not torch.equal(p, initial_adapters[name]) for name, p in trainable.items())

    model.eval()
    reference, intermediates = _evaluate(model, input_ids, targets)
    snapshot = {key: value.detach().clone() for key, value in model.state_dict().items()}
    controller.export(model, directory)
    assert set(model.state_dict()) == set(snapshot)
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key]), key
    assert identities == {name: id(p) for name, p in model.named_parameters()}
    assert model.config.to_dict() == model_config
    assert config == settings
    assert all(not child.training for child in model.modules())
    metadata = json.loads((directory / "manifest.json").read_text())
    expected_keys = {} if qat_experts_only else {dense_name + ".weight": ("fp8", [fp8_block_size, fp8_block_size])}
    if target_experts:
        expected_keys.update(
            {
                experts_name + ".gate_and_up_projs": (expert_format, list(expert_config.block_size)),
                experts_name + ".down_projs": (expert_format, list(expert_config.block_size)),
            }
        )
    assert {
        entry["weight_key"]: (entry["config"]["format"], entry["config"]["block_size"]) for entry in metadata["weights"]
    } == expected_keys
    ordinary = load_file(str(directory / "model.safetensors"))
    packed = load_file(str(directory / "quantized.safetensors"))
    assert not any("lora_" in key for key in ordinary)
    assert set(packed) == {key + suffix for key in expected_keys for suffix in (".payload", ".scales")}
    assert len(packed) == 2 * len(expected_keys) and all(value.dtype == torch.uint8 for value in packed.values())
    for key, (weight_format, (rows, cols)) in expected_keys.items():
        shape = original[key].shape
        if key != dense_name + ".weight":
            shape = (*shape[:-2], shape[-1], shape[-2])
        payload_shape = shape if weight_format == "fp8" else (*shape[:-1], shape[-1] // 2)
        assert packed[key + ".payload"].shape == payload_shape, key
        assert packed[key + ".scales"].shape == (*shape[:-2], shape[-2] // rows, shape[-1] // cols), key
    merged_keys = {dense_name + ".weight"} if qat_experts_only else set()
    if qat_experts_only:
        attention = model.get_submodule(dense_name)
        merged = attention.weight + attention.scale * (attention.lora_B.weight @ attention.lora_A.weight)
        assert torch.equal(ordinary[dense_name + ".weight"], merged.cpu())
        assert not torch.equal(ordinary[dense_name + ".weight"], original[dense_name + ".weight"].cpu())

    fresh = _tiny_model(architecture, dtype, glm_hybrid=glm_hybrid).eval()
    fresh_config = copy.deepcopy(fresh.config.to_dict())
    assert fresh_config == model_config
    QAT.load_quantized_checkpoint(fresh, directory)
    assert fresh.config.to_dict() == fresh_config
    assert not any(isinstance(module, (LinearLoRA, GroupedExpertsLoRA)) for module in fresh.modules())
    for name, p in fresh.named_parameters():
        assert p.device.type == "cpu"
        if name not in expected_keys and name not in merged_keys:
            assert torch.equal(p, original[name].cpu()), name
    # An untargeted attention adapter changes operation ordering when merged:
    # xW + (xA)B versus x(W + BA). Permit float32 accumulation roundoff or a
    # few BF16 ulps at unit scale, but keep all original round trips bit-exact.
    rtol, atol = ((2e-2, 2e-2) if dtype == torch.bfloat16 else (1e-4, 1e-5)) if qat_experts_only else (0, 0)
    if device.type == "cuda":
        fresh.to(device=device)
        if qat_experts_only:
            _assert_exact_expert_reload(model, fresh, experts_name)
        actual, actual_intermediates = _evaluate(fresh, input_ids, targets)
        for name in targets:
            torch.testing.assert_close(actual_intermediates[name], intermediates[name], rtol=rtol, atol=atol, msg=name)
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
        fresh.cpu()
        model.cpu()
        input_ids = input_ids.cpu()
        reference, intermediates = _evaluate(model, input_ids, targets)
    if qat_experts_only:
        _assert_exact_expert_reload(model, fresh, experts_name)
    actual, actual_intermediates = _evaluate(fresh, input_ids, targets)
    # Loading another model must not write through borrowed source state.
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), snapshot[key].cpu()), key
    assert model.config.to_dict() == model_config
    for name in targets:
        torch.testing.assert_close(actual_intermediates[name], intermediates[name], rtol=rtol, atol=atol, msg=name)
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)


@torch.no_grad()
def _assert_exact_expert_reload(source: nn.Module, fresh: nn.Module, name: str) -> None:
    """Isolate expert arithmetic from upstream attention's floating merge error."""
    experts = source.get_submodule(name)
    weight = experts.gate_and_up_projs
    x = torch.randn(6, 128, device=weight.device, dtype=weight.dtype)
    mask = torch.ones(6, device=weight.device, dtype=torch.bool)
    # Both tiny routed experts execute; identical inputs and routing avoid
    # attributing upstream floating-point differences to quantized expert math.
    indices = torch.tensor([[0, 1]], device=weight.device).expand(6, -1)
    probabilities = torch.tensor([[0.25, 0.75]], device=weight.device, dtype=weight.dtype).expand(6, -1)
    expected = experts(x, mask, probabilities, indices)
    actual = fresh.get_submodule(name)(x, mask, probabilities, indices)
    assert expected.shape == (6, 128) and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["glm5_next", "deepseek_v4"])
@pytest.mark.parametrize("fp8_block_size", [32, 128])
@pytest.mark.parametrize("target_experts", [False, True], ids=["dense", "dense-and-experts"])
def test_tiny_lora_qat_models_cpu(tmp_path: Path, architecture: str, fp8_block_size: int, target_experts: bool) -> None:
    run_tiny_model_roundtrip(
        architecture,
        dtype=torch.float32,
        device=torch.device("cpu"),
        fp8_block_size=fp8_block_size,
        directory=tmp_path,
        target_experts=target_experts,
    )


@pytest.mark.parametrize("architecture", ["glm5_next", "deepseek_v4"])
def test_tiny_expert_only_qat_with_attention_lora_cpu(tmp_path: Path, architecture: str) -> None:
    run_tiny_model_roundtrip(
        architecture,
        dtype=torch.float32,
        device=torch.device("cpu"),
        fp8_block_size=32,
        directory=tmp_path,
        qat_experts_only=True,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "qat_experts_only", [False, True], ids=["all-trained-targets-qat", "independent-experts-only-qat"]
)
@pytest.mark.parametrize(
    "architecture,expert_format,glm_hybrid",
    [
        pytest.param("glm5_next", "fp8", False, id="glm5.3-flash-fp8-128-sparse-only"),
        pytest.param("glm5_next", "fp8", True, id="glm5.3-flash-fp8-128-dsa-kda-hybrid"),
        pytest.param("deepseek_v4", "mxfp4", False, id="deepseek-v4-mxfp4"),
    ],
)
def test_tiny_checkpoint_format_lora_qat_cpu(
    tmp_path: Path,
    dtype: torch.dtype,
    architecture: str,
    expert_format: Literal["mxfp4", "fp8"],
    glm_hybrid: bool,
    qat_experts_only: bool,
) -> None:
    """Match expert weight formats, not full pretrained checkpoints or activations.

    GLM source: https://huggingface.co/zai-org/GLM-5.3-Flash (glm5_next,
    FP8 E4M3, 128x128). Tiny weights use production random initialization;
    this tests weight-only reference QAT, not the checkpoint's dynamic
    activation quantization or its complete exclusion list.
    """
    run_tiny_model_roundtrip(
        architecture,
        dtype=dtype,
        device=torch.device("cpu"),
        fp8_block_size=128,
        directory=tmp_path,
        expert_format=expert_format,
        glm_hybrid=glm_hybrid,
        qat_experts_only=qat_experts_only,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.compiler.set_stance("force_eager")
def test_tiny_deepseek_v4_production_initialization_and_hidden_output(dtype: torch.dtype) -> None:
    torch.manual_seed(19)
    model = _tiny_model("deepseek_v4", dtype).eval()
    input_ids = torch.tensor([[1, 5, 9, 13, 17, 21], [2, 6, 10, 14, 18, 22]])
    with torch.no_grad(), sdpa_kernel(SDPBackend.MATH):
        output = model(input_ids, output_hidden_states=True)
    assert output.logits.shape == (2, 6, 64)
    assert output.hidden_states.shape == (2, 6, 128)
    assert output.logits.dtype == output.hidden_states.dtype == dtype
    assert torch.isfinite(output.logits).all()
    assert torch.isfinite(output.hidden_states).all()
