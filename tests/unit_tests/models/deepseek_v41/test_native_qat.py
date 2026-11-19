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

"""QAT module contracts and unchanged parameter namespace."""

import pytest
import torch

from nemo_automodel.components.models.deepseek_v41.qat import NativeQATLinear, NativeQATGroupedLinear, NativeQATMLP
from nemo_automodel.components.models.deepseek_v41.quantization import quantize_native


class _SingleRankDispatcher:
    """Stand-in for the flex dispatcher on one rank that owns every expert.

    Each token is replicated once per selected expert and the copies are sorted
    by expert, as the real dispatch delivers them; the combine sums the copies
    of a token back together.
    """

    def __init__(self, experts: int) -> None:
        self.experts = experts

    def token_permutation2(self, hidden_states, num_local_tokens, token_probs, token_indices):
        tokens, selected = token_indices.shape
        valid = token_indices.reshape(-1) >= 0
        self.source = torch.arange(tokens).repeat_interleave(selected)[valid]
        flat = token_indices.reshape(-1)[valid]
        order = flat.argsort(stable=True)
        self.source = self.source[order]
        self.tokens = tokens
        counts = torch.bincount(flat, minlength=self.experts)
        return hidden_states[self.source], counts, token_probs.reshape(-1)[valid][order]

    def token_unpermutation(self, values):
        return torch.zeros(self.tokens, values.shape[-1], dtype=values.dtype).index_add(0, self.source, values)


def test_linear_matches_dequantized_operands_and_ste_gradients():
    torch.manual_seed(119)
    layer = NativeQATLinear(64, 32, bias=False, dtype=torch.float32)
    x = torch.randn(2, 3, 64, requires_grad=True)
    upstream = torch.randn(2, 3, 32)
    output = layer(x)
    qx = quantize_native(x.detach().reshape(-1, 64)).reshape_as(x)
    qw = quantize_native(layer.weight.detach(), weight=True)
    torch.testing.assert_close(output, torch.nn.functional.linear(qx, qw), rtol=0, atol=0)
    output.backward(upstream)
    torch.testing.assert_close(x.grad, upstream @ qw, rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, upstream.flatten(0, 1).T @ qx.flatten(0, 1), rtol=0, atol=0)
    assert set(layer.state_dict()) == {"weight"}


def test_grouped_linear_preserves_independent_groups_and_backward():
    torch.manual_seed(120)
    layer = NativeQATGroupedLinear(64, 64, 2, bias=False)
    x = torch.randn(3, 2, 64, requires_grad=True)
    actual = layer(x)
    qx = quantize_native(x.detach())
    qw = quantize_native(layer.weight.detach(), weight=True).reshape(2, 32, 64)
    expected = torch.stack([qx[:, group] @ qw[group].T for group in range(2)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    actual.backward(torch.randn_like(actual))
    assert torch.isfinite(layer.weight.grad).all() and torch.isfinite(x.grad).all()
    assert set(layer.state_dict()) == {"weight"}


def test_shared_mlp_qat_preserves_state_dict_and_trainability():
    torch.manual_seed(121)
    layer = NativeQATMLP(64, 64, "torch", dtype=torch.float32, swiglu_limit=10.)
    x = torch.randn(2, 3, 64, requires_grad=True)
    actual = layer(x)
    actual.backward(torch.randn_like(actual))
    assert set(layer.state_dict()) == {"gate_proj.weight", "up_proj.weight", "down_proj.weight"}
    for parameter in layer.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0


def test_routed_qat_grouped_gemm_matches_per_expert_forward_and_gradients():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.qat import NativeQATExpertsDeepEP
    from nemo_automodel.components.moe.config import MoEConfig

    torch.manual_seed(122)
    config = MoEConfig(dim=64, inter_dim=64, moe_inter_dim=64, n_routed_experts=3,
                       n_activated_experts=2, n_shared_experts=1, n_expert_groups=0, n_limited_groups=0,
                       train_gate=True, gate_bias_update_factor=0., aux_loss_coeff=0.,
                       score_func="sqrtsoftplus", route_scale=1., norm_topk_prob=True,
                       swiglu_limit=10., dtype=torch.bfloat16)
    backend = BackendConfig(experts="torch_mm", dispatcher="hybridep")
    layer = NativeQATExpertsDeepEP(config, backend=backend, dispatcher_backend="hybridep")
    activation = layer.expert_activation
    x = torch.randn(5, 64, dtype=torch.bfloat16, requires_grad=True)
    w1 = torch.randn(3, 64, 128, dtype=torch.bfloat16, requires_grad=True)
    w2 = torch.randn(3, 64, 64, dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([2, 0, 3], dtype=torch.int32)
    probs = torch.ones(5, 1, dtype=torch.bfloat16)
    actual = layer.compute_grouped(x, w1, w2, counts, probs)
    reference_inputs = [v.detach().clone().requires_grad_() for v in (x, w1, w2)]
    rx, rw1, rw2 = reference_inputs
    outputs = []
    start = 0
    for expert, count in enumerate(counts.tolist()):
        if count:
            hidden = quantize_native(rx[start:start + count]) @ quantize_native(rw1[expert].T, fp4=True, weight=True).T
            hidden = activation(hidden, probs[start:start + count])
            outputs.append(quantize_native(hidden) @ quantize_native(rw2[expert].T, fp4=True, weight=True).T)
        start += count
    expected = torch.cat(outputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    upstream = torch.randn_like(actual)
    actual.backward(upstream)
    expected.backward(upstream)
    for value, reference in zip((x, w1, w2), reference_inputs, strict=True):
        torch.testing.assert_close(value.grad, reference.grad, rtol=0.02, atol=0.0625)
    assert torch.count_nonzero(w1.grad[1]) == 0
    assert torch.count_nonzero(w2.grad[1]) == 0


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_engram_wkv_quantizes_weight_and_leaves_released_table_rows_unchanged(tmp_path, dtype):
    import json

    from safetensors.torch import save_file

    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
    from nemo_automodel.components.models.deepseek_v41.engram import DeepseekV41Engram

    torch.manual_seed(31)
    tensors = {
        "layers.1.engram.embed.weight": torch.randn(17, 64).to(torch.float8_e4m3fn),
        "layers.1.engram.embed.scale": torch.randint(123, 130, (17, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu),
    }
    save_file(tensors, tmp_path / "table.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "table.safetensors" for name in tensors}})
    )
    sizes = dict(
        hidden_size=32, hc_mult=2, engram_layer_ids=[1], engram_num_embeddings=[17], engram_head_dim=64,
        engram_n_heads=2, engram_max_ngram_size=2, engram_host_checkpoint=str(tmp_path), dtype=dtype,
        initializer_range=0.1,
    )
    backend = BackendConfig(linear="torch")
    plain = DeepseekV41Engram(DeepseekV41TextConfig(**sizes), 1, backend)
    qat = DeepseekV41Engram(DeepseekV41TextConfig(native_qat=True, **sizes), 1, backend)
    assert type(plain.wkv) is torch.nn.Linear and type(qat.wkv) is NativeQATLinear
    qat.load_state_dict(plain.state_dict(), strict=True)

    ids = torch.tensor([[[0, 16], [2, 2], [0, 2]], [[16, 0], [5, 9], [16, 16]]])
    rows = qat.embed(ids).flatten(-2)
    # Released rows already lie on the MXFP8 grid in the activation quantizer's own blocks.
    torch.testing.assert_close(quantize_native(rows.reshape(-1, 128)).reshape_as(rows), rows, rtol=0, atol=0)

    # The only difference from the unquantized module is the wkv weight Q/DQ.
    quantized = quantize_native(plain.wkv.weight.detach(), weight=True)
    assert not torch.equal(quantized, plain.wkv.weight)
    x = torch.randn(2, 3, 2, 32, dtype=getattr(torch, dtype))
    output = qat(x, ids)
    assert not torch.equal(output, plain(x, ids))
    with torch.no_grad():
        plain.wkv.weight.copy_(quantized)
    torch.testing.assert_close(output, plain(x, ids), rtol=0, atol=0)

    output.sum().backward()
    assert qat.wkv.weight.grad is not None and torch.count_nonzero(qat.wkv.weight.grad)
    assert set(qat.state_dict()) == set(plain.state_dict())

    with pytest.raises(ValueError, match="32-channel blocks per head"):
        DeepseekV41TextConfig(native_qat=True, **{**sizes, "engram_head_dim": 48})


def test_native_qat_requires_hybridep_and_expert_parallelism():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

    text = DeepseekV41TextConfig(
        native_qat=True, vocab_size=64, hidden_size=32, moe_intermediate_size=32, num_hidden_layers=2,
        head_dim=32, qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=1, index_head_dim=32,
        n_routed_experts=2, num_experts_per_tok=1, engram_layer_ids=[], engram_num_embeddings=[], dtype="float32",
    )
    config = DeepseekV41Config(text_config=text, vision_config=DeepseekV41VisionConfig(num_hidden_layers=0))
    common = dict(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm")
    with torch.device("meta"), pytest.raises(ValueError, match="hybridep dispatcher"):
        DeepseekV41ForCausalLM(config, backend=BackendConfig(dispatcher="torch", **common))
    # World size one: the MoE falls back to all-gather experts, which have no QAT compute.
    with torch.device("meta"), pytest.raises(ValueError, match="expert parallelism"), pytest.warns(UserWarning):
        DeepseekV41ForCausalLM(config, backend=BackendConfig(dispatcher="hybridep", **common))


def test_native_qat_model_initialization_forward_backward_and_optimizer_step(monkeypatch):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig, DeepseekV41VisionConfig
    from nemo_automodel.components.models.deepseek_v41.model import DeepseekV41ForCausalLM

    torch.manual_seed(123)
    text = DeepseekV41TextConfig(
        native_qat=True, vocab_size=64, hidden_size=32, moe_intermediate_size=32,
        num_hidden_layers=6, num_attention_heads=2, head_dim=32, qk_rope_head_dim=16,
        q_lora_rank=32, o_lora_rank=32, o_groups=1, n_routed_experts=2, num_experts_per_tok=2,
        compress_ratios=[0, 0, 2, 2, 1, 1], kv_source_layer_ids=[2, 4],
        index_source_layer_ids=[2, 4, 5], index_n_heads=2, index_head_dim=32, index_topk=2,
        candidate_source_layer_id=4, candidate_topk_blocks=2, candidate_block_size=2,
        engram_layer_ids=[], engram_num_embeddings=[], num_nextn_predict_layers=0,
        dspark_block_size=0, dspark_noise_token_id=0, dtype="float32",
    )
    config = DeepseekV41Config(text_config=text, vision_config=DeepseekV41VisionConfig(num_hidden_layers=0))
    backend = BackendConfig(attn="eager", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="hybridep")
    # Build the expert-parallel model on one CPU process: report two ranks to the
    # MoE constructor, then give every layer a single-rank stand-in dispatcher.
    monkeypatch.setattr("nemo_automodel.components.moe.layers.get_world_size_safe", lambda: 2)
    monkeypatch.setattr(torch.nn.Parameter, "to_local", lambda self: self, raising=False)
    with torch.device("meta"):
        model = DeepseekV41ForCausalLM(config, backend=backend)
    model.to_empty(device="cpu")
    model.initialize_weights(torch.device("cpu"), dtype=torch.float32)
    for layer in model.model.layers.values():
        assert type(layer.ffn.experts).__name__ == "NativeQATExpertsDeepEP"
        layer.ffn.experts.token_dispatcher = _SingleRankDispatcher(2)
        layer.ffn.experts.ep_size, layer.ffn.experts.n_routed_experts = 1, 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    inputs = torch.randint(0, 64, (1, 8))
    before = model.model.layers["0"].attn.wq_a.weight.detach().clone()
    for _ in range(2):
        optimizer.zero_grad()
        output = model(inputs, labels=inputs)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None, name
                assert torch.isfinite(parameter.grad).all(), name
        optimizer.step()
    assert not torch.equal(model.model.layers["0"].attn.wq_a.weight, before)


def _released(raw: torch.Tensor, fp4: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode weights in the released checkpoint layout.

    Args:
        raw: FP32 tensor of shape [rows, columns], both divisible by 32.
        fp4: Pack E2M1 codes two per byte with one scale per row and 32
            columns; otherwise E4M3 values with 32x32 scales.

    Returns:
        The Q/DQ weights [rows, columns] (FP32), the checkpoint values
        ([rows, columns / 2] INT8 for FP4, [rows, columns] E4M3 otherwise) and
        the E8M0 scales ([rows, columns / 32] for FP4, [rows / 32, columns / 32]
        otherwise).
    """
    grid = quantize_native(raw, fp4=fp4, weight=True)
    row_block = 1 if fp4 else 32
    shape = (grid.shape[0] // row_block, row_block, grid.shape[1] // 32, 32)
    # The scale follows the maximum before rounding, as in the release: where
    # that maximum rounds down, the largest stored code is 3 (or 224), not 4-6.
    maximum = raw.to(torch.bfloat16).float().reshape(shape).abs().amax((1, 3)).clamp_min(2.0**-100)
    scale = torch.exp2(torch.ceil(torch.log2(maximum / (6 if fp4 else 448))))
    codes = (grid.reshape(shape) / scale[:, None, :, None]).reshape(grid.shape)
    if not fp4:
        return grid, codes.to(torch.float8_e4m3fn), scale.to(torch.float8_e8m0fnu)
    index = torch.bucketize(codes.abs(), torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])).to(torch.uint8)
    index |= torch.signbit(codes).to(torch.uint8) << 3
    return grid, (index[:, 0::2] | (index[:, 1::2] << 4)).view(torch.int8), scale.to(torch.float8_e8m0fnu)


@pytest.mark.parametrize("fp4", [False, True])
def test_spread_masters_keep_released_values_and_pass_small_updates(fp4):
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import dequantize_checkpoint_weight

    torch.manual_seed(0)
    raw = torch.randn(256, 1024) * 0.02
    raw[:32, :32] = 0
    grid, weight, scale = _released(raw, fp4)
    centred = dequantize_checkpoint_weight(weight, scale, dtype=torch.float32)
    spread = dequantize_checkpoint_weight(weight, scale, dtype=torch.float32, spread_seed=5)
    assert torch.equal(centred, grid)
    assert torch.equal(spread, dequantize_checkpoint_weight(weight, scale, dtype=torch.float32, spread_seed=5))
    assert not torch.equal(spread, dequantize_checkpoint_weight(weight, scale, dtype=torch.float32, spread_seed=6))
    # The quantized model is the released one; an all-zero block has no cell to move in.
    assert torch.equal(quantize_native(spread, fp4=fp4, weight=True), grid)
    assert (spread[:32, :32] == 0).all()
    assert (spread != grid).float().mean() > 0.9
    if fp4:
        # Blocks stored with a largest code of 3 are spread as well.
        largest = (grid.abs().reshape(256, 32, 32).amax(-1) / scale.float()).round()
        assert (largest == 3).any()
        assert ((spread != grid).reshape(256, 32, 32).float().mean(-1)[largest == 3] > 0.5).all()
    # An update far below the cell size is lost at the cell centre and, on
    # average, fully present in the quantized weights of spread masters.
    direction = torch.sign(torch.randn_like(grid))
    direction[:32, :32] = 0
    update = 1e-6 * direction
    expressed = [
        ((quantize_native(start + update, fp4=fp4, weight=True) - grid) * direction).sum() / update.abs().sum()
        for start in (centred, spread)
    ]
    assert expressed[0] < 0.01
    assert 0.7 < expressed[1] < 1.3


def test_cell_positions_depend_only_on_seed_and_global_index():
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import _cell_positions

    full = _cell_positions(77, torch.arange(64), torch.arange(128), 128)
    shard = _cell_positions(77, torch.arange(10, 30), torch.arange(32, 96), 128)
    assert torch.equal(full[10:30, 32:96], shard)
    assert not torch.equal(full, _cell_positions(78, torch.arange(64), torch.arange(128), 128))
    assert full.min() >= 0 and full.max() < 1
    assert abs(full.mean() - 0.5) < 0.02 and abs(full.std() - 12**-0.5) < 0.02


def test_spread_masters_reject_unsupported_layouts_and_configuration():
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41TextConfig
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import dequantize_checkpoint_weight

    _, weight, scale = _released(torch.randn(32, 64), False)
    with pytest.raises(ValueError, match="FP32 output"):
        dequantize_checkpoint_weight(weight, scale, dtype=torch.bfloat16, spread_seed=1)
    rowwise = torch.ones(32, 2).to(torch.float8_e8m0fnu)
    with pytest.raises(ValueError, match="block scales"):
        dequantize_checkpoint_weight(weight, rowwise, dtype=torch.float32, rowwise=True, spread_seed=1)
    with pytest.raises(ValueError, match="requires native_qat"):
        DeepseekV41TextConfig(native_qat_master_seed=1)
    with pytest.raises(ValueError, match="native_qat_master_seed"):
        DeepseekV41TextConfig(native_qat=True, native_qat_master_seed=-1)


def test_adapter_spreads_qat_weights_and_keeps_engram_tables():
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config, DeepseekV41TextConfig
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import DeepseekV41StateDictAdapter
    from nemo_automodel.components.moe.config import MoEConfig

    torch.manual_seed(1)
    text = DeepseekV41TextConfig(
        native_qat=True, native_qat_master_seed=3, vocab_size=64, hidden_size=32, moe_intermediate_size=32,
        num_hidden_layers=2, head_dim=32, qk_rope_head_dim=16, q_lora_rank=32, o_lora_rank=32, o_groups=1,
        index_head_dim=32, n_routed_experts=2, num_experts_per_tok=1, engram_layer_ids=[1],
        engram_num_embeddings=[4], engram_head_dim=32, dtype="float32",
    )
    moe = MoEConfig(
        dim=32, inter_dim=32, moe_inter_dim=32, n_routed_experts=2, n_shared_experts=1, n_activated_experts=1,
        n_expert_groups=0, n_limited_groups=0, train_gate=True, gate_bias_update_factor=0, aux_loss_coeff=0,
        score_func="sqrtsoftplus", route_scale=1.5, norm_topk_prob=True, dtype=torch.float32,
    )
    backend = BackendConfig(linear="torch", attn="sdpa", dispatcher="hybridep", experts="torch_mm")
    adapter = DeepseekV41StateDictAdapter(DeepseekV41Config(text_config=text), moe, backend, dtype=torch.float32)
    dense, dense_value, dense_scale = _released(torch.randn(32, 64) * 0.02, False)
    expert, expert_value, expert_scale = _released(torch.randn(32, 32) * 0.02, True)
    table = torch.randn(4, 32).to(torch.float8_e4m3fn)
    state = {
        "layers.0.attn.wq_a.weight": dense_value,
        "layers.0.attn.wq_a.scale": dense_scale,
        "layers.0.ffn.experts.0.w2.weight": expert_value,
        "layers.0.ffn.experts.0.w2.scale": expert_scale,
        "layers.1.engram.embed.weight": table,
        "layers.1.engram.embed.scale": torch.ones(4, 1).to(torch.float8_e8m0fnu),
    }
    masters = adapter._dequantize(state)
    assert set(masters) == {"layers.0.attn.wq_a.weight", "layers.0.ffn.experts.0.w2.weight", "layers.1.engram.embed.weight"}
    for key, grid, fp4 in (("layers.0.attn.wq_a.weight", dense, False), ("layers.0.ffn.experts.0.w2.weight", expert, True)):
        assert not torch.equal(masters[key], grid)
        assert torch.equal(quantize_native(masters[key], fp4=fp4, weight=True), grid)
    assert torch.equal(masters["layers.1.engram.embed.weight"], table.float())


def test_hybridep_experts_forward_uses_the_qat_compute_between_dispatch_and_combine():
    """The flex-dispatcher forward must hand local matrices and received tokens to compute_grouped."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v41.qat import NativeQATExpertsDeepEP
    from nemo_automodel.components.moe.config import MoEConfig
    from nemo_automodel.components.moe.experts import GroupedExpertsDeepEP

    torch.manual_seed(124)
    config = MoEConfig(dim=64, inter_dim=64, moe_inter_dim=64, n_routed_experts=2,
                       n_activated_experts=1, n_shared_experts=1, n_expert_groups=0, n_limited_groups=0,
                       train_gate=True, gate_bias_update_factor=0., aux_loss_coeff=0.,
                       score_func="sqrtsoftplus", route_scale=1., norm_topk_prob=True,
                       swiglu_limit=10., dtype=torch.bfloat16)
    layer = NativeQATExpertsDeepEP(config, backend=BackendConfig(experts="torch_mm", dispatcher="hybridep"),
                                   dispatcher_backend="hybridep")

    class _Local(torch.nn.Parameter):
        def to_local(self):
            return self

    layer.gate_and_up_projs = _Local(torch.randn(2, 64, 128, dtype=torch.bfloat16))
    layer.down_projs = _Local(torch.randn(2, 64, 64, dtype=torch.bfloat16))

    layer.token_dispatcher = _SingleRankDispatcher(2)
    layer.ep_size, layer.n_routed_experts = 1, 2
    x = torch.randn(6, 64, dtype=torch.bfloat16)
    indices = torch.tensor([[1], [0], [1], [1], [0], [0]])
    weights = torch.ones(6, 1, dtype=torch.bfloat16)
    with torch.no_grad():
        actual = layer(x, torch.ones(6, dtype=torch.bool), weights, indices)
        expected = torch.empty_like(actual)
        for expert in range(2):
            rows = indices[:, 0] == expert
            hidden = quantize_native(x[rows]) @ quantize_native(layer.gate_and_up_projs[expert].T, fp4=True, weight=True).T
            hidden = layer.expert_activation(hidden, weights[rows])
            expected[rows] = quantize_native(hidden) @ quantize_native(layer.down_projs[expert].T, fp4=True, weight=True).T
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # The unquantized compute gives a different result, so the hook is in effect.
        plain = GroupedExpertsDeepEP.compute_grouped(
            layer, x[rows], layer.gate_and_up_projs[1:], layer.down_projs[1:], torch.tensor([3]), weights[rows])
        assert not torch.equal(plain, actual[rows])
