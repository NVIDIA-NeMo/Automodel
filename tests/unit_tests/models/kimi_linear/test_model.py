# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from nemo_automodel._transformers.models.common import BackendConfig
from nemo_automodel._transformers.models.kimi_linear.config import KimiLinear48BConfig
from nemo_automodel._transformers.models.kimi_linear.model import KimiLinear48BForCausalLM
from tests.unit_tests.models.kimi_linear.test_cp import _FakeCPMesh


def _tiny_kimi_config(*, use_kda: bool = False) -> KimiLinear48BConfig:
    linear_attn_config = (
        {"kda_layers": [1], "full_attn_layers": [2]} if use_kda else {"kda_layers": [], "full_attn_layers": [1, 2]}
    )
    return KimiLinear48BConfig(
        vocab_size=128,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=None,
        kv_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=8,
        mla_use_nope=True,
        num_experts=4,
        num_experts_per_token=2,
        num_shared_experts=0,
        moe_intermediate_size=8,
        first_k_dense_replace=1,
        linear_attn_config={
            **linear_attn_config,
            "num_heads": 2,
            "head_dim": 8,
            "short_conv_kernel_size": 4,
        },
        torch_dtype="float32",
    )


def _require_fla() -> None:
    pytest.importorskip("fla")
    pytest.importorskip("fla.modules")
    pytest.importorskip("fla.ops.kda")
    pytest.importorskip("fla.ops.kda.gate")


def _backend_config() -> BackendConfig:
    return BackendConfig(
        attn="eager",
        linear="torch",
        rms_norm="torch_fp32",
        experts="torch",
        dispatcher="torch",
        enable_hf_state_dict_adapter=True,
    )


def _assert_floating_state_finite(model: torch.nn.Module) -> None:
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        if tensor.is_floating_point():
            assert torch.isfinite(tensor).all(), name


def test_update_moe_gate_bias_no_op_when_factor_zero():
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    moe_layer = model.model.layers["1"].mlp

    assert moe_layer.gate.bias_update_factor == 0.0
    with patch.object(moe_layer.gate, "update_bias") as mock_update_bias:
        model.update_moe_gate_bias()

    mock_update_bias.assert_not_called()


def test_tiny_kimi_with_kda_initializes_fp32_params_when_fla_available():
    _require_fla()

    model = KimiLinear48BForCausalLM(_tiny_kimi_config(use_kda=True), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    kda_attn = model.model.layers["0"].self_attn

    assert model.model.layers["0"].is_linear_attn
    assert kda_attn.A_log.dtype == torch.float32
    assert kda_attn.dt_bias.dtype == torch.float32
    assert torch.isfinite(kda_attn.A_log).all()
    assert torch.isfinite(kda_attn.dt_bias).all()
    _assert_floating_state_finite(model)


def test_kimi_moe_uses_hf_routing_numerics():
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    moe_layer = model.model.layers["1"].mlp

    assert moe_layer.gate.router_weights_fp32
    assert moe_layer.gate.router_weight_uses_score_correction_bias
    assert moe_layer.experts.config.apply_router_weight_after_down


def test_kimi_defaults_gate_precision_without_mutating_backend_config():
    backend = _backend_config()

    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=backend)

    assert backend.gate_precision is None
    assert model.backend is not backend
    assert model.backend.gate_precision == torch.float32


def test_initialize_weights_respects_explicit_buffer_device_on_cpu():
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    explicit_device = torch.device("meta")

    with (
        patch("torch.cuda.is_available", return_value=False),
        patch.object(model.model, "init_weights") as mock_model_init,
    ):
        model.initialize_weights(buffer_device=explicit_device, dtype=torch.float32)

    mock_model_init.assert_called_once()
    assert mock_model_init.call_args.args[0] == explicit_device


def test_checkpoint_free_initialize_and_eval_forward_runs_hf_order_moe():
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    _assert_floating_state_finite(model)

    moe_layer = model.model.layers["1"]
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
    with (
        patch.object(moe_layer, "_moe_infer_hf_order", wraps=moe_layer._moe_infer_hf_order) as mock_hf_order,
        torch.inference_mode(),
    ):
        output = model(input_ids=input_ids)

    mock_hf_order.assert_called_once()
    assert output.logits.shape == (2, 3, model.vocab_size)
    assert torch.isfinite(output.logits).all()


def test_hf_order_eval_moe_matches_standard_grouped_experts_path():
    """Pin the eval-only HF-ordered expert loop to the canonical MoE path.

    ``KimiDecoderLayer._moe_infer_hf_order`` exists to reproduce HF's expert
    ordering at inference; this is what catches either path drifting from the
    other, so it must keep passing before the two are merged or one is dropped.
    """
    torch.manual_seed(0)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()
    moe_layer = model.model.layers["1"]
    moe = moe_layer.mlp
    hidden_states = torch.randn(2, 3, model.config.hidden_size)

    with torch.inference_mode():
        hf_order = moe_layer._moe_infer_hf_order(moe, moe.experts, hidden_states)
        standard = moe(hidden_states, None)

    torch.testing.assert_close(hf_order, standard, rtol=1e-5, atol=1e-6)


def test_packed_mask_blocks_attention_across_documents():
    torch.manual_seed(0)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32)
    perturbed = input_ids.clone()
    perturbed[0, 3:] = torch.tensor([11, 12, 13])

    with torch.inference_mode():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).logits
        changed = model(input_ids=perturbed, attention_mask=attention_mask).logits

    # Document 1 owns positions 0..2 and must not see the rewritten document 2.
    torch.testing.assert_close(baseline[:, :3], changed[:, :3], rtol=1e-5, atol=1e-6)
    assert not torch.allclose(baseline[:, 3:], changed[:, 3:])


def test_padding_only_query_rows_stay_finite():
    torch.manual_seed(0)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    input_ids = torch.tensor([[3, 4, 5, 0]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.int32)

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    assert torch.isfinite(logits).all()


def test_prepare_model_inputs_for_cp_returns_contiguous_cp_sharder():
    from nemo_automodel._transformers.models.kimi_linear.cp import shard_batch_for_kimi_cp
    from nemo_automodel.components.distributed.context_parallel.sharder import contiguous_local_indices

    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())

    updates = model.prepare_model_inputs_for_cp({"input_ids": torch.zeros(1, 4, dtype=torch.long)})

    assert set(updates) == {"cp_sharder"}
    sharder = updates["cp_sharder"]
    assert sharder.shard_batch is shard_batch_for_kimi_cp
    # KDA's recurrent state needs contiguous per-rank slices, not the
    # load-balanced head/tail layout the framework uses by default.
    assert sharder.local_token_global_indices is contiguous_local_indices
    assert model._owns_cp_attention is True


class _StubDeviceMesh(dict):
    """Minimal ``cp``-only device mesh, as the CP dispatch indexes it."""

    def __init__(self, cp_size: int, cp_rank: int = 0) -> None:
        super().__init__()
        self["cp"] = _FakeCPMesh(cp_size, cp_rank)
        self.mesh_dim_names = ["cp"]


def test_cp_dispatch_shards_kimi_batch_contiguously_not_load_balanced():
    """The generic CP dispatch must pick up Kimi's sharder, not the default layout.

    Kimi's KDA layers are silently wrong under the framework's default head/tail
    round-robin layout, so this asserts the public sharder entry point resolves
    the model hook and produces contiguous rank-1 tokens plus the document map.
    """
    from nemo_automodel.components.distributed.context_parallel.sharder import ContextParallelSharder

    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
    batch = {"input_ids": input_ids, "labels": input_ids.clone()}

    sharder = ContextParallelSharder(model, _StubDeviceMesh(cp_size=2, cp_rank=1), batch)
    _, sharded = sharder.shard(batch)

    # Contiguous second half, not the round-robin head/tail pair ([2, 3, 4, 5]).
    assert sharded["input_ids"].tolist() == [[4, 5, 6, 7]]
    assert sharded["kimi_packed_context"].seq_start == 4
    assert sharder.shard_layout.padded_seq_len == 8


def test_setup_cp_attention_records_mesh_on_every_attention_block():
    _require_fla()
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(use_kda=True), backend=_backend_config())
    sentinel = object()

    for block in model.model.layers.values():
        block.self_attn.setup_cp_attention(sentinel)

    assert [block.self_attn._cp_mesh for block in model.model.layers.values()] == [sentinel, sentinel]


def test_thd_packed_inputs_run_through_the_batched_layers():
    """THD packing squeezes the batch axis off; the model must restore it."""
    torch.manual_seed(0)
    model = KimiLinear48BForCausalLM(_tiny_kimi_config(), backend=_backend_config())
    model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)
    model.eval()

    input_ids = torch.tensor([[3, 4, 5, 6, 7, 8]], dtype=torch.long)
    cu_seqlens = torch.tensor([[0, 3, 6]], dtype=torch.int32)

    with torch.inference_mode():
        thd = model(
            input_ids=input_ids,
            position_ids=torch.tensor([[0, 1, 2, 0, 1, 2]], dtype=torch.long),
            qkv_format="thd",
            cu_seqlens=cu_seqlens,
        ).logits
        bshd = model(
            input_ids=input_ids,
            attention_mask=torch.tensor([[1, 1, 1, 2, 2, 2]], dtype=torch.int32),
        ).logits

    assert thd.shape == (1, 6, model.vocab_size)
    assert torch.isfinite(thd).all()
    # Both routes describe the same two documents, so they must agree.
    torch.testing.assert_close(thd, bshd, rtol=1e-5, atol=1e-6)
