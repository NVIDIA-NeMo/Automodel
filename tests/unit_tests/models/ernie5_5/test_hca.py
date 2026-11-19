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

"""Forward, gradient and input-guard tests for the ERNIE 5.5 HCA attention.

The numerical tests use rows of 300 tokens (two compressed keys) and non-zero
virtual-head and gate parameters unless a test states otherwise.
"""

from typing import Any
from unittest.mock import create_autospec

import pytest
import torch
import torch.nn.functional as F
from torch.distributed.device_mesh import DeviceMesh

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.utils import cast_model_to_dtype
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
)
from nemo_automodel.components.models.ernie5_5.checkpoint_layout import map_hca_to_hf
from nemo_automodel.components.models.ernie5_5.hca import Ernie5_5HCAAttention

# No test of this file uses CUDA.
pytestmark = pytest.mark.cpu_only

HIDDEN, HEADS, HEAD_DIM, ROPE, Q_RANK, O_RANK, WINDOW, RATIO = 16, 2, 8, 4, 6, 4, 8, 128
LONG = 300  # Two complete compression windows and a remainder.
FP32_NAMES = (
    "sinks_param",
    "compressor.wkv",
    "compressor.wgate",
    "compressor.ape_param",
    "vha_postmix_U",
    "vha_postmix_V",
)


def _config(
    dtype: torch.dtype = torch.float32,
    *,
    o_groups: int = 2,
    ratio: int = RATIO,
    dropout: float = 0.0,
    fuse_qnorm_rope: bool = False,
) -> DeepseekV4Config:
    """Build a one-layer attention config with two heads of eight channels."""
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=HIDDEN,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=HEADS,
        num_key_value_heads=1,
        head_dim=HEAD_DIM,
        qk_rope_head_dim=ROPE,
        q_lora_rank=Q_RANK,
        o_lora_rank=O_RANK,
        o_groups=o_groups,
        n_routed_experts=2,
        n_shared_experts=1,
        num_experts_per_tok=1,
        max_position_embeddings=1024,
        compress_ratios=[ratio],
        sliding_window=WINDOW,
        attention_dropout=dropout,
        num_hash_layers=0,
        hc_mult=1,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype=str(dtype).removeprefix("torch."),
        fuse_qnorm_rope=fuse_qnorm_rope,
        compress_rope_theta=160000.0,
    )


def _backend(attn: str = "sdpa", rope: str = "torch") -> BackendConfig:
    """Build a backend with torch experts, dispatcher and linear layers and the given attention and rotary names."""
    return BackendConfig(attn=attn, rope=rope, experts="torch", dispatcher="torch", linear="torch")


def _cast(module: DeepseekV4Attention, dtype: torch.dtype) -> None:
    """Cast a layer with ``cast_model_to_dtype``, keeping the parameters named in ``FP32_NAMES`` in FP32."""
    module._keep_in_fp32_modules_strict = list(FP32_NAMES)
    cast_model_to_dtype(module, dtype)


def _attention(dtype: torch.dtype = torch.float32, *, o_groups: int = 2, dropout: float = 0.0) -> Ernie5_5HCAAttention:
    """Build an HCA layer whose parameters, including virtual-head and gate ones, are all non-trivial."""
    config = _config(dtype, o_groups=o_groups, dropout=dropout)
    module = Ernie5_5HCAAttention(config, 0, _backend(), vha_rank=2)
    _cast(module, dtype)
    generator = torch.Generator().manual_seed(55)
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.copy_(1.0 + 0.1 * noise if name.endswith("norm.weight") else 0.4 * noise)
    return module


def _parent_twin(module: Ernie5_5HCAAttention, wo_a_weight: torch.Tensor | None = None) -> DeepseekV4Attention:
    """Build the DeepSeek V4 attention that shares every inherited parameter of ``module``.

    Args:
        module: ERNIE layer to copy the inherited parameters from.
        wo_a_weight: Optional replacement for the grouped projection weight, of shape
            [groups * o_lora_rank, heads * head_dim / groups].

    Returns:
        A ``DeepseekV4Attention`` in the training mode and parameter dtypes of ``module``.
    """
    dtype = module.wq_a.weight.dtype
    parent = DeepseekV4Attention(module.config, 0, _backend())
    _cast(parent, dtype)
    state = {name: value for name, value in module.state_dict().items() if not name.startswith(("vha.", "gate_proj."))}
    if wo_a_weight is not None:
        state["wo_a.weight"] = wo_a_weight
    parent.load_state_dict(state, strict=True)
    return parent.train(module.training)


def _inputs(
    sequence: int, *, batch: int = 1, dtype: torch.dtype = torch.float32
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Build hidden states and attention arguments for rows that each hold one document.

    Returns:
        Hidden states of shape [batch, sequence, hidden], and the keyword arguments of the attention
        call: compress-rope cosine and sine, each of shape [batch, sequence, rotary_channels], passed
        as both rotary tables; the rotary module; and the additive causal sliding-window mask of
        shape [batch, 1, sequence, sequence].
    """
    rotary = DeepseekV4RotaryEmbedding(
        rope_theta=160000.0,
        head_dim=HEAD_DIM,
        partial_rotary_factor=ROPE / HEAD_DIM,
        rope_scaling={
            "type": "yarn",
            "factor": 16,
            "original_max_position_embeddings": 65536,
            "beta_fast": 32,
            "beta_slow": 1,
        },
    )
    positions = torch.arange(sequence).expand(batch, -1)
    cos_sin = rotary(torch.empty(batch, sequence, 1, dtype=dtype), positions)
    mask = build_causal_padding_mask(
        None, seq_len=sequence, dtype=dtype, device=torch.device("cpu"), batch_size=batch, sliding_window=WINDOW
    )
    hidden = torch.randn(batch, sequence, HIDDEN, generator=torch.Generator().manual_seed(7)).to(dtype)
    kwargs = {
        "position_embeddings": cos_sin,
        "position_embeddings_compress": cos_sin,
        "rotary_compress": rotary,
        "attention_mask": mask,
    }
    return hidden, kwargs


@pytest.mark.parametrize(
    ("dtype", "sequence", "dropout"),
    [
        (torch.float32, 64, 0.0),
        (torch.float32, LONG, 0.0),
        (torch.float32, LONG, 0.1),
        (torch.bfloat16, 64, 0.0),
        (torch.bfloat16, LONG, 0.0),
    ],
)
def test_identity_additions_give_exactly_half_of_parent(dtype: torch.dtype, sequence: int, dropout: float) -> None:
    """A zero mixing factor and a zero gate weight leave the parent's forward scaled by sigmoid(0)."""
    module = _attention(dtype, dropout=dropout)
    with torch.no_grad():
        module.vha_postmix_V.zero_()
        module.gate_proj.weight.zero_()
    parent = _parent_twin(module)
    hidden, kwargs = _inputs(sequence, batch=2, dtype=dtype)
    ernie_hidden = hidden.clone().requires_grad_(True)
    parent_hidden = hidden.clone().requires_grad_(True)

    torch.manual_seed(3)
    output, probabilities = module(ernie_hidden, **kwargs)
    torch.manual_seed(3)
    parent_output, parent_probabilities = parent(parent_hidden, **kwargs)
    output.float().sum().backward()
    parent_output.float().sum().backward()

    # Halving is exact in binary floating point, so the comparison is bitwise.
    assert torch.equal(2 * output, parent_output)
    assert torch.equal(probabilities, parent_probabilities)
    assert torch.equal(2 * ernie_hidden.grad, parent_hidden.grad)
    assert torch.equal(2 * module.wq_a.weight.grad, parent.wq_a.weight.grad)


def test_virtual_head_mix_permutes_heads_before_the_grouped_projection() -> None:
    """With ``V U^T = P - I`` the mix is the head swap ``P``, which equals swapping the projection's inputs."""
    module = _attention(o_groups=1)
    with torch.no_grad():
        module.vha_postmix_U.copy_(torch.eye(2))
        module.vha_postmix_V.copy_(torch.tensor([[-1.0, 1.0], [1.0, -1.0]]))
        module.gate_proj.weight.zero_()
    weight = module.wo_a.weight.detach()
    swapped_inputs = torch.cat([weight[:, HEAD_DIM:], weight[:, :HEAD_DIM]], dim=1)
    parent = _parent_twin(module, wo_a_weight=swapped_inputs)
    unswapped_parent = _parent_twin(module)
    hidden, kwargs = _inputs(LONG)

    output, _ = module(hidden, **kwargs)
    expected, _ = parent(hidden, **kwargs)
    unswapped, _ = unswapped_parent(hidden, **kwargs)

    # The two sides add the same products in a different order.
    torch.testing.assert_close(2 * output, expected, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(2 * output, unswapped, rtol=1e-2, atol=1e-3)


def test_gate_scales_the_grouped_projection_by_the_sigmoid_of_the_query_latent() -> None:
    """The output is ``wo_b(wo_a(...) * sigmoid(gate_proj(q_norm(...))))``."""
    module = _attention()
    hidden, kwargs = _inputs(LONG)
    seen: dict[str, torch.Tensor] = {}
    handles = [
        module.q_norm.register_forward_hook(lambda _module, _args, result: seen.update(latent=result)),
        module.wo_a.register_forward_hook(lambda _module, _args, result: seen.update(projected=result)),
    ]
    output, _ = module(hidden, **kwargs)
    for handle in handles:
        handle.remove()

    gate = torch.sigmoid(F.linear(seen["latent"], module.gate_proj.weight))
    expected = F.linear(seen["projected"].flatten(2) * gate, module.wo_b.weight)
    assert float(gate.max() - gate.min()) > 0.5
    torch.testing.assert_close(output, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("name", ["vha.vha_postmix_U", "vha.vha_postmix_V", "gate_proj.weight"])
def test_output_changes_with_virtual_head_and_gate_parameters(name: str) -> None:
    """Each ERNIE parameter takes part in the forward."""
    module = _attention()
    hidden, kwargs = _inputs(LONG)
    before, _ = module(hidden, **kwargs)
    with torch.no_grad():
        module.get_parameter(name).add_(0.25)
    after, _ = module(hidden, **kwargs)
    assert float((after - before).abs().max()) > 1e-3


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_gradients_reach_virtual_head_gate_and_compressor_parameters(dtype: torch.dtype) -> None:
    """Every parameter of a row with compressed keys receives a finite, non-zero gradient."""
    module = _attention(dtype)
    hidden, kwargs = _inputs(LONG, batch=2, dtype=dtype)
    hidden.requires_grad_(True)
    output, _ = module(hidden, **kwargs)
    output.float().square().sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.dtype == parameter.dtype, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name
    assert torch.isfinite(hidden.grad).all() and torch.count_nonzero(hidden.grad) > 0


def test_row_shorter_than_one_window_does_not_use_the_compressor() -> None:
    """Fewer than 128 tokens give no compressed key, and the compressor stays out of the graph."""
    module = _attention()
    hidden, kwargs = _inputs(100, batch=2)
    output, probabilities = module(hidden, **kwargs)
    output.square().sum().backward()

    assert output.shape == (2, 100, HIDDEN) and torch.isfinite(output).all()
    assert probabilities.shape == (2, HEADS, 100, 100)
    assert all(parameter.grad is None for parameter in module.compressor.parameters())
    assert module.vha_postmix_V.grad is not None and module.gate_proj.weight.grad is not None
    with torch.no_grad():
        module.compressor.ape_param.weight.add_(1.0)
        module.compressor.wkv.weight.add_(1.0)
    changed, _ = module(hidden, **kwargs)
    assert torch.equal(changed, output)


def test_compressed_keys_are_visible_only_after_their_window_is_complete() -> None:
    """Compressed key ``p`` is attended exactly by the queries at or after token ``128 * (p + 1) - 1``."""
    module = _attention().eval()
    hidden, kwargs = _inputs(LONG)
    with torch.no_grad():
        output, probabilities = module(hidden, **kwargs)
        assert probabilities.shape == (1, HEADS, LONG, LONG + 2)
        for key, first_query in ((0, RATIO - 1), (1, 2 * RATIO - 1)):
            column = probabilities[0, :, :, LONG + key]
            assert torch.count_nonzero(column[:, :first_query]) == 0
            assert bool((column[:, first_query:] > 0).all())

        # Token 200 is in window 1. Earlier outputs must not move; a query beyond the sliding
        # window of token 200 sees the change only through compressed key 1.
        changed_hidden = hidden.clone()
        changed_hidden[0, 200] += 1.0
        changed, _ = module(changed_hidden, **kwargs)
    moved = (changed[0] - output[0]).abs().amax(dim=-1)
    assert torch.equal(changed[0, :200], output[0, :200])
    assert float(moved[200 : 200 + WINDOW].min()) > 0.0
    assert float(moved[200 + WINDOW : 2 * RATIO - 1].max()) == 0.0
    assert float(moved[2 * RATIO - 1 :].min()) > 0.0


@pytest.mark.parametrize(
    "mask_shape",
    [
        (2, 1, LONG, LONG - 1),  # Too few key columns: the missing ones would be padded as visible.
        (2, 1, LONG, LONG + 1),  # Too many key columns: the compressed-key mask would be misplaced.
        (2, 1, LONG, RATIO),
        (2, 1, LONG - 1, LONG),
        (1, 1, LONG, LONG),
        (2, LONG, LONG),
        (2, LONG),
    ],
)
def test_attention_mask_with_a_wrong_shape_is_rejected(mask_shape: tuple[int, ...]) -> None:
    """The mask must be [batch, 1 or heads, sequence, sequence]; nothing is padded or cropped."""
    module = _attention()
    hidden, kwargs = _inputs(LONG, batch=2)
    kwargs["attention_mask"] = torch.zeros(mask_shape)
    with pytest.raises(ValueError, match="explicit 4D additive attention mask"):
        module(hidden, **kwargs)


def test_missing_boolean_and_wrong_head_count_masks_are_rejected() -> None:
    """A mask is required, must be additive, and must have one head or one per attention head."""
    module = _attention()
    hidden, kwargs = _inputs(LONG)
    mask = kwargs.pop("attention_mask")
    with pytest.raises(ValueError, match="explicit 4D additive attention mask"):
        module(hidden, **kwargs)
    with pytest.raises(ValueError, match="explicit 4D additive attention mask"):
        module(hidden, **kwargs, attention_mask=mask == 0)
    with pytest.raises(ValueError, match="1 or 2 heads"):
        module(hidden, **kwargs, attention_mask=mask.expand(1, 3, LONG, LONG))


def test_per_head_mask_matches_the_shared_mask() -> None:
    """A mask with one plane per head is extended over the compressed keys like the shared one."""
    module = _attention().eval()
    hidden, kwargs = _inputs(LONG, batch=2)
    per_head = dict(kwargs, attention_mask=kwargs["attention_mask"].expand(2, HEADS, LONG, LONG).contiguous())
    with torch.no_grad():
        shared_output, shared_probabilities = module(hidden, **kwargs)
        output, probabilities = module(hidden, **per_head)
    assert torch.equal(output, shared_output)
    assert torch.equal(probabilities, shared_probabilities)


def test_inputs_outside_the_single_document_path_are_rejected() -> None:
    """Packed rows, offsets, explicit positions, context parallelism and unknown arguments all raise."""
    module = _attention()
    hidden, kwargs = _inputs(LONG)
    positions = torch.arange(LONG).unsqueeze(0)
    with pytest.raises(NotImplementedError, match="Packed"):
        module(hidden, **kwargs, packed_seq_ids=torch.ones(1, LONG, dtype=torch.long))
    with pytest.raises(NotImplementedError, match="Nonzero"):
        module(hidden, **kwargs, start_pos=1)
    with pytest.raises(NotImplementedError, match="position_ids"):
        module(hidden, **kwargs, position_ids=positions)
    with pytest.raises(NotImplementedError, match="position_ids"):
        module(hidden, **kwargs, position_ids=positions % RATIO)
    with pytest.raises(NotImplementedError, match="context parallelism"):
        module(hidden, **kwargs, _dsv4_cp_group=object())
    with pytest.raises(TypeError, match="vision_token_types"):
        module(hidden, **kwargs, vision_token_types=torch.zeros(1, LONG, dtype=torch.long))
    with pytest.raises(TypeError, match="_dsv4_complete_compressor_windows"):
        module(hidden, **kwargs, _dsv4_complete_compressor_windows=True)
    for name in ("position_embeddings_compress", "rotary_compress"):
        incomplete = {key: value for key, value in kwargs.items() if key != name}
        with pytest.raises(ValueError, match="position_embeddings_compress and rotary_compress"):
            module(hidden, **incomplete)
    output, _ = module(hidden, **kwargs, packed_seq_ids=None, _dsv4_cp_group=None)
    assert torch.isfinite(output).all()


def test_context_parallel_setup_is_rejected() -> None:
    """The inherited context-parallel hook must not silently enable an unvalidated path."""
    module = _attention()
    with pytest.raises(NotImplementedError, match="context parallelism"):
        module.setup_cp_attention(create_autospec(DeviceMesh, instance=True))


@pytest.mark.parametrize(
    ("attn", "rope", "message"),
    [
        ("tilelang", "torch", "got attn='tilelang'"),
        ("cudnn", "torch", "got attn='cudnn'"),
        ("te", "torch", "got attn='te'"),
        ("flex", "torch", "got attn='flex'"),
        ("fa4", "torch", "got attn='fa4'"),
        ("eagre", "torch", "got attn='eagre'"),
        ("eager", "triton", "Triton rotary kernel"),
    ],
)
def test_backends_without_a_reference_comparison_are_rejected(attn: str, rope: str, message: str) -> None:
    """Every attention name outside the dense allow-list, and the Triton rotation, fail at construction."""
    backend = _backend()
    backend.attn, backend.rope = attn, rope
    with pytest.raises(NotImplementedError, match=message):
        Ernie5_5HCAAttention(_config(), 0, backend, vha_rank=2)


@pytest.mark.parametrize("attn", ["eager", "torch", "sdpa"])
def test_dense_backend_names_select_the_same_eager_path(attn: str) -> None:
    """The three accepted names give bitwise-identical outputs; none selects a different kernel."""
    hidden, kwargs = _inputs(LONG)
    torch.manual_seed(3)
    reference = Ernie5_5HCAAttention(_config(), 0, _backend(attn="eager"), vha_rank=2)
    torch.manual_seed(3)
    module = Ernie5_5HCAAttention(_config(), 0, _backend(attn=attn), vha_rank=2)
    module.load_state_dict(reference.state_dict())
    expected, _ = reference(hidden, **kwargs)
    actual, _ = module(hidden, **kwargs)
    assert torch.equal(actual, expected)
    assert Ernie5_5HCAAttention(_config(), 0, None, vha_rank=2).backend.attn == "eager"


def test_unsupported_configurations_are_rejected() -> None:
    """Only ratio-128 layers with a positive mixing rank and unfused query rotation are built."""
    with pytest.raises(NotImplementedError, match="fuse_qnorm_rope"):
        Ernie5_5HCAAttention(_config(fuse_qnorm_rope=True), 0, _backend(), vha_rank=2)
    with pytest.raises(ValueError, match="compression ratio 128; layer 0 has 4"):
        Ernie5_5HCAAttention(_config(ratio=4), 0, _backend(), vha_rank=2)
    with pytest.raises(ValueError, match="vha_rank must be positive; got 0"):
        Ernie5_5HCAAttention(_config(), 0, _backend(), vha_rank=0)


def test_parameter_names_match_the_checkpoint_mapping_and_init_weights_resets_the_additions() -> None:
    """The layer keeps the 15 mapped parameters; ``init_weights`` zeroes V and redraws U and the gate."""
    module = _attention()
    assert len(map_hca_to_hf(module.state_dict())) == 15
    module.init_weights(torch.device("cpu"))
    assert torch.count_nonzero(module.vha_postmix_V) == 0
    assert torch.count_nonzero(module.vha_postmix_U) > 0
    assert 0.0 < float(module.gate_proj.weight.abs().max()) < 0.2
