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

"""DSV4 attention through real cuDNN sparse kernels versus the eager masked path."""

import pytest
import torch
import torch.nn as nn

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.common.cudnn_sparse_attention import is_cudnn_sparse_attention_available
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.layers import (
    DeepseekV4Attention,
    DeepseekV4RotaryEmbedding,
    build_causal_padding_mask,
)

pytestmark = [
    pytest.mark.timeout(300),
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability()[0] < 9
        or not is_cudnn_sparse_attention_available(),
        reason="requires SM90+ and cuDNN Frontend sparse attention",
    ),
]

_BATCH, _SEQUENCE = 2, 132
_FP32_SUBMODULES = {
    "sinks_param",
    "compressor.wkv",
    "compressor.wgate",
    "compressor.ape_param",
    "compressor.indexer.wkv",
    "compressor.indexer.wgate",
    "compressor.indexer.ape_param",
}


def _config(ratio: int) -> DeepseekV4Config:
    """Tiny attention whose head dimension matches the cuDNN sparse kernels.

    Args:
        ratio: Compression ratio of the single attention layer.

    Returns:
        Config with a 128-token sliding window, so 132-token sequences exercise
        windowed keys, and an indexer top-k covering the complete ratio-4 pool.
    """
    return DeepseekV4Config(
        vocab_size=32,
        hidden_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=16,
        o_lora_rank=16,
        o_groups=1,
        n_routed_experts=2,
        n_shared_experts=0,
        num_experts_per_tok=1,
        max_position_embeddings=256,
        compress_ratios=[ratio],
        sliding_window=128,
        index_head_dim=128,
        index_n_heads=2,
        # Selecting the whole causal pool avoids top-k tie differences between backends.
        index_topk=_SEQUENCE // 4,
        attention_dropout=0.0,
        num_hash_layers=0,
        hc_mult=1,
        num_nextn_predict_layers=0,
        rms_norm_eps=1e-6,
        torch_dtype="bfloat16",
    )


def _attention(config: DeepseekV4Config, attn: str, dtype: torch.dtype, source: nn.Module | None) -> nn.Module:
    """Build one attention layer on CUDA with the production precision split.

    Args:
        config: Single-layer attention config.
        attn: Attention backend, ``"sdpa"`` for the eager masked path or ``"cudnn"``.
        dtype: Parameter dtype outside the FP32 compressor projections and sinks.
        source: Module whose state is copied, or None to initialize new weights.

    Returns:
        Attention module in training mode with a frozen ratio-4 indexer.
    """
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", rope_fusion=False)
    module = DeepseekV4Attention(config, layer_idx=0, backend=backend).cuda()
    if source is None:
        torch.manual_seed(1301)
        module.init_weights(torch.device("cuda"), init_std=0.2)
        # Zero-initialized sinks would leave their per-head contribution untested.
        with torch.no_grad():
            module.sinks_param.weight.normal_()
    else:
        module.load_state_dict(source.state_dict())
    module.to(dtype)
    # The production FSDP policy keeps exactly these attention submodules in FP32.
    for name, submodule in module.named_modules():
        if name in _FP32_SUBMODULES:
            submodule.float()
    if module.compressor is not None and module.compressor.indexer is not None:
        module.compressor.indexer.requires_grad_(False)
    return module.train()


def _run(
    module: nn.Module,
    hidden: torch.Tensor,
    upstream: torch.Tensor,
    rotary: tuple[nn.Module, nn.Module],
    masked: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Run forward/backward and one SGD step.

    Args:
        module: Attention layer.
        hidden: Input of shape [batch, sequence, hidden] in the module activation dtype.
        upstream: Output cotangent with hidden's shape.
        rotary: Main and compressed rotary modules.
        masked: Pass the explicit causal sliding-window mask; otherwise use the
            canonical unmasked path whose queries are proven nonempty.

    Returns:
        Output [batch, sequence, hidden], input gradient with the same shape,
        and a name-to-tensor map of parameter gradients.
    """
    x = hidden.detach().clone().requires_grad_()
    positions = torch.arange(_SEQUENCE, device="cuda").unsqueeze(0).expand(_BATCH, -1)
    main, compressed = rotary
    kwargs = {}
    mask = None
    if masked:
        mask = build_causal_padding_mask(
            None, seq_len=_SEQUENCE, dtype=x.dtype, device=x.device, batch_size=_BATCH, sliding_window=128
        )
    else:
        kwargs["_dsv4_all_queries_nonempty"] = True
    output, _ = module(
        x,
        position_embeddings=main(x, positions),
        attention_mask=mask,
        position_embeddings_compress=compressed(x, positions),
        rotary_compress=compressed,
        position_ids=positions,
        **kwargs,
    )
    output.backward(upstream)
    grads = {name: p.grad.detach().clone() for name, p in module.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for p in module.parameters():
            if p.requires_grad:
                p.sub_(0.5 * p.grad)
    return output.detach(), x.grad.detach(), grads


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error of actual against expected, computed in FP64."""
    return ((actual.double() - expected.double()).norm() / expected.double().norm()).item()


@pytest.mark.parametrize("ratio", [0, 4, 128])
def test_cudnn_attention_matches_eager_masked_attention(ratio: int) -> None:
    config = _config(ratio)
    torch.manual_seed(1303)
    seed_module = _attention(config, "sdpa", torch.float32, None)
    rotary = tuple(
        DeepseekV4RotaryEmbedding(
            rope_theta=float(theta), head_dim=512, partial_rotary_factor=64 / 512, rope_scaling=None
        ).cuda()
        for theta in (config.rope_theta, config.compress_rope_theta)
    )
    with torch.no_grad():
        # Every run shares BF16-representable weights, including the FP32 oracle.
        for parameter in seed_module.parameters():
            parameter.copy_(parameter.bfloat16())
    hidden = torch.randn(_BATCH, _SEQUENCE, config.hidden_size, device="cuda").bfloat16()
    upstream = torch.randn_like(hidden)

    oracle = _attention(config, "sdpa", torch.float32, seed_module)
    expected = _run(oracle, hidden.float(), upstream.float(), rotary, masked=True)
    runs = {}
    modules = {}
    for name, attn, masked in (("eager", "sdpa", True), ("cudnn", "cudnn", False), ("cudnn_masked", "cudnn", True)):
        modules[name] = _attention(config, attn, torch.bfloat16, seed_module)
        runs[name] = _run(modules[name], hidden, upstream, rotary, masked)

    # BF16 tolerance: cuDNN must be no less accurate than the existing BF16
    # eager path, both measured against the FP32 oracle.
    for name in ("cudnn", "cudnn_masked"):
        output, input_grad, grads = runs[name]
        eager_output, eager_input_grad, eager_grads = runs["eager"]
        for actual, reference, label in (
            (output, eager_output, "output"),
            (input_grad, eager_input_grad, "input gradient"),
            *((grads[key], eager_grads[key], key) for key in eager_grads),
        ):
            assert torch.isfinite(actual).all(), f"{name} {label}"
        for actual, eager, oracle_value, label in (
            (output, eager_output, expected[0], "output"),
            (input_grad, eager_input_grad, expected[1], "input gradient"),
            *((grads[key], eager_grads[key], expected[2][key], key) for key in eager_grads),
        ):
            error = _relative_l2(actual, oracle_value)
            eager_error = _relative_l2(eager, oracle_value)
            assert error <= 2 * eager_error + 2e-3, f"{name} {label}: {error:.3e} vs eager {eager_error:.3e}"

    # The two cuDNN branches select identical keys; only empty-row filtering
    # differs. The forward is deterministic. Backward accumulates key/value
    # gradients atomically, so repeated identical runs already differ by up to
    # ~3e-4 relative L2; 1e-3 still rejects any branch-dependent gradient.
    masked, unmasked = runs["cudnn_masked"], runs["cudnn"]
    torch.testing.assert_close(masked[0], unmasked[0], rtol=0, atol=0)
    for actual, reference in ((masked[1], unmasked[1]), *((masked[2][k], unmasked[2][k]) for k in unmasked[2])):
        assert _relative_l2(actual, reference) <= 1e-3

    # One SGD step moves every trainable parameter like the FP32 oracle, within
    # the eager BF16 error, and leaves the frozen indexer untouched.
    initial = dict(seed_module.named_parameters())

    def update(module: nn.Module, key: str) -> torch.Tensor:
        """Return the step's FP32 parameter change, with the parameter's own shape."""
        return dict(module.named_parameters())[key].float() - initial[key].float()

    for name in ("cudnn", "cudnn_masked"):
        for key in runs["eager"][2]:
            error = _relative_l2(update(modules[name], key), update(oracle, key))
            eager_error = _relative_l2(update(modules["eager"], key), update(oracle, key))
            assert error <= 2 * eager_error + 2e-3, f"{name} {key} update: {error:.3e} vs eager {eager_error:.3e}"
        if ratio == 4:
            for key, parameter in modules[name].compressor.indexer.named_parameters(prefix="compressor.indexer"):
                assert parameter.grad is None, key
                torch.testing.assert_close(parameter.float(), initial[key], rtol=0, atol=0)
