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

"""Diagnose mHC coefficients on authentic released layer20 input streams.

The primary oracle remains the pinned, unchanged hc_split_sinkhorn. Alternative
native arithmetic is explicitly diagnostic and does not alter production code.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch
from run_decoder_reference_parity import _holder, _Options, _reference_runtime
from run_reference_parity import (
    REFERENCE_HASHES,
    REVISION,
    _CheckpointReader,
    _import_reference,
    _load_official,
    _tensor_metrics,
)
from safetensors import safe_open
from torch.nn import functional as F

from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sinkhorn_normalize
from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection, DeepseekV41Mix


def _metrics(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    return {
        **asdict(_tensor_metrics(expected, actual)),
        "exact": torch.equal(expected, actual),
        "changed_elements": int((expected != actual).sum()),
        "finite": bool(torch.isfinite(actual).all()),
    }


def _affine(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    pre = torch.sigmoid(mixes[..., :4] * scale[0] + base[:4]) + 1e-6
    post = 2 * torch.sigmoid(mixes[..., 4:8] * scale[1] + base[4:8])
    logits = mixes[..., 8:] * scale[2] + base[8:]
    return pre, post, logits.unflatten(-1, (4, 4))


def _torch_coefficients(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    pre, post, logits = _affine(mixes, scale, base)
    comb = logits.softmax(-1) + 1e-6
    comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    for _ in range(19):
        comb = comb / (comb.sum(-1, keepdim=True) + 1e-6)
        comb = comb / (comb.sum(-2, keepdim=True) + 1e-6)
    return pre, post, comb


def _affine_logits(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    scales = torch.cat((scale[:1].expand(4), scale[1:2].expand(4), scale[2:3].expand(16)))
    return mixes * scales + base


def _sigmoid_and_sinkhorn(logits: torch.Tensor):
    return (
        torch.sigmoid(logits[..., :4]) + 1e-6,
        2 * torch.sigmoid(logits[..., 4:8]),
        dsv4_sinkhorn_normalize(logits[..., 8:].unflatten(-1, (4, 4)), backend="tilelang", repeat=20, eps=1e-6),
    )


def _addcmul_coefficients(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor):
    scales = torch.cat((scale[:1].expand(4), scale[1:2].expand(4), scale[2:3].expand(16)))
    return _sigmoid_and_sinkhorn(torch.addcmul(base, mixes, scales))


def _gradient_report(mixes: torch.Tensor, scale: torch.Tensor, base: torch.Tensor) -> dict:
    """Compare existing TileKernels backward with the same smooth FP64 equations."""
    torch.manual_seed(719)
    inputs = [value.detach().clone().requires_grad_(True) for value in (mixes, scale, base)]
    reference_inputs = [value.detach().double().requires_grad_(True) for value in inputs]
    actual = _addcmul_coefficients(*inputs)
    expected = _torch_coefficients(*reference_inputs)
    upstream = [torch.randn_like(value) for value in actual]
    sum((value * gradient).sum() for value, gradient in zip(actual, upstream, strict=True)).backward()
    sum((value * gradient.double()).sum() for value, gradient in zip(expected, upstream, strict=True)).backward()
    report = {
        name: _metrics(expected_input.grad.float(), actual_input.grad)
        for name, expected_input, actual_input in zip(("mixes", "scale", "base"), reference_inputs, inputs, strict=True)
    }
    for expected_input, actual_input in zip(reference_inputs, inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad.float(), rtol=0.001, atol=0.0001)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--reference-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one available GPU for this coefficient diagnostic")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    reference = _import_reference(args.reference_dir)
    config = DeepseekV41Config.from_pretrained(args.checkpoint, local_files_only=True).text_config
    assert config.hc_mult == 4 and config.hc_sinkhorn_iters == 20 and config.hc_eps == 1e-6
    with safe_open(str(args.reference_artifact), framework="pt", device="cpu") as artifact:
        hidden = artifact.get_tensor("final_streams").cuda()
        previous_mix = artifact.get_tensor("final_pre_mix").cuda()
    reader = _CheckpointReader(args.checkpoint)
    fn = reader.tensor("layers.20.hc_attn_fn").cuda()
    scale = reader.tensor("layers.20.hc_attn_scale").cuda()
    base = reader.tensor("layers.20.hc_attn_base").cuda()
    attn_norm_weight = reader.tensor("layers.20.attn_norm.weight").cuda()
    ffn_norm_weight = reader.tensor("layers.20.ffn_norm.weight").cuda()
    reader.close()
    native = DeepseekV41HyperConnection(config).cuda()
    with torch.no_grad():
        native.fn.copy_(fn)
        native.scale.copy_(scale)
        native.base.copy_(base)
    flat = hidden.flatten(2).float()
    mixes = F.linear(flat, fn) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + config.rms_norm_eps)
    expected = reference.hc_split_sinkhorn(mixes, scale, base, 4, 20, 1e-6)
    actual = native(hidden)
    production_tilelang = DeepseekV41HyperConnection(config, sinkhorn_backend="tilelang").cuda()
    production_tilelang.load_state_dict(native.state_dict())
    production_mix = production_tilelang(hidden)
    compiled_affine = torch.compile(_affine, fullgraph=True)
    compiled_all = torch.compile(_torch_coefficients, fullgraph=True)
    affine_outputs = compiled_affine(mixes, scale, base)
    torch_affine = _affine(mixes, scale, base)
    compiled_logits = torch.compile(_affine_logits, fullgraph=True)(mixes, scale, base)
    scales = torch.cat((scale[:1].expand(4), scale[1:2].expand(4), scale[2:3].expand(16)))
    addcmul_logits = torch.addcmul(base, mixes, scales)
    variants = {
        "production": (actual.pre, actual.post, actual.comb),
        "production_tilelang": (production_mix.pre, production_mix.post, production_mix.comb),
        "torch_coefficients": _torch_coefficients(mixes, scale, base),
        "compiled_all": compiled_all(mixes, scale, base),
        "compiled_affine_tilekernels_sinkhorn": (
            *affine_outputs[:2],
            dsv4_sinkhorn_normalize(affine_outputs[2], backend="tilelang", repeat=20, eps=1e-6),
        ),
        "torch_affine_tilekernels_sinkhorn": (
            *torch_affine[:2],
            dsv4_sinkhorn_normalize(torch_affine[2], backend="tilelang", repeat=20, eps=1e-6),
        ),
        "compiled_affine_eager_sigmoid_tilekernels_sinkhorn": _sigmoid_and_sinkhorn(compiled_logits),
        "addcmul_affine_eager_sigmoid_tilekernels_sinkhorn": _sigmoid_and_sinkhorn(addcmul_logits),
    }
    results = {
        name: {part: _metrics(a, b) for part, a, b in zip(("pre", "post", "comb"), expected, values, strict=True)}
        for name, values in variants.items()
    }
    for part in results["production_tilelang"].values():
        assert part["exact"], "The production TileLang mHC must preserve the released coefficient rounding"
    gradient_hidden = hidden[:, :32].detach().clone().requires_grad_(True)
    gradient_mix = production_tilelang(gradient_hidden)
    sum(value.square().sum() for value in (gradient_mix.pre, gradient_mix.post, gradient_mix.comb)).backward()
    production_gradients = {
        name: {"finite": bool(torch.isfinite(value.grad).all()), "norm": float(value.grad.float().norm())}
        for name, value in (("hidden", gradient_hidden), *production_tilelang.named_parameters())
    }
    assert all(value["finite"] and value["norm"] > 0 for value in production_gradients.values())
    runtime_args = _reference_runtime(
        reference, _Options(args.checkpoint, args.reference_dir, args.output, sequence_length=hidden.shape[1])
    )
    with torch.device("cuda"), reference.set_dtype(torch.bfloat16):
        official_attention = reference.Attention(20, runtime_args)
        attn_norm = reference.RMSNorm(config.hidden_size, config.rms_norm_eps)
        ffn_norm = reference.RMSNorm(config.hidden_size, config.rms_norm_eps)
    with torch.no_grad():
        attn_norm.weight.copy_(attn_norm_weight)
        ffn_norm.weight.copy_(ffn_norm_weight)
    attention_audit = _load_official(
        _holder(official_attention, 20, native=False, attention_only=True), args.checkpoint
    )
    with torch.inference_mode(), torch.device("cuda"), reference.set_dtype(torch.bfloat16):
        attention_input = attn_norm(reference.Block.hc_pre(None, hidden, previous_mix))
        attention_output = official_attention(attention_input, 0)
        expanded = reference.Block.hc_post(None, attention_output, hidden, expected[1], expected[2])
        collapsed = reference.Block.hc_pre(None, expanded, expected[0])
        expansion_report = {}
        for name in ("production", "production_tilelang", "addcmul_affine_eager_sigmoid_tilekernels_sinkhorn"):
            mix = DeepseekV41Mix(*variants[name])
            native_expanded = native.expand(attention_output, hidden, mix)
            native_collapsed = native.collapse(native_expanded, mix.pre)
            expansion_report[name] = {
                "after_attention_expand": _metrics(expanded, native_expanded),
                "ffn_collapse": _metrics(collapsed, native_collapsed),
                "ffn_norm": _metrics(ffn_norm(collapsed), ffn_norm(native_collapsed)),
            }
    kernel = reference.hc_split_sinkhorn.__globals__["hc_split_sinkhorn_kernel"](4, 20, 1e-6)
    source_path = args.output.with_suffix(".official_kernel.cu")
    source_path.parent.mkdir(parents=True, exist_ok=True)
    source_path.write_text(kernel.get_kernel_source())
    report = {
        "reference_revision": REVISION,
        "reference_source_hashes": REFERENCE_HASHES,
        "reference_artifact": str(args.reference_artifact),
        "scope": "Authentic layer20 initial streams; unchanged official mHC split and native-only arithmetic alternatives",
        "source_hc_parameters_loaded_exactly": True,
        "source_kernel_generated_code": str(source_path),
        "input_shape": list(hidden.shape),
        "results": results,
        "real_attention_state_audit": asdict(attention_audit),
        "real_expand_collapse": expansion_report,
        "gradients_against_fp64_equations": _gradient_report(mixes, scale, base),
        "production_module_gradients": production_gradients,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
