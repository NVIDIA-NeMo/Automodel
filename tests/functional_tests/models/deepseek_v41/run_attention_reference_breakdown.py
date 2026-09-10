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

"""Localize released layer0 attention and mHC differences on actual prefix inputs.

The pinned oracle stays unchanged. Component replay calls its original methods
and verifies the final output against the independently saved streaming oracle.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch
from run_reference_parity import _CheckpointReader, _import_reference, _load_official, _tensor_metrics
from safetensors.torch import load_file
from torch import nn


def main() -> None:
    """Measure each stage on exactly the released pretrained layer0 input."""
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.deepseek_v4.optimized_kernels import dsv4_sparse_attention
    from nemo_automodel.components.models.deepseek_v41.attention import DeepseekV41Attention, _apply_rope
    from nemo_automodel.components.models.deepseek_v41.config import DeepseekV41Config
    from nemo_automodel.components.models.deepseek_v41.layers import DeepseekV41HyperConnection
    from nemo_automodel.components.models.deepseek_v41.quantization import quantize_cache
    from nemo_automodel.components.models.deepseek_v41.state_dict_adapter import dequantize_checkpoint_weight

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("highest")
    source = load_file(str(options.artifact), device="cpu")
    x = source["layer.0.attn_norm"].to(device)
    reference = _import_reference(options.checkpoint / "inference")
    values = json.loads((options.checkpoint / "inference/config.json").read_text())
    values.update(dtype="bf16", expert_dtype=None, max_batch_size=1, max_seq_len=x.shape[1])
    args = reference.ModelArgs(**values)
    reference.world_size, reference.rank, reference.default_dtype = 1, 0, torch.bfloat16
    config = DeepseekV41Config.from_pretrained(options.checkpoint).text_config
    with torch.device(device), reference.set_dtype(torch.bfloat16):
        official = reference.Attention(0, args).eval().requires_grad_(False)
        native = DeepseekV41Attention(config, 0, BackendConfig(attn="tilelang", linear="torch", rms_norm="torch_fp32"))
        hc = DeepseekV41HyperConnection(config).eval().requires_grad_(False)
    owner = nn.Module()
    owner.layers = nn.ModuleDict({"0": nn.Module()})
    owner.layers["0"].attn = official
    official_audit = _load_official(owner, options.checkpoint)
    reader = _CheckpointReader(options.checkpoint)
    with torch.no_grad():
        for name, destination in native.named_parameters():
            original = "attn_sink" if name == "sinks_param.weight" else name
            key = f"layers.0.attn.{original}"
            weight = reader.tensor(key)
            scale_key = key.removesuffix("weight") + "scale"
            if key.endswith("weight") and scale_key in reader.weight_map:
                weight = dequantize_checkpoint_weight(
                    weight.to(device), reader.tensor(scale_key).to(device), dtype=destination.dtype
                )
            destination.copy_(weight)
        for name, destination in hc.named_parameters():
            destination.copy_(reader.tensor(f"layers.0.hc_attn_{name}"))
    reader.close()
    metrics = {}

    def compare(name: str, expected: torch.Tensor, actual: torch.Tensor) -> None:
        """Record arbitrary paired stage tensors and the count of unequal elements."""
        entry = asdict(_tensor_metrics(expected, actual))
        entry["unequal_elements"] = torch.count_nonzero(expected != actual.to(expected.device)).item()
        metrics[name] = entry
        logging.info("%s: %s", name, entry)

    with torch.inference_mode(), torch.device(device):
        positions = torch.arange(x.shape[1], device=device).unsqueeze(0)
        angles = native.rotary_emb(positions)
        freqs = official.freqs_cis[: x.shape[1]]
        compare("rope.cos", freqs.real.unsqueeze(0), angles.cos())
        compare("rope.sin", freqs.imag.unsqueeze(0), angles.sin())
        oqr, nqr = official.q_norm(official.wq_a(x)), native.q_norm(native.wq_a(x))
        compare("query_latent", oqr, nqr)
        oq = official.wq_b(oqr).unflatten(-1, (official.n_local_heads, official.head_dim))
        nq = native.wq_b(nqr).unflatten(-1, (native.num_heads, native.head_dim))
        compare("query_before_rope", oq, nq)
        reference.apply_rotary_emb(oq[..., -args.rope_head_dim :], freqs)
        nq = _apply_rope(nq, angles)
        compare("query_after_rope", oq, nq)
        okv, slots = official._window_kv(x, freqs, 0)
        nkv = quantize_cache(_apply_rope(native.kv_norm(native.wkv(x)), angles), format="fp8", block_size=32)
        compare("window_kv", okv, nkv)
        oo = reference.sparse_attn(oq, okv, official.attn_sink, slots, official.softmax_scale)
        no_common = dsv4_sparse_attention(
            oq, okv, native.attn_sink, slots, official.softmax_scale, backend="tilelang", reference_rounding=True
        )
        no = dsv4_sparse_attention(
            nq, nkv, native.attn_sink, slots, official.softmax_scale, backend="tilelang", reference_rounding=True
        )
        compare("kernel_common_inputs", oo, no_common)
        compare("kernel_actual_inputs", oo, no)
        reference.apply_rotary_emb(oo[..., -args.rope_head_dim :], freqs, True)
        no = _apply_rope(no, angles, inverse=True)
        compare("inverse_rope", oo, no)
        grouped = oo.view(1, x.shape[1], official.n_local_groups, -1)
        weight = official.wo_a.weight.view(official.n_local_groups, official.o_lora_rank, -1)
        oa = torch.einsum("bsgd,grd->bsgr", grouped, weight)
        compare("grouped_projection_common_inputs", oa, native.wo_a(grouped))
        na = native.wo_a(no.reshape(1, x.shape[1], native.num_groups, -1))
        compare("grouped_projection_actual_inputs", oa, na)
        out = official.wo_b(oa.flatten(2))
        compare("output_projection_common_inputs", out, native.wo_b(oa.flatten(2)))
        compare("output_projection_actual_inputs", out, native.wo_b(na.flatten(2)))
        compare("oracle_replay_matches_saved", source["layer.0.attn"].to(device), out)

        streams = source["layer.0.stream_input"].to(device)
        fake = SimpleNamespace(
            norm_eps=args.norm_eps, hc_mult=args.hc_mult, hc_sinkhorn_iters=args.hc_sinkhorn_iters, hc_eps=args.hc_eps
        )
        pre, post, comb = reference.Block.hc_mixes(fake, streams, hc.fn, hc.scale, hc.base)
        mix = hc(streams)
        compare("mhc.pre", pre, mix.pre)
        compare("mhc.post", post, mix.post)
        compare("mhc.comb", comb, mix.comb)
        expanded = reference.Block.hc_post(fake, out, streams, post, comb)
        compare("mhc.expand_common_inputs", expanded, hc.expand(out, streams, mix))
        collapsed = reference.Block.hc_pre(fake, expanded, pre)
        compare("mhc.collapse_common_inputs", collapsed, hc.collapse(expanded, mix.pre))
    query = nq.clone().requires_grad_(True)
    key_value = nkv.clone().requires_grad_(True)
    sink = native.attn_sink.detach().clone().requires_grad_(True)
    trained = dsv4_sparse_attention(
        query, key_value, sink, slots.clone(), official.softmax_scale, backend="tilelang", reference_rounding=True
    )
    trained.float().square().mean().backward()
    gradients = {
        name: bool(torch.isfinite(tensor.grad).all())
        for name, tensor in (("query", query), ("kv", key_value), ("sink", sink))
    }
    logging.info("Finite backward: %s", gradients)
    options.output.write_text(
        json.dumps(
            {"official_state_audit": asdict(official_audit), "stages": metrics, "gradients_finite": gradients}, indent=2
        )
        + "\n"
    )
    if not all(gradients.values()):
        raise ValueError("Sparse attention backward produced nonfinite gradients")


if __name__ == "__main__":
    main()
