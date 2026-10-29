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

"""Stream original released layers on two devices with independent weight reads.

Only metrics and input IDs are exported; source weight payloads stay on the cluster.
Use --layers 4 --tokens 4096 for full-vocabulary prefix coverage, or --layers 72
with a bounded document for full-depth teacher-forced numerical coverage.
"""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from torch.nn import functional as F

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral_large4.attention import SequenceLayout
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4Block
from nemo_automodel.components.models.mistral_large4.rope_utils import MistralLarge4Rotary
from nemo_automodel.components.models.mistral_large4.state_dict_adapter import MistralLarge4StateDictAdapter
from tests.unit_tests.models.mistral_large4.math_reference import rms
from tests.unit_tests.models.mistral_large4.released_reference import reference_block


class ReleasedReader:
    """Own independent native-file reads, hashes, and bounded decode scratch."""

    def __init__(self, source: Path, device: torch.device, dtype: torch.dtype, *, native: bool) -> None:
        self.source, self.device, self.dtype, self.native = source, device, dtype, native
        self.index = json.loads((source / "consolidated.safetensors.index.json").read_text())["weight_map"]
        self.hashes: dict[str, str] = {}
        self.bytes = 0

    def read(self, key: str) -> torch.Tensor:
        """Return independent source storage with original tensor shape and dtype."""
        with safe_open(self.source / self.index[key], framework="pt", device="cpu") as handle:
            value = handle.get_tensor(key)
        self.hashes[key] = hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()
        self.bytes += value.numel() * value.element_size()
        return value.to(self.device)

    def weight(self, key: str) -> torch.Tensor:
        """Return a decoded matrix [output, input] or original vector [channels]."""
        raw = self.read(key)
        sensitive = key == "lerope.alpha" or key.endswith("gate.weight") or ".indexer.k_norm." in key
        dtype = torch.float32 if sensitive else self.dtype
        if raw.dtype != torch.float8_e4m3fn:
            return raw.to(dtype)
        scale_key = key.removesuffix(".weight") + ".qscale_weight"
        scales = self.read(scale_key)
        if self.native:
            decoded = MistralLarge4StateDictAdapter(dtype=dtype).from_hf({key: raw, scale_key: scales})
            return next(iter(decoded.values()))
        # Independent numeric FP32 block-grid reconstruction; no adapter helpers.
        decoded = torch.empty(raw.shape, device=self.device, dtype=dtype)
        rows, columns = raw.shape
        if rows % 128 or columns % 128:
            raise ValueError(f"Released matrix is not block aligned: {key}")
        for start in range(0, rows, 1024):
            stop = min(start + 1024, rows)
            tile = raw[start:stop].float().reshape((stop - start) // 128, 128, columns // 128, 128)
            decoded[start:stop] = (tile * scales[start // 128 : stop // 128, None, :, None]).reshape(
                stop - start, columns
            )
        return decoded

    def layer(self, layer: int) -> dict[str, torch.Tensor]:
        """Return source-key layer tensors, each matrix [output, input] or vector."""
        return {
            key: self.weight(key)
            for key in self.index
            if key.startswith(f"layers.{layer}.") and not key.endswith(".qscale_weight")
        }


def difference(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float]:
    """Compare same-layout arbitrary-rank tensors with FP64 metric accumulation.

    Args:
        actual: Floating tensor [...], arbitrary leading dimensions.
        expected: Independently computed floating tensor of identical shape.

    Returns:
        Scalar maximum absolute error, relative L2 and cosine similarity.
    """
    a, b = actual.detach().double(), expected.detach().to(actual.device).double()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    err = a - b
    return {
        "max_abs": err.abs().max().item(),
        "mean_abs": err.abs().mean().item(),
        "p95_abs": err.abs().flatten().quantile(0.95).item(),
        "relative_l2": (err.norm() / b.norm().clamp_min(1e-30)).item(),
        "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
    }


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    config = MistralLarge4Config.from_native(args.source)
    config.fp8_indexer = config.fp8_kv_cache = False
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    readers = [ReleasedReader(args.source, d, dtype, native=(i == 0)) for i, d in enumerate(devices)]
    frozen = json.loads(args.inputs.read_text())
    ids = torch.tensor(frozen["input_ids"][: args.tokens], dtype=torch.long)
    assert ids.numel() == args.tokens
    positions = [torch.arange(args.tokens, device=d) for d in devices]
    embeddings = [reader.weight("tok_embeddings.weight") for reader in readers]
    x = [F.embedding(ids.to(d), weight) for d, weight in zip(devices, embeddings)]
    del embeddings
    alpha = [reader.weight("lerope.alpha") for reader in readers]
    rotary = MistralLarge4Rotary(config).to(devices[0])
    rotary.alpha.copy_(alpha[0])
    tables = rotary(positions[0])
    layout = SequenceLayout(
        positions[0], torch.zeros_like(positions[0]), torch.ones_like(positions[0], dtype=torch.bool)
    )
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
    )
    caches: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [{}, {}]
    selections: list[torch.Tensor | None] = [None, None]
    report = {
        "scope": f"released original first {args.layers} layers; all {args.tokens} positions and full {config.vocab_size} vocabulary",
        "dtype": args.dtype,
        "layers": [],
        "input_sha256": hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        "checkpoint_metadata_sha256": hashlib.sha256((args.source / "params.json").read_bytes()).hexdigest(),
        "tolerances": {
            "hidden_relative_l2": 5e-5 if dtype == torch.float32 else 0.02,
            "logit_relative_l2": 1e-4 if dtype == torch.float32 else 0.02,
            "mean_kl": 1e-6 if dtype == torch.float32 else 0.001,
            "max_kl": 1e-4 if dtype == torch.float32 else 0.02,
        },
        "passed": False,
    }
    start = time.monotonic()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for layer in range(args.layers):
        tick = time.monotonic()
        source = config.kv_source_layers[layer]
        with torch.device("meta"):
            block = MistralLarge4Block(config, backend, layer_idx=layer, dtype=dtype)
        block.to_empty(device=devices[0])
        native_state = readers[0].layer(layer)
        mapped = MistralLarge4StateDictAdapter(
            dtype=dtype, grouped_experts=True, n_routed_experts=config.n_routed_experts
        ).from_hf(native_state)
        prefix = f"model.layers.{layer}."
        block.load_state_dict({k.removeprefix(prefix): v for k, v in mapped.items()}, strict=True)
        del mapped, native_state
        x[0], kv, selections[0] = block(x[0], tables, layout, caches[0].get(source), selections[0])
        if layer in config.kv_source_layers[layer + 1 : args.layers]:
            caches[0][layer] = kv
        del block, kv
        gc.collect()
        torch.cuda.empty_cache()
        state = readers[1].layer(layer)
        state["lerope.alpha"] = alpha[1]
        out = reference_block(
            state, x[1], positions[1], config, layer=layer, shared_kv=caches[1].get(source), indices=selections[1]
        )
        x[1], selections[1] = out.hidden, out.indices
        if layer in config.kv_source_layers[layer + 1 : args.layers]:
            caches[1][layer] = out.kv
        metrics = difference(x[0], x[1])
        # Index order is not semantic; compare selected key sets including padding.
        selected_equal = selections[0].sort(-1).values == selections[1].to(devices[0]).sort(-1).values
        metrics.update(
            layer=layer, seconds=time.monotonic() - tick, selector_set_agreement=selected_equal.float().mean().item()
        )
        report["layers"].append(metrics)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(metrics), flush=True)
        del state, out, selected_equal
        gc.collect()
        torch.cuda.empty_cache()
    norm = [reader.weight("norm.weight") for reader in readers]
    hidden = [rms(h, w, config.rms_norm_eps) for h, w in zip(x, norm)]
    heads = [reader.weight("output.weight") for reader in readers]
    kl, target, top1, squared, norm_squared, maximum = [], [], [], 0.0, 0.0, 0.0
    for start_token in range(0, args.tokens, 64):
        stop = min(start_token + 64, args.tokens)
        logits = [30 * torch.tanh(F.linear(h[start_token:stop], w).float() / 30) for h, w in zip(hidden, heads)]
        a, b = logits[0], logits[1].to(devices[0])
        error = a.double() - b.double()
        squared += error.square().sum().item()
        norm_squared += b.double().square().sum().item()
        maximum = max(maximum, error.abs().max().item())
        lp_a, lp_b = a.double().log_softmax(-1), b.double().log_softmax(-1)
        kl.append((lp_b.exp() * (lp_b - lp_a)).sum(-1).cpu())
        top1.append((a.argmax(-1) == b.argmax(-1)).cpu())
        labels = ids.roll(-1)[start_token:stop].to(devices[0])
        target.append((lp_a.gather(1, labels[:, None]) - lp_b.gather(1, labels[:, None])).abs().cpu())
    all_kl, all_targets = torch.cat(kl), torch.cat(target)
    report["logits"] = {
        "max_abs": maximum,
        "relative_l2": (squared / max(norm_squared, 1e-30)) ** 0.5,
        "kl_mean": all_kl.mean().item(),
        "kl_p95": all_kl.quantile(0.95).item(),
        "kl_max": all_kl.max().item(),
        "top1_agreement": torch.cat(top1).float().mean().item(),
        "target_logprob_error_mean": all_targets[:-1].mean().item(),
        "target_logprob_error_max": all_targets[:-1].max().item(),
        "positions": args.tokens,
        "vocab": config.vocab_size,
    }
    report["seconds"] = time.monotonic() - start
    report["source_hashes_equal"] = readers[0].hashes == readers[1].hashes
    report["source_tensors_read"] = len(readers[0].hashes)
    report["source_bytes_per_reader"] = [r.bytes for r in readers]
    report["peak_cuda_bytes"] = [torch.cuda.max_memory_allocated(d) for d in devices]
    tol = report["tolerances"]
    report["passed"] = (
        report["source_hashes_equal"]
        and all(m["relative_l2"] <= tol["hidden_relative_l2"] for m in report["layers"])
        and report["logits"]["relative_l2"] <= tol["logit_relative_l2"]
        and report["logits"]["kl_mean"] <= tol["mean_kl"]
        and report["logits"]["kl_max"] <= tol["max_kl"]
    )
    args.output.write_text(json.dumps(report, indent=2))
    args.output.with_suffix(".source-hashes.json").write_text(json.dumps(readers[0].hashes, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}), flush=True)
    assert report["passed"], "Released numerical comparison failed; see recorded metrics"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
