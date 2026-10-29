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

"""All-gradient BF16 check for an original released MoE block on two GPUs."""

import argparse
import gc
import json
from pathlib import Path

import torch

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.mistral_large4.attention import SequenceLayout
from nemo_automodel.components.models.mistral_large4.config import MistralLarge4Config
from nemo_automodel.components.models.mistral_large4.model import MistralLarge4Block
from nemo_automodel.components.models.mistral_large4.rope_utils import MistralLarge4Rotary
from nemo_automodel.components.models.mistral_large4.state_dict_adapter import MistralLarge4StateDictAdapter
from tests.functional_tests.models.mistral_large4.validate_released import ReleasedReader, difference
from tests.unit_tests.models.mistral_large4.released_reference import reference_block
from tests.unit_tests.models.mistral_large4.test_model import tensor_error


def run(source: Path, inputs: Path, output: Path) -> None:
    """Compare original layer 1, input and LeRoPE gradients, and FP32 SGD deltas."""
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    config = MistralLarge4Config.from_native(source)
    config.fp8_indexer = config.fp8_kv_cache = False
    dtype = torch.bfloat16
    devices = [torch.device("cuda:0"), torch.device("cuda:1")]
    readers = [ReleasedReader(source, d, dtype, native=(i == 0)) for i, d in enumerate(devices)]
    ids = torch.tensor(json.loads(inputs.read_text())["input_ids"][20:25])
    positions = [torch.arange(4093, 4098, device=d) for d in devices]
    x = [r.weight("tok_embeddings.weight")[ids.to(d)].detach().requires_grad_() for r, d in zip(readers, devices)]
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch_fp32",
        rope="torch",
        rope_fusion=False,
        experts="torch",
        dispatcher="torch",
    )
    with torch.device("meta"):
        block = MistralLarge4Block(config, backend, layer_idx=1, dtype=dtype)
    block.to_empty(device=devices[0])
    native_state = readers[0].layer(1)
    adapter = MistralLarge4StateDictAdapter(dtype=dtype, grouped_experts=True, n_routed_experts=config.n_routed_experts)
    block.load_state_dict({k.removeprefix("model.layers.1."): v for k, v in adapter.from_hf(native_state).items()})
    del native_state
    native = adapter.to_hf({"model.layers.1." + k: v for k, v in block.named_parameters()})
    rotary = MistralLarge4Rotary(config).to(devices[0])
    with torch.no_grad():
        rotary.alpha.copy_(readers[0].weight("lerope.alpha"))
    native["lerope.alpha"] = rotary.alpha
    state = readers[1].layer(1)
    state["lerope.alpha"] = readers[1].weight("lerope.alpha")
    for name, value in state.items():
        value.requires_grad_(native[name].requires_grad)
        assert torch.equal(native[name].detach(), value.to(devices[0])), name
    layout = SequenceLayout(
        positions[0], torch.zeros_like(positions[0]), torch.ones_like(positions[0], dtype=torch.bool)
    )
    actual = block(x[0], rotary(positions[0]), layout, None, None)[0]
    expected = reference_block(state, x[1], positions[1], config, layer=1).hidden
    upstream = torch.randn(actual.shape, generator=torch.Generator().manual_seed(1201)).to(devices[0])
    (actual.float() * upstream).sum().backward()
    (expected.float() * upstream.to(devices[1])).sum().backward()
    gradients = adapter.to_hf(
        {"model.layers.1." + k: v.grad for k, v in block.named_parameters() if v.grad is not None}
    )
    gradients["lerope.alpha"] = rotary.alpha.grad
    gradients["input"] = x[0].grad
    report = {
        "scope": "released original layer 1, all 128 experts, BF16 storage/compute, protected FP32 router/LeRoPE; five tokens at positions 4093..4097",
        "hidden": difference(actual, expected),
        "gradients": [],
        "updates": [],
        "strict_gradient_failures": 0,
        "tolerances": {
            "per_tensor_rms_absolute": 1.6e-5,
            "per_tensor_rms_relative": 1 / 64,
            "global_relative_l2": 0.005,
        },
        "passed": False,
    }
    native["input"] = x[0]
    state["input"] = x[1]
    total_error, total_norm, update_error, update_norm = 0.0, 0.0, 0.0, 0.0
    for name, value in native.items():
        if not value.requires_grad:
            assert name not in gradients and state[name].grad is None
            continue
        assert name in gradients and state[name].grad is not None, name
        reference_gradient = state[name].grad.to(devices[0])
        error = tensor_error(name, gradients[name], reference_gradient)
        total_error += error.squared_error
        total_norm += error.reference_squared_norm
        report["gradients"].append(
            {
                "name": name,
                "rms_error": error.rms_error,
                "reference_rms": error.reference_rms,
                "relative_l2": error.relative_l2,
                "max_abs": error.max_absolute,
                "passed": error.rms_error <= 1.6e-5 + error.reference_rms / 64,
            }
        )
        report["strict_gradient_failures"] += int(
            not torch.allclose(gradients[name].float(), reference_gradient.float(), atol=1.6e-5, rtol=4e-5)
        )
        if name != "input":
            before = value.detach().float()
            left = torch.nn.Parameter(before.clone())
            right = torch.nn.Parameter(before.clone())
            left.grad = gradients[name].float().clone()
            right.grad = reference_gradient.float().clone()
            torch.optim.SGD([left], lr=0.01).step()
            torch.optim.SGD([right], lr=0.01).step()
            delta = tensor_error(name, left - before, right - before)
            update_error += delta.squared_error
            update_norm += delta.reference_squared_norm
            report["updates"].append(
                {
                    "name": name,
                    "relative_l2": delta.relative_l2,
                    "rms_error": delta.rms_error,
                    "passed": delta.rms_error <= 1.6e-7 + delta.reference_rms / 64,
                }
            )
            del before, left, right
        del reference_gradient
    report["global_gradient_relative_l2"] = (total_error / max(total_norm, 1e-30)) ** 0.5
    report["global_update_relative_l2"] = (update_error / max(update_norm, 1e-30)) ** 0.5
    report["source_hashes_equal"] = readers[0].hashes == readers[1].hashes
    report["peak_cuda_bytes"] = [torch.cuda.max_memory_allocated(d) for d in devices]
    report["passed"] = (
        report["source_hashes_equal"]
        and all(g["passed"] for g in report["gradients"])
        and all(g["passed"] for g in report["updates"])
        and report["global_gradient_relative_l2"] <= 0.005
        and report["global_update_relative_l2"] <= 0.005
    )
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("gradients", "updates")}), flush=True)
    assert report["passed"], "Released BF16 gradients failed the pre-existing frozen budget"
    del block, state, native
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.inputs, args.output)
