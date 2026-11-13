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

"""Tiny-config FP32 parity: native Inkling 2 vision tower vs Transformers ``Inkling2VisionModel``.

Loads the same random weights into both towers, encodes two images with different grid sizes in
one batch on CPU in float32, and compares ``pooler_output``, ``last_hidden_state`` and the gradient
of every parameter for a scalar loss with a random upstream projection. Every reported relative
error (max absolute difference over the reference's max absolute value) must stay below 1e-5.

Requires a Transformers build with ``inkling2`` (huggingface/transformers@109a7d19bf or later); the
pinned Transformers 5.18.0 does not have it, so this is a development script, not a unit test.

Run directly::

    python tests/unit_tests/models/inkling2/parity_check_inkling2_vision.py [--attn eager|sdpa]
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

from nemo_automodel.components.models.inkling2.vision import Inkling2VisionModel

# Two images, (time, rows, columns) in language-model tokens; with merge 2 and a 6x6 position grid,
# their 6x8 and 10x4 patch grids exercise both up- and down-sampling of the position embedding.
IMAGE_GRID_THW = [[1, 3, 4], [1, 5, 2]]

# Tiny vision config, in ``Inkling2VisionConfig`` argument names (shared by the Transformers and
# AutoModel classes). The unit tests build the AutoModel config from the same values.
TINY_VISION_CONFIG = {
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "patch_size": 4,
    "spatial_merge_size": 2,
    "posemb_grid_size": 6,
    "projection_hidden_dim": 96,
    "projector_bias": False,
    "text_hidden_size": 48,
    "layer_norm_eps": 1e-5,
    "mlp_ratio": 4.0,
}

# Gradient norms recorded by ``--golden`` for these parameters.
GOLDEN_GRAD_PARAMETERS = (
    "embeddings.patch_embedding.weight",
    "embeddings.position_embedding.weight",
    "embeddings.class_embedding",
    "layers.0.self_attn.q_proj.weight",
    "layers.1.lambda_2",
    "merger.linear_1.weight",
)


def build_tiny_vision_config():
    """Return the tiny Transformers ``Inkling2VisionConfig``."""
    from transformers.models.inkling2.configuration_inkling2 import Inkling2VisionConfig

    return Inkling2VisionConfig(**TINY_VISION_CONFIG)


def fill_parameters_(module: torch.nn.Module, seed: int = 1) -> None:
    """Fill every parameter, in sorted name order, with seeded noise.

    No gradient path is trivially zero or one afterwards: layer norms are near one, layer scales
    near 0.1 and every other tensor is ``0.1 * N(0, 1)``.
    """
    generator = torch.Generator().manual_seed(seed)
    parameters = dict(module.named_parameters())
    with torch.no_grad():
        for name in sorted(parameters):
            parameter = parameters[name]
            noise = torch.randn(parameter.shape, generator=generator)
            if name.endswith(("layernorm_before.weight", "layernorm_after.weight", "pre_layernorm.weight")):
                parameter.copy_(1.0 + 0.1 * noise)
            elif "lambda_" in name:
                parameter.copy_(0.1 + 0.05 * noise)
            else:
                parameter.copy_(0.1 * noise)


def make_inputs(
    patch_size: int, spatial_merge_size: int, num_channels: int, text_hidden_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return seeded ``(pixel_values, image_grid_thw, upstream)`` for ``IMAGE_GRID_THW``.

    ``pixel_values`` has shape ``[lm_tokens, merge * patch, merge * patch, channels]`` with values
    in ``[-1, 1]``, ``image_grid_thw`` shape ``[images, 3]`` and ``upstream`` shape
    ``[lm_tokens, text_hidden]``, the random projection that turns ``pooler_output`` into a scalar.
    """
    grid = torch.tensor(IMAGE_GRID_THW)
    side = patch_size * spatial_merge_size
    generator = torch.Generator().manual_seed(2)
    tokens = int(grid.prod(-1).sum())
    pixel_values = torch.rand(tokens, side, side, num_channels, generator=generator) * 2 - 1
    upstream = torch.randn(tokens, text_hidden_size, generator=generator)
    return pixel_values, grid, upstream


def _group(name: str) -> str:
    """Map a parameter name to its reporting group."""
    if name.startswith("layers."):
        _, _, rest = name.split(".", 2)
        return "layers.*." + rest.split(".")[0] + (".*" if "." in rest else "")
    return name.rsplit(".", 1)[0] + ".*" if "." in name else name


def _errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    """Return (max abs error, max abs error / max abs reference)."""
    abs_err = (actual - expected).abs().max().item()
    scale = expected.abs().max().item()
    return abs_err, abs_err / scale if scale > 0 else abs_err


def run(attn_implementation: str) -> dict[str, tuple[float, float]]:
    """Run the comparison and return ``{tensor group: (max abs error, max relative error)}``."""
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel as HFInkling2VisionModel

    torch.manual_seed(0)
    config = build_tiny_vision_config()
    config._attn_implementation = attn_implementation
    reference = HFInkling2VisionModel(config).float()
    fill_parameters_(reference)
    native = Inkling2VisionModel(config).float()
    native.load_state_dict(reference.state_dict(), strict=True)
    reference.train()
    native.train()  # attention_dropout is 0.0; train mode keeps the autograd path of real training

    pixel_values, grid, upstream = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )
    ref_out = reference(pixel_values=pixel_values, image_grid_thw=grid)
    nat_out = native(pixel_values, grid)
    (ref_out.pooler_output * upstream).sum().backward()
    (nat_out.pooler_output * upstream).sum().backward()

    results: dict[str, tuple[float, float]] = {
        "pooler_output": _errors(nat_out.pooler_output, ref_out.pooler_output),
        "last_hidden_state": _errors(nat_out.last_hidden_state, ref_out.last_hidden_state),
    }
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    ref_params = dict(reference.named_parameters())
    for name, parameter in native.named_parameters():
        ref_grad = ref_params[name].grad
        if parameter.grad is None or ref_grad is None:
            raise AssertionError(f"missing gradient for {name}")
        grouped["grad " + _group(name)].append(_errors(parameter.grad, ref_grad))
    for group, errors in grouped.items():
        results[group] = (max(e[0] for e in errors), max(e[1] for e in errors))
    return results


def golden() -> dict[str, list[float]]:
    """Return the reference statistics that the unit tests pin, computed with Transformers."""
    from transformers.models.inkling2.modeling_inkling2 import Inkling2VisionModel as HFInkling2VisionModel

    config = build_tiny_vision_config()
    config._attn_implementation = "sdpa"
    reference = HFInkling2VisionModel(config).float()
    fill_parameters_(reference)
    pixel_values, grid, upstream = make_inputs(
        config.patch_size, config.spatial_merge_size, config.num_channels, config.text_hidden_size
    )
    output = reference(pixel_values=pixel_values, image_grid_thw=grid)
    (output.pooler_output * upstream).sum().backward()
    parameters = dict(reference.named_parameters())
    return {
        "pooler_row_sums": output.pooler_output.sum(-1).tolist(),
        "pooler_first_row": output.pooler_output[0, :8].tolist(),
        "grad_norms": [parameters[name].grad.norm().item() for name in GOLDEN_GRAD_PARAMETERS],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--attn", choices=["eager", "sdpa"], default="eager", help="reference attention backend")
    parser.add_argument("--golden", action="store_true", help="print the reference values pinned by the unit tests")
    args = parser.parse_args()
    if args.golden:
        for key, values in golden().items():
            print(f"{key} = {values!r}")
        return
    results = run(args.attn)
    print(f"| tensor ({args.attn} reference) | max abs err | max rel err |")
    print("|---|---|---|")
    for name, (abs_err, rel_err) in results.items():
        print(f"| {name} | {abs_err:.3e} | {rel_err:.3e} |")
    worst = max(rel for _, rel in results.values())
    if worst >= 1e-5:
        raise SystemExit(f"FAILED: worst relative error {worst:.3e} >= 1e-5")
    print(f"PASSED: worst relative error {worst:.3e}")


if __name__ == "__main__":
    main()
