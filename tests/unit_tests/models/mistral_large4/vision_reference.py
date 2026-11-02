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

"""Independent native-vision oracle: explicit attention, complex RoPE, unfold merge.

Derived from the pinned vendor equations; does not import production vision math.
HF Pixtral is a separate FP32 check after a test-only reversible Q/K permutation.
"""

import torch
from torch.nn import functional as F

from tests.unit_tests.models.mistral_large4.math_reference import rms


def reference_images(state: dict[str, torch.Tensor], pixels: list[torch.Tensor], c) -> tuple[torch.Tensor, ...]:
    """Compute each image independently using released parameter names.

    Args:
        state: Differentiable released tensors, matrices [output, input], norms
            [channels], convolution [output, input, patch_height, patch_width].
        pixels: Per-image floating [channels, height, width] tensors.
        c: Native vision configuration.

    Returns:
        Per-image [merged_patches, text_hidden] projected features.
    """
    outputs = []
    dim = c.hidden_size // c.num_attention_heads
    for pixels_one in pixels:
        patch = F.conv2d(pixels_one[None], state["vision_encoder.patch_conv.weight"], stride=c.patch_size)
        h, w = patch.shape[-2:]
        x = rms(patch[0].flatten(1).T, state["vision_encoder.ln_pre.weight"], 1e-5)
        frequencies = c.rope_theta ** (-torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim)
        phase = torch.stack(
            [torch.cat((r * frequencies[::2], col * frequencies[1::2])) for r in range(h) for col in range(w)]
        )
        rotations = torch.polar(torch.ones_like(phase), phase)[:, None]
        for layer in range(c.num_hidden_layers):
            prefix = f"vision_encoder.transformer.layers.{layer}."
            y = rms(x, state[prefix + "attention_norm.weight"], 1e-5)
            q, k, v = [
                F.linear(y, state[prefix + "attention.w" + name + ".weight"]).view(-1, c.num_attention_heads, dim)
                for name in ("q", "k", "v")
            ]
            q = (
                torch.view_as_real(
                    torch.view_as_complex(q.float().reshape(h * w, c.num_attention_heads, -1, 2)) * rotations
                )
                .flatten(-2)
                .to(x.dtype)
            )
            k = (
                torch.view_as_real(
                    torch.view_as_complex(k.float().reshape(h * w, c.num_attention_heads, -1, 2)) * rotations
                )
                .flatten(-2)
                .to(x.dtype)
            )
            probability = (torch.einsum("thd,shd->hts", q.float(), k.float()) / dim**0.5).softmax(-1).to(x.dtype)
            attended = torch.einsum("hts,shd->thd", probability, v).reshape(h * w, -1)
            x = x + F.linear(attended, state[prefix + "attention.wo.weight"])
            y = rms(x, state[prefix + "ffn_norm.weight"], 1e-5)
            a = F.linear(y, state[prefix + "feed_forward.w1.weight"])
            b = F.linear(y, state[prefix + "feed_forward.w3.weight"])
            x = x + F.linear(F.silu(a) * b, state[prefix + "feed_forward.w2.weight"])
        x = rms(x, state["pre_mm_projector_norm.weight"], 1e-5)
        merged = F.unfold(
            x.T.reshape(1, c.hidden_size, h, w), kernel_size=c.spatial_merge_size, stride=c.spatial_merge_size
        )[0].T
        merged = F.linear(merged, state["patch_merger.merging_layer.weight"])
        projected = F.linear(
            F.gelu(F.linear(merged, state["vision_language_adapter.w_in.weight"])),
            state["vision_language_adapter.w_out.weight"],
        )
        outputs.append(projected)
    return tuple(outputs)
