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

"""FP32 forward/backward parity of the native Inkling 2 text model against Transformers.

Requires a Transformers build with ``inkling2`` (huggingface/transformers#50682, commit
109a7d19bf667168837f32929e00b791e195d14d or later); the pinned release has none, so this is a
development script, not a unit test.

Random native weights are exported with the state-dict adapter to the released ``model.llm.*``
layout, saved as safetensors and loaded by ``Inkling2ForCausalLM.from_pretrained``, so the
reference sees the weights through its own checkpoint conversion. Both models run the same
padded batch on CPU in FP32; the script compares logits on valid tokens and the gradient of a
random linear functional of those logits for every parameter.

Run directly:  python tests/unit_tests/models/inkling2/parity_check_inkling2.py
"""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import torch
from safetensors.torch import save_file

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.inkling2.configuration import Inkling2Config
from nemo_automodel.components.models.inkling2.model import Inkling2ForConditionalGeneration

# Seven layers: block boundaries at 0 and 6; full attention at 2 (inside a block) and 6 (at a boundary).
TINY_TEXT_CONFIG = dict(
    hidden_size=64,
    num_hidden_layers=7,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    swa_num_attention_heads=4,
    swa_num_key_value_heads=2,
    swa_head_dim=16,
    sliding_window_size=8,
    d_rel=4,
    rel_extent=12,
    vocab_size=256,
    unpadded_vocab_size=250,
    moe_intermediate_size=32,
    n_routed_experts=8,
    num_experts_per_tok=2,
    n_layers_per_attn_res_block=6,
    layer_types=[
        "hybrid_sliding",
        "hybrid_sliding",
        "hybrid",
        "hybrid_sliding",
        "hybrid_sliding",
        "hybrid_sliding",
        "hybrid",
    ],
    # Small thresholds and caps so the clips, log scaling and softcaps are active on tiny weights.
    log_scaling_n_floor=4,
    qk_l2_clip_threshold=1.0,
    rel_l2_clip_threshold=0.5,
    mlp_gate_softcap=0.5,
    mlp_up_softcap=0.5,
    final_logit_softcapping=0.5,
    eos_token_id=1,
)


def _randomize(model: torch.nn.Module, seed: int) -> None:
    """Fill every parameter and buffer with seeded values away from degenerate initializations."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
            if name.endswith(("norm.weight", "layernorm.weight", ".scale")):
                tensor.copy_(1.0 + 0.2 * noise)
            elif name.endswith("e_score_correction_bias"):
                tensor.copy_(0.05 * noise)
            else:
                tensor.copy_(0.15 * noise)


def _hf_to_native_name(name: str) -> str:
    return name if name == "lm_head.weight" else "model.language_model." + name.removeprefix("model.")


def _hf_grad_to_native(name: str, grad: torch.Tensor) -> torch.Tensor:
    """Bring a Transformers gradient into the native layout.

    Args:
        name: Transformers parameter name.
        grad: Gradient in the Transformers layout; ``gate_up_proj`` is ``[experts, 2 * intermediate, hidden]``
            with ``[gate | up]`` rows and ``down_proj`` is ``[experts, hidden, intermediate]``.

    Returns:
        The gradient in the native layout: ``[experts, hidden, 2 * intermediate]`` with interleaved gate/up
        columns, ``[experts, intermediate, hidden]``, or ``grad`` unchanged for other parameters.
    """
    if name.endswith("mlp.experts.gate_up_proj"):
        # [E, 2I, H] with [gate | up] rows -> interleaved rows -> [E, H, 2I].
        experts, rows, hidden = grad.shape
        interleaved = grad.reshape(experts, 2, rows // 2, hidden).transpose(1, 2).reshape(experts, rows, hidden)
        return interleaved.transpose(-1, -2)
    if name.endswith("mlp.experts.down_proj"):
        return grad.transpose(-1, -2)
    return grad


def _group(name: str) -> str:
    for token, group in (
        ("experts", "experts"),
        ("mlp.gate", "router"),
        ("sconv", "kv_convs"),
        ("convolution", "post_convs"),
        ("_input.query", "attn_residual_queries"),
        ("final_input", "attn_residual_queries"),
        ("self_attn", "attention"),
        ("embed_tokens", "embedding"),
        ("lm_head", "lm_head"),
    ):
        if token in name:
            return group
    return "norms"


def main() -> None:
    try:
        from transformers.models.inkling2.modeling_inkling2 import Inkling2ForCausalLM
    except ImportError as exc:
        raise RuntimeError("This parity script needs a Transformers build that includes inkling2") from exc

    torch.manual_seed(0)
    config = Inkling2Config(text_config=TINY_TEXT_CONFIG, vision_config=None, torch_dtype="float32")
    text_config = config.text_config
    input_ids = torch.randint(2, text_config.unpadded_vocab_size, (2, 24))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :5] = 0  # left padding in the second row
    valid = attention_mask.bool()
    upstream = torch.randn(2, 24, text_config.unpadded_vocab_size)

    results = {}
    for attn in ("eager", "sdpa"):
        backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
        native = Inkling2ForConditionalGeneration(config, backend=backend)
        _randomize(native, seed=1)
        adapter = native.state_dict_adapter
        raw_state = {key: value.detach().contiguous() for key, value in adapter.to_hf(native.state_dict()).items()}

        # Level 1: adapter round trip is exact.
        roundtrip = adapter.from_hf(raw_state)
        native_state = native.state_dict()
        assert roundtrip.keys() == native_state.keys(), set(roundtrip) ^ set(native_state)
        for key, value in native_state.items():
            assert torch.equal(roundtrip[key], value), key

        with tempfile.TemporaryDirectory() as directory:
            hf_config = {"model_type": "inkling2_text", "architectures": ["Inkling2ForCausalLM"], **TINY_TEXT_CONFIG}
            hf_config["transformers_version"] = "5.23.0.dev0"
            Path(directory, "config.json").write_text(json.dumps(hf_config))
            save_file(raw_state, str(Path(directory, "model.safetensors")))
            reference = Inkling2ForCausalLM.from_pretrained(directory, dtype=torch.float32)
        reference.config._attn_implementation = "eager"
        reference.train()
        native.train()

        # Without autograd, CPU SDPA may pick its fused kernel instead of the math fallback.
        with torch.no_grad():
            no_grad_logits = native(input_ids=input_ids, attention_mask=attention_mask).logits
        native_logits = native(input_ids=input_ids, attention_mask=attention_mask).logits
        reference_logits = reference(input_ids=input_ids, attention_mask=attention_mask).logits
        no_grad_diff = (no_grad_logits - reference_logits)[valid].abs().max().item()
        (native_logits * upstream)[valid].sum().backward()
        (reference_logits * upstream)[valid].sum().backward()

        logit_diff = (native_logits - reference_logits)[valid].abs().max().item()
        logit_scale = reference_logits[valid].abs().max().item()
        native_params = dict(native.named_parameters())
        grads = defaultdict(lambda: [0.0, 0.0])
        for name, parameter in reference.named_parameters():
            native_name = _hf_to_native_name(name).replace("experts.gate_up_proj", "experts.gate_and_up_projs")
            native_name = native_name.replace("experts.down_proj", "experts.down_projs")
            native_grad = native_params.pop(native_name).grad
            reference_grad = _hf_grad_to_native(name, parameter.grad)
            entry = grads[_group(native_name)]
            entry[0] = max(entry[0], (native_grad - reference_grad).abs().max().item())
            entry[1] = max(entry[1], reference_grad.abs().max().item())
        assert not native_params, f"native parameters without a reference: {sorted(native_params)}"
        results[attn] = (logit_diff, no_grad_diff, logit_scale, dict(grads))

    for attn, (logit_diff, no_grad_diff, logit_scale, grads) in results.items():
        print(
            f"[{attn}] logits max_abs={logit_diff:.3e} no_grad max_abs={no_grad_diff:.3e} "
            f"(ref max {logit_scale:.3e}, rel {logit_diff / logit_scale:.3e})"
        )
        for group, (diff, scale) in sorted(grads.items()):
            print(
                f"[{attn}]   grad {group:22s} max_abs={diff:.3e} ref_max={scale:.3e} rel={diff / max(scale, 1e-30):.3e}"
            )
        assert max(logit_diff, no_grad_diff) <= 1e-5, f"{attn} logits differ: {logit_diff}, {no_grad_diff}"
        for group, (diff, scale) in grads.items():
            assert diff <= 1e-4 * max(scale, 1e-6), f"{attn} {group} gradients differ: {diff} vs {scale}"
    print("PARITY PASSED")


if __name__ == "__main__":
    main()
