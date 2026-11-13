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

"""FP32 forward/backward parity of native Inkling 2 against Transformers (text only and image + text).

Requires a Transformers build with ``inkling2`` (huggingface/transformers#50682, commit
109a7d19bf667168837f32929e00b791e195d14d or later); the pinned release has none, so this is a
development script, not a unit test.

Random native weights are exported with the state-dict adapter to the released ``model.llm.*``
layout, saved as safetensors and loaded by ``Inkling2ForCausalLM.from_pretrained`` (text only) or
``Inkling2ForConditionalGeneration.from_pretrained`` (vision tower, two images scattered into a
padded batch), so the reference sees the weights through its own checkpoint conversion. Both
models run the same batch on CPU in FP32; the script compares logits on valid tokens and the
gradient of a random linear functional of those logits for every parameter.

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


# Tiny vision tower for the multimodal check; its projector writes the text width above.
TINY_VISION_CONFIG = dict(
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    patch_size=4,
    spatial_merge_size=2,
    posemb_grid_size=6,
    projection_hidden_dim=48,
    text_hidden_size=TINY_TEXT_CONFIG["hidden_size"],
)
IMAGE_TOKEN_ID = 249
# Two images of 3x4 and 5x2 language-model tokens, in the first and second row.
IMAGE_GRID_THW = [[1, 3, 4], [1, 5, 2]]


def _randomize(model: torch.nn.Module, seed: int) -> None:
    """Fill every parameter and buffer with seeded values away from degenerate initializations."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            noise = torch.randn(tensor.shape, generator=generator, dtype=torch.float32)
            if name.endswith(("norm.weight", "layernorm.weight", ".scale", "_before.weight", "_after.weight")):
                tensor.copy_(1.0 + 0.2 * noise)
            elif name.endswith("e_score_correction_bias"):
                tensor.copy_(0.05 * noise)
            elif ".lambda_" in name:
                tensor.copy_(0.1 + 0.05 * noise)
            else:
                tensor.copy_(0.15 * noise)


def multimodal_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return seeded ``(input_ids, attention_mask, pixel_values, image_grid_thw)`` for two images.

    ``input_ids`` has shape ``[2, 24]``: row 0 holds the 12 placeholders of the first image after
    four text tokens, row 1 is left-padded by 3 and holds the 10 placeholders of the second image.
    ``pixel_values`` has shape ``[22, merge * patch, merge * patch, channels]`` in ``[-1, 1]``.
    """
    generator = torch.Generator().manual_seed(4)
    input_ids = torch.randint(2, IMAGE_TOKEN_ID, (2, 24), generator=generator)
    input_ids[0, 4:16] = IMAGE_TOKEN_ID
    input_ids[1, 6:16] = IMAGE_TOKEN_ID
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :3] = 0
    side = TINY_VISION_CONFIG["patch_size"] * TINY_VISION_CONFIG["spatial_merge_size"]
    grid = torch.tensor(IMAGE_GRID_THW)
    pixel_values = torch.rand(int(grid.prod(-1).sum()), side, side, 3, generator=generator) * 2 - 1
    return input_ids, attention_mask, pixel_values, grid


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


def _native_name(name: str, text_prefix: str) -> str:
    """Map a Transformers parameter name to the native one; ``text_prefix`` is the reference's text prefix."""
    if name.startswith(text_prefix):
        name = "model.language_model." + name.removeprefix(text_prefix)
    return name.replace("experts.gate_up_proj", "experts.gate_and_up_projs").replace(
        "experts.down_proj", "experts.down_projs"
    )


def _group(name: str) -> str:
    for token, group in (
        ("vision_tower", "vision_tower"),
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


def _load_reference(
    reference_cls: type,
    config_json: dict,
    raw_state: dict[str, torch.Tensor],
    dtype: torch.dtype = torch.float32,
) -> torch.nn.Module:
    """Save ``raw_state`` as a released-layout checkpoint and load it through Transformers' conversion.

    Args:
        reference_cls: Transformers model class.
        config_json: ``config.json`` contents.
        raw_state: Released keys mapped to contiguous tensors of the released layouts.
        dtype: Model dtype; Transformers keeps its FP32 modules in FP32.

    Returns:
        The reference model in training mode, with no missing or unexpected keys.
    """
    with tempfile.TemporaryDirectory() as directory:
        # A transformers_version keeps Transformers from adding the MXFP8 quantization config.
        Path(directory, "config.json").write_text(json.dumps({**config_json, "transformers_version": "5.23.0.dev0"}))
        save_file(raw_state, str(Path(directory, "model.safetensors")))
        reference, info = reference_cls.from_pretrained(directory, dtype=dtype, output_loading_info=True)
    assert not any(info.values()), info
    # The text model supports eager attention only; the vision tower runs SDPA, like the native one.
    reference.config._attn_implementation = "eager"
    if reference_cls.__name__ == "Inkling2ForConditionalGeneration":
        reference.model.vision_tower.config._attn_implementation = "sdpa"
    return reference.train()


def _record_router_choices(model: torch.nn.Module, index_position: int) -> dict[str, torch.Tensor]:
    """Record each router's latest expert choice, sorted along the top-k axis.

    Args:
        model: Model whose routers are the modules named ``*.mlp.gate``.
        index_position: Position of the expert-index tensor ``[tokens, top_k]`` in the router output
            (1 for AutoModel's ``Gate``, 2 for the Transformers router).

    Returns:
        A mapping filled during later forwards: router name to a long tensor of shape ``[tokens, top_k]``.
    """
    choices: dict[str, torch.Tensor] = {}

    def record(name: str):
        def hook(module: torch.nn.Module, args: tuple, output: tuple) -> None:
            choices[name] = output[index_position].detach().sort(dim=-1).values

        return hook

    for name, module in model.named_modules():
        if name.endswith("mlp.gate"):
            module.register_forward_hook(record(name))
    return choices


def _router_mismatches(native: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], valid: torch.Tensor) -> int:
    """Count valid tokens whose top-k expert set differs, summed over layers.

    Args:
        native: Router name to sorted expert indices ``[tokens, top_k]`` of the native model.
        reference: The same for the reference, keyed by its own router names in layer order.
        valid: Boolean tensor of shape ``[batch, sequence]``; tokens are its flattened positions.

    Returns:
        Number of (layer, valid token) pairs with a different expert set.
    """
    assert len(native) == len(reference) > 0, (len(native), len(reference))
    flat_valid = valid.flatten()
    return sum(
        int((native_choice != reference_choice).any(-1)[flat_valid].sum())
        for native_choice, reference_choice in zip(native.values(), reference.values())
    )


def _compare(
    native: torch.nn.Module,
    reference: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    valid: torch.Tensor,
    text_prefix: str,
) -> dict[str, float | dict[str, list[float]]]:
    """Run both FP32 models on ``inputs`` and compare logits, router choices and parameter gradients.

    Args:
        native: Native model in training mode.
        reference: Transformers model holding the same weights.
        inputs: Forward keyword arguments; ``input_ids`` has shape ``[batch, sequence]``.
        valid: Boolean tensor of shape ``[batch, sequence]`` selecting the compared positions.
        text_prefix: Prefix of the reference's text-model parameters.

    Returns:
        Max abs logit difference with and without autograd, reference logit max, router mismatches
        and ``{group: [max abs grad difference, reference grad max]}``.
    """
    native_choices = _record_router_choices(native, 1)
    reference_choices = _record_router_choices(reference, 2)
    with torch.no_grad():
        no_grad_logits = native(**inputs).logits
    native_logits = native(**inputs).logits
    reference_logits = reference(**inputs).logits
    upstream = torch.randn(native_logits.shape, generator=torch.Generator().manual_seed(5))
    (native_logits * upstream)[valid].sum().backward()
    (reference_logits * upstream)[valid].sum().backward()

    native_params = dict(native.named_parameters())
    grads: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for name, parameter in reference.named_parameters():
        native_grad = native_params.pop(_native_name(name, text_prefix)).grad
        reference_grad = _hf_grad_to_native(name, parameter.grad)
        entry = grads[_group(name)]
        entry[0] = max(entry[0], (native_grad - reference_grad).abs().max().item())
        entry[1] = max(entry[1], reference_grad.abs().max().item())
    assert not native_params, f"native parameters without a reference: {sorted(native_params)}"
    return {
        "logit_diff": (native_logits - reference_logits)[valid].abs().max().item(),
        "no_grad_logit_diff": (no_grad_logits - reference_logits)[valid].abs().max().item(),
        "logit_scale": reference_logits[valid].abs().max().item(),
        "router_mismatches": _router_mismatches(native_choices, reference_choices, valid),
        "grads": dict(grads),
    }


def _compare_bf16(
    native: torch.nn.Module,
    reference: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    valid: torch.Tensor,
) -> dict[str, float]:
    """Compare BF16 forwards: logits on valid tokens and router choices.

    Args:
        native: Native BF16 model.
        reference: Transformers BF16 model holding the same weights.
        inputs: Forward keyword arguments; ``input_ids`` has shape ``[batch, sequence]``.
        valid: Boolean tensor of shape ``[batch, sequence]`` selecting the compared positions.

    Returns:
        Max abs logit difference, reference logit max, fraction of bitwise-equal logits and router mismatches.
    """
    native_choices = _record_router_choices(native, 1)
    reference_choices = _record_router_choices(reference, 2)
    with torch.no_grad():
        native_logits = native(**inputs).logits[valid]
        reference_logits = reference(**inputs).logits[valid]
    return {
        "logit_diff": (native_logits - reference_logits).abs().max().item(),
        "logit_scale": reference_logits.abs().max().item(),
        "exact_fraction": (native_logits == reference_logits).float().mean().item(),
        "router_mismatches": _router_mismatches(native_choices, reference_choices, valid),
    }


def _native_model(config: Inkling2Config, attn: str) -> Inkling2ForConditionalGeneration:
    backend = BackendConfig(attn=attn, linear="torch", rms_norm="torch", experts="torch", dispatcher="torch")
    native = Inkling2ForConditionalGeneration(config, backend=backend)
    _randomize(native, seed=1)
    return native.train()


def _raw_state(native: Inkling2ForConditionalGeneration) -> dict[str, torch.Tensor]:
    """Export native weights to the released layout and check the adapter round trip is exact."""
    adapter = native.state_dict_adapter
    raw_state = {key: value.detach().contiguous() for key, value in adapter.to_hf(native.state_dict()).items()}
    roundtrip = adapter.from_hf(raw_state)
    native_state = native.state_dict()
    assert roundtrip.keys() == native_state.keys(), set(roundtrip) ^ set(native_state)
    for key, value in native_state.items():
        assert torch.equal(roundtrip[key], value), key
    return raw_state


def main() -> None:
    try:
        from transformers.models.inkling2.modeling_inkling2 import Inkling2ForCausalLM
        from transformers.models.inkling2.modeling_inkling2 import (
            Inkling2ForConditionalGeneration as HFInkling2ForConditionalGeneration,
        )
    except ImportError as exc:
        raise RuntimeError("This parity script needs a Transformers build that includes inkling2") from exc

    torch.manual_seed(0)
    input_ids = torch.randint(2, TINY_TEXT_CONFIG["unpadded_vocab_size"], (2, 24))
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :5] = 0  # left padding in the second row
    mm_input_ids, mm_attention_mask, pixel_values, image_grid_thw = multimodal_inputs()
    scenarios = {
        # Text only: Inkling2ForCausalLM.
        "text": (
            {"vision_config": None},
            Inkling2ForCausalLM,
            {"model_type": "inkling2_text", "architectures": ["Inkling2ForCausalLM"], **TINY_TEXT_CONFIG},
            {"input_ids": input_ids, "attention_mask": attention_mask},
            "model.",
        ),
        # Image + text: Inkling2ForConditionalGeneration with the vision tower, two images.
        "image+text": (
            {"vision_config": TINY_VISION_CONFIG, "image_token_id": IMAGE_TOKEN_ID},
            HFInkling2ForConditionalGeneration,
            {
                "model_type": "inkling2",
                "architectures": ["Inkling2ForConditionalGeneration"],
                "image_token_id": IMAGE_TOKEN_ID,
                "text_config": TINY_TEXT_CONFIG,
                "vision_config": TINY_VISION_CONFIG,
            },
            {
                "input_ids": mm_input_ids,
                "attention_mask": mm_attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            },
            "model.language_model.",
        ),
    }

    failures = []
    for scenario, (config_kwargs, reference_cls, config_json, inputs, text_prefix) in scenarios.items():
        valid = inputs["attention_mask"].bool()
        for attn in ("eager", "sdpa"):
            run = f"{scenario}/{attn}"
            config = Inkling2Config(text_config=TINY_TEXT_CONFIG, torch_dtype="float32", **config_kwargs)
            native = _native_model(config, attn)
            reference = _load_reference(reference_cls, config_json, _raw_state(native))
            result = _compare(native, reference, inputs, valid, text_prefix)
            print(
                f"[{run} fp32] logits max_abs={result['logit_diff']:.3e} "
                f"no_grad max_abs={result['no_grad_logit_diff']:.3e} (ref max {result['logit_scale']:.3e}) "
                f"router top-k mismatches={result['router_mismatches']}"
            )
            for group, (diff, scale) in sorted(result["grads"].items()):
                print(f"[{run} fp32]   grad {group:22s} max_abs={diff:.3e} ref_max={scale:.3e} rel={diff / scale:.3e}")
            if max(result["logit_diff"], result["no_grad_logit_diff"]) > 1e-5 or result["router_mismatches"]:
                failures.append(f"{run} fp32 logits/router")
            failures += [
                f"{run} fp32 grad {group}"
                for group, (diff, scale) in result["grads"].items()
                if diff > 1e-4 * max(scale, 1e-6)
            ]

            # BF16: the same weights rounded to BF16 outside the FP32 modules both models keep.
            config = Inkling2Config(text_config=TINY_TEXT_CONFIG, torch_dtype="bfloat16", **config_kwargs)
            native = _native_model(config, attn)
            reference = _load_reference(reference_cls, config_json, _raw_state(native), dtype=torch.bfloat16)
            result = _compare_bf16(native, reference, inputs, valid)
            print(
                f"[{run} bf16] logits max_abs={result['logit_diff']:.3e} (ref max {result['logit_scale']:.3e}, "
                f"rel {result['logit_diff'] / result['logit_scale']:.3e}) bitwise-equal={result['exact_fraction']:.4f} "
                f"router top-k mismatches={result['router_mismatches']}"
            )
            # Eager uses the reference's ops in the same order; SDPA may pick a fused CPU kernel.
            if attn == "eager" and result["exact_fraction"] < 0.99:
                failures.append(f"{run} bf16 logits")
    assert not failures, failures
    print("PARITY PASSED")


if __name__ == "__main__":
    main()
