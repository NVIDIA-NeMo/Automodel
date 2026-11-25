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

"""Generate the reference dumps that ``test_reference_parity.py`` reads.

The reference is the Hugging Face Transformers implementation of Hy4-flash, pull request 50903 at commit ``eb1d5db4``
(``transformers`` 5.24.0.dev0). It must be first on ``PYTHONPATH``; AutoModel is not imported. Each dump holds a tiny
random model's weights under the released key names, its inputs and the reference outputs::

    PYTHONPATH=<reference transformers> python make_reference_dumps.py text ref_tiny_97.pt --length 97
    PYTHONPATH=<reference transformers> python make_reference_dumps.py image-text ref_vlm.pt
    PYTHONPATH=<reference transformers> python make_reference_dumps.py vision ref_vision.pt

The tests find them through ``HY4_REF_DUMP``, ``HY4_REF_VLM_DUMP`` and ``HY4_REF_VISION_DUMP``.
"""

import argparse
import re
from pathlib import Path

import torch
import transformers
from tiny_kwargs import TINY_TEXT_KWARGS
from torch import nn

IMAGE_TOKEN_ID = 250
VISION_KWARGS = dict(
    hidden_size=32,
    intermediate_size=48,
    num_hidden_layers=2,
    num_attention_heads=4,
    patch_size=4,
    spatial_merge_size=2,
    out_hidden_size=TINY_TEXT_KWARGS["hidden_size"],
    max_image_size=32,
    rms_norm_eps=1e-5,
)
# (reference pattern, released replacement) of the text backbone and of the vision tower.
TEXT_RENAMES = (
    (r"\.self_attn\.sinks$", ".self_attn.learnable_sink_param"),
    (r"\.self_attn\.gate_proj\.weight$", ".self_attn.linear_gate.weight"),
    (r"\.post_self_attn_layernorm\.weight$", ".self_attn.post_proj_norm.weight"),
    (r"\.mlp\.pre_expert_proj\.weight$", ".mlp.routed_expert_down_proj.weight"),
    (r"\.mlp\.post_expert_proj\.weight$", ".mlp.routed_expert_up_proj.weight"),
    (r"\.attn_hc\.(fn|base|scale)$", r".hc_attn_layer.hc_pre.hc_\1"),
    (r"\.ffn_hc\.(fn|base|scale)$", r".hc_mlp_layer.hc_pre.hc_\1"),
    (r"^hc_head\.hc_(fn|base|scale)$", r"hc_head.hc_head_\1"),
)
VISION_RENAMES = (
    (r"layer_norm1", "input_layernorm"),
    (r"layer_norm2", "post_attention_layernorm"),
    (r"mlp\.fc1", "mlp.dense_h_to_4h"),
    (r"mlp\.fc2", "mlp.dense_4h_to_h"),
    (r"patch_merger\.before_rms", "perceive.norm"),
    (r"patch_merger\.proj_conv", "perceive.proj.0"),
    (r"patch_merger\.proj_out", "perceive.proj.2"),
    (r"patch_merger\.image_newline", "perceive.image_newline"),
)


def _rename(key: str, renames: tuple[tuple[str, str], ...]) -> str:
    for pattern, replacement in renames:
        key = re.sub(pattern, replacement, key)
    return key


def _released_text_key(key: str, mlp_layer_types: list[str]) -> str | None:
    """Released name of a reference text-backbone key, or ``None`` for keys the release does not carry."""
    if "rotary_emb" in key:
        return None
    norm = re.search(r"layers\.(\d+)\.post_mlp_layernorm\.weight$", key)
    if norm:
        if mlp_layer_types[int(norm.group(1))] != "dense":
            return None  # MoE layers keep this norm at ``mlp.combine_output_norm.weight`` already
        key = re.sub(r"\.post_mlp_layernorm\.weight$", ".mlp.post_fc2_norm.weight", key)
    return "model." + _rename(key, TEXT_RENAMES)


def _perturb(model: nn.Module, scale: float) -> None:
    """Move every floating parameter off its initial value so no weight is exactly one or zero."""
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.dtype.is_floating_point:
                parameter.add_(scale * torch.randn_like(parameter))
        for name, buffer in model.named_buffers():
            if name.endswith("e_score_correction_bias"):
                buffer.copy_(0.1 * torch.randn_like(buffer))


def dump_text(path: Path, length: int, seed: int) -> None:
    """Text backbone: logits, final hidden states, every layer's output and the sub-module outputs."""
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashTextConfig
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashTextModel

    torch.manual_seed(seed)
    config = HYV4FlashTextConfig(**TINY_TEXT_KWARGS)
    config._attn_implementation = "eager"
    model = HYV4FlashTextModel(config).float().eval()
    _perturb(model, 0.05)
    lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False).float()
    nn.init.normal_(lm_head.weight, std=0.1)
    ids = torch.randint(3, 280, (1, length))
    ids[0, 5:8] = ids[0, 2:5]  # repeated n-grams so Engram rows collide
    positions = torch.arange(length)[None]
    ngram = model.build_ngram_tokens(ids, positions)
    engram_mask = ids != config.engram_pad_id

    layer_out: list[torch.Tensor] = []
    sub_out: dict[str, torch.Tensor] = {}
    for index, layer in enumerate(model.layers):
        layer.register_forward_hook(lambda module, args, out: layer_out.append(out.detach().clone()))
        indexer = getattr(layer.self_attn, "compressed_indexer", None)
        if indexer is not None:
            indexer.register_forward_hook(
                lambda module, args, out, key=f"{index}.selected": sub_out.__setitem__(
                    key, (out[0, 0] > -1).detach().clone()
                )
            )
        for name in ("engram_layer", "mlp", "self_attn"):
            module = getattr(layer, name, None)
            if module is not None:
                module.register_forward_hook(
                    lambda module, args, out, key=f"{index}.{name}": sub_out.__setitem__(
                        key, (out[0] if isinstance(out, tuple) else out).detach().clone()
                    )
                )
    with torch.no_grad():
        hidden = model(
            input_ids=ids, position_ids=positions, ngram_tokens=ngram, engram_mask=engram_mask, use_cache=False
        ).last_hidden_state
        logits = lm_head(hidden)

    state = {}
    for key, value in model.state_dict().items():
        released = _released_text_key(key, config.mlp_layer_types)
        if released is not None:
            state[released] = value.detach().clone()
    state["lm_head.weight"] = lm_head.weight.detach().clone()
    torch.save(
        {
            "state": state,
            "ids": ids,
            "logits": logits,
            "hidden": hidden,
            "layer_out": layer_out,
            "sub_out": sub_out,
            "transformers": transformers.__version__,
        },
        path,
    )


def dump_image_text(path: Path, seed: int) -> None:
    """Composite model: logits and M-RoPE positions of one text / image / text sequence."""
    from transformers.models.hy_v4_flash.configuration_hy_v4_flash import HYV4FlashConfig
    from transformers.models.hy_v4_flash.modeling_hy_v4_flash import HYV4FlashForConditionalGeneration

    torch.manual_seed(seed)
    config = HYV4FlashConfig(
        text_config=dict(TINY_TEXT_KWARGS),
        vision_config={**VISION_KWARGS, "model_type": "hy_v4_vl_vision"},
        image_token_id=IMAGE_TOKEN_ID,
    )
    config._attn_implementation = "eager"
    config.text_config._attn_implementation = "eager"
    model = HYV4FlashForConditionalGeneration(config).float().eval()
    _perturb(model, 0.05)
    grid = torch.tensor([[1, 4, 6]])  # merged: 2 x 3 (+1 newline per row) = 8 image tokens
    image_tokens = 2 * (3 + 1)
    before, after = torch.randint(3, 240, (7,)), torch.randint(3, 240, (9,))
    ids = torch.cat([before, torch.full((image_tokens,), IMAGE_TOKEN_ID), after])[None]
    types = (ids == IMAGE_TOKEN_ID).long()
    pixels = torch.randn(4 * 6, 3 * 4 * 4)
    with torch.no_grad():
        logits = model(
            input_ids=ids, pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=types, use_cache=False
        ).logits
        positions, _ = model.model.get_rope_index(ids, mm_token_type_ids=types, image_grid_thw=grid)

    state = {}
    for key, value in model.state_dict().items():
        if key.startswith("model.language_model."):
            released = _released_text_key(key.removeprefix("model.language_model."), config.text_config.mlp_layer_types)
        elif key.startswith("model.vision_tower."):
            released = "vit." + _rename(key.removeprefix("model.vision_tower."), VISION_RENAMES)
        elif key == "lm_head.weight":
            released = key
        else:
            raise KeyError(f"reference key without a released name: {key}")
        if released is not None:
            state[released] = value.detach().clone()
    torch.save(
        {
            "state": state,
            "ids": ids,
            "types": types,
            "pixels": pixels,
            "grid": grid,
            "logits": logits,
            "position_ids": positions,
            "vision_kwargs": VISION_KWARGS,
            "image_token_id": IMAGE_TOKEN_ID,
            "transformers": transformers.__version__,
        },
        path,
    )


def dump_vision(path: Path, seed: int) -> None:
    """Vision tower alone: the merged features of three images of different grids."""
    from transformers.models.hy_v4_vl.configuration_hy_v4_vl import HYV4VLVisionConfig
    from transformers.models.hy_v4_vl.modeling_hy_v4_vl import HYV4VLVisionModel

    torch.manual_seed(seed)
    config = HYV4VLVisionConfig(**VISION_KWARGS)
    config._attn_implementation = "eager"
    model = HYV4VLVisionModel(config).float().eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.1 * torch.randn_like(parameter))
    grid = torch.tensor([[1, 4, 6], [1, 2, 4], [2, 4, 4]])
    pixels = torch.randn(int(grid.prod(dim=1).sum()), 3 * 4 * 4)
    with torch.no_grad():
        features = model(pixels, grid).pooler_output[0]
    state = {"vit." + _rename(key, VISION_RENAMES): value.detach().clone() for key, value in model.state_dict().items()}
    torch.save(
        {
            "state": state,
            "pixels": pixels,
            "grid": grid,
            "features": features,
            "kwargs": VISION_KWARGS,
            "transformers": transformers.__version__,
        },
        path,
    )


def main() -> None:
    """Write one dump."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("kind", choices=["text", "image-text", "vision"])
    parser.add_argument("path", type=Path)
    parser.add_argument("--length", type=int, default=97, help="tokens of the text dump")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()
    if arguments.kind == "text":
        dump_text(arguments.path, arguments.length, arguments.seed)
    elif arguments.kind == "image-text":
        dump_image_text(arguments.path, arguments.seed)
    else:
        dump_vision(arguments.path, arguments.seed)
    print(f"saved {arguments.path} with transformers {transformers.__version__} from {transformers.__file__}")


if __name__ == "__main__":
    main()
