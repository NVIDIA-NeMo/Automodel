# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Generate LFM3 and LFM3-VL goldens from the native Transformers implementation.

Run on CPU in an environment with ``transformers==5.23.0``, the first release
that ships both models; the release Automodel pins has neither, so the tests
cannot build this reference themselves. No NeMo code, network call or trained
weight takes part:

    python generate_reference.py reference.npz

Weights are stored under the names Transformers writes to disk, which are the
names the Automodel modules use.
"""

import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import transformers
from safetensors.torch import load_file
from transformers import Lfm3Config, Lfm3ForCausalLM, Lfm3VlConfig, Lfm3VlForConditionalGeneration

SEED = 1739
IMAGE_TOKEN, VIDEO_TOKEN, FIRST_TEXT_TOKEN = 10, 11, 14
TEXT_CONFIG = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 5,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "sliding_window": 4,
    "max_position_embeddings": 128,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
}
VISION_CONFIG = {
    "hidden_size": 32,
    "intermediate_size": 48,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "patch_size": 4,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
}
VL_CONFIG = {
    "projector_hidden_size": 40,
    "image_token_id": IMAGE_TOKEN,
    "video_token_id": VIDEO_TOKEN,
    "vision_start_token_id": 12,
    "vision_end_token_id": 13,
}
GENERATE = {"max_new_tokens": 8, "do_sample": False, "eos_token_id": None, "pad_token_id": 0}
GRADIENTS = (
    "model.language_model.embed_tokens.weight",
    "model.language_model.layers.0.conv.conv.weight",
    "model.language_model.layers.2.self_attn.gate_proj.weight",
    "model.language_model.layers.4.self_attn.q_proj.weight",
    "model.vision_tower.patchifier.weight",
    "model.vision_tower.layers.0.conv.spatial_conv.weight",
    "model.vision_tower.layers.2.self_attn.q_layernorm.weight",
    "model.multi_modal_projector.linear_1.weight",
)


def native_name(name: str) -> str:
    """Map a parameter's on-disk name to its name in the native model, which renames a few on load."""
    name = name.replace(".gate_proj.", ".g_proj.")
    if name.startswith("model.vision_tower."):
        name = (
            name.replace("patchifier", "patch_embed").replace("q_layernorm", "q_norm").replace("k_layernorm", "k_norm")
        )
    return name


def main(archive_path: Path) -> None:
    torch.manual_seed(SEED)
    vl = Lfm3VlForConditionalGeneration(
        Lfm3VlConfig(text_config={**TEXT_CONFIG, "model_type": "lfm3"}, vision_config=VISION_CONFIG, **VL_CONFIG)
    ).eval()
    with torch.no_grad():
        for parameter in vl.parameters():
            parameter.normal_(mean=1.0 if parameter.ndim == 1 else 0.0, std=0.2)
    with tempfile.TemporaryDirectory() as directory:
        vl.save_pretrained(directory)
        state = load_file(Path(directory) / "model.safetensors")
    text = Lfm3ForCausalLM(Lfm3Config(**TEXT_CONFIG)).eval()
    text.model.load_state_dict(vl.model.language_model.state_dict())

    # Two images (patch grids 1x4x6 and 1x2x4: 6 and 2 tokens) and one video (2x4x4: two frames of 4 tokens).
    ids = torch.randint(FIRST_TEXT_TOKEN, TEXT_CONFIG["vocab_size"], (2, 30))
    text_ids = ids.clone()
    ids[0, 2:8], ids[0, 12:16], ids[0, 19:23], ids[1, 5:7] = IMAGE_TOKEN, VIDEO_TOKEN, VIDEO_TOKEN, IMAGE_TOKEN
    mask = torch.ones_like(ids)
    mask[1, :3] = 0
    media = {
        "pixel_values": torch.randn(24 + 8, 96),
        "image_grid_thw": torch.tensor([[1, 4, 6], [1, 2, 4]]),
        "pixel_values_videos": torch.randn(32, 96),
        "video_grid_thw": torch.tensor([[2, 4, 4]]),
    }
    arrays = {f"state.{name}": value for name, value in state.items()}
    arrays |= {"input.ids": ids, "input.text_ids": text_ids, "input.mask": mask}
    arrays |= {f"input.{name}": value for name, value in media.items()}

    parameters = dict(vl.named_parameters())
    vl(input_ids=ids, attention_mask=mask, labels=ids, **media).loss.backward()
    arrays |= {f"expected.vl.grad.{name}": parameters[native_name(name)].grad for name in GRADIENTS}
    with torch.no_grad():
        output = vl(input_ids=ids, attention_mask=mask, labels=ids, **media)
        arrays |= {"expected.vl.logits": output.logits, "expected.vl.loss": output.loss}
        arrays["expected.vl.generated"] = vl.generate(input_ids=ids, attention_mask=mask, **media, **GENERATE)
        output = text(input_ids=text_ids, attention_mask=mask, labels=text_ids)
        arrays |= {"expected.text.logits": output.logits, "expected.text.loss": output.loss}
        arrays["expected.text.generated"] = text.generate(input_ids=text_ids, attention_mask=mask, **GENERATE)

    np.savez_compressed(archive_path, **{name: value.detach().numpy() for name, value in arrays.items()})
    metadata = {
        "source": "transformers.models.lfm3 and transformers.models.lfm3_vl",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "array_count": len(arrays),
        "seed": SEED,
        "dtype": "float32",
        "text_config": TEXT_CONFIG,
        "vision_config": VISION_CONFIG,
        "vl_config": VL_CONFIG,
        "generate": GENERATE,
    }
    archive_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
