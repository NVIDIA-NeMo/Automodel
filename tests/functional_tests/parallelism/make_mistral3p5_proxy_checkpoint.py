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

"""Build a small random Mistral3.5 checkpoint for the parallelism parity tests.

Same shape as `examples/vlm_finetune/mistral3p5/mistral3p5_128b_medpix.yaml`,
just small enough for a 2-GPU runner. Writes the weights plus the real
processor, so the test has one directory to point at.

The processor is saved from the cached `mistralai/Mistral-Medium-3.5-128B`,
which is why `vocab_size` and `image_token_index` stay at their real values --
the processor emits those token ids and the embedding has to cover them.

Usage:
    python make_mistral3p5_proxy_checkpoint.py --output-dir DIR
"""

import argparse
from pathlib import Path

import torch

SOURCE_MODEL = "mistralai/Mistral-Medium-3.5-128B"

# Kept at the real values because the processor depends on them.
VOCAB_SIZE = 131072
IMAGE_TOKEN_INDEX = 10
SPATIAL_MERGE_SIZE = 2
# Shrunk, except the vision patch size, which has to match the processor.
TEXT_HIDDEN = 256
VISION_HIDDEN = 128


def build_config():
    """Build the shrunk Mistral3 config.

    Returns:
        A ``Mistral3Config`` with a ministral3 text tower and a pixtral vision
        tower, both scaled down but keeping the fields the processor and the
        parallelism code depend on.
    """
    from transformers.models.mistral3.configuration_mistral3 import Mistral3Config

    text_config = {
        "model_type": "ministral3",
        "vocab_size": VOCAB_SIZE,
        "hidden_size": TEXT_HIDDEN,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 4096,
        "rms_norm_eps": 1e-5,
        # Same yarn setup as the real config, retuned for the shorter context:
        # factor 1.0 with original == max keeps it consistent. The fields are
        # not optional -- ministral3 attention reads
        # `original_max_position_embeddings` and `llama_4_scaling_beta`.
        "rope_parameters": {
            "rope_type": "yarn",
            "type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 1.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 4.0,
            "beta_slow": 1.0,
            "mscale": 1.0,
            "mscale_all_dim": 0.0,
            "llama_4_scaling_beta": 0,
        },
        "bos_token_id": 1,
        "eos_token_id": 2,
        "tie_word_embeddings": False,
        "use_cache": False,
        # pad_token_id is deliberately left out: it would give the embedding a
        # padding_idx, and under FSDP2 that makes weight initialization get
        # skipped. Harmless here because the checkpoint carries real weights,
        # but it keeps the config honest about what the test relies on.
    }
    vision_config = {
        "model_type": "pixtral",
        "hidden_size": VISION_HIDDEN,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "head_dim": 32,
        "image_size": 1024,
        # Must match the processor: PixtralImageProcessor uses patch_size 14,
        # and a mismatch makes the image-token count disagree with the number
        # of vision features.
        "patch_size": 14,
    }
    config = Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        image_token_index=IMAGE_TOKEN_INDEX,
        spatial_merge_size=SPATIAL_MERGE_SIZE,
        multimodal_projector_bias=False,
        projector_hidden_act="gelu",
        tie_word_embeddings=False,
    )
    config.architectures = ["Mistral3ForConditionalGeneration"]
    return config


def main() -> None:
    """Write the proxy checkpoint and its processor."""
    parser = argparse.ArgumentParser(description="Build a Mistral3.5 proxy checkpoint")
    parser.add_argument("--output-dir", required=True, help="Directory to write the checkpoint into")
    parser.add_argument("--seed", type=int, default=1234, help="Seed for weight initialization")
    args = parser.parse_args()

    # Built with the stock HF class, not the NeMo subclass: this only produces
    # an on-disk artifact, and NeMo's save_pretrained needs a live checkpointer.
    # The saved `architectures` still routes from_pretrained to the NeMo class.
    from transformers import AutoProcessor
    from transformers.models.mistral3.modeling_mistral3 import Mistral3ForConditionalGeneration

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model = Mistral3ForConditionalGeneration(build_config()).to(torch.bfloat16)

    embed_absmax = float(model.get_input_embeddings().weight.detach().abs().max())
    if embed_absmax == 0.0:
        raise RuntimeError(
            "Proxy checkpoint would be all zeros: HF initialization did not run. "
            "A zero model emits uniform logits and zero gradients, which makes the "
            "parity comparison meaningless."
        )

    model.save_pretrained(str(output_dir))
    AutoProcessor.from_pretrained(SOURCE_MODEL).save_pretrained(str(output_dir))

    print(f"Wrote Mistral3.5 proxy checkpoint to {output_dir} (embed absmax {embed_absmax:.6f})")


if __name__ == "__main__":
    main()
