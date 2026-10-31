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

"""Check full tiny Omni weight coverage, media paths, and decision backward."""

import argparse
import json

import numpy as np
import torch
from d1_publisher_fixtures import initialize_nonunit_rmsnorm
from PIL import Image
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nemo_automodel.components.models.d1.config import D1OmniConfig
from nemo_automodel.components.models.d1.model import (
    OMNI_REPOSITORY,
    OMNI_REVISION,
    D1OmniModel,
)


def main() -> None:
    """Run the explicit publisher comparison; fetch pinned code only when invoked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    config = D1OmniConfig(
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "norm_eps": 1e-5,
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_ffn_dim_multiplier": 1.0,
            "block_multiple_of": 16,
            "rope_theta": 1e6,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 256,
            "vision_use_head": False,
        },
        audio_config={
            "feat_in": 128,
            "n_layers": 1,
            "d_model": 64,
            "subsampling_conv_channels": 8,
            "ff_expansion_factor": 2,
            "n_heads": 1,
            "conv_kernel_size": 9,
            "residual_width": 32,
        },
        projector_hidden_size=64,
        head_layers=2,
        max_length=128,
        temperatures={"choice:3-5": 1.4, "score:3-5": 1.7, "noul:2": 1.5},
    )

    released_class = get_class_from_dynamic_module(
        "modeling_d1.D1OmniModel", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
    )
    released_config = released_class.config_class(**config.to_dict())
    released = released_class(released_config).eval().to(device)
    nonunit_rmsnorms = initialize_nonunit_rmsnorm(released)
    native = D1OmniModel(config, trust_remote_code=True).eval().to(device)
    expected = released.state_dict()
    assert set(native.state_dict()) == set(expected)
    native.load_state_dict(expected, strict=True)
    question_class = get_class_from_dynamic_module(
        "prompt.Question", OMNI_REPOSITORY, revision=OMNI_REVISION, code_revision=OMNI_REVISION
    )
    questions = [
        (
            question_class("choice", "Choose an action", {"A": "first", "B": "second", "C": "third"}),
            [1, 3, 5],
            0,
            "choice:3-5",
        ),
        (
            question_class("score", "Rate risk", ["none", "low", "medium", "high", "critical"]),
            [1, 2, 3, 4, 5],
            1,
            "score:3-5",
        ),
        (question_class("noul", "Is this evidence suspicious?"), [1, 3], 2, "noul:2"),
    ]
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
    valid = torch.ones_like(ids, dtype=torch.bool)
    positions = torch.tensor([[1, 3]], device=device)
    markers = torch.ones_like(positions, dtype=torch.bool)
    types = torch.tensor([2], device=device)
    image = Image.new("RGB", (32, 32), color=(100, 50, 200))
    waveform = np.sin(np.arange(8000, dtype=np.float32) * 0.1) * 0.1
    checks = []
    for modality in ("text", "image", "audio"):
        if modality == "image":
            reference_prefix = released.vision([image])
            prefix = native.encode_images([image])
        elif modality == "audio":
            reference_prefix = released.audio(waveform)
            prefix = native.encode_audio(waveform)
        else:
            reference_prefix = prefix = None
        if prefix is not None:
            torch.testing.assert_close(prefix, reference_prefix, rtol=0, atol=0)
            lengths = torch.tensor([prefix.shape[1]], device=device)
        else:
            lengths = None
        for question, marker_indices, type_id, temperature_key in questions:
            positions = torch.tensor([marker_indices], device=device)
            markers = torch.ones_like(positions, dtype=torch.bool)
            types = torch.tensor([type_id], device=device)
            reference_probabilities = torch.tensor(
                released._forward([(reference_prefix, ids[0].tolist(), marker_indices, question, modality == "text")]),
                device=device,
            )
            result = native(ids, valid, positions, markers, types, prefix, lengths)
            logits = result.logits
            if modality == "text":
                logits = logits / config.temperatures[temperature_key]
            probabilities = logits.softmax(-1)
            if question.type == "noul":
                probabilities = probabilities.flip(-1)
            torch.testing.assert_close(probabilities, reference_probabilities, atol=2e-6, rtol=2e-6)
            checks.append(
                {
                    "modality": modality,
                    "question_type": question.type,
                    "options": question.options,
                    "prefix_length": 0 if prefix is None else prefix.shape[1],
                    "public_probability_max_abs_diff": float(
                        (probabilities - reference_probabilities).abs().max().detach()
                    ),
                }
            )
    training = native(ids, valid, positions, markers, types, labels=torch.tensor([1], device=device))
    training.loss.backward()
    assert native.encoder.embed_tokens.weight.grad.abs().sum() > 0
    assert native.head.scorer[3].weight.grad.abs().sum() > 0
    print(
        json.dumps(
            {
                "scope": "AutoModel full tiny Omni model with released perception modules",
                "source_revision": OMNI_REVISION,
                "checkpoint_keys": len(expected),
                "strict_full_weight_load": True,
                "nonunit_rmsnorms": nonunit_rmsnorms,
                "modalities": checks,
                "training_loss": float(training.loss.detach()),
                "embedding_and_head_backward": "passed",
                "torch_version": torch.__version__,
                "device": str(device),
            }
        )
    )


if __name__ == "__main__":
    main()
