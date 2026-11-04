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

"""Compare D1-3B's tiny native AutoModel backbone with its pinned release."""

import argparse
import json

import torch
from d1_publisher_fixtures import initialize_nonunit_rmsnorm
from transformers import Lfm2VlConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from nemo_automodel.components.models.d1.releases import RELEASES
from nemo_automodel.components.models.d1.vl_model import Lfm2VlForConditionalGeneration


def tiny_config() -> Lfm2VlConfig:
    return Lfm2VlConfig(
        architectures=["Lfm2VlForConditionalGeneration"],
        image_token_id=31,
        tie_word_embeddings=True,
        projector_hidden_size=64,
        projector_use_layernorm=False,
        projector_bias=True,
        text_config={
            "vocab_size": 32,
            "hidden_size": 64,
            "intermediate_size": 96,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "layer_types": ["conv", "full_attention"],
            "conv_L_cache": 3,
            "conv_bias": False,
            "block_auto_adjust_ff_dim": False,
            "pad_token_id": 0,
            "use_cache": False,
        },
        vision_config={
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "patch_size": 16,
            "num_patches": 16,
            "vision_use_head": False,
        },
    )


def main() -> None:
    """Run the explicit publisher comparison; fetch pinned code only when invoked."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(7)
    repository, revision, _ = RELEASES["d1-3b"]
    reference_class = get_class_from_dynamic_module(
        "lfm2_vl.Lfm2VlForConditionalGeneration", repository, revision=revision, code_revision=revision
    )
    config = tiny_config()
    reference = reference_class(config).eval().to(device)
    # HF's released from_pretrained path reties after replacing the language stack.
    reference.tie_weights()
    nonunit_rmsnorms = initialize_nonunit_rmsnorm(reference)
    native = Lfm2VlForConditionalGeneration(config).eval().to(device)
    assert set(reference.state_dict()) == set(native.state_dict())
    native.load_state_dict(reference.state_dict(), strict=True)
    assert reference.lm_head.weight is reference.get_input_embeddings().weight
    assert native.lm_head.weight is native.get_input_embeddings().weight
    checks = []
    for dtype in (torch.float32, torch.bfloat16):
        reference.to(dtype)
        native.to(dtype)
        for modality in ("text", "image"):
            if modality == "image":
                ids = torch.tensor([[1, 31, 31, 31, 31, 2, 3]], device=device)
                vision = {
                    "pixel_values": torch.randn(1, 16, 768, dtype=dtype, device=device),
                    "spatial_shapes": torch.tensor([[4, 4]], device=device),
                    "pixel_attention_mask": torch.ones(1, 16, dtype=torch.int32, device=device),
                }
            else:
                ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7]], device=device)
                vision = {}
            expected = reference(ids, use_cache=False, **vision).logits
            actual = native(ids, use_cache=False, **vision).logits
            tolerance = 2e-5 if dtype == torch.float32 else 0.01
            torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
            cosine = torch.nn.functional.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0)
            assert cosine > 0.9999
            native.zero_grad()
            reference.zero_grad()
            probe = torch.randn_like(actual, dtype=torch.float32)
            (actual.float() * probe).mean().backward()
            (expected.float() * probe).mean().backward()
            max_gradient_error = 0.0
            parameters_checked = 0
            for name, parameter in reference.named_parameters():
                if parameter.grad is not None:
                    comparison = dict(native.named_parameters())[name].grad
                    torch.testing.assert_close(comparison, parameter.grad, atol=tolerance, rtol=tolerance)
                    max_gradient_error = max(max_gradient_error, float((comparison - parameter.grad).abs().max()))
                    parameters_checked += 1
            checks.append(
                {
                    "modality": modality,
                    "dtype": str(dtype),
                    "max_logit_abs_error": float((actual - expected).abs().max().detach()),
                    "logit_cosine": float(cosine.detach()),
                    "max_parameter_gradient_error": max_gradient_error,
                    "parameters_with_gradients_checked": parameters_checked,
                }
            )
    print(
        json.dumps(
            {
                "scope": "AutoModel tiny D1-3B image/text backbone",
                "source_revision": revision,
                "checkpoint_keys": len(reference.state_dict()),
                "strict_full_weight_load": True,
                "nonunit_rmsnorms": nonunit_rmsnorms,
                "tied_identity": "preserved",
                "torch_version": torch.__version__,
                "device": str(device),
                "checks": checks,
            }
        )
    )


if __name__ == "__main__":
    main()
