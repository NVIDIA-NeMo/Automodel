# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
"""CPU validation script for LLaVA-OneVision-1.5 integration.

This script validates the implementation on CPU without requiring GPU.
It tests:
  1. Model loading (config + architecture)
  2. Data pipeline (dataset builder + collate function)
  3. Forward pass shape correctness

Usage:
    python tests/integration/test_llava_onevision_cpu.py
"""

import logging
import sys
from pathlib import Path

import torch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_model_config():
    """Test 1: Verify model config and architecture can be instantiated."""
    logger.info("=" * 80)
    logger.info("Test 1: Model Config & Architecture")
    logger.info("=" * 80)

    from nemo_automodel.components.models.llava_onevision.model import (
        LLaVAOneVision1_5_ForConditionalGeneration,
        Llavaonevision1_5Config,
        RiceConfig,
    )

    # Create a small test config
    vision_config = RiceConfig(
        depth=2,
        hidden_size=64,
        intermediate_size=128,
        num_heads=4,
        patch_size=14,
        spatial_merge_size=2,
        text_hidden_size=64,
    )

    text_config = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=1000,
        rope_theta=1e6,
        max_position_embeddings=128,
    )

    config = Llavaonevision1_5Config(
        vision_config=vision_config,
        text_config=text_config,
        image_token_id=100,
        video_token_id=101,
        vision_start_token_id=98,
        vision_end_token_id=99,
    )

    logger.info("  ✓ Llavaonevision1_5Config created")
    logger.info(f"    - Vision depth: {config.vision_config.depth}")
    logger.info(f"    - Vision hidden size: {config.vision_config.hidden_size}")
    logger.info(f"    - Text hidden size: {config.text_config.hidden_size}")
    logger.info(f"    - Vocab size: {config.text_config.vocab_size}")
    logger.info(f"    - Image token ID: {config.image_token_id}")

    # Instantiate model
    model = LLaVAOneVision1_5_ForConditionalGeneration(config, attn_implementation="eager")
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"  ✓ Model instantiated with {param_count:,} parameters")

    return config, model


def test_forward_pass_shape(config, model):
    """Test 2: Verify forward pass returns correct tensor shapes."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 2: Forward Pass Shape (Text-Only)")
    logger.info("=" * 80)

    batch_size = 2
    seq_len = 16

    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True,
        )

    # Verify output shapes
    assert outputs.logits.shape == (batch_size, seq_len, config.text_config.vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, config.text_config.vocab_size)}, got {outputs.logits.shape}"
    )

    logger.info("  ✓ Forward pass successful")
    logger.info(f"    - Input IDs shape: {input_ids.shape}")
    logger.info(f"    - Logits shape: {outputs.logits.shape}")
    logger.info(f"    - Loss: {outputs.loss.item():.4f}" if outputs.loss is not None else "    - Loss: None")

    return True


def test_forward_pass_with_image(config, model):
    """Test 3: Verify forward pass with image features."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 3: Forward Pass with Image Features")
    logger.info("=" * 80)

    batch_size = 1
    seq_len = 32
    num_patches = 4  # Small number for CPU testing

    # Create input with image token
    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    # Insert image token at a specific position
    input_ids[0, 5] = config.image_token_id
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # Create dummy pixel values and grid
    # Shape: [num_patches, C*P*P] where C=3, P=14
    pixel_values = torch.randn(num_patches, 3 * 14 * 14)
    # Grid dimensions: [num_images, 3] as (T, H, W)
    image_grid_thw = torch.tensor([[1, 2, 2]])  # 1 image, 2x2 grid after merging

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    assert outputs.logits.shape == (batch_size, seq_len, config.text_config.vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, config.text_config.vocab_size)}, got {outputs.logits.shape}"
    )

    logger.info("  ✓ Forward pass with image features successful")
    logger.info(f"    - Input IDs shape: {input_ids.shape}")
    logger.info(f"    - Pixel values shape: {pixel_values.shape}")
    logger.info(f"    - Image grid THW: {image_grid_thw}")
    logger.info(f"    - Logits shape: {outputs.logits.shape}")

    return True


def test_forward_pass_with_multiple_images(config, model):
    """Test 3b: Multi-image forward pass exercises the per-segment CLS-strip path.

    With a single image the strip loop only runs once, so ``hidden_states[s+1:e+1]``
    happens to be correct. With >=2 images, reading the source slice from the
    pre-CLS ``cu`` instead of the post-CLS ``cu_seqlens`` drops the tail patch of
    each later segment and pulls in the next segment's CLS.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 3b: Forward Pass with Multiple Image Features")
    logger.info("=" * 80)

    batch_size = 1
    seq_len = 32
    # Two images: 4 patches each pre-merge, 1 token each post spatial_merge_size=2.
    image_grid_thw = torch.tensor([[1, 2, 2], [1, 2, 2]])
    total_patches = int((image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]).sum().item())
    pixel_values = torch.randn(total_patches, 3 * 14 * 14)

    input_ids = torch.randint(0, config.text_config.vocab_size, (batch_size, seq_len))
    input_ids[0, 5] = config.image_token_id  # first image placeholder
    input_ids[0, 15] = config.image_token_id  # second image placeholder
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            return_dict=True,
        )

    assert outputs.logits.shape == (batch_size, seq_len, config.text_config.vocab_size), (
        f"Expected logits shape {(batch_size, seq_len, config.text_config.vocab_size)}, got {outputs.logits.shape}"
    )
    assert torch.isfinite(outputs.logits).all().item(), "Non-finite logits with multi-image input."

    logger.info("  ✓ Multi-image forward pass successful")
    logger.info(f"    - Grid THW: {image_grid_thw.tolist()}")
    logger.info(f"    - Pixel patches: {total_patches}")
    logger.info(f"    - Logits shape: {outputs.logits.shape}")

    return True


def test_collate_function():
    """Test 4: Verify collate function works correctly."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 4: Collate Function")
    logger.info("=" * 80)

    from nemo_automodel.components.datasets.vlm.collate_fns import llava_onevision_collate_fn

    # Create mock examples
    examples = [
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": None},  # Will be handled by processor
                        {"type": "text", "text": "Describe this image."},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "This is a test response."}]},
            ],
        },
        {
            "conversation": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is 2+2?"},
                    ],
                },
                {"role": "assistant", "content": [{"type": "text", "text": "4"}]},
            ],
        },
    ]

    logger.info("  ✓ Collate function imported successfully")
    logger.info(f"    - Created {len(examples)} mock examples")

    # Note: We can't fully test the collate function without a real processor
    # This test verifies the function can be imported and has correct signature
    import inspect

    sig = inspect.signature(llava_onevision_collate_fn)
    params = list(sig.parameters.keys())
    assert "examples" in params, "Missing 'examples' parameter"
    assert "processor" in params, "Missing 'processor' parameter"

    logger.info(f"  ✓ Collate function signature correct: {params}")

    return True


def test_dataset_builder():
    """Test 5: Verify dataset builder can be imported."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 5: Dataset Builder")
    logger.info("=" * 80)

    import inspect

    from nemo_automodel.components.datasets.vlm.datasets import make_llava_onevision_dataset

    sig = inspect.signature(make_llava_onevision_dataset)
    params = list(sig.parameters.keys())

    assert "path_or_dataset" in params, "Missing 'path_or_dataset' parameter"
    assert "split" in params, "Missing 'split' parameter"

    logger.info("  ✓ Dataset builder imported successfully")
    logger.info(f"    - Function signature: {params}")
    logger.info("    - Default dataset: liuhaotian/LLaVA-Instruct-150K")

    return True


def test_registry_registration():
    """Test 6: Verify model is registered in the architecture mapping."""
    logger.info("")
    logger.info("=" * 80)
    logger.info("Test 6: Registry Registration")
    logger.info("=" * 80)

    from nemo_automodel._transformers.registry import ModelRegistry

    arch_name = "LLaVAOneVision1_5_ForConditionalGeneration"

    assert arch_name in ModelRegistry.model_arch_name_to_cls, f"{arch_name} not found in MODEL_ARCH_MAPPING"

    model_cls = ModelRegistry.model_arch_name_to_cls[arch_name]
    logger.info("  ✓ Model registered in MODEL_ARCH_MAPPING")
    logger.info(f"    - Architecture name: {arch_name}")
    logger.info(f"    - Model class: {model_cls.__name__}")
    logger.info(f"    - Module: {model_cls.__module__}")

    return True


def main():
    """Run all CPU validation tests."""
    logger.info("LLaVA-OneVision-1.5 CPU Validation")
    logger.info("=" * 80)
    logger.info("This script validates the implementation on CPU without GPU.")
    logger.info("Tests: config, forward pass, collate fn, dataset builder, registry")
    logger.info("")

    results = {}

    try:
        # Test 1: Model config
        config, model = test_model_config()
        results["Model Config"] = True

        # Test 2: Forward pass (text-only)
        results["Forward Pass (Text)"] = test_forward_pass_shape(config, model)

        # Test 3: Forward pass with image
        results["Forward Pass (Image)"] = test_forward_pass_with_image(config, model)

        # Test 3b: multi-image regression — covers the per-segment CLS-strip path.
        results["Forward Pass (Multi-Image)"] = test_forward_pass_with_multiple_images(config, model)

        # Test 4: Collate function
        results["Collate Function"] = test_collate_function()

        # Test 5: Dataset builder
        results["Dataset Builder"] = test_dataset_builder()

        # Test 6: Registry
        results["Registry"] = test_registry_registration()

    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results["Overall"] = False

    # Summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"  {status}: {test_name}")
        if not passed:
            all_passed = False

    if all_passed:
        logger.info("")
        logger.info("✅ All CPU validation tests passed!")
        logger.info("The implementation is ready for PR submission.")
        logger.info("Note: Actual training requires GPU access (A100/H100 recommended).")
        return 0
    else:
        logger.info("")
        logger.info("❌ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
