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

"""Real-checkpoint Bolt E2B parity and two-GPU FSDP2 training checks.

Run parity on one GPU, or training on two GPUs, from the repository root::

    python tests/functional_tests/models/gemma4/run_bolt_e2b.py --mode parity --model-path /checkpoint
    torchrun --standalone --nproc-per-node=2 tests/functional_tests/models/gemma4/run_bolt_e2b.py \
        --mode lora --model-path /checkpoint --output-dir /node-local/bolt-e2b

Use a node-local output directory. The source checkpoint is read without modification.
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.distributed.tensor import DTensor
from transformers import AutoModelForImageTextToText, AutoProcessor

from nemo_automodel import NeMoAutoModelForImageTextToText
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.datasets.vlm.collate_fns import default_collate_fn
from nemo_automodel.components.distributed.config import DistributedSetup, FSDP2Config
from nemo_automodel.components.distributed.mesh import ParallelismSizes
from nemo_automodel.components.models.gemma4_moe.model import Gemma4ForConditionalGeneration
from nemo_automodel.components.utils.model_utils import FreezeConfig


def _local(tensor: torch.Tensor) -> torch.Tensor:
    """Access a tensor's local storage without changing its layout.

    Args:
        tensor: Tensor of arbitrary shape, or a DTensor sharded over its data-parallel mesh.

    Returns:
        Tensor with the same axis order and per-rank local shape, aliasing the input storage.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def _parity(model_path: str, device: torch.device, with_image: bool, attention: str) -> dict[str, float]:
    """Compare real BF16 weights and text logits through the public loading entry point."""
    inputs = {"input_ids": torch.tensor([[2, 1001, 1002, 1003, 1004, 1005, 1006, 1007]], device=device)}
    if with_image:
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
        processor.tokenizer.padding_side = "right"
        examples = [
            {
                "conversation": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": Image.new("RGB", (96, 96), "gray")},
                            {"type": "text", "text": "Describe this image."},
                        ],
                    },
                    {"role": "assistant", "content": [{"type": "text", "text": "A plain gray image."}]},
                ]
            }
        ]
        batch = default_collate_fn(examples, processor=processor, max_length=512)
        assert (batch["labels"] != -100).any()
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in batch.items()
            if key not in ("labels", "n_images_per_sample")
        }
    start = time.monotonic()
    reference = (
        AutoModelForImageTextToText.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation=attention,
            local_files_only=True,
        )
        .to(device)
        .eval()
    )
    native = NeMoAutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        attn_implementation=attention,
        local_files_only=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
    ).eval()
    assert isinstance(native, Gemma4ForConditionalGeneration)
    assert native.lm_head.weight is native.model.language_model.embed_tokens.weight
    actual_state = native.state_dict()
    assert set(actual_state) == set(reference.state_dict())
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(actual_state[name], tensor, atol=0, rtol=0, msg=name)
    with torch.no_grad():
        expected = reference(**inputs, use_cache=False).logits.float()
        actual = native(**inputs, use_cache=False).logits.float()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.01)
    result = {
        "max_abs_logit_difference": (actual - expected).abs().max().item(),
        "logit_cosine_similarity": F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item(),
        "load_and_check_seconds": time.monotonic() - start,
    }
    del native, reference
    gc.collect()
    torch.cuda.empty_cache()
    return result


def _train(
    model_path: str, output_dir: Path, steps: int, peft: bool, device: torch.device, attention: str
) -> dict[str, float | list[float]]:
    """Train with FSDP2 and verify adapter/model and optimizer restoration."""
    setup = DistributedSetup.build(
        strategy=FSDP2Config(
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                output_dtype=torch.bfloat16,
                cast_forward_inputs=False,
            )
        ),
        parallelism_sizes=ParallelismSizes(dp_size=dist.get_world_size()),
        activation_checkpointing=True,
    )
    adapter = (
        PeftConfig(
            target_modules=[
                "*language_model.layers.*.self_attn.q_proj",
                "*language_model.layers.*.self_attn.k_proj",
                "*language_model.layers.*.self_attn.v_proj",
                "*language_model.layers.*.self_attn.o_proj",
                "*language_model.layers.*.mlp.gate_proj",
                "*language_model.layers.*.mlp.up_proj",
                "*language_model.layers.*.mlp.down_proj",
            ],
            dim=16,
            alpha=32,
            use_triton=True,
        )
        if peft
        else None
    )
    model = NeMoAutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if peft else torch.float32,
        distributed_setup=setup,
        peft_config=adapter,
        freeze_config=FreezeConfig(freeze_vision_tower=True, freeze_audio_tower=True),
        attn_implementation=attention,
        local_files_only=True,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        text_config={"use_cache": False},
    ).train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=2e-4 if peft else 1e-5)
    ids = torch.tensor([[2, 1001, 1002, 1003, 1004, 1005, 1006, 1007]], device=device)
    losses = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(input_ids=ids, use_cache=False).logits
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), ids[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [_local(p.grad) for p in trainable if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], losses
    if peft:
        checkpointer = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(output_dir),
            model_save_format="safetensors",
            is_peft=True,
            save_consolidated=False,
        ).build(dp_rank=dist.get_rank(), tp_rank=0, pp_rank=0, moe_mesh=None)
        saved = output_dir / "resume"
        checkpointer.save_model(model, str(saved), peft_config=adapter)
        checkpointer.save_optimizer(optimizer, model, str(saved))
        checkpointer.async_wait()
        expected = [_local(p).detach().clone() for p in trainable]
        with torch.no_grad():
            for p in trainable:
                _local(p).zero_()
        checkpointer.load_model(model, str(saved / "model"))
        for p, value in zip(trainable, expected, strict=True):
            torch.testing.assert_close(_local(p), value, atol=0, rtol=0)
        restored_optimizer = torch.optim.AdamW(trainable, lr=optimizer.param_groups[0]["lr"])
        checkpointer.load_optimizer(restored_optimizer, model, str(saved))
        assert len(restored_optimizer.state) == len(optimizer.state)
        for p in trainable:
            for name in ("step", "exp_avg", "exp_avg_sq"):
                if p in optimizer.state:
                    torch.testing.assert_close(
                        _local(restored_optimizer.state[p][name]),
                        _local(optimizer.state[p][name]),
                        atol=0,
                        rtol=0,
                    )
        checkpointer.finalize()
    return {"losses": losses, "peak_gpu_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30}


def main() -> None:
    """Run one bounded real-checkpoint check and write machine-readable evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("parity", "image-parity", "lora", "full"), required=True)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--attn-implementation", choices=("eager", "sdpa"), default="eager")
    parser.add_argument("--output-dir", type=Path, default=Path("/tmp/bolt-e2b-validation"))
    args = parser.parse_args()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in ("parity", "image-parity"):
        result = _parity(
            args.model_path, device, with_image=args.mode == "image-parity", attention=args.attn_implementation
        )
    else:
        dist.init_process_group("nccl")
        try:
            result = _train(
                args.model_path, args.output_dir, args.steps, args.mode == "lora", device, args.attn_implementation
            )
        finally:
            dist.destroy_process_group()
    if int(os.environ.get("RANK", "0")) == 0:
        report = {"mode": args.mode, "model_path": args.model_path, "attention": args.attn_implementation, **result}
        (args.output_dir / f"{args.mode}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
