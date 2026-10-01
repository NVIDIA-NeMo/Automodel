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

"""Text-to-image generation with the native HunyuanImage-3.0 model under expert parallelism.

The MoE backbone runs natively (sharded with EP + FSDP2 across all ranks), loaded from the base release or from a
consolidated training checkpoint. Everything else follows the released sampler: the release input builder (chat
template, size tokens, classifier-free-guidance pair, attention mask), its flow-matching scheduler and guidance, and
its VAE decode. Each step runs the full sequence (the release reuses a KV cache after the first step; the result is the
same).

Example (one node, 8 GPUs):

    torchrun --nproc-per-node=8 examples/diffusion/generate/generate_hunyuan_image3.py \\
        --model /path/to/HunyuanImage-3.0 --output-dir ./hunyuan_image3_outputs \\
        --prompts "a cat sitting on a windowsill watching the rain"

    # Finetuned weights (consolidated HF-layout safetensors written by training):
    torchrun --nproc-per-node=8 examples/diffusion/generate/generate_hunyuan_image3.py \\
        --model /path/to/HunyuanImage-3.0 --weights /ckpts/run/epoch_X_step_Y/model/consolidated ...

    # LoRA checkpoint written by training (adapter on top of the base weights):
    torchrun --nproc-per-node=8 examples/diffusion/generate/generate_hunyuan_image3.py \\
        --model /path/to/HunyuanImage-3.0 --peft-checkpoint /ckpts/run/epoch_X_step_Y ...
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel.components._peft.lora import PeftConfig
from nemo_automodel.components.checkpoint.config import CheckpointingConfig
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.components.distributed.init_utils import initialize_distributed
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.hunyuan_image3.rope import build_2d_positions
from nemo_automodel.recipes._dist_utils import create_distributed_setup_from_config


def load_release_builder(model_dir: str, device: torch.device):
    """Release model without decoder layers: input builder, scheduler, guidance, VAE and image post-processing."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Use the release classes directly. Once nemo_automodel registers its native config for this model_type, the
    # Auto factories resolve to it, and it lacks the release's image/tokenizer defaults.
    release_cls = get_class_from_dynamic_module("hunyuan.HunyuanImage3ForCausalMM", model_dir)
    config = release_cls.config_class.from_pretrained(model_dir)
    config.num_hidden_layers = 0
    for key in ("moe_topk", "moe_intermediate_size", "num_shared_expert"):
        if isinstance(getattr(config, key, None), list):
            setattr(config, key, [])
    config.moe_impl = "eager"
    builder = release_cls.from_pretrained(
        model_dir, config=config, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": device}
    ).eval()
    builder.load_tokenizer(AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True))
    return builder


def load_peft_config(peft_checkpoint: str) -> PeftConfig:
    """LoRA config saved by training (``model/automodel_peft_config.json`` plus rank/alpha from ``adapter_config.json``)."""
    model_dir = Path(peft_checkpoint) / "model"
    config = json.loads((model_dir / "automodel_peft_config.json").read_text())
    adapter = json.loads((model_dir / "adapter_config.json").read_text())
    config.setdefault("dim", adapter["r"])
    config.setdefault("alpha", adapter["lora_alpha"])
    return PeftConfig.from_dict(config)


def load_native_model(model_dir: str, weights: str | None, world_size: int, peft_checkpoint: str | None = None):
    """Native backbone sharded with EP across all ranks.

    ``weights`` overrides the base checkpoint location; ``peft_checkpoint`` loads a training LoRA checkpoint on top.
    """
    cfg = ConfigNode(
        {
            "distributed": {
                "strategy": "fsdp2",
                "tp_size": 1,
                "cp_size": 1,
                "pp_size": 1,
                "ep_size": world_size,
                "activation_checkpointing": False,
            }
        }
    )
    setup = create_distributed_setup_from_config(cfg, world_size=world_size)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=False)
    config.include_vae_and_vision = False
    config.remote_code_dir = model_dir
    config._name_or_path = weights or model_dir
    backend = BackendConfig(
        attn="sdpa",
        linear="torch",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        gate_precision="float32",
        enable_hf_state_dict_adapter=True,
    )
    peft_config = load_peft_config(peft_checkpoint) if peft_checkpoint else None
    model = NeMoAutoModelForCausalLM.from_config(
        config=config,
        backend=backend,
        distributed_setup=setup,
        load_base_model=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        use_liger_kernel=False,
        use_sdpa_patching=False,
        peft_config=peft_config,
    )
    if peft_checkpoint:
        # Same loader the training recipe uses to resume LoRA checkpoints (EP-sharded expert adapters included).
        checkpointer = CheckpointingConfig(
            enabled=True,
            checkpoint_dir=str(Path(peft_checkpoint).parent),
            model_save_format="safetensors",
            is_peft=True,
            save_consolidated=False,
            model_repo_id=model_dir,
        ).build(dp_rank=dist.get_rank(), tp_rank=0, pp_rank=0, moe_mesh=setup.mesh_context.moe_mesh)
        checkpointer.load_model(model, str(Path(peft_checkpoint) / "model"))
    model.requires_grad_(False)
    return model.eval()


@torch.no_grad()
def generate(model, builder, prompt: str, seed: int, height: int, width: int, steps: int, guidance: float, device):
    """Release sampler with the native model predicting the flow velocity; returns a PIL image (rank 0)."""
    from importlib import import_module

    pipe = builder.pipeline
    release = import_module(type(pipe).__module__)
    inputs = builder.prepare_model_inputs(prompt=prompt, mode="gen_image", image_size=(height, width), seed=seed)
    out = inputs["tokenizer_output"]
    input_ids = inputs["input_ids"].to(device)
    attention_mask = builder._prepare_attention_mask_for_generation(input_ids, builder.generation_config, inputs).to(
        device
    )
    info = inputs["batch_gen_image_info"][0]
    token_h, token_w = info.token_height, info.token_width
    positions = torch.stack(
        [
            build_2d_positions(input_ids.shape[1], [(s.start, token_h, token_w) for s in out.gen_image_slices[row]])
            for row in range(input_ids.shape[0])
        ]
    ).to(device)

    timesteps, _ = release.retrieve_timesteps(pipe.scheduler, steps, device)
    generator = torch.Generator(device).manual_seed(seed)
    latents = pipe.prepare_latents(
        batch_size=1,
        latent_channel=builder.config.vae["latent_channels"],
        image_size=[info.image_height, info.image_width],
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    for t in timesteps:
        latent_input = torch.cat([latents] * input_ids.shape[0])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = model(
                input_ids,
                rope_positions=positions,
                attention_mask=attention_mask,
                mode="gen_image",
                images=latent_input,
                timestep=t.repeat(latent_input.shape[0]),
                image_mask=inputs["image_mask"].to(device),
                timestep_scatter_index=inputs["gen_timestep_scatter_index"].to(device),
            ).diffusion_prediction.float()
        pred_cond, pred_uncond = pred.chunk(2)
        pred = pipe.cfg_operator(pred_cond, pred_uncond, guidance, step=0)
        latents = pipe.scheduler.step(pred, t, latents, return_dict=False)[0]

    vae = builder.vae
    latents = latents / vae.config.scaling_factor
    if getattr(vae.config, "shift_factor", None):
        latents = latents + vae.config.shift_factor
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        image = vae.decode(latents.unsqueeze(2), return_dict=False)[0].squeeze(2)
    return pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True])[0]


def main():
    """Parse arguments, load the native model and the release builder, and generate one image per prompt."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="Release checkpoint directory (config, remote code, tokenizer, VAE).")
    p.add_argument("--weights", default=None, help="Optional directory of HF-layout safetensors to load instead.")
    p.add_argument(
        "--peft-checkpoint", default=None, help="Optional training checkpoint step directory with a LoRA adapter."
    )
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=42, help="Prompt i uses seed + i.")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=5.0)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    info = initialize_distributed("nccl", timeout_minutes=60)
    model = load_native_model(a.model, a.weights, info.world_size, a.peft_checkpoint)
    builder = load_release_builder(a.model, info.device)
    if info.is_main:
        os.makedirs(a.output_dir, exist_ok=True)
    for i, prompt in enumerate(a.prompts):
        # retrieve_timesteps -> scheduler.set_timesteps resets the scheduler's step index per prompt.
        image = generate(model, builder, prompt, a.seed + i, a.height, a.width, a.steps, a.guidance_scale, info.device)
        if info.is_main:
            safe = "".join(c if c.isalnum() or c in " _-" else "" for c in prompt)[:50].strip().replace(" ", "_")
            path = os.path.join(a.output_dir, f"sample_{i:03d}_{safe}.png")
            image.save(path)
            print("saved", path, flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
