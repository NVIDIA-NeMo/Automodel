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

"""Matched Qwen3.5-VL packed-training probe for AutoModel and Accelerate FSDP2."""

from __future__ import annotations

import inspect
import json
import logging
import os
import pathlib
import time
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn.functional as F

from nemo_automodel._transformers.utils import apply_cache_compatibility_patches, resolve_get_rope_index
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.distributed.utils import FirstRankPerNode
from nemo_automodel.components.models.common.packing import configure_packing
from nemo_automodel.recipes._typed_config import RecipeConfig

logger = logging.getLogger(__name__)


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors nested in a batch to one device.

    Args:
        value: A tensor of arbitrary shape or a nested mapping/list/tuple containing tensors.
        device: Destination device for every tensor.

    Returns:
        The same container topology with independently transferred tensors of unchanged shape.
    """
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _profile_enabled() -> bool:
    return os.environ.get("CANVA_TORCH_PROFILE", "0") == "1"


def _make_profiler(*, warmup_steps: int, rank: int) -> torch.profiler.profile | None:
    if not _profile_enabled() or rank != 0:
        return None
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=warmup_steps, warmup=1, active=3, repeat=1),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    )


def _emit_profile(profiler: torch.profiler.profile, *, framework: str) -> None:
    try:
        operator_table = profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=100)
    except AttributeError:
        operator_table = profiler.key_averages().table(sort_by="self_device_time_total", row_limit=100)
    logger.info("CANVA_OPERATOR_TABLE framework=%s\n%s", framework, operator_table)

    kernels: dict[str, dict[str, float | int]] = {}
    for event in profiler.events():
        if "cuda" not in str(getattr(event, "device_type", "")).lower():
            continue
        duration_us = getattr(event, "self_device_time_total", None)
        if duration_us is None:
            duration_us = getattr(event, "self_cuda_time_total", 0.0)
        entry = kernels.setdefault(event.name, {"calls": 0, "self_cuda_time_us": 0.0})
        entry["calls"] = int(entry["calls"]) + 1
        entry["self_cuda_time_us"] = float(entry["self_cuda_time_us"]) + float(duration_us)

    summary = [
        {"name": name, **values}
        for name, values in sorted(kernels.items(), key=lambda item: float(item[1]["self_cuda_time_us"]), reverse=True)[
            :100
        ]
    ]
    logger.info(
        "CANVA_KERNEL_SUMMARY=%s", json.dumps({"framework": framework, "kernels": summary}, separators=(",", ":"))
    )

    output_dir = pathlib.Path("/opt/nemo-ci/canva_profile")
    output_dir.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(output_dir / f"{framework}_rank0.json"))


def _log_environment(*, framework: str, attention: str, rank: int, world_size: int) -> None:
    if rank != 0:
        return
    device = torch.cuda.current_device()
    payload = {
        "framework": framework,
        "attention": attention,
        "world_size": world_size,
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    logger.info("CANVA_ENV=%s", json.dumps(payload, separators=(",", ":")))


def _run_automodel(cfg: RecipeConfig) -> None:
    from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM

    trainer = FinetuneRecipeForVLM(cfg)
    trainer.setup()
    rank = trainer.dist_env.rank
    world_size = trainer.dist_env.world_size
    attention = str(cfg.get("model.backend.attn", "unknown"))
    framework = f"automodel_{attention}"
    _log_environment(framework=framework, attention=attention, rank=rank, world_size=world_size)

    warmup_steps = int(cfg.get("benchmark_probe.warmup_steps", 5))
    profiler = _make_profiler(warmup_steps=warmup_steps, rank=rank)
    if profiler is None:
        trainer.run_train_validation_loop()
        return

    original_step = trainer._run_train_optim_step

    def profiled_step(batches: list[dict[str, Any]], max_grad_norm: float | None = None) -> Any:
        """Profile one AutoModel optimizer step without changing its result.

        Args:
            batches: Microbatch mappings containing packed ``input_ids`` and ``labels`` tensors of shape
                [batch, sequence] for neat packing or [tokens] for THD packing, plus VLM media tensors.
            max_grad_norm: Optional gradient-norm clipping threshold.

        Returns:
            The original trainer metrics sample.
        """
        result = original_step(batches, max_grad_norm=max_grad_norm)
        profiler.step()
        return result

    trainer._run_train_optim_step = profiled_step
    profiler.__enter__()
    try:
        trainer.run_train_validation_loop()
    finally:
        profiler.__exit__(None, None, None)
    _emit_profile(profiler, framework=framework)


def _accepted_forward_keys(model: torch.nn.Module) -> set[str]:
    return set(inspect.signature(model.forward).parameters)


def _run_accelerate(cfg: RecipeConfig) -> None:
    from accelerate import Accelerator, FullyShardedDataParallelPlugin
    from transformers import AutoModelForImageTextToText

    attention = str(cfg.get("benchmark_probe.attention", "flash_attention_4"))
    accumulation_steps = int(cfg.get("benchmark_probe.gradient_accumulation_steps", 1))
    max_steps = int(cfg.get("step_scheduler.max_steps", 12))
    warmup_steps = int(cfg.get("benchmark_probe.warmup_steps", 5))
    model_name = str(cfg.get("model.pretrained_model_name_or_path"))

    fsdp_plugin = FullyShardedDataParallelPlugin(
        fsdp_version=2,
        reshard_after_forward=True,
        auto_wrap_policy="transformer_based_wrap",
        transformer_cls_names_to_wrap=["Qwen3_5DecoderLayer", "Qwen3_5VisionBlock"],
        mixed_precision_policy={
            "param_dtype": torch.bfloat16,
            "reduce_dtype": torch.bfloat16,
            "output_dtype": torch.bfloat16,
        },
    )
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=accumulation_steps,
        fsdp_plugin=fsdp_plugin,
    )
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    _log_environment(framework="accelerate_fsdp2", attention=attention, rank=rank, world_size=world_size)

    apply_cache_compatibility_patches()
    configure_packing(attn_implementation=attention)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation=attention,
    )
    model.config.use_cache = False
    model.train()
    accepted_forward_keys = _accepted_forward_keys(model)

    dataloader_config = cfg.vlm_dataloader
    if dataloader_config is None:
        raise ValueError("Accelerate comparison requires the same VLM dataloader config as the AutoModel arm")
    dataloader_build = dataloader_config.build(
        pretrained_model_name_or_path=model_name,
        dp_rank=rank,
        dp_world_size=world_size,
        batch_size=int(cfg.get("step_scheduler.local_batch_size", 1)),
        dataset_build_context=FirstRankPerNode(),
        get_rope_index=resolve_get_rope_index(model),
        packing_attn_implementation=attention,
        cp_size=1,
    )
    dataloader = dataloader_build.dataloader

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("optimizer.lr", 1.0e-5)),
        betas=tuple(cfg.get("optimizer.betas", [0.9, 0.95])),
        eps=float(cfg.get("optimizer.eps", 1.0e-8)),
        weight_decay=float(cfg.get("optimizer.weight_decay", 0.1)),
    )
    model, optimizer = accelerator.prepare(model, optimizer)
    profiler = _make_profiler(warmup_steps=warmup_steps, rank=rank)
    if profiler is not None:
        profiler.__enter__()

    data_iter = iter(dataloader)
    optimizer_step = 0
    window_start = time.perf_counter()
    window_tokens = 0
    optimizer.zero_grad(set_to_none=True)
    while optimizer_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = _move_to_device(batch, accelerator.device)
        labels = batch.pop("labels")
        local_tokens = labels.numel()
        window_tokens += local_tokens
        inputs = {key: value for key, value in batch.items() if key in accepted_forward_keys and value is not None}

        with accelerator.accumulate(model):
            output = model(**inputs)
            logits = output.logits
            local_loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]).float(), labels.reshape(-1), ignore_index=-100, reduction="sum"
            )
            local_label_tokens = labels.ne(-100).sum()
            global_label_tokens = accelerator.reduce(local_label_tokens, reduction="sum").clamp_min(1)
            loss = local_loss_sum * world_size * accumulation_steps / global_label_tokens
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if not accelerator.sync_gradients:
            continue

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - window_start
        global_tokens = accelerator.reduce(
            torch.tensor(window_tokens, device=accelerator.device, dtype=torch.long), reduction="sum"
        ).item()
        optimizer_step += 1
        if rank == 0:
            logger.info(
                "CANVA_STEP framework=accelerate_fsdp2 step=%d loss=%.6f tps=%.2f tps_per_gpu=%.2f mem_gib=%.2f",
                optimizer_step,
                float((local_loss_sum / local_label_tokens.clamp_min(1)).detach()),
                global_tokens / elapsed,
                global_tokens / elapsed / world_size,
                torch.cuda.max_memory_allocated() / 1024**3,
            )
        if profiler is not None:
            profiler.step()
        window_start = time.perf_counter()
        window_tokens = 0

    if profiler is not None:
        profiler.__exit__(None, None, None)
        _emit_profile(profiler, framework="accelerate_fsdp2")
    accelerator.wait_for_everyone()


def main(config: str = "examples/vlm_finetune/qwen3_5/qwen3_5_4b_canva_automodel_sdpa.yaml") -> None:
    """Run the benchmark arm selected by ``benchmark_probe.framework``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = RecipeConfig(parse_args_and_load_config(config))
    framework = str(cfg.get("benchmark_probe.framework", "automodel"))
    if framework == "automodel":
        _run_automodel(cfg)
    elif framework == "accelerate":
        _run_accelerate(cfg)
    else:
        raise ValueError(f"Unsupported benchmark_probe.framework={framework!r}")


if __name__ == "__main__":
    main()
