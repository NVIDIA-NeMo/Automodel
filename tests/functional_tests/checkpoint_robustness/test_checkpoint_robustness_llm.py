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

"""Train -> checkpoint -> reload via automodel & vanilla HF from consolidated, verify logits match via KL divergence.

Launch: torchrun --nproc-per-node=<N> -m pytest <this_file> -c <config.yaml>
    [--kl_threshold <float>] [--hf_kl_threshold <float>]
    [--cross_tp_size <int>] [--cross_tp_kl_threshold <float>]
    [--tokenizer_name <str>]
    [--source_load_kl_threshold <float>] [--source_load_mean_kl_threshold <float>]
    [--check_source_load_parity] [--check_fused_qkv_keys] [--check_phantom_keys] [--check_resume]
    [--hf_source_post_load_dequantize]
    [--max_vram_gb <float>] [--max_cpu_gb <float>]
"""

from __future__ import annotations

import gc
import os
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import datasets
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import (
    _MODELS_REQUIRING_BUFFER_REINIT,
    _reinit_non_persistent_buffers,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.recipes.base_recipe import BaseRecipe
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction
from nemo_automodel.shared.utils import dtype_from_str

datasets.disable_caching()

# Llama token IDs for "The quick brown fox jumps over the lazy dog"
_DEFAULT_INPUT_IDS = [791, 4996, 14198, 39935, 35308, 927, 279, 16053, 5679]
_DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog"


def _extract_custom_args(argv):
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {
        "--kl_threshold",
        "--hf_kl_threshold",
        "--cross_tp_size",
        "--cross_tp_kl_threshold",
        "--experts_implementation",
        "--tokenizer_name",
        "--max_vram_gb",
        "--max_cpu_gb",
        "--resume_loss_threshold",
        "--source_load_cosine_threshold",
        "--source_load_kl_threshold",
        "--source_load_mean_kl_threshold",
    }
    boolean_keys = {
        "--trust_remote_code",
        "--check_source_load_parity",
        "--check_fused_qkv_keys",
        "--check_phantom_keys",
        "--check_resume",
        "--hf_device_map_auto",
        "--hf_source_post_load_dequantize",
        "--skip_hf_reload",
    }
    custom = {}
    remaining = []
    i = 0
    while i < len(argv):
        if argv[i] in custom_keys:
            custom[argv[i].lstrip("-")] = argv[i + 1]
            i += 2
        elif argv[i] in boolean_keys:
            custom[argv[i].lstrip("-")] = True
            i += 1
        else:
            remaining.append(argv[i])
            i += 1

    # Read ci.checkpoint_robustness from the YAML config as defaults.
    # CLI args take precedence over YAML values.
    config_path = None
    for j, arg in enumerate(remaining):
        if arg == "--config" and j + 1 < len(remaining):
            config_path = remaining[j + 1]
            break
    if config_path:
        import yaml

        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f)
        ci_robustness = raw_cfg.get("ci", {}).get("checkpoint_robustness") or {}
        no_check_resume = ci_robustness.pop("no_check_resume", False)
        for k, v in ci_robustness.items():
            if k not in custom:
                if "." in k:
                    # Dotted keys are config overrides (e.g. distributed.tp_size),
                    # route them to the config parser instead of the custom dict.
                    remaining.extend([f"--{k}", str(v)])
                elif isinstance(v, bool) and v:
                    custom[k] = True
                elif not isinstance(v, bool):
                    custom[k] = str(v)
        # Enable check_resume by default unless no_check_resume is set
        if not no_check_resume and "check_resume" not in custom:
            custom["check_resume"] = True

    return custom, remaining


def _get_input_ids(tokenizer_name: str | None) -> list[int]:
    """Return input IDs for the test prompt, using dynamic tokenization if tokenizer_name is set."""
    if tokenizer_name is None:
        return _DEFAULT_INPUT_IDS
    from nemo_automodel import NeMoAutoTokenizer

    tokenizer = NeMoAutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    )
    return tokenizer.encode(_DEFAULT_PROMPT, add_special_tokens=False)


def _load_hf_fp8_dequantized_config(
    pretrained_model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
    revision: str | None = None,
    token: str | bool | None = None,
):
    """Return an HF config that dequantizes a fine-grained FP8 checkpoint, if applicable."""
    from transformers import AutoConfig

    config_kwargs: dict[str, str | bool] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
    quantization_config = getattr(config, "quantization_config", None)
    if isinstance(quantization_config, dict):
        quant_method = quantization_config.get("quant_method")
    else:
        quant_method = getattr(quantization_config, "quant_method", None)
    if getattr(quant_method, "value", quant_method) != "fp8":
        return None

    if isinstance(quantization_config, dict):
        config.quantization_config = {**quantization_config, "dequantize": True}
    else:
        quantization_config.dequantize = True
    return config


def _dequantize_hf_fp8_weights_in_place(model, output_dtype: torch.dtype) -> int:
    """Dequantize native per-tensor HF FP8 modules without their runtime kernel.

    Some MoE checkpoints cannot use Transformers' load-time ``dequantize=True``
    conversion, while their native FP8 modules require the optional ``kernels``
    package at forward time. Load those modules with their native weight/scale
    layout first, then replace only the FP8 weight parameters with dequantized
    tensors. ``FP8Linear`` dispatches to its ordinary PyTorch path once a weight
    uses more than one byte per element. ``FP8Experts`` also needs its configured
    experts implementation reset to ``eager`` because its wrapper selects the
    grouped FP8 kernel independently of the weight dtype. This helper intentionally
    accepts only the scalar and per-expert scalar scale layouts used by the
    Mistral4 checkpoint; block-wise layouts should use Transformers' normal
    load-time conversion.
    """
    parameter_pairs = (
        ("weight", "weight_scale_inv"),
        ("gate_up_proj", "gate_up_proj_scale_inv"),
        ("up_proj", "up_proj_scale_inv"),
        ("down_proj", "down_proj_scale_inv"),
    )
    converted = 0
    converted_expert_weights = False
    for module in model.modules():
        for weight_name, scale_name in parameter_pairs:
            weight = getattr(module, weight_name, None)
            scale = getattr(module, scale_name, None)
            if not isinstance(weight, torch.Tensor) or not isinstance(scale, torch.Tensor):
                continue
            if weight.element_size() > 1:
                continue
            scale = scale.squeeze()
            if scale.numel() == 1:
                broadcast_scale = scale
            elif weight.ndim == 3 and scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
                broadcast_scale = scale.view(-1, 1, 1)
            else:
                raise ValueError(
                    f"Unsupported post-load FP8 scale layout for {type(module).__name__}.{weight_name}: "
                    f"weight={tuple(weight.shape)}, scale={tuple(scale.shape)}"
                )
            dequantized = (weight.float() * broadcast_scale.float()).to(output_dtype)
            setattr(
                module,
                weight_name,
                torch.nn.Parameter(dequantized, requires_grad=bool(getattr(weight, "requires_grad", False))),
            )
            converted += 1
            converted_expert_weights |= weight.ndim == 3

    assert converted > 0, "Post-load HF FP8 dequantization requested, but no FP8 weight/scale pairs were found"
    if converted_expert_weights:
        model.set_experts_implementation("eager")
    return converted


def _post_load_dequant_max_memory() -> dict[int, int]:
    """Reserve enough automatic-device-map headroom for FP8-to-BF16 expansion."""
    return {
        index: int(torch.cuda.get_device_properties(index).total_memory * 0.35)
        for index in range(torch.cuda.device_count())
    }


def _rss_gb() -> float:
    """Current RSS in GB from /proc/self/statm."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * page_size / 1024**3


def _kl_divergence_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> torch.Tensor:
    """Per-token KL(reference || candidate) for full [B, T, V] logits."""
    assert reference_logits.shape == candidate_logits.shape
    vocab_size = reference_logits.shape[-1]
    ref_log_probs = F.log_softmax(reference_logits.float(), dim=-1).reshape(-1, vocab_size)
    cand_log_probs = F.log_softmax(candidate_logits.float(), dim=-1).reshape(-1, vocab_size)
    return F.kl_div(cand_log_probs, ref_log_probs, reduction="none", log_target=True).sum(-1)


def _cosine_similarity_from_logits(reference_logits: torch.Tensor, candidate_logits: torch.Tensor) -> float:
    """Cosine similarity over flattened float32 logits."""
    return F.cosine_similarity(reference_logits.flatten().float(), candidate_logits.flatten().float(), dim=0).item()


def _materialize_config_value(value):
    """Convert a config value into the object that recipe instantiation would pass as a kwarg."""
    if isinstance(value, ConfigNode):
        if hasattr(value, "_target_"):
            return value.instantiate()
        return {
            k: _materialize_config_value(v)
            for k, v in value.__dict__.items()
            if k not in ("raise_on_missing_attr", "_raw_config", "_original_strings")
        }
    if isinstance(value, list):
        return [_materialize_config_value(v) for v in value]
    return value


def _model_kwargs_from_config(model_cfg: ConfigNode) -> dict:
    """Return kwargs from the recipe model config without invoking the model target."""
    return {
        k: _materialize_config_value(v)
        for k, v in model_cfg.__dict__.items()
        if k not in ("_target_", "raise_on_missing_attr", "_raw_config", "_original_strings")
    }


def _resolve_source_load_dtype(model_kwargs: dict) -> torch.dtype:
    """Mirror NeMoAuto's practical source-load dtype default for the HF reference model."""
    torch_dtype = model_kwargs.get("torch_dtype", "auto")
    if torch_dtype == "auto":
        return torch.bfloat16
    if isinstance(torch_dtype, str):
        return dtype_from_str(torch_dtype)
    return torch_dtype


def _get_trust_remote_code_attn_implementation(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str:
    """Select the vanilla-HF attention implementation for a remote-code model."""
    from transformers import AutoConfig

    config_kwargs: dict[str, str | bool] = {"trust_remote_code": True}
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

    # Nemotron-H remote-code checkpoints do not share optimized attention backend
    # support: FlashAttention fails in Nemotron-3 Super's varlen path, while
    # Nemotron-Nano v2 declares SDPA unsupported. Eager is their common HF
    # reference path. Other remote-code models (notably Nemotron-Flash) still
    # require FA2.
    return "eager" if config.model_type == "nemotron_h" else "flash_attention_2"


def _hf_source_load_kwargs(
    model_kwargs: dict,
    *,
    pretrained_model_name_or_path: str | Path,
    source_dtype: torch.dtype,
    trust_remote_code: bool,
    experts_implementation: str | None,
    device: torch.device,
    hf_device_map_auto: bool,
) -> dict:
    """Build the HF-safe subset of recipe model kwargs for the source-load reference."""
    hf_allowed_keys = {
        "attn_implementation",
        "config",
        "quantization_config",
        "revision",
        "token",
        "trust_remote_code",
    }
    hf_kwargs = {k: v for k, v in model_kwargs.items() if k in hf_allowed_keys}
    hf_kwargs["torch_dtype"] = source_dtype
    hf_kwargs["trust_remote_code"] = trust_remote_code or bool(hf_kwargs.get("trust_remote_code", False))
    hf_kwargs["local_files_only"] = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    if hf_kwargs["trust_remote_code"] and "attn_implementation" not in hf_kwargs:
        hf_kwargs["attn_implementation"] = _get_trust_remote_code_attn_implementation(
            pretrained_model_name_or_path,
            revision=hf_kwargs.get("revision"),
            token=hf_kwargs.get("token"),
        )
    if experts_implementation and not trust_remote_code:
        hf_kwargs["experts_implementation"] = experts_implementation
        hf_kwargs["trust_remote_code"] = False
    if hf_device_map_auto:
        hf_kwargs["device_map"] = "auto"
    if (
        "device_map" not in hf_kwargs
        and not hf_kwargs["trust_remote_code"]
        and hf_kwargs.get("quantization_config") is None
    ):
        hf_kwargs["device_map"] = {"": device}
    return hf_kwargs


def _lm_head_embedding_aliased(model) -> bool | None:
    """Return lm_head/input-embedding aliasing when real local storage is inspectable."""
    # FSDP2/TP wrappers may expose distinct local storages for logically tied
    # parameters, so only use this as a real storage check before sharding.
    if dist.is_initialized() and dist.get_world_size() > 1:
        return None
    lm_head = getattr(model, "lm_head", None)
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if lm_head is None or get_input_embeddings is None:
        return None
    embeddings = get_input_embeddings()
    if embeddings is None or not hasattr(lm_head, "weight") or not hasattr(embeddings, "weight"):
        return None
    lm_head_weight = lm_head.weight
    embedding_weight = embeddings.weight
    if isinstance(lm_head_weight, DTensor) or isinstance(embedding_weight, DTensor):
        return None
    try:
        lm_head_ptr = lm_head_weight.data_ptr()
        embedding_ptr = embedding_weight.data_ptr()
    except RuntimeError:
        return None
    if lm_head_ptr == 0 or embedding_ptr == 0:
        return None
    return lm_head_ptr == embedding_ptr


def _explicit_tie_word_embeddings(config) -> bool | None:
    """Return an explicit tie_word_embeddings flag from a top-level or text config."""
    tie_word_embeddings = getattr(config, "tie_word_embeddings", None)
    if tie_word_embeddings is not None:
        return bool(tie_word_embeddings)
    text_config = getattr(config, "text_config", None)
    tie_word_embeddings = getattr(text_config, "tie_word_embeddings", None)
    return None if tie_word_embeddings is None else bool(tie_word_embeddings)


def _release_model_memory() -> None:
    """Release standalone model memory between source-load parity phases."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _preinit_global_rank() -> int:
    """Return the torchrun global rank before torch.distributed is initialized."""
    if dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def _preinit_world_size() -> int:
    """Return the torchrun world size before torch.distributed is initialized."""
    if dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def _sanitize_sync_id(value: str) -> str:
    """Return a filesystem-friendly sync identifier."""
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _source_load_run_id() -> str:
    """Return a launch-scoped ID shared by ranks for pre-init file sync."""
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if slurm_job_id:
        slurm_step_id = os.environ.get("SLURM_STEP_ID", "step")
        slurm_restart_count = os.environ.get("SLURM_RESTART_COUNT", "0")
        return _sanitize_sync_id(f"slurm_{slurm_job_id}_{slurm_step_id}_{slurm_restart_count}")

    torch_run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if torch_run_id and torch_run_id.lower() not in ("local", "none", "default"):
        restart_count = os.environ.get("TORCHELASTIC_RESTART_COUNT", "0")
        return _sanitize_sync_id(f"torchelastic_{torch_run_id}_{restart_count}")

    master_port = os.environ.get("MASTER_PORT", "unknown")
    world_size = os.environ.get("WORLD_SIZE", "1")
    # Local fallback is intended for single-node torchrun/debug runs. Multi-node
    # non-SLURM launches should provide a meaningful TORCHELASTIC_RUN_ID so all
    # nodes agree on the same marker path.
    return _sanitize_sync_id(f"local_ppid_{os.getppid()}_port_{master_port}_world_{world_size}")


def _source_load_sync_paths(cfg) -> tuple[Path, Path, Path]:
    """Return sync directory and done/fail paths for pre-init source-load parity."""
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    sync_dir = checkpoint_dir.parent / f".source_load_parity_{_source_load_run_id()}"
    return sync_dir, sync_dir / "done", sync_dir / "fail"


def _wait_for_source_load_rank0(done_path: Path, fail_path: Path) -> None:
    """Wait for rank 0 to finish source-load parity before process-group init."""
    timeout_s = int(os.environ.get("SOURCE_LOAD_PARITY_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if done_path.exists():
            return
        if fail_path.exists():
            raise RuntimeError(f"Rank 0 source-load parity failed:\n{fail_path.read_text()}")
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 source-load parity")


def _cleanup_source_load_sync(cfg) -> None:
    """Best-effort cleanup of pre-init source-load sync markers."""
    sync_dir, done_path, fail_path = _source_load_sync_paths(cfg)
    if not sync_dir.exists():
        return
    for path in (done_path, fail_path):
        path.unlink(missing_ok=True)
    try:
        sync_dir.rmdir()
    except OSError:
        pass


def _hf_reload_sync_paths(cfg) -> tuple[Path, Path]:
    """Return sync directory and done path for the rank-0-only HF reload."""
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    sync_dir = checkpoint_dir.parent / f".hf_reload_{_source_load_run_id()}"
    return sync_dir, sync_dir / "done"


def _wait_for_hf_reload_rank0(done_path: Path) -> None:
    """Wait without an active collective for rank 0 to finish the vanilla-HF reload."""
    timeout_s = int(os.environ.get("HF_RELOAD_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if done_path.exists():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 vanilla-HF reload")


def _prepare_hf_reload_sync(cfg) -> tuple[Path, Path] | None:
    """Prepare ranks for a long rank-0-only HF reload without starting an NCCL wait."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None

    sync_dir, done_path = _hf_reload_sync_paths(cfg)
    if _rank0():
        sync_dir.mkdir(parents=True, exist_ok=True)
        done_path.unlink(missing_ok=True)
    _barrier()  # ensure all ranks released recipe memory and rank 0 reset the marker
    if not _rank0():
        _wait_for_hf_reload_rank0(done_path)
    return sync_dir, done_path


def _finish_hf_reload_sync(
    sync_paths: tuple[Path, Path] | None,
    error_message: str | None = None,
) -> str | None:
    """Release waiting ranks and propagate a rank-0 HF parity failure."""
    if sync_paths is None:
        return error_message

    sync_dir, done_path = sync_paths
    if _rank0():
        status = "ok\n" if error_message is None else f"error\n{error_message}"
        done_path.write_text(status)
    _barrier()
    status = done_path.read_text()
    _barrier()
    if _rank0():
        done_path.unlink(missing_ok=True)
        try:
            sync_dir.rmdir()
        except OSError:
            pass
    if status.startswith("error\n"):
        return status.removeprefix("error\n")
    return None


def _prepare_source_load_reference(
    cfg,
    input_ids: list[int],
    *,
    hf_model_cls: type,
    trust_remote_code: bool,
    experts_implementation: str | None,
    hf_device_map_auto: bool,
    hf_source_post_load_dequantize: bool,
) -> tuple[torch.Tensor, bool | None, bool | None] | None:
    """Compute vanilla HF source-load reference logits before trainer construction."""
    if _preinit_world_size() > 1:
        sync_dir, done_path, fail_path = _source_load_sync_paths(cfg)
        if _preinit_global_rank() != 0:
            _wait_for_source_load_rank0(done_path, fail_path)
            return None
        sync_dir.mkdir(parents=True, exist_ok=True)
        done_path.unlink(missing_ok=True)
        fail_path.unlink(missing_ok=True)
    else:
        done_path = None
        fail_path = None

    if _preinit_global_rank() != 0:
        return None

    try:
        result = _prepare_source_load_reference_rank0(
            cfg,
            input_ids,
            hf_model_cls=hf_model_cls,
            trust_remote_code=trust_remote_code,
            experts_implementation=experts_implementation,
            hf_device_map_auto=hf_device_map_auto,
            hf_source_post_load_dequantize=hf_source_post_load_dequantize,
        )
    except Exception:
        if fail_path is not None:
            fail_path.write_text(traceback.format_exc())
        raise
    else:
        if done_path is not None:
            done_path.write_text("ok\n")
        return result


def _prepare_source_load_reference_rank0(
    cfg,
    input_ids: list[int],
    *,
    hf_model_cls: type,
    trust_remote_code: bool,
    experts_implementation: str | None,
    hf_device_map_auto: bool,
    hf_source_post_load_dequantize: bool,
) -> tuple[torch.Tensor, bool | None, bool | None]:
    """Rank-0 implementation of vanilla HF source-load reference capture."""
    from contextlib import nullcontext

    from nemo_automodel._transformers.utils import apply_cache_compatibility_patches

    apply_cache_compatibility_patches()

    model_kwargs = _model_kwargs_from_config(cfg.model)
    original_pretrained_path = model_kwargs.get("pretrained_model_name_or_path")
    assert original_pretrained_path is not None, "source-load parity requires model.pretrained_model_name_or_path"
    source_dtype = _resolve_source_load_dtype(model_kwargs)
    trust_remote_code = trust_remote_code or bool(model_kwargs.get("trust_remote_code", False))

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    hf_kwargs = _hf_source_load_kwargs(
        model_kwargs,
        pretrained_model_name_or_path=original_pretrained_path,
        source_dtype=source_dtype,
        trust_remote_code=trust_remote_code,
        experts_implementation=experts_implementation,
        device=device,
        hf_device_map_auto=hf_device_map_auto,
    )
    if hf_source_post_load_dequantize and hf_kwargs.get("device_map") == "auto" and torch.cuda.is_available():
        # Accelerate sizes the automatic map for the on-disk FP8 tensors. The
        # post-load BF16 representation needs roughly twice that memory, so cap
        # each GPU's FP8 placement at 35% and retain headroom for the forward.
        hf_kwargs["max_memory"] = _post_load_dequant_max_memory()
    # Dense FP8 checkpoints use Transformers' load-time dequantization by
    # default. Some MoE checkpoints need to retain their native weight/scale
    # layout during loading and are dequantized immediately afterwards.
    if (
        not hf_source_post_load_dequantize
        and "config" not in hf_kwargs
        and hf_kwargs.get("quantization_config") is None
    ):
        fp8_config = _load_hf_fp8_dequantized_config(
            original_pretrained_path,
            trust_remote_code=hf_kwargs["trust_remote_code"],
            revision=hf_kwargs.get("revision"),
            token=hf_kwargs.get("token"),
        )
        if fp8_config is not None:
            hf_kwargs["config"] = fp8_config

    try:
        from nemo_automodel._transformers.model_init import no_hf_meta_device

        no_meta = no_hf_meta_device() if trust_remote_code else nullcontext()
    except ImportError:
        no_meta = nullcontext()

    print(f"\n[Phase 0] Source-load reference: vanilla HF for {original_pretrained_path}")
    with no_meta:
        if "device_map" in hf_kwargs:
            hf_model = hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
        else:
            hf_model = _fix_meta_rotary_embeddings(
                hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
            ).to(device)
    if hf_source_post_load_dequantize:
        converted = _dequantize_hf_fp8_weights_in_place(hf_model, source_dtype)
        print(f"[Phase 0] Post-load dequantized {converted} HF FP8 weight tensors to {source_dtype}.")
    _reinit_rotary_per_module(hf_model, device)
    if trust_remote_code:
        from nemo_automodel._transformers.v4_patches.rotary import fix_rotary_embeddings, should_fix_rotary_embeddings

        if should_fix_rotary_embeddings([hf_model]):
            fix_rotary_embeddings([hf_model])

    hf_logits = _get_logits(hf_model, input_ids, device)
    hf_aliased = _lm_head_embedding_aliased(hf_model)
    explicit_tie_word_embeddings = _explicit_tie_word_embeddings(hf_model.config)
    del hf_model
    _release_model_memory()
    return hf_logits, hf_aliased, explicit_tie_word_embeddings


def _compare_source_load_parity(
    source_reference: tuple[torch.Tensor, bool | None, bool | None] | None,
    candidate_logits: torch.Tensor,
    candidate_model,
    *,
    source_load_kl_threshold: float,
    source_load_mean_kl_threshold: float,
    source_load_cosine_threshold: float,
) -> str | None:
    """Compare the vanilla HF source-load reference against the constructed trainer model.

    Args:
        source_reference: Rank-0 tuple containing logits of shape [batch, sequence, vocab], the HF input/output
            embedding alias state, and the explicit tie-word-embeddings setting. Other ranks pass ``None``.
        candidate_logits: Constructed trainer logits of shape [batch, sequence, vocab].
        candidate_model: Constructed trainer model used to inspect input/output embedding aliasing.
        source_load_kl_threshold: Maximum allowed per-token KL divergence.
        source_load_mean_kl_threshold: Maximum allowed mean per-token KL divergence.
        source_load_cosine_threshold: Minimum allowed cosine similarity over flattened logits.

    Returns:
        Synchronized failure traceback when source-load parity fails, otherwise ``None``. The caller may defer this
        failure until independent checkpoint reload and resume phases have completed.
    """
    candidate_aliased = _lm_head_embedding_aliased(candidate_model)
    failure_message = None
    if _rank0():
        try:
            assert source_reference is not None, "rank 0 source-load reference was not captured"
            hf_logits, hf_aliased, explicit_tie_word_embeddings = source_reference
            assert hf_logits.shape == candidate_logits.shape, (
                f"Source-load parity shape mismatch: HF logits {hf_logits.shape} vs trainer logits "
                f"{candidate_logits.shape}"
            )
            kl_source = _kl_divergence_from_logits(hf_logits, candidate_logits)
            max_kl_source = kl_source.max().item()
            mean_kl_source = kl_source.mean().item()
            p95_kl_source = torch.quantile(kl_source, 0.95).item()
            cosine_source = _cosine_similarity_from_logits(hf_logits, candidate_logits)
            print(
                f"[Phase 0] Source-load vs constructed-trainer max KL: {max_kl_source:.6e} "
                f"(threshold: {source_load_kl_threshold:.6e}); mean KL: {mean_kl_source:.6e} "
                f"(threshold: {source_load_mean_kl_threshold:.6e}); p95 KL: {p95_kl_source:.6e}; "
                f"cosine={cosine_source:.8f} "
                f"(threshold: {source_load_cosine_threshold:.8f}); "
                f"hf_aliased={hf_aliased}; trainer_aliased={candidate_aliased}; "
                f"tie_word_embeddings={explicit_tie_word_embeddings}"
            )

            assert max_kl_source <= source_load_kl_threshold, (
                f"KL divergence between original HF source load and constructed trainer model too large: "
                f"max per-token KL = {max_kl_source:.6e} > threshold {source_load_kl_threshold:.6e}"
            )
            assert mean_kl_source <= source_load_mean_kl_threshold, (
                f"Mean KL divergence between original HF source load and constructed trainer model too large: "
                f"mean per-token KL = {mean_kl_source:.6e} > threshold {source_load_mean_kl_threshold:.6e}"
            )
            assert cosine_source >= source_load_cosine_threshold, (
                f"Cosine similarity between original HF source load and constructed trainer model too low: "
                f"cosine = {cosine_source:.8f} < threshold {source_load_cosine_threshold:.8f}"
            )
            if hf_aliased is not None and candidate_aliased is not None:
                assert hf_aliased == candidate_aliased, (
                    f"Source-load lm_head aliasing mismatch: HF aliased={hf_aliased}, "
                    f"trainer aliased={candidate_aliased}"
                )
            if explicit_tie_word_embeddings is not None and candidate_aliased is not None:
                assert candidate_aliased == explicit_tie_word_embeddings, (
                    f"Constructed trainer lm_head aliasing does not match config.tie_word_embeddings="
                    f"{explicit_tie_word_embeddings}: aliased={candidate_aliased}"
                )
        except Exception:
            failure_message = traceback.format_exc()

    # Keep every rank on the same control-flow path when rank 0 detects a Phase 0
    # mismatch. The caller records the failure and continues with the independent
    # checkpoint reload and resume phases.
    if dist.is_initialized():
        payload = [failure_message]
        dist.broadcast_object_list(payload, src=0)
        failure_message = payload[0]
    if failure_message is not None and _rank0():
        print("[Phase 0] Source-load parity failed; deferring failure until later checkpoint phases complete.")
    return failure_message


def _get_logits_pp(trainer, input_ids, device) -> torch.Tensor:
    """Run forward through the PP schedule and return logits on every rank.

    The raw ``model_parts[0].forward`` can't be called directly on non-first PP
    stages (they expect float hidden states, not int token IDs). Mirror the
    KD recipe's trick: swap the schedule's loss_fn for a capture closure, run
    ``schedule.eval`` on the first stage, then broadcast the captured last-stage
    logits along the PP group.
    """
    schedule = trainer.pp.info.schedule
    pp_batch_size = trainer.pipeline_config.pp_batch_size
    orig_seq_len = len(input_ids)

    # PP recv buffer shapes are locked at first forward. r0.4.0 lacks
    # AutoPipeline.update_seq_len (added in #1689) to resize on the fly, so
    # discover the locked seq_len from the stages and pad input_ids to match
    # for the forward pass. Captured logits are sliced back to orig_seq_len.
    def _discover_pp_seq_len() -> int:
        pp_seq_len = getattr(trainer.pp, "pp_seq_len", None)
        if pp_seq_len:
            return pp_seq_len
        for stage in getattr(trainer.pp.info, "stages", None) or ():
            for meta in getattr(stage, "inputs_meta", None) or ():
                if meta.ndim >= 2 and meta.shape[1] > 0:
                    return meta.shape[1]
        ds_seq_length = trainer.cfg.get("dataset.seq_length", None)
        return ds_seq_length or orig_seq_len

    pp_seq_len = _discover_pp_seq_len()
    if orig_seq_len < pp_seq_len:
        input_ids = list(input_ids) + [0] * (pp_seq_len - orig_seq_len)

    # Replicate the prompt to pp_batch_size so the schedule's batch split is valid.
    ids = torch.tensor([input_ids] * pp_batch_size, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(ids)
    targets = torch.zeros_like(ids) if trainer.pp.info.has_last_stage else None

    captured = [None]

    def _capture_loss_fn(logits, target, **_):
        captured[0] = logits.detach().float().clone()
        return logits.new_tensor(0.0, dtype=logits.dtype)

    saved_loss_fn = schedule._loss_fn
    schedule._loss_fn = _capture_loss_fn
    try:
        for m in trainer.model_parts:
            m.eval()
        # Use no_grad rather than inference_mode: FSDP2's wait_for_unshard reads
        # tensor._version on unsharded params, which is not available for
        # inference-mode tensors ("Inference tensors do not track version counter").
        with torch.no_grad():
            losses = [] if trainer.pp.info.has_last_stage else None
            if trainer.pp.info.has_first_stage:
                schedule.eval(ids, target=targets, losses=losses, attention_mask=attention_mask)
            else:
                schedule.eval(target=targets, losses=losses, attention_mask=attention_mask)
    finally:
        schedule._loss_fn = saved_loss_fn

    config = trainer.model_parts[0].config
    vocab_size = getattr(config, "vocab_size", None)
    if vocab_size is None:
        vocab_size = getattr(getattr(config, "text_config", None), "vocab_size", None)
    assert vocab_size is not None, "could not resolve vocab_size from model config"

    buf = torch.zeros((1, orig_seq_len, vocab_size), device=device, dtype=torch.float32)
    if trainer.pp.info.has_last_stage and captured[0] is not None:
        buf.copy_(captured[0][:1, :orig_seq_len, :])

    pp_mesh = trainer.device_mesh["pp"]
    pp_group = pp_mesh.get_group()
    src = dist.get_global_rank(pp_group, pp_mesh.size() - 1)
    dist.broadcast(buf, src=src, group=pp_group)

    return buf.cpu()


def _get_logits(model, input_ids, device, trainer=None) -> torch.Tensor:
    """Forward pass returning float32 logits on CPU."""
    if trainer is not None and getattr(trainer, "pp_enabled", False):
        return _get_logits_pp(trainer, input_ids, device)

    model.eval()
    ids = torch.tensor([input_ids], device=device)
    attention_mask = torch.ones_like(ids)
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=attention_mask, use_cache=False)
        logits = out.logits if hasattr(out, "logits") else out
        if isinstance(logits, DTensor):
            logits = logits.full_tensor()
        return logits.float().cpu()


def _reinit_rotary_per_module(model, default_device):
    """Recompute DeciLM / Gemma3 style non-persistent rotary buffers on each
    module's own device.

    HF `from_pretrained` in transformers 5.x leaves ``inv_freq`` uninitialized
    for models whose rotary buffers are computed in ``__init__`` and never
    saved to the state dict (e.g. nemotron-nas, gemma3). With
    ``device_map='auto'`` each rotary module can live on a different GPU, so
    we drive the recompute per-module using its own inv_freq device rather
    than a single fixed device.
    """
    model_type = getattr(model.config, "model_type", None)
    if model_type not in _MODELS_REQUIRING_BUFFER_REINIT:
        return model
    for mod in model.modules():
        inv = getattr(mod, "inv_freq", None)
        if inv is None:
            continue
        mod_device = inv.device
        if mod_device.type == "meta":
            mod_device = next((p.device for p in mod.parameters()), default_device)
        _reinit_non_persistent_buffers(mod, mod_device, model_type=model_type)
    return model


def _fix_meta_rotary_embeddings(model):
    """Re-materialize RotaryEmbedding tensors stuck on meta device.

    The HF remote Baichuan code creates inv_freq/cos_cached/sin_cached as
    plain tensor attributes (not registered buffers), so HF's meta-device
    init never materializes them.
    """
    for _name, mod in model.named_modules():
        if hasattr(mod, "inv_freq") and mod.inv_freq.device.type == "meta":
            dim = mod.inv_freq.shape[0] * 2
            mod.inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
            max_pos = mod.max_seq_len_cached
            t = torch.arange(max_pos, dtype=torch.float32)
            freqs = torch.outer(t, mod.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            mod.cos_cached = emb.cos()[None, None, :, :].to(torch.float32)
            mod.sin_cached = emb.sin()[None, None, :, :].to(torch.float32)
    return model


def _prepopulate_hf_dynamic_modules_cache(local_dir: Path | str) -> None:
    """Copy every ``.py`` from ``local_dir`` into HF's dynamic-modules cache.

    Works around a transformers<=5.5.x bug in the local-dir branch of
    ``dynamic_module_utils.get_cached_module_file``: it only copies the
    modeling file's *direct* relative imports into
    ``HF_MODULES_CACHE/transformers_modules/<submodule>/``. Transitive
    imports (e.g. ``fused_mha_with_cache.py`` imports ``.triton_attention``)
    are later discovered by ``get_relative_import_files`` at module-load
    time and fail with ``FileNotFoundError`` because they never got copied.

    Pre-seeding the cache dir with all ``.py`` files from the consolidated
    dir makes the filecmp-gated copies no-ops and ensures every transitive
    import is resolvable.
    """
    import shutil

    try:
        from transformers.dynamic_module_utils import (
            HF_MODULES_CACHE,
            TRANSFORMERS_DYNAMIC_MODULE_NAME,
            _sanitize_module_name,
        )
    except ImportError:
        return

    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        return
    submodule = _sanitize_module_name(local_dir.name)
    dst = Path(HF_MODULES_CACHE) / TRANSFORMERS_DYNAMIC_MODULE_NAME / submodule
    dst.mkdir(parents=True, exist_ok=True)
    for src_py in local_dir.rglob("*.py"):
        if src_py.name == "__init__.py":
            continue
        rel = src_py.relative_to(local_dir)
        dst_py = dst / rel
        dst_py.parent.mkdir(parents=True, exist_ok=True)
        if not dst_py.exists():
            shutil.copy2(src_py, dst_py)


def _tp_size_from_argv(argv) -> int:
    """Peek at --distributed.tp_size / --config YAML without constructing the cfg.

    Returns 1 if no TP setting is found. Used before cfg parsing to pick a
    reasonable default kl_threshold.
    """
    for i, a in enumerate(argv):
        if a == "--distributed.tp_size" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except (TypeError, ValueError):
                return 1
    config_path = None
    for i, a in enumerate(argv):
        if a == "--config" and i + 1 < len(argv):
            config_path = argv[i + 1]
            break
    if config_path:
        try:
            import yaml

            with open(config_path) as f:
                raw_cfg = yaml.safe_load(f) or {}
            tp = (raw_cfg.get("distributed") or {}).get("tp_size", 1)
            return int(tp) if tp is not None else 1
        except Exception:
            pass
    return 1


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _barrier():
    if dist.is_initialized():
        dist.barrier()


def _release_recipe_memory(recipe) -> None:
    """Release a recipe's GPU-resident state between checkpoint-robustness phases.

    Each phase builds a full FSDP2 model + optimizer. A bare ``del`` is not
    enough: the per-part optimizers are reachable from the model (they are built
    over ``model.parts``), so the optimizer state (Adam moments are the bulk)
    lingers. Clear the optimizer state in place and drop the recipe's references,
    then collect — letting the prior phase's model + optimizer be reclaimed
    before the next phase allocates its own, keeping the inter-phase baseline low.
    """
    if recipe is None:
        return
    optimizers = getattr(recipe, "optimizer", None)
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers] if optimizers is not None else []
    for opt in optimizers:
        try:
            opt.state.clear()
            opt.param_groups.clear()
        except Exception:
            pass
    recipe.model_parts = None
    recipe.optimizer = None
    if getattr(recipe, "lr_scheduler", None) is not None:
        recipe.lr_scheduler = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_checkpoint_robustness(
    *,
    recipe_cls: type[BaseRecipe],
    hf_model_cls: type,
    input_ids_loader: Callable[[str | None], list[int]] = _get_input_ids,
) -> None:
    """Run checkpoint robustness for one recipe and Hugging Face auto-model class.

    Args:
        recipe_cls: Recipe class used for training, checkpoint reload, and resume phases.
        hf_model_cls: Hugging Face auto-model class used for source and consolidated loads.
        input_ids_loader: Domain-specific tokenizer used to encode the parity prompt.
    """
    custom_args, config_argv = _extract_custom_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + config_argv
    # When tensor parallelism is active the forward pass uses row-parallel
    # all-reduces and cuBLASLt plan caches whose order of accumulation is
    # process-dependent; this produces ULP-level bf16 drift between the
    # trainer's and restored model's logits even with bit-identical weights.
    # Use a small tolerance when TP>1; keep strict 0 otherwise so real
    # save/load regressions in non-TP setups still fail.
    _tp_size = _tp_size_from_argv(config_argv)
    _default_kl_threshold = "1e-5" if _tp_size > 1 else "0"
    kl_threshold = float(custom_args.get("kl_threshold", _default_kl_threshold))
    hf_kl_threshold = float(custom_args.get("hf_kl_threshold", "5e-3"))
    cross_tp_size = int(custom_args.get("cross_tp_size", "0"))
    cross_tp_kl_threshold = float(custom_args.get("cross_tp_kl_threshold", "5e-3"))
    trust_remote_code = bool(custom_args.get("trust_remote_code", False))
    experts_implementation = custom_args.get("experts_implementation", None)
    tokenizer_name = custom_args.get("tokenizer_name", None)
    max_vram_gb = float(custom_args.get("max_vram_gb", "0"))
    max_cpu_gb = float(custom_args.get("max_cpu_gb", "0"))
    check_fused_qkv_keys = bool(custom_args.get("check_fused_qkv_keys", False))
    check_phantom_keys = bool(custom_args.get("check_phantom_keys", False))
    check_resume = bool(custom_args.get("check_resume", False))
    resume_loss_threshold = float(custom_args.get("resume_loss_threshold", "5e-3"))
    hf_device_map_auto = bool(custom_args.get("hf_device_map_auto", False))
    hf_source_post_load_dequantize = bool(custom_args.get("hf_source_post_load_dequantize", False))
    skip_hf_reload = bool(custom_args.get("skip_hf_reload", False))
    check_source_load_parity = bool(custom_args.get("check_source_load_parity", False))
    source_load_kl_threshold = float(custom_args.get("source_load_kl_threshold", "5e-3"))
    source_load_mean_kl_threshold = float(custom_args.get("source_load_mean_kl_threshold", "1e-3"))
    source_load_cosine_threshold = float(custom_args.get("source_load_cosine_threshold", "0.9999"))
    deferred_failures: list[str] = []

    input_ids = input_ids_loader(tokenizer_name)
    cfg = parse_args_and_load_config()

    source_load_reference = None
    if check_source_load_parity:
        source_load_reference = _prepare_source_load_reference(
            cfg,
            input_ids,
            hf_model_cls=hf_model_cls,
            trust_remote_code=trust_remote_code,
            experts_implementation=experts_implementation,
            hf_device_map_auto=hf_device_map_auto,
            hf_source_post_load_dequantize=hf_source_post_load_dequantize,
        )
        _barrier()

    # Phase 1: Construct the model, optionally compare it against the raw HF
    # source-load reference, then train and checkpoint.
    torch.cuda.reset_peak_memory_stats()
    trainer = recipe_cls(cfg)
    trainer.setup()

    if check_source_load_parity:
        device = next(trainer.model_parts[0].parameters()).device
        trainer_source_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
        source_load_failure = _compare_source_load_parity(
            source_load_reference,
            trainer_source_logits,
            trainer.model_parts[0],
            source_load_kl_threshold=source_load_kl_threshold,
            source_load_mean_kl_threshold=source_load_mean_kl_threshold,
            source_load_cosine_threshold=source_load_cosine_threshold,
        )
        if source_load_failure is not None:
            deferred_failures.append(f"Phase 0 source-load parity:\n{source_load_failure}")
        del trainer_source_logits, source_load_reference
        _barrier()
        if _rank0():
            _cleanup_source_load_sync(cfg)
        _barrier()

        # Do not train with a model that has already run a no-grad parity
        # forward. FSDP2 and non-reentrant activation-checkpoint wrappers keep
        # forward bookkeeping that is expected to match the first backward;
        # reusing the probed model can make that bookkeeping nondeterministic.
        # A fresh recipe is also the clearest separation between the optional
        # diagnostic and the checkpoint lifecycle under test.
        _release_recipe_memory(trainer)
        del trainer
        _barrier()
        cfg = parse_args_and_load_config()
        trainer = recipe_cls(cfg)
        trainer.setup()

    trainer.run_train_validation_loop()

    # Memory tracking after training
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_cpu_gb = _rss_gb()
    if _rank0():
        print(f"\n[Memory] Peak VRAM: {peak_vram_gb:.2f} GB, Peak CPU RSS: {peak_cpu_gb:.2f} GB")
    if max_vram_gb > 0:
        assert peak_vram_gb <= max_vram_gb, f"Peak VRAM {peak_vram_gb:.2f} GB exceeds threshold {max_vram_gb:.2f} GB"
    if max_cpu_gb > 0:
        assert peak_cpu_gb <= max_cpu_gb, f"Peak CPU RSS {peak_cpu_gb:.2f} GB exceeds threshold {max_cpu_gb:.2f} GB"

    # Phase 2: Capture reference logits before teardown
    device = next(trainer.model_parts[0].parameters()).device
    reference_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)

    # Phase 3: Reload automodel from consolidated checkpoint
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    ckpt_step_dirs = sorted(checkpoint_dir.glob("epoch_*_step_*"))
    assert len(ckpt_step_dirs) > 0, f"No checkpoint subdirectories found under {checkpoint_dir}"
    ckpt_step_dir = ckpt_step_dirs[-1]
    consolidated_dir = ckpt_step_dir / "model" / "consolidated"

    is_peft = hasattr(cfg, "peft")
    original_pretrained_path = cfg.model.pretrained_model_name_or_path
    # Materialize an explicit YAML quantization subtree for the vanilla HF
    # reload. Passing ConfigNode directly would recurse in HF's deepcopy.
    # Source FP8 configs without an override are detected and dequantized below.
    _raw_qc = getattr(cfg.model, "quantization_config", None)
    if _raw_qc is not None and hasattr(_raw_qc, "instantiate"):
        try:
            original_quantization_config = _raw_qc.instantiate()
        except Exception:
            original_quantization_config = None
    else:
        original_quantization_config = _raw_qc

    _release_recipe_memory(trainer)
    del trainer

    # Phantom key check: scan consolidated safetensors for leaked quantization keys
    if check_phantom_keys and _rank0():
        from safetensors import safe_open

        assert consolidated_dir.exists(), f"Phantom key check: {consolidated_dir} does not exist"
        sf_files = sorted(consolidated_dir.glob("*.safetensors"))
        assert len(sf_files) > 0, f"Phantom key check: no .safetensors files in {consolidated_dir}"
        for sf_path in sf_files:
            with safe_open(str(sf_path), framework="pt") as f:
                for key in f.keys():
                    assert "_blocks" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"
                    assert "_scales" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"
        print(f"[Phantom keys] Scanned {len(sf_files)} files, no _blocks/_scales keys ✓")

    # Pre-populate HF dynamic module cache on rank 0 to prevent filesystem races
    # when all ranks simultaneously load trust_remote_code models from local paths.
    # On shared filesystems (e.g. Lustre), concurrent shutil.copy2 calls from
    # multiple ranks cause PermissionError. Also seed all transitive .py
    # imports so transformers' local-dir branch (which only copies direct
    # imports of the modeling file) doesn't fail on files imported
    # indirectly (e.g. Nemotron-Flash's triton_attention.py).
    if not is_peft:
        if _rank0():
            from transformers import AutoConfig

            _prepopulate_hf_dynamic_modules_cache(consolidated_dir)
            try:
                AutoConfig.from_pretrained(str(consolidated_dir), trust_remote_code=True)
            except Exception:
                pass
        _barrier()

    cfg = parse_args_and_load_config()
    if not is_peft:
        cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
        cfg.checkpoint.enabled = False
    restored_trainer = recipe_cls(cfg)
    restored_trainer.setup()

    restored_logits = _get_logits(restored_trainer.model_parts[0], input_ids, device, trainer=restored_trainer)

    kl_restored = _kl_divergence_from_logits(reference_logits, restored_logits)
    max_kl_restored = kl_restored.max().item()
    if _rank0():
        print(f"\n[Phase 3] Automodel-from-consolidated max KL: {max_kl_restored:.6e} (threshold: {kl_threshold:.6e})")
    assert max_kl_restored <= kl_threshold, (
        f"KL divergence between original and automodel-from-consolidated too large: "
        f"max per-token KL = {max_kl_restored:.6e} > threshold {kl_threshold:.6e}"
    )

    # Phase 4: Load into vanilla HF (rank 0 only)
    _release_recipe_memory(restored_trainer)
    del restored_trainer
    hf_reload_sync_paths = _prepare_hf_reload_sync(cfg)

    hf_reload_error = None
    if skip_hf_reload:
        if _rank0():
            print("[Phase 4] Skipped (ci.checkpoint_robustness.skip_hf_reload=true).")
    elif _rank0():
        from contextlib import nullcontext

        # Nemotron-Flash's custom ``LlamaRotaryEmbedding.__init__`` does
        # ``torch.arange(...).to(device)`` which blows up under transformers 5.x's
        # unconditional ``torch.device("meta")`` init context. Wrap HF loads in
        # ``no_hf_meta_device`` so the model is built on a real device; we rely on
        # this only for trust_remote_code models since standard HF models init
        # correctly under meta.
        try:
            from nemo_automodel._transformers.model_init import no_hf_meta_device

            _no_meta = no_hf_meta_device() if trust_remote_code else nullcontext()
        except ImportError:
            _no_meta = nullcontext()

        hf_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )
        # Remote-code models can ship attention names that transformers 5.x
        # rejects. Select a supported implementation while keeping Nemotron-H
        # off HF's incompatible FlashAttention varlen path.
        if trust_remote_code and "attn_implementation" not in hf_kwargs:
            config_path = original_pretrained_path if is_peft else consolidated_dir
            hf_kwargs["attn_implementation"] = _get_trust_remote_code_attn_implementation(config_path)
        if experts_implementation and not trust_remote_code:
            hf_kwargs["experts_implementation"] = experts_implementation
            hf_kwargs["trust_remote_code"] = False
        if hf_device_map_auto:
            hf_kwargs["device_map"] = "auto"
        if original_quantization_config is not None:
            hf_kwargs["quantization_config"] = original_quantization_config
        else:
            config_path = original_pretrained_path if is_peft else consolidated_dir
            fp8_config = _load_hf_fp8_dequantized_config(
                config_path,
                trust_remote_code=trust_remote_code,
            )
            if fp8_config is not None:
                hf_kwargs["config"] = fp8_config
        # Load the reference model straight onto the target GPU. Materialising a
        # 14B checkpoint on CPU and then ``.to(device)`` costs ~50-225s, and that
        # rank-0-only stall trips the NCCL watchdog while the other ranks idle at
        # the post-phase ``_barrier()`` below (the failure mode this test hit on
        # large models). ``device_map`` places weights on the GPU directly (~12s).
        # trust_remote_code models need the ``_no_meta`` real-device init, which is
        # incompatible with device_map's meta dispatch, and the auto/quantized
        # paths set ``device_map`` themselves -- so restrict this to standard-HF loads.
        if "device_map" not in hf_kwargs and not trust_remote_code and original_quantization_config is None:
            hf_kwargs["device_map"] = {"": device}

        if is_peft:
            from peft import PeftModel

            with _no_meta:
                if "device_map" in hf_kwargs:
                    base_model = hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
                else:
                    base_model = _fix_meta_rotary_embeddings(
                        hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
                    ).to(device)
            # Re-init non-persistent rotary buffers for ``model_type`` values
            # in ``_MODELS_REQUIRING_BUFFER_REINIT`` (``nemotron-nas``,
            # ``gemma3``) — their ``inv_freq`` is computed in ``__init__`` and
            # never written to the checkpoint; meta-device init leaves
            # garbage values after ``from_pretrained``.
            _reinit_rotary_per_module(base_model, device)
            # For Nemotron-Flash (``model_type=="nemotron_flash"``) the
            # ``inv_freq`` buffer also lands garbage under HF load but its
            # NTK formula is non-standard, so route through the dedicated
            # ``fix_rotary_embeddings`` patch which installs Flash's own NTK
            # formula and mirrors Flash's native forward.
            if trust_remote_code:
                from nemo_automodel._transformers.v4_patches.rotary import (
                    fix_rotary_embeddings,
                    should_fix_rotary_embeddings,
                )

                if should_fix_rotary_embeddings([base_model]):
                    fix_rotary_embeddings([base_model])
            peft_model = PeftModel.from_pretrained(base_model, str(ckpt_step_dir / "model"))
            hf_logits = _get_logits(peft_model, input_ids, device)

            # PEFT fused QKV key verification
            if check_fused_qkv_keys:
                from safetensors import safe_open

                adapter_path = ckpt_step_dir / "model" / "adapter_model.safetensors"
                assert adapter_path.exists(), f"adapter_model.safetensors not found at {adapter_path}"
                with safe_open(str(adapter_path), framework="pt") as f:
                    adapter_keys = list(f.keys())
                combined_keys = [k for k in adapter_keys if "qkv_proj" in k or "gate_up_proj" in k]
                assert len(combined_keys) == 0, (
                    f"Fused QKV check failed: adapter_model.safetensors contains combined projection keys: "
                    f"{combined_keys}"
                )
                print(f"[Fused QKV] No combined projection keys in adapter ({len(adapter_keys)} keys checked) ✓")

            del peft_model, base_model
        else:
            _prepopulate_hf_dynamic_modules_cache(consolidated_dir)
            with _no_meta:
                if "device_map" in hf_kwargs:
                    hf_model = hf_model_cls.from_pretrained(str(consolidated_dir), **hf_kwargs)
                else:
                    hf_model = _fix_meta_rotary_embeddings(
                        hf_model_cls.from_pretrained(str(consolidated_dir), **hf_kwargs)
                    ).to(device)
            # Re-init non-persistent rotary buffers for nemotron-nas / gemma3
            # (``_MODELS_REQUIRING_BUFFER_REINIT`` allow-list). See PEFT branch
            # above for details.
            _reinit_rotary_per_module(hf_model, device)
            # For Nemotron-Flash: install NTK inv_freq via dedicated patch.
            if trust_remote_code:
                from nemo_automodel._transformers.v4_patches.rotary import (
                    fix_rotary_embeddings,
                    should_fix_rotary_embeddings,
                )

                if should_fix_rotary_embeddings([hf_model]):
                    fix_rotary_embeddings([hf_model])
            hf_logits = _get_logits(hf_model, input_ids, device)
            del hf_model

        kl_hf = _kl_divergence_from_logits(reference_logits, hf_logits)
        max_kl_hf = kl_hf.max().item()
        print(f"[Phase 4] HF-loaded max KL: {max_kl_hf:.6e} (threshold: {hf_kl_threshold:.6e})")
        if max_kl_hf > hf_kl_threshold:
            hf_reload_error = (
                "KL divergence between original and HF-loaded model too large: "
                f"max per-token KL = {max_kl_hf:.6e} > threshold {hf_kl_threshold:.6e}"
            )

    hf_reload_error = _finish_hf_reload_sync(hf_reload_sync_paths, hf_reload_error)
    assert hf_reload_error is None, hf_reload_error

    # Phase 5 (optional): Cross-TP — reload consolidated with a different TP size
    if cross_tp_size > 0 and not is_peft:
        cfg = parse_args_and_load_config()
        cfg.model.pretrained_model_name_or_path = str(consolidated_dir)
        cfg.checkpoint.enabled = False
        cfg.distributed.tp_size = cross_tp_size
        cfg.distributed.dp_size = None
        cross_tp_trainer = recipe_cls(cfg)
        cross_tp_trainer.setup()

        cross_tp_logits = _get_logits(cross_tp_trainer.model_parts[0], input_ids, device, trainer=cross_tp_trainer)

        kl_cross_tp = _kl_divergence_from_logits(reference_logits, cross_tp_logits)
        max_kl_cross_tp = kl_cross_tp.max().item()
        if _rank0():
            print(
                f"[Phase 5] Cross-TP (tp_size={cross_tp_size}) max KL: "
                f"{max_kl_cross_tp:.6e} (threshold: {cross_tp_kl_threshold:.6e})"
            )
        assert max_kl_cross_tp <= cross_tp_kl_threshold, (
            f"KL divergence between original and cross-TP model too large: "
            f"max per-token KL = {max_kl_cross_tp:.6e} > threshold {cross_tp_kl_threshold:.6e}"
        )

        _release_recipe_memory(cross_tp_trainer)
        del cross_tp_trainer
        _barrier()

    # Phase 6 (optional): Training resumption — verify loss continuity
    # Phase 1 trained for max_steps (e.g. 5) and checkpointed. We now train a fresh baseline
    # for max_steps+3 (no checkpoint save), then resume from the checkpoint and train to
    # max_steps+3. For SFT, losses should match to ~4 decimal places.
    if check_resume:
        import json
        import shutil
        import tempfile

        # Baseline: fresh continuous run for max_steps+3, saving losses to a temp dir
        baseline_dir = tempfile.mkdtemp(prefix="resume_baseline_")
        cfg = parse_args_and_load_config()
        original_max_steps = cfg.step_scheduler.max_steps
        resume_max_steps = original_max_steps + 3
        cfg.step_scheduler.max_steps = resume_max_steps
        cfg.checkpoint.checkpoint_dir = baseline_dir
        cfg.checkpoint.enabled = False
        # Phase 1 computed lr_decay_steps = min(total_epoch_steps, original_max_steps).
        # With resume_max_steps the baseline would compute a *different* lr_decay_steps,
        # causing the LR curve (and thus model weights) at step N to diverge from
        # Phase 1's checkpoint.  Pin lr_decay_steps to match Phase 1.
        if hasattr(cfg, "lr_scheduler") and cfg.lr_scheduler is not None:
            cfg.lr_scheduler.lr_decay_steps = original_max_steps
        baseline_trainer = recipe_cls(cfg)
        baseline_trainer.setup()
        baseline_trainer.run_train_validation_loop()

        baseline_losses = {}
        baseline_jsonl = Path(baseline_dir) / "training.jsonl"
        if _rank0() and baseline_jsonl.exists():
            with open(baseline_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["step"] >= original_max_steps:
                        baseline_losses[entry["step"]] = entry["loss"]

        _release_recipe_memory(baseline_trainer)
        del baseline_trainer
        shutil.rmtree(baseline_dir, ignore_errors=True)

        # Resume: reload from Phase 1 checkpoint and train to resume_max_steps.
        cfg = parse_args_and_load_config()
        cfg.checkpoint.restore_from = str(ckpt_step_dir)
        cfg.step_scheduler.max_steps = resume_max_steps
        resume_trainer = recipe_cls(cfg)
        resume_trainer.setup()
        resume_trainer.run_train_validation_loop()

        # Compare losses at the overlapping steps
        resume_jsonl = checkpoint_dir / "training.jsonl"
        if _rank0():
            assert baseline_losses, "Phase 6: baseline_losses is empty — no steps to compare"
            assert resume_jsonl.exists(), f"Phase 6: {resume_jsonl} not found"

            resume_losses = {}
            with open(resume_jsonl) as f:
                for line in f:
                    entry = json.loads(line)
                    if entry["step"] in baseline_losses:
                        resume_losses[entry["step"]] = entry["loss"]

            matched_steps = 0
            for step in sorted(baseline_losses):
                if step in resume_losses:
                    matched_steps += 1
                    bl = baseline_losses[step]
                    rl = resume_losses[step]
                    diff = abs(bl - rl)
                    print(f"[Phase 6] Step {step}: baseline_loss={bl:.6f}, resume_loss={rl:.6f}, diff={diff:.6e}")
                    if not is_peft:
                        assert diff < resume_loss_threshold, (
                            f"SFT loss mismatch after resume at step {step}: "
                            f"baseline={bl:.6f}, resume={rl:.6f}, diff={diff:.6e}"
                        )

            assert matched_steps > 0, (
                f"Phase 6: no overlapping steps found between baseline ({sorted(baseline_losses.keys())}) "
                f"and resume ({sorted(resume_losses.keys())})"
            )
            print(f"[Phase 6] Training resumption verified ({matched_steps} steps compared) ✓")

        _release_recipe_memory(resume_trainer)
        del resume_trainer
        _barrier()

    # Skip the atexit-registered destroy_process_group() call. MoE models with expert
    # parallelism create NCCL sub-groups (DeepEP) that leave pending collective state,
    # causing destroy_process_group() to hang and SIGABRT. Since the process is about to
    # exit, the OS reclaims all resources safely.
    import atexit

    from nemo_automodel.components.distributed.init_utils import destroy_global_state

    atexit.unregister(destroy_global_state)

    if deferred_failures:
        raise AssertionError(
            "Checkpoint robustness completed with deferred failures:\n\n" + "\n\n".join(deferred_failures)
        )


def test_checkpoint_robustness() -> None:
    """Run checkpoint robustness with the LLM finetune recipe."""
    from transformers import AutoModelForCausalLM

    run_checkpoint_robustness(
        recipe_cls=TrainFinetuneRecipeForNextTokenPrediction,
        hf_model_cls=AutoModelForCausalLM,
    )


if __name__ == "__main__":
    test_checkpoint_robustness()
