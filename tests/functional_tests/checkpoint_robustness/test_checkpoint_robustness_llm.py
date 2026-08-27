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

"""Train, checkpoint, and validate AutoModel and vanilla-HF reloads.

Launch with ``torchrun --nproc-per-node=<N> -m <this_module> --config <config.yaml>``.

The CI launcher runs phases in isolated processes by default through ``--isolated_phase``. Accepted phase names are
``source_load_reference``, ``source_load_parity``, ``train_and_save``, ``automodel_reload``, ``hf_reload``, ``resume``,
and ``cross_tp_reload``. Direct invocation without ``--isolated_phase`` retains the compatibility single-process
lifecycle.

See ``tests/ci_tests/README.md#checkpoint-robustness`` for the public phase contract, tolerance profiles, and supported
recipe controls.
"""

from __future__ import annotations

import gc
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from collections.abc import Callable
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from nemo_automodel.recipes.base_recipe import BaseRecipe

import datasets
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from nemo_automodel.components.checkpoint.checkpointing import (
    _MODELS_REQUIRING_BUFFER_REINIT,
    _reinit_non_persistent_buffers,
)
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.components.config.loader import ConfigNode
from nemo_automodel.shared.utils import dtype_from_str
from tests.functional_tests.checkpoint_robustness.parity_metrics import (
    _apply_parity_threshold_overrides,
    _compute_parity_metrics,
    _normalize_parity_profile_overrides,
    _normalize_parity_threshold_overrides,
    _parity_failures,
    _resolve_parity_thresholds,
    _select_parity_profile,
    _validate_logits,
)
from tests.functional_tests.checkpoint_robustness.resume_trajectory import (
    _checkpoint_for_completed_steps,
    _checkpoint_state_snapshot,
    _configure_resumed_run,
    _configure_uninterrupted_run,
    _disable_checkpoint_saves_after_restore,
    _gather_rank_failures,
    _load_reference_trajectory,
    _persist_reference_trajectory,
    _persist_training_reproducibility,
    _report_resume_comparison,
    _report_training_reproducibility,
    _resolve_resume_loss_tolerance,
    _restored_state_mismatch,
    _resume_plan_from_config,
    _TrainingReproducibilityRecorder,
    _TrajectoryRecorder,
)
from tests.functional_tests.checkpoint_robustness.shape_diagnostics import (
    _build_shape_diagnostic_report,
    _normalize_shape_diagnostic_config,
    _persist_shape_diagnostic_report,
    _ShapeDiagnosticConfig,
)

datasets.disable_caching()

_PARITY_DOCUMENT_PATH = Path(__file__).with_name("parity_document.mdx")
_PARITY_DOCUMENT_SHA256 = "8f734b2ee925ab82afb56dfa3a512108b70d3c54a2489f7978a036420da34cdb"  # pragma: allowlist secret
_ROUTER_DIAGNOSTIC_MODEL_TYPE = "glm4_moe_lite"
_REMOVED_CHECKPOINT_ROBUSTNESS_FIELDS = {
    "automodel_reload_cosine_threshold",
    "automodel_reload_mean_kl_threshold",
    "automodel_reload_p95_kl_threshold",
    "check_hf_reload",
    "check_resume",
    "check_source_load_parity",
    "cosine_threshold",
    "cross_tp_kl_threshold",
    "hf_cosine_threshold",
    "hf_kl_threshold",
    "kl_threshold",
    "no_check_resume",
    "skip_automodel_logit_parity",
    "skip_hf_logit_parity",
    "source_load_cosine_threshold",
    "source_load_kl_threshold",
    "source_load_mean_kl_threshold",
}


@dataclass(frozen=True)
class _LogitParityPolicy:
    """Configuration and enforcement state for one full-logit comparison."""

    phase: str
    comparison: str
    comparison_kind: Literal["same_implementation", "cross_framework", "cross_topology"]
    profile: str
    enforce: bool = True
    mean_kl_threshold_override: float | None = None
    p95_kl_threshold_override: float | None = None
    cosine_threshold_override: float | None = None
    gate_sequence_length: int | None = None


def _parse_boolean_fixture_value(value: object, *, key: str) -> bool:
    """Normalize a YAML boolean fixture value without string-truthiness surprises."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    raise ValueError(f"ci.checkpoint_robustness.{key} must be true or false, got {value!r}")


def _extract_custom_args(argv: list[str]) -> tuple[dict[str, object], list[str]]:
    """Separate test-specific CLI flags from config parser arguments."""
    custom_keys = {
        "--isolated_phase",
        "--cross_tp_size",
        "--experts_implementation",
        "--hf_adapter_ignored_key_prefix",
        "--hf_device_map_cpu_max_memory_gib",
        "--hf_device_map_max_memory_gib",
        "--hf_reload_timeout_seconds",
        "--tokenizer_name",
        "--max_vram_gb",
        "--max_cpu_gb",
        "--training_reproducibility_loss_threshold",
        "--parity_sequence_length",
        "--cross_framework_gate_sequence_length",
        "--parity_threshold_overrides",
        "--parity_tolerance_profile",
        "--parity_tolerance_profile_overrides",
        "--resume_first_loss_threshold",
        "--resume_loss_threshold",
        "--resume_tolerance_profile",
    }
    boolean_keys = {
        "--trust_remote_code",
        "--check_fused_qkv_keys",
        "--check_phantom_keys",
        "--hf_device_map_auto",
        "--hf_source_post_load_dequantize",
        "--capture_router_diagnostics",
        "--skip_resume",
        "--skip_source_load_parity",
        "--skip_source_load_logit_parity",
        "--skip_hf_reload",
        "--skip_automodel_reload_logit_parity",
        "--skip_hf_reload_logit_parity",
    }
    boolean_config_keys = {key.lstrip("-") for key in boolean_keys}
    custom: dict[str, object] = {}
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
    cli_custom_keys = set(custom)

    # Read ci.checkpoint_robustness from the YAML config as defaults.
    # CLI args take precedence over YAML values.
    config_path = None
    for j, arg in enumerate(remaining):
        if arg == "--config" and j + 1 < len(remaining):
            config_path = remaining[j + 1]
            break
    ci_robustness: dict = {}
    if config_path:
        import yaml

        with open(config_path) as f:
            raw_cfg = yaml.safe_load(f) or {}
        ci_robustness = raw_cfg.get("ci", {}).get("checkpoint_robustness") or {}
        removed_fields = sorted(_REMOVED_CHECKPOINT_ROBUSTNESS_FIELDS & ci_robustness.keys())
        if removed_fields:
            raise ValueError("Removed checkpoint-robustness fields are not supported: " + ", ".join(removed_fields))
        default_on_control_keys = {
            "parity_threshold_overrides",
            "parity_tolerance_profile_overrides",
            "skip_resume",
            "skip_source_load_parity",
        }
        for k, v in ci_robustness.items():
            if k in default_on_control_keys:
                continue
            if k == "shape_diagnostic":
                continue
            if k not in custom:
                if "." in k:
                    # Dotted keys are config overrides (e.g. distributed.tp_size),
                    # route them to the config parser instead of the custom dict.
                    remaining.extend([f"--{k}", str(v)])
                elif k in boolean_config_keys:
                    custom[k] = _parse_boolean_fixture_value(v, key=k)
                elif not isinstance(v, bool):
                    custom[k] = str(v)

    raw_threshold_overrides = custom.get("parity_threshold_overrides")
    if raw_threshold_overrides is None:
        raw_threshold_overrides = ci_robustness.get("parity_threshold_overrides")
    if isinstance(raw_threshold_overrides, str):
        import yaml

        raw_threshold_overrides = yaml.safe_load(raw_threshold_overrides)
    if raw_threshold_overrides is not None:
        custom["parity_threshold_overrides"] = _normalize_parity_threshold_overrides(raw_threshold_overrides)

    raw_profile_overrides = custom.get("parity_tolerance_profile_overrides")
    if raw_profile_overrides is None:
        raw_profile_overrides = ci_robustness.get("parity_tolerance_profile_overrides")
    if isinstance(raw_profile_overrides, str):
        import yaml

        raw_profile_overrides = yaml.safe_load(raw_profile_overrides)
    if raw_profile_overrides is not None:
        custom["parity_tolerance_profile_overrides"] = _normalize_parity_profile_overrides(raw_profile_overrides)

    if "skip_source_load_parity" in cli_custom_keys:
        source_load_parity_enabled = False
    elif "skip_source_load_parity" in ci_robustness:
        source_load_parity_enabled = not _parse_boolean_fixture_value(
            ci_robustness["skip_source_load_parity"], key="skip_source_load_parity"
        )
    else:
        source_load_parity_enabled = True
    custom["source_load_parity_enabled"] = source_load_parity_enabled
    if not source_load_parity_enabled:
        custom["skip_source_load_parity"] = True

    if "skip_resume" in cli_custom_keys:
        resume_enabled = False
    elif "skip_resume" in ci_robustness:
        resume_enabled = not _parse_boolean_fixture_value(ci_robustness["skip_resume"], key="skip_resume")
    else:
        resume_enabled = True
    custom["resume_enabled"] = resume_enabled
    if not resume_enabled:
        custom["skip_resume"] = True

    parity_sequence_length = int(custom.get("parity_sequence_length", "2048"))
    if parity_sequence_length <= 0:
        raise ValueError(f"parity_sequence_length must be positive, got {parity_sequence_length}")
    if "cross_framework_gate_sequence_length" in custom:
        cross_framework_gate_sequence_length = int(custom["cross_framework_gate_sequence_length"])
        if cross_framework_gate_sequence_length <= 0:
            raise ValueError(
                f"cross_framework_gate_sequence_length must be positive, got {cross_framework_gate_sequence_length}"
            )
        if cross_framework_gate_sequence_length > parity_sequence_length:
            raise ValueError(
                "cross_framework_gate_sequence_length must not exceed parity_sequence_length "
                f"({cross_framework_gate_sequence_length} > {parity_sequence_length})"
            )
    else:
        cross_framework_gate_sequence_length = None
    raw_shape_diagnostic = ci_robustness.get("shape_diagnostic")
    shape_diagnostic = _normalize_shape_diagnostic_config(
        raw_shape_diagnostic,
        parity_sequence_length=parity_sequence_length,
    )
    shape_diagnostic_lengths = shape_diagnostic.lengths(
        parity_sequence_length=parity_sequence_length,
        gate_sequence_length=cross_framework_gate_sequence_length,
    )
    if shape_diagnostic.sweep_lengths and not source_load_parity_enabled:
        raise ValueError("shape_diagnostic.sweep_lengths requires source-load parity to be enabled")
    if shape_diagnostic_lengths:
        custom["shape_diagnostic"] = shape_diagnostic
    if "hf_reload_timeout_seconds" in custom and int(custom["hf_reload_timeout_seconds"]) <= 0:
        raise ValueError("hf_reload_timeout_seconds must be positive")
    _resolve_parity_thresholds(str(custom.get("parity_tolerance_profile", "standard")), "same_implementation")

    return custom, remaining


def _get_input_ids(tokenizer_name: str | None) -> list[int]:
    """Tokenize the repository's long-form finetuning guide for parity testing."""
    if tokenizer_name is None:
        raise ValueError("tokenizer_name is required to tokenize the checkpoint parity document")
    from nemo_automodel import NeMoAutoTokenizer

    tokenizer = NeMoAutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=True,
        local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    )
    return tokenizer.encode(_get_parity_document(), add_special_tokens=False)


def _get_parity_document() -> str:
    """Load and validate the fixed long-form document shared by LLM and VLM parity tests."""
    try:
        document_bytes = _PARITY_DOCUMENT_PATH.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Unable to load checkpoint parity document: {_PARITY_DOCUMENT_PATH}") from exc
    document_sha256 = hashlib.sha256(document_bytes).hexdigest()
    if document_sha256 != _PARITY_DOCUMENT_SHA256:
        raise RuntimeError(
            "Checkpoint parity document changed unexpectedly: "
            f"expected sha256={_PARITY_DOCUMENT_SHA256}, got sha256={document_sha256}"
        )
    return document_bytes.decode("utf-8")


def _load_hf_config(
    pretrained_model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
    revision: str | None = None,
    token: str | bool | None = None,
):
    """Load the HF config used by a vanilla-reference model."""
    from transformers import AutoConfig

    config_kwargs: dict[str, str | bool] = {
        "trust_remote_code": trust_remote_code,
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    return AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)


def _load_hf_fp8_dequantized_config(
    pretrained_model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
    revision: str | None = None,
    token: str | bool | None = None,
):
    """Return an HF config that dequantizes a fine-grained FP8 checkpoint, if applicable."""
    config = _load_hf_config(
        pretrained_model_name_or_path,
        trust_remote_code=trust_remote_code,
        revision=revision,
        token=token,
    )
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


def _repair_legacy_partial_rotary_config(config) -> bool:
    """Restore a legacy partial-rotary spec dropped by newer Transformers configs.

    Checkpoints such as MiniMax-M2.* express partial RoPE only through the
    legacy ``rotary_dim`` config field. Transformers 5.x in-tree configs keep
    ``rotary_dim`` as a plain attribute while their models read only
    ``rope_parameters["partial_rotary_factor"]``, so the vanilla reference
    silently rotates the full head dimension with the wrong frequency ladder
    and becomes a deterministic but invalid reference (AMINT-286). Derive the
    missing factor as ``rotary_dim / head_dim``.

    Args:
        config: Loaded HF config for the vanilla reference model.

    Returns:
        True when the config's rope parameters were repaired; False when the
        config has no legacy spec or already carries a partial factor.
    """
    rotary_dim = getattr(config, "rotary_dim", None)
    head_dim = getattr(config, "head_dim", None)
    if not rotary_dim or not head_dim or rotary_dim == head_dim:
        return False
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        if rope_parameters.get("partial_rotary_factor"):
            return False
        rope_parameters["partial_rotary_factor"] = rotary_dim / head_dim
        return True
    if rope_parameters is not None and hasattr(rope_parameters, "partial_rotary_factor"):
        if rope_parameters.partial_rotary_factor:
            return False
        rope_parameters.partial_rotary_factor = rotary_dim / head_dim
        return True
    return False


def _is_nemo_owned_config(config) -> bool:
    """Return True when a config object is an AutoModel component config."""
    return type(config).__module__.startswith("nemo_automodel")


def _replace_nemo_owned_reference_config(
    config,
    pretrained_model_name_or_path: str | Path,
    *,
    trust_remote_code: bool,
    revision: str | None = None,
    token: str | bool | None = None,
):
    """Re-resolve a vanilla-reference config hijacked by AutoModel registrations.

    nemo_automodel registers its component config classes into Transformers'
    ``CONFIG_MAPPING`` (``_CUSTOM_CONFIG_REGISTRATIONS``), and a locally
    registered ``model_type`` wins over checkpoint remote code even with
    ``trust_remote_code=True``. Inside the harness process, AutoConfig then
    resolves checkpoints such as Kimi-Linear to an AutoModel-owned class while
    the model class still comes from the checkpoint's ``auto_map``, and
    ``from_pretrained`` rejects the pair with a ``config_class`` mismatch
    (AMINT-288). Resolve the checkpoint's own config class from its
    ``auto_map`` instead, preserving a load-time FP8 ``dequantize`` request.

    The same registrations also shadow in-tree config classes: for built-in
    references (``trust_remote_code=False``), AutoConfig pairs the
    AutoModel-owned config with the in-tree model, which then crashes on
    attribute contracts the local class does not carry (the in-tree
    minimax_m3_vl vision tower reads ``vision_config.temporal_patch_size``).
    Resolve the in-tree class from Transformers' own name table, which
    registration cannot shadow.

    Args:
        config: Config resolved for the vanilla reference load.
        pretrained_model_name_or_path: Checkpoint the reference loads from.
        trust_remote_code: Whether the reference load trusts remote code.
        revision: Optional checkpoint revision.
        token: Optional Hub token.

    Returns:
        Tuple of the faithful config and whether a replacement happened.
    """
    if not _is_nemo_owned_config(config):
        return config, False

    load_kwargs: dict[str, str | bool] = {
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    if token is not None:
        load_kwargs["token"] = token

    config_cls = None
    auto_map = getattr(config, "auto_map", None) or {}
    class_reference = auto_map.get("AutoConfig")
    if trust_remote_code and class_reference:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_cls = get_class_from_dynamic_module(class_reference, pretrained_model_name_or_path, **load_kwargs)
    else:
        import transformers
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

        class_name = CONFIG_MAPPING_NAMES.get(getattr(config, "model_type", None) or "")
        if class_name is not None:
            config_cls = getattr(transformers, class_name, None)
    if config_cls is None:
        return config, False
    replacement = config_cls.from_pretrained(pretrained_model_name_or_path, **load_kwargs)

    original_quantization = getattr(config, "quantization_config", None)
    requested_dequantize = (
        original_quantization.get("dequantize")
        if isinstance(original_quantization, dict)
        else getattr(original_quantization, "dequantize", None)
    )
    if requested_dequantize:
        replacement_quantization = getattr(replacement, "quantization_config", None)
        if isinstance(replacement_quantization, dict):
            replacement.quantization_config = {**replacement_quantization, "dequantize": True}
        elif replacement_quantization is not None:
            replacement_quantization.dequantize = True
    return replacement, True


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


def _hf_device_map_max_memory(
    max_memory_gib: str | float | None,
    cpu_max_memory_gib: str | float | None = None,
) -> dict[int | str, str] | None:
    """Build optional GPU and CPU caps for vanilla-HF automatic placement."""
    if max_memory_gib is None:
        return None
    max_memory_gib = float(max_memory_gib)
    if max_memory_gib <= 0:
        raise ValueError("hf_device_map_max_memory_gib must be positive")
    device_count = torch.cuda.device_count()
    if device_count == 0:
        raise RuntimeError("hf_device_map_max_memory_gib requires at least one visible CUDA device")
    max_memory: dict[int | str, str] = {index: f"{max_memory_gib:g}GiB" for index in range(device_count)}
    if cpu_max_memory_gib is not None:
        cpu_max_memory_gib = float(cpu_max_memory_gib)
        if cpu_max_memory_gib <= 0:
            raise ValueError("hf_device_map_cpu_max_memory_gib must be positive")
        max_memory["cpu"] = f"{cpu_max_memory_gib:g}GiB"
    return max_memory


def _peft_adapter_load_kwargs(hf_kwargs: dict[str, object]) -> dict[str, object]:
    """Build PEFT adapter-load arguments for an AutoModel checkpoint.

    PEFT removes Accelerate's base-model hooks and dispatches the wrapped model
    again when any base module is CPU- or disk-offloaded. Without the original
    limits, that second pass sizes its map from GPUs that already contain the
    base model and can overcommit them while reattaching hooks.

    AutoModel already saves adapter keys in the final HF namespace. Disable the
    base checkpoint's conversion map so PEFT does not remap those keys again.
    """
    return {
        "key_mapping": {},
        **{key: hf_kwargs[key] for key in ("device_map", "max_memory") if key in hf_kwargs},
    }


def _patch_remote_masking_api_compatibility() -> None:
    """Adapt remote model code to masking kwargs removed or renamed by Transformers."""
    import transformers.masking_utils as masking_utils

    for function_name in ("create_causal_mask", "create_sliding_window_causal_mask"):
        mask_function = getattr(masking_utils, function_name)
        if getattr(mask_function, "_nemo_removed_kwargs_patched", False):
            continue
        parameters = inspect.signature(mask_function).parameters.values()
        parameter_names = {parameter.name for parameter in parameters}
        accepts_var_keyword = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters)
        drop_cache_position = "cache_position" not in parameter_names and not accepts_var_keyword
        # Transformers v5.x renamed ``input_embeds`` to ``inputs_embeds``;
        # pre-v5 remote code (e.g. Kimi-Linear) still passes the old keyword.
        rename_input_embeds = (
            "input_embeds" not in parameter_names and "inputs_embeds" in parameter_names and not accepts_var_keyword
        )
        if not drop_cache_position and not rename_input_embeds:
            continue

        @wraps(mask_function)
        def compatible_mask_function(
            *args,
            _mask_function=mask_function,
            _drop_cache_position=drop_cache_position,
            _rename_input_embeds=rename_input_embeds,
            **kwargs,
        ):
            if _drop_cache_position:
                kwargs.pop("cache_position", None)
            if _rename_input_embeds and "input_embeds" in kwargs:
                kwargs["inputs_embeds"] = kwargs.pop("input_embeds")
            return _mask_function(*args, **kwargs)

        compatible_mask_function._nemo_removed_kwargs_patched = True  # type: ignore[attr-defined]
        setattr(masking_utils, function_name, compatible_mask_function)


def _patch_remote_fla_api_compatibility() -> None:
    """Adapt remote model code to the renamed fla-core KDA gate API.

    Kimi-Linear's pre-0.4.2 remote code calls
    ``fused_kda_gate(g, A_log, head_k_dim, g_bias=...)`` with a flat
    ``g`` of shape ``[..., heads * head_k_dim]`` that the old kernel reshaped
    internally. fla-core 0.4.2 renamed the API to
    ``fused_kda_gate(g, A_log, dt_bias=None, lower_bound=None, ...)`` and
    expects ``g`` pre-reshaped to ``[..., heads, head_k_dim]``. Translate
    legacy calls when the installed function no longer accepts the old form;
    an installed fla that still accepts ``g_bias`` is left untouched.
    """
    try:
        import fla.ops.kda as kda_ops
        import fla.ops.kda.gate as kda_gate
    except ImportError:
        return

    gate_function = kda_gate.fused_kda_gate
    if getattr(gate_function, "_nemo_legacy_kda_gate_patched", False):
        return
    parameter_names = set(inspect.signature(gate_function).parameters)
    if "g_bias" in parameter_names or "dt_bias" not in parameter_names:
        return

    @wraps(gate_function)
    def compatible_fused_kda_gate(g, A_log, *args, _gate_function=gate_function, **kwargs):
        """Translate a legacy KDA gate call onto the renamed fla-core API.

        Args:
            g: Gate projection. Legacy callers pass a flat Tensor of shape
                [..., heads * head_k_dim] together with a positional
                ``head_k_dim``; new-style callers pass [..., heads, head_k_dim].
            A_log: Per-head log-decay Tensor of shape [heads].
            *args: A leading int is the legacy positional ``head_k_dim``; a
                following tensor is the legacy positional ``g_bias``.
            **kwargs: Legacy ``g_bias``/``beta``/``threshold`` keywords are
                translated or rejected; everything else passes through.

        Returns:
            Gate Tensor of shape [..., heads, head_k_dim] from the new API.
        """
        if args and isinstance(args[0], int):
            head_k_dim = args[0]
            remaining = list(args[1:])
            if remaining:
                kwargs.setdefault("g_bias", remaining.pop(0))
            if remaining:
                raise TypeError("Unexpected extra positional arguments for legacy fused_kda_gate call")
            g_bias = kwargs.pop("g_bias", None)
            beta = kwargs.pop("beta", 1.0)
            threshold = kwargs.pop("threshold", 20.0)
            if beta != 1.0 or threshold != 20.0:
                raise TypeError(
                    "Legacy fused_kda_gate beta/threshold overrides are not supported by the installed fla API"
                )
            return _gate_function(g.view(*g.shape[:-1], -1, head_k_dim), A_log, dt_bias=g_bias, **kwargs)
        return _gate_function(g, A_log, *args, **kwargs)

    compatible_fused_kda_gate._nemo_legacy_kda_gate_patched = True  # type: ignore[attr-defined]
    kda_gate.fused_kda_gate = compatible_fused_kda_gate
    if getattr(kda_ops, "fused_kda_gate", None) is gate_function:
        kda_ops.fused_kda_gate = compatible_fused_kda_gate


def _rss_gb() -> float:
    """Current RSS in GB from /proc/self/statm."""
    page_size = os.sysconf("SC_PAGE_SIZE")
    with open("/proc/self/statm") as f:
        rss_pages = int(f.read().split()[1])
    return rss_pages * page_size / 1024**3


def _fit_input_ids_to_sequence_length(input_ids: list[int], sequence_length: int) -> list[int]:
    """Truncate a tokenized document to the requested parity length."""
    if not input_ids:
        raise ValueError("Tokenized parity document must not be empty")
    if sequence_length <= 0:
        raise ValueError(f"parity_sequence_length must be positive, got {sequence_length}")
    if len(input_ids) < sequence_length:
        raise ValueError(
            f"Tokenized parity document contains {len(input_ids)} tokens, but parity_sequence_length requires "
            f"{sequence_length}; choose a shorter sequence length or a longer parity document"
        )
    return input_ids[:sequence_length]


def _compare_logits(
    artifact_dir: Path,
    reference_logits: torch.Tensor,
    candidate_logits: torch.Tensor,
    policy: _LogitParityPolicy,
    *,
    diagnostics: dict[str, object] | None = None,
) -> str | None:
    """Compute, persist, and optionally enforce one full-logit comparison.

    Args:
        artifact_dir: Directory that owns checkpoint-robustness artifacts.
        reference_logits: Reference tensor of shape [..., vocab], with arbitrary leading token dimensions.
        candidate_logits: Candidate tensor of shape [..., vocab], matching ``reference_logits`` exactly.
        policy: Comparison identity, numerical profile, targeted overrides, and enforcement state.
        diagnostics: Optional diagnostic evidence to embed in the emitted schema-v3 record.

    Returns:
        A failure message when an enforced gate fails, otherwise ``None``.
    """
    metrics = _compute_parity_metrics(reference_logits, candidate_logits)
    full_sequence_length = reference_logits.shape[-2]
    gate_sequence_length = full_sequence_length if policy.gate_sequence_length is None else policy.gate_sequence_length
    if gate_sequence_length <= 0:
        raise ValueError(f"{policy.comparison} gate_sequence_length must be positive, got {gate_sequence_length}")
    if policy.gate_sequence_length is not None and policy.comparison_kind != "cross_framework":
        raise ValueError("gate_sequence_length is supported only for cross-framework comparisons")
    if gate_sequence_length > full_sequence_length:
        raise ValueError(
            f"{policy.comparison} gate_sequence_length={gate_sequence_length} exceeds the captured "
            f"sequence length {full_sequence_length}"
        )
    uses_gate_prefix = gate_sequence_length < full_sequence_length
    if uses_gate_prefix:
        gate_metrics = _compute_parity_metrics(
            reference_logits[..., :gate_sequence_length, :],
            candidate_logits[..., :gate_sequence_length, :],
        )
    else:
        gate_metrics = metrics
    profile_thresholds = _resolve_parity_thresholds(policy.profile, policy.comparison_kind)
    threshold_overrides = {
        "mean_kl": policy.mean_kl_threshold_override,
        "p95_kl": policy.p95_kl_threshold_override,
        "cosine_similarity": policy.cosine_threshold_override,
    }
    uses_threshold_overrides = any(value is not None for value in threshold_overrides.values())
    active_profile_thresholds = _apply_parity_threshold_overrides(
        profile_thresholds,
        mean_kl=policy.mean_kl_threshold_override,
        p95_kl=policy.p95_kl_threshold_override,
        cosine_similarity=policy.cosine_threshold_override,
    )
    profile_failures = _parity_failures(gate_metrics, profile_thresholds)
    active_failures = _parity_failures(gate_metrics, active_profile_thresholds)
    full_sequence_profile_failures = _parity_failures(metrics, profile_thresholds)
    full_sequence_active_failures = _parity_failures(metrics, active_profile_thresholds)
    threshold_mode = "profile_with_numeric_overrides" if uses_threshold_overrides else "profile"
    payload = {
        "schema_version": 3,
        "parity_document_sha256": _PARITY_DOCUMENT_SHA256,
        "phase": policy.phase,
        "comparison": policy.comparison,
        "comparison_kind": policy.comparison_kind,
        "profile": policy.profile,
        "profile_thresholds": profile_thresholds.to_dict(),
        "threshold_overrides": threshold_overrides,
        "active_thresholds": active_profile_thresholds.to_dict(),
        "threshold_mode": threshold_mode,
        "enforced": policy.enforce,
        "passed": not policy.enforce or not active_failures,
        "within_active_thresholds": not active_failures,
        "would_pass_profile": not profile_failures,
        "full_sequence_within_active_thresholds": not full_sequence_active_failures,
        "full_sequence_would_pass_profile": not full_sequence_profile_failures,
        "failures": list(active_failures) if policy.enforce else [],
        "threshold_failures": list(active_failures),
        "profile_failures": list(profile_failures),
        "full_sequence_threshold_failures": list(full_sequence_active_failures),
        "full_sequence_profile_failures": list(full_sequence_profile_failures),
        "full_sequence_length": full_sequence_length,
        "gate_sequence_length": gate_sequence_length,
        "uses_gate_prefix": uses_gate_prefix,
        "reference_logits": {
            "dtype": str(reference_logits.dtype),
            "shape": list(reference_logits.shape),
        },
        "candidate_logits": {
            "dtype": str(candidate_logits.dtype),
            "shape": list(candidate_logits.shape),
        },
        "metrics": metrics.to_dict(),
        "gate_metrics": gate_metrics.to_dict(),
    }
    diagnostic_keys = set(diagnostics or ())
    diagnostic_collisions = diagnostic_keys & payload.keys()
    if diagnostic_collisions:
        raise ValueError(f"Diagnostic keys collide with parity record fields: {sorted(diagnostic_collisions)}")
    if diagnostics:
        payload.update(diagnostics)
    report_dir = artifact_dir / "parity_metrics"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{policy.phase}_{policy.comparison}.json"
    temporary_report_path = report_path.with_suffix(".tmp")
    temporary_report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary_report_path.replace(report_path)
    logged_payload = {key: value for key, value in payload.items() if key not in diagnostic_keys}
    if diagnostics:
        logged_payload["embedded_diagnostics"] = sorted(diagnostics)
    logged_payload["report_path"] = str(report_path)
    print(f"CHECKPOINT_PARITY_METRICS {json.dumps(logged_payload, sort_keys=True)}")

    if not policy.enforce:
        print(
            f"[{policy.phase}] {policy.comparison} metrics are informational; "
            f"would_pass_active_thresholds={not active_failures}, would_pass_profile={not profile_failures}"
        )
        return None
    if not active_failures:
        return None
    gate_scope = f"first {gate_sequence_length} tokens" if uses_gate_prefix else "full sequence"
    return f"{policy.comparison} parity failed on {gate_scope}: " + "; ".join(active_failures)


def _comparison_threshold_overrides(custom_args: dict[str, object], comparison: str) -> dict[str, float]:
    """Return normalized overrides for one comparison."""
    all_overrides = _normalize_parity_threshold_overrides(custom_args.get("parity_threshold_overrides"))
    return all_overrides.get(comparison, {})


def _comparison_profile(custom_args: dict[str, object], comparison: str) -> str:
    """Return the comparison profile, falling back to the global profile."""
    return _select_parity_profile(
        str(custom_args.get("parity_tolerance_profile", "standard")),
        custom_args.get("parity_tolerance_profile_overrides"),
        comparison,
    )


def _source_load_parity_policy(custom_args: dict[str, object], *, enforce: bool = True) -> _LogitParityPolicy:
    """Build the Phase 0 source-load policy."""
    overrides = _comparison_threshold_overrides(custom_args, "source_load")
    return _LogitParityPolicy(
        phase="phase_0",
        comparison="source_load",
        comparison_kind="cross_framework",
        profile=_comparison_profile(custom_args, "source_load"),
        enforce=enforce and not bool(custom_args.get("skip_source_load_logit_parity", False)),
        mean_kl_threshold_override=overrides.get("mean_kl"),
        p95_kl_threshold_override=overrides.get("p95_kl"),
        cosine_threshold_override=overrides.get("cosine_similarity"),
        gate_sequence_length=(
            int(custom_args["cross_framework_gate_sequence_length"])
            if "cross_framework_gate_sequence_length" in custom_args
            else None
        ),
    )


def _repeatability_policy(*, phase: str, comparison: str, profile: str) -> _LogitParityPolicy:
    """Build an informational policy for two forwards through one loaded model."""
    return _LogitParityPolicy(
        phase=phase,
        comparison=comparison,
        comparison_kind="same_implementation",
        profile=profile,
        enforce=False,
    )


def _automodel_reload_parity_policy(custom_args: dict[str, object]) -> _LogitParityPolicy:
    """Build the Phase 2 AutoModel model-reload policy."""
    overrides = _comparison_threshold_overrides(custom_args, "automodel_reload")
    return _LogitParityPolicy(
        phase="phase_2",
        comparison="automodel_model_reload",
        comparison_kind="same_implementation",
        profile=_comparison_profile(custom_args, "automodel_reload"),
        enforce=not bool(custom_args.get("skip_automodel_reload_logit_parity", False)),
        mean_kl_threshold_override=overrides.get("mean_kl"),
        p95_kl_threshold_override=overrides.get("p95_kl"),
        cosine_threshold_override=overrides.get("cosine_similarity"),
    )


def _hf_reload_parity_policy(custom_args: dict[str, object]) -> _LogitParityPolicy:
    """Build the Phase 3 vanilla-HF export-reload policy."""
    overrides = _comparison_threshold_overrides(custom_args, "hf_reload")
    return _LogitParityPolicy(
        phase="phase_3",
        comparison="hf_export_reload",
        comparison_kind="cross_framework",
        profile=_comparison_profile(custom_args, "hf_reload"),
        enforce=not bool(custom_args.get("skip_hf_reload_logit_parity", False)),
        mean_kl_threshold_override=overrides.get("mean_kl"),
        p95_kl_threshold_override=overrides.get("p95_kl"),
        cosine_threshold_override=overrides.get("cosine_similarity"),
        gate_sequence_length=(
            int(custom_args["cross_framework_gate_sequence_length"])
            if "cross_framework_gate_sequence_length" in custom_args
            else None
        ),
    )


def _cross_tp_parity_policy(custom_args: dict[str, object]) -> _LogitParityPolicy:
    """Build the optional Phase 5 cross-topology policy."""
    overrides = _comparison_threshold_overrides(custom_args, "cross_tp")
    return _LogitParityPolicy(
        phase="phase_5",
        comparison="cross_tp_reload",
        comparison_kind="cross_topology",
        profile=_comparison_profile(custom_args, "cross_tp"),
        mean_kl_threshold_override=overrides.get("mean_kl"),
        p95_kl_threshold_override=overrides.get("p95_kl"),
        cosine_threshold_override=overrides.get("cosine_similarity"),
    )


def _tensor_digest(tensor: torch.Tensor) -> dict[str, object]:
    """Return exact dtype, shape, and byte-level SHA-256 metadata for a tensor."""
    local_tensor = tensor.detach()
    if isinstance(local_tensor, DTensor):
        local_tensor = local_tensor.to_local()
    cpu_tensor = local_tensor.contiguous().cpu()
    raw_bytes = cpu_tensor.view(torch.uint8).numpy()
    return {
        "dtype": str(cpu_tensor.dtype),
        "shape": list(cpu_tensor.shape),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }


def _trainable_parameter_digests(model_parts: list[torch.nn.Module]) -> dict[str, dict[str, object]]:
    """Hash every rank-local trainable parameter for exact PEFT save/reload comparison."""
    digests: dict[str, dict[str, object]] = {}
    for part_index, model_part in enumerate(model_parts):
        for name, parameter in model_part.named_parameters():
            if not parameter.requires_grad:
                continue
            key = f"part_{part_index}:{name.replace('_checkpoint_wrapped_module.', '')}"
            digests[key] = _tensor_digest(parameter)
    return digests


def _accelerate_offloaded_tensor(model: torch.nn.Module, parameter_name: str) -> torch.Tensor:
    """Read a meta parameter's backing tensor from its nearest Accelerate offload hook."""

    def hooks(hook):
        for child_hook in getattr(hook, "hooks", (hook,)):
            if child_hook is hook:
                yield hook
            else:
                yield from hooks(child_hook)

    name_parts = parameter_name.split(".")
    for module_end in range(len(name_parts) - 1, -1, -1):
        module_name = ".".join(name_parts[:module_end])
        local_name = ".".join(name_parts[module_end:])
        module = model if not module_name else model.get_submodule(module_name)
        hf_hook = getattr(module, "_hf_hook", None)
        if hf_hook is None:
            continue
        for hook in hooks(hf_hook):
            weights_map = getattr(hook, "weights_map", None)
            if weights_map is None:
                continue
            try:
                tensor = weights_map[local_name]
            except KeyError:
                continue
            assert not tensor.is_meta, f"Accelerate backing tensor is still meta for {parameter_name}"
            return tensor
    raise AssertionError(f"No Accelerate backing tensor found for meta parameter {parameter_name}")


def _assert_peft_adapter_matches_checkpoint(
    peft_model: torch.nn.Module,
    adapter_path: Path,
    ignored_key_prefix: str | None = None,
) -> tuple[int, int]:
    """Verify that vanilla PEFT loaded every HF-supported adapter tensor exactly."""
    from peft import get_peft_model_state_dict
    from safetensors import safe_open

    assert adapter_path.exists(), f"adapter_model.safetensors not found at {adapter_path}"
    # This comparison is against AutoModel's adapter-only safetensors export.
    # PEFT's default ``save_embedding_layers="auto"`` treats a targeted output
    # head as an embedding and adds its base-layer weight, even though that
    # tensor is not adapter state and is intentionally absent from the file.
    loaded_adapter = get_peft_model_state_dict(peft_model, save_embedding_layers=False)
    normalized_parameter_names = {}
    if any(tensor.is_meta for tensor in loaded_adapter.values()):
        for parameter_name, _ in peft_model.named_parameters():
            name_parts = parameter_name.split(".")
            if "lora_" not in parameter_name or len(name_parts) < 2:
                continue
            normalized_name = ".".join((*name_parts[:-2], name_parts[-1]))
            normalized_parameter_names[normalized_name] = parameter_name
    with safe_open(str(adapter_path), framework="pt") as saved_adapter:
        expected_keys = set(saved_adapter.keys())
        loaded_keys = set(loaded_adapter)
        missing = sorted(expected_keys - loaded_keys)
        ignored_missing = [key for key in missing if ignored_key_prefix and key.startswith(ignored_key_prefix)]
        required_missing = sorted(set(missing) - set(ignored_missing))
        unexpected = sorted(loaded_keys - expected_keys)
        assert not required_missing and not unexpected, (
            "Vanilla PEFT adapter key mismatch: "
            f"missing={required_missing[:10]}, unexpected={unexpected[:10]}, "
            f"ignored_missing={ignored_missing[:10]}"
        )

        mismatches = []
        matched_keys = expected_keys - set(ignored_missing)
        for key in sorted(matched_keys):
            expected_digest = _tensor_digest(saved_adapter.get_tensor(key))
            loaded_tensor = loaded_adapter[key]
            if loaded_tensor.is_meta:
                assert key in normalized_parameter_names, f"No PEFT parameter found for meta adapter tensor {key}"
                loaded_tensor = _accelerate_offloaded_tensor(peft_model, normalized_parameter_names[key])
            loaded_digest = _tensor_digest(loaded_tensor)
            if expected_digest != loaded_digest:
                mismatches.append(f"{key}: checkpoint={expected_digest}, loaded={loaded_digest}")
                if len(mismatches) == 10:
                    break

    assert not mismatches, "Vanilla PEFT adapter tensor mismatch:\n" + "\n".join(mismatches)
    return len(matched_keys), len(ignored_missing)


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


def _model_pretrained_path(model_cfg: ConfigNode, model_kwargs: dict | None = None) -> str | Path:
    """Resolve the source checkpoint for from-pretrained and config-based recipes."""
    direct_path = getattr(model_cfg, "pretrained_model_name_or_path", None)
    if direct_path:
        return direct_path

    nested_config = getattr(model_cfg, "config", None)
    nested_path = getattr(nested_config, "pretrained_model_name_or_path", None)
    if nested_path:
        return nested_path
    nested_name_or_path = getattr(nested_config, "name_or_path", None)
    if nested_name_or_path:
        return nested_name_or_path

    if model_kwargs is not None:
        materialized_config = model_kwargs.get("config")
        for attribute in ("pretrained_model_name_or_path", "name_or_path", "_name_or_path"):
            materialized_path = getattr(materialized_config, attribute, None)
            if materialized_path:
                return materialized_path

    raise ValueError(
        "Checkpoint robustness requires model.pretrained_model_name_or_path or "
        "model.config.pretrained_model_name_or_path"
    )


def _set_model_pretrained_path(model_cfg: ConfigNode, pretrained_model_name_or_path: str | Path) -> None:
    """Retarget both from-pretrained and config-based recipes to an exported checkpoint."""
    path = str(pretrained_model_name_or_path)
    nested_config = getattr(model_cfg, "config", None)
    if nested_config is not None and (
        hasattr(nested_config, "pretrained_model_name_or_path") or hasattr(nested_config, "name_or_path")
    ):
        if hasattr(nested_config, "pretrained_model_name_or_path"):
            nested_config.pretrained_model_name_or_path = path
        if hasattr(nested_config, "name_or_path"):
            nested_config.name_or_path = path
        return
    model_cfg.pretrained_model_name_or_path = path


def _resolve_hf_model_class(
    pretrained_model_name_or_path: str | Path,
    default_model_cls: type,
    *,
    revision: str | None = None,
    token: str | bool | None = None,
) -> type:
    """Select the vanilla-HF auto-model class supported by the checkpoint."""
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, PretrainedConfig
    from transformers.models.auto.modeling_auto import (
        MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    )

    config_kwargs: dict[str, str | bool] = {
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **config_kwargs)
    auto_map = config_dict.get("auto_map") or {}
    supported_classes = {
        model_cls.__name__: model_cls for model_cls in (AutoModelForImageTextToText, AutoModelForCausalLM)
    }

    if auto_map:
        if default_model_cls.__name__ in auto_map:
            return default_model_cls
        advertised_classes = [model_cls for name, model_cls in supported_classes.items() if name in auto_map]
        if len(advertised_classes) == 1:
            return advertised_classes[0]
        return default_model_cls

    model_type = config_dict.get("model_type")
    native_mappings = {
        AutoModelForCausalLM: MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        AutoModelForImageTextToText: MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    }
    native_classes = [model_cls for model_cls, mapping in native_mappings.items() if model_type in mapping]
    if len(native_classes) == 1:
        return native_classes[0]
    return default_model_cls


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
    from transformers import PretrainedConfig

    config_kwargs: dict[str, str | bool] = {
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **config_kwargs)

    # Remote-code checkpoints do not share optimized attention backend support:
    # these models reject the recipe backend under the pinned Transformers
    # version. Eager is their common vanilla-HF reference path.
    eager_model_types = {"deepseek_v4", "nemotron-nas", "nemotron_flash", "nemotron_h", "step3p7"}
    return "eager" if config_dict.get("model_type") in eager_model_types else "flash_attention_2"


def _resolve_hf_attn_implementation(
    pretrained_model_name_or_path: str | Path,
    requested_implementation: str | None,
    *,
    hf_model_cls: type,
    trust_remote_code: bool,
    revision: str | None = None,
    token: str | bool | None = None,
) -> str | None:
    """Use the recipe backend when vanilla HF supports it, otherwise use eager."""
    if trust_remote_code:
        compatible_implementation = _get_trust_remote_code_attn_implementation(
            pretrained_model_name_or_path,
            revision=revision,
            token=token,
        )
        if compatible_implementation == "eager" or requested_implementation is None:
            return compatible_implementation
        return requested_implementation

    if requested_implementation not in {"sdpa", "flash_attention_2"}:
        return requested_implementation

    from transformers import AutoConfig

    config_kwargs: dict[str, str | bool] = {
        "local_files_only": os.environ.get("HF_HUB_OFFLINE", "0") == "1",
    }
    if revision is not None:
        config_kwargs["revision"] = revision
    if token is not None:
        config_kwargs["token"] = token
    config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)
    try:
        concrete_model_cls = hf_model_cls._model_mapping[type(config)]
    except (AttributeError, KeyError):
        return requested_implementation

    support_attribute = {
        "sdpa": "_supports_sdpa",
        "flash_attention_2": "_supports_flash_attn",
    }[requested_implementation]
    if not bool(getattr(concrete_model_cls, support_attribute, False)):
        return "eager"
    return requested_implementation


def _hf_source_load_kwargs(
    model_kwargs: dict,
    *,
    pretrained_model_name_or_path: str | Path,
    source_dtype: torch.dtype,
    trust_remote_code: bool,
    experts_implementation: str | None,
    hf_model_cls: type,
    device: torch.device,
    hf_device_map_auto: bool,
    hf_device_map_max_memory_gib: str | float | None = None,
    hf_device_map_cpu_max_memory_gib: str | float | None = None,
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
    recipe_config = hf_kwargs.get("config")
    if recipe_config is not None and _is_nemo_owned_config(recipe_config):
        # AutoModel component configs are never valid for a vanilla-HF
        # reference; remote and in-tree model classes both reject them with a
        # config_class mismatch. Let the reference resolve the checkpoint's
        # own config instead.
        del hf_kwargs["config"]
    hf_kwargs["torch_dtype"] = source_dtype
    hf_kwargs["trust_remote_code"] = trust_remote_code
    hf_kwargs["local_files_only"] = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    attn_implementation = _resolve_hf_attn_implementation(
        pretrained_model_name_or_path,
        hf_kwargs.get("attn_implementation"),
        hf_model_cls=hf_model_cls,
        trust_remote_code=hf_kwargs["trust_remote_code"],
        revision=hf_kwargs.get("revision"),
        token=hf_kwargs.get("token"),
    )
    if attn_implementation is not None:
        hf_kwargs["attn_implementation"] = attn_implementation
    if experts_implementation and not trust_remote_code:
        hf_kwargs["experts_implementation"] = experts_implementation
        hf_kwargs["trust_remote_code"] = False
    if hf_device_map_auto:
        hf_kwargs["device_map"] = "auto"
        # References too large for uncapped automatic placement (the 427B
        # MiniMax-M3 fills every GPU and OOMs on the fused-expert concat
        # transient) need the same GPU caps + CPU spill as the HF reload path.
        max_memory = _hf_device_map_max_memory(hf_device_map_max_memory_gib, hf_device_map_cpu_max_memory_gib)
        if max_memory is not None:
            hf_kwargs["max_memory"] = max_memory
            print(f"[Phase 0] Automatic device-map memory limits: {max_memory}")
    if (
        "device_map" not in hf_kwargs
        and not hf_kwargs["trust_remote_code"]
        and hf_kwargs.get("quantization_config") is None
    ):
        hf_kwargs["device_map"] = {"": device}
    return hf_kwargs


def _hf_model_load_context(*, trust_remote_code: bool, has_device_map: bool) -> AbstractContextManager[None]:
    """Return the model-initialization context for a vanilla Hugging Face load.

    Hugging Face device-map dispatch relies on meta initialization to preserve
    non-persistent buffers such as Llama RoPE frequencies. Disabling that path
    before a device-mapped load can replace those buffers with uninitialized storage.
    The real-device fallback is therefore limited to remote-code loads whose
    placement is not already owned by ``device_map``.
    """
    if not trust_remote_code or has_device_map:
        return nullcontext()

    from nemo_automodel._transformers.model_init import no_hf_meta_device

    return no_hf_meta_device()


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
    try:
        inspect.signature(get_input_embeddings).bind()
    except TypeError:
        # Some remote-code VLMs require input IDs to select an embedding path.
        # That accessor cannot provide the storage-only alias check used here.
        return None
    except ValueError:
        # Some extension callables do not expose an inspectable signature. Let
        # the normal zero-argument accessor path handle those implementations.
        pass
    try:
        embeddings = get_input_embeddings()
    except TypeError:
        # Some wrappers expose a zero-argument accessor but delegate to an
        # input-dependent inner model, so storage aliasing is not observable.
        return None
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


def _normalize_peft_no_split_modules(model) -> None:
    """Adapt Transformers 5 device-map metadata to the PEFT/Accelerate contract."""
    no_split_modules = getattr(model, "_no_split_modules", None)
    if isinstance(no_split_modules, set):
        # Transformers 5 normalizes this metadata to a set, while the
        # PEFT/Accelerate versions in CI expect a list or tuple.
        model._no_split_modules = sorted(no_split_modules)


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


# Leaf names distinctive enough to register as layout-independent fp32
# aliases. Transformers matches _keep_in_fp32_modules_strict entries as
# unanchored substrings, so only leaves that cannot collide with unrelated
# modules qualify — a generic leaf such as ``proj`` or ``scale`` (from
# Gemma4's ``router.proj``/``router.scale``) would pin ``q_proj``,
# ``down_proj``, and friends fp32 across the whole bf16 reference.
_HF_FP32_LEAF_ALIASES = ("e_score_correction_bias",)


def _hf_fp32_module_names(hf_config: object) -> tuple[str, ...]:
    """Infer vanilla-HF fp32 names from AutoModel's model-owned checkpoint contract."""
    from nemo_automodel._transformers.model_init import _resolve_custom_model_cls_for_config
    from nemo_automodel.components.models.common.gated_delta_net_fp32 import (
        FP32_GDN_PARAM_NAMES,
        has_gated_delta_net_fp32_checkpoint_contract,
    )

    module_names = list(FP32_GDN_PARAM_NAMES) if has_gated_delta_net_fp32_checkpoint_contract(hf_config) else []
    model_cls = _resolve_custom_model_cls_for_config(hf_config)
    for name in getattr(model_cls, "_keep_in_fp32_modules_strict", None) or ():
        if name not in module_names:
            module_names.append(name)
        # AutoModel strict names use AutoModel module paths, but vanilla HF
        # layouts can hang the same tensor off a different parent (in-tree
        # MiniMax-M2 keeps e_score_correction_bias on ``mlp``, not
        # ``mlp.gate``), so the AutoModel-path entry silently fails to match
        # and the reference's router bias was cast to bf16 — scrambling 30-73%
        # of knife-edge top-k selections per layer (AMINT-286). Also register
        # the distinctive leaf so any layout keeps the tensor in fp32.
        leaf = name.rsplit(".", 1)[-1]
        if leaf in _HF_FP32_LEAF_ALIASES and leaf not in module_names:
            module_names.append(leaf)
    return tuple(module_names)


@contextmanager
def _keep_hf_modules_in_fp32(hf_config: object):
    """Apply AutoModel's fp32 checkpoint contract while loading a vanilla HF model."""
    module_names = _hf_fp32_module_names(hf_config)
    if not module_names:
        yield
        return

    from transformers import PreTrainedModel

    attr = "_keep_in_fp32_modules_strict"
    previous = getattr(PreTrainedModel, attr, None)
    setattr(PreTrainedModel, attr, set(previous or ()) | set(module_names))
    try:
        yield
    finally:
        setattr(PreTrainedModel, attr, previous)


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


def _input_ids_sync_paths(cfg) -> tuple[Path, Path, Path, Path]:
    """Return pre-init input-ID payload and status paths."""
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    sync_dir = checkpoint_dir.parent / f".checkpoint_robustness_input_ids_{_source_load_run_id()}"
    return sync_dir, sync_dir / "input_ids.json", sync_dir / "done", sync_dir / "fail"


def _wait_for_input_ids_rank0(payload_path: Path, done_path: Path, fail_path: Path) -> list[int]:
    """Wait for rank 0 to publish the checkpoint-robustness prompt IDs."""
    timeout_s = int(os.environ.get("CHECKPOINT_INPUT_IDS_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fail_path.exists():
            raise RuntimeError(f"Rank 0 input-ID loading failed:\n{fail_path.read_text()}")
        if done_path.exists():
            try:
                payload = payload_path.read_text()
            except FileNotFoundError:
                pass
            else:
                return [int(token_id) for token_id in json.loads(payload)]
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 input-ID publication")


def _load_input_ids_once(
    cfg,
    input_ids_loader: Callable[[str | None], list[int]],
    tokenizer_name: str | None,
    *,
    sequence_length: int,
) -> list[int]:
    """Load and expand dynamic input IDs once before distributed initialization.

    The tokenizer and processor imports are I/O-heavy on shared filesystems.
    Loading on every worker can turn a cold import into a multi-node import
    storm, so rank 0 writes the small result for the other ranks to read.
    """
    if tokenizer_name is None or _preinit_world_size() == 1:
        return _fit_input_ids_to_sequence_length(input_ids_loader(tokenizer_name), sequence_length)

    sync_dir, payload_path, done_path, fail_path = _input_ids_sync_paths(cfg)
    if _preinit_global_rank() != 0:
        return _wait_for_input_ids_rank0(payload_path, done_path, fail_path)

    sync_dir.mkdir(parents=True, exist_ok=True)
    if done_path.exists():
        return _wait_for_input_ids_rank0(payload_path, done_path, fail_path)

    payload_path.unlink(missing_ok=True)
    done_path.unlink(missing_ok=True)
    fail_path.unlink(missing_ok=True)
    try:
        input_ids = _fit_input_ids_to_sequence_length(input_ids_loader(tokenizer_name), sequence_length)
        temporary_payload_path = payload_path.with_suffix(".tmp")
        temporary_payload_path.write_text(json.dumps(input_ids))
        temporary_payload_path.replace(payload_path)
    except Exception:
        fail_path.write_text(traceback.format_exc())
        raise
    else:
        done_path.write_text("ok\n")
        return input_ids


def _cleanup_input_ids_sync(cfg) -> None:
    """Best-effort cleanup of pre-init input-ID synchronization files."""
    sync_dir, payload_path, done_path, fail_path = _input_ids_sync_paths(cfg)
    if not sync_dir.exists():
        return
    for path in (payload_path, payload_path.with_suffix(".tmp"), done_path, fail_path):
        path.unlink(missing_ok=True)
    try:
        sync_dir.rmdir()
    except OSError:
        pass


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


def _wait_for_hf_reload_rank0(done_path: Path, *, timeout_s: int | None = None) -> None:
    """Wait without an active collective for rank 0 to finish the vanilla-HF reload."""
    if timeout_s is None:
        timeout_s = int(os.environ.get("HF_RELOAD_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if done_path.exists():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 vanilla-HF reload")


def _prepare_hf_reload_sync(cfg, *, timeout_s: int | None = None) -> tuple[Path, Path] | None:
    """Prepare ranks for a long rank-0-only HF reload without starting an NCCL wait."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return None

    sync_dir, done_path = _hf_reload_sync_paths(cfg)
    if _rank0():
        sync_dir.mkdir(parents=True, exist_ok=True)
        done_path.unlink(missing_ok=True)
    _barrier()  # ensure all ranks released recipe memory and rank 0 reset the marker
    if not _rank0():
        _wait_for_hf_reload_rank0(done_path, timeout_s=timeout_s)
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


def _record_deferred_failure(
    deferred_failures: list[str],
    phase: str,
    failure_message: str | None,
) -> None:
    """Record a numerical comparison failure without blocking independent phases."""
    if failure_message is None:
        return
    deferred_failures.append(f"{phase}:\n{failure_message}")
    if _rank0():
        print(f"[{phase}] Comparison failed; deferring failure until later checkpoint phases complete.")


def _broadcast_rank0_failure(failure_message: str | None) -> str | None:
    """Broadcast one rank-0 comparison result so every worker follows the same path."""
    if not dist.is_initialized():
        return failure_message
    payload = [failure_message]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


def _prepare_source_load_reference(
    cfg,
    input_ids: list[int],
    *,
    hf_model_cls: type,
    trust_remote_code: bool | None,
    experts_implementation: str | None,
    hf_device_map_auto: bool,
    hf_device_map_max_memory_gib: str | float | None = None,
    hf_device_map_cpu_max_memory_gib: str | float | None = None,
    hf_source_post_load_dequantize: bool,
    parity_tolerance_profile: str = "standard",
    capture_router_diagnostics: bool = False,
    shape_diagnostic: _ShapeDiagnosticConfig | None = None,
    cross_framework_gate_sequence_length: int | None = None,
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
            hf_device_map_max_memory_gib=hf_device_map_max_memory_gib,
            hf_device_map_cpu_max_memory_gib=hf_device_map_cpu_max_memory_gib,
            hf_source_post_load_dequantize=hf_source_post_load_dequantize,
            parity_tolerance_profile=parity_tolerance_profile,
            capture_router_diagnostics=capture_router_diagnostics,
            shape_diagnostic=shape_diagnostic,
            cross_framework_gate_sequence_length=cross_framework_gate_sequence_length,
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
    trust_remote_code: bool | None,
    experts_implementation: str | None,
    hf_device_map_auto: bool,
    hf_device_map_max_memory_gib: str | float | None = None,
    hf_device_map_cpu_max_memory_gib: str | float | None = None,
    hf_source_post_load_dequantize: bool,
    parity_tolerance_profile: str = "standard",
    capture_router_diagnostics: bool = False,
    shape_diagnostic: _ShapeDiagnosticConfig | None = None,
    cross_framework_gate_sequence_length: int | None = None,
) -> tuple[torch.Tensor, bool | None, bool | None]:
    """Rank-0 implementation of vanilla HF source-load reference capture."""
    from nemo_automodel._transformers.utils import apply_cache_compatibility_patches

    apply_cache_compatibility_patches()
    _patch_remote_masking_api_compatibility()
    _patch_remote_fla_api_compatibility()
    artifact_dir = _robustness_artifact_dir(cfg)
    for router_path in _router_diagnostic_paths(artifact_dir):
        router_path.unlink(missing_ok=True)
    _shape_diagnostic_report_path(artifact_dir, "phase_0").unlink(missing_ok=True)
    if capture_router_diagnostics:
        if shape_diagnostic is not None:
            for sequence_length in shape_diagnostic.lengths(
                parity_sequence_length=len(input_ids),
                gate_sequence_length=cross_framework_gate_sequence_length,
            ):
                _shape_router_capture_path(artifact_dir, sequence_length).unlink(missing_ok=True)

    model_kwargs = _model_kwargs_from_config(cfg.model)
    original_pretrained_path = _model_pretrained_path(cfg.model, model_kwargs)
    hf_model_cls = _resolve_hf_model_class(
        original_pretrained_path,
        hf_model_cls,
        revision=model_kwargs.get("revision"),
        token=model_kwargs.get("token"),
    )
    source_dtype = _resolve_source_load_dtype(model_kwargs)
    if trust_remote_code is None:
        trust_remote_code = bool(model_kwargs.get("trust_remote_code", False))

    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    hf_kwargs = _hf_source_load_kwargs(
        model_kwargs,
        pretrained_model_name_or_path=original_pretrained_path,
        source_dtype=source_dtype,
        trust_remote_code=trust_remote_code,
        experts_implementation=experts_implementation,
        hf_model_cls=hf_model_cls,
        device=device,
        hf_device_map_auto=hf_device_map_auto,
        hf_device_map_max_memory_gib=hf_device_map_max_memory_gib,
        hf_device_map_cpu_max_memory_gib=hf_device_map_cpu_max_memory_gib,
    )
    requested_attn_implementation = model_kwargs.get("attn_implementation")
    if hf_kwargs.get("attn_implementation") != requested_attn_implementation:
        print(
            "[Phase 0] Vanilla-HF attention compatibility fallback: "
            f"requested={requested_attn_implementation!r}, selected={hf_kwargs.get('attn_implementation')!r}"
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

    hf_config = hf_kwargs.get("config")
    if hf_config is None:
        hf_config = _load_hf_config(
            original_pretrained_path,
            trust_remote_code=hf_kwargs["trust_remote_code"],
            revision=hf_kwargs.get("revision"),
            token=hf_kwargs.get("token"),
        )
    hf_config, replaced_reference_config = _replace_nemo_owned_reference_config(
        hf_config,
        original_pretrained_path,
        trust_remote_code=hf_kwargs["trust_remote_code"],
        revision=hf_kwargs.get("revision"),
        token=hf_kwargs.get("token"),
    )
    if replaced_reference_config:
        # Pass the faithful config explicitly so from_pretrained's internal
        # AutoConfig resolution cannot re-select the AutoModel-owned class.
        hf_kwargs["config"] = hf_config
    if _repair_legacy_partial_rotary_config(hf_config):
        # The repaired spec only reaches the model when the config object is
        # passed explicitly; from_pretrained otherwise re-reads config.json.
        hf_kwargs["config"] = hf_config

    model_load_context = _hf_model_load_context(
        trust_remote_code=trust_remote_code,
        has_device_map="device_map" in hf_kwargs,
    )

    print(f"\n[Phase 0] Source-load reference: vanilla HF for {original_pretrained_path}")
    with model_load_context, _keep_hf_modules_in_fp32(hf_config):
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

    with _router_diagnostic_capture_context(
        hf_model,
        _robustness_artifact_dir(cfg),
        framework="hf",
        enabled=capture_router_diagnostics,
    ):
        hf_logits = _get_logits(hf_model, input_ids, device)
    repeated_hf_logits = _get_logits(hf_model, input_ids, device)
    _compare_logits(
        _robustness_artifact_dir(cfg),
        hf_logits,
        repeated_hf_logits,
        _repeatability_policy(
            phase="phase_0",
            comparison="hf_source_self_repeat",
            profile=parity_tolerance_profile,
        ),
    )
    del repeated_hf_logits
    if shape_diagnostic is not None:
        _run_hf_shape_diagnostic(
            hf_model,
            input_ids,
            device,
            hf_logits,
            artifact_dir=artifact_dir,
            config=shape_diagnostic,
            gate_sequence_length=cross_framework_gate_sequence_length,
            capture_router_diagnostics=capture_router_diagnostics,
            phase="phase_0",
        )
    hf_aliased = _lm_head_embedding_aliased(hf_model)
    explicit_tie_word_embeddings = _explicit_tie_word_embeddings(hf_model.config)
    del hf_model
    _release_model_memory()
    return hf_logits, hf_aliased, explicit_tie_word_embeddings


def _compare_source_load_parity(
    source_reference: tuple[torch.Tensor, bool | None, bool | None] | None,
    candidate_logits: torch.Tensor,
    candidate_aliased: bool | None,
    *,
    artifact_dir: Path,
    policy: _LogitParityPolicy,
) -> str | None:
    """Compare the vanilla HF source-load reference against the constructed trainer model.

    Args:
        source_reference: Rank-0 tuple containing logits of shape [batch, sequence, vocab], the HF input/output
            embedding alias state, and the explicit tie-word-embeddings setting. Other ranks pass ``None``.
        candidate_logits: Constructed trainer logits of shape [batch, sequence, vocab].
        candidate_aliased: Constructed trainer input/output embedding alias state.
        artifact_dir: Directory that owns checkpoint-robustness artifacts.
        policy: Source-load metric profile, legacy overrides, and enforcement state.

    Returns:
        Synchronized failure traceback when source-load parity fails, otherwise ``None``. The caller may defer this
        failure until independent checkpoint reload and resume phases have completed.
    """
    failure_message = None
    if _rank0():
        try:
            assert source_reference is not None, "rank 0 source-load reference was not captured"
            hf_logits, hf_aliased, explicit_tie_word_embeddings = source_reference
            assert hf_logits.shape == candidate_logits.shape, (
                f"Source-load parity shape mismatch: HF logits {hf_logits.shape} vs trainer logits "
                f"{candidate_logits.shape}"
            )
            diagnostics: dict[str, object] = {}
            diagnostic_failure = None
            try:
                shape_report_path = _shape_diagnostic_report_path(artifact_dir, "phase_0")
                if shape_report_path.exists():
                    diagnostics["shape_diagnostic"] = json.loads(shape_report_path.read_text())
                hf_router_capture, automodel_router_capture, router_report = _router_diagnostic_paths(artifact_dir)
                if hf_router_capture.exists() or automodel_router_capture.exists():
                    if not hf_router_capture.exists() or not automodel_router_capture.exists():
                        raise AssertionError(
                            "Incomplete Phase 0 router diagnostics: expected both HF and AutoModel captures, got "
                            f"hf={hf_router_capture.exists()} automodel={automodel_router_capture.exists()}"
                        )
                    from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
                        compare_glm_router_captures,
                    )

                    router_diagnostics = compare_glm_router_captures(
                        hf_path=hf_router_capture,
                        automodel_path=automodel_router_capture,
                        report_path=router_report,
                        reference_logits=hf_logits,
                        candidate_logits=candidate_logits,
                    )
                    diagnostics["router_diagnostics"] = router_diagnostics
            except Exception as exc:
                diagnostic_failure = traceback.format_exc()
                diagnostics["diagnostic_failure"] = {
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            parity_failure = _compare_logits(
                artifact_dir,
                hf_logits,
                candidate_logits,
                policy,
                diagnostics=diagnostics,
            )
            comparison_failures = []
            if parity_failure is not None:
                comparison_failures.append(parity_failure)
            if diagnostic_failure is not None:
                comparison_failures.append(f"Phase 0 diagnostics failed:\n{diagnostic_failure}")
            if comparison_failures:
                raise AssertionError("\n".join(comparison_failures))
            print(
                f"[Phase 0] Source-load aliases: hf_aliased={hf_aliased}; "
                f"trainer_aliased={candidate_aliased}; tie_word_embeddings={explicit_tie_word_embeddings}"
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

    # PyTorch pipeline stages preallocate activation buffers for one sequence
    # shape. Resize those buffers before this parity-only forward just as the
    # training recipes do before every schedule step.
    trainer.pp.update_seq_len(orig_seq_len)

    # Replicate the prompt to pp_batch_size so the schedule's batch split is valid.
    ids = torch.tensor([input_ids] * pp_batch_size, device=device, dtype=torch.long)
    attention_mask = torch.ones_like(ids)
    targets = torch.zeros_like(ids) if trainer.pp.info.has_last_stage else None

    captured = [None]

    def _capture_loss_fn(output, target, **_):
        """Capture the main logits from a pipeline loss input.

        Args:
            output: Tensor of shape [microbatch, sequence, vocab], or an MTP tuple whose first element has that shape.
            target: Tensor of shape [microbatch, sequence]. Unused.
            **_: Unused loss keyword arguments.

        Returns:
            Scalar zero tensor on the logits device.
        """
        logits = output[0] if isinstance(output, tuple) else output
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


def _prepare_consolidated_hf_cache_once(cfg, consolidated_dir: Path) -> None:
    """Prepare remote-code files once before an isolated distributed setup.

    Every worker reaches this function before the recipe initializes a process
    group, so ``dist.get_rank()`` cannot select the writer. A small shared-file
    marker lets pre-init global rank 0 finish all cache writes before the other
    workers import the consolidated checkpoint's dynamic modules.
    """
    expected_payload = str(consolidated_dir.resolve())
    sync_dir = _robustness_artifact_dir(cfg) / "hf_dynamic_modules_cache"
    done_path = sync_dir / "done"
    fail_path = sync_dir / "fail"

    def is_ready() -> bool:
        try:
            return done_path.read_text() == expected_payload
        except FileNotFoundError:
            return False

    def prepare_cache() -> None:
        from transformers import AutoConfig

        _prepopulate_hf_dynamic_modules_cache(consolidated_dir)
        try:
            AutoConfig.from_pretrained(str(consolidated_dir), trust_remote_code=True)
        except Exception:
            pass

    if is_ready():
        return
    if _preinit_world_size() == 1:
        prepare_cache()
        return
    if _preinit_global_rank() == 0:
        sync_dir.mkdir(parents=True, exist_ok=True)
        done_path.unlink(missing_ok=True)
        fail_path.unlink(missing_ok=True)
        try:
            prepare_cache()
            temporary_done_path = done_path.with_suffix(".tmp")
            temporary_done_path.write_text(expected_payload)
            temporary_done_path.replace(done_path)
        except Exception:
            fail_path.write_text(traceback.format_exc())
            raise
        return

    timeout_s = int(os.environ.get("HF_DYNAMIC_MODULE_CACHE_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fail_path.exists():
            raise RuntimeError(f"Rank 0 dynamic-module cache preparation failed:\n{fail_path.read_text()}")
        if is_ready():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 dynamic-module cache preparation")


def _rank0() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def _report_phase(message: str) -> None:
    """Report high-level checkpoint-robustness progress on rank 0."""
    if _preinit_global_rank() == 0:
        stream = sys.__stdout__ or sys.stdout
        print(f"[checkpoint_robustness][{time.strftime('%H:%M:%S')}] {message}", file=stream, flush=True)


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


def _checkpoint_paths(cfg) -> tuple[Path, Path, Path]:
    """Locate the latest checkpoint and its consolidated model directory."""
    checkpoint_dir = Path(cfg.checkpoint.checkpoint_dir)
    ckpt_step_dirs = list(checkpoint_dir.glob("epoch_*_step_*"))
    assert ckpt_step_dirs, f"No checkpoint subdirectories found under {checkpoint_dir}"

    def checkpoint_position(path: Path) -> tuple[int, int]:
        name_parts = path.name.split("_")
        return int(name_parts[1]), int(name_parts[3])

    ckpt_step_dir = max(ckpt_step_dirs, key=checkpoint_position)
    return checkpoint_dir, ckpt_step_dir, ckpt_step_dir / "model" / "consolidated"


def _materialize_hf_quantization_config(cfg):
    """Materialize a YAML quantization subtree for vanilla-HF loading."""
    raw_quantization_config = getattr(cfg.model, "quantization_config", None)
    if raw_quantization_config is not None and hasattr(raw_quantization_config, "instantiate"):
        try:
            return raw_quantization_config.instantiate()
        except Exception:
            return None
    return raw_quantization_config


def _run_hf_reload_standing_shape_diagnostic(
    hf_model: torch.nn.Module,
    input_ids: list[int],
    device: torch.device,
    hf_logits: torch.Tensor,
    *,
    artifact_dir: Path,
    custom_args: dict[str, object],
) -> None:
    """Automatically contextualize a shortened Phase 3 cross-framework gate.

    Args:
        hf_model: Loaded vanilla-HF model used for standalone forwards.
        input_ids: Token IDs for the full parity prompt.
        device: Device holding ``hf_model`` and its inputs.
        hf_logits: Full-forward logits of shape [batch, sequence, vocab].
        artifact_dir: Directory that owns checkpoint-robustness artifacts.
        custom_args: Validated checkpoint-robustness fixture settings.
    """
    if "cross_framework_gate_sequence_length" not in custom_args:
        return
    gate_sequence_length = int(custom_args["cross_framework_gate_sequence_length"])
    if gate_sequence_length >= len(input_ids):
        return
    _run_hf_shape_diagnostic(
        hf_model,
        input_ids,
        device,
        hf_logits,
        artifact_dir=artifact_dir,
        config=_ShapeDiagnosticConfig(),
        gate_sequence_length=gate_sequence_length,
        capture_router_diagnostics=False,
        phase="phase_3",
    )


def _collect_hf_reload_shape_diagnostics(
    hf_model: torch.nn.Module,
    input_ids: list[int],
    device: torch.device,
    hf_logits: torch.Tensor,
    *,
    artifact_dir: Path,
    custom_args: dict[str, object],
) -> tuple[dict[str, object], str | None]:
    """Collect optional Phase 3 shape evidence without suppressing parity output."""
    diagnostics: dict[str, object] = {}
    try:
        _run_hf_reload_standing_shape_diagnostic(
            hf_model,
            input_ids,
            device,
            hf_logits,
            artifact_dir=artifact_dir,
            custom_args=custom_args,
        )
        shape_report_path = _shape_diagnostic_report_path(artifact_dir, "phase_3")
        if shape_report_path.exists():
            diagnostics["shape_diagnostic"] = json.loads(shape_report_path.read_text())
    except Exception as exc:
        diagnostics["diagnostic_failure"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
        return diagnostics, traceback.format_exc()
    return diagnostics, None


def _run_vanilla_hf_reload(
    cfg,
    input_ids: list[int],
    reference_logits: torch.Tensor,
    *,
    hf_model_cls: type,
    custom_args: dict[str, object],
) -> str | None:
    """Load the saved model with vanilla HF and validate its adapter and forward pass.

    Args:
        cfg: Resolved checkpoint-robustness recipe configuration.
        input_ids: Token IDs for one text-only parity prompt.
        reference_logits: Tensor of shape [batch, sequence, vocab] captured before checkpoint save.
        hf_model_cls: Hugging Face auto-model class used for the reload.
        custom_args: Checkpoint-robustness fixture settings.

    Returns:
        An error message when loading or parity fails, otherwise ``None``.
    """
    try:
        from nemo_automodel._transformers.utils import apply_cache_compatibility_patches

        # Match Phase 0's vanilla-HF setup. Exported trust-remote-code models can
        # still carry Transformers-v4 list-form ``_tied_weights_keys``.
        apply_cache_compatibility_patches()
        _patch_remote_masking_api_compatibility()
        _patch_remote_fla_api_compatibility()
        _, ckpt_step_dir, consolidated_dir = _checkpoint_paths(cfg)
        artifact_dir = _robustness_artifact_dir(cfg)
        _shape_diagnostic_report_path(artifact_dir, "phase_3").unlink(missing_ok=True)
        is_peft = hasattr(cfg, "peft")
        model_kwargs = _model_kwargs_from_config(cfg.model)
        original_pretrained_path = _model_pretrained_path(cfg.model, model_kwargs)
        original_quantization_config = _materialize_hf_quantization_config(cfg)
        configured_trust_remote_code = custom_args.get("trust_remote_code")
        trust_remote_code = (
            bool(model_kwargs.get("trust_remote_code", False))
            if configured_trust_remote_code is None
            else bool(configured_trust_remote_code)
        )
        experts_implementation = custom_args.get("experts_implementation", None)
        hf_device_map_auto = bool(custom_args.get("hf_device_map_auto", False))
        check_fused_qkv_keys = bool(custom_args.get("check_fused_qkv_keys", False))
        hf_adapter_ignored_key_prefix = custom_args.get("hf_adapter_ignored_key_prefix")
        device = torch.device("cuda", torch.cuda.current_device())
        config_path = original_pretrained_path if is_peft else consolidated_dir
        hf_model_cls = _resolve_hf_model_class(
            config_path,
            hf_model_cls,
            revision=model_kwargs.get("revision"),
            token=model_kwargs.get("token"),
        )

        hf_kwargs = dict(
            torch_dtype=torch.bfloat16,
            trust_remote_code=trust_remote_code,
            local_files_only=os.environ.get("HF_HUB_OFFLINE", "0") == "1",
        )
        for key in ("revision", "token"):
            if model_kwargs.get(key) is not None:
                hf_kwargs[key] = model_kwargs[key]
        # Keep the recipe backend when vanilla HF supports it. Some model
        # implementations reject that backend under the pinned Transformers
        # version, so their independent HF reference uses eager instead.
        attn_implementation = _resolve_hf_attn_implementation(
            config_path,
            model_kwargs.get("attn_implementation"),
            hf_model_cls=hf_model_cls,
            trust_remote_code=trust_remote_code,
            revision=model_kwargs.get("revision"),
            token=model_kwargs.get("token"),
        )
        if attn_implementation is not None:
            hf_kwargs["attn_implementation"] = attn_implementation
        if attn_implementation != model_kwargs.get("attn_implementation") and _rank0():
            print(
                "[Phase 3] Vanilla-HF attention compatibility fallback: "
                f"requested={model_kwargs.get('attn_implementation')!r}, selected={attn_implementation!r}"
            )
        if experts_implementation and not trust_remote_code:
            hf_kwargs["experts_implementation"] = experts_implementation
            hf_kwargs["trust_remote_code"] = False
        if hf_device_map_auto:
            hf_kwargs["device_map"] = "auto"
            max_memory = _hf_device_map_max_memory(
                custom_args.get("hf_device_map_max_memory_gib"),
                custom_args.get("hf_device_map_cpu_max_memory_gib"),
            )
            if max_memory is not None:
                hf_kwargs["max_memory"] = max_memory
                print(f"[HF reload] Automatic device-map memory limits: {max_memory}")
        if original_quantization_config is not None:
            hf_kwargs["quantization_config"] = original_quantization_config
        else:
            fp8_config = _load_hf_fp8_dequantized_config(
                config_path,
                trust_remote_code=trust_remote_code,
            )
            if fp8_config is not None:
                hf_kwargs["config"] = fp8_config
        hf_config = hf_kwargs.get("config")
        if hf_config is None:
            hf_config = _load_hf_config(
                config_path,
                trust_remote_code=trust_remote_code,
                revision=model_kwargs.get("revision"),
                token=model_kwargs.get("token"),
            )
        hf_config, replaced_reference_config = _replace_nemo_owned_reference_config(
            hf_config,
            config_path,
            trust_remote_code=trust_remote_code,
            revision=model_kwargs.get("revision"),
            token=model_kwargs.get("token"),
        )
        if replaced_reference_config:
            # Pass the faithful config explicitly so from_pretrained's internal
            # AutoConfig resolution cannot re-select the AutoModel-owned class.
            hf_kwargs["config"] = hf_config
        if _repair_legacy_partial_rotary_config(hf_config):
            # The repaired spec only reaches the model when the config object
            # is passed explicitly; from_pretrained otherwise re-reads
            # config.json (the consolidated export copies the source config,
            # so it carries the same legacy rotary_dim field).
            hf_kwargs["config"] = hf_config
        # Load the reference model straight onto the target GPU. Materialising a
        # 14B checkpoint on CPU and then ``.to(device)`` costs ~50-225s, and that
        # rank-0-only stall trips the NCCL watchdog while the other ranks idle at
        # a collective. ``device_map`` places weights on GPUs directly.
        if "device_map" not in hf_kwargs and not trust_remote_code and original_quantization_config is None:
            hf_kwargs["device_map"] = {"": device}

        model_load_context = _hf_model_load_context(
            trust_remote_code=trust_remote_code,
            has_device_map="device_map" in hf_kwargs,
        )

        diagnostic_failure = None
        if is_peft:
            from peft import PeftModel

            with model_load_context, _keep_hf_modules_in_fp32(hf_config):
                if "device_map" in hf_kwargs:
                    base_model = hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
                else:
                    base_model = _fix_meta_rotary_embeddings(
                        hf_model_cls.from_pretrained(original_pretrained_path, **hf_kwargs)
                    ).to(device)
            placement_counts: dict[str, int] = {}
            for placement in getattr(base_model, "hf_device_map", {}).values():
                placement_name = str(placement)
                placement_counts[placement_name] = placement_counts.get(placement_name, 0) + 1
            print(f"[HF reload] Base-model device-map placements: {placement_counts}")
            _reinit_rotary_per_module(base_model, device)
            if trust_remote_code:
                from nemo_automodel._transformers.v4_patches.rotary import (
                    fix_rotary_embeddings,
                    should_fix_rotary_embeddings,
                )

                if should_fix_rotary_embeddings([base_model]):
                    fix_rotary_embeddings([base_model])
            _normalize_peft_no_split_modules(base_model)
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(ckpt_step_dir / "model"),
                autocast_adapter_dtype=False,
                **_peft_adapter_load_kwargs(hf_kwargs),
            )
            adapter_path = ckpt_step_dir / "model" / "adapter_model.safetensors"
            matched_adapter_tensors, ignored_adapter_tensors = _assert_peft_adapter_matches_checkpoint(
                peft_model,
                adapter_path,
                ignored_key_prefix=hf_adapter_ignored_key_prefix,
            )
            print(f"[HF reload] Exact saved-adapter fingerprints matched ({matched_adapter_tensors} tensors)")
            if ignored_adapter_tensors:
                print(
                    "[HF reload] Saved adapter tensors absent from vanilla HF were allowed by the configured prefix "
                    f"{hf_adapter_ignored_key_prefix!r} ({ignored_adapter_tensors} tensors)"
                )
            hf_logits = _get_logits(peft_model, input_ids, device)
            repeated_hf_logits = _get_logits(peft_model, input_ids, device)
            _compare_logits(
                _robustness_artifact_dir(cfg),
                hf_logits,
                repeated_hf_logits,
                _repeatability_policy(
                    phase="phase_3",
                    comparison="hf_export_self_repeat",
                    profile=_comparison_profile(custom_args, "hf_reload"),
                ),
            )
            del repeated_hf_logits
            diagnostics, diagnostic_failure = _collect_hf_reload_shape_diagnostics(
                peft_model,
                input_ids,
                device,
                hf_logits,
                artifact_dir=artifact_dir,
                custom_args=custom_args,
            )

            if check_fused_qkv_keys:
                from safetensors import safe_open

                with safe_open(str(adapter_path), framework="pt") as f:
                    adapter_keys = list(f.keys())
                combined_keys = [key for key in adapter_keys if "qkv_proj" in key or "gate_up_proj" in key]
                assert not combined_keys, (
                    f"Fused QKV check failed: adapter_model.safetensors contains combined projection keys: "
                    f"{combined_keys}"
                )
                print(f"[Fused QKV] No combined projection keys in adapter ({len(adapter_keys)} keys checked) ✓")

            del peft_model, base_model
        else:
            _prepopulate_hf_dynamic_modules_cache(consolidated_dir)
            with model_load_context, _keep_hf_modules_in_fp32(hf_config):
                if "device_map" in hf_kwargs:
                    hf_model = hf_model_cls.from_pretrained(str(consolidated_dir), **hf_kwargs)
                else:
                    hf_model = _fix_meta_rotary_embeddings(
                        hf_model_cls.from_pretrained(str(consolidated_dir), **hf_kwargs)
                    ).to(device)
            _reinit_rotary_per_module(hf_model, device)
            if trust_remote_code:
                from nemo_automodel._transformers.v4_patches.rotary import (
                    fix_rotary_embeddings,
                    should_fix_rotary_embeddings,
                )

                if should_fix_rotary_embeddings([hf_model]):
                    fix_rotary_embeddings([hf_model])
            hf_logits = _get_logits(hf_model, input_ids, device)
            repeated_hf_logits = _get_logits(hf_model, input_ids, device)
            _compare_logits(
                _robustness_artifact_dir(cfg),
                hf_logits,
                repeated_hf_logits,
                _repeatability_policy(
                    phase="phase_3",
                    comparison="hf_export_self_repeat",
                    profile=_comparison_profile(custom_args, "hf_reload"),
                ),
            )
            del repeated_hf_logits
            diagnostics, diagnostic_failure = _collect_hf_reload_shape_diagnostics(
                hf_model,
                input_ids,
                device,
                hf_logits,
                artifact_dir=artifact_dir,
                custom_args=custom_args,
            )
            del hf_model

        hf_reload_error = _compare_logits(
            artifact_dir,
            reference_logits,
            hf_logits,
            _hf_reload_parity_policy(custom_args),
            diagnostics=diagnostics,
        )
        del hf_logits
        _release_model_memory()
        failures = []
        if hf_reload_error is not None:
            failures.append(hf_reload_error)
        if diagnostic_failure is not None:
            failures.append(f"Phase 3 diagnostics failed:\n{diagnostic_failure}")
        return "\n".join(failures) if failures else None
    except Exception as exc:
        _release_model_memory()
        return f"Vanilla HF reload failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _robustness_artifact_dir(cfg) -> Path:
    """Return the shared directory used to pass small artifacts between isolated phases."""
    return Path(cfg.checkpoint.checkpoint_dir) / ".checkpoint_robustness"


def _router_diagnostic_paths(artifact_dir: Path) -> tuple[Path, Path, Path]:
    """Return the Phase 0 HF capture, AutoModel capture, and summary paths."""
    router_dir = artifact_dir / "router_diagnostics"
    return router_dir / "phase_0_hf.pt", router_dir / "phase_0_automodel.pt", router_dir / "phase_0_summary.json"


def _shape_diagnostic_report_path(artifact_dir: Path, phase: str) -> Path:
    """Return the full non-gating HF shape report path for one phase."""
    return artifact_dir / "shape_diagnostics" / f"{phase}_hf_shape.json"


def _shape_router_capture_path(artifact_dir: Path, sequence_length: int) -> Path:
    """Return the vanilla-HF router capture path for one standalone shape probe."""
    return artifact_dir / "shape_diagnostics" / "router_captures" / f"hf_{sequence_length}.pt"


def _router_diagnostic_capture_context(
    model: torch.nn.Module,
    artifact_dir: Path,
    *,
    framework: Literal["hf", "automodel"],
    enabled: bool,
    output_path: Path | None = None,
) -> AbstractContextManager[None]:
    """Return an opt-in GLM router capture context for one Phase 0 forward."""
    if not enabled:
        return nullcontext()
    from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
        capture_glm_automodel_routers,
        capture_glm_hf_routers,
    )

    hf_path, automodel_path, _report_path = _router_diagnostic_paths(artifact_dir)
    if framework == "hf":
        return capture_glm_hf_routers(model, hf_path if output_path is None else output_path)
    if output_path is not None:
        raise ValueError("A custom router capture output_path is supported only for vanilla-HF shape diagnostics")
    return capture_glm_automodel_routers(model, automodel_path)


def _validate_router_diagnostic_config(cfg, custom_args: dict[str, object]) -> None:
    """Reject router capture for unsupported model families and pipeline execution."""
    if not custom_args.get("capture_router_diagnostics", False):
        return
    distributed_config = getattr(cfg, "distributed", None)
    pp_size = int(getattr(distributed_config, "pp_size", 1))
    if pp_size > 1:
        raise ValueError(
            "capture_router_diagnostics does not support pipeline parallelism because global rank 0 owns only "
            f"one local pipeline stage; set distributed.pp_size=1 or disable capture (got pp_size={pp_size})"
        )
    model_kwargs = _model_kwargs_from_config(cfg.model)
    pretrained_path = _model_pretrained_path(cfg.model, model_kwargs)
    configured_trust_remote_code = custom_args.get("trust_remote_code")
    trust_remote_code = (
        bool(model_kwargs.get("trust_remote_code", False))
        if configured_trust_remote_code is None
        else bool(configured_trust_remote_code)
    )
    model_config = _load_hf_config(
        pretrained_path,
        trust_remote_code=trust_remote_code,
        revision=model_kwargs.get("revision"),
        token=model_kwargs.get("token"),
    )
    model_type = getattr(model_config, "model_type", None)
    if model_type != _ROUTER_DIAGNOSTIC_MODEL_TYPE:
        raise ValueError(
            "capture_router_diagnostics currently supports only model_type="
            f"{_ROUTER_DIAGNOSTIC_MODEL_TYPE!r}, got {model_type!r}"
        )


def _run_hf_shape_diagnostic(
    hf_model: torch.nn.Module,
    input_ids: list[int],
    device: torch.device,
    base_logits: torch.Tensor,
    *,
    artifact_dir: Path,
    config: _ShapeDiagnosticConfig,
    gate_sequence_length: int | None,
    capture_router_diagnostics: bool,
    phase: Literal["phase_0", "phase_3"],
) -> None:
    """Run informational HF-full-prefix versus HF-standalone forwards.

    Args:
        hf_model: Loaded vanilla-HF model used for standalone forwards.
        input_ids: Token IDs for the full parity prompt.
        device: Device holding ``hf_model`` and its inputs.
        base_logits: Full-forward logits of shape [batch, sequence, vocab].
        artifact_dir: Directory that owns checkpoint-robustness artifacts.
        config: Validated standalone-forward lengths.
        gate_sequence_length: Optional standing cross-framework gate length.
        capture_router_diagnostics: Whether to capture router selections for each standalone forward.
        phase: Checkpoint phase that owns the diagnostic report.
    """
    diagnostic_lengths = config.lengths(
        parity_sequence_length=len(input_ids),
        gate_sequence_length=gate_sequence_length,
    )
    if not diagnostic_lengths:
        raise ValueError("Shape diagnostic has no standalone-forward lengths")

    standalone_logits: dict[int, torch.Tensor] = {}
    router_summaries: dict[int, dict[str, object]] = {}
    hf_router_capture = _router_diagnostic_paths(artifact_dir)[0]
    for sequence_length in diagnostic_lengths:
        router_capture_path = _shape_router_capture_path(artifact_dir, sequence_length)
        with _router_diagnostic_capture_context(
            hf_model,
            artifact_dir,
            framework="hf",
            enabled=capture_router_diagnostics,
            output_path=router_capture_path,
        ):
            candidate_logits = _get_logits(hf_model, input_ids[:sequence_length], device)
        standalone_logits[sequence_length] = candidate_logits
        if capture_router_diagnostics:
            from tests.functional_tests.checkpoint_robustness.router_diagnostics import (
                summarize_glm_router_shape_captures,
            )

            router_summaries[sequence_length] = summarize_glm_router_shape_captures(
                base_path=hf_router_capture,
                standalone_path=router_capture_path,
                reference_logits=base_logits[..., :sequence_length, :],
                candidate_logits=candidate_logits,
            )

    report = _build_shape_diagnostic_report(
        base_logits,
        standalone_logits,
        parity_document_sha256=_PARITY_DOCUMENT_SHA256,
        phase=phase,
        gate_sequence_length=gate_sequence_length,
        sweep_lengths=config.sweep_lengths,
        router_diagnostics=router_summaries or None,
    )
    _persist_shape_diagnostic_report(report, _shape_diagnostic_report_path(artifact_dir, phase))


def _source_load_artifact_paths(cfg) -> tuple[Path, Path]:
    """Return the persisted Phase 0 reference-logit and metadata paths."""
    artifact_dir = _robustness_artifact_dir(cfg)
    return artifact_dir / "source_load_reference_logits.pt", artifact_dir / "source_load_reference_metadata.json"


def _wait_for_source_load_artifacts(reference_path: Path, metadata_path: Path, fail_path: Path) -> None:
    """Wait for rank 0 to publish the isolated Phase 0 reference artifacts."""
    timeout_s = int(os.environ.get("SOURCE_LOAD_PARITY_TIMEOUT_SECONDS", "1800"))
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if fail_path.exists():
            raise RuntimeError(f"Rank 0 source-load artifact publication failed:\n{fail_path.read_text()}")
        if reference_path.exists() and metadata_path.exists():
            return
        time.sleep(5)
    raise TimeoutError(f"Timed out waiting {timeout_s}s for rank 0 source-load artifact publication")


def _disable_distributed_atexit_teardown() -> None:
    """Let process exit reclaim distributed resources after an isolated phase."""
    import atexit

    from nemo_automodel.components.distributed.init_utils import destroy_global_state

    atexit.unregister(destroy_global_state)


def _raise_distributed_failure(failure_message: str | None) -> None:
    """Raise the same phase failure on every initialized distributed rank."""
    if dist.is_initialized():
        payload = [failure_message]
        dist.broadcast_object_list(payload, src=0)
        failure_message = payload[0]
    if failure_message is not None:
        if _preinit_global_rank() == 0:
            phase_marker = failure_message.partition("\n")[0]
            print(f"[checkpoint_robustness][phase-error] {phase_marker}", file=sys.stderr, flush=True)
        raise AssertionError(failure_message)


def _run_process_isolated_checkpoint_phase(
    phase: str,
    *,
    custom_args: dict[str, object],
    recipe_cls: type[BaseRecipe],
    hf_model_cls: type,
    input_ids_loader: Callable[[str | None], list[int]],
) -> None:
    """Run one large-model checkpoint phase and then return directly to the launcher.

    The launcher starts every phase in a fresh distributed process group. This
    mirrors a real checkpoint restart and avoids relying on Python object
    destruction to release an entire trainer graph before constructing another.

    Args:
        phase: Isolated checkpoint phase selected by the launcher.
        custom_args: Test-specific values extracted from the resolved recipe.
        recipe_cls: Recipe class used for training, reload, and resume.
        hf_model_cls: Hugging Face auto-model class used for vanilla-HF reload.
        input_ids_loader: Domain-specific prompt tokenizer.
    """
    supported_phases = {
        "source_load_reference",
        "source_load_parity",
        "train_and_save",
        "automodel_reload",
        "hf_reload",
        "resume",
        "cross_tp_reload",
    }
    if phase not in supported_phases:
        raise ValueError(f"Unsupported isolated checkpoint phase {phase!r}; expected one of {sorted(supported_phases)}")
    if custom_args.get("skip_resume", False) and phase == "resume":
        raise ValueError(f"Process-isolated phase {phase!r} conflicts with skip_resume=true")
    if phase == "cross_tp_reload" and int(custom_args.get("cross_tp_size", "0")) <= 0:
        raise ValueError("Process-isolated cross_tp_reload requires cross_tp_size > 0")

    _disable_distributed_atexit_teardown()
    cfg = parse_args_and_load_config()
    _validate_router_diagnostic_config(cfg, custom_args)
    tokenizer_name = custom_args.get("tokenizer_name", None)
    parity_sequence_length = int(custom_args.get("parity_sequence_length", "2048"))

    if phase == "source_load_reference":
        if not custom_args.get("source_load_parity_enabled", False):
            raise ValueError("Isolated source_load_reference requires Phase 0 to be enabled")

        reference_path, metadata_path = _source_load_artifact_paths(cfg)
        source_load_fail_path = _source_load_sync_paths(cfg)[2] if _preinit_world_size() > 1 else None
        if _preinit_global_rank() == 0:
            for path in (
                reference_path,
                reference_path.with_suffix(".tmp"),
                metadata_path,
                metadata_path.with_suffix(".tmp"),
            ):
                path.unlink(missing_ok=True)

        _report_phase("Isolated Phase 0a source load: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )
        _report_phase("Isolated Phase 0a source load: starting vanilla-HF reference load")
        source_load_reference = _prepare_source_load_reference(
            cfg,
            input_ids,
            hf_model_cls=hf_model_cls,
            trust_remote_code=custom_args.get("trust_remote_code"),
            experts_implementation=custom_args.get("experts_implementation", None),
            hf_device_map_auto=bool(custom_args.get("hf_device_map_auto", False)),
            hf_device_map_max_memory_gib=custom_args.get("hf_device_map_max_memory_gib"),
            hf_device_map_cpu_max_memory_gib=custom_args.get("hf_device_map_cpu_max_memory_gib"),
            hf_source_post_load_dequantize=bool(custom_args.get("hf_source_post_load_dequantize", False)),
            parity_tolerance_profile=_comparison_profile(custom_args, "source_load"),
            capture_router_diagnostics=bool(custom_args.get("capture_router_diagnostics", False)),
            shape_diagnostic=custom_args.get("shape_diagnostic"),
            cross_framework_gate_sequence_length=(
                int(custom_args["cross_framework_gate_sequence_length"])
                if "cross_framework_gate_sequence_length" in custom_args
                else None
            ),
        )
        if _preinit_global_rank() == 0:
            assert source_load_reference is not None, "rank 0 source-load reference was not captured"
            reference_logits, hf_aliased, explicit_tie_word_embeddings = source_load_reference
            token_count, vocab_size = _validate_logits(reference_logits)
            print(
                f"[Phase 0] Vanilla-HF source forward produced finite logits for "
                f"{token_count} tokens and vocab_size={vocab_size}"
            )
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_reference_path = reference_path.with_suffix(".tmp")
            temporary_metadata_path = metadata_path.with_suffix(".tmp")
            try:
                torch.save(reference_logits, temporary_reference_path)
                temporary_reference_path.replace(reference_path)
                temporary_metadata_path.write_text(
                    json.dumps(
                        {
                            "explicit_tie_word_embeddings": explicit_tie_word_embeddings,
                            "hf_aliased": hf_aliased,
                        },
                        sort_keys=True,
                    )
                )
                temporary_metadata_path.replace(metadata_path)
            except Exception:
                if source_load_fail_path is not None:
                    source_load_fail_path.write_text(traceback.format_exc())
                raise
        else:
            assert source_load_fail_path is not None
            _wait_for_source_load_artifacts(reference_path, metadata_path, source_load_fail_path)
        _report_phase("Isolated Phase 0a source load: reference persisted; exiting phase")
        return

    if phase == "source_load_parity":
        if not custom_args.get("source_load_parity_enabled", False):
            raise ValueError("Isolated source_load_parity requires Phase 0 to be enabled")

        _report_phase("Isolated Phase 0b source parity: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )
        reference_path, metadata_path = _source_load_artifact_paths(cfg)
        assert reference_path.exists(), f"Source-load reference logits not found at {reference_path}"
        assert metadata_path.exists(), f"Source-load reference metadata not found at {metadata_path}"

        source_trainer = recipe_cls(cfg)
        source_trainer.setup()
        _report_phase("Isolated Phase 0b source parity: trainer setup complete; starting parity forward")
        if tokenizer_name is not None and dist.is_initialized() and dist.get_world_size() > 1:
            _barrier()
            if _rank0():
                _cleanup_input_ids_sync(cfg)
            _barrier()

        device = next(source_trainer.model_parts[0].parameters()).device
        with _router_diagnostic_capture_context(
            source_trainer.model_parts[0],
            _robustness_artifact_dir(cfg),
            framework="automodel",
            enabled=bool(custom_args.get("capture_router_diagnostics", False)),
        ):
            trainer_source_logits = _get_logits(
                source_trainer.model_parts[0],
                input_ids,
                device,
                trainer=source_trainer,
            )
        source_load_reference = None
        if _rank0():
            metadata = json.loads(metadata_path.read_text())
            source_load_reference = (
                torch.load(reference_path, map_location="cpu", weights_only=True),
                metadata["hf_aliased"],
                metadata["explicit_tie_word_embeddings"],
            )
        source_load_failure = _compare_source_load_parity(
            source_load_reference,
            trainer_source_logits,
            _lm_head_embedding_aliased(source_trainer.model_parts[0]),
            artifact_dir=_robustness_artifact_dir(cfg),
            policy=_source_load_parity_policy(custom_args),
        )
        _barrier()
        if _rank0():
            _cleanup_source_load_sync(cfg)
        _barrier()
        if source_load_failure is not None:
            source_load_failure = (
                "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE "
                f"phase=source_load_parity check=source_load_parity\n{source_load_failure}"
            )
        _raise_distributed_failure(source_load_failure)
        _report_phase("Isolated Phase 0b source parity: comparison complete; exiting phase")
        return

    if phase == "hf_reload":
        if custom_args.get("skip_hf_reload", False):
            raise ValueError("Process-isolated hf_reload conflicts with skip_hf_reload=true")

        _report_phase("Isolated Phase 3 vanilla-HF export reload: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )

        # The HF model is sharded by one rank-0 process over all GPUs on its
        # node. A CPU process group keeps the remaining workers synchronized
        # without reserving CUDA memory or starting a long NCCL wait.
        dist.init_process_group(
            backend="gloo",
            timeout=timedelta(minutes=cfg.get("dist_env", {}).get("timeout_minutes", 1)),
        )
        if tokenizer_name is not None:
            _barrier()
            if _rank0():
                _cleanup_input_ids_sync(cfg)
            _barrier()

        reference_path = _robustness_artifact_dir(cfg) / "reference_logits.pt"
        hf_reload_timeout_s = (
            int(custom_args["hf_reload_timeout_seconds"]) if "hf_reload_timeout_seconds" in custom_args else None
        )
        hf_reload_sync_paths = _prepare_hf_reload_sync(cfg, timeout_s=hf_reload_timeout_s)
        hf_reload_error = None
        if _rank0():
            if not reference_path.exists():
                hf_reload_error = f"Reference logits not found at {reference_path}"
            else:
                reference_logits = torch.load(reference_path, map_location="cpu", weights_only=True)
                hf_reload_error = _run_vanilla_hf_reload(
                    cfg,
                    input_ids,
                    reference_logits,
                    hf_model_cls=hf_model_cls,
                    custom_args=custom_args,
                )
        hf_reload_error = _finish_hf_reload_sync(hf_reload_sync_paths, hf_reload_error)
        if hf_reload_error is not None:
            hf_reload_error = (
                f"CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=hf_reload check=hf_reload_parity\n{hf_reload_error}"
            )
        _raise_distributed_failure(hf_reload_error)
        _report_phase("Isolated Phase 3 vanilla-HF export reload: parity complete; exiting phase")
        return

    if phase == "train_and_save":
        _report_phase("Isolated Phase 1 train/save/reference: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )
        resume_plan = None
        if custom_args.get("resume_enabled", False):
            resume_plan = _resume_plan_from_config(cfg)
            _configure_uninterrupted_run(cfg, resume_plan)

        torch.cuda.reset_peak_memory_stats()
        _report_phase("Isolated Phase 1 train/save/reference: starting trainer setup")
        trainer = recipe_cls(cfg)
        trainer.setup()
        resume_recorder = None
        if resume_plan is not None:
            resume_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=True)
            resume_recorder.attach(trainer)
        reproducibility_recorder = None
        reproducibility_dir = os.environ.get("AUTOMODEL_REPRODUCIBILITY_DIR")
        if reproducibility_dir is not None:
            reproducibility_recorder = _TrainingReproducibilityRecorder(trainer)
            reproducibility_recorder.attach()
        _report_phase("Isolated Phase 1 train/save/reference: trainer setup complete")
        if tokenizer_name is not None and dist.is_initialized() and dist.get_world_size() > 1:
            _barrier()
            if _rank0():
                _cleanup_input_ids_sync(cfg)
            _barrier()

        _report_phase("Isolated Phase 1 train/save/reference: starting training and checkpoint")
        trainer.run_train_validation_loop()
        _report_phase("Isolated Phase 1 train/save/reference: training and checkpoint complete")

        if resume_recorder is not None:
            _persist_reference_trajectory(resume_recorder)
            _barrier()
            if _rank0():
                print("[Resume correctness] Persisted the uninterrupted post-checkpoint trajectory")
        if reproducibility_recorder is not None:
            artifact_dir = Path(reproducibility_dir)
            _persist_training_reproducibility(
                reproducibility_recorder,
                artifact_dir,
                lifecycle="checkpoint",
            )
            _barrier()
            _report_training_reproducibility(
                artifact_dir,
                reproducibility_recorder,
                loss_threshold=float(custom_args.get("training_reproducibility_loss_threshold", "5e-2")),
            )

        peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
        peak_cpu_gb = _rss_gb()
        if _rank0():
            print(f"\n[Memory] Peak VRAM: {peak_vram_gb:.2f} GB, Peak CPU RSS: {peak_cpu_gb:.2f} GB")
        max_vram_gb = float(custom_args.get("max_vram_gb", "0"))
        max_cpu_gb = float(custom_args.get("max_cpu_gb", "0"))
        if max_vram_gb > 0:
            assert peak_vram_gb <= max_vram_gb, (
                f"Peak VRAM {peak_vram_gb:.2f} GB exceeds threshold {max_vram_gb:.2f} GB"
            )
        if max_cpu_gb > 0:
            assert peak_cpu_gb <= max_cpu_gb, f"Peak CPU RSS {peak_cpu_gb:.2f} GB exceeds threshold {max_cpu_gb:.2f} GB"

        _report_phase("Isolated Phase 1 train/save/reference: capturing reference logits")
        device = next(trainer.model_parts[0].parameters()).device
        reference_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
        token_count, vocab_size = _validate_logits(reference_logits)
        repeated_reference_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
        if _rank0():
            _compare_logits(
                _robustness_artifact_dir(cfg),
                reference_logits,
                repeated_reference_logits,
                _repeatability_policy(
                    phase="phase_1",
                    comparison="automodel_reference_self_repeat",
                    profile=str(custom_args.get("parity_tolerance_profile", "standard")),
                ),
            )
            print(
                f"[Phase 1] Reference forward produced finite logits for "
                f"{token_count} tokens and vocab_size={vocab_size}"
            )
        del repeated_reference_logits
        _checkpoint_paths(cfg)
        artifact_dir = _robustness_artifact_dir(cfg)
        if _rank0():
            artifact_dir.mkdir(parents=True, exist_ok=True)
            torch.save(reference_logits, artifact_dir / "reference_logits.pt")
        _barrier()
        if hasattr(cfg, "peft"):
            rank = dist.get_rank() if dist.is_initialized() else 0
            trainable_digests = _trainable_parameter_digests(trainer.model_parts)
            (artifact_dir / f"trainable_parameter_digests_rank_{rank}.json").write_text(
                json.dumps(trainable_digests, sort_keys=True)
            )
            _barrier()
        _report_phase("Isolated Phase 1 train/save/reference: reference artifacts persisted; exiting phase")
        return

    if phase == "automodel_reload":
        _report_phase("Isolated Phase 2 AutoModel model reload: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )
        checkpoint_dir, ckpt_step_dir, consolidated_dir = _checkpoint_paths(cfg)
        reference_path = _robustness_artifact_dir(cfg) / "reference_logits.pt"
        assert reference_path.exists(), f"Reference logits not found at {reference_path}"
        is_peft = hasattr(cfg, "peft")

        if custom_args.get("check_phantom_keys", False) and _preinit_global_rank() == 0:
            from safetensors import safe_open

            assert consolidated_dir.exists(), f"Phantom key check: {consolidated_dir} does not exist"
            sf_files = sorted(consolidated_dir.glob("*.safetensors"))
            assert sf_files, f"Phantom key check: no .safetensors files in {consolidated_dir}"
            for sf_path in sf_files:
                with safe_open(str(sf_path), framework="pt") as f:
                    for key in f.keys():
                        assert "_blocks" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"
                        assert "_scales" not in key, f"Phantom mxfp4 key leaked: {key} in {sf_path.name}"

        if not is_peft:
            _prepare_consolidated_hf_cache_once(cfg, consolidated_dir)
            _set_model_pretrained_path(cfg.model, consolidated_dir)
            cfg.checkpoint.enabled = False

        _report_phase("Isolated Phase 2 AutoModel model reload: starting trainer setup")
        restored_trainer = recipe_cls(cfg)
        restored_trainer.setup()
        _report_phase("Isolated Phase 2 AutoModel model reload: trainer setup complete")
        if tokenizer_name is not None and dist.is_initialized() and dist.get_world_size() > 1:
            _barrier()
            if _rank0():
                _cleanup_input_ids_sync(cfg)
            _barrier()

        device = next(restored_trainer.model_parts[0].parameters()).device
        restored_logits = _get_logits(
            restored_trainer.model_parts[0],
            input_ids,
            device,
            trainer=restored_trainer,
        )
        if is_peft:
            # Capture both sides after a forward. FSDP may change a parameter's
            # rank-local view during its first unshard/reshard lifecycle, so
            # hashing the restored model immediately after setup is not
            # comparable to the post-forward training reference.
            artifact_dir = _robustness_artifact_dir(cfg)
            rank = dist.get_rank() if dist.is_initialized() else 0
            expected_digests_path = artifact_dir / f"trainable_parameter_digests_rank_{rank}.json"
            local_digest_failure = None
            if not expected_digests_path.exists():
                local_digest_failure = f"rank {rank}: missing trainable-parameter digest {expected_digests_path}"
            else:
                expected_digests = json.loads(expected_digests_path.read_text())
                restored_digests = _trainable_parameter_digests(restored_trainer.model_parts)
                if restored_digests != expected_digests:
                    missing = sorted(set(expected_digests) - set(restored_digests))
                    unexpected = sorted(set(restored_digests) - set(expected_digests))
                    mismatched = sorted(
                        key
                        for key in set(expected_digests) & set(restored_digests)
                        if expected_digests[key] != restored_digests[key]
                    )
                    local_digest_failure = (
                        f"rank {rank}: trainable parameters differ after reload; missing={missing[:5]}, "
                        f"unexpected={unexpected[:5]}, mismatched={mismatched[:5]}"
                    )

            digest_failures = [local_digest_failure]
            if dist.is_initialized():
                digest_failures = [None] * dist.get_world_size()
                dist.all_gather_object(digest_failures, local_digest_failure)
            failure_message = None
            if _rank0():
                failures = [failure for failure in digest_failures if failure is not None]
                if failures:
                    failure_message = (
                        "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE "
                        "phase=automodel_reload check=trainable_parameter_fingerprint\n"
                        "Trainable PEFT parameter fingerprint mismatch:\n" + "\n".join(failures)
                    )
                else:
                    print(
                        f"[Isolated AutoModel reload] exact trainable-parameter fingerprints matched on "
                        f"{len(digest_failures)} ranks"
                    )
            _raise_distributed_failure(failure_message)

        reload_policy = _automodel_reload_parity_policy(custom_args)
        failure_message = None
        if _rank0():
            reference_logits = torch.load(reference_path, map_location="cpu", weights_only=True)
            failure_message = _compare_logits(
                _robustness_artifact_dir(cfg),
                reference_logits,
                restored_logits,
                reload_policy,
            )
        failure_message = _broadcast_rank0_failure(failure_message)
        if reload_policy.profile == "relaxed" or not reload_policy.enforce or failure_message is not None:
            repeated_restored_logits = _get_logits(
                restored_trainer.model_parts[0],
                input_ids,
                device,
                trainer=restored_trainer,
            )
            if _rank0():
                _compare_logits(
                    _robustness_artifact_dir(cfg),
                    restored_logits,
                    repeated_restored_logits,
                    _repeatability_policy(
                        phase="phase_2",
                        comparison="automodel_reload_self_repeat",
                        profile=reload_policy.profile,
                    ),
                )
            del repeated_restored_logits
        if failure_message is not None:
            failure_message = (
                "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=automodel_reload check=full_logit_parity\n" + failure_message
            )
        _raise_distributed_failure(failure_message)
        _report_phase(
            f"Isolated AutoModel reload: parity complete for {ckpt_step_dir.relative_to(checkpoint_dir)}; exiting phase"
        )
        return

    if phase == "cross_tp_reload":
        if hasattr(cfg, "peft"):
            raise ValueError("Process-isolated cross_tp_reload does not support PEFT checkpoints")
        _report_phase("Isolated Phase 5 cross-TP reload: loading prompt input IDs")
        input_ids = _load_input_ids_once(
            cfg,
            input_ids_loader,
            tokenizer_name,
            sequence_length=parity_sequence_length,
        )
        checkpoint_dir, ckpt_step_dir, consolidated_dir = _checkpoint_paths(cfg)
        reference_path = _robustness_artifact_dir(cfg) / "reference_logits.pt"
        if not reference_path.exists():
            raise FileNotFoundError(f"Reference logits not found at {reference_path}")

        _prepare_consolidated_hf_cache_once(cfg, consolidated_dir)
        _set_model_pretrained_path(cfg.model, consolidated_dir)
        cfg.checkpoint.enabled = False
        cfg.distributed.tp_size = int(custom_args["cross_tp_size"])
        cfg.distributed.dp_size = None

        _report_phase("Isolated Phase 5 cross-TP reload: starting trainer setup")
        cross_tp_trainer = recipe_cls(cfg)
        cross_tp_trainer.setup()
        _report_phase("Isolated Phase 5 cross-TP reload: trainer setup complete")
        if tokenizer_name is not None and dist.is_initialized() and dist.get_world_size() > 1:
            _barrier()
            if _rank0():
                _cleanup_input_ids_sync(cfg)
            _barrier()

        device = next(cross_tp_trainer.model_parts[0].parameters()).device
        cross_tp_logits = _get_logits(
            cross_tp_trainer.model_parts[0],
            input_ids,
            device,
            trainer=cross_tp_trainer,
        )
        failure_message = None
        if _rank0():
            reference_logits = torch.load(reference_path, map_location="cpu", weights_only=True)
            failure_message = _compare_logits(
                _robustness_artifact_dir(cfg),
                reference_logits,
                cross_tp_logits,
                _cross_tp_parity_policy(custom_args),
            )
            if failure_message is not None:
                failure_message = (
                    "CHECKPOINT_ROBUSTNESS_PHASE_FAILURE phase=cross_tp_reload check=full_logit_parity\n"
                    + failure_message
                )
        _raise_distributed_failure(failure_message)
        _release_recipe_memory(cross_tp_trainer)
        _report_phase(
            f"Isolated Phase 5 cross-TP reload: parity complete for "
            f"{ckpt_step_dir.relative_to(checkpoint_dir)}; exiting phase"
        )
        return

    resume_plan = _resume_plan_from_config(cfg)
    reference_trajectory = _load_reference_trajectory(resume_plan)
    checkpoint_path = _checkpoint_for_completed_steps(resume_plan, resume_plan.boundary_step)
    _configure_resumed_run(cfg, resume_plan, checkpoint_path)

    _report_phase("Isolated Phase 4 native resume: starting setup and optimizer checkpoint load")
    resume_trainer = recipe_cls(cfg)
    resume_trainer.setup()
    # The restore is complete. Phase 4 validates the continuation but does not
    # need to publish another large distributed checkpoint at its final step.
    _disable_checkpoint_saves_after_restore(resume_trainer)
    restored_state = _checkpoint_state_snapshot(resume_trainer, state_is_being_saved=False)
    local_failure = _restored_state_mismatch(reference_trajectory["boundary_state"], restored_state)
    failure_message = _gather_rank_failures(local_failure, check="restored_state")
    _raise_distributed_failure(failure_message)
    if _rank0():
        print(
            "[Resume correctness] Optimizer counters/groups, LR scheduler, and RNG matched exactly after load; "
            "model parameters and optimizer tensors are compared at the first pre-update point, after both "
            "branches have entered the same FSDP lifecycle"
        )

    resume_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=False)
    resume_recorder.attach(resume_trainer)
    _report_phase("Isolated Phase 4 native resume: checkpoint state verified; starting shared-trajectory continuation")
    resume_trainer.run_train_validation_loop()
    _report_phase("Isolated Phase 4 native resume: training complete")

    resumed_trajectory = resume_recorder.to_dict()
    resume_tolerance = _resolve_resume_loss_tolerance(
        custom_args.get("resume_tolerance_profile", "standard"),
        first_step_override=custom_args.get("resume_first_loss_threshold"),
        later_step_override=custom_args.get("resume_loss_threshold"),
    )
    comparison_report = _report_resume_comparison(
        resume_plan,
        reference_trajectory,
        resumed_trajectory,
        resume_tolerance,
    )
    local_failure = comparison_report["blocking_failure"]
    failure_message = _gather_rank_failures(local_failure, check="shared_trajectory")
    _raise_distributed_failure(failure_message)
    _report_phase("Isolated Phase 4 native resume: shared-trajectory checkpoint continuation verified; exiting phase")


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
    isolated_phase = custom_args.get("isolated_phase")
    if isolated_phase is not None:
        _run_process_isolated_checkpoint_phase(
            str(isolated_phase),
            custom_args=custom_args,
            recipe_cls=recipe_cls,
            hf_model_cls=hf_model_cls,
            input_ids_loader=input_ids_loader,
        )
        return
    cross_tp_size = int(custom_args.get("cross_tp_size", "0"))
    trust_remote_code = custom_args.get("trust_remote_code")
    experts_implementation = custom_args.get("experts_implementation", None)
    tokenizer_name = custom_args.get("tokenizer_name", None)
    parity_sequence_length = int(custom_args.get("parity_sequence_length", "2048"))
    max_vram_gb = float(custom_args.get("max_vram_gb", "0"))
    max_cpu_gb = float(custom_args.get("max_cpu_gb", "0"))
    check_phantom_keys = bool(custom_args.get("check_phantom_keys", False))
    resume_enabled = bool(custom_args.get("resume_enabled", False))
    resume_tolerance = _resolve_resume_loss_tolerance(
        custom_args.get("resume_tolerance_profile", "standard"),
        first_step_override=custom_args.get("resume_first_loss_threshold"),
        later_step_override=custom_args.get("resume_loss_threshold"),
    )
    training_reproducibility_loss_threshold = float(custom_args.get("training_reproducibility_loss_threshold", "5e-2"))
    hf_device_map_auto = bool(custom_args.get("hf_device_map_auto", False))
    hf_source_post_load_dequantize = bool(custom_args.get("hf_source_post_load_dequantize", False))
    skip_hf_reload = bool(custom_args.get("skip_hf_reload", False))
    source_load_parity_enabled = bool(custom_args.get("source_load_parity_enabled", False))
    deferred_failures: list[str] = []

    cfg = parse_args_and_load_config()
    _validate_router_diagnostic_config(cfg, custom_args)
    resume_plan = _resume_plan_from_config(cfg) if resume_enabled else None
    if resume_plan is not None:
        _configure_uninterrupted_run(cfg, resume_plan)
    input_ids = _load_input_ids_once(
        cfg,
        input_ids_loader,
        tokenizer_name,
        sequence_length=parity_sequence_length,
    )

    source_load_reference = None
    if source_load_parity_enabled:
        _report_phase("Phase 0: starting vanilla-HF source-load reference")
        source_load_reference = _prepare_source_load_reference(
            cfg,
            input_ids,
            hf_model_cls=hf_model_cls,
            trust_remote_code=trust_remote_code,
            experts_implementation=experts_implementation,
            hf_device_map_auto=hf_device_map_auto,
            hf_device_map_max_memory_gib=custom_args.get("hf_device_map_max_memory_gib"),
            hf_device_map_cpu_max_memory_gib=custom_args.get("hf_device_map_cpu_max_memory_gib"),
            hf_source_post_load_dequantize=hf_source_post_load_dequantize,
            parity_tolerance_profile=_comparison_profile(custom_args, "source_load"),
            capture_router_diagnostics=bool(custom_args.get("capture_router_diagnostics", False)),
            shape_diagnostic=custom_args.get("shape_diagnostic"),
            cross_framework_gate_sequence_length=(
                int(custom_args["cross_framework_gate_sequence_length"])
                if "cross_framework_gate_sequence_length" in custom_args
                else None
            ),
        )
        _barrier()
        _report_phase("Phase 0: vanilla-HF source-load reference complete")

    # Phase 1: Construct the model, optionally compare it against the raw HF
    # source-load reference, then train and checkpoint.
    torch.cuda.reset_peak_memory_stats()
    _report_phase("Phase 1: starting initial trainer setup")
    trainer = recipe_cls(cfg)
    trainer.setup()
    _report_phase("Phase 1: initial trainer setup complete")
    if tokenizer_name is not None and dist.is_initialized() and dist.get_world_size() > 1:
        _barrier()
        if _rank0():
            _cleanup_input_ids_sync(cfg)
        _barrier()

    if source_load_parity_enabled:
        _report_phase("Phase 0: starting constructed-trainer parity forward")
        device = next(trainer.model_parts[0].parameters()).device
        with _router_diagnostic_capture_context(
            trainer.model_parts[0],
            _robustness_artifact_dir(cfg),
            framework="automodel",
            enabled=bool(custom_args.get("capture_router_diagnostics", False)),
        ):
            trainer_source_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
        source_load_failure = _compare_source_load_parity(
            source_load_reference,
            trainer_source_logits,
            _lm_head_embedding_aliased(trainer.model_parts[0]),
            artifact_dir=_robustness_artifact_dir(cfg),
            policy=_source_load_parity_policy(custom_args),
        )
        _record_deferred_failure(deferred_failures, "Phase 0 source-load parity", source_load_failure)
        del trainer_source_logits, source_load_reference
        _barrier()
        if _rank0():
            _cleanup_source_load_sync(cfg)
        _barrier()
        _report_phase("Phase 0: constructed-trainer parity forward complete")

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
        if resume_plan is not None:
            _configure_uninterrupted_run(cfg, resume_plan)
        _report_phase("Phase 1: starting fresh trainer setup after source parity")
        trainer = recipe_cls(cfg)
        trainer.setup()
        _report_phase("Phase 1: fresh trainer setup complete")

    resume_recorder = None
    if resume_plan is not None:
        resume_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=True)
        resume_recorder.attach(trainer)
    reproducibility_recorder = None
    reproducibility_dir = os.environ.get("AUTOMODEL_REPRODUCIBILITY_DIR")
    if reproducibility_dir is not None:
        reproducibility_recorder = _TrainingReproducibilityRecorder(trainer)
        reproducibility_recorder.attach()
    _report_phase("Phase 1: starting train and checkpoint")
    trainer.run_train_validation_loop()
    _report_phase("Phase 1: train and checkpoint complete")
    if resume_recorder is not None:
        _persist_reference_trajectory(resume_recorder)
        _barrier()
        if _rank0():
            print(
                "[Resume correctness] Phase 1 continued through the comparison steps from the exact "
                "checkpoint-producing trajectory"
            )
    if reproducibility_recorder is not None:
        artifact_dir = Path(reproducibility_dir)
        _persist_training_reproducibility(
            reproducibility_recorder,
            artifact_dir,
            lifecycle="checkpoint",
        )
        _barrier()
        _report_training_reproducibility(
            artifact_dir,
            reproducibility_recorder,
            loss_threshold=training_reproducibility_loss_threshold,
        )

    # Memory tracking after training
    peak_vram_gb = torch.cuda.max_memory_allocated() / 1024**3
    peak_cpu_gb = _rss_gb()
    if _rank0():
        print(f"\n[Memory] Peak VRAM: {peak_vram_gb:.2f} GB, Peak CPU RSS: {peak_cpu_gb:.2f} GB")
    if max_vram_gb > 0:
        assert peak_vram_gb <= max_vram_gb, f"Peak VRAM {peak_vram_gb:.2f} GB exceeds threshold {max_vram_gb:.2f} GB"
    if max_cpu_gb > 0:
        assert peak_cpu_gb <= max_cpu_gb, f"Peak CPU RSS {peak_cpu_gb:.2f} GB exceeds threshold {max_cpu_gb:.2f} GB"

    # Phase 1 also captures the reference distribution before teardown. It is
    # persisted by the isolated runner for the independent reload processes.
    _report_phase("Phase 1: starting reference-logits capture")
    device = next(trainer.model_parts[0].parameters()).device
    reference_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
    token_count, vocab_size = _validate_logits(reference_logits)
    repeated_reference_logits = _get_logits(trainer.model_parts[0], input_ids, device, trainer=trainer)
    if _rank0():
        _compare_logits(
            _robustness_artifact_dir(cfg),
            reference_logits,
            repeated_reference_logits,
            _repeatability_policy(
                phase="phase_1",
                comparison="automodel_reference_self_repeat",
                profile=str(custom_args.get("parity_tolerance_profile", "standard")),
            ),
        )
        print(
            f"[Phase 1] Reference forward produced finite logits for {token_count} tokens and vocab_size={vocab_size}"
        )
    del repeated_reference_logits
    _report_phase("Phase 1: reference-logits capture complete")

    # Locate the Phase 1 checkpoint used by the reload and resume checks.
    if resume_plan is not None:
        ckpt_step_dir = _checkpoint_for_completed_steps(resume_plan, resume_plan.final_max_steps)
    else:
        _, ckpt_step_dir, _ = _checkpoint_paths(cfg)
    consolidated_dir = ckpt_step_dir / "model" / "consolidated"

    is_peft = hasattr(cfg, "peft")

    _release_recipe_memory(trainer)
    del trainer

    # Phase 2: Reload AutoModel from the exported HF-format consolidated weights.
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
        _set_model_pretrained_path(cfg.model, consolidated_dir)
        cfg.checkpoint.enabled = False
    _report_phase("Phase 2: starting AutoModel model reload setup")
    restored_trainer = recipe_cls(cfg)
    restored_trainer.setup()
    _report_phase("Phase 2: AutoModel model reload setup complete")

    _report_phase("Phase 2: starting restored-logits capture")
    restored_logits = _get_logits(restored_trainer.model_parts[0], input_ids, device, trainer=restored_trainer)
    _report_phase("Phase 2: restored-logits capture complete")

    reload_policy = _automodel_reload_parity_policy(custom_args)
    automodel_reload_error = None
    if _rank0():
        automodel_reload_error = _compare_logits(
            _robustness_artifact_dir(cfg),
            reference_logits,
            restored_logits,
            reload_policy,
        )
    automodel_reload_error = _broadcast_rank0_failure(automodel_reload_error)
    if reload_policy.profile == "relaxed" or not reload_policy.enforce or automodel_reload_error is not None:
        repeated_restored_logits = _get_logits(
            restored_trainer.model_parts[0],
            input_ids,
            device,
            trainer=restored_trainer,
        )
        if _rank0():
            _compare_logits(
                _robustness_artifact_dir(cfg),
                restored_logits,
                repeated_restored_logits,
                _repeatability_policy(
                    phase="phase_2",
                    comparison="automodel_reload_self_repeat",
                    profile=reload_policy.profile,
                ),
            )
        del repeated_restored_logits
    _record_deferred_failure(deferred_failures, "Phase 2 AutoModel model reload parity", automodel_reload_error)

    _release_recipe_memory(restored_trainer)
    del restored_trainer

    # Phase 3: Load the same exported weights into vanilla HF (rank 0 only).
    _report_phase("Phase 3: starting vanilla-HF export reload")
    hf_reload_timeout_s = (
        int(custom_args["hf_reload_timeout_seconds"]) if "hf_reload_timeout_seconds" in custom_args else None
    )
    hf_reload_sync_paths = _prepare_hf_reload_sync(cfg, timeout_s=hf_reload_timeout_s)

    hf_reload_error = None
    if skip_hf_reload:
        if _rank0():
            print("[Phase 3] Skipped (ci.checkpoint_robustness.skip_hf_reload=true).")
    elif _rank0():
        hf_reload_error = _run_vanilla_hf_reload(
            cfg,
            input_ids,
            reference_logits,
            hf_model_cls=hf_model_cls,
            custom_args=custom_args,
        )

    hf_reload_error = _finish_hf_reload_sync(hf_reload_sync_paths, hf_reload_error)
    _record_deferred_failure(deferred_failures, "Phase 3 vanilla-HF export reload parity", hf_reload_error)
    _report_phase("Phase 3: vanilla-HF export reload complete")

    # Phase 4: restore the exact Phase 1 boundary and replay its continuation.
    if resume_enabled:
        assert resume_plan is not None
        reference_trajectory = _load_reference_trajectory(resume_plan)
        checkpoint_path = _checkpoint_for_completed_steps(resume_plan, resume_plan.boundary_step)
        cfg = parse_args_and_load_config()
        _configure_resumed_run(cfg, resume_plan, checkpoint_path)
        _report_phase("Phase 4: starting native-checkpoint resume setup and load")
        resume_trainer = recipe_cls(cfg)
        resume_trainer.setup()
        # The restore is complete. Phase 4 validates the continuation but does not
        # need to publish another large distributed checkpoint at its final step.
        _disable_checkpoint_saves_after_restore(resume_trainer)
        restored_state = _checkpoint_state_snapshot(resume_trainer, state_is_being_saved=False)
        local_failure = _restored_state_mismatch(reference_trajectory["boundary_state"], restored_state)
        failure_message = _gather_rank_failures(local_failure, check="restored_state")
        _raise_distributed_failure(failure_message)
        if _rank0():
            print(
                "[Resume correctness] Restored optimizer counters/groups, LR scheduler, and RNG exactly at the "
                "Phase 1 boundary; model parameters and optimizer tensors are compared at the first pre-update "
                "point after both branches have entered the same FSDP lifecycle"
            )

        resumed_recorder = _TrajectoryRecorder(resume_plan, capture_boundary_state=False)
        resumed_recorder.attach(resume_trainer)
        _report_phase("Phase 4: checkpoint state verified; starting shared-trajectory continuation")
        resume_trainer.run_train_validation_loop()
        _report_phase("Phase 4: resumed training complete")

        resumed_trajectory = resumed_recorder.to_dict()
        comparison_report = _report_resume_comparison(
            resume_plan,
            reference_trajectory,
            resumed_trajectory,
            resume_tolerance,
        )
        local_failure = comparison_report["blocking_failure"]
        failure_message = _gather_rank_failures(local_failure, check="shared_trajectory")
        _raise_distributed_failure(failure_message)
        if _rank0():
            print(
                f"[Resume correctness] Shared-trajectory continuation verified for "
                f"{resume_plan.continuation_steps} steps; profile={resume_tolerance.profile}, "
                f"first-step atol/rtol={resume_tolerance.first_step_atol:.3e}/"
                f"{resume_tolerance.first_step_rtol:.3e}, later-step atol/rtol="
                f"{resume_tolerance.later_step_atol:.3e}/{resume_tolerance.later_step_rtol:.3e}"
            )

        _release_recipe_memory(resume_trainer)
        del resume_trainer
        _barrier()
        _report_phase("Phase 4: resume comparison complete")

    # Phase 5 (optional): reload the exported weights with a different TP size.
    if cross_tp_size > 0 and not is_peft:
        _report_phase("Phase 5: starting optional cross-TP consolidated reload")
        cfg = parse_args_and_load_config()
        _set_model_pretrained_path(cfg.model, consolidated_dir)
        cfg.checkpoint.enabled = False
        cfg.distributed.tp_size = cross_tp_size
        cfg.distributed.dp_size = None
        cross_tp_trainer = recipe_cls(cfg)
        cross_tp_trainer.setup()

        cross_tp_logits = _get_logits(cross_tp_trainer.model_parts[0], input_ids, device, trainer=cross_tp_trainer)
        cross_tp_error = None
        if _rank0():
            cross_tp_error = _compare_logits(
                _robustness_artifact_dir(cfg),
                reference_logits,
                cross_tp_logits,
                _cross_tp_parity_policy(custom_args),
            )
        cross_tp_error = _broadcast_rank0_failure(cross_tp_error)
        _record_deferred_failure(deferred_failures, "Phase 5 cross-TP consolidated reload parity", cross_tp_error)

        _release_recipe_memory(cross_tp_trainer)
        del cross_tp_trainer
        _barrier()
        _report_phase("Phase 5: optional cross-TP consolidated reload complete")

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
    _report_phase("All enabled phases complete")


def test_checkpoint_robustness() -> None:
    """Run checkpoint robustness with the LLM finetune recipe."""
    from transformers import AutoModelForCausalLM

    from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction

    run_checkpoint_robustness(
        recipe_cls=TrainFinetuneRecipeForNextTokenPrediction,
        hf_model_cls=AutoModelForCausalLM,
    )


if __name__ == "__main__":
    test_checkpoint_robustness()
