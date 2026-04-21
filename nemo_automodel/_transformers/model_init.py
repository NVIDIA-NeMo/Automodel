# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Model resolution and initialization helpers.

Functions for resolving which model class to use (custom vs HF), downloading
weights, applying config overrides, and instantiating the model.
"""

import gc
import inspect
import json
import logging
import os
import threading
from contextlib import contextmanager

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

# For models that still accesses config.pad_token_id after v5 removes it in PretrainedConfig
if not hasattr(PretrainedConfig, "pad_token_id"):
    PretrainedConfig.pad_token_id = None

from nemo_automodel._transformers.utils import _patch_layer_types_validator, apply_qwen3_omni_config_patch

apply_qwen3_omni_config_patch()
# Relax transformers v5 strict ``layer_types`` validation early so that
# remote-code configs (e.g. nvidia/Nemotron-Flash-1B) with non-canonical
# taxonomies can be loaded by ``AutoConfig`` / ``AutoTokenizer`` before the
# recipe gets a chance to call ``apply_cache_compatibility_patches``.
_patch_layer_types_validator()

import nemo_automodel.components.checkpoint.utils as checkpoint_utils
import nemo_automodel.components.distributed.utils as dist_utils
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.distributed.init_utils import get_local_world_size_preinit, get_world_size_safe
from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin
from nemo_automodel.components.utils.model_utils import resolve_trust_remote_code, skip_random_init
from nemo_automodel.shared.utils import dtype_from_str

logger = logging.getLogger(__name__)

# Thread-local: when True, HF's get_init_context must not add torch.device("meta")
# so that model init runs on real device (used when retrying after "Cannot copy out of meta tensor").
_hf_meta_device_disabled = threading.local()


def _get_hf_meta_device_disabled():
    return getattr(_hf_meta_device_disabled, "value", False)


@contextmanager
def no_hf_meta_device():
    """Disable HuggingFace's meta device in get_init_context so model is built on real device."""
    prev = _get_hf_meta_device_disabled()
    _hf_meta_device_disabled.value = True
    try:
        yield
    finally:
        _hf_meta_device_disabled.value = prev


def _filter_meta_device_from_init_context(contexts):
    """Remove torch.device('meta') from HF init context list when we want real-device init."""
    return [c for c in contexts if not (isinstance(c, torch.device) and getattr(c, "type", None) == "meta")]


def _is_remote_code_class(cls) -> bool:
    """Return True if ``cls`` is a dynamically-loaded ``trust_remote_code`` class.

    Such classes live under ``transformers_modules.*`` (the HF module cache)
    and commonly perform ``.to(device)`` on meta tensors during ``__init__``,
    which explodes when HF wraps init with ``torch.device("meta")``.
    """
    mod = getattr(cls, "__module__", "") or ""
    return mod.startswith("transformers_modules.") or ".transformers_modules." in mod


def _patched_get_init_context(cls, *args, **kwargs):
    """Wrapper around PreTrainedModel.get_init_context that strips meta device when requested."""
    original = _patched_get_init_context.__wrapped__
    contexts = original(cls, *args, **kwargs)
    if _get_hf_meta_device_disabled() or _is_remote_code_class(cls):
        return _filter_meta_device_from_init_context(contexts)
    return contexts


# Bind original and install patch (classmethod-safe)
_original_get_init_context = PreTrainedModel.get_init_context.__func__
_patched_get_init_context.__wrapped__ = _original_get_init_context
PreTrainedModel.get_init_context = classmethod(_patched_get_init_context)


def _get_mixin_wrapped_class(model_class: type) -> type:
    """
    Get a class that combines HFCheckpointingMixin with the original model class.

    If the class already has the mixin, returns it unchanged.

    Args:
        model_class: The original model class (e.g., LlamaForCausalLM)

    Returns:
        A class that inherits from both HFCheckpointingMixin and model_class
    """
    # Custom models already inherit HFCheckpointingMixin
    if issubclass(model_class, HFCheckpointingMixin):
        return model_class

    # Create wrapper class that looks identical to original
    return type(
        model_class.__name__,
        (HFCheckpointingMixin, model_class),
        {
            "__module__": model_class.__module__,
            "__qualname__": model_class.__qualname__,
        },
    )


@contextmanager
def local_torch_dtype(
    dtype: torch.dtype, model_class_name: str | None = None, default_dtype: torch.dtype = torch.bfloat16
):
    """
    Locally change the torch default dtype to `dtype`, and restore the old one upon exiting the context.
    If `model_class_name` is provided, it's used to provide a more helpful error message if `dtype` is not valid.
    """
    # Just a more helping error before we set `torch.set_default_dtype` later on which would crash in this case
    if isinstance(dtype, str):
        dtype = default_dtype
    if not dtype.is_floating_point:
        if model_class_name is not None:
            error_message = (
                f"{model_class_name} cannot be instantiated under `dtype={dtype}` as it's not a floating-point dtype"
            )
        else:
            error_message = f"Cannot set `{dtype}` as torch's default as it's not a floating-point dtype"
        raise ValueError(error_message)
    original_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(original_dtype)


def _is_config_compatible_with_custom_model(arch_name: str, config) -> bool:
    """
    Check if a HuggingFace config is compatible with our custom model implementation.

    Some architectures (e.g., NemotronHForCausalLM) are shared between different model versions
    (v2 vs v3) but our custom implementation only supports specific versions. This function
    validates that the config has the required attributes for the custom implementation.

    Args:
        arch_name: The architecture name (e.g., "NemotronHForCausalLM")
        config: The HuggingFace config object

    Returns:
        True if the config is compatible with our custom implementation, False otherwise
    """
    # NemotronHForCausalLM: Our custom implementation is for v3 (MoE model)
    # v3 requires n_routed_experts, v2 does not have this attribute
    if arch_name == "NemotronHForCausalLM":
        return hasattr(config, "n_routed_experts") and config.n_routed_experts is not None

    # All other architectures are assumed compatible
    return True


def _resolve_custom_model_cls_for_config(config):
    """Resolve the custom model class for *config*, if the config is compatible."""
    architectures = get_architectures(config)
    if not architectures:
        return None

    arch_name = architectures[0]
    if not ModelRegistry.has_custom_model(arch_name):
        return None

    # Some architecture names are shared across multiple upstream variants.
    # Screen them here before asking the registry for the custom implementation.
    if not _is_config_compatible_with_custom_model(arch_name, config):
        return None

    return ModelRegistry.resolve_custom_model_cls(arch_name, config)


def get_hf_config(pretrained_model_name_or_path, attn_implementation, **kwargs):
    """
    Get the HF config for the model.
    """
    kwargs = kwargs.copy()
    trust_remote_code = kwargs.pop("trust_remote_code", resolve_trust_remote_code(pretrained_model_name_or_path))
    hf_config = kwargs.get("config", None)
    if hf_config is None:
        # Filter out nested dict kwargs before passing to AutoConfig.from_pretrained.
        # Nested dicts (e.g. text_config={"key": val}) would replace entire sub-configs
        # with incomplete dicts, losing all other fields. These nested overrides are
        # instead handled by _consume_config_overrides which deep-merges them.
        nested_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if isinstance(kwargs[k], dict)}  # noqa: F841
        try:
            hf_config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs,
                trust_remote_code=trust_remote_code,
                attn_implementation=attn_implementation,
            )
        except ValueError as e:
            err = str(e)
            if "does not recognize this architecture" in err:
                raise ValueError(
                    f"{e}\n\n"
                    f"The checkpoint '{pretrained_model_name_or_path}' has a model type not "
                    f"recognized by the installed version of NeMo Automodel. "
                    f"This usually means your installed package is out of date.\n\n"
                    f"To fix this, try upgrading:\n"
                    f"  pip install --upgrade nemo_automodel\n"
                    f"or install from source:\n"
                    f"  pip install git+https://github.com/NVIDIA-NeMo/Automodel.git"
                ) from e
            # Some upstream configs (e.g. stepfun-ai/Step-3.5-Flash) ship
            # layer_types longer than num_hidden_layers, which newer transformers
            # versions reject during config instantiation. Fix the raw dict and retry.
            if "num_hidden_layers" in err and ("layer_types" in err or "layer types" in err):
                hf_config = _load_config_with_layer_types_fix(
                    pretrained_model_name_or_path,
                    attn_implementation,
                    trust_remote_code=trust_remote_code,
                    **kwargs,
                )
            else:
                raise
    return hf_config


def _load_config_with_layer_types_fix(pretrained_model_name_or_path, attn_implementation, trust_remote_code, **kwargs):
    """Load an HF config after truncating ``layer_types`` to ``num_hidden_layers``.

    Works around buggy upstream configs whose ``layer_types`` list is longer than
    ``num_hidden_layers`` (e.g. stepfun-ai/Step-3.5-Flash).
    """
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    config_dict, _ = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
    n = config_dict.get("num_hidden_layers")
    lt = config_dict.get("layer_types")
    if isinstance(n, int) and isinstance(lt, list) and len(lt) > n:
        logger.warning(
            "Truncating layer_types (len=%d) to num_hidden_layers=%d for %s",
            len(lt),
            n,
            pretrained_model_name_or_path,
        )
        config_dict["layer_types"] = lt[:n]

    config_cls = None
    auto_map = config_dict.get("auto_map") or {}
    if trust_remote_code and "AutoConfig" in auto_map:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        config_cls = get_class_from_dynamic_module(auto_map["AutoConfig"], pretrained_model_name_or_path)
    if config_cls is None:
        model_type = config_dict.get("model_type")
        config_cls = CONFIG_MAPPING.get(model_type)
    if config_cls is None:
        raise ValueError(
            f"Could not resolve config class for {pretrained_model_name_or_path} "
            f"(model_type={config_dict.get('model_type')!r})"
        )
    return config_cls.from_dict(config_dict, attn_implementation=attn_implementation)


def get_is_hf_model(config, force_hf):
    """Determine whether the model should use the HF (not custom) implementation."""
    if force_hf:
        return True
    return _resolve_custom_model_cls_for_config(config) is None


def _download_model_weights(hf_config, pretrained_model_name_or_path):
    if not os.path.isdir(pretrained_model_name_or_path):
        if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
            logger.info(
                "HF_HUB_OFFLINE=1: skipping weight download for %s (using cached weights)",
                pretrained_model_name_or_path,
            )
            return
        num_nodes = (get_world_size_safe() % get_local_world_size_preinit()) + 1  # 1-indexed
        if num_nodes > 1:
            logger.info(
                "Downloading model weights on %d nodes. This incurs high storage usage. "
                "It is recommended to download once with `hf download` and pass in the "
                "downloaded path to the `pretrained_model_name_or_path` argument.",
                num_nodes,
            )
        # Import via module reference (vs bound name) so unit tests can patch
        # `nemo_automodel.components.distributed.utils.FirstRankPerNode`.
        with dist_utils.FirstRankPerNode():
            snapshot_download(pretrained_model_name_or_path)


def _setup_bnb_loading_kwargs(kwargs: dict) -> None:
    """Configure kwargs for HF from_pretrained to work with BitsAndBytes quantization.

    Sets ``device_map`` so HF loads+quantizes per-shard on the current GPU, and
    disables the async weight loader introduced in transformers v5 which can
    materialize many full-precision tensors concurrently before the quantizer
    runs, causing OOM on memory-constrained systems.
    """
    kwargs.setdefault("device_map", {"": torch.cuda.current_device()})
    prev = os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")
    if prev is None:
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
        logger.info("Set HF_DEACTIVATE_ASYNC_LOAD=1 for BnB-compatible synchronous weight loading.")
    logger.info("BnB loading: device_map=%s", kwargs["device_map"])


def _resolve_model_dir(pretrained_model_name_or_path: str) -> str:
    """Resolve a HF repo id or local path to a local directory with model files."""
    if os.path.isdir(pretrained_model_name_or_path):
        return pretrained_model_name_or_path
    return snapshot_download(pretrained_model_name_or_path, local_files_only=True)


def _has_safetensors(model_dir: str) -> bool:
    """Check whether a model directory contains safetensors checkpoint files."""
    if os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")):
        return True
    if os.path.exists(os.path.join(model_dir, "model.safetensors")):
        return True
    return False


def _stream_load_bnb_weights(model, model_dir, device, torch_dtype):
    """Load safetensor shards one-at-a-time, quantizing BnB Params4bit on the fly.

    Peak memory ≈ (accumulated quantized weights) + (one bf16 weight tensor)
    instead of (full bf16 model) with standard HF loading.
    """
    import bitsandbytes as bnb
    from safetensors import safe_open

    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        shard_files = list(dict.fromkeys(index["weight_map"].values()))
    else:
        shard_files = ["model.safetensors"]

    # Build name → (module, attr_name, param_or_buffer) index
    param_map: dict[str, tuple] = {}
    for name, param in model.named_parameters():
        parts = name.rsplit(".", 1)
        mod = model.get_submodule(parts[0]) if len(parts) == 2 else model
        param_map[name] = (mod, parts[-1], param)
    for name, buf in model.named_buffers():
        if name not in param_map:
            parts = name.rsplit(".", 1)
            mod = model.get_submodule(parts[0]) if len(parts) == 2 else model
            param_map[name] = (mod, parts[-1], buf)

    loaded_keys: set[str] = set()
    device = torch.device(device) if not isinstance(device, torch.device) else device

    for shard_idx, shard_file in enumerate(shard_files):
        shard_path = os.path.join(model_dir, shard_file)
        logger.info(
            "Streaming BnB shard %d/%d: %s",
            shard_idx + 1,
            len(shard_files),
            shard_file,
        )

        with safe_open(shard_path, framework="pt") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)

                if key not in param_map:
                    logger.debug("Skipping key not in model: %s", key)
                    del tensor
                    continue

                mod, attr, old_param = param_map[key]

                if isinstance(old_param, bnb.nn.Params4bit):
                    if torch_dtype is not None:
                        tensor = tensor.to(dtype=torch_dtype)
                    new_param = bnb.nn.Params4bit(
                        data=tensor,
                        requires_grad=False,
                        compress_statistics=old_param.compress_statistics,
                        quant_type=old_param.quant_type,
                        quant_storage=old_param.quant_storage,
                        module=mod if isinstance(mod, bnb.nn.Linear4bit) else None,
                        bnb_quantized=False,
                    )
                    del tensor
                    new_param._quantize(device)
                    mod._parameters[attr] = new_param
                else:
                    target_dtype = torch_dtype if torch_dtype is not None else tensor.dtype
                    materialized = tensor.to(device=device, dtype=target_dtype)
                    del tensor
                    if isinstance(old_param, torch.nn.Parameter):
                        mod._parameters[attr] = torch.nn.Parameter(materialized, requires_grad=old_param.requires_grad)
                    else:
                        mod._buffers[attr] = materialized

                loaded_keys.add(key)

        gc.collect()
        torch.cuda.empty_cache()

    # Tie weights before validating: safetensors typically stores only one copy
    # of a tied pair (e.g. Llama's lm_head.weight tied to embed_tokens.weight),
    # so the untied sibling is still on meta at this point. tie_weights()
    # re-establishes the Python-level alias so both sides point at the loaded
    # tensor.
    if hasattr(model, "tie_weights"):
        model.tie_weights()

    # Any param/buffer still on meta after load+tie is a real missing key —
    # forward pass would silently produce NaN.  Fail loudly instead.
    missing: list[str] = []
    for name, (_, _, _) in param_map.items():
        if name in loaded_keys:
            continue
        current = _get_model_tensor(model, name)
        if current is None or (hasattr(current, "device") and current.device.type == "meta"):
            missing.append(name)

    if missing:
        preview = ", ".join(missing[:10])
        more = f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""
        raise RuntimeError(
            f"Streaming BnB load left {len(missing)} tensor(s) unmaterialized after tie_weights: {preview}{more}"
        )

    logger.info(
        "Streaming BnB load complete: %d tensors loaded (%d additional tied after load)",
        len(loaded_keys),
        len(param_map) - len(loaded_keys),
    )


def _streaming_bnb_supported(cls, hf_config) -> bool:
    """Whether streaming BnB can safely load HF safetensors directly into the target class.

    The streaming loader maps safetensors keys 1:1 onto ``model.named_parameters()``.
    Two cases break that 1:1 assumption and must fall back to the standard HF loader:

    1. Automodel's custom implementations fuse projections (e.g. MoE
       ``mlp.experts.gate_up_proj``) and rely on a ``state_dict_adapter`` to translate
       HF-style keys on load. Detected via the ``HFCheckpointingMixin`` marker.
    2. Vanilla HF classes whose safetensors use a legacy layout that HF's loader
       reshapes/renames at load time (e.g. Mixtral ``block_sparse_moe.experts.*.w1`` →
       fused ``mlp.experts.gate_up_proj``). Detected via HF's per-model-type
       ``get_checkpoint_conversion_mapping`` — any non-empty mapping means the streaming
       path would leave fused tensors on meta device.
    """
    try:
        model_cls = cls._model_mapping[type(hf_config)]
    except (KeyError, TypeError):
        return False
    try:
        from nemo_automodel.components.models.common.hf_checkpointing_mixin import HFCheckpointingMixin

        if issubclass(model_cls, HFCheckpointingMixin):
            return False
    except ImportError:
        pass
    try:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    except ImportError:
        return True
    model_type = getattr(hf_config, "model_type", None)
    if model_type and get_checkpoint_conversion_mapping(model_type):
        return False
    return True


def _init_model_bnb_streaming(
    cls, pretrained_model_name_or_path, hf_config, attn_implementation, torch_dtype, quantization_config, **kwargs
):
    """Create model on meta device, replace Linear→Linear4bit, stream-load+quantize.

    This avoids materializing the full bf16 model in memory, which is critical
    for unified-memory systems (e.g. DGX Spark) where CPU and GPU share the
    same physical memory pool.

    Returns ``(is_custom_model=False, model)`` so the caller treats it like an
    HF-loaded model with weights already present.
    """
    from transformers.initialization import no_init_weights
    from transformers.integrations.bitsandbytes import replace_with_bnb_linear

    from nemo_automodel.components.utils.model_utils import init_empty_weights

    if isinstance(torch_dtype, str) and torch_dtype != "auto":
        torch_dtype = dtype_from_str(torch_dtype)
    if torch_dtype == "auto":
        torch_dtype = getattr(hf_config, "torch_dtype", torch.bfloat16)
        if isinstance(torch_dtype, str):
            torch_dtype = dtype_from_str(torch_dtype)

    device = torch.cuda.current_device()

    # 1. Download weights if needed
    _download_model_weights(hf_config, pretrained_model_name_or_path)

    # 2. Resolve to local directory & verify safetensors
    model_dir = _resolve_model_dir(pretrained_model_name_or_path)
    if not _has_safetensors(model_dir):
        raise FileNotFoundError(f"Streaming BnB loading requires safetensors checkpoint, but none found in {model_dir}")

    # 3. Create model skeleton on meta device (zero memory)
    logger.info("Creating model skeleton on meta device for streaming BnB quantization")
    with no_init_weights(), init_empty_weights():
        model = cls._from_config_parent_class(
            hf_config,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

    # 4. Replace nn.Linear → bnb.nn.Linear4bit (still on meta, no memory)
    modules_to_not_convert = getattr(quantization_config, "llm_int8_skip_modules", None)
    if modules_to_not_convert is None:
        modules_to_not_convert = getattr(model, "_keep_in_fp32_modules", None)
    model = replace_with_bnb_linear(
        model,
        modules_to_not_convert=modules_to_not_convert,
        quantization_config=quantization_config,
    )

    # 5. Stream-load weights, quantizing each tensor on the fly
    _stream_load_bnb_weights(model, model_dir, device, torch_dtype)

    # 6. Store quantization_config on the model (HF convention)
    model.config.quantization_config = quantization_config
    model.is_quantized = True

    # 7. Wrap with HFCheckpointingMixin
    try:
        hf_model_cls = cls._model_mapping[type(hf_config)]
    except KeyError:
        hf_model_cls = type(model)
    model.__class__ = _get_mixin_wrapped_class(hf_model_cls)

    return False, model


def _get_model_tensor(model, name: str):
    """Return a parameter or buffer by its fully-qualified state-dict key."""
    try:
        return model.get_parameter(name)
    except (AttributeError, ValueError):
        pass
    try:
        return model.get_buffer(name)
    except (AttributeError, ValueError):
        return None


def _restore_loaded_model_dtype(
    model, pretrained_model_name_or_path, hf_config, quantization_config, load_kwargs
) -> None:
    """Restore each loaded tensor to the exact dtype stored in the checkpoint.

    Some modules allocate parameters in a wider dtype than the checkpoint.
    HuggingFace then copies the checkpoint tensor into that existing tensor,
    which upcasts the loaded value. We fix that by re-inspecting checkpoint
    tensor dtypes per key and restoring each loaded parameter/buffer to the
    dtype that was actually stored in the file.
    """
    if quantization_config is not None or getattr(hf_config, "quantization_config", None) is not None:
        return

    try:
        checkpoint_dtypes = checkpoint_utils._get_checkpoint_tensor_dtypes(
            pretrained_model_name_or_path, hf_config, load_kwargs
        )
    except Exception as exc:
        logger.warning(
            "Failed to inspect checkpoint tensor dtypes for %s; leaving loaded dtypes unchanged: %s",
            pretrained_model_name_or_path,
            exc,
        )
        return

    if not checkpoint_dtypes:
        return

    restored_dtype_by_tensor_id: dict[int, torch.dtype] = {}
    restored_count = 0
    for name, checkpoint_dtype in checkpoint_dtypes.items():
        tensor = _get_model_tensor(model, name)
        if tensor is None or tensor.dtype == checkpoint_dtype:
            continue

        seen_dtype = restored_dtype_by_tensor_id.get(id(tensor))
        if seen_dtype is not None and seen_dtype != checkpoint_dtype:
            logger.warning(
                "Skipping conflicting checkpoint dtypes for aliased tensor %s: %s vs %s",
                name,
                seen_dtype,
                checkpoint_dtype,
            )
            continue

        try:
            tensor.data = tensor.data.to(dtype=checkpoint_dtype)
        except (RuntimeError, TypeError) as exc:
            logger.warning("Failed to restore checkpoint dtype for %s to %s: %s", name, checkpoint_dtype, exc)
            continue

        restored_dtype_by_tensor_id[id(tensor)] = checkpoint_dtype
        restored_count += 1

    if restored_count > 0:
        logger.info("Restored checkpoint dtypes for %d tensors from %s", restored_count, pretrained_model_name_or_path)


def __init_model(
    cls,
    pretrained_model_name_or_path_or_config,
    attn_implementation,
    torch_dtype,
    quantization_config,
    force_hf,
    *model_args,
    **kwargs,
):
    torch_dtype = dtype_from_str(torch_dtype) if torch_dtype != "auto" else torch_dtype
    is_pretrained_init = isinstance(pretrained_model_name_or_path_or_config, str)  # The caller is .from_pretrained
    hf_config = (
        get_hf_config(pretrained_model_name_or_path_or_config, attn_implementation, **kwargs)
        if is_pretrained_init
        else pretrained_model_name_or_path_or_config
    )
    pretrained_model_name_or_path = (
        pretrained_model_name_or_path_or_config if is_pretrained_init else getattr(hf_config, "name_or_path")
    )
    architectures = get_architectures(hf_config)

    # Streaming BnB loading: when quantization is requested and we're loading from a
    # pretrained checkpoint, use streaming quantization to avoid materializing the full
    # bf16 model in memory. This is critical for unified-memory systems (DGX Spark)
    # and large models (70B+). Can be disabled with AUTOMODEL_BNB_STREAMING=0.
    _bnb_streaming = os.environ.get("AUTOMODEL_BNB_STREAMING", "1") != "0"
    if (
        quantization_config is not None
        and is_pretrained_init
        and not force_hf
        and _bnb_streaming
        and _streaming_bnb_supported(cls, hf_config)
    ):
        try:
            logger.info("Using streaming BnB quantization for memory-efficient loading")
            return _init_model_bnb_streaming(
                cls,
                pretrained_model_name_or_path,
                hf_config,
                attn_implementation,
                torch_dtype,
                quantization_config,
                **kwargs,
            )
        except FileNotFoundError:
            logger.warning(
                "Streaming BnB loading unavailable (no safetensors checkpoint); falling back to standard HF loading."
            )

    # 1. if force_hf is True, use HF model class wrapped with mixin
    if force_hf:
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            _setup_bnb_loading_kwargs(kwargs)
        if is_pretrained_init:
            with skip_random_init():
                model = cls._from_pretrained_parent_class(
                    pretrained_model_name_or_path,
                    *model_args,
                    torch_dtype=torch_dtype,
                    attn_implementation=attn_implementation,
                    **kwargs,
                )
            _restore_loaded_model_dtype(model, pretrained_model_name_or_path, hf_config, quantization_config, kwargs)
        else:
            model = cls._from_config_parent_class(
                hf_config,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        # Get HF model class and wrap with mixin
        hf_model_cls = type(model)
        try:
            if len(architectures) > 0 and architectures[0] != "NemotronHForCausalLM":
                hf_model_cls = cls._model_mapping[type(hf_config)]
        except KeyError:
            pass  # fallback to use the model class from the model object
        model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
        return False, model

    # 2. If we have a custom model implementation available, we prioritize that over HF
    model_cls = _resolve_custom_model_cls_for_config(hf_config)
    if model_cls is not None:
        if quantization_config is not None:
            # BnB quantization is tightly integrated with HF's from_pretrained weight
            # loading pipeline.  Custom model constructors only create the architecture
            # (no weight loading, no quantization), so we must fall through to the HF
            # path which handles load + quantize atomically.
            logger.info(
                "BnB quantization requested; using HuggingFace model loader for %s "
                "(custom implementations do not support BnB quantization natively).",
                architectures[0],
            )
        else:
            # Download model weights on local rank 0; skip for from_config or local paths
            if pretrained_model_name_or_path:
                _download_model_weights(hf_config, pretrained_model_name_or_path)
            logger.info(f"Using custom model implementation for {architectures[0]}")
            kwargs.pop("trust_remote_code", None)
            # Treat config-related kwargs as config overrides (HF behavior) and
            # avoid forwarding them into model __init__.
            init_param_names = _get_init_param_names(model_cls)
            _consume_config_overrides(hf_config, kwargs, init_param_names=init_param_names)
            kwargs = _filter_kwargs_for_init(model_cls, kwargs)
            # Coerce plain-dict backend (e.g. from CLI --model.backend.attn sdpa) to BackendConfig
            if "backend" in kwargs and isinstance(kwargs["backend"], dict):
                from nemo_automodel.components.models.common.utils import BackendConfig

                kwargs["backend"] = BackendConfig(**kwargs["backend"])
            # Override config's torch_dtype with user-requested dtype so model __init__ uses correct dtype
            if torch_dtype != "auto":
                hf_config.torch_dtype = torch_dtype
            with local_torch_dtype(torch_dtype, model_cls.__name__):
                return True, model_cls(hf_config, *model_args, **kwargs)

    # 3. fallback to HF model class wrapped with mixin
    model = None
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
        _setup_bnb_loading_kwargs(kwargs)
    if is_pretrained_init:
        with skip_random_init():
            model = cls._from_pretrained_parent_class(
                pretrained_model_name_or_path,
                *model_args,
                torch_dtype=torch_dtype,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        _restore_loaded_model_dtype(model, pretrained_model_name_or_path, hf_config, quantization_config, kwargs)
    else:
        model = cls._from_config_parent_class(
            hf_config,
            *model_args,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            **kwargs,
        )

    # Get HF model class and wrap with mixin
    hf_model_cls = type(model)
    try:
        if len(architectures) > 0 and architectures[0] != "NemotronHForCausalLM":
            hf_model_cls = cls._model_mapping[type(hf_config)]
    except KeyError:
        pass  # fallback to use the model class from the model object
    model.__class__ = _get_mixin_wrapped_class(hf_model_cls)
    return False, model


def _tie_weights_nemo(model):
    if not hasattr(model, "_nemo_tied_weights_keys"):
        return

    def get_module_by_fqn(model, fqn):
        from functools import reduce

        fqn = fqn.split(".")
        if fqn[-1] == "weight":
            fqn = fqn[:-1]
        return reduce(getattr, fqn, model)

    for k, v in model._nemo_tied_weights_keys.items():
        get_module_by_fqn(model, k).weight = get_module_by_fqn(model, v).weight


def _init_model(
    cls,
    pretrained_model_name_or_path_or_config,
    attn_implementation,
    torch_dtype,
    quantization_config,
    force_hf,
    *model_args,
    **kwargs,
):
    is_custom_model, model = __init_model(
        cls,
        pretrained_model_name_or_path_or_config,
        attn_implementation,
        torch_dtype,
        quantization_config,
        force_hf,
        *model_args,
        **kwargs,
    )
    # https://github.com/NVIDIA-NeMo/Automodel/blob/a3a57176f68add7917faaa32f19228f49fcbb1ba/examples/llm_finetune/nemotron_flash/nemotron_flash_1b_squad.yaml#L41
    # this happens in nemotron_flash, where we load using force_hf, and the model is pre 5.x
    #
    # for safety, we tied weights after _model_init. We could do the tying in post_init, but it could be overwritten.
    # So the sequence is roughly:
    #   1. HF constructs NemotronFlashForCausalLM(config).
    #   2. Inside that constructor, self.post_init() runs.
    #   3. Only after construction returns does from_pretrained() finish loading/applying checkpoint weights.
    #   4. That later load can assign lm_head.weight and model.embed_tokens.weight separately, which breaks any alias we create inside post_init().

    if hasattr(model, "_nemo_tied_weights_keys"):
        _tie_weights_nemo(model)
    return is_custom_model, model


def get_architectures(hf_config):
    """
    Get the architectures from the HF config.
    """
    architectures = []
    if hasattr(hf_config, "architectures"):
        architectures = hf_config.architectures or []
    return architectures


def _get_init_param_names(model_cls) -> set[str]:
    """
    Best-effort extraction of explicit __init__ parameter names (excluding `self`).

    Returns an empty set if the signature cannot be inspected.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return set()
    return {k for k in sig.parameters.keys() if k != "self"}


def _consume_config_overrides(config, kwargs: dict, *, init_param_names: set[str] | None = None) -> None:
    """
    Mimic HF from_pretrained behavior: treat config-related kwargs as config overrides,
    not model __init__ kwargs.

    For custom model implementations we instantiate via `model_cls(config, **kwargs)`,
    so passing config flags like `output_hidden_states` would crash. This helper moves
    such keys onto the config and removes them from `kwargs`.
    """
    if init_param_names is None:
        init_param_names = set()
    # Prefer `to_dict()` to capture the canonical set of config fields.
    try:
        config_keys = set(config.to_dict().keys())
    except Exception:
        config_keys = set(getattr(config, "__dict__", {}).keys())

    for k in list(kwargs.keys()):
        # If the model explicitly declares this kwarg, keep it for __init__.
        if k in init_param_names:
            continue
        # Otherwise, if it looks like a config field, apply it to config.
        if k in config_keys:
            val = kwargs.pop(k)
            # Deep-merge dict overrides into existing sub-config objects (e.g.
            # text_config={"router_aux_loss_coef": 0}) instead of replacing the
            # entire sub-config, which would lose all other fields.
            if isinstance(val, dict):
                existing = getattr(config, k, None)
                if existing is not None and hasattr(existing, "to_dict"):
                    for sub_k, sub_v in val.items():
                        setattr(existing, sub_k, sub_v)
                    continue
            setattr(config, k, val)


def _filter_kwargs_for_init(model_cls, kwargs: dict) -> dict:
    """
    Filter kwargs down to what `model_cls.__init__` explicitly accepts.

    If the constructor has a `**kwargs` parameter (VAR_KEYWORD) or signature cannot be
    inspected, returns kwargs unchanged.
    """
    try:
        sig = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return kwargs

    allowed = set(sig.parameters.keys())
    allowed.discard("self")
    # We pass `config` positionally.
    allowed.discard("config")
    return {k: v for k, v in kwargs.items() if k in allowed}


def resolve_sdpa_method(
    sdpa_method: list | None = None,
    device_mesh=None,
    activation_checkpointing: bool = False,
) -> list["SDPBackend"] | None:  # noqa: F821
    """Resolve SDPA backend list from config strings or runtime constraints.

    When *sdpa_method* is provided (e.g. from YAML), string values are
    converted to :class:`torch.nn.attention.SDPBackend` enum members.
    Already-resolved ``SDPBackend`` values are passed through unchanged.
    When ``None``, automatic defaults are applied based on context
    parallelism and activation checkpointing settings.

    Valid string values (case-insensitive): ``flash_attention``,
    ``efficient_attention``, ``math``, ``cudnn_attention``.

    Args:
        sdpa_method: List of backend name strings or SDPBackend enum values,
            or ``None`` to use automatic defaults.
        device_mesh: Device mesh for distributed training.
        activation_checkpointing: Whether activation checkpointing is enabled.

    Returns:
        Ordered list of :class:`SDPBackend` members, or ``None`` to use
        PyTorch's default selection.
    """
    from torch.nn.attention import SDPBackend

    _NAME_TO_BACKEND = dict(SDPBackend.__members__)

    if sdpa_method is not None:
        backends = []
        for entry in sdpa_method:
            if isinstance(entry, str):
                key = entry.upper()
                if key not in _NAME_TO_BACKEND:
                    raise ValueError(f"Unknown SDPA backend '{entry}'. Valid values: {sorted(_NAME_TO_BACKEND.keys())}")
                backends.append(_NAME_TO_BACKEND[key])
            else:
                backends.append(entry)
        return backends

    # Auto-select based on runtime constraints
    cp_size = 1
    if device_mesh is not None and "cp" in device_mesh.mesh_dim_names:
        cp_size = device_mesh["cp"].size()

    if cp_size > 1:
        # CP with DTensor only supports flash and efficient backends;
        # MATH is not compatible with DTensor.
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
    elif activation_checkpointing:
        # For activation checkpointing, disable cudnn SDPA backend because
        # it may not be selected during recomputation, causing:
        # "Recomputed values have different metadata than during forward pass."
        return [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]

    return None
