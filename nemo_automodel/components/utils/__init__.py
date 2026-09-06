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

import importlib as _importlib

_LAZY_ATTRS = {
    "FreezeConfig": (".model_utils", "FreezeConfig"),
    "ModuleSelector": (".model_utils", "ModuleSelector"),
    "VLM_INPUT_KEYS": (".model_utils", "VLM_INPUT_KEYS"),
    "apply_parameter_freezing": (".model_utils", "apply_parameter_freezing"),
    "build_compile_config": (".compile_utils", "build_compile_config"),
    "calculate_mfu": (".flops_utils", "calculate_mfu"),
    "compile_model": (".compile_utils", "compile_model"),
    "compile_module_inplace": (".compile_utils", "compile_module_inplace"),
    "count_model_parameters": (".model_utils", "count_model_parameters"),
    "enable_radio_vit_fused_attn": (".model_utils", "enable_radio_vit_fused_attn"),
    "filter_forward_kwargs": (".model_utils", "filter_forward_kwargs"),
    "freeze_deepseek_v4_indexer_params": (".model_utils", "freeze_deepseek_v4_indexer_params"),
    "freeze_minimax_m3_indexer_params": (".model_utils", "freeze_minimax_m3_indexer_params"),
    "freeze_unused_kv_sharing_params": (".model_utils", "freeze_unused_kv_sharing_params"),
    "get_flops_formula_for_hf_config": (".flops_utils", "get_flops_formula_for_hf_config"),
    "init_empty_weights": (".model_utils", "init_empty_weights"),
    "nemotronh_mtp_flops": (".flops_utils", "_nemotronh_mtp_flops"),
    "parse_freeze_config": (".model_utils", "parse_freeze_config"),
    "print_trainable_parameters": (".model_utils", "print_trainable_parameters"),
    "resolve_trust_remote_code": (".model_utils", "resolve_trust_remote_code"),
    "skip_random_init": (".model_utils", "skip_random_init"),
    "squeeze_input_for_thd": (".model_utils", "squeeze_input_for_thd"),
    "supports_logits_to_keep": (".model_utils", "_supports_logits_to_keep"),
    "supports_seq_lens": (".model_utils", "_supports_seq_lens"),
}

__all__ = sorted(_LAZY_ATTRS.keys())


def __getattr__(name: str) -> object:
    """Load an exported component symbol on first access."""
    if name in _LAZY_ATTRS:
        module_path, attr_name = _LAZY_ATTRS[name]
        module = _importlib.import_module(module_path, __name__)
        attr = getattr(module, attr_name)
        globals()[name] = attr
        return attr
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Return the component's exported symbols."""
    return sorted(__all__)
