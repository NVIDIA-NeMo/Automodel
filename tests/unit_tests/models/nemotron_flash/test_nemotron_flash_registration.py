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

import ast
from pathlib import Path

from transformers import AutoConfig

from nemo_automodel.components.models.nemotron_flash.configuration import NemotronFlashConfig

MODELING_FILE = (
    Path(__file__).resolve().parents[4] / "nemo_automodel/components/models/nemotron_flash/modeling_nemotron_flash.py"
)


def _get_method_args(class_name: str, method_name: str) -> ast.arguments:
    tree = ast.parse(MODELING_FILE.read_text())
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == method_name:
                    return child.args
    raise AssertionError(f"Could not find {class_name}.{method_name} in {MODELING_FILE}")


def test_nemotron_flash_config_registered():
    import nemo_automodel._transformers.registry  # noqa: F401

    cfg = AutoConfig.for_model("nemotron_flash")
    assert isinstance(cfg, NemotronFlashConfig)
    assert cfg.model_type == "nemotron_flash"


def test_nemotron_flash_in_registry():
    from nemo_automodel._transformers.registry import MODEL_ARCH_MAPPING

    mapping = dict(MODEL_ARCH_MAPPING)
    assert "NemotronFlashForCausalLM" in mapping
    assert mapping["NemotronFlashForCausalLM"][0] == "nemo_automodel.components.models.nemotron_flash.model"


def test_nemotron_flash_forward_accepts_padding_mask():
    model_args = _get_method_args("NemotronFlashModel", "forward")
    causal_lm_args = _get_method_args("NemotronFlashForCausalLM", "forward")

    assert "padding_mask" in [arg.arg for arg in model_args.args]
    assert model_args.kwarg is not None and model_args.kwarg.arg == "kwargs"

    causal_lm_arg_names = [arg.arg for arg in causal_lm_args.args]
    assert "padding_mask" in causal_lm_arg_names
    assert "logits_to_keep" in causal_lm_arg_names
    assert causal_lm_args.kwarg is not None and causal_lm_args.kwarg.arg == "kwargs"
