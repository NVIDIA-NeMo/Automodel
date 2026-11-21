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

"""Check that a LoRA sequence-classification checkpoint keeps the trained classifier.

L2_Seq_Cls_LoRA_FP32.sh trains one step with checkpointing, then runs this script with
the same config. The recipe trains the classifier in full next to the LoRA adapters
(freeze_config.unfreeze_modules), so the adapter checkpoint must store the classifier,
list it in modules_to_save, load it in HF PEFT, and restore it when training resumes.
"""

import json
from pathlib import Path

import torch
from peft import PeftModel
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers import BertForSequenceClassification

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_seq_cls import TrainFinetuneRecipeForSequenceClassification
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

_CONFIG = "tests/functional_tests/llm_pretrain_and_kd/llm_seq_cls/seq_cls_bert_lora_fp32.yaml"
_HEAD_KEYS = {"base_model.model.classifier.weight", "base_model.model.classifier.bias"}


def main() -> None:
    """Check the latest checkpoint's adapter files, its HF PEFT reload, and the recipe resume."""
    cfg = parse_args_and_load_config(_CONFIG)
    checkpoints = sorted(Path(cfg.checkpoint.checkpoint_dir).glob("epoch_*_step_*"))
    assert checkpoints, f"No checkpoint was written under {cfg.checkpoint.checkpoint_dir}"
    adapter_dir = checkpoints[-1] / "model"
    adapter = load_file(str(adapter_dir / "adapter_model.safetensors"))
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))

    assert adapter_config["task_type"] == "SEQ_CLS", adapter_config
    assert adapter_config["modules_to_save"] == ["classifier"], adapter_config
    assert _HEAD_KEYS <= adapter.keys(), f"The adapter does not store the classifier: {sorted(adapter)}"

    # HF PEFT wraps the classifier of a SEQ_CLS adapter and loads it from the adapter file.
    base = BertForSequenceClassification(cfg.model.config.instantiate())
    head = PeftModel.from_pretrained(base, str(adapter_dir)).base_model.model.classifier.modules_to_save["default"]
    for name, parameter in head.named_parameters():
        torch.testing.assert_close(parameter, adapter[f"base_model.model.classifier.{name}"], rtol=0, atol=0)

    # Resuming loads the same checkpoint: every trainable parameter, LoRA and classifier, comes back.
    recipe = TrainFinetuneRecipeForSequenceClassification(cfg)
    recipe.setup()
    restored = {}
    for name, parameter in recipe.model_parts[0].named_parameters():
        if parameter.requires_grad:
            value = parameter.full_tensor() if isinstance(parameter, DTensor) else parameter
            restored[f"base_model.model.{canonical_parameter_fqn(name)}"] = value.detach().cpu()
    assert restored.keys() == adapter.keys(), sorted(set(restored) ^ set(adapter))
    for key, value in adapter.items():
        torch.testing.assert_close(restored[key], value, rtol=0, atol=0)
    print(f"LoRA classifier checkpoint at {adapter_dir} is complete and reloads correctly")


if __name__ == "__main__":
    main()
