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

"""Run existing MoE LoRA recipes with storage and actual GEMM dtype assertions."""

from collections import Counter
from functools import partial

import torch
from torch import nn
from torch.distributed.tensor import DTensor
from torch.utils._python_dispatch import TorchDispatchMode

from nemo_automodel.components._peft.lora_experts import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


class _ExpertComputeDtypes(TorchDispatchMode):
    """Inspect GEMM operands inside expert forwards, including AC recomputation.

    Inspect operands rather than module parameter dtypes: FSDP may expose BF16
    compute copies while the original adapter storage remains FP32.
    """

    def __init__(self) -> None:
        super().__init__()
        self.active: list[str] = []
        self.gemms: Counter[str] = Counter()

    def enter(self, name: str, module: nn.Module, inputs: tuple) -> None:
        assert inputs[0].dtype == torch.bfloat16, (name, "expert input", inputs[0].dtype)
        self.active.append(name)

    def leave(self, name: str, module: nn.Module, inputs: tuple, output: torch.Tensor | None) -> None:
        assert self.active.pop() == name
        # Non-reentrant AC can stop recomputation before an output is produced.
        if output is not None:
            assert output.dtype == torch.bfloat16, (name, "expert output", output.dtype)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if self.active and func._schema.name in {"aten::mm", "aten::bmm", "aten::addmm", "aten::_grouped_mm"}:
            operands = args[1:3] if func._schema.name == "aten::addmm" else args[:2]
            assert len(operands) == 2
            for operand in operands:
                assert operand.dtype == torch.bfloat16, (self.active[-1], func, operand.dtype)
            self.gemms[self.active[-1]] += 1
        return func(*args, **(kwargs or {}))


def _assert_storage(experts: dict[str, nn.Module], *, check_gradients: bool = False) -> None:
    for name, module in experts.items():
        adapter_count = base_count = 0
        for param_name, param in module.named_parameters():
            is_adapter = param_name.startswith("lora_")
            expected = torch.float32 if is_adapter else torch.bfloat16
            assert param.dtype == expected, (name, param_name, "storage", param.dtype, expected)
            local = param.to_local() if isinstance(param, DTensor) else param
            assert local.dtype == expected and local.numel() > 0, (name, param_name)
            assert param.requires_grad == is_adapter, (name, param_name, "requires_grad")
            if is_adapter:
                adapter_count += 1
                if check_gradients:
                    assert param.grad is not None, (name, param_name, "missing gradient")
                    assert param.grad.dtype == torch.float32, (name, param_name, "gradient", param.grad.dtype)
                    grad = param.grad.to_local() if isinstance(param.grad, DTensor) else param.grad
                    assert torch.isfinite(grad).all(), (name, param_name, "non-finite gradient")
            else:
                base_count += 1
                assert param.grad is None, (name, param_name, "frozen base gradient")
        assert adapter_count == 4 and base_count >= 2, (name, adapter_count, base_count)


def main() -> None:
    """Exercise the real recipe, including initialization, optimizer steps, and resume."""
    cfg = parse_args_and_load_config("tests/functional_tests/hf_peft/deepseek_moe_lora_small_for_test.yaml")
    trainer = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    trainer.setup()
    experts = {
        f"part{index}.{name}": module
        for index, part in enumerate(trainer.model_parts)
        for name, module in part.named_modules()
        if isinstance(module, (GroupedExpertsLoRA, GroupedExpertsDeepEPLoRA))
    }
    assert experts, "The recipe must actually install MoE LoRA experts"
    # setup() includes initialization and restores a checkpoint when resuming.
    _assert_storage(experts)
    audit = _ExpertComputeDtypes()
    handles = []
    optimizer_steps = 0

    def before_step(optimizer, args, kwargs):
        _assert_storage(experts, check_gradients=True)

    def after_step(optimizer, args, kwargs):
        nonlocal optimizer_steps
        _assert_storage(experts)
        optimizer_steps += 1

    try:
        for name, module in experts.items():
            handles.append(module.register_forward_pre_hook(partial(audit.enter, name)))
            handles.append(module.register_forward_hook(partial(audit.leave, name), always_call=True))
        for optimizer in trainer.optimizer:
            handles.append(optimizer.register_step_pre_hook(before_step))
            handles.append(optimizer.register_step_post_hook(after_step))
        with audit:
            trainer.run_train_validation_loop()
        _assert_storage(experts)
        assert optimizer_steps > 0, "The recipe must perform an optimizer update"
        assert not audit.active
        for name in experts:
            assert audit.gemms[name] > 0, (name, "no expert GEMMs observed")
        print(
            f"MOE_LORA_PRECISION_PASS expert_storage=bf16 lora_storage=fp32 "
            f"compute=bf16 optimizer_steps={optimizer_steps} expert_gemms={dict(audit.gemms)}",
            flush=True,
        )
    finally:
        for handle in handles:
            handle.remove()


if __name__ == "__main__":
    main()
