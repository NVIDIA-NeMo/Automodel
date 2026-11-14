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

"""Disk-backed CPU LoRA QAT resume and independent packed-checkpoint inference."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig

# Static programs deliberately import production modules, not this test or conftest.
# Only the trusted temporary checkpoint directory is passed through argv.
_RESUME_SCRIPT = """
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nemo_automodel._transformers.qat import QAT
from nemo_automodel.components._peft.lora import PeftConfig, apply_lora_to_linear_modules
from nemo_automodel.components.quantization.qat import QATConfig, QATRule
from nemo_automodel.components.quantization.weight_qat import WeightQuantizationConfig

torch.set_num_threads(1)
torch.manual_seed(999)
directory = Path(sys.argv[1])
model = nn.Sequential(
    nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
    nn.ReLU(),
    nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
)
base = load_file(str(directory / "base.safetensors"))
model.load_state_dict(base, strict=True)
assert apply_lora_to_linear_modules(
    model,
    PeftConfig(target_modules=["0", "2"], dim=4, alpha=8, use_memory_efficient_lora=False),
) == 2
controller = QAT(QATConfig(rules=(
    QATRule(("0",), WeightQuantizationConfig("fp8", (32, 32))),
    QATRule(("2",), WeightQuantizationConfig("mxfp4", (1, 32))),
)))
controller.prepare(model)
adapters = load_file(str(directory / "adapters.safetensors"))
assert set(adapters) == {f"{layer}.lora_{factor}.weight" for layer in (0, 2) for factor in ("A", "B")}
incompatible = model.load_state_dict(adapters, strict=False)
assert set(incompatible.missing_keys) == set(base)
assert not incompatible.unexpected_keys
for name, parameter in model.named_parameters():
    assert parameter.requires_grad == (name in adapters)
    if name in adapters:
        assert torch.equal(parameter, adapters[name])
        assert torch.count_nonzero(parameter) > 0
    else:
        assert torch.equal(parameter, base[name])

# A deliberately different LR verifies that load_state_dict restores param groups.
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad], lr=0.7, foreach=False
)
optimizer.load_state_dict(torch.load(directory / "optimizer.pt", map_location="cpu", weights_only=True))
assert optimizer.param_groups[0]["lr"] == 0.01
assert len(optimizer.state) == len(adapters)
assert all(state["step"].item() == 2 for state in optimizer.state.values())
batch = load_file(str(directory / "batch.safetensors"))
model.eval()
with torch.no_grad():
    before = model(batch["inputs"])
    assert torch.equal(before, batch["checkpoint_output"])
model.train()
optimizer.zero_grad(set_to_none=True)
loss = nn.functional.mse_loss(model(batch["inputs"]), batch["target"])
loss.backward()
optimizer.step()
for name, parameter in model.named_parameters():
    if name in base:
        assert not parameter.requires_grad
        assert parameter.grad is None
        assert torch.equal(parameter, base[name])
    else:
        assert not torch.equal(parameter, adapters[name])
model.eval()
with torch.no_grad():
    after = model(batch["inputs"])
save_file({"before": before, "after": after}, str(directory / "resumed_outputs.safetensors"))
save_file(model.state_dict(), str(directory / "resumed_state.safetensors"))
torch.save(optimizer.state_dict(), directory / "resumed_optimizer.pt")
controller.export(model, directory / "export")
"""

_LOAD_SCRIPT = """
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from nemo_automodel._transformers.qat import QAT

torch.set_num_threads(1)
torch.manual_seed(12345)
directory = Path(sys.argv[1])
model = nn.Sequential(
    nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
    nn.ReLU(),
    nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
)
assert set(model.state_dict()) == {"0.weight", "0.bias", "2.weight", "2.bias"}
QAT.load_quantized_checkpoint(model, directory / "export")
model.eval()
inputs = load_file(str(directory / "batch.safetensors"))["inputs"]
with torch.no_grad():
    output = model(inputs)
save_file({"output": output}, str(directory / "loaded_output.safetensors"))
"""


@pytest.mark.timeout(60)
def test_saved_adapter_optimizer_resume_export_and_fresh_process_load(tmp_path: Path) -> None:
    """Require bit-exact resume and export on CPU float32 [batch=4, hidden=32] data."""
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(91)
            model = nn.Sequential(
                nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
                nn.ReLU(),
                nn.Linear(32, 32, bias=True, dtype=torch.float32, device="cpu"),
            )
            base = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            save_file(base, str(tmp_path / "base.safetensors"))
            assert (
                apply_lora_to_linear_modules(
                    model,
                    PeftConfig(target_modules=["0", "2"], dim=4, alpha=8, use_memory_efficient_lora=False),
                )
                == 2
            )
            controller = QAT(QATConfig(
                rules=(
                    QATRule(("0",), WeightQuantizationConfig("fp8", (32, 32))),
                    QATRule(("2",), WeightQuantizationConfig("mxfp4", (1, 32))),
                )
            ))
            controller.prepare(model)
            adapter_names = {f"{layer}.lora_{factor}.weight" for layer in (0, 2) for factor in ("A", "B")}
            assert {name for name, p in model.named_parameters() if p.requires_grad} == adapter_names
            initial_adapters = {name: model.state_dict()[name].clone() for name in adapter_names}
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01, foreach=False)
            # Fixed [batch, hidden] inputs and regression targets shared through disk.
            inputs = torch.randn(4, 32, dtype=torch.float32, device="cpu")
            target = torch.randn(4, 32, dtype=torch.float32, device="cpu")
            model.train()
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(model(inputs), target)
                loss.backward()
                optimizer.step()
            adapters = {name: model.state_dict()[name].clone() for name in adapter_names}
            for name, adapter in adapters.items():
                assert torch.count_nonzero(adapter) > 0
                assert not torch.equal(adapter, initial_adapters[name]), name
            for name, parameter in model.named_parameters():
                if name in base:
                    assert not parameter.requires_grad
                    assert parameter.grad is None
                    assert torch.equal(parameter, base[name]), name
            assert len(optimizer.state) == 4
            assert all(state["step"].item() == 2 for state in optimizer.state.values())
            save_file(adapters, str(tmp_path / "adapters.safetensors"))
            # These optimizer files are created by this test, never external pickle input.
            torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
            model.eval()
            with torch.no_grad():
                checkpoint_output = model(inputs)
            save_file(
                {"inputs": inputs, "target": target, "checkpoint_output": checkpoint_output},
                str(tmp_path / "batch.safetensors"),
            )

            # Uninterrupted third step is the independent reference for optimizer resume.
            model.train()
            optimizer.zero_grad(set_to_none=True)
            nn.functional.mse_loss(model(inputs), target).backward()
            optimizer.step()
            model.eval()
            with torch.no_grad():
                expected_output = model(inputs)
            assert not torch.equal(expected_output, checkpoint_output)

            repo_root = Path(__file__).resolve().parents[3]
            env = dict(os.environ)
            env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
            env["OMP_NUM_THREADS"] = env["MKL_NUM_THREADS"] = "1"
            for script in (_RESUME_SCRIPT, _LOAD_SCRIPT):
                result = subprocess.run(
                    [sys.executable, "-c", script, str(tmp_path)],
                    cwd=repo_root,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=25,
                    check=False,
                )
                assert result.returncode == 0, result.stdout + result.stderr

            resumed_outputs = load_file(str(tmp_path / "resumed_outputs.safetensors"))
            assert torch.equal(resumed_outputs["before"], checkpoint_output)
            assert torch.equal(resumed_outputs["after"], expected_output)
            resumed_state = load_file(str(tmp_path / "resumed_state.safetensors"))
            assert set(resumed_state) == set(model.state_dict())
            for name, tensor in model.state_dict().items():
                assert torch.equal(resumed_state[name], tensor), name
                if name in base:
                    assert torch.equal(tensor, base[name]), name
                else:
                    assert not torch.equal(tensor, adapters[name]), name
            resumed_optimizer = torch.load(tmp_path / "resumed_optimizer.pt", map_location="cpu", weights_only=True)
            expected_optimizer = optimizer.state_dict()
            assert resumed_optimizer["param_groups"] == expected_optimizer["param_groups"]
            assert set(resumed_optimizer["state"]) == set(expected_optimizer["state"])
            for index, state in expected_optimizer["state"].items():
                assert set(state) == set(resumed_optimizer["state"][index]) == {"step", "exp_avg", "exp_avg_sq"}
                assert state["step"].item() == 3
                assert torch.count_nonzero(state["exp_avg"]) > 0
                assert torch.count_nonzero(state["exp_avg_sq"]) > 0
                for key, tensor in state.items():
                    assert torch.equal(resumed_optimizer["state"][index][key], tensor), (index, key)

            exported = tmp_path / "export"
            manifest = json.loads((exported / "manifest.json").read_text())
            assert [(entry["weight_key"], entry["config"]) for entry in manifest["weights"]] == [
                ("0.weight", {"format": "fp8", "block_size": [32, 32], "scale_format": "e8m0"}),
                ("2.weight", {"format": "mxfp4", "block_size": [1, 32], "scale_format": "e8m0"}),
            ]
            packed = load_file(str(exported / "quantized.safetensors"))
            assert set(packed) == {f"{layer}.weight.{part}" for layer in (0, 2) for part in ("payload", "scales")}
            assert all(tensor.dtype == torch.uint8 for tensor in packed.values())
            ordinary = load_file(str(exported / "model.safetensors"))
            assert set(ordinary) == {"0.bias", "2.bias"}
            assert all(torch.equal(tensor, base[name]) for name, tensor in ordinary.items())
            loaded_output = load_file(str(tmp_path / "loaded_output.safetensors"))["output"]
            assert torch.equal(loaded_output, expected_output)
    finally:
        torch.set_num_threads(previous_threads)
