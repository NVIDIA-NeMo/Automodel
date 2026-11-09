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

"""Real single-rank placement through the public Diffusers entry points."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from diffusers import DiffusionPipeline, Transformer2DModel

from nemo_automodel._diffusers.auto_diffusion_pipeline import NeMoAutoDiffusionPipeline
from nemo_automodel.components.distributed import DDPConfig, MeshContext
from nemo_automodel.components.distributed.ddp import parallelize_ddp
from nemo_automodel.components.distributed.init_utils import initialize_distributed


def _run_single_rank(case: str, device_index: int, output_dir: Path) -> None:
    """Verify placement, training and weight reload in an isolated process."""
    report = {"status": "failed", "case": case, "device_index": device_index, "torch": torch.__version__}
    try:
        assert not dist.is_initialized()
        os.environ.update(WORLD_SIZE="1", LOCAL_WORLD_SIZE="1", RANK="0", LOCAL_RANK=str(device_index))
        initialize_distributed("nccl", timeout_minutes=1)
        assert str(dist.get_backend()) == "gloo"
        assert dist.get_world_size() == 1
        assert torch.cuda.current_device() == device_index
        torch.manual_seed(42)
        expected_device = torch.device("cpu" if "cpu" in case else f"cuda:{device_index}")
        requested_device = None if "default" in case or case == "direct-cuda" else expected_device
        move_to_device = not case.endswith("no-move")
        context = MeshContext(strategy_config=DDPConfig())
        original_context = dict(context.__dict__)

        if case == "direct-cuda":
            model = torch.nn.Linear(4, 2).to(device=expected_device, dtype=torch.bfloat16)
            model = parallelize_ddp(model, DDPConfig())
        elif case.startswith("pretrained"):
            # Replace only external weight loading; placement and parallelization are real.
            loaded = NeMoAutoDiffusionPipeline(transformer=torch.nn.Linear(4, 2))
            with patch.object(DiffusionPipeline, "from_pretrained", return_value=loaded):
                pipe = NeMoAutoDiffusionPipeline.from_pretrained(
                    str(output_dir),
                    device=requested_device,
                    torch_dtype=torch.float32,
                    mesh_context=context,
                    move_to_device=move_to_device,
                    load_for_training=True,
                )
            model = pipe.transformer
        else:
            # Exercise actual local Diffusers configuration loading without downloaded weights.
            transformer = Transformer2DModel(
                num_attention_heads=1,
                attention_head_dim=4,
                in_channels=4,
                out_channels=4,
                num_layers=1,
                norm_num_groups=1,
                sample_size=2,
            )
            transformer.save_config(output_dir / "transformer")
            pipe = NeMoAutoDiffusionPipeline.from_config(
                str(output_dir),
                pipeline_spec={"transformer_cls": "Transformer2DModel", "load_full_pipeline": False},
                device=requested_device,
                torch_dtype=torch.float32,
                mesh_context=context,
                move_to_device=move_to_device,
            )
            model = pipe.transformer

        assert context.__dict__ == original_context, "loading must not mutate caller-owned mesh context"
        assert all(parameter.device == expected_device for parameter in model.parameters())
        dtype = next(model.parameters()).dtype
        if expected_device.type == "cpu":
            assert dtype == torch.float32
        shape = (1, 4, 2, 2) if case.startswith("config") else (3, 4)
        inputs = torch.randn(shape, device=expected_device, dtype=dtype)
        model.eval()
        output = model(inputs)
        output = output.sample if case.startswith("config") else output
        target = torch.randn_like(output)
        before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = (output.float() - target.float()).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        optimizer.step()
        assert any(not torch.equal(before[name], value.detach().cpu()) for name, value in model.state_dict().items())
        with torch.no_grad():
            prediction = model(inputs)
            prediction = prediction.sample if case.startswith("config") else prediction
            prediction = prediction.cpu()
        checkpoint = output_dir / "weights.pt"
        torch.save({name: value.detach().cpu() for name, value in model.state_dict().items()}, checkpoint)
        # Clear parameters before reload so equality cannot pass from untouched weights.
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        with torch.no_grad():
            restored = model(inputs)
            restored = restored.sample if case.startswith("config") else restored
            restored = restored.cpu()
        torch.testing.assert_close(restored, prediction, rtol=0, atol=0)
        root = Path(__file__).resolve().parents[3]
        paths = [
            "nemo_automodel/components/distributed/ddp.py",
            "nemo_automodel/components/distributed/mesh.py",
            "nemo_automodel/components/distributed/model_parallelizer.py",
            "nemo_automodel/_diffusers/auto_diffusion_pipeline.py",
        ]
        report.update(
            status="passed",
            backend=str(dist.get_backend()),
            parameter_devices=sorted({str(parameter.device) for parameter in model.parameters()}),
            dtype=str(dtype),
            loss=loss.detach().item(),
            reload_max_abs_error=(restored - prediction).abs().max().item(),
            gpu=torch.cuda.get_device_name(device_index),
            source_sha256={path: hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths},
        )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.parametrize(
    "case,device_index",
    [
        ("direct-cuda", 0),
        ("direct-cuda", 1),
        ("pretrained-cpu", 0),
        ("config-cpu", 0),
        ("pretrained-cpu-no-move", 0),
        ("config-cpu-no-move", 0),
        ("pretrained-default", 0),
        ("config-default", 0),
        ("pretrained-cuda", 1),
    ],
)
def test_single_rank_gloo_device_train_save_reload(case: str, device_index: int, tmp_path: Path) -> None:
    """Run public CPU overrides and default CUDA placement on a CUDA host."""
    if not torch.cuda.is_available() or torch.cuda.device_count() <= device_index:
        pytest.skip(f"requires CUDA device {device_index}")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), case, str(device_index), str(tmp_path)],
        # These tiny CPU/Gloo cases also run in GPU CI jobs, where the unit-test
        # launcher does not bound host thread pools. Keep each child bounded.
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["status"] == "passed"
    expected = "cpu" if "cpu" in case else f"cuda:{device_index}"
    assert report["parameter_devices"] == [expected]
    assert report["reload_max_abs_error"] == 0
    print(f"DDP_DEVICE_REPORT={tmp_path / 'report.json'}")


if __name__ == "__main__":
    _run_single_rank(sys.argv[1], int(sys.argv[2]), Path(sys.argv[3]))
