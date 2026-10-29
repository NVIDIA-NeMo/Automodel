# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise single-process rendezvous with the real Gloo backend."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch.distributed as dist


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="requires Gloo")
@pytest.mark.parametrize("remove_hash_store", [False, True])
def test_single_process_store_lifecycle(tmp_path, remove_hash_store):
    source = Path(__file__).resolve().parents[3] / "nemo_automodel/components/distributed/init_utils.py"
    script = r"""
import importlib.util
import sys
import torch

spec = importlib.util.spec_from_file_location("tested_init_utils", sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
if sys.argv[2] == "True" and hasattr(torch.distributed, "HashStore"):
    del torch.distributed.HashStore
for attempt in range(2):
    info = module.initialize_distributed("nccl")
    assert info.rank == 0 and info.world_size == 1 and info.is_main
    assert torch.distributed.get_backend() == "gloo"
    value = torch.tensor([3.0])
    torch.distributed.all_reduce(value)
    assert value.item() == 3.0
    if attempt == 0:
        module.destroy_global_state()
        assert not torch.distributed.is_initialized()
print("single-process rendezvous passed")
"""
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES="", RANK="0", LOCAL_RANK="0", WORLD_SIZE="1")
    env.update({key: str(tmp_path) for key in ("TMP", "TEMP", "TMPDIR")})
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(remove_hash_store)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "single-process rendezvous passed" in result.stdout
    assert "Exception ignored in atexit callback" not in result.stderr
    assert not list(tmp_path.glob("automodel-store-*"))
