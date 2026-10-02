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

import os
import subprocess
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock

import pytest
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous import c10d_rendezvous_backend
from torch.distributed.run import config_from_args, get_args_parser


@pytest.mark.parametrize("timeout", ["", "37"])
def test_finetune_and_checkpoint_launches_apply_connection_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, timeout: str
) -> None:
    """Exercise shell command generation and the actual C10d timeout consumer."""
    launcher = Path(__file__).resolve().parents[3] / "tests/ci_tests/scripts/finetune_launcher.sh"
    argv_log = tmp_path / "torchrun.argv"
    stubs = tmp_path / "stubs.sh"
    stubs.write_text(
        'cd() { return 0; }\n'
        'python3() { printf "%s\\n" "$TEST_CONFIG"; }\n'
        'torchrun() { printf "%s\\0" "$@" >> "$TEST_ARGV"; printf "\\0" >> "$TEST_ARGV"; }\n'
    )
    env = {
        **os.environ,
        "BASH_ENV": str(stubs),
        "TEST_CONFIG": str(tmp_path / "config.yaml"),
        "TEST_ARGV": str(argv_log),
        "CONFIG_PATH": "examples/llm_finetune/glm/glm_5.1_lora.yaml",
        "PIPELINE_DIR": str(tmp_path),
        "TEST_NAME": "glm_5.1_lora",
        "TEST_LEVEL": "release",
        "TEST_SCRIPT_PATH": "train.py",
        "TEST_NODE_COUNT": "2",
        "CONFIG_NPROC_PER_NODE": "1",
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
        "SLURM_JOB_ID": "1234",
        "EXEC_CMD": "",
        "FINETUNE_ARGS": "",
        "RDZV_TIMEOUT": timeout,
        "HAS_ROBUSTNESS": "true",
        "CHECKPOINT_ROBUSTNESS_PROCESS_ISOLATION": "true",
        "CHECKPOINT_ROBUSTNESS_PHASES": "train_and_save automodel_reload resume",
        "REQUIRE_FINITE_METRICS": "false",
    }
    subprocess.run(["bash", str(launcher)], env=env, capture_output=True, text=True, check=True, timeout=10)

    commands = argv_log.read_bytes().split(b"\0\0")
    assert commands.pop() == b""
    assert len(commands) == 4  # Ordinary finetune plus three fresh checkpoint processes.
    tcp_store = Mock()
    monkeypatch.setattr(c10d_rendezvous_backend, "TCPStore", tcp_store)
    for command in commands:
        args = get_args_parser().parse_args(command.decode().split("\0"))
        config, _, _ = config_from_args(args)
        parameters = RendezvousParameters(
            backend=config.rdzv_backend,
            endpoint=config.rdzv_endpoint,
            run_id=config.run_id,
            min_nodes=config.min_nodes,
            max_nodes=config.max_nodes,
            is_host=True,
            **config.rdzv_configs,
        )
        c10d_rendezvous_backend._create_tcp_store(parameters)
        expected = int(timeout or 600)
        assert tcp_store.call_args.kwargs["timeout"] == timedelta(seconds=expected)
        assert config.rdzv_configs["join_timeout"] == str(expected)
