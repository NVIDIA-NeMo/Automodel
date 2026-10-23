# 独立验证 PR A 的两个辅助函数(从 RL-main 的 setup.py 用 AST 抽出,避免导入整个 nemo_rl/nemo_automodel 栈):逻辑与 #4481 一致。
import ast, datetime, pathlib, sys, types
from typing import Optional
from types import SimpleNamespace
from unittest.mock import patch
import pytest, torch
SRC = pathlib.Path(__file__).parent / "RL-main/nemo_rl/models/automodel/setup.py"
tree = ast.parse(SRC.read_text())
wanted = {"_distributed_timeout_minutes", "_init_process_group"}
funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
assert {f.name for f in funcs} == wanted, [f.name for f in funcs]
mod = types.ModuleType("helpers"); mod.__dict__.update({"datetime": datetime, "torch": torch, "Optional": Optional, "PolicyConfig": dict, "RuntimeConfig": object})
exec(compile(ast.Module(body=funcs, type_ignores=[]), str(SRC), "exec"), mod.__dict__)
_distributed_timeout_minutes, _init_process_group = mod._distributed_timeout_minutes, mod._init_process_group

def _init_call(automodel_cfg, cpu_offload=False):
    with patch("torch.distributed.init_process_group") as mock_init:
        _init_process_group({"automodel_cfg": automodel_cfg}, SimpleNamespace(cpu_offload=cpu_offload))
    return mock_init.call_args

@pytest.mark.parametrize("minutes", [45, 0.5])
def test_forwards_timeout(minutes):
    assert _init_call({"distributed_timeout_minutes": minutes}).kwargs == {"backend": "nccl", "timeout": datetime.timedelta(minutes=minutes)}

def test_cpu_offload_backend():
    assert _init_call({"distributed_timeout_minutes": 30}, cpu_offload=True).kwargs == {"backend": "cuda:nccl,cpu:gloo", "timeout": datetime.timedelta(minutes=30)}

@pytest.mark.parametrize("cfg", [{}, {"distributed_timeout_minutes": None}])
def test_unset_keeps_default(cfg):
    assert _init_call(cfg).kwargs == {"backend": "nccl"}

@pytest.mark.parametrize("minutes", [0, -5])
def test_non_positive_rejected(minutes):
    with pytest.raises(ValueError, match=r"^automodel_cfg\.distributed_timeout_minutes must be positive"):
        _distributed_timeout_minutes({"automodel_cfg": {"distributed_timeout_minutes": minutes}})
