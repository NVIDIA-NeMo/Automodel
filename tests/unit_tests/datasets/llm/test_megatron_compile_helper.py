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

"""Exercise real JIT builds, cache reuse, and the NumPy helper bindings."""

import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from nemo_automodel.components.datasets.llm.megatron import megatron_utils
from nemo_automodel.components.datasets.llm.megatron.helpers import build_sample_idx
from nemo_automodel.components.datasets.llm.megatron.megatron_utils import compile_helper


@pytest.fixture(scope="module")
def helpers(tmp_path_factory):
    cache = tmp_path_factory.mktemp("megatron_extensions")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(megatron_utils, "_HELPER_CACHE_ROOT", cache)
        monkeypatch.setenv("MAX_JOBS", "1")
        compile_helper.cache_clear()
        module = compile_helper()
    yield module
    compile_helper.cache_clear()


# A cold C++ compilation exceeds the normal 5s unit-test budget.
@pytest.mark.runtime_budget(60, hard_timeout=120, reason="Compiles the real CPU extension once in the module fixture.")
@pytest.mark.parametrize("add_extra_token", [False, True])
def test_sample_indices(helpers, add_extra_token):
    sizes = np.array([4, 4], dtype=np.int32)
    documents = np.array([0, 1], dtype=np.int32)
    expected = [[0, 0], [0, 2], [1, 0], [1, 2]]
    if not add_extra_token:
        # Without the lookahead token, an exact boundary stays at the end of
        # the preceding document rather than the start of the next one.
        expected[2] = [0, 4]
        expected.append([1, 4])
    actual = build_sample_idx(sizes, documents, 2, 1, 8, add_extra_token_to_sequence=add_extra_token)
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.int32
    actual64 = helpers.build_sample_idx_int64(sizes, documents, 2, 1, 8, True, int(add_extra_token))
    np.testing.assert_array_equal(actual64, expected)
    assert actual64.dtype == np.int64
    assert compile_helper() is helpers


@pytest.mark.runtime_budget(
    60, hard_timeout=120, reason="Compiles the real CPU extension when this test runs independently."
)
def test_blending_indices(helpers):
    dataset_index = np.zeros(8, dtype=np.int16)
    sample_index = np.zeros(8, dtype=np.int64)
    helpers.build_blending_indices(dataset_index, sample_index, np.array([0.25, 0.75]), 2, 8, False)
    np.testing.assert_array_equal(np.bincount(dataset_index), [2, 6])
    for dataset in range(2):
        np.testing.assert_array_equal(sample_index[dataset_index == dataset], np.arange([2, 6][dataset]))

    helpers.build_exhaustive_blending_indices(dataset_index, sample_index, np.array([3, 5], dtype=np.int64), 2)
    np.testing.assert_array_equal(np.bincount(dataset_index), [3, 5])
    for dataset in range(2):
        np.testing.assert_array_equal(sample_index[dataset_index == dataset], np.arange([3, 5][dataset]))


# Delayed source writes expose overlap before PyTorch's internal build lock.
@pytest.mark.runtime_budget(
    90, hard_timeout=120, reason="Races two real cold-cache JIT loads and checks reuse in a fresh interpreter."
)
def test_concurrent_build_and_cache_reuse(tmp_path):
    env = dict(os.environ, TEST_HELPER_CACHE=str(tmp_path), MAX_JOBS="1")
    script = """
import os
import time
from pathlib import Path
import numpy as np
import torch.utils.cpp_extension as cpp_extension
from nemo_automodel.components.datasets.llm.megatron import megatron_utils
from nemo_automodel.components.datasets.llm.megatron.megatron_utils import compile_helper
cache = Path(os.environ["TEST_HELPER_CACHE"])
megatron_utils._HELPER_CACHE_ROOT = cache
if os.environ.get("RACE_SOURCE_WRITES"):
    (cache / f"ready-{os.getpid()}").touch()
    deadline = time.monotonic() + 30
    while len(list(cache.glob("ready-*"))) < 2:
        assert time.monotonic() < deadline, "second worker did not start"
        time.sleep(0.01)
    original_write = cpp_extension._maybe_write
    def slow_write(filename, content):
        # Exclusive creation fails if another rank enters source generation.
        active = cache / "source-write-active"
        with active.open("x"):
            try:
                time.sleep(0.5)
                original_write(filename, content)
            finally:
                active.unlink()
    cpp_extension._maybe_write = slow_write
module = compile_helper()
actual = module.build_sample_idx_int32(np.array([8], dtype=np.int32), np.array([0], dtype=np.int32), 2, 1, 8, True, 1)
np.testing.assert_array_equal(actual, [[0, 0], [0, 2], [0, 4], [0, 6]])
assert compile_helper() is module
print(Path(module.__file__).resolve())
"""
    processes = [
        subprocess.Popen(
            [sys.executable, "-c", script],
            env=dict(env, RACE_SOURCE_WRITES="1"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]
    try:
        paths = []
        for process in processes:
            stdout, stderr = process.communicate(timeout=100)
            assert process.returncode == 0, stderr
            paths.append(Path(stdout.strip()))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait()
    assert paths[0] == paths[1]
    assert paths[0].is_relative_to(tmp_path)
    mtime = paths[0].stat().st_mtime_ns
    subprocess.run([sys.executable, "-c", script], env=env, check=True, capture_output=True, timeout=30)
    assert paths[0].stat().st_mtime_ns == mtime


def test_compile_failure_is_raised_and_can_be_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import fcntl

    monkeypatch.setattr(megatron_utils, "_HELPER_CACHE_ROOT", tmp_path)
    compile_helper.cache_clear()
    with patch("torch.utils.cpp_extension.load_inline", side_effect=RuntimeError("compiler failed")):
        with pytest.raises(RuntimeError, match="compiler failed"):
            compile_helper()
    assert compile_helper.cache_info().currsize == 0
    # A failed build must release the OS lock as well as leave the LRU empty.
    lock_path = next(tmp_path.glob("*/compile.lock"))
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cache_stays_node_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path / "shared-torch-cache"))
    monkeypatch.setenv("TMPDIR", str(tmp_path / "shared-tmp"))
    compile_helper.cache_clear()
    try:
        with patch("torch.utils.cpp_extension.load_inline") as load:
            compile_helper()
        build_directory = Path(load.call_args.kwargs["build_directory"])
        assert build_directory.is_relative_to(Path("/tmp") / f"nemo_automodel_{os.getuid()}")
        assert not (tmp_path / "shared-torch-cache").exists()
        assert not (tmp_path / "shared-tmp").exists()
    finally:
        compile_helper.cache_clear()
