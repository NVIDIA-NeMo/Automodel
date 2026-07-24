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
import importlib
import os
import sys
import types
from pathlib import Path
from shutil import rmtree

import pytest
import torch

os.environ.setdefault("HF_CACHE", "/home/TestData/lite/hf_cache")
os.environ.setdefault("HF_HOME", "/home/TestData/HF_HOME")

# ---------------------------------------------------------------------------
# Shim: ``transformers.initialization`` was added in transformers >=4.48.
# Older versions keep ``no_init_weights`` in ``transformers.modeling_utils``.
# Provide a thin compatibility module so that
# ``from transformers.initialization import no_init_weights`` works regardless
# of the installed version.
# ---------------------------------------------------------------------------
if "transformers.initialization" not in sys.modules:
    try:
        importlib.import_module("transformers.initialization")
    except ModuleNotFoundError:
        from transformers.modeling_utils import no_init_weights

        _compat = types.ModuleType("transformers.initialization")
        _compat.no_init_weights = no_init_weights
        sys.modules["transformers.initialization"] = _compat

# Ensure tests import the in-repo sources (not an installed site-packages copy).
# This is important when `nemo_automodel` is also installed in the environment.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_addoption(parser):
    """Additional command-line arguments passed to pytest.
    For now:
        --cpu: use CPU during testing (DEFAULT: GPU)
        --use_local_test_data: use local test data/skip downloading from URL/GitHub (DEFAULT: False)
    """
    parser.addoption(
        "--cpu", action="store_true", help="pass that argument to use CPU during testing (DEFAULT: False = GPU)"
    )
    parser.addoption(
        "--with_downloads",
        action="store_true",
        help="pass this argument to active tests which download models from the cloud.",
    )


@pytest.fixture
def device(request):
    """Simple fixture returning string denoting the device [CPU | GPU]"""
    if request.config.getoption("--cpu"):
        return "CPU"
    else:
        return "GPU"


@pytest.fixture(autouse=True)
def run_only_on_device_fixture(request, device):
    """Fixture to skip tests based on the device"""
    if request.node.get_closest_marker("run_only_on"):
        if request.node.get_closest_marker("run_only_on").args[0] != device:
            pytest.skip("skipped on this device: {}".format(device))


@pytest.fixture(autouse=True)
def downloads_weights(request, device):
    """Fixture to validate if the with_downloads flag is passed if necessary"""
    if request.node.get_closest_marker("with_downloads"):
        if not request.config.getoption("--with_downloads"):
            pytest.skip(
                "To run this test, pass --with_downloads option. It will download (and cache) models from cloud."
            )


@pytest.fixture(autouse=True)
def cleanup_local_folder():
    """Cleanup local experiments folder"""
    # Asserts in fixture are not recommended, but I'd rather stop users from deleting expensive training runs
    assert not Path("./NeMo_experiments").exists()
    assert not Path("./nemo_experiments").exists()

    yield

    if Path("./NeMo_experiments").exists():
        rmtree("./NeMo_experiments", ignore_errors=True)
    if Path("./nemo_experiments").exists():
        rmtree("./nemo_experiments", ignore_errors=True)


@pytest.fixture(autouse=True)
def reset_env_vars():
    """Reset environment variables"""
    # Store the original environment variables before the test
    original_env = dict(os.environ)

    # Run the test
    yield

    # After the test, restore the original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture(autouse=True)
def enforce_torch_memory_limit(request):
    """Enforce opt-in per-test PyTorch allocator budgets."""
    marker = request.node.get_closest_marker("torch_memory_limit")
    if marker is None:
        yield
        return

    cpu_limit_mb = marker.kwargs.get("cpu_mb")
    cuda_limit_mb = marker.kwargs.get("cuda_mb")
    if cpu_limit_mb is None and cuda_limit_mb is None:
        pytest.fail("torch_memory_limit requires cpu_mb and/or cuda_mb")

    cuda_device = None
    cuda_allocated_before = 0
    if cuda_limit_mb is not None and torch.cuda.is_available():
        cuda_device = torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_device)
        cuda_allocated_before = torch.cuda.memory_allocated(cuda_device)

    if cpu_limit_mb is None:
        yield
        profiler = None
    else:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU],
            profile_memory=True,
            acc_events=True,
        ) as profiler:
            yield

    bytes_per_mb = 1024**2
    if profiler is not None:
        cpu_allocated_bytes = sum(max(0, event.self_cpu_memory_usage) for event in profiler.key_averages())
        assert cpu_allocated_bytes <= cpu_limit_mb * bytes_per_mb, (
            f"test allocated {cpu_allocated_bytes / bytes_per_mb:.1f} MiB through the PyTorch CPU allocator; "
            f"limit is {cpu_limit_mb} MiB"
        )

    if cuda_device is not None:
        cuda_peak_delta_bytes = torch.cuda.max_memory_allocated(cuda_device) - cuda_allocated_before
        assert cuda_peak_delta_bytes <= cuda_limit_mb * bytes_per_mb, (
            f"test allocated {cuda_peak_delta_bytes / bytes_per_mb:.1f} MiB through the PyTorch CUDA allocator; "
            f"limit is {cuda_limit_mb} MiB"
        )


def pytest_configure(config):
    """Initial configuration of conftest.
    The function checks if test_data.tar.gz is present in tests/.data.
    If so, compares its size with github's test_data.tar.gz.
    If file absent or sizes not equal, function downloads the archive from github and unpacks it.
    """
    config.addinivalue_line(
        "markers",
        "run_only_on(device): runs the test only on a given device [CPU | GPU]",
    )
    config.addinivalue_line(
        "markers",
        "with_downloads: runs the test using data present in tests/.data",
    )
    config.addinivalue_line(
        "markers",
        "torch_memory_limit(cpu_mb=None, cuda_mb=None): limits per-test PyTorch allocator usage",
    )
