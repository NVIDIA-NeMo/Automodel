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

"""Verify native-extension compatibility while building the container image."""

import importlib
import tempfile
from importlib.metadata import version

import torch
from packaging.version import Version
from torch.utils.cpp_extension import load_inline


def main() -> None:
    """Compile a C++20 PyTorch extension and load required native modules."""
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("The installed PyTorch build does not provide torch._grouped_mm")

    cpp_source = """
        #include <torch/extension.h>

        static_assert(__cplusplus >= 202002L, "C++20 or later is required");

        long cxx_standard() {
            return __cplusplus;
        }
    """
    with tempfile.TemporaryDirectory(prefix="automodel-cxx20-") as build_directory:
        probe = load_inline(
            name="automodel_cxx20_probe",
            cpp_sources=cpp_source,
            functions=["cxx_standard"],
            extra_cflags=["-std=c++20"],
            build_directory=build_directory,
            with_cuda=False,
            verbose=True,
            keep_intermediates=False,
        )
        if probe.cxx_standard() < 202002:
            raise RuntimeError("The PyTorch extension was not compiled with C++20 or later")

    installed_te_version = Version(version("transformer-engine"))
    if installed_te_version < Version("2.19.0"):
        raise RuntimeError(f"Transformer Engine 2.19.0 or later is required, found {installed_te_version}")

    # Importing the framework binding eagerly catches stale extensions whose
    # PyTorch or CUDA symbols cannot be resolved by the final image runtime.
    importlib.import_module("transformer_engine.pytorch")


if __name__ == "__main__":
    main()
