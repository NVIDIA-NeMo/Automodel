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

import argparse
import importlib
import tempfile
from importlib.metadata import version

import torch
from packaging.version import Version
from torch.utils.cpp_extension import load_inline


def main() -> None:
    """Compile a C++20 PyTorch extension and load required native modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-image", choices=("cuda", "pytorch"), default="cuda")
    args = parser.parse_args()

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
    # The PyTorch image uses the upgraded native stack. The CUDA image keeps
    # the package minimum from pyproject.toml and its separate dependency lock.
    minimum_te_version = Version("2.20.2" if args.base_image == "pytorch" else "2.19.0")
    if installed_te_version < minimum_te_version:
        raise RuntimeError(
            f"Transformer Engine {minimum_te_version} or later is required, found {installed_te_version}"
        )

    if installed_te_version >= Version("2.20.2"):
        installed_frontend_version = Version(version("nvidia-cudnn-frontend"))
        if installed_frontend_version < Version("1.29.0"):
            raise RuntimeError(
                f"Transformer Engine 2.20.2 requires cuDNN Frontend 1.29.0 or later, found {installed_frontend_version}"
            )
        importlib.import_module("cudnn")

    # Importing the framework binding eagerly catches stale extensions whose
    # PyTorch or CUDA symbols cannot be resolved by the final image runtime.
    importlib.import_module("transformer_engine.pytorch")

    flash_mla_version = version("flash-mla")
    if flash_mla_version != "1.0.0+b7643bd":
        raise RuntimeError(f"FlashMLA 1.0.0+b7643bd is required, found {flash_mla_version}")
    flash_mla = importlib.import_module("flash_mla")
    if not callable(getattr(flash_mla, "flash_mla_sparse_fwd", None)):
        raise RuntimeError("FlashMLA does not provide the required flash_mla_sparse_fwd kernel")

    # MoK recipes import both modules; loading them also checks the compiled
    # mok._C extension against the final image's PyTorch/CUDA libraries.
    importlib.import_module("mok.functional")
    importlib.import_module("mok.ops")


if __name__ == "__main__":
    main()
