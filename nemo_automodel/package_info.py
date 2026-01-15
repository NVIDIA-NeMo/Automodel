# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

try:
    import contextlib
    import sys
    from pathlib import Path

    @contextlib.contextmanager
    def temp_sys_path(path: Path):
        sys.path.insert(0, str(path))
        try:
            yield
        finally:
            sys.path.pop(0)

    with temp_sys_path(Path(__file__).parent.parent / "build_utils"):
        from build_backend import dynamic_version_info

        __version__, __git_version__ = dynamic_version_info()
except ModuleNotFoundError:
    # Fallbacks for running directly from the source tree before _version.py is generated
    __git_version__ = "gitunknown"
    __version__ = "0.0.0"

__package_name__ = "nemo_automodel"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Automodel"
__download_url__ = "https://github.com/NVIDIA-NeMo/Automodel/releases"
__description__ = "NeMo Automodel - Delivers zero-day integration with Hugging Face models, \
                   automating fine-tuning and pretraining with built-in parallelism, \
                   custom-kernels and optimized recipes"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch"
