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

"""Storage-orientation contract between grouped-GEMM modules and layout-aware optimizers."""


class TransposedMatrixModule:
    """Marker base class for modules whose matrix parameters store ``[..., input, output]``.

    ``nn.Linear`` stores ``[output, input]``. Grouped-GEMM expert modules (``GroupedExperts``,
    ``GroupedExpertsDeepEP`` and their LoRA and MXFP4 subclasses) store every trainable matrix
    parameter, including LoRA factors, as ``[experts, input, output]`` so each expert computes
    ``x @ W``. Optimizers whose update depends on the output-neuron axis (Muown) check
    ``isinstance(module, TransposedMatrixModule)`` to normalize along that axis. Biases are not
    matrices and are routed by the optimizer's own rules. The class carries no state or methods.
    """
