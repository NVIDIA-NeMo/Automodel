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

"""Model-private kernels for MiniMax M3 sparse attention (MSA).

Holds the SM100 training backward for main attention: the KV-parallel CuTe DSL
kernel, its ``sum(O * dO)`` delta preprocess, and the CPU-side task/CTA schedule
derived from the forward work items. Nothing is re-exported: ``msa.py`` resolves
the backward launcher by module path so a MiniMax M3 import never pulls in the
CuTe DSL. The sparse prefill forward is not vendored; it comes from the optional
``fmha_sm100`` package at first use.
"""
