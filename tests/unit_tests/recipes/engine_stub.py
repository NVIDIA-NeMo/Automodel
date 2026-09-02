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


class RecipeEngineStub:
    """Minimal Engine stand-in for recipe unit tests, matching Engine's signatures."""

    def __init__(self, module, *, optimizer=None, grad_norm=0.0):
        self.module = module
        self.optimizer = optimizer
        self.grad_norm = grad_norm
        self.gradient_accumulation_steps = None

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def backward(self, loss, retain_graph=False, scale_wrt_gas=True):
        del scale_wrt_gas
        loss.backward(retain_graph=retain_graph)

    def step(self):
        if self.optimizer is not None:
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

    def get_global_grad_norm(self):
        return self.grad_norm

    def set_gradient_accumulation_steps(self, steps):
        self.gradient_accumulation_steps = steps
