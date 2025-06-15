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

# taken and edited from https://github.com/pytorch/pytorch/blob/6ebe9a4f47e9cd1c9ccd467bcdfdea9445fd98d6/torch/distributed/checkpoint/_hf_planner.py # pylint: disable=line-too-long

# mypy: allow-untyped-defs

from torch.distributed.checkpoint.default_planner import (
    create_default_local_load_plan,
    DefaultLoadPlanner,
)
from torch.distributed.checkpoint.planner import LoadPlan


__all__ = ["_HuggingFaceSavePlanner", "_HuggingFaceLoadPlanner"]


class _HuggingFaceLoadPlanner(DefaultLoadPlanner):
    def __init__(self, allow_tensor_resize: bool = False):
        super().__init__()
        self.allow_tensor_resize = allow_tensor_resize

    def create_local_plan(self) -> LoadPlan:
        assert self.metadata is not None

        # check_md_size is added to avoid the check if we're allowing tensor resize.
        # This will be deprecated in favor of _load_state_dict_from_keys and then we
        # can remove this planner all together.
        return create_default_local_load_plan(
            self.state_dict,
            self.metadata,
            not self.allow_partial_load,
            check_md_size=not self.allow_tensor_resize,
        )