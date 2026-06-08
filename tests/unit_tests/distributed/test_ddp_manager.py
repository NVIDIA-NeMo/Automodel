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

from unittest.mock import MagicMock

import torch.nn as nn

from nemo_automodel.components.distributed import ddp as ddp_mod
from nemo_automodel.components.distributed.config import DDPConfig


def test_ddp_manager_forwards_ddp_constructor_flags(monkeypatch):
    monkeypatch.setattr(ddp_mod.dist, "is_available", lambda: True, raising=True)
    monkeypatch.setattr(ddp_mod.dist, "is_initialized", lambda: True, raising=True)
    monkeypatch.setattr(ddp_mod.dist, "get_rank", lambda: 0, raising=True)
    monkeypatch.setattr(ddp_mod.dist, "get_world_size", lambda: 2, raising=True)

    ddp_ctor = MagicMock(return_value="wrapped")
    monkeypatch.setattr(ddp_mod, "DDP", ddp_ctor, raising=True)

    manager = ddp_mod.DDPManager(
        DDPConfig(
            backend="gloo",
            broadcast_buffers=True,
            find_unused_parameters=True,
            static_graph=True,
            bucket_cap_mb=64,
            gradient_as_bucket_view=True,
        )
    )

    model = nn.Linear(2, 2)
    assert manager.parallelize(model) == "wrapped"

    ddp_ctor.assert_called_once()
    assert ddp_ctor.call_args.args[0] is model
    assert ddp_ctor.call_args.kwargs["device_ids"] is None
    assert ddp_ctor.call_args.kwargs["broadcast_buffers"] is True
    assert ddp_ctor.call_args.kwargs["find_unused_parameters"] is True
    assert ddp_ctor.call_args.kwargs["static_graph"] is True
    assert ddp_ctor.call_args.kwargs["bucket_cap_mb"] == 64
    assert ddp_ctor.call_args.kwargs["gradient_as_bucket_view"] is True
