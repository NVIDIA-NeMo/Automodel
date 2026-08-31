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

from torch import nn

from nemo_automodel.shared.model_utils import iter_transformer_and_mtp_blocks


def test_iter_transformer_and_mtp_blocks_finds_inner_decoder_mtp_layers():
    backbone_block = nn.Linear(2, 2)
    mtp_block = nn.Linear(2, 2)
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([backbone_block])
    decoder.mtp_layers = nn.ModuleList([mtp_block])
    model = nn.Module()
    model.model = decoder

    yielded = list(iter_transformer_and_mtp_blocks(model))

    assert yielded == [
        (decoder.layers, "0", backbone_block),
        (decoder.mtp_layers, "0", mtp_block),
    ]
