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

"""Offline reference parity for the Omni numerical trunk and marker head."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from transformers import Lfm2Config

from nemo_automodel.components.models.d1.decision_head import D1OmniHeadConfig
from nemo_automodel.components.models.d1.encoder import D1OmniEncoder


def test_omni_encoder_head_and_all_gradients_match_pinned_publisher():
    path = Path(__file__).with_name("fixtures") / "omni_reference.npz"
    metadata = json.loads(path.with_suffix(".json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == metadata["archive_sha256"]
    with np.load(path, allow_pickle=False) as archive:
        values = {name: torch.from_numpy(archive[name]) for name in archive.files}
    assert len(values) == metadata["array_count"]
    with torch.device("cpu"):
        encoder = D1OmniEncoder(Lfm2Config(**metadata["text_config"])).float().eval()
        head = D1OmniHeadConfig(64, metadata["head_layers"]).build().float().eval()
    for namespace, module in (("encoder", encoder), ("head", head)):
        prefix = f"state.{namespace}."
        module.load_state_dict(
            {name.removeprefix(prefix): value for name, value in values.items() if name.startswith(prefix)}, strict=True
        )
    perturbation = values["input.perturbation"].clone().requires_grad_()
    prefix = values["input.prefix"]
    valid = values["input.valid"]
    hidden = encoder(encoder.embed_tokens(values["input.ids"]) + perturbation, valid, prefix)
    torch.testing.assert_close(hidden, values["expected.hidden"], atol=3e-6, rtol=3e-5)
    lengths = valid.sum(dim=1) - prefix
    text = torch.nn.utils.rnn.pad_sequence(
        [
            hidden[index, int(offset) : int(offset + length)]
            for index, (offset, length) in enumerate(zip(prefix, lengths))
        ],
        batch_first=True,
    )
    text_valid = torch.arange(text.shape[1])[None] < lengths[:, None]
    logits = head(text, text_valid, values["input.positions"], values["input.markers"], values["input.types"])
    torch.testing.assert_close(logits, values["expected.logits"], atol=3e-6, rtol=3e-5)
    logits.backward(values["input.upstream"])
    torch.testing.assert_close(perturbation.grad, values["expected.input_gradient"], atol=3e-6, rtol=3e-5)
    for namespace, module in (("encoder", encoder), ("head", head)):
        for name, parameter in module.named_parameters():
            key = f"gradient.{namespace}.{name}"
            if key in metadata["absent_gradients"]:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, values[key], atol=3e-6, rtol=3e-5, msg=key)
