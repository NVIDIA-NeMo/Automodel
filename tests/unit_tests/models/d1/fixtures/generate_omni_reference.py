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

"""Generate offline Omni trunk/head goldens from the pinned publisher encoder.py.

Pass a local copy of encoder.py from the source URL in omni_reference.json.
The source SHA256 is checked before importing. No NeMo code, perception modules,
network calls, or trained publisher weights participate in generation.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch


def main() -> None:
    """Write random tiny publisher state, outputs, and all gradients as numeric NPZ arrays."""
    source = Path(sys.argv[1])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "5f2e20319ee42a7367febf06c7653d7b3f9b28ef99a64cb9ec7739563dcfcffe":
        raise ValueError("publisher encoder source differs from the pinned reference")
    spec = importlib.util.spec_from_file_location("d1_omni_publisher_reference", source)
    publisher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = publisher
    spec.loader.exec_module(publisher)
    torch.set_num_threads(1)
    torch.manual_seed(1739)
    config = dict(
        vocab_size=32,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        layer_types=["conv", "full_attention"],
        norm_eps=1e-5,
        conv_L_cache=3,
        conv_bias=False,
        block_ffn_dim_multiplier=1.0,
        block_multiple_of=16,
        block_auto_adjust_ff_dim=True,
        rope_theta=1e6,
    )
    encoder = publisher.Trunk(config).float().eval()
    head = publisher.DecisionHead(64, 2).float().eval()
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, publisher.RMSNorm):
                module.weight.copy_(torch.linspace(0.65, 1.35, module.weight.numel()))
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7], [8, 9, 3, 2, 1, 0, 0], [10, 11, 5, 6, 7, 8, 0]])
    valid = torch.tensor([[True] * 7, [True] * 5 + [False] * 2, [True] * 6 + [False]])
    prefix = torch.tensor([0, 2, 1])
    positions = torch.tensor([[1, 3, 5], [0, 1, 2], [1, 3, 0]])
    markers = torch.tensor([[True] * 3, [True] * 3, [True, True, False]])
    types = torch.tensor([0, 1, 2])
    perturbation = (torch.randn(3, 7, 64) * 0.1).requires_grad_()
    hidden = encoder(encoder.embed_tokens(ids) + perturbation, valid, prefix)
    lengths = valid.sum(dim=1) - prefix
    text = torch.nn.utils.rnn.pad_sequence(
        [
            hidden[index, int(offset) : int(offset + length)]
            for index, (offset, length) in enumerate(zip(prefix, lengths))
        ],
        batch_first=True,
    )
    text_valid = torch.arange(text.shape[1])[None] < lengths[:, None]
    logits = head(text, text_valid, positions, markers, types)
    upstream = torch.randn_like(logits).masked_fill(~markers, 0)
    logits.backward(upstream)
    tensors = {
        "input.ids": ids,
        "input.valid": valid,
        "input.prefix": prefix,
        "input.positions": positions,
        "input.markers": markers,
        "input.types": types,
        "input.perturbation": perturbation.detach(),
        "input.upstream": upstream,
        "expected.hidden": hidden.detach(),
        "expected.logits": logits.detach(),
        "expected.input_gradient": perturbation.grad,
    }
    absent = []
    for namespace, module in (("encoder", encoder), ("head", head)):
        tensors.update({f"state.{namespace}.{name}": value for name, value in module.state_dict().items()})
        for name, parameter in module.named_parameters():
            key = f"gradient.{namespace}.{name}"
            if parameter.grad is None:
                absent.append(key)
            else:
                tensors[key] = parameter.grad
    output = Path(__file__).with_name("omni_reference.npz")
    np.savez_compressed(output, allow_pickle=False, **{name: value.detach().numpy() for name, value in tensors.items()})
    metadata = {
        "source_url": "https://huggingface.co/LiquidAI/d1-omni-600M/blob/02b55d7076f15129e59ab3f94783f32c4b088674/encoder.py",
        "source_sha256": digest,
        "archive_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "torch_version": torch.__version__,
        "seed": 1739,
        "dtype": "float32",
        "text_config": config,
        "head_layers": 2,
        "absent_gradients": absent,
        "array_count": len(tensors),
        "description": "Random tiny publisher trunk/head; no trained weights. Text/media-prefix/padding and all question types.",
    }
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
