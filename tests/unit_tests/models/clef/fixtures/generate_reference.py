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

"""Generate random tiny-head goldens using only the pinned publisher implementation.

Download joint_schema_model.py from the source URL recorded in reference.json,
then pass its local path as the sole argument. Generation verifies its SHA256
before importing it; no NeMo implementation or publisher weights are used.
"""

import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import torch


class _CharacterTokenizer:
    def __call__(self, text: str, *, add_special_tokens: bool) -> SimpleNamespace:
        return SimpleNamespace(input_ids=[ord(character) for character in text])


def main() -> None:
    """Write reproducible CPU float32 reference outputs beside this generator."""
    source = Path(sys.argv[1])
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != "0e304cf7c6500e8bb59bef7e2afd2c6373f82596dfb3b57d1aa93c175e2dc3a3":
        raise ValueError("publisher source differs from the pinned CLEF reference")
    spec = importlib.util.spec_from_file_location("clef_publisher_reference", source)
    publisher = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = publisher
    spec.loader.exec_module(publisher)

    torch.set_num_threads(1)
    torch.manual_seed(1739)
    config = dict(hidden_size=4, width=4, routing_layers=1, layers=1, heads=1, feedforward=8, dropout=0.0)
    head = publisher.JointSchemaHead(**config).eval()
    hidden = torch.randn(2, 8, 4, requires_grad=True)
    lexical = torch.randn(16, 4, requires_grad=True)
    input_ids = torch.tensor([[0, 2, 3, 4, 5, 6, 7, 8], [0, 8, 7, 6, 5, 4, 0, 0]])
    mask = torch.tensor([[1] * 8, [1] * 6 + [0, 0]])
    records = [
        publisher.EncodedRecord(
            tuple(input_ids[0].tolist()),
            (
                publisher.EncodedQuestion("flag", 0, (0, 2), ((2, 3), (3, 4)), ("true", "false")),
                publisher.EncodedQuestion("pick", 1, (1, 3), ((3, 4), (4, 5), (5, 6)), ("a", "b", "c")),
                publisher.EncodedQuestion("level", 2, (2, 4), ((4, 5), (6, 7)), ("0", "1")),
            ),
            "first",
        ),
        publisher.EncodedRecord(
            tuple(input_ids[1, :6].tolist()),
            (publisher.EncodedQuestion("pick", 1, (0, 2), ((2, 4), (4, 6)), ("a", "b")),),
            "second",
        ),
    ]
    logits = head(hidden, input_ids, mask, records, lexical)
    flat = [row for record in logits for row in record]
    targets = [1, 2, 1, 0]
    loss = torch.stack(
        [
            torch.nn.functional.cross_entropy(row.unsqueeze(0), torch.tensor([target]))
            for row, target in zip(flat, targets)
        ]
    ).mean()
    loss.backward()
    record = {
        "id": "prompt",
        "state": {"z": "observation", "a": "caf\u00e9"},
        "questions": {
            "flag": {"type": "noul", "instructions": "", "criteria": {"true": "Yes"}},
            "pick": {"type": "choice", "instructions": {"task": "Choose"}, "criteria": {"z": None, "a": "Alpha"}},
            "level": {"type": "score", "criteria": ["Low", "High"]},
        },
    }
    encodings = []
    for max_state_tokens in (None, 7):
        encoded = publisher.encode_record(_CharacterTokenizer(), record, max_state_tokens=max_state_tokens)
        encodings.append(dict(max_state_tokens=max_state_tokens, expected=asdict(encoded)))
    result = {
        "provenance": {
            "source_url": "https://huggingface.co/Cloudflare/clef/blob/ed3eed331870db2eff4b0db01237128ede8a00ce/joint_schema_model.py",
            "source_sha256": digest,
            "torch_version": torch.__version__,
            "seed": 1739,
            "dtype": "float32",
            "description": "Random tiny publisher head; no trained model weights. Character-tokenizer prompt contract.",
        },
        "prompt_record": record,
        "encodings": encodings,
        "head_config": config,
        "state_dict": {name: value.tolist() for name, value in head.state_dict().items()},
        "hidden_states": hidden.detach().tolist(),
        "lexical_rows": lexical.detach().tolist(),
        "input_ids": input_ids.tolist(),
        "attention_mask": mask.tolist(),
        "records": [asdict(record) for record in records],
        "targets": targets,
        "logits": [[row.detach().tolist() for row in record] for record in logits],
        "loss": loss.item(),
        "hidden_gradient": hidden.grad.tolist(),
        "lexical_gradient": lexical.grad.tolist(),
        "parameter_gradients": {name: parameter.grad.tolist() for name, parameter in head.named_parameters()},
    }
    Path(__file__).with_name("reference.json").write_text(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
