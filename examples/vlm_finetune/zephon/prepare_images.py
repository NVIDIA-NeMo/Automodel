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

"""Prepare a small finite image-conversation source for the Zephon VLM example."""

import json
from pathlib import Path

from PIL import Image


def main() -> None:
    """Write local PNG fixtures and native-format conversation JSONL."""
    root = Path("work/zephon-vlm").resolve()
    images = root / "images"
    corpus = root / "corpus"
    images.mkdir(parents=True, exist_ok=True)
    corpus.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, color, size in (("red", "red", (64, 96)), ("blue", "blue", (128, 64))):
        path = images / f"{name}.png"
        Image.new("RGB", size, color=color).save(path)
        paths.append(path)
    documents = []
    for index in range(128):
        chosen = paths if index % 3 == 0 else [paths[index % 2]]
        content = [{"type": "image", "image": str(path)} for path in chosen]
        content.append({"type": "text", "text": "Name the colors."})
        documents.append(
            {
                "conversation": [
                    {"role": "user", "content": content},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": " and ".join(path.stem for path in chosen)}],
                    },
                ]
            }
        )
    (corpus / "images.jsonl").write_text("".join(json.dumps(document) + "\n" for document in documents))
    print(f"Wrote {len(documents)} image conversations to {corpus}")


if __name__ == "__main__":
    main()
