# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import argparse

import nemo_automodel._cli.app as app


def test_cli_accepts_pretrain(tmp_path, monkeypatch):
    parser = app.build_parser()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("dummy: 1")
    args = parser.parse_args(["pretrain", "llm", "-c", str(cfg)])
    assert args.command == "pretrain"
    assert args.domain == "llm"
    assert args.config == cfg 