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

import cli.app as app


def test_cli_accepts_config_positional(tmp_path):
    parser = app.build_parser()
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("recipe:\n  _target_: nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction\n")
    args, _ = parser.parse_known_args([str(cfg)])
    assert args.config == cfg
    assert args.nproc_per_node is None
