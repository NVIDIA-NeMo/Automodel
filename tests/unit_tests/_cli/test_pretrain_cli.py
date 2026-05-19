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

import nemo_automodel.cli.app as app


def test_cli_accepts_config_positional(tmp_path):
    parser = app.build_parser()
    cfg = tmp_path / "cfg.yaml"
    recipe_target = "nemo_automodel.recipes.llm.train_ft.TrainFinetuneRecipeForNextTokenPrediction"
    cfg.write_text(f"recipe:\n  _target_: {recipe_target}\n")

    args, _ = parser.parse_known_args([str(cfg)])
    config_path, config, generated_config = app._resolve_invocation(args, parser)

    assert args.inputs == [str(cfg)]
    assert config_path == cfg.resolve()
    assert config["recipe"]["_target_"] == recipe_target
    assert generated_config is False
    assert args.nproc_per_node is None
