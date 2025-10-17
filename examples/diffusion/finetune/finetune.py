from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.diffusion.finetune import TrainWan21DiffusionRecipe


def main(default_config_path="examples/diffusion/finetune/wan2_1_t2v_flow.yaml"):
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainWan21DiffusionRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()


