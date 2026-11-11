"""CPU checks of the world-action fine-tuning recipe: its discovery and its seeding."""

from nemo_automodel.cli.utils import resolve_recipe_name
from nemo_automodel.recipes.wam.train import draw_seed


def test_recipe_is_discoverable_by_its_bare_name():
    assert resolve_recipe_name("TrainWAMFinetuneRecipe") == "nemo_automodel.recipes.wam.train.TrainWAMFinetuneRecipe"


def test_draw_seed_separates_ranks_micro_steps_and_steps():
    seeds = {draw_seed(3, step, micro, rank, 8) for step in range(3) for micro in range(6) for rank in range(8)}
    assert len(seeds) == 3 * 6 * 8
    assert draw_seed(3, 0, 0, 0, 8) == 3 * 1_000_003
