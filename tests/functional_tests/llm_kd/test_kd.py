from tests.utils.test_utils import run_test_script


TEST_FOLDER = "llm_kd"
KD_SCRIPT = "L2_KD_Transformer_SFT.sh"


class TestKDRecipe:
    def test_kd_recipe_runs(self):
        run_test_script(TEST_FOLDER, KD_SCRIPT)


