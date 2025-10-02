# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import types
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_automodel.recipes.llm.benchmark import BenchmarkingRecipeForNextTokenPrediction


@pytest.fixture
def patch_torch_distributed_for_benchmark():
    """
    Patch torch.distributed for benchmark tests.

    This fixture provides stubs for torch.distributed methods that are commonly used
    in the benchmarking code but require a multi-GPU/multi-process environment.
    """
    # Save original torch.distributed
    original_distributed = torch.distributed

    # Distributed stubs
    dist_stub = types.ModuleType("torch.distributed")

    # Minimal API surface
    dist_stub.get_world_size = lambda: 1
    dist_stub.get_rank = lambda: 0
    dist_stub.barrier = lambda group=None: None
    dist_stub.is_initialized = lambda: False
    dist_stub.send = lambda tensor, dst: None
    dist_stub.recv = lambda tensor, src: None

    def _all_gather(dest: torch.Tensor, src: torch.Tensor, group=None, async_op=False):
        """Dummy all_gather for single process."""
        dest.copy_(src)
        return None

    dist_stub.all_gather_into_tensor = _all_gather
    dist_stub._all_gather_base = _all_gather

    # Patch torch.distributed
    torch.distributed = dist_stub

    yield

    # Restore original
    torch.distributed = original_distributed


@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    config = SimpleNamespace(
        benchmark=SimpleNamespace(
            warmup_steps=10,
            peak_tflops=989,
            nsys_start=-1,
            nsys_end=-1,
            nsys_ranks=[],
        ),
        step_scheduler=SimpleNamespace(
            max_steps=30,
            global_batch_size=256,
            local_batch_size=4,
            ckpt_every_steps=1000,
            val_every_steps=1000,
            num_epochs=1,
        ),
        dataset=SimpleNamespace(
            seq_len=2048,
            vocab_size=50257,
            batch_size=4,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(
                pretrained_model_name_or_path="gpt2",
            ),
        ),
    )
    return config


@pytest.fixture
def mock_recipe(mock_config):
    """Create a mock benchmarking recipe instance."""
    with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
        recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)
        recipe.cfg = mock_config
        recipe.timers = MagicMock()
        recipe.dist_env = MagicMock()
        recipe.dist_env.rank = 0
        recipe.dist_env.world_size = 8
        recipe.dist_env.device = torch.device("cpu")
        recipe.dist_env.is_main = True
        recipe.model_parts = [MagicMock()]
        recipe.model_parts[0].config = SimpleNamespace(
            hidden_size=768,
            num_hidden_layers=12,
            vocab_size=50257,
            num_attention_heads=12,
            intermediate_size=3072,
        )
        recipe.optimizer = [MagicMock()]
        recipe.dataloader = MagicMock()
        recipe.val_dataloader = None
        recipe.pp_enabled = False
        recipe.tflops = 1000.0
        # Set benchmark-specific attributes
        recipe._bench_steps = 30
        recipe._bench_warmup_steps = 10
        recipe._bench_peak_tflops = 989
        recipe._bench_nsys_start = -1
        recipe._bench_nsys_end = -1
        recipe._bench_nsys_ranks = []
        recipe._bench_seq_len = 2048
        return recipe


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeInitialization:
    """Test initialization of BenchmarkingRecipeForNextTokenPrediction."""

    def test_init_extracts_benchmark_params(self, mock_config):
        """Test that __init__ correctly extracts benchmarking parameters from benchmark section."""
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._bench_steps == 30  # from step_scheduler.max_steps
            assert recipe._bench_warmup_steps == 10  # from benchmark.warmup_steps
            assert recipe._bench_peak_tflops == 989  # from benchmark.peak_tflops
            assert recipe._bench_nsys_start == -1  # from benchmark.nsys_start
            assert recipe._bench_nsys_end == -1  # from benchmark.nsys_end
            assert recipe._bench_nsys_ranks == []  # from benchmark.nsys_ranks
            assert recipe._bench_seq_len == 2048  # from dataset.seq_len

    def test_init_infers_max_steps_from_step_scheduler(self, mock_config):
        """Test that max_steps is inferred from step_scheduler."""
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._bench_steps == mock_config.step_scheduler.max_steps

    def test_init_infers_vocab_size(self, mock_config):
        """Test that vocab_size is inferred from model config."""
        mock_model_config = MagicMock()
        mock_model_config.vocab_size = 50257

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            with patch("transformers.AutoConfig.from_pretrained", return_value=mock_model_config):
                recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

                assert mock_config.dataset.vocab_size == 50257

    def test_init_sets_batch_size_from_scheduler(self, mock_config):
        """Test that batch_size is set from step_scheduler."""
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert mock_config.dataset.batch_size == 4


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeSetup:
    """Test setup method of BenchmarkingRecipeForNextTokenPrediction."""

    @patch("nemo_automodel.recipes.llm.benchmark.get_flops_formula_for_hf_config")
    def test_setup_clears_val_dataloader(self, mock_get_flops, mock_recipe):
        """Test that setup clears the validation dataloader."""
        mock_flops_formula = MagicMock(return_value=1e15)
        mock_get_flops.return_value = mock_flops_formula

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"):
            mock_recipe.setup()

            assert mock_recipe.val_dataloader is None

    @patch("nemo_automodel.recipes.llm.benchmark.get_flops_formula_for_hf_config")
    def test_setup_calculates_tflops(self, mock_get_flops, mock_recipe):
        """Test that setup calculates TFLOPs correctly."""
        expected_flops = 1e15
        mock_flops_formula = MagicMock(return_value=expected_flops)
        mock_get_flops.return_value = mock_flops_formula

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"):
            mock_recipe.setup()

            expected_tflops = expected_flops / (10**12)
            assert mock_recipe.tflops == expected_tflops
            mock_flops_formula.assert_called_once()


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeRunBenchmark:
    """Test run_benchmark method of BenchmarkingRecipeForNextTokenPrediction."""

    def test_run_benchmark_sets_models_to_train_mode(self, mock_recipe):
        """Test that run_benchmark sets all models to training mode."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        mock_recipe._forward_backward_step = MagicMock()
        # Mock timers to return a dict with the expected structure
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        # Mock timer objects for active_time calls
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }
        # Need 30 iterations * 8 ga_steps = 240 batches
        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 240
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

            for model_part in mock_recipe.model_parts:
                model_part.train.assert_called_once()

    def test_run_benchmark_calculates_gradient_accumulation_steps(self, mock_recipe):
        """Test that gradient accumulation steps are calculated correctly."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        mock_recipe._forward_backward_step = MagicMock()
        # Mock timers to return a dict with the expected structure
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        # Mock timer objects for active_time calls
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }
        # Need 30 iterations * 8 ga_steps = 240 batches
        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 240
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

            # global_batch_size=256, local_batch_size=4, dp_size=8
            # ga_steps = 256 / (4 * 8) = 8
            expected_ga_steps = 8
            # Verify forward_backward_step was called expected_ga_steps times per iteration
            assert mock_recipe._forward_backward_step.call_count == 30 * expected_ga_steps

    def test_run_benchmark_zero_grads_per_iteration(self, mock_recipe):
        """Test that gradients are zeroed at the start of each iteration."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        mock_recipe._forward_backward_step = MagicMock()
        # Mock timers to return a dict with the expected structure
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        # Mock timer objects for active_time calls
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }
        # Need 30 iterations * 8 ga_steps = 240 batches
        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 240
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

            # Should be called 30 times (once per iteration)
            assert mock_recipe.optimizer[0].zero_grad.call_count == 30

    def test_run_benchmark_optimizer_step_per_iteration(self, mock_recipe):
        """Test that optimizer step is called once per iteration."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        mock_recipe._forward_backward_step = MagicMock()
        # Mock timers to return a dict with the expected structure
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        # Mock timer objects for active_time calls
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }
        # Need 30 iterations * 8 ga_steps = 240 batches
        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 240
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

            # Should be called 30 times (once per iteration)
            assert mock_recipe.optimizer[0].step.call_count == 30


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeHelpers:
    """Test helper methods and edge cases."""

    def test_benchmark_with_invalid_ga_config(self, mock_recipe):
        """Test that invalid gradient accumulation config raises assertion."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        # Set invalid batch sizes that will cause assertion error
        # global_batch_size=16, local_batch_size=4, dp_size=8
        # ga_steps = 16 / (4 * 8) = 0.5 (not divisible)
        mock_recipe.cfg.step_scheduler.local_batch_size = 4
        mock_recipe.cfg.step_scheduler.global_batch_size = 16

        with pytest.raises(AssertionError, match="Global batch size must be divisible"):
            mock_recipe.run_benchmark()

    def test_init_requires_benchmark_section(self):
        """Test that __init__ raises error if benchmark section is missing."""
        config_without_benchmark = SimpleNamespace(
            step_scheduler=SimpleNamespace(max_steps=30),
            dataset=SimpleNamespace(seq_len=2048),
        )

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            with pytest.raises(AttributeError):
                recipe = BenchmarkingRecipeForNextTokenPrediction(config_without_benchmark)
