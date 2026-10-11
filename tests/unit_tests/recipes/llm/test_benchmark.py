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
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import LlamaConfig, PretrainedConfig

from nemo_automodel.components.loss.masked_ce import MaskedCrossEntropy
from nemo_automodel.components.moe.megatron.moe_utils import MoEAuxLossAutoScaler
from nemo_automodel.components.utils.flops_utils import llama2_flops
from nemo_automodel.recipes.llm.benchmark import (
    BenchmarkingRecipeForNextTokenPrediction,
    _CudaMemoryPeak,
    _infer_vocab_size,
    _IterationMetrics,
    main,
)


class ConfigNamespace(SimpleNamespace):
    """A SimpleNamespace that also supports dict-like .get() method."""

    def get(self, key, default=None):
        """Get attribute with a default, supporting dotted keys (like ConfigNode/RecipeConfig.get)."""
        obj = self
        for part in key.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return default
        return obj


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
def mock_config(monkeypatch):
    """Create a mock configuration for testing."""
    monkeypatch.setattr(
        "transformers.AutoConfig.from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(vocab_size=50257),
    )
    config = ConfigNamespace(
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
def mock_recipe(mock_config, monkeypatch):
    """Create a mock benchmarking recipe instance."""
    monkeypatch.setattr(torch.cuda, "set_device", MagicMock())
    monkeypatch.setattr(torch.cuda.nvtx, "range_push", MagicMock())
    monkeypatch.setattr(torch.cuda.nvtx, "range_pop", MagicMock())
    with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
        recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)
        recipe.cfg = mock_config
        recipe.timers = MagicMock()
        recipe.dist_env = MagicMock()
        recipe.dist_env.rank = 0
        recipe.dist_env.world_size = 8
        recipe.dist_env.device = torch.device("cpu")
        recipe.dist_env.is_main = True
        monkeypatch.setattr(
            recipe,
            "_collect_cuda_memory_peaks",
            lambda: [
                _CudaMemoryPeak(rank, (rank + 1) * 1024, (rank + 1) * 2048, rank, 0)
                for rank in range(recipe.dist_env.world_size)
            ],
        )
        recipe.model_parts = [MagicMock()]
        recipe.model_parts[0].config = LlamaConfig(
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
        recipe._bench_all_token_head_tflops = 1000.0
        recipe._bench_iterations = [_IterationMetrics(i, 0.5, 1000.0, 1000.0, 0, 0) for i in range(30)]
        # Set benchmark-specific attributes
        recipe._bench_steps = 30
        recipe._bench_warmup_steps = 10
        recipe._bench_peak_tflops = 989
        recipe._bench_nsys_start = -1
        recipe._bench_nsys_end = -1
        recipe._bench_nsys_ranks = []
        recipe._bench_seq_len = 2048
        recipe._bench_json_output_path = None
        recipe._wandb_enabled = False
        recipe.wandb_run = None
        recipe.partial_cuda_graph_manager = None
        recipe._partial_cuda_graph_capture_pending = False
        recipe.step_scheduler = SimpleNamespace(step=0, gc_every_steps=None)
        recipe._dp_allreduce = MagicMock(side_effect=lambda x, include_cp=False: x)
        recipe.device_mesh = None
        recipe.moe_mesh = None
        recipe.loss_fn = MaskedCrossEntropy()
        recipe.max_grad_norm = None
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

    @pytest.mark.parametrize("scope", ["model", "text"])
    def test_init_accepts_flops_scope(self, mock_config: ConfigNamespace, scope: str) -> None:
        mock_config.benchmark.flops_scope = scope
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)
        assert recipe._bench_flops_scope == scope

    def test_init_rejects_invalid_flops_scope(self, mock_config: ConfigNamespace) -> None:
        mock_config.benchmark.flops_scope = "vision"
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__") as parent:
            with pytest.raises(ValueError, match="benchmark.flops_scope must be"):
                BenchmarkingRecipeForNextTokenPrediction(mock_config)
        parent.assert_not_called()

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
                BenchmarkingRecipeForNextTokenPrediction(mock_config)

                assert mock_config.dataset.vocab_size == 50257

    def test_infer_vocab_size_string_target(self, mock_config):
        """Test _infer_vocab_size with a string _target_ (custom config class)."""
        mock_model_config = MagicMock()
        mock_model_config.vocab_size = 131072

        model_cfg = ConfigNamespace(
            config=ConfigNamespace(
                _target_="some.package.module.CustomConfig",
                pretrained_model_name_or_path="some-model",
            )
        )

        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.CustomConfig.from_pretrained.return_value = mock_model_config
            mock_import.return_value = mock_module

            vocab_size = _infer_vocab_size(model_cfg)

            mock_import.assert_called_once_with("some.package.module")
            mock_module.CustomConfig.from_pretrained.assert_called_once_with("some-model")
            assert vocab_size == 131072

    def test_infer_vocab_size_trust_remote_code_at_model_level(self, mock_config):
        """trust_remote_code at model level (not nested under config) must flow to AutoConfig."""
        mock_model_config = MagicMock(spec=["vocab_size"])
        mock_model_config.vocab_size = 163840

        model_cfg = ConfigNamespace(
            trust_remote_code=True,
            config=ConfigNamespace(
                _target_="transformers.AutoConfig.from_pretrained",
                pretrained_model_name_or_path="moonshotai/Kimi-K2-Base",
            ),
        )

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_model_config) as mock_autoconfig:
            vocab_size = _infer_vocab_size(model_cfg)
            assert vocab_size == 163840
            mock_autoconfig.assert_called_once_with("moonshotai/Kimi-K2-Base", trust_remote_code=True)

    def test_infer_vocab_size_retries_layer_type_validation_error(self, mock_config):
        """Step-style MTP metadata should not prevent benchmark vocab inference."""
        model_cfg = ConfigNamespace(
            trust_remote_code=True,
            config=ConfigNamespace(
                _target_="transformers.AutoConfig.from_pretrained",
                pretrained_model_name_or_path="stepfun-ai/Step-3.5-Flash",
            ),
        )
        model_config = MagicMock(spec=["vocab_size"])
        model_config.vocab_size = 151936
        validation_error = ValueError("num_hidden_layers must equal layer_types")

        with (
            patch(
                "transformers.AutoConfig.from_pretrained",
                side_effect=[validation_error, model_config],
            ) as mock_autoconfig,
            patch("nemo_automodel._transformers.v4_patches.layer_types.relax_layer_types_validator") as relax,
        ):
            assert _infer_vocab_size(model_cfg) == 151936

        relax.assert_called_once()
        assert mock_autoconfig.call_count == 2

    def test_infer_vocab_size_method_target(self, mock_config):
        """`_target_` resolved to a classmethod (e.g. `Class.from_pretrained`) should be invoked directly."""
        mock_model_config = MagicMock(spec=["vocab_size"])
        mock_model_config.vocab_size = 129280

        def factory(path):
            assert path == "deepseek-ai/DeepSeek-V3.2"
            return mock_model_config

        model_cfg = ConfigNamespace(
            config=ConfigNamespace(
                _target_=factory,
                pretrained_model_name_or_path="deepseek-ai/DeepSeek-V3.2",
            )
        )

        vocab_size = _infer_vocab_size(model_cfg)
        assert vocab_size == 129280

    def test_infer_vocab_size_vl_text_config(self, mock_config):
        """Test _infer_vocab_size falls back to text_config.vocab_size for VL models."""
        mock_model_config = MagicMock(spec=[])
        mock_model_config.text_config = MagicMock()
        mock_model_config.text_config.vocab_size = 151936

        model_cfg = ConfigNamespace(
            config=ConfigNamespace(
                _target_="transformers.AutoConfig.from_pretrained",
                pretrained_model_name_or_path="some-vl-model",
            )
        )

        with patch("transformers.AutoConfig.from_pretrained", return_value=mock_model_config):
            vocab_size = _infer_vocab_size(model_cfg)
            assert vocab_size == 151936

    def test_init_sets_batch_size_from_scheduler(self, mock_config):
        """Test that batch_size is set from step_scheduler."""
        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert mock_config.dataset.batch_size == 4


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeSetup:
    """Test setup method of BenchmarkingRecipeForNextTokenPrediction."""

    @pytest.mark.parametrize("scope", ["model", "text"])
    def test_setup_requires_explicit_text_scope_for_composite(
        self, mock_recipe: BenchmarkingRecipeForNextTokenPrediction, scope: str
    ) -> None:
        text = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
        )
        mock_recipe.model_parts[0].config = PretrainedConfig(text_config=text)
        mock_recipe._bench_flops_scope = scope
        with (
            patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"),
            patch.object(mock_recipe, "_mtp_tflops", return_value=0.0),
        ):
            if scope == "model":
                with pytest.raises(ValueError, match="No FLOPs formula.*flops_scope='model'"):
                    mock_recipe.setup()
            else:
                mock_recipe.setup()
                expected = llama2_flops(text, gbs=256, seq_len=2048) / 1e12
                assert mock_recipe.tflops == pytest.approx(expected)

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

        # Mock _forward_backward_step to append loss to loss_buffer
        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)
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

        # Mock _forward_backward_step to append loss to loss_buffer
        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)
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

        # Mock _forward_backward_step to append loss to loss_buffer
        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)
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
        graph_manager = MagicMock()
        sync_tp_replicas = MagicMock()
        mock_recipe.partial_cuda_graph_manager = graph_manager
        mock_recipe._partial_cuda_graph_capture_pending = True

        # Mock _forward_backward_step to append loss to loss_buffer
        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)
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

        with (
            patch("torch.distributed.barrier"),
            patch(
                "nemo_automodel.recipes.llm.benchmark.synchronize_tp_replica_gradients",
                sync_tp_replicas,
            ),
        ):
            mock_recipe.run_benchmark()

            # Should be called 30 times (once per iteration)
            assert mock_recipe.optimizer[0].step.call_count == 30
            assert sync_tp_replicas.call_count == 30
            sync_tp_replicas.assert_called_with(mock_recipe.model_parts, mock_recipe.device_mesh)
            graph_manager.capture.assert_called_once_with()
            assert mock_recipe._partial_cuda_graph_capture_pending is False

    def test_run_benchmark_calls_gc_hook_per_iteration(self, mock_recipe):
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        mock_recipe._maybe_collect_garbage = MagicMock()

        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }
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

        assert mock_recipe._maybe_collect_garbage.call_count == 30


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
        config_without_benchmark = ConfigNamespace(
            step_scheduler=SimpleNamespace(max_steps=30),
            dataset=SimpleNamespace(seq_len=2048),
        )

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            with pytest.raises(AttributeError):
                BenchmarkingRecipeForNextTokenPrediction(config_without_benchmark)

    def test_main_closes_partial_cuda_graphs_when_setup_fails(self):
        recipe = MagicMock()
        graph_manager = MagicMock()
        recipe.partial_cuda_graph_manager = graph_manager
        recipe.setup.side_effect = RuntimeError("setup failed")

        with (
            patch("nemo_automodel.recipes.llm.benchmark.parse_args_and_load_config", return_value=MagicMock()),
            patch(
                "nemo_automodel.recipes.llm.benchmark.BenchmarkingRecipeForNextTokenPrediction",
                return_value=recipe,
            ),
            pytest.raises(RuntimeError, match="setup failed"),
        ):
            main("config.yaml")

        recipe.run_benchmark.assert_not_called()
        graph_manager.close.assert_called_once_with()
        assert recipe.partial_cuda_graph_manager is None


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeWandBIntegration:
    """Test WandB integration features."""

    def test_wandb_enabled_when_config_present(self, mock_config):
        """Test that wandb is enabled when wandb config is present."""
        mock_config.wandb = SimpleNamespace(project="test-project", name="test-run")

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._wandb_enabled is True

    def test_wandb_disabled_when_config_absent(self, mock_config):
        """Test that wandb is disabled when wandb config is absent."""
        # mock_config doesn't have wandb attribute by default

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._wandb_enabled is False

    @patch("nemo_automodel.recipes.llm.benchmark.get_flops_formula_for_hf_config")
    @patch("wandb.run")
    def test_setup_logs_to_wandb_when_enabled(self, mock_wandb_run, mock_get_flops, mock_recipe):
        """Test that setup logs timers to wandb when enabled."""
        mock_recipe._wandb_enabled = True
        # Use __dict__ to bypass state tracking
        mock_recipe.__dict__["wandb_run"] = mock_wandb_run

        mock_flops_formula = MagicMock(return_value=1e15)
        mock_get_flops.return_value = mock_flops_formula

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"):
            mock_recipe.setup()

            # Verify wandb logging was called
            mock_recipe.timers.write_to_wandb.assert_called_once_with(
                names=["setup"],
                writer=mock_wandb_run,
                iteration=0,
                normalizer=1.0,
                reset=False,
                barrier=True,
            )

    @patch("nemo_automodel.recipes.llm.benchmark.get_flops_formula_for_hf_config")
    def test_setup_skips_wandb_when_disabled(self, mock_get_flops, mock_recipe):
        """Test that setup skips wandb logging when disabled."""
        mock_recipe._wandb_enabled = False
        mock_recipe.wandb_run = None

        mock_flops_formula = MagicMock(return_value=1e15)
        mock_get_flops.return_value = mock_flops_formula

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"):
            mock_recipe.setup()

            # Verify wandb logging was NOT called
            mock_recipe.timers.write_to_wandb.assert_not_called()

    def test_log_iteration_metrics_with_wandb(self, mock_recipe):
        """Test that iteration metrics are logged to wandb."""
        mock_recipe._wandb_enabled = True
        mock_recipe.wandb_run = MagicMock()
        mock_recipe.timers._get_global_min_max_time = MagicMock(return_value={"iteration": (0.0, 1.5)})

        with patch("nemo_automodel.recipes.llm.benchmark.calculate_mfu", return_value=50.0):
            mock_recipe._log_iteration_metrics("iteration", ga_steps=4, peak_tflops=989, rank=0, iteration=10)

            # Verify wandb logging was called with correct parameters
            expected_timer_names = [
                "iteration",
                "optimizer",
                "forward_backward_0",
                "forward_backward_1",
                "forward_backward_2",
                "forward_backward_3",
            ]
            mock_recipe.timers.write_to_wandb.assert_called_once_with(
                names=expected_timer_names,
                writer=mock_recipe.wandb_run,
                iteration=10,
                normalizer=1.0,
                reset=False,
                barrier=False,
            )


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeJSONOutput:
    """Test JSON output functionality."""

    def test_json_output_path_stored(self, mock_config):
        """Test that json_output_path is stored from config."""
        mock_config.benchmark.json_output_path = "/tmp/benchmark_results.json"

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._bench_json_output_path == "/tmp/benchmark_results.json"

    def test_json_output_default_none(self, mock_config):
        """Test that json_output_path defaults to None."""
        # mock_config.benchmark doesn't have json_output_path

        with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
            recipe = BenchmarkingRecipeForNextTokenPrediction(mock_config)

            assert recipe._bench_json_output_path is None

    @pytest.mark.parametrize("scope", ["model", "text"])
    def test_log_benchmark_summary_creates_json_file(
        self, mock_recipe: BenchmarkingRecipeForNextTokenPrediction, tmp_path: Path, scope: str
    ) -> None:
        """Test that benchmark summary is written to JSON file."""
        json_file = tmp_path / "benchmark_summary.json"
        mock_recipe._bench_json_output_path = str(json_file)
        mock_recipe._bench_flops_scope = scope

        # Mock timers
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 10.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        with patch("torch.distributed.barrier"):
            mock_recipe._log_benchmark_summary(steps=30, warmup_steps=10, peak_tflops=989, rank=0)

        # Verify JSON file was created
        assert json_file.exists()

        # Verify JSON content
        import json

        with open(json_file) as f:
            summary_data = json.load(f)

        assert summary_data["flops_scope"] == scope
        assert summary_data["total_steps"] == 30
        assert summary_data["warmup_steps"] == 10
        assert summary_data["training_steps"] == 20
        assert summary_data["peak_tflops"] == 989
        assert summary_data["world_size"] == 8
        assert summary_data["global_batch_size"] == 256
        assert summary_data["local_batch_size"] == 4
        assert summary_data["seq_len"] == 2048
        assert "avg_iter_time_seconds" in summary_data
        assert "avg_mfu_percent" in summary_data
        assert (
            summary_data["cuda_memory_peak_scope"] == "setup_warmup_and_measured_iterations_before_summary_collectives"
        )
        assert len(summary_data["cuda_memory_peaks"]) == 8
        assert summary_data["cuda_memory_peaks"][7] == {
            "rank": 7,
            "max_allocated_bytes": 8192,
            "max_reserved_bytes": 16384,
            "num_alloc_retries": 7,
            "num_ooms": 0,
        }
        assert summary_data["cuda_allocator_counter_scope"] == "process_lifetime_before_summary_collectives"
        assert summary_data["cuda_max_allocated_bytes"] == 8192
        assert summary_data["cuda_max_reserved_bytes"] == 16384

    def test_log_benchmark_summary_skips_json_when_none(self, mock_recipe, tmp_path):
        """Test that JSON output is skipped when path is None."""
        mock_recipe._bench_json_output_path = None

        # Mock timers
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 10.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        with patch("torch.distributed.barrier"):
            mock_recipe._log_benchmark_summary(steps=30, warmup_steps=10, peak_tflops=989, rank=0)

        # Verify no JSON file was created
        assert not any(tmp_path.glob("*.json"))


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeSummaryData:
    """Test benchmark summary data structure and WandB logging."""

    @pytest.mark.parametrize("scope", ["model", "text"])
    @patch("wandb.log")
    @patch("wandb.Table")
    def test_summary_wandb_table_structure(
        self,
        mock_table_class: MagicMock,
        mock_wandb_log: MagicMock,
        mock_recipe: BenchmarkingRecipeForNextTokenPrediction,
        scope: str,
    ) -> None:
        """Test that wandb table identifies the selected FLOPs scope."""
        mock_recipe._bench_flops_scope = scope
        mock_recipe.wandb_run = MagicMock()
        mock_table_instance = MagicMock()
        mock_table_class.return_value = mock_table_instance

        # Mock timers
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 100.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        with patch("torch.distributed.barrier"):
            with patch("wandb.finish"):
                mock_recipe._log_benchmark_summary(steps=30, warmup_steps=10, peak_tflops=989, rank=0)

        # Verify Table was created with correct columns
        mock_table_class.assert_called_once()
        call_kwargs = mock_table_class.call_args[1]
        assert call_kwargs["columns"] == ["Metric", "Value"]

        # Verify data includes expected metrics
        data = call_kwargs["data"]
        metric_names = [row[0] for row in data]
        assert "Total Steps" in metric_names
        assert "Warmup Steps" in metric_names
        assert "Training Steps" in metric_names
        expected_mfu_label = "Avg MFU (%)" if scope == "model" else "Avg text-backbone MFU (%)"
        assert expected_mfu_label in metric_names
        assert "TFLOPs/GPU/s" in metric_names
        assert "World Size" in metric_names
        assert "Global Batch Size" in metric_names
        assert "Sequence Length" in metric_names

    @pytest.mark.parametrize("scope", ["model", "text"])
    @patch("wandb.log")
    def test_summary_wandb_scalar_metrics(
        self, mock_wandb_log: MagicMock, mock_recipe: BenchmarkingRecipeForNextTokenPrediction, scope: str
    ) -> None:
        """Test that scalar metrics retain their FLOPs scope in W&B."""
        mock_recipe._bench_flops_scope = scope
        mock_recipe.wandb_run = MagicMock()

        # Mock timers
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 100.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        with patch("torch.distributed.barrier"):
            with patch("wandb.Table"):
                with patch("wandb.finish"):
                    mock_recipe._log_benchmark_summary(steps=30, warmup_steps=10, peak_tflops=989, rank=0)

        # Verify scalar metrics were logged
        calls = mock_wandb_log.call_args_list
        # Find the call with scalar metrics
        scalar_call = None
        for call in calls:
            if "summary/avg_iter_time_seconds" in call[0][0]:
                scalar_call = call
                break

        assert scalar_call is not None
        scalar_metrics = scalar_call[0][0]
        assert "summary/avg_iter_time_seconds" in scalar_metrics
        assert "summary/avg_mfu_percent" in scalar_metrics
        assert scalar_metrics["summary/flops_scope"] == scope
        assert "summary/training_time_seconds" in scalar_metrics
        assert "summary/tflops_per_gpu" in scalar_metrics

    @patch("wandb.finish")
    def test_wandb_finish_called(self, mock_wandb_finish, mock_recipe):
        """Test that wandb.finish() is called at the end."""
        mock_recipe.wandb_run = MagicMock()

        # Mock timers
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 100.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        with patch("torch.distributed.barrier"):
            with patch("wandb.log"):
                with patch("wandb.Table"):
                    mock_recipe._log_benchmark_summary(steps=30, warmup_steps=10, peak_tflops=989, rank=0)

        # Verify wandb.finish was called
        mock_wandb_finish.assert_called_once()


@pytest.mark.usefixtures("patch_torch_distributed_for_benchmark")
class TestBenchmarkingRecipeLossCalculation:
    """Test loss calculation and synchronization."""

    def test_loss_buffer_accumulation(self, mock_recipe):
        """Test that losses are accumulated in buffer during gradient accumulation."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)

        # Track loss_buffer contents
        captured_loss_buffers = []

        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5 + ga_step_idx * 0.1))
                captured_loss_buffers.append(list(loss_buffer))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)

        # Mock timers
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        # Run only 1 iteration to test
        mock_recipe._bench_steps = 1
        mock_recipe._bench_warmup_steps = 0

        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 8
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

        # Verify loss_buffer was populated correctly (8 GA steps)
        assert len(captured_loss_buffers[-1]) == 8

    def test_dp_allreduce_called_for_loss(self, mock_recipe):
        """Test that DP allreduce is called for loss synchronization."""
        mock_recipe._get_dp_group_size = MagicMock(return_value=8)
        dp_allreduce_calls = []

        def track_allreduce(tensor, include_cp=False):
            dp_allreduce_calls.append({"tensor_shape": tensor.shape, "include_cp": include_cp})
            return tensor

        # Use __dict__ to bypass state tracking
        mock_recipe.__dict__["_dp_allreduce"] = MagicMock(side_effect=track_allreduce)

        # Mock forward_backward_step
        def mock_forward_backward_step(ga_step_idx, batch, loss_buffer=None, **kwargs):
            if loss_buffer is not None:
                loss_buffer.append(torch.tensor(0.5))

        mock_recipe._forward_backward_step = MagicMock(side_effect=mock_forward_backward_step)

        # Mock timers
        mock_recipe.timers._get_global_min_max_time = MagicMock(
            return_value={"iteration_warmup": (0.0, 1.0), "iteration": (0.0, 1.0)}
        )
        mock_timer = MagicMock()
        mock_timer.active_time.return_value = 1.0
        mock_recipe.timers._timers = {
            "setup": mock_timer,
            "iteration": mock_timer,
            "iteration_warmup": mock_timer,
        }

        # Run only 1 iteration
        mock_recipe._bench_steps = 1
        mock_recipe._bench_warmup_steps = 0

        mock_recipe.dataloader.__iter__ = MagicMock(
            return_value=iter(
                [
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "labels": torch.tensor([[1, 2, 3]]),
                        "position_ids": torch.tensor([[0, 1, 2]]),
                    }
                ]
                * 8
            )
        )

        with patch("torch.distributed.barrier"):
            mock_recipe.run_benchmark()

        # Verify DP allreduce was called
        # Should be called twice per iteration: once for num_label_tokens, once for loss
        assert len(dp_allreduce_calls) >= 2

        # Verify loss reduction includes context parallelism
        loss_reduction_calls = [call for call in dp_allreduce_calls if call["include_cp"]]
        assert len(loss_reduction_calls) >= 1


class _BenchmarkGradientModel(torch.nn.Module):
    """Tiny real logits model; an optional auxiliary loss has its own parameter."""

    def __init__(self, *, with_auxiliary: bool = False) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.2, -0.3], [-0.4, 0.6], [0.7, 0.1]]))
        self.auxiliary = torch.nn.Parameter(torch.tensor(0.3)) if with_auxiliary else None

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute logits, optionally attaching the production auxiliary scaler.

        Args:
            input_ids: Integer token IDs of shape [batch, sequence], each zero or one.

        Returns:
            Logits of shape [batch, sequence, vocab=3].
        """
        features = torch.nn.functional.one_hot(input_ids, num_classes=2).float()
        logits = torch.nn.functional.linear(features, self.weight)
        if self.auxiliary is not None:
            auxiliary_loss = self.auxiliary.square() * (1.0 + input_ids.float().mean())
            logits = MoEAuxLossAutoScaler.apply(logits, auxiliary_loss)
        return logits


def _gradient_batches(rank: int, ga_steps: int, ignore_index: int) -> list[dict[str, torch.Tensor]]:
    """Build unequal supervision, retaining both dataset and loss ignore sentinels."""
    batches = []
    for microbatch in range(ga_steps):
        input_ids = torch.tensor([[(rank + microbatch + token) % 2 for token in range(4)]])
        labels = torch.tensor([[(2 * rank + microbatch + token) % 3 for token in range(4)]])
        valid_count = 1 + (rank + 2 * microbatch) % 4
        labels[:, valid_count:] = ignore_index
        if valid_count < 4:
            labels[:, -1] = -100
        batches.append({"input_ids": input_ids, "labels": labels})
    return batches


def _gradient_recipe(
    model: torch.nn.Module,
    batches: list[dict[str, torch.Tensor]],
    *,
    ga_steps: int,
    rank: int = 0,
    world_size: int = 1,
    ignore_index: int = -100,
    max_grad_norm: float | None = None,
) -> BenchmarkingRecipeForNextTokenPrediction:
    """Construct the CPU fixture without replacing the native loop or backward.

    Args:
        model: Local model or real DDP wrapper producing [batch, sequence, vocab] logits.
        batches: Microbatches containing input_ids and labels, both [batch=1, sequence=4].
        ga_steps: Number of microbatches in the optimizer step.
        rank: Real process rank.
        world_size: Real data-parallel process count.
        ignore_index: Loss-specific ignored label.
        max_grad_norm: Optional global gradient norm limit.

    Returns:
        Recipe using native loss, forward/backward, DP reduction, and SGD.
    """
    config = ConfigNamespace(
        benchmark=SimpleNamespace(warmup_steps=0, peak_tflops=989, nsys_start=-1, nsys_end=-1, nsys_ranks=[]),
        step_scheduler=SimpleNamespace(max_steps=1, local_batch_size=1, global_batch_size=ga_steps * world_size),
        dataset=SimpleNamespace(seq_len=4),
    )
    with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
        recipe = BenchmarkingRecipeForNextTokenPrediction(config)
    recipe.cfg = config
    recipe.tflops = 1.0
    recipe.dist_env = SimpleNamespace(rank=rank, world_size=world_size, device=torch.device("cpu"), is_main=rank == 0)
    recipe.model_parts = [model]
    recipe.optimizer = [torch.optim.SGD(model.parameters(), lr=0.07)]
    recipe.dataloader = batches
    recipe.loss_fn = MaskedCrossEntropy(ignore_index=ignore_index)
    recipe.max_grad_norm = max_grad_norm
    recipe.device_mesh = None
    recipe.moe_mesh = None
    recipe.pp_enabled = False
    recipe.tokenizer = None
    recipe.domain_mixture = None
    recipe.te_fp8 = None
    recipe.distributed_config = SimpleNamespace(defer_fsdp_grad_sync=True)
    recipe.partial_cuda_graph_manager = None
    recipe._partial_cuda_graph_capture_pending = False
    recipe.step_scheduler = SimpleNamespace(step=0, gc_every_steps=None)
    recipe.timers = MagicMock()
    recipe._wandb_enabled = True
    recipe.wandb_run = MagicMock()
    recipe._log_iteration_metrics = MagicMock()
    recipe._log_benchmark_summary = MagicMock()
    recipe._collect_moe_load_balance = MagicMock()
    recipe._log_moe_metrics = MagicMock()
    return recipe


def _mean_gradient_oracle(
    batches: list[dict[str, torch.Tensor]],
    *,
    ignore_index: int,
    max_grad_norm: float | None,
    with_auxiliary: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float]:
    """Differentiate one concatenated valid-token mean, independently of native helpers.

    Args:
        batches: All ranks' microbatches, each containing [1, sequence=4] IDs and labels.
        ignore_index: Loss-specific ignored label in addition to dataset sentinel -100.
        max_grad_norm: Optional clipping bound applied to the complete reference gradient.
        with_auxiliary: Include the mathematical mean of per-microbatch auxiliary losses.

    Returns:
        Named complete reference gradients, post-SGD parameters, and scalar main loss.
        Weight tensors have shape [vocab=3, hidden=2]; auxiliary tensors are scalars.
    """
    model = _BenchmarkGradientModel(with_auxiliary=with_auxiliary)
    ids = torch.cat([batch["input_ids"].reshape(-1) for batch in batches])
    labels = torch.cat([batch["labels"].reshape(-1) for batch in batches])
    valid = (labels != -100) & (labels != ignore_index)
    features = torch.nn.functional.one_hot(ids, num_classes=2).float()
    logits = features @ model.weight.T
    main_loss = (
        torch.nn.functional.cross_entropy(logits[valid], labels[valid], reduction="mean")
        if valid.any()
        else logits.sum() * 0.0
    )
    objective = main_loss
    if with_auxiliary:
        objective = (
            objective
            + model.auxiliary.square()
            * torch.stack([1.0 + batch["input_ids"].float().mean() for batch in batches]).mean()
        )
    objective.backward()
    gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    if max_grad_norm is not None:
        norm = torch.sqrt(sum(gradient.double().square().sum() for gradient in gradients.values()))
        coefficient = min(1.0, max_grad_norm / (norm.item() + 1e-6))
        gradients = {name: gradient * coefficient for name, gradient in gradients.items()}
    updated = {name: parameter.detach() - 0.07 * gradients[name] for name, parameter in model.named_parameters()}
    return gradients, updated, main_loss.item()


def _run_gradient_recipe(recipe: BenchmarkingRecipeForNextTokenPrediction) -> None:
    """Disable CUDA-only instrumentation while retaining the native CPU lifecycle."""
    with (
        patch.object(torch.cuda, "is_available", return_value=False),
        patch.object(torch.cuda.nvtx, "range_push"),
        patch.object(torch.cuda.nvtx, "range_pop"),
        patch.object(torch.cuda, "max_memory_allocated", return_value=0),
    ):
        recipe.run_benchmark()


def _assert_gradient_result(
    model: _BenchmarkGradientModel,
    recipe: BenchmarkingRecipeForNextTokenPrediction,
    expected: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], float],
) -> None:
    """Compare complete gradients/SGD updates and the independently derived mean loss.

    Args:
        model: Complete local model after its native optimizer step.
        recipe: Executed recipe whose rank-zero loss is retained by a logging mock.
        expected: Named gradients and parameters [vocab=3, hidden=2] (optional scalar
            auxiliary), followed by the scalar reference main loss.
    """
    gradients, updated, main_loss = expected
    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter.grad, gradients[name], rtol=3e-6, atol=3e-7)
        torch.testing.assert_close(parameter, updated[name], rtol=3e-6, atol=3e-7)
    if recipe.dist_env.rank == 0:
        losses = [call.args[0]["loss"] for call in recipe.wandb_run.log.call_args_list if "loss" in call.args[0]]
        assert losses == pytest.approx([main_loss], rel=3e-6, abs=3e-7)


@pytest.mark.parametrize("ga_steps", [1, 3])
@pytest.mark.parametrize("ignore_index", [-100, -7])
@pytest.mark.parametrize("max_grad_norm", [None, 0.1])
def test_benchmark_global_mean_gradient(ga_steps: int, ignore_index: int, max_grad_norm: float | None) -> None:
    """Actual inherited backward must match a single valid-token mean and SGD."""
    batches = _gradient_batches(0, ga_steps, ignore_index)
    model = _BenchmarkGradientModel()
    recipe = _gradient_recipe(model, batches, ga_steps=ga_steps, ignore_index=ignore_index, max_grad_norm=max_grad_norm)
    expected = _mean_gradient_oracle(batches, ignore_index=ignore_index, max_grad_norm=max_grad_norm)
    _run_gradient_recipe(recipe)
    _assert_gradient_result(model, recipe, expected)


def test_benchmark_zero_supervised_tokens() -> None:
    """An all-ignored window has finite zero main loss and no SGD update."""
    batches = _gradient_batches(0, 2, -100)
    for batch in batches:
        batch["labels"].fill_(-100)
    model = _BenchmarkGradientModel()
    recipe = _gradient_recipe(model, batches, ga_steps=2, max_grad_norm=0.1)
    expected = _mean_gradient_oracle(batches, ignore_index=-100, max_grad_norm=0.1)
    _run_gradient_recipe(recipe)
    _assert_gradient_result(model, recipe, expected)


@pytest.mark.parametrize("ga_steps", [1, 3])
def test_benchmark_auxiliary_gradient_scale(ga_steps: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """A stale auxiliary scale must be replaced by the current accumulation mean."""
    monkeypatch.setattr(MoEAuxLossAutoScaler, "main_loss_backward_scale", torch.tensor(7.0))
    batches = _gradient_batches(0, ga_steps, -100)
    model = _BenchmarkGradientModel(with_auxiliary=True)
    recipe = _gradient_recipe(model, batches, ga_steps=ga_steps)
    expected = _mean_gradient_oracle(batches, ignore_index=-100, max_grad_norm=None, with_auxiliary=True)
    _run_gradient_recipe(recipe)
    _assert_gradient_result(model, recipe, expected)


def _benchmark_dp_gradient_worker(rank: int, rendezvous: str) -> None:
    """Exercise real two-process Gloo DDP with unequal rank-local supervision."""
    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=90)
    )
    errors = []
    try:
        for ga_steps in (1, 2):
            for max_grad_norm in (None, 0.1):
                model = _BenchmarkGradientModel()
                wrapped = torch.nn.parallel.DistributedDataParallel(model)
                batches = _gradient_batches(rank, ga_steps, -7)
                all_batches = [batch for peer in range(2) for batch in _gradient_batches(peer, ga_steps, -7)]
                recipe = _gradient_recipe(
                    wrapped,
                    batches,
                    ga_steps=ga_steps,
                    rank=rank,
                    world_size=2,
                    ignore_index=-7,
                    max_grad_norm=max_grad_norm,
                )
                expected = _mean_gradient_oracle(all_batches, ignore_index=-7, max_grad_norm=max_grad_norm)
                _run_gradient_recipe(recipe)
                try:
                    _assert_gradient_result(model, recipe, expected)
                except AssertionError as exc:
                    errors.append(f"rank={rank}, GA={ga_steps}, clip={max_grad_norm}: {exc}")
        gathered = [None, None]
        torch.distributed.all_gather_object(gathered, errors)
        assert not any(gathered), str(gathered)
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.runtime_budget(
    90,
    hard_timeout=300,
    reason="Two spawned native-recipe workers require cold imports; spawn isolates prior parent autograd threads.",
)
def test_benchmark_global_mean_gradient_two_process(tmp_path: Path) -> None:
    """Native loss/loop plus actual DP averaging must equal the global mean oracle."""
    # Fork can inherit an initialized autograd thread pool from earlier model
    # tests. Fresh interpreters preserve correctness for integrated suite order.
    torch.multiprocessing.spawn(
        _benchmark_dp_gradient_worker,
        args=((tmp_path / "benchmark_dp_rendezvous").as_uri(),),
        nprocs=2,
        join=True,
    )


def test_prepacked_from_pretrained_config_keeps_dataset_factory_arguments(tmp_path: Path) -> None:
    """A real native YAML parser and load_from_disk factory accept no mock-dataset keys."""
    from datasets import Dataset

    from nemo_automodel.components.config._arg_parser import parse_args_and_load_config

    dataset_path = tmp_path / "packed"
    Dataset.from_dict({"input_ids": [[1, 2, 3, 4]], "labels": [[-100, 2, 3, 4]]}).save_to_disk(dataset_path)
    config_path = tmp_path / "benchmark.yaml"
    config_path.write_text(
        f"""recipe: BenchmarkingRecipeForNextTokenPrediction
benchmark:
  warmup_steps: 1
  peak_tflops: 989
  nsys_start: -1
  nsys_end: -1
  nsys_ranks: []
step_scheduler:
  max_steps: 3
  local_batch_size: 1
  global_batch_size: 1
model:
  _target_: nemo_automodel.NeMoAutoModelForCausalLM.from_pretrained
  pretrained_model_name_or_path: /unused/pretrained-checkpoint
dataset:
  _target_: datasets.load_from_disk
  dataset_path: {dataset_path}
packed_sequence:
  prepacked: true
  packed_sequence_size: 4
"""
    )
    config = parse_args_and_load_config(str(config_path), argv=[])
    with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.__init__"):
        recipe = BenchmarkingRecipeForNextTokenPrediction(config)
    assert recipe._bench_seq_len == 4
    assert config.get("dataset.seq_len") is None
    assert config.get("dataset.batch_size") is None
    assert config.get("dataset.vocab_size") is None
    assert config.dataset.instantiate()[0]["input_ids"] == [1, 2, 3, 4]


@pytest.mark.parametrize("world_size", [1, 8, 128])
def test_summary_uses_sum_of_actual_work_and_slowest_rank_times(mock_recipe, tmp_path: Path, world_size: int) -> None:
    """Warmup, rank-zero active_time and the final pack's FLOPs cannot bias the summary."""
    import json

    mock_recipe._bench_json_output_path = str(tmp_path / "summary.json")
    mock_recipe.dist_env.world_size = world_size
    mock_recipe.wandb_run = MagicMock()
    mock_recipe._bench_iterations = [
        _IterationMetrics(0, 100.0, 90000.0, 99000.0, 100, 20),
        _IterationMetrics(1, 4.0, 2000.0, 2100.0, 100, 40),
        _IterationMetrics(2, 6.0, 5000.0, 5400.0, 100, 50),
    ]
    mock_recipe.tflops = 5000.0
    timer = MagicMock()
    timer.active_time.return_value = 0.01
    mock_recipe.timers._timers = {"setup": timer, "iteration": timer, "iteration_warmup": timer}
    with patch("torch.distributed.barrier"), patch("wandb.log") as wandb_log, patch("wandb.Table"):
        mock_recipe._log_benchmark_summary(steps=3, warmup_steps=1, peak_tflops=989, rank=0)
    result = json.loads(Path(mock_recipe._bench_json_output_path).read_text())
    assert result["training_time_seconds"] == 10.0
    assert result["warmup_time_seconds"] == 100.0
    assert result["avg_iter_time_seconds"] == 5.0
    assert result["mean_tflops_per_step"] == 3500.0
    assert result["tflops_per_gpu"] == 3500.0 / world_size
    assert result["avg_mfu_percent"] == pytest.approx(7000 / (world_size * 10 * 989) * 100)
    assert result["all_token_head_mfu_percent"] == pytest.approx(7500 / (world_size * 10 * 989) * 100)
    scalar_metrics = next(call.args[0] for call in wandb_log.call_args_list if "summary/tflops_per_gpu" in call.args[0])
    assert scalar_metrics["summary/mean_tflops_per_step"] == 3500.0
    assert scalar_metrics["summary/tflops_per_gpu"] == 3500.0 / world_size
    assert scalar_metrics["summary/cuda_max_allocated_bytes"] == 1024 * world_size
    assert scalar_metrics["summary/cuda_max_reserved_bytes"] == 2048 * world_size
    assert [item["seconds"] for item in result["iterations"]] == [100.0, 4.0, 6.0]


def test_iteration_retains_cross_rank_maximum(mock_recipe) -> None:
    mock_recipe._bench_iterations = []
    mock_recipe.timers._get_global_min_max_time.return_value = {"iteration": (1.0, 4.0)}
    mock_recipe._log_iteration_metrics("iteration", ga_steps=1, peak_tflops=989, rank=0, iteration=10)
    assert len(mock_recipe._bench_iterations) == 1
    assert mock_recipe._bench_iterations[0].seconds == 4.0


def _batch_flops_mesh_worker(rank: int, rendezvous: str) -> None:
    """Use real Gloo DP, CP and PP groups to expose duplicated logical-batch counts."""
    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.models.step5.configuration import Step5Config
    from nemo_automodel.components.models.step5.flops import step5_flops
    from nemo_automodel.components.utils.flops_utils import get_batch_flops_counter

    torch.set_num_threads(1)
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    try:
        model_config = Step5Config(num_nextn_predict_layers=0)
        for pp_size, dp_size, cp_size in ((1, 2, 2), (2, 1, 2)):
            mesh = init_device_mesh("cpu", (pp_size, dp_size, cp_size), mesh_dim_names=("pp", "dp", "cp"))
            mesh["dp", "cp"]._flatten(mesh_dim_name="dp_cp")
            dp_rank = mesh["dp"].get_local_rank()
            lengths = [3, 5] if dp_rank == 0 else [8]
            positions = torch.cat([torch.arange(length) for length in lengths]).unsqueeze(0)
            labels = torch.zeros_like(positions)
            labels[:, 0] = -100
            batch = {"input_ids": torch.zeros_like(positions), "labels": labels, "position_ids": positions}
            recipe = _gradient_recipe(_BenchmarkGradientModel(), [batch], ga_steps=1, rank=rank, world_size=4)
            recipe.device_mesh = mesh
            recipe._batch_flops_counter = get_batch_flops_counter(model_config)
            recipe._record_batch_flops([batch], num_label_tokens=7 * dp_size)
            documents = [3, 5] if dp_size == 1 else [3, 5, 8]
            full = sum(step5_flops(model_config, seq_len=length) for length in documents)
            supervised = full - dp_size * 6 * model_config.hidden_size * model_config.vocab_size
            assert recipe.tflops == pytest.approx(supervised / 1e12)
            assert recipe._bench_all_token_head_tflops == pytest.approx(full / 1e12)
            assert recipe._bench_step_tokens == 8 * dp_size
            # Inspect real DTensor local shards without gathering or changing dtype.
            import json

            from torch.distributed.tensor import Shard, distribute_tensor

            wrapped = distribute_tensor(torch.arange(16.0).reshape(8, 2), mesh["dp"], [Shard(0)])
            recipe.model_parts[0].register_parameter("wrapped", torch.nn.Parameter(wrapped))
            audit_dir = Path(rendezvous.removeprefix("file://")).parent / f"audit-pp{pp_size}"
            recipe._bench_loaded_model_audit_path = str(audit_dir)
            recipe._write_loaded_model_audit()
            audit = json.loads((audit_dir / f"rank-{rank}.json").read_text())
            record = next(item for item in audit["tensors"] if item["name"] == "wrapped")
            assert record["global_shape"] == [8, 2]
            assert record["local_shape"] == [8 // dp_size, 2]
            assert record["dtype"] == "torch.float32"
            assert record["placements"] == ["S(0)"]
            torch.testing.assert_close(wrapped.to_local(), torch.arange(16.0).reshape(8, 2).chunk(dp_size)[dp_rank])
    finally:
        torch.distributed.destroy_process_group()


def test_batch_flops_sum_dp_without_cp_or_pp_duplication(tmp_path: Path) -> None:
    """Four actual ranks exercise DP2×CP2 and PP2×CP2 with the native DP reducer."""
    torch.multiprocessing.start_processes(
        _batch_flops_mesh_worker,
        args=((tmp_path / "batch_flops_mesh").as_uri(),),
        nprocs=4,
        join=True,
        start_method="fork",
    )


def test_model_owned_batch_flops_reject_unaccounted_peft(mock_recipe) -> None:
    """A full-training model formula must never overwrite a PEFT-adjusted numerator."""
    from nemo_automodel.components.models.step5.configuration import Step5Config

    mock_recipe.model_parts[0].config = Step5Config(num_nextn_predict_layers=0)
    mock_recipe.cfg.peft = SimpleNamespace(rank=8)
    with patch("nemo_automodel.recipes.llm.benchmark.TrainFinetuneRecipeForNextTokenPrediction.setup"):
        with pytest.raises(ValueError, match="without PEFT"):
            mock_recipe.setup()


def test_native_benchmark_records_actual_batch_hashes_and_detects_changed_labels(tmp_path: Path) -> None:
    """Actual CPU rows, including ignored labels and masks, determine the audit."""
    import hashlib
    import json

    labels = torch.tensor([[-100, 2, 3, -7]])
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "labels": labels,
        "position_ids": torch.tensor([[0, 1, 0, 1]]),
        "attention_mask": torch.tensor([[1, 1, 2, 2]]),
    }
    recipe = _gradient_recipe(_BenchmarkGradientModel(), [batch], ga_steps=1, ignore_index=-7)
    recipe._bench_batch_metadata_path = str(tmp_path / "batches")
    recipe.device_mesh = None
    recipe.dataloader = SimpleNamespace(sampler=range(1))
    expected = hashlib.sha256(b"".join(batch[key][0].numpy().tobytes() for key in batch)).hexdigest()
    recipe._record_batch_metadata([batch], step=0)
    batch["labels"][0, 1] = 7
    recipe._record_batch_metadata([batch], step=1)
    records = [json.loads(line) for line in (tmp_path / "batches/rank-0.jsonl").read_text().splitlines()]
    assert records[0]["rows"][0]["sha256"] == expected
    assert records[0]["rows"][0]["supervised_tokens"] == 2
    assert records[0]["sampler"] == "range"
    assert records[0]["rows"][0]["sha256"] != records[1]["rows"][0]["sha256"]
    assert records[0]["rows"][0]["tensor_sha256"]["input_ids"] == records[1]["rows"][0]["tensor_sha256"]["input_ids"]


def test_loaded_model_audit_is_read_only_and_samples_noncontiguous_parameters(tmp_path: Path) -> None:
    """The native audit observes fp32 buffers, scalars and strided tensors without repair."""
    import hashlib
    import json

    model = torch.nn.Module()
    model.register_parameter("strided", torch.nn.Parameter(torch.arange(10000.0).reshape(100, 100).t()))
    model.register_parameter("norm", torch.nn.Parameter(torch.arange(13.0)))
    model.register_buffer("router_bias", torch.tensor([34.0625, 34.125], dtype=torch.float32))
    model.register_buffer("scalar", torch.tensor(4.0))
    model.register_buffer("empty", torch.empty(0))
    labels = torch.tensor([[1, 2]])
    recipe = _gradient_recipe(model, [{"input_ids": labels, "labels": labels}], ga_steps=1)
    recipe.device_mesh = None
    recipe._bench_loaded_model_audit_path = str(tmp_path / "loaded")
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    recipe._write_loaded_model_audit()
    audit = json.loads((tmp_path / "loaded/rank-0.json").read_text())
    tensors = {item["name"]: item for item in audit["tensors"]}
    assert tensors["router_bias"]["dtype"] == "torch.float32"
    assert tensors["router_bias"]["sample_values"] == [34.0625, 34.125]
    assert (
        tensors["router_bias"]["full_local_sha256"] == hashlib.sha256(model.router_bias.numpy().tobytes()).hexdigest()
    )
    assert tensors["strided"]["hashed_elements"] == 16
    assert tensors["strided"]["full_local_sha256"] is None
    assert tensors["strided"]["sample_values"][0] == 0.0
    assert tensors["strided"]["sample_values"][-1] == 9999.0
    assert tensors["scalar"]["sample_values"] == [4.0]
    assert tensors["empty"]["sample_values"] == []
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name])


def _benchmark_memory_peak_worker(rank: int, rendezvous: str, world_size: int) -> None:
    """Real Gloo collection with CUDA counters modeled before temporary allocation."""
    torch.distributed.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=world_size, timeout=timedelta(seconds=30)
    )
    recipe = object.__new__(BenchmarkingRecipeForNextTokenPrediction)
    recipe.dist_env = SimpleNamespace(device=torch.device("cpu"), world_size=world_size)
    peaks = [1024 * (rank + 1), 4096 * (rank + 1)]
    counters = {"num_alloc_retries": rank + 3, "num_ooms": rank}
    tensor_factory = torch.tensor

    def allocate_collective_buffer(data: list[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        # The reporting buffers must not increase the snapshot being reported.
        peaks[:] = [10**9, 2 * 10**9]
        counters.update(num_alloc_retries=10**9, num_ooms=10**9)
        return tensor_factory(data, dtype=dtype, device=device)

    try:
        with (
            patch.object(torch.cuda, "max_memory_allocated", side_effect=lambda device: peaks[0]),
            patch.object(torch.cuda, "max_memory_reserved", side_effect=lambda device: peaks[1]),
            patch.object(torch.cuda, "memory_stats", side_effect=lambda device: counters.copy()),
            patch.object(torch.cuda, "reset_peak_memory_stats", side_effect=AssertionError("must not reset peaks")),
            patch.object(torch, "tensor", side_effect=allocate_collective_buffer),
        ):
            result = recipe._collect_cuda_memory_peaks()
        assert result == [
            _CudaMemoryPeak(peer, 1024 * (peer + 1), 4096 * (peer + 1), peer + 3, peer) for peer in range(world_size)
        ]
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.runtime_budget(
    90, hard_timeout=300, reason="Spawned benchmark imports isolate distributed workers from prior autograd threads."
)
def test_benchmark_memory_peaks_use_precollective_snapshot(tmp_path: Path, world_size: int) -> None:
    """Every global rank's bytes survive aggregation without counting reporting buffers."""
    torch.multiprocessing.spawn(
        _benchmark_memory_peak_worker,
        args=((tmp_path / "memory_peaks_rendezvous").as_uri(), world_size),
        nprocs=world_size,
        join=True,
    )


@pytest.mark.parametrize("initialized", [False, True])
def test_setup_selects_local_device_before_cuda_timer(initialized: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh workers select their GPU; existing distributed contexts stay untouched."""
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: initialized)
    select_device = MagicMock()
    monkeypatch.setattr(torch.cuda, "set_device", select_device)
    recipe = object.__new__(BenchmarkingRecipeForNextTokenPrediction)

    def observe_timer(name: str, log_level: int) -> None:
        assert name == "setup" and log_level == 1
        if initialized:
            select_device.assert_not_called()
        else:
            select_device.assert_called_once_with(3)
        raise RuntimeError("observed setup timer")

    recipe.timers = observe_timer
    with pytest.raises(RuntimeError, match="observed setup timer"):
        recipe.setup()
