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

import pytest
import torch
from unittest.mock import Mock, MagicMock, patch
from torch.distributed.pipelining.schedules import (
    PipelineScheduleSingle,
    PipelineScheduleMulti,
)

from nemo_automodel.components.distributed.pipelining.functional import (
    stage_ids_this_rank,
    generate_hf_model_fqn_per_model_part,
    calculate_virtual_stages,
    split_model_into_stages,
    build_pipeline_schedule,
    pipeline_model,
)


class TestStageIdsThisRank:
    """Test stage_ids_this_rank function - no mocks needed as it's pure calculation."""

    def test_loop_style_single_stage_per_rank(self):
        # Test with 1 stage per rank
        assert stage_ids_this_rank(0, 4, 4, "loop") == (0,)
        assert stage_ids_this_rank(1, 4, 4, "loop") == (1,)
        assert stage_ids_this_rank(2, 4, 4, "loop") == (2,)
        assert stage_ids_this_rank(3, 4, 4, "loop") == (3,)

    def test_loop_style_multiple_stages_per_rank(self):
        # Test with 2 stages per rank
        assert stage_ids_this_rank(0, 4, 8, "loop") == (0, 4)
        assert stage_ids_this_rank(1, 4, 8, "loop") == (1, 5)
        assert stage_ids_this_rank(2, 4, 8, "loop") == (2, 6)
        assert stage_ids_this_rank(3, 4, 8, "loop") == (3, 7)

    def test_v_style(self):
        # Test V-style scheduling (assumes 2 stages per rank)
        assert stage_ids_this_rank(0, 4, 8, "v") == (0, 7)
        assert stage_ids_this_rank(1, 4, 8, "v") == (1, 6)
        assert stage_ids_this_rank(2, 4, 8, "v") == (2, 5)
        assert stage_ids_this_rank(3, 4, 8, "v") == (3, 4)

    def test_invalid_stage_distribution(self):
        # Test when stages not evenly divisible by pp_size
        with pytest.raises(AssertionError):
            stage_ids_this_rank(0, 4, 5, "loop")

    def test_v_style_invalid_stages_per_rank(self):
        # Test V-style with != 2 stages per rank
        with pytest.raises(AssertionError):
            stage_ids_this_rank(0, 4, 12, "v")  # 3 stages per rank


class TestGenerateHfModelFqnPerModelPart:
    """Test generate_hf_model_fqn_per_model_part function - no mocks needed."""

    def test_single_stage(self):
        result = generate_hf_model_fqn_per_model_part(
            num_stages=1,
            num_layers=4,
            include_embeddings=True,
            include_lm_head=True,
            include_rotary_emb=True,
        )
        assert len(result) == 1
        assert "model.embed_tokens" in result[0]
        assert "model.layers.0" in result[0]
        assert "model.layers.3" in result[0]
        assert "model.norm" in result[0]
        assert "lm_head" in result[0]
        assert "model.rotary_emb" in result[0]

    def test_multiple_stages_even_distribution(self):
        result = generate_hf_model_fqn_per_model_part(
            num_stages=4,
            num_layers=8,
        )
        assert len(result) == 4
        # First stage has embeddings + 2 layers
        assert "model.embed_tokens" in result[0]
        assert "model.layers.0" in result[0]
        assert "model.layers.1" in result[0]
        # Middle stages have 2 layers each
        assert "model.layers.2" in result[1]
        assert "model.layers.3" in result[1]
        # Last stage has layers + norm + lm_head
        assert "model.layers.6" in result[3]
        assert "model.layers.7" in result[3]
        assert "model.norm" in result[3]
        assert "lm_head" in result[3]

    def test_uneven_distribution(self):
        # 10 layers across 3 stages: 4, 3, 3
        result = generate_hf_model_fqn_per_model_part(
            num_stages=3,
            num_layers=10,
        )
        assert len(result) == 3
        # First stage gets extra layer (4 layers)
        assert len([m for m in result[0] if "layers." in m]) == 4
        # Other stages get 3 layers each
        assert len([m for m in result[1] if "layers." in m]) == 3
        assert len([m for m in result[2] if "layers." in m]) == 3

    def test_without_embeddings_and_lm_head(self):
        result = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            include_embeddings=False,
            include_lm_head=False,
            include_rotary_emb=False,
        )
        # First stage should not have embeddings
        assert "model.embed_tokens" not in result[0]
        # Last stage should not have lm_head
        assert "lm_head" not in result[1]
        # No stage should have rotary_emb
        assert all("model.rotary_emb" not in stage for stage in result)

    def test_custom_fqn_prefix(self):
        result = generate_hf_model_fqn_per_model_part(
            num_stages=2,
            num_layers=4,
            fqn_prefix="custom.",
        )
        assert "custom.embed_tokens" in result[0]
        assert "custom.layers.0" in result[0]
        assert "custom.norm" in result[1]

    def test_invalid_num_stages(self):
        with pytest.raises(ValueError):
            generate_hf_model_fqn_per_model_part(0, 4)

        with pytest.raises(ValueError):
            generate_hf_model_fqn_per_model_part(5, 4)  # More stages than layers


class TestCalculateVirtualStages:
    """Test calculate_virtual_stages function - no mocks needed."""

    def test_with_layers_per_stage_single_schedule(self):
        # Single stage schedule with valid config - needs rounding
        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=32,  # This will give exactly 1 stage per rank
            pp_size=4,
            is_single_stage_schedule=True,
            round_to_pp_multiple="up",
        )
        assert num_virtual == 4  # ceil(32/32) + 1 = 2, rounded up to 4
        assert stages_per_rank == 1

    def test_with_layers_per_stage_multi_schedule(self):
        # Multi stage schedule with valid config
        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=4,
            pp_size=4,
            is_single_stage_schedule=False,
            round_to_pp_multiple="down",
        )
        assert num_virtual == 8  # ceil(32/4) + 1 = 9, rounded down to 8
        assert stages_per_rank == 2

    def test_round_up(self):
        # Test rounding up when not divisible
        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=5,
            pp_size=4,
            is_single_stage_schedule=False,
            round_to_pp_multiple="up",
        )
        assert num_virtual == 8  # ceil(32/5) + 1 = 8
        assert stages_per_rank == 2

    def test_round_down(self):
        # Test rounding down
        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=3,
            pp_size=4,
            is_single_stage_schedule=False,
            round_to_pp_multiple="down",
        )
        assert num_virtual == 12  # ceil(32/3) + 1 = 12
        assert stages_per_rank == 3

    def test_invalid_round_option(self):
        with pytest.raises(ValueError, match="Invalid value for round_to_pp_multiple"):
            calculate_virtual_stages(
                num_layers=32,
                layers_per_stage=7,  # This gives 6 stages which is not divisible by 4
                pp_size=4,
                is_single_stage_schedule=False,
                round_to_pp_multiple="invalid",  # Invalid option should trigger error
            )

    def test_invalid_stages_not_divisible(self):
        with pytest.raises(ValueError, match="must be divisible by"):
            calculate_virtual_stages(
                num_layers=32,
                layers_per_stage=7,  # ceil(32/7) + 1 = 6, not divisible by 4
                pp_size=4,
                is_single_stage_schedule=False,
                round_to_pp_multiple=None,  # Explicitly set to None to ensure error is raised
            )

    def test_single_schedule_multiple_stages_error(self):
        with pytest.raises(ValueError, match="Single stage schedule requires exactly 1 stage"):
            calculate_virtual_stages(
                num_layers=32,
                layers_per_stage=8,  # This gives 5 stages total (ceil(32/8) + 1 = 5)
                pp_size=4,
                is_single_stage_schedule=True,
                round_to_pp_multiple="up",  # Round 5 up to 8, giving 2 stages per rank
            )

    def test_multi_schedule_single_stage_error(self):
        with pytest.raises(ValueError, match="Multi-stage schedule requires at least 2 stages"):
            calculate_virtual_stages(
                num_layers=32,
                layers_per_stage=32,  # This gives 2 stages total (ceil(32/32) + 1 = 2)
                pp_size=2,  # With 2 PP ranks, that's 1 stage per rank
                is_single_stage_schedule=False,
            )

    def test_without_layers_per_stage(self):
        # Default behavior when layers_per_stage is None
        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=None,
            pp_size=4,
            is_single_stage_schedule=True,
        )
        assert num_virtual == 4
        assert stages_per_rank == 1

        num_virtual, stages_per_rank = calculate_virtual_stages(
            num_layers=32,
            layers_per_stage=None,
            pp_size=4,
            is_single_stage_schedule=False,
        )
        assert num_virtual == 8
        assert stages_per_rank == 2


class TestSplitModelIntoStages:
    """Test split_model_into_stages function with mocks."""
    @patch('nemo_automodel.components.distributed.pipelining.functional.calculate_virtual_stages')
    @patch('nemo_automodel.components.distributed.pipelining.functional.generate_hf_model_fqn_per_model_part')
    def test_auto_generate_module_names(self, mock_generate_fqn, mock_calc_stages):
        # Setup mocks
        mock_pp_mesh = Mock()
        mock_pp_mesh.get_local_rank.return_value = 0
        mock_pp_mesh.size.return_value = 2

        mock_model = Mock()
        mock_model.model = Mock()
        mock_model.model.layers = [Mock() for _ in range(4)]
        mock_model.model.rotary_emb = Mock()
        mock_model.lm_head = Mock()

        # Mock virtual stages calculation
        mock_calc_stages.return_value = (2, 1)

        # Mock FQN generation
        mock_generate_fqn.return_value = [
            ["model.embed_tokens", "model.layers.0"],
            ["model.layers.1", "model.norm"],
        ]

        with patch('nemo_automodel.components.distributed.pipelining.functional.PipelineStage'), \
             patch('nemo_automodel.components.distributed.pipelining.functional.get_schedule_class') as mock_get_schedule_class, \
             patch('nemo_automodel.components.distributed.pipelining.functional.stage_ids_this_rank') as mock_stage_ids, \
             patch('copy.deepcopy') as mock_deepcopy:

            # Make sure get_schedule_class returns an actual class
            mock_get_schedule_class.return_value = PipelineScheduleSingle

            # Mock stage_ids_this_rank
            mock_stage_ids.return_value = (0,)

            # Mock deepcopy to return a mock with proper structure
            mock_copy = Mock()
            mock_copy.named_children.return_value = []
            mock_deepcopy.return_value = mock_copy

            stages, models = split_model_into_stages(
                mock_model,
                mock_pp_mesh,
                "pp",
                "PipelineScheduleSingle",
                torch.device("cuda:0"),
                layers_per_stage=2,
            )

            # Verify FQN generation was called
            mock_generate_fqn.assert_called_once()


class TestBuildPipelineSchedule:
    """Test build_pipeline_schedule function."""

    @patch('nemo_automodel.components.distributed.pipelining.functional.get_schedule_class')
    def test_build_schedule_single(self, mock_get_schedule):
        # Create a mock schedule class that properly inherits from PipelineScheduleSingle
        class MockScheduleSingle(PipelineScheduleSingle):
            def __init__(self, *args, **kwargs):
                self.stage = args[0] if args else None
                self.n_microbatches = kwargs.get('n_microbatches', 0)
                self.loss_fn = kwargs.get('loss_fn', None)

            def _step_microbatches(self, *args, **kwargs):
                # Mock implementation of abstract method
                pass

        mock_get_schedule.return_value = MockScheduleSingle

        # Mock stages
        mock_stage = Mock()
        stages = [mock_stage]

        # Mock loss function
        loss_fn = Mock()

                # Call function
        schedule = build_pipeline_schedule(
            pipeline_parallel_schedule_csv=None,
            pipeline_parallel_schedule="PipelineScheduleSingle",
            microbatch_size=2,
            local_batch_size=8,
            stages=stages,
            loss_fn=loss_fn,
        )

        # Verify schedule was created correctly
        assert isinstance(schedule, MockScheduleSingle)
        assert schedule.stage == mock_stage
        assert schedule.n_microbatches == 4
        assert schedule.loss_fn == loss_fn

    @patch('nemo_automodel.components.distributed.pipelining.functional.get_schedule_class')
    def test_build_schedule_multi(self, mock_get_schedule):
        # Create a mock schedule class that properly inherits from PipelineScheduleMulti
        class MockScheduleMulti(PipelineScheduleMulti):
            def __init__(self, *args, **kwargs):
                self.stages = args[0] if args else None
                self.n_microbatches = kwargs.get('n_microbatches', 0)
                self.loss_fn = kwargs.get('loss_fn', None)

        mock_get_schedule.return_value = MockScheduleMulti

        # Mock stages
        stages = [Mock(), Mock()]

        # Mock loss function
        loss_fn = Mock()

                # Call function
        schedule = build_pipeline_schedule(
            pipeline_parallel_schedule_csv=None,
            pipeline_parallel_schedule="PipelineScheduleMulti",
            microbatch_size=2,
            local_batch_size=8,
            stages=stages,
            loss_fn=loss_fn,
        )

        # Verify schedule was created correctly
        assert isinstance(schedule, MockScheduleMulti)
        assert schedule.stages == stages
        assert schedule.n_microbatches == 4
        assert schedule.loss_fn == loss_fn

    def test_invalid_batch_size(self):
        # Test when batch size not divisible by microbatch size
        with pytest.raises(ValueError, match="must be divisible by"):
            build_pipeline_schedule(
                pipeline_parallel_schedule_csv=None,
                pipeline_parallel_schedule="PipelineScheduleSingle",
                microbatch_size=3,
                local_batch_size=8,
                stages=[Mock()],
                loss_fn=Mock(),
            )

    @patch('os.path.isfile')
    def test_csv_schedule(self, mock_isfile):
        # Mock file exists
        mock_isfile.return_value = True

        # Create a mock _PipelineScheduleRuntime class that can be used with issubclass
        class MockPipelineScheduleRuntime:
            def __init__(self, *args, **kwargs):
                self.stage = args[0] if args else None
                self.n_microbatches = kwargs.get('n_microbatches', 0)
                self.loss_fn = kwargs.get('loss_fn', None)
                self._load_csv = Mock()
                self._mock_instance = self  # Store reference for assertions

        # Patch _PipelineScheduleRuntime with our mock class
        with patch('nemo_automodel.components.distributed.pipelining.functional._PipelineScheduleRuntime', MockPipelineScheduleRuntime):
            # Call with CSV
            schedule = build_pipeline_schedule(
                pipeline_parallel_schedule_csv="/path/to/schedule.csv",
                pipeline_parallel_schedule=None,
                microbatch_size=2,
                local_batch_size=8,
                stages=[Mock()],
                loss_fn=Mock(),
            )

            # Verify CSV was loaded
            schedule._load_csv.assert_called_once_with("/path/to/schedule.csv")
            assert isinstance(schedule, MockPipelineScheduleRuntime)

    def test_csv_file_not_found(self):
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(FileNotFoundError):
                build_pipeline_schedule(
                    pipeline_parallel_schedule_csv="/nonexistent/file.csv",
                    pipeline_parallel_schedule=None,
                    microbatch_size=2,
                    local_batch_size=8,
                    stages=[Mock()],
                    loss_fn=Mock(),
                )


class TestPipelineModel:
    """Test pipeline_model function."""

    @patch('nemo_automodel.components.distributed.pipelining.functional.split_model_into_stages')
    @patch('nemo_automodel.components.distributed.pipelining.functional.build_pipeline_schedule')
    def test_basic_pipeline_model(self, mock_build_schedule, mock_split_stages):
        # Setup mocks
        mock_world_mesh = MagicMock()
        mock_pp_mesh = Mock()
        mock_pp_mesh.size.return_value = 2
        mock_world_mesh.__getitem__.return_value = mock_pp_mesh

        mock_moe_mesh = Mock()

        # Mock model
        mock_model = Mock()

        # Mock split_model_into_stages return
        mock_stage1 = Mock()
        mock_stage1.is_first = True
        mock_stage1.is_last = False
        mock_stage1.submod = Mock()

        mock_stage2 = Mock()
        mock_stage2.is_first = False
        mock_stage2.is_last = True
        mock_stage2.submod = Mock()

        mock_model1 = Mock()
        mock_model2 = Mock()

        mock_split_stages.return_value = ([mock_stage1, mock_stage2], [mock_model1, mock_model2])

        # Mock schedule
        mock_schedule = Mock()
        mock_build_schedule.return_value = mock_schedule

        # Call function
        schedule, models, has_first, has_last, stages = pipeline_model(
            model=mock_model,
            world_mesh=mock_world_mesh,
            moe_mesh=mock_moe_mesh,
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            layers_per_stage=4,
            pipeline_parallel_schedule_csv=None,
            pipeline_parallel_schedule="PipelineScheduleSingle",
            microbatch_size=2,
            local_batch_size=8,
            device=torch.device("cuda:0"),
            loss_fn=Mock(),
        )

        assert schedule == mock_schedule
        assert models == [mock_model1, mock_model2]
        assert has_first is True
        assert has_last is True
        assert stages == [mock_stage1, mock_stage2]

    def test_pipeline_size_validation(self):
        # Test assertion when pp_size <= 1
        mock_world_mesh = MagicMock()
        mock_pp_mesh = Mock()
        mock_pp_mesh.size.return_value = 1
        mock_world_mesh.__getitem__.return_value = mock_pp_mesh

        with pytest.raises(AssertionError):
            pipeline_model(
                model=Mock(),
                world_mesh=mock_world_mesh,
                moe_mesh=Mock(),
                pp_axis_name="pp",
                dp_axis_names=("dp",),
                layers_per_stage=4,
                pipeline_parallel_schedule_csv=None,
                pipeline_parallel_schedule="PipelineScheduleSingle",
                microbatch_size=2,
                local_batch_size=8,
                device=torch.device("cuda:0"),
            )

    @patch('nemo_automodel.components.distributed.pipelining.functional.split_model_into_stages')
    @patch('nemo_automodel.components.distributed.pipelining.functional.build_pipeline_schedule')
    def test_with_parallelization_fn(self, mock_build_schedule, mock_split_stages):
        # Setup mocks
        mock_world_mesh = MagicMock()
        mock_pp_mesh = Mock()
        mock_pp_mesh.size.return_value = 2
        mock_world_mesh.__getitem__.return_value = mock_pp_mesh

        # Mock parallelization function
        mock_parallelize_fn = Mock()

        # Mock stages and models
        mock_stage = Mock()
        mock_stage.is_first = True
        mock_stage.is_last = False
        mock_stage.submod = Mock()

        mock_model = Mock()
        mock_split_stages.return_value = ([mock_stage], [mock_model])

        # Call with parallelization
        pipeline_model(
            model=Mock(),
            world_mesh=mock_world_mesh,
            moe_mesh=Mock(),
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            layers_per_stage=4,
            pipeline_parallel_schedule_csv=None,
            pipeline_parallel_schedule="PipelineScheduleSingle",
            microbatch_size=2,
            local_batch_size=8,
            device=torch.device("cuda:0"),
            parallelize_fn=mock_parallelize_fn,
        )

        # Verify parallelize_fn was called
        mock_parallelize_fn.assert_called_once()
        call_kwargs = mock_parallelize_fn.call_args[1]
        assert call_kwargs['pp_enabled'] is True
        assert call_kwargs['dp_axis_names'] == ("dp",)
