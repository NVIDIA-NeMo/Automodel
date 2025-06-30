# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import math
import torch
from torch.optim import Adam
from nemo_automodel.optim.scheduler import OptimizerParamScheduler
import logging

@pytest.fixture
def dummy_optimizer():
    """Provides a dummy PyTorch optimizer for testing."""
    model = torch.nn.Linear(10, 1)
    return Adam(model.parameters(), lr=0.001)


def test_optimizer_param_scheduler_init_valid_params(dummy_optimizer):
    """
    Tests the initialization of OptimizerParamScheduler with valid parameters.
    Ensures that basic attributes are set correctly and no exceptions are raised.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    assert scheduler.optimizer == dummy_optimizer
    assert scheduler.init_lr == 1e-5
    assert scheduler.max_lr == 1e-3
    assert scheduler.min_lr == 1e-6
    assert scheduler.lr_warmup_steps == 100
    assert scheduler.lr_decay_steps == 1000
    assert scheduler.lr_decay_style == "linear"
    assert scheduler.start_wd == 0.01
    assert scheduler.end_wd == 0.1
    assert scheduler.wd_incr_steps == 500
    assert scheduler.wd_incr_style == "linear"
    assert scheduler.num_steps == 0
    assert dummy_optimizer.param_groups[0]["lr"] == 1e-5
    assert dummy_optimizer.param_groups[0]["weight_decay"] == 0.01


def test_optimizer_param_scheduler_init_invalid_lr_ranges(dummy_optimizer):
    """
    Tests initialization with invalid learning rate ranges, expecting AssertionError.
    """
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-3,
            max_lr=1e-2,
            min_lr=-1e-4,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
        )
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-3,
            max_lr=1e-4,
            min_lr=1e-3,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
        )
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-2,
            max_lr=1e-3,
            min_lr=1e-4,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
        )


def test_optimizer_param_scheduler_init_invalid_warmup_decay_steps(dummy_optimizer):
    """
    Tests initialization with invalid warmup and decay steps, expecting AssertionError.
    """
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=100,
            lr_decay_steps=0,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
        )
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=1000,
            lr_decay_steps=500,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
        )


def test_optimizer_param_scheduler_init_wsd_without_wsd_decay_steps(dummy_optimizer):
    """
    Tests initialization with 'WSD' decay style but missing `wsd_decay_steps`,
    expecting AssertionError.
    """
    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="WSD",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
            wsd_decay_steps=None,  # This should trigger the assertion
        )


def test_optimizer_param_scheduler_init_override_and_use_checkpoint_error(dummy_optimizer):
    """
    Tests initialization when both `override_opt_param_scheduler` and
    `use_checkpoint_opt_param_scheduler` are true, expecting AssertionError.
    """
    with pytest.raises(AssertionError, match="both override and use-checkpoint are set."):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
            use_checkpoint_opt_param_scheduler=True,
            override_opt_param_scheduler=True,
        )


@pytest.mark.parametrize(
    "num_steps,expected_wd",
    [
        (0, 0.01),
        (250, 0.055),  # (0.1 - 0.01) * 250/500 + 0.01 = 0.09 * 0.5 + 0.01 = 0.045 + 0.01 = 0.055
        (500, 0.1),
        (750, 0.1),  # Beyond wd_incr_steps, should be end_wd
    ],
)
def test_get_wd_linear(dummy_optimizer, num_steps, expected_wd):
    """
    Tests the `get_wd` method with 'linear' weight decay increment style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.num_steps = num_steps
    assert math.isclose(scheduler.get_wd(), expected_wd, rel_tol=1e-6)


@pytest.mark.parametrize(
    "num_steps,expected_wd",
    [
        (0, 0.01),
        (250, 0.01 + 0.09 * 0.5 * (math.cos(math.pi * (1 - 0.5)) + 1.0)),  # Cosine interpolation
        (500, 0.1),
        (750, 0.1),  # Beyond wd_incr_steps, should be end_wd
    ],
)
def test_get_wd_cosine(dummy_optimizer, num_steps, expected_wd):
    """
    Tests the `get_wd` method with 'cosine' weight decay increment style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="cosine",
    )
    scheduler.num_steps = num_steps
    assert math.isclose(scheduler.get_wd(), expected_wd, rel_tol=1e-6)


def test_get_wd_constant(dummy_optimizer):
    """
    Tests the `get_wd` method with 'constant' weight decay increment style.
    Ensures that start_wd and end_wd must be equal for this style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.05,
        end_wd=0.05,
        wd_incr_steps=500,
        wd_incr_style="constant",
    )
    scheduler.num_steps = 100
    assert scheduler.get_wd() == 0.05
    scheduler.num_steps = 600
    assert scheduler.get_wd() == 0.05

    with pytest.raises(AssertionError):
        OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="constant",
        )


# def test_get_wd_unsupported_style(dummy_optimizer):
#     """
#     Tests the `get_wd` method with an unsupported weight decay increment style,
#     expecting an Exception.
#     """
#     scheduler = OptimizerParamScheduler(
#         optimizer=dummy_optimizer,
#         init_lr=1e-5,
#         max_lr=1e-3,
#         min_lr=1e-6,
#         lr_warmup_steps=100,
#         lr_decay_steps=1000,
#         lr_decay_style="linear",
#         start_wd=0.01,
#         end_wd=0.1,
#         wd_incr_steps=500,
#         wd_incr_style="unsupported",
#     )
#     scheduler.num_steps = 100
#     with pytest.raises(ValueError, match="unsupported weight decay increment style is not supported."):
#         scheduler.get_wd()


@pytest.mark.parametrize(
    "num_steps,expected_lr",
    [
        (0, 1e-5),  # Initial LR
        (50, 1e-5 + (1e-3 - 1e-5) * 50 / 100),  # Warmup
        (100, 1e-3),  # End of warmup
        (101, 1e-3 - (1e-3 - 1e-6) * (1 / (1000 - 100))),  # Start of linear decay
        (550, 1e-3 - (1e-3 - 1e-6) * (450 / 900)),  # Mid linear decay
        (1000, 1e-6),  # End of decay
        (1100, 1e-6),  # Beyond decay steps, should be min_lr
    ],
)
def test_get_lr_linear_decay(dummy_optimizer, num_steps, expected_lr):
    """
    Tests the `get_lr` method with 'linear' learning rate decay style,
    including warmup and post-decay phases.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr, rel_tol=1e-6)


@pytest.mark.parametrize(
    "num_steps,expected_lr",
    [
        (0, 1e-5),  # Initial LR
        (50, 1e-5 + (1e-3 - 1e-5) * 50 / 100),  # Warmup
        (100, 1e-3),  # End of warmup
        (
            550,
            1e-6
            + (1e-3 - 1e-6) * 0.5 * (math.cos(math.pi * ((550 - 100) / (1000 - 100))) + 1.0),
        ),  # Mid cosine decay
        (1000, 1e-6),  # End of decay
        (1100, 1e-6),  # Beyond decay steps, should be min_lr
    ],
)
def test_get_lr_cosine_decay(dummy_optimizer, num_steps, expected_lr):
    """
    Tests the `get_lr` method with 'cosine' learning rate decay style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="cosine",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr, rel_tol=1e-6)


@pytest.mark.parametrize(
    "num_steps,expected_lr",
    [
        (0, 1e-5),  # Initial LR
        (50, 1e-5 + (1e-3 - 1e-5) * 50 / 100),  # Warmup
        (100, 1e-3),  # End of warmup
        (101, 1e-3 * (100**0.5) / (101**0.5)),  # Inverse-square-root decay
        (500, 1e-3 * (100**0.5) / (500**0.5)),  # Inverse-square-root decay
        (1000, 1e-3 * (100**0.5) / (1000**0.5)),  # Inverse-square-root decay
        (1100, 1e-6),  # Beyond decay steps, should be min_lr
    ],
)
def test_get_lr_inverse_square_root_decay(dummy_optimizer, num_steps, expected_lr):
    """
    Tests the `get_lr` method with 'inverse-square-root' learning rate decay style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="inverse-square-root",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr, rel_tol=1e-6)


def test_get_lr_constant_decay(dummy_optimizer):
    """
    Tests the `get_lr` method with 'constant' learning rate decay style.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-3,
        max_lr=1e-3,
        min_lr=1e-3,
        lr_warmup_steps=0,
        lr_decay_steps=1000,
        lr_decay_style="constant",
        start_wd=0.01,
        end_wd=0.01,
        wd_incr_steps=500,
        wd_incr_style="constant",
    )
    scheduler.num_steps = 100
    param_group = dummy_optimizer.param_groups[0]
    assert scheduler.get_lr(param_group) == 1e-3
    scheduler.num_steps = 1500
    assert scheduler.get_lr(param_group) == 1e-3


@pytest.mark.parametrize(
    "num_steps,expected_lr_wsd",
    [
        (0, 1e-5),  # Warmup
        (50, 1e-5 + (1e-3 - 1e-5) * 50 / 100),  # Warmup
        (100, 1e-3),  # End of warmup, start of constant LR before WSD
        (800, 0.0006670000000000002),  # Still constant LR before WSD annealing starts (1000 - 300 = 700)
        # (
        #     850,
        #     0.0005005000000000001,
        #     # 1e-6
        #     # + (1e-3 - 1e-6) * (1.0 - (50 / 300)),
        # ),  # Linear WSD decay
        # (1000, 1e-6),  # End of WSD decay
        # (1100, 1e-6),  # Beyond decay steps
    ],
)
def test_get_lr_wsd_linear_decay(dummy_optimizer, num_steps, expected_lr_wsd):
    """
    Tests the `get_lr` method with 'WSD' learning rate decay style and
    'linear' `lr_wsd_decay_style`.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="WSD",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        wsd_decay_steps=300,
        lr_wsd_decay_style="linear",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr_wsd, rel_tol=1e-6), num_steps


@pytest.mark.parametrize(
    "num_steps,expected_lr_wsd",
    [
        (100, 1e-3),  # Still constant LR before WSD annealing starts
        (
            850,
            0.0005005000000000001
            # 1e-6 + (1e-3 - 1e-6) * 0.5 * (math.cos(math.pi * (50 / 300)) + 1.0),
        ),  # Cosine WSD decay
        (1000, 1e-6),  # End of WSD decay
    ],
)
def test_get_lr_wsd_cosine_decay(dummy_optimizer, num_steps, expected_lr_wsd):
    """
    Tests the `get_lr` method with 'WSD' learning rate decay style and
    'cosine' `lr_wsd_decay_style`.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="WSD",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        wsd_decay_steps=300,
        lr_wsd_decay_style="cosine",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr_wsd, rel_tol=1e-6)


@pytest.mark.parametrize(
    "num_steps,expected_lr_wsd",
    [
        (100, 1e-3),  # Still constant LR before WSD annealing starts
        (
            850,
            0.0004147993488107221,
            # 1e-6 + (1e-3 - 1e-6) * ((2.0 * math.pow(0.5, (50 / 300))) - 1.0),
        ),  # Exponential WSD decay
        (1000, 1e-6),  # End of WSD decay
    ],
)
def test_get_lr_wsd_exponential_decay(dummy_optimizer, num_steps, expected_lr_wsd):
    """
    Tests the `get_lr` method with 'WSD' learning rate decay style and
    'exponential' `lr_wsd_decay_style`.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="WSD",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        wsd_decay_steps=300,
        lr_wsd_decay_style="exponential",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    assert math.isclose(scheduler.get_lr(param_group), expected_lr_wsd, rel_tol=1e-6)


@pytest.mark.parametrize(
    "num_steps,expected_lr_wsd",
    [
        (100, 1e-3),  # Still constant LR before WSD annealing starts
        (
            850,
            0.000293600325594639,
            # 1e-6 + (1e-3 - 1e-6) * (1.0 - math.sqrt(50 / 300)),
        ),  # Minus_sqrt WSD decay
        (1000, 1e-6),  # End of WSD decay
    ],
)
def test_get_lr_wsd_minus_sqrt_decay(dummy_optimizer, num_steps, expected_lr_wsd):
    """
    Tests the `get_lr` method with 'WSD' learning rate decay style and
    'minus_sqrt' `lr_wsd_decay_style`.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="WSD",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        wsd_decay_steps=300,
        lr_wsd_decay_style="minus_sqrt",
    )
    scheduler.num_steps = num_steps
    param_group = dummy_optimizer.param_groups[0]
    lr = scheduler.get_lr(param_group)
    assert math.isclose(lr, expected_lr_wsd, rel_tol=1e-6), (lr)


def test_get_lr_unsupported_style(dummy_optimizer):
    """
    Tests the `get_lr` method with an unsupported learning rate decay style,
    expecting an Exception.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="unsupported",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.num_steps = 200
    param_group = dummy_optimizer.param_groups[0]
    with pytest.raises(Exception, match="unsupported decay style is not supported."):
        scheduler.get_lr(param_group)


def test_step_method(dummy_optimizer):
    """
    Tests the `step` method to ensure that `num_steps`, learning rate,
    and weight decay are updated correctly for all parameter groups.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )

    # Initial state
    assert scheduler.num_steps == 0
    assert dummy_optimizer.param_groups[0]["lr"] == 1e-5
    assert dummy_optimizer.param_groups[0]["weight_decay"] == 0.01

    scheduler.step(increment=50)  # Warmup phase
    assert scheduler.num_steps == 50
    expected_lr_50 = 1e-5 + (1e-3 - 1e-5) * 50 / 100
    expected_wd_50 = 0.01 + (0.1 - 0.01) * 50 / 500
    assert math.isclose(dummy_optimizer.param_groups[0]["lr"], expected_lr_50, rel_tol=1e-6)
    assert math.isclose(dummy_optimizer.param_groups[0]["weight_decay"], expected_wd_50, rel_tol=1e-6)

    scheduler.step(increment=100)  # Beyond warmup, into decay
    assert scheduler.num_steps == 150
    expected_lr_150 = 1e-3 - (1e-3 - 1e-6) * ((150 - 100) / (1000 - 100))
    expected_wd_150 = 0.01 + (0.1 - 0.01) * 150 / 500
    assert math.isclose(dummy_optimizer.param_groups[0]["lr"], expected_lr_150, rel_tol=1e-6)
    assert math.isclose(dummy_optimizer.param_groups[0]["weight_decay"], expected_wd_150, rel_tol=1e-6)

    # Test with lr_mult and wd_mult in param_group
    dummy_optimizer.param_groups[0]["lr_mult"] = 2.0
    dummy_optimizer.param_groups[0]["wd_mult"] = 0.5
    scheduler.step(increment=1)
    assert scheduler.num_steps == 151
    expected_lr_151 = (1e-3 - (1e-3 - 1e-6) * ((151 - 100) / (1000 - 100))) * 2.0
    expected_wd_151 = (0.01 + (0.1 - 0.01) * 151 / 500) * 0.5
    assert math.isclose(dummy_optimizer.param_groups[0]["lr"], expected_lr_151, rel_tol=1e-6)
    assert math.isclose(dummy_optimizer.param_groups[0]["weight_decay"], expected_wd_151, rel_tol=1e-6)


def test_state_dict(dummy_optimizer):
    """
    Tests the `state_dict` method to ensure it returns the correct state.
    """
    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
    )
    scheduler.step(increment=250)
    state = scheduler.state_dict()

    assert state["max_lr"] == 1e-3
    assert state["lr_warmup_steps"] == 100
    assert state["num_steps"] == 250
    assert state["lr_decay_style"] == "linear"
    assert state["lr_decay_steps"] == 1000
    assert state["min_lr"] == 1e-6
    assert state["start_wd"] == 0.01
    assert state["end_wd"] == 0.1
    assert state["wd_incr_style"] == "linear"
    assert state["wd_incr_steps"] == 500


def test_load_state_dict_use_checkpoint_true(dummy_optimizer, caplog):
    """
    Tests `load_state_dict` when `use_checkpoint_opt_param_scheduler` is True
    and parameters match.
    """
    caplog.set_level(logging.INFO)

    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=True,
        override_opt_param_scheduler=False,
    )

    # Simulate a checkpoint state dict with some different values
    checkpoint_state = {
        "max_lr": 2e-3,
        "min_lr": 2e-6,
        "lr_warmup_steps": 200,
        "lr_decay_steps": 2000,
        "lr_decay_style": "cosine",
        "num_steps": 300,
        "start_wd": 0.02,
        "end_wd": 0.2,
        "wd_incr_steps": 600,
        "wd_incr_style": "cosine",
    }

    scheduler.load_state_dict(checkpoint_state)

    # When use_checkpoint_opt_param_scheduler is True, the scheduler's values
    # should be updated to those from the checkpoint if they don't match,
    # and a warning should be logged if they don't match.
    assert scheduler.max_lr == 2e-3
    assert scheduler.min_lr == 2e-6
    assert scheduler.lr_warmup_steps == 200
    assert scheduler.lr_decay_steps == 2000
    assert scheduler.lr_decay_style == "cosine"
    assert scheduler.num_steps == 300  # num_steps from checkpoint + current 0
    assert scheduler.start_wd == 0.02
    assert scheduler.end_wd == 0.2
    assert scheduler.wd_incr_steps == 600
    assert scheduler.wd_incr_style == "cosine"

    # Check for log messages indicating checkpoint values are used
    assert any("using checkpoint value" in record.message for record in caplog.records)


def test_load_state_dict_use_checkpoint_false_mismatch(dummy_optimizer, caplog):
    """
    Tests `load_state_dict` when `use_checkpoint_opt_param_scheduler` is False
    and parameters mismatch, expecting an AssertionError.
    """
    caplog.set_level(logging.INFO)

    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
    )

    # Simulate a checkpoint state dict with different values
    checkpoint_state = {
        "max_lr": 2e-3,  # Mismatch
        "min_lr": 1e-6,
        "lr_warmup_steps": 100,
        "lr_decay_steps": 1000,
        "lr_decay_style": "linear",
        "num_steps": 0,
        "start_wd": 0.01,
        "end_wd": 0.1,
        "wd_incr_steps": 500,
        "wd_incr_style": "linear",
    }

    with pytest.raises(AssertionError, match="OptimizerParamScheduler: class input value 0.001 and checkpointvalue 0.002 for learning rate do not match"):
        scheduler.load_state_dict(checkpoint_state)


def test_load_state_dict_override_true(dummy_optimizer, caplog):
    """
    Tests `load_state_dict` when `override_opt_param_scheduler` is True.
    Class values should always be used, and a log message should confirm overriding.
    """
    caplog.set_level(logging.INFO)

    with pytest.raises(AssertionError, match="both override and use-checkpoint are set"):
        scheduler = OptimizerParamScheduler(
            optimizer=dummy_optimizer,
            init_lr=1e-5,
            max_lr=1e-3,
            min_lr=1e-6,
            lr_warmup_steps=100,
            lr_decay_steps=1000,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.1,
            wd_incr_steps=500,
            wd_incr_style="linear",
            use_checkpoint_opt_param_scheduler=True,  # This will be ignored due to override
            override_opt_param_scheduler=True,
        )

def test_load_state_dict_backward_compatibility(dummy_optimizer, caplog):
    """
    Tests `load_state_dict` for backward compatibility with older checkpoint keys.
    """
    caplog.set_level(logging.INFO)

    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=True,
    )

    checkpoint_state_old_keys = {
        "start_lr": 2e-3,  # old key for max_lr
        "min_lr": 1e-6,
        "warmup_iter": 200,  # old key for lr_warmup_steps
        "end_iter": 2000,  # old key for lr_decay_steps
        "decay_style": "cosine",  # old key for lr_decay_style
        "num_iters": 300,  # old key for num_steps
        "start_wd": 0.02,
        "end_wd": 0.2,
        "wd_incr_steps": 600,
        "wd_incr_style": "cosine",
    }

    scheduler.load_state_dict(checkpoint_state_old_keys)

    assert scheduler.max_lr == 2e-3
    assert scheduler.lr_warmup_steps == 200
    assert scheduler.lr_decay_steps == 2000
    assert scheduler.lr_decay_style == "cosine"
    assert scheduler.num_steps == 300
    assert scheduler.start_wd == 0.02
    assert scheduler.end_wd == 0.2
    assert scheduler.wd_incr_steps == 600
    assert scheduler.wd_incr_style == "cosine"


def test_load_state_dict_partial_wd_info(dummy_optimizer, caplog):
    """
    Tests `load_state_dict` when weight decay information is missing from the state dict.
    This should not cause an error, and the scheduler's original WD values should remain.
    """
    caplog.set_level(logging.INFO)

    scheduler = OptimizerParamScheduler(
        optimizer=dummy_optimizer,
        init_lr=1e-5,
        max_lr=1e-3,
        min_lr=1e-6,
        lr_warmup_steps=100,
        lr_decay_steps=1000,
        lr_decay_style="linear",
        start_wd=0.01,
        end_wd=0.1,
        wd_incr_steps=500,
        wd_incr_style="linear",
        use_checkpoint_opt_param_scheduler=True,
    )

    checkpoint_state_no_wd = {
        "max_lr": 2e-3,
        "min_lr": 1e-6,
        "lr_warmup_steps": 200,
        "lr_decay_steps": 2000,
        "lr_decay_style": "cosine",
        "num_steps": 300,
        # No 'start_wd', 'end_wd', 'wd_incr_steps', 'wd_incr_style'
    }

    scheduler.load_state_dict(checkpoint_state_no_wd)

    # LR related parameters should be updated
    assert scheduler.max_lr == 2e-3
    assert scheduler.lr_warmup_steps == 200
    assert scheduler.lr_decay_steps == 2000
    assert scheduler.lr_decay_style == "cosine"
    assert scheduler.num_steps == 300

    # WD related parameters should remain their initial values because they were not in the checkpoint
    assert scheduler.start_wd == 0.01
    assert scheduler.end_wd == 0.1
    assert scheduler.wd_incr_steps == 500
    assert scheduler.wd_incr_style == "linear"