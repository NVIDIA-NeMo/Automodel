# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace

from tests.functional_tests.checkpoint_robustness.resume_trajectory import (
    _TrainingReproducibilityRecorder,
    _compare_training_reproducibility,
    _configure_resumed_run,
    _configure_uninterrupted_run,
    _restored_state_mismatch,
    _resume_plan_from_config,
    _trajectory_mismatch,
)


def _config(max_steps: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        step_scheduler=SimpleNamespace(max_steps=max_steps, ckpt_every_steps=max_steps),
        lr_scheduler=SimpleNamespace(lr_decay_steps=None),
        checkpoint=SimpleNamespace(
            checkpoint_dir="/tmp/checkpoint-robustness",
            restore_from=None,
            save_consolidated=True,
        ),
    )


def _trajectory(*, first_loss: float, second_loss: float, first_batch: str = "batch-5") -> dict:
    return {
        "boundary_step": 5,
        "continuation_steps": 2,
        "boundary_state": {},
        "steps": {
            "5": {"batch_digest": first_batch, "loss": first_loss, "lr": 1e-4},
            "6": {"batch_digest": "batch-6", "loss": second_loss, "lr": 5e-5},
        },
    }


def _training_run(
    *,
    fingerprint: dict[str, str] | None = None,
    first_batch: str = "batch-0",
    first_loss: float = 1.0,
    second_loss: float = 0.9,
) -> dict:
    return {
        "fingerprint_components": fingerprint or {"model": "same", "dataset": "same"},
        "steps": {
            "0": {"batch_digest": first_batch, "loss": first_loss, "lr": 1e-4},
            "1": {"batch_digest": "batch-1", "loss": second_loss, "lr": 5e-5},
        },
    }


def test_shared_resume_plan_extends_phase_one_from_the_checkpoint_boundary(tmp_path):
    cfg = _config()
    cfg.checkpoint.checkpoint_dir = str(tmp_path)
    plan = _resume_plan_from_config(cfg, continuation_steps=3)

    _configure_uninterrupted_run(cfg, plan)

    assert plan.boundary_step == 5
    assert plan.comparison_steps == (5, 6, 7)
    assert cfg.step_scheduler.max_steps == 8
    assert cfg.step_scheduler.ckpt_every_steps == 5
    assert cfg.lr_scheduler.lr_decay_steps == 5
    assert cfg.checkpoint.save_consolidated == "final"

    checkpoint_path = tmp_path / "epoch_0_step_4"
    _configure_resumed_run(cfg, plan, checkpoint_path)
    assert cfg.checkpoint.restore_from == str(checkpoint_path)
    assert cfg.checkpoint.checkpoint_dir == str(plan.resume_checkpoint_dir)
    assert cfg.checkpoint.save_consolidated is False


def test_resume_state_check_detects_omitted_dataloader_state():
    reference = {
        "step_scheduler": {"step": 5, "epoch": 0},
        "optimizer_steps": [{"5.0": 2}],
        "optimizer_groups": [[{"lr": 1e-4, "weight_decay": 0.01}]],
        "lr_scheduler_digest": "lr",
        "rng_digest": "rng",
        "dataloader_digest": "batch-position-5",
    }
    restored = {key: value for key, value in reference.items() if key != "dataloader_digest"}

    mismatch = _restored_state_mismatch(reference, restored)

    assert mismatch == "restored snapshot omitted required stateful dataloader position (dataloader_digest)"


def test_shared_trajectory_detects_shifted_dataloader_position():
    reference = _trajectory(first_loss=1.0, second_loss=0.9)
    resumed = _trajectory(first_loss=1.0, second_loss=0.9, first_batch="batch-6")

    mismatch = _trajectory_mismatch(
        reference,
        resumed,
        first_loss_threshold=1e-6,
        later_loss_threshold=5e-3,
    )

    assert mismatch == "resumed batch identity differs at step 5; stateful dataloader position was not restored"


def test_shared_trajectory_uses_stricter_first_loss_threshold():
    reference = _trajectory(first_loss=1.0, second_loss=0.9)
    resumed = _trajectory(first_loss=1.0 + 5e-7, second_loss=0.904)

    assert (
        _trajectory_mismatch(
            reference,
            resumed,
            first_loss_threshold=1e-6,
            later_loss_threshold=5e-3,
        )
        is None
    )

    resumed["steps"]["5"]["loss"] = 1.0 + 2e-6
    mismatch = _trajectory_mismatch(
        reference,
        resumed,
        first_loss_threshold=1e-6,
        later_loss_threshold=5e-3,
    )
    assert "first-step_threshold=1.000000e-06" in mismatch


def test_training_reproducibility_requires_matching_configuration_fingerprint():
    normal = _training_run()
    checkpoint = _training_run(fingerprint={"model": "same", "dataset": "different"})

    report = _compare_training_reproducibility(normal, checkpoint, loss_threshold=5e-3)

    assert report == {
        "status": "not_comparable",
        "reason": "configuration fingerprint mismatch",
        "mismatched_components": ["dataset"],
    }


def test_training_reproducibility_reports_nonblocking_loss_envelope():
    normal = _training_run()
    checkpoint = _training_run(first_loss=1.001, second_loss=0.91)

    report = _compare_training_reproducibility(normal, checkpoint, loss_threshold=5e-3)

    assert report["status"] == "outside_tolerance"
    assert report["overlapping_steps"] == [0, 1]
    assert report["max_difference_step"] == 1
    assert abs(report["max_loss_difference"] - 1e-2) < 1e-12


def test_training_reproducibility_detects_independent_batch_order_divergence():
    normal = _training_run()
    checkpoint = _training_run(first_batch="different-batch")

    report = _compare_training_reproducibility(normal, checkpoint, loss_threshold=5e-3)

    assert report["status"] == "diverged"
    assert report["reason"] == "batch identity mismatch"
    assert report["step"] == 0


def test_training_reproducibility_recorder_captures_runtime_batch_and_metrics():
    trainer = SimpleNamespace(
        cfg=SimpleNamespace(
            to_dict=lambda: {
                "model": {"name": "tiny"},
                "rng": {"seed": 123},
                "dataset": {"name": "samples"},
                "dataloader": {"shuffle": True},
                "distributed": {"tp_size": 1},
                "optimizer": {"lr": 1e-4},
                "lr_scheduler": {"lr_decay_steps": 2},
                "loss_fn": {"name": "loss"},
            }
        ),
        step_scheduler=SimpleNamespace(
            step=0,
            global_batch_size=2,
            local_batch_size=1,
            grad_acc_steps=2,
            epoch=0,
            num_epochs=1,
            val_every_steps=2,
        ),
        lr_scheduler=SimpleNamespace(state_dict=lambda: {"last_epoch": 0}),
    )
    trainer._run_train_optim_step = lambda batches: SimpleNamespace(metrics={"loss": 0.75, "lr": 1e-4})
    recorder = _TrainingReproducibilityRecorder(trainer)
    recorder.attach()

    trainer._run_train_optim_step([{"input_ids": [1, 2, 3]}])

    payload = recorder.to_dict()
    assert set(payload["fingerprint_components"]) == {
        "model_and_initialization",
        "seed",
        "dataset_and_ordering",
        "batch_and_topology",
        "optimizer",
        "lr_scheduler",
        "loss_and_backend",
    }
    assert payload["steps"]["0"]["loss"] == 0.75
    assert payload["steps"]["0"]["lr"] == 1e-4
    assert len(payload["steps"]["0"]["batch_digest"]) == 64
