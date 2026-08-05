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

import json
from copy import deepcopy
from types import SimpleNamespace

import torch
from torch.utils.data import TensorDataset
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from nemo_automodel.components.training.rng import StatefulRNG
from nemo_automodel.components.training.step_scheduler import StepScheduler
from tests.functional_tests.checkpoint_robustness.resume_trajectory import (
    _checkpoint_state_snapshot,
    _compare_training_reproducibility,
    _configure_resumed_run,
    _configure_uninterrupted_run,
    _report_training_reproducibility,
    _restored_state_mismatch,
    _resume_plan_from_config,
    _ResumePlan,
    _state_digest,
    _TrainingReproducibilityRecorder,
    _trajectory_mismatch,
    _TrajectoryRecorder,
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


def test_resume_state_check_detects_omitted_rng_state():
    reference = {
        "step_scheduler": {"step": 5, "epoch": 0},
        "optimizer_steps": [{"5.0": 2}],
        "optimizer_groups": [[{"lr": 1e-4, "weight_decay": 0.01}]],
        "lr_scheduler_digest": "lr",
        "rng_digest": "rng",
    }
    restored = {key: value for key, value in reference.items() if key != "rng_digest"}

    mismatch = _restored_state_mismatch(reference, restored)

    assert mismatch == "restored snapshot omitted required RNG state (rng_digest)"


def test_stateful_sampler_restore_uses_next_batch_instead_of_raw_state_digest():
    dataset = TensorDataset(torch.arange(8))

    def make_dataloader() -> StatefulDataLoader:
        sampler = StatefulDistributedSampler(
            dataset,
            seed=42,
            drop_last=True,
            num_replicas=1,
            rank=0,
            shuffle=True,
        )
        return StatefulDataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)

    uninterrupted = make_dataloader()
    uninterrupted_iterator = iter(uninterrupted)
    next(uninterrupted_iterator)
    checkpoint_state = uninterrupted.state_dict()
    checkpoint_digest = _state_digest(checkpoint_state)

    resumed = make_dataloader()
    resumed.load_state_dict(checkpoint_state)

    assert _state_digest(resumed.state_dict()) != checkpoint_digest
    expected_batch = next(uninterrupted_iterator)
    resumed_batch = next(iter(resumed))
    assert torch.equal(resumed_batch[0], expected_batch[0])


def test_shared_trajectory_harness_runs_checkpoint_and_resume_locally(tmp_path):
    plan = _ResumePlan(checkpoint_dir=tmp_path, boundary_step=2, continuation_steps=2)
    checkpoints: dict[int, dict[str, object]] = {}

    def make_trainer() -> SimpleNamespace:
        dataset = TensorDataset(torch.arange(16, dtype=torch.float32))
        sampler = StatefulDistributedSampler(
            dataset,
            seed=42,
            drop_last=True,
            num_replicas=1,
            rank=0,
            shuffle=True,
        )
        dataloader = StatefulDataLoader(dataset, batch_size=2, sampler=sampler, num_workers=0)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
        trainer = SimpleNamespace(
            dataloader=dataloader,
            parameter=parameter,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            rng=StatefulRNG(seed=123),
            step_scheduler=StepScheduler(
                global_batch_size=2,
                local_batch_size=2,
                dp_size=1,
                dataloader=dataloader,
                ckpt_every_steps=plan.boundary_step,
                save_checkpoint_every_epoch=False,
                max_steps=plan.final_max_steps,
                preemption_signal=None,
            ),
        )

        def train_step(batches: list[tuple[torch.Tensor]]) -> SimpleNamespace:
            """Run one deterministic optimizer step over scalar sample tensors.

            Args:
                batches: List of microbatches, each containing a tensor of shape [batch].

            Returns:
                Namespace containing scalar loss and learning-rate metrics.
            """
            optimizer.zero_grad()
            target = batches[0][0].mean()
            loss = (parameter - target).square()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            return SimpleNamespace(metrics={"loss": float(loss.detach()), "lr": optimizer.param_groups[0]["lr"]})

        def save_checkpoint(
            epoch: int,
            step: int,
            train_loss: float,
            val_loss: dict[str, float] | None = None,
            best_metric_key: str = "default",
        ) -> None:
            del epoch, train_loss, val_loss, best_metric_key
            checkpoints[int(step)] = {
                "model": parameter.detach().clone(),
                "optimizer": deepcopy(optimizer.state_dict()),
                "lr_scheduler": deepcopy(lr_scheduler.state_dict()),
                "rng": deepcopy(trainer.rng.state_dict()),
                "dataloader": deepcopy(dataloader.state_dict()),
                "step_scheduler": deepcopy(trainer.step_scheduler.state_dict()),
            }

        trainer._run_train_optim_step = train_step
        trainer.save_checkpoint = save_checkpoint
        return trainer

    def run(trainer: SimpleNamespace) -> None:
        for epoch in trainer.step_scheduler.epochs:
            trainer.step_scheduler.set_epoch(epoch)
            for batches in trainer.step_scheduler:
                log_data = trainer._run_train_optim_step(batches)
                if trainer.step_scheduler.is_ckpt_step:
                    trainer.save_checkpoint(
                        epoch,
                        trainer.step_scheduler.step,
                        log_data.metrics["loss"],
                    )

    uninterrupted = make_trainer()
    reference_recorder = _TrajectoryRecorder(plan, capture_boundary_state=True)
    reference_recorder.attach(uninterrupted)
    run(uninterrupted)
    reference = reference_recorder.to_dict()

    resumed = make_trainer()
    checkpoint = checkpoints[plan.boundary_step - 1]
    with torch.no_grad():
        resumed.parameter.copy_(checkpoint["model"])
    resumed.optimizer.load_state_dict(checkpoint["optimizer"])
    resumed.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    resumed.rng.load_state_dict(checkpoint["rng"])
    resumed.dataloader.load_state_dict(checkpoint["dataloader"])
    resumed.step_scheduler.load_state_dict(checkpoint["step_scheduler"])

    restored_state = _checkpoint_state_snapshot(resumed, state_is_being_saved=False)
    assert _restored_state_mismatch(reference["boundary_state"], restored_state) is None

    resumed_recorder = _TrajectoryRecorder(plan, capture_boundary_state=False)
    resumed_recorder.attach(resumed)
    run(resumed)
    assert (
        _trajectory_mismatch(
            reference,
            resumed_recorder.to_dict(),
            first_loss_threshold=0.0,
            later_loss_threshold=0.0,
        )
        is None
    )


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


def test_training_reproducibility_persists_prominent_nonblocking_alert(tmp_path, capsys):
    (tmp_path / "normal_rank_0.json").write_text(json.dumps(_training_run()))
    checkpoint_recorder = SimpleNamespace(to_dict=lambda: _training_run(second_loss=0.96))

    _report_training_reproducibility(tmp_path, checkpoint_recorder, loss_threshold=5e-2)

    output = capsys.readouterr().out
    assert "[Training reproducibility][non-blocking][ALERT]" in output
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["blocking"] is False
    assert report["status"] == "alert"
    assert report["reports"][0]["status"] == "outside_tolerance"


def test_training_reproducibility_detects_independent_batch_order_divergence():
    normal = _training_run()
    checkpoint = _training_run(first_batch="different-batch")

    report = _compare_training_reproducibility(normal, checkpoint, loss_threshold=5e-3)

    assert report["status"] == "diverged"
    assert report["reason"] == "batch identity mismatch"
    assert report["step"] == 0


def test_training_reproducibility_recorder_captures_runtime_batch_and_metrics():
    step_scheduler = StepScheduler(
        global_batch_size=2,
        local_batch_size=1,
        dp_size=1,
        dataloader=[0, 1],
        max_steps=2,
        preemption_signal=None,
    )
    assert not hasattr(step_scheduler, "global_batch_size")
    assert not hasattr(step_scheduler, "local_batch_size")
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
                "step_scheduler": {"global_batch_size": 2, "local_batch_size": 1},
            }
        ),
        step_scheduler=step_scheduler,
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
