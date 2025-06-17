import pytest

# List of CLI overrides forwarded by the functional-test shell scripts.
# Registering them with pytest prevents the test discovery phase from
# aborting with "file or directory not found: --<option>" errors.
_OVERRIDES = [
    "config",
    "model.pretrained_model_name_or_path",
    "step_scheduler.max_steps",
    "step_scheduler.grad_acc_steps",
    "dataset.tokenizer.pretrained_model_name_or_path",
    "validation_dataset.tokenizer.pretrained_model_name_or_path",
    "dataset.dataset_name",
    "validation_dataset.dataset_name",
    "dataset.limit_dataset_samples",
    "step_scheduler.ckpt_every_steps",
    "checkpoint.enabled",
    "checkpoint.checkpoint_dir",
    "checkpoint.model_save_format",
    "dataloader.batch_size",
    "checkpoint.save_consolidated",
]


def pytest_addoption(parser: pytest.Parser):
    """Register the NeMo-Automodel CLI overrides so that pytest accepts them.
    The functional test launchers forward these arguments after a ``--``
    separator.  If pytest is unaware of an option it treats it as a file
    path and aborts collection.  Declaring each option here is enough to
    convince pytest that they are legitimate flags while still keeping
    them intact in ``sys.argv`` for the application code to parse later.
    """
    for opt in _OVERRIDES:
        # ``dest`` must be a valid Python identifier, so replace dots.
        dest = opt.replace(".", "_")
        parser.addoption(f"--{opt}", dest=dest, action="store", help=f"(passthrough) {opt}") 
