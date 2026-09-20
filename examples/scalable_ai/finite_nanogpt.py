"""Bounded variant of NeMo Automodel's NanogptDataset for validation.

``NanogptDataset`` is an infinite stream (its file iterator restarts the shard list forever, which is what
training wants), so a validation epoch over it never terminates. This subclass stops after ``max_samples``
samples per iterator (i.e. per data-parallel rank / dataloader worker).
"""

from nemo_automodel.components.datasets.llm.nanogpt_dataset import NanogptDataset


class FiniteNanogptDataset(NanogptDataset):
    """NanogptDataset that stops after ``max_samples`` samples per iterator (validation use)."""

    def __init__(self, *args, max_samples: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_samples = int(max_samples)

    def __iter__(self):
        for i, sample in enumerate(super().__iter__()):
            if i >= self.max_samples:
                return
            yield sample

    def __len__(self) -> int:  # type: ignore[override]
        return self.max_samples
