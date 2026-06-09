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

"""Tests for ``build_eagle3_dataloader`` worker configuration.

The EAGLE recipes initialize CUDA (the target model) before iterating the
dataloader. With ``num_workers > 0`` the workers must therefore (a) not inherit
the parent's live CUDA context -- ``fork`` does, and aborts the worker with
``cudaErrorInitializationError`` -- and (b) stay alive across epochs so the pool
is not re-forked at every epoch boundary. These tests pin both behaviors.
"""

from __future__ import annotations

from types import SimpleNamespace

from nemo_automodel.components.datasets.llm import eagle3


def _patch_dataset(monkeypatch, n: int = 4):
    """Replace ChatDataset with a tiny in-memory map-style dataset (no I/O)."""
    rows = [{"input_ids": [0], "loss_mask": [0], "attention_mask": [1]} for _ in range(n)]
    monkeypatch.setattr(eagle3, "ChatDataset", lambda *a, **k: rows)


def _build(**over):
    kwargs = dict(data_path="x", tokenizer=None, seq_length=8, batch_size=2, shuffle=False)
    kwargs.update(over)
    return eagle3.build_eagle3_dataloader(**kwargs)


def test_no_workers_keeps_defaults(monkeypatch):
    _patch_dataset(monkeypatch)
    monkeypatch.setattr(eagle3.torch.cuda, "is_available", lambda: False)
    dl = _build(num_workers=0)
    assert dl.num_workers == 0
    # persistent_workers / multiprocessing_context must NOT be forced on when
    # there are no worker processes (DataLoader rejects persistent_workers with
    # num_workers == 0).
    assert dl.persistent_workers is False
    assert dl.multiprocessing_context is None


def test_workers_are_persistent(monkeypatch):
    _patch_dataset(monkeypatch)
    monkeypatch.setattr(eagle3.torch.cuda, "is_available", lambda: False)
    dl = _build(num_workers=4)
    assert dl.num_workers == 4
    # Persistent across epochs -> no re-fork at the epoch boundary.
    assert dl.persistent_workers is True
    # No CUDA -> no need to override the start method.
    assert dl.multiprocessing_context is None


def test_workers_use_forkserver_when_cuda_available(monkeypatch):
    _patch_dataset(monkeypatch)
    monkeypatch.setattr(eagle3.torch.cuda, "is_available", lambda: True)
    dl = _build(num_workers=4)
    assert dl.persistent_workers is True
    # forkserver -> workers start clean, without the parent's CUDA context.
    assert dl.multiprocessing_context.get_start_method() == "forkserver"


def test_dp_mesh_sampler_keys_on_dp_axis(monkeypatch):
    """With a dp_mesh, the sampler distributes by dp size/rank so the cp ranks in
    a dp group pull the identical sample (CP shards its sequence across them),
    instead of keying on the global world rank."""
    _patch_dataset(monkeypatch, n=8)
    monkeypatch.setattr(eagle3.torch.cuda, "is_available", lambda: False)
    dp_mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: 1)
    dl = _build(distributed=True, dp_mesh=dp_mesh)
    assert isinstance(dl.sampler, eagle3.DistributedSampler)
    # Keyed on the dp sub-axis, not the global world.
    assert dl.sampler.num_replicas == 2
    assert dl.sampler.rank == 1
