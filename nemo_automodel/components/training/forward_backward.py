# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
from typing import Any

import torch

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch value to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) if item is not None else None for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def forward_backward_step(
    *,
    idx: int,
    batch: MutableMapping[str, Any],
    device: torch.device,
    device_mesh: Any,
    model_parts: list[torch.nn.Module],
    distributed_config: Any,
    loss_fn: Callable[..., torch.Tensor] | None,
    calculate_loss_fn: Callable[..., torch.Tensor],
    loss_buffer: list[torch.Tensor],
    num_label_tokens: int | None,
    num_batches: int,
    is_train: bool,
    pp_enabled: bool,
    pp: Any | None,
    dp_group_size: int,
    make_cp_batch_and_ctx_fn: Callable[..., tuple[Callable[[], Any], MutableMapping[str, Any]]],
    get_sync_ctx_fn: Callable[..., Any],
    filter_forward_kwargs_fn: Callable[[torch.nn.Module, MutableMapping[str, Any]], MutableMapping[str, Any]],
    make_cp_batch_kwargs: dict[str, Any] | None = None,
    prepare_batch_before_cp: Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]] | None = None,
    model_context_factory: Callable[[], Any] | None = None,
    pp_batch_context_factory: Callable[[MutableMapping[str, Any]], Any] | None = None,
    filter_pp_batch: bool = True,
    hidden_states_error_message: str = (
        "FusedLinearCrossEntropy requires the model to output hidden states. "
        "Set `model.output_hidden_states=True` in the config."
    ),
) -> None:
    """Run one LLM/VLM forward-backward step.

    Recipes supply hooks for modality-specific preparation while sharing the
    common CP, PP, loss, and backward control flow.
    """
    batch = {key: move_to_device(value, device) for key, value in batch.items()}

    if prepare_batch_before_cp is not None:
        batch = prepare_batch_before_cp(batch)

    train_ctx, batch = make_cp_batch_and_ctx_fn(device_mesh, batch, **(make_cp_batch_kwargs or {}))
    labels = batch.pop("labels")
    model_ctx = model_context_factory() if model_context_factory is not None else nullcontext()

    if pp_enabled:
        if pp is None:
            raise ValueError("pp must be provided when pp_enabled=True")

        with train_ctx(), model_ctx:
            losses = [] if pp.info.has_last_stage else None
            targets = labels.clone() if pp.info.has_last_stage else None

            input_ids = batch.pop("input_ids")
            pp.update_seq_len(input_ids.shape[1])

            pp_batch = batch
            if filter_pp_batch:
                pp_batch = {
                    key: value
                    for key, value in batch.items()
                    if value is not None and not (isinstance(value, dict) and len(value) == 0)
                }

            pp_batch_ctx = (
                pp_batch_context_factory(pp_batch) if pp_batch_context_factory is not None else nullcontext()
            )
            schedule_fn = pp.info.schedule.step if is_train else pp.info.schedule.eval
            with pp_batch_ctx:
                if pp.info.has_first_stage:
                    schedule_fn(input_ids, target=targets, losses=losses, **pp_batch)
                else:
                    schedule_fn(target=targets, losses=losses, **pp_batch)

        if pp.info.has_last_stage:
            local_loss = torch.sum(torch.stack(losses))
        else:
            local_loss = torch.tensor(0.0, device=device)
        loss_buffer.append(local_loss.clone().detach())
        return

    model = model_parts[0]
    sync_ctx = (
        get_sync_ctx_fn(
            model,
            idx == num_batches - 1,
            defer_fsdp_grad_sync=getattr(distributed_config, "defer_fsdp_grad_sync", True),
        )
        if is_train
        else nullcontext()
    )

    with train_ctx(), sync_ctx, model_ctx:
        batch = filter_forward_kwargs_fn(model, batch)
        if isinstance(loss_fn, FusedLinearCrossEntropy):
            out = model(logits_to_keep=1, **batch)
            hidden_states = get_final_hidden_states(out)
            if hidden_states is None:
                raise ValueError(hidden_states_error_message)
        else:
            out = model(**batch)
            hidden_states = get_final_hidden_states(out)

        local_loss = calculate_loss_fn(
            loss_fn,
            logits=getattr(out, "logits", out),
            labels=labels,
            model=model,
            hidden_states=hidden_states,
            num_label_tokens=num_label_tokens,
        )
        loss_buffer.append(local_loss.clone().detach())
        if is_train:
            (local_loss * dp_group_size).backward()


__all__ = ["forward_backward_step", "move_to_device"]
