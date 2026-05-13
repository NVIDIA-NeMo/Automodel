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

import logging
from collections.abc import Callable, MutableMapping
from contextlib import nullcontext
from typing import Any

import torch

from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states


def move_to_device(value: Any, device: torch.device) -> Any:
    """Recursively move tensors in a batch value to a device."""
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: move_to_device(v, device) if v is not None else None for k, v in value.items()}
    if isinstance(value, list):
        return [move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(v, device) for v in value)
    return value


def forward_backward_step(
    *,
    idx: int,
    batch: MutableMapping[str, Any],
    device: torch.device,
    device_mesh: Any,
    model_parts: list[torch.nn.Module],
    distributed_config: Any,
    loss_fn: Callable,
    calculate_loss_fn: Callable[..., torch.Tensor],
    loss_buffer: list[torch.Tensor],
    num_label_tokens: int | None,
    num_batches: int,
    is_train: bool,
    pp_enabled: bool,
    pp: Any | None,
    dp_group_size: int,
    make_cp_batch_and_ctx_fn: Callable,
    get_sync_ctx_fn: Callable,
    filter_forward_kwargs_fn: Callable,
    make_cp_batch_kwargs: dict[str, Any] | None = None,
    pre_cp_transform: Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]] | None = None,
    fp8_context_factory: Callable[[], Any] | None = None,
    pp_batch_context_factory: Callable[[MutableMapping[str, Any]], Any] | None = None,
    pp_eval_enabled: bool = True,
    pp_validation_skip_message: str = "Skipping forward pass for validation because pipeline parallelism is enabled",
) -> None:
    """Run one forward/backward microstep for LLM and VLM recipes.

    Recipe-specific behavior is expressed through hooks:
    - ``pre_cp_transform`` may rewrite the batch before CP chunking.
    - ``pp_batch_context_factory`` may stage already-prepared PP-only batch data.
    """
    batch = {k: move_to_device(v, device) for k, v in batch.items()}

    if pre_cp_transform is not None:
        batch = pre_cp_transform(batch)

    train_ctx, batch = make_cp_batch_and_ctx_fn(device_mesh, batch, **(make_cp_batch_kwargs or {}))
    labels = batch.pop("labels")
    fp8_ctx = fp8_context_factory() if fp8_context_factory is not None else nullcontext()

    if pp_enabled:
        if pp is None:
            raise ValueError("pp must be provided when pp_enabled=True")
        if not is_train and not pp_eval_enabled:
            logging.info(pp_validation_skip_message)
            return

        with train_ctx(), fp8_ctx:
            losses = [] if pp.info.has_last_stage else None
            if pp.info.has_last_stage:
                targets = labels.clone()
            else:
                targets = None

            input_ids = batch.pop("input_ids")
            pp.update_seq_len(input_ids.shape[1])

            batch_filtered = {
                k: v for k, v in batch.items() if v is not None and not (isinstance(v, dict) and len(v) == 0)
            }
            pp_batch_ctx = (
                pp_batch_context_factory(batch_filtered)
                if pp_batch_context_factory is not None
                else nullcontext()
            )
            schedule_fn = pp.info.schedule.step if is_train else pp.info.schedule.eval
            with pp_batch_ctx:
                if pp.info.has_first_stage:
                    schedule_fn(input_ids, target=targets, losses=losses, **batch_filtered)
                else:
                    schedule_fn(target=targets, losses=losses, **batch_filtered)

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
    with train_ctx(), sync_ctx, fp8_ctx:
        batch = filter_forward_kwargs_fn(model, batch)
        if isinstance(loss_fn, FusedLinearCrossEntropy):
            out = model(logits_to_keep=1, **batch)
            hidden_states = get_final_hidden_states(out)
            if hidden_states is None:
                raise ValueError(
                    "FusedLinearCrossEntropy requires the model to output hidden states. "
                    "Set `model.output_hidden_states=True` in the config."
                )
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
