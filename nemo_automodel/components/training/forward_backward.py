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
import torch.distributed as dist

from nemo_automodel.components.distributed.cp_utils import make_cp_batch_and_ctx
from nemo_automodel.components.distributed.mesh_utils import get_flat_mesh
from nemo_automodel.components.distributed.utils import get_sync_ctx
from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
from nemo_automodel.components.training.model_output_utils import get_final_hidden_states
from nemo_automodel.components.utils.model_utils import filter_forward_kwargs


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


def _get_dp_group_size(device_mesh: Any, *, include_cp: bool = True) -> int:
    if device_mesh is None:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
        return 1

    mesh_dim_names = getattr(device_mesh, "mesh_dim_names", ())
    cp_size = device_mesh["cp"].size() if include_cp and "cp" in mesh_dim_names else 1
    if cp_size > 1:
        return get_flat_mesh(device_mesh, "dp_cp").get_group().size()
    return get_flat_mesh(device_mesh, "dp").get_group().size()


def calculate_loss(
    loss_fn: Callable[..., torch.Tensor],
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    model: torch.nn.Module,
    hidden_states: Any,
    num_label_tokens: int | None,
) -> torch.Tensor:
    loss_fn_kwargs = {"num_label_tokens": num_label_tokens}
    if isinstance(loss_fn, FusedLinearCrossEntropy):
        lm_head = None
        if hasattr(model, "get_output_embeddings"):
            lm_head = model.get_output_embeddings().weight
        else:
            for name, param in model.named_parameters(remove_duplicate=False):
                if "lm_head" in name and name.endswith(".weight"):
                    lm_head = param
                    break
        if lm_head is None:
            raise ValueError("lm_head.weight not found in model")

        lm_head = lm_head.full_tensor() if hasattr(lm_head, "full_tensor") else lm_head
        loss_fn_kwargs.update(
            {
                "hidden_states": hidden_states,
                "labels": labels,
                "lm_weight": lm_head,
            }
        )
    else:
        loss_fn_kwargs.update(
            {
                "logits": logits,
                "labels": labels,
            }
        )

    return loss_fn(**loss_fn_kwargs)


def forward_backward_step(
    *,
    batch: MutableMapping[str, Any],
    device: torch.device,
    device_mesh: Any,
    model_parts: list[torch.nn.Module],
    distributed_config: Any,
    loss_fn: Callable[..., torch.Tensor] | None,
    num_label_tokens: int | None,
    is_train: bool,
    pp: Any | None,
    is_last_microbatch: bool = True,
    make_cp_batch_kwargs: dict[str, Any] | None = None,
    prepare_batch_before_cp: Callable[[MutableMapping[str, Any]], MutableMapping[str, Any]] | None = None,
    model_context_factory: Callable[[], Any] | None = None,
    pp_batch_context_factory: Callable[[MutableMapping[str, Any]], Any] | None = None,
    hidden_states_error_message: str = (
        "FusedLinearCrossEntropy requires the model to output hidden states. "
        "Set `model.output_hidden_states=True` in the config."
    ),
) -> torch.Tensor:
    """Run one LLM/VLM forward-backward step.

    Recipes supply hooks for modality-specific preparation while sharing the
    common CP, PP, loss, and backward control flow.
    """
    pp_enabled = pp is not None
    batch = {key: move_to_device(value, device) for key, value in batch.items()}

    if prepare_batch_before_cp is not None:
        batch = prepare_batch_before_cp(batch)

    train_ctx, batch = make_cp_batch_and_ctx(device_mesh, batch, **(make_cp_batch_kwargs or {}))
    labels = batch.pop("labels")
    model_ctx = model_context_factory() if model_context_factory is not None else nullcontext()

    if pp_enabled:
        with train_ctx(), model_ctx:
            losses = [] if pp.info.has_last_stage else None
            targets = labels.clone() if pp.info.has_last_stage else None

            input_ids = batch.pop("input_ids")
            pp.update_seq_len(input_ids.shape[1])

            pp_batch = batch
            if pp_batch_context_factory is None:
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
        return local_loss.detach()

    model = model_parts[0]
    sync_ctx = (
        get_sync_ctx(
            model,
            is_last_microbatch,
            defer_fsdp_grad_sync=getattr(distributed_config, "defer_fsdp_grad_sync", True),
        )
        if is_train
        else nullcontext()
    )

    with train_ctx(), sync_ctx, model_ctx:
        batch = filter_forward_kwargs(model, batch)
        if isinstance(loss_fn, FusedLinearCrossEntropy):
            out = model(logits_to_keep=1, **batch)
            hidden_states = get_final_hidden_states(out)
            if hidden_states is None:
                raise ValueError(hidden_states_error_message)
        else:
            out = model(**batch)
            hidden_states = get_final_hidden_states(out)

        if loss_fn is None:
            raise ValueError("loss_fn must be provided for non-PP forward-backward")

        local_loss = calculate_loss(
            loss_fn=loss_fn,
            logits=getattr(out, "logits", out),
            labels=labels,
            model=model,
            hidden_states=hidden_states,
            num_label_tokens=num_label_tokens,
        )
        if is_train:
            dp_group_size = _get_dp_group_size(device_mesh, include_cp=True)
            (local_loss * dp_group_size).backward()
        return local_loss.detach()


__all__ = ["calculate_loss", "forward_backward_step", "move_to_device"]
