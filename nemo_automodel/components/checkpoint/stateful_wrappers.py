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

from functools import partial
from typing import Any, Optional

import torch
import transformer_engine.pytorch.module.base as te_base
import transformer_engine.pytorch.ops.op as te_ops

# The Conflict:
# PyTorch DCP passes an _EXTRA_STATE sentinel for missing keys, but Transformer Engine (TE)
# throws a RuntimeError if it receives anything other than None or a Tensor.
#
# The Fix (Monkeypatch):
# Intercept set_extra_state calls. If the input is the _EXTRA_STATE sentinel, return early
# (doing nothing) to safely ignore the missing state without crashing TE.
_original_set_extra_state = te_base.TransformerEngineBaseModule.set_extra_state
_original_op_set_extra_state = te_ops.BasicOperation.set_extra_state

def _safe_set_extra_state(self, state):
    if state is not None and "EXTRA_STATE" in str(type(state)):
        return
    return _original_set_extra_state(self, state)

def _safe_op_set_extra_state(self, state):
    if state is not None and "EXTRA_STATE" in str(type(state)):
        return
    return _original_op_set_extra_state(self, state)

te_base.TransformerEngineBaseModule.set_extra_state = _safe_set_extra_state
te_ops.BasicOperation.set_extra_state = _safe_op_set_extra_state

from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)

from nemo_automodel.components.checkpoint.utils import is_tied_word_embeddings

_PREFIX = "model."


def _is_quantized_module(module: torch.nn.Module) -> bool:
    """Check if a module is a BitsAndBytes quantized type.

    Detects quantization by checking for `quant_state` attribute which is
    common across BitsAndBytes quantized module types (Params4bit, Int8Params, etc.).
    """
    return getattr(module, "quant_state", None) is not None


def _has_quantized_params(model: torch.nn.Module) -> bool:
    """Check if model has any BitsAndBytes quantized modules."""
    return any(map(_is_quantized_module, model.modules()))


def _get_peft_state_dict(model: torch.nn.Module) -> dict[str, Any]:
    """Extract only trainable PEFT adapter weights, bypassing DCP for quantized models.

    This function directly iterates over model parameters to collect trainable weights,
    avoiding PyTorch DCP's state_dict traversal which fails on BitsAndBytes quantized
    modules (Params4bit, Int8Params, etc.).
    """
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param = param.full_tensor() if hasattr(param, "full_tensor") else param
            state_dict[name] = param.detach().cpu()
    return state_dict


def _drop_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX) -> None:
    """
    Remove the *first* occurrence of `prefix` on every key in-place.
    """
    for k in list(sd.keys()):
        if k.startswith(prefix):
            sd[k[len(prefix) :]] = sd.pop(k)


def _add_outer_prefix(sd: dict[str, Any], prefix: str = _PREFIX, skip_keys: list[str] = []) -> None:
    """
    Prepend `prefix` once to every key in-place (inverse of `_drop_outer_prefix`).
    """
    for k in list(sd.keys()):
        if not k.startswith(prefix) and k not in skip_keys:
            sd[prefix + k] = sd.pop(k)


def _rename_dora_keys_to_hf(sd: dict[str, Any]) -> None:
    """
    Rename DoRA magnitude keys to match HF PEFT's saved checkpoint format in-place.

    HF PEFT's ``get_peft_model_state_dict`` strips the adapter name and the
    ``.weight`` suffix from ``lora_magnitude_vector.<adapter>.<weight>`` so the
    round-trip format on disk is simply ``<module>.lora_magnitude_vector``.
    When loading, ``set_peft_model_state_dict`` re-inserts the adapter name
    and the ``.weight`` suffix automatically, so we must NOT include them here.
    """
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude"):
            sd[k[: -len(".lora_magnitude")] + ".lora_magnitude_vector"] = sd.pop(k)


def _rename_dora_keys_from_hf(sd: dict[str, Any]) -> None:
    """
    Reverse of _rename_dora_keys_to_hf: convert HF PEFT key format back to internal names.

    Handles both the current on-disk format (``<module>.lora_magnitude_vector``)
    and the legacy format that included ``.default.weight`` for robustness.
    """
    for k in list(sd.keys()):
        if k.endswith(".lora_magnitude_vector.default.weight"):
            sd[k[: -len(".lora_magnitude_vector.default.weight")] + ".lora_magnitude"] = sd.pop(k)
        elif k.endswith(".lora_magnitude_vector"):
            sd[k[: -len(".lora_magnitude_vector")] + ".lora_magnitude"] = sd.pop(k)


def _get_lm_head_weight_and_name(model: torch.nn.Module) -> Optional[tuple[torch.Tensor, str]]:
    for name, param in model.named_parameters(remove_duplicate=False):
        if "lm_head" in name and name.endswith(".weight"):
            normalized_name = name.replace("_orig_mod.", "")
            return param, normalized_name

    return None, None


# modified from pytorch tutorial https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html
class ModelState:
    """
    Helper class for tracking model state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model to track.
    """

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        is_peft: bool = False,
        is_init_step: bool = False,
        skip_task_head_prefixes: list[str] | None = None,
    ):
        """
        Initialize a ModelState instance for distributed checkpointing.

        The constructor records the model reference, detects whether the model
        ties its language-model head to the input embeddings, and stores the
        desired serialization backend so that DCP can correctly save and restore
        the model's parameters and buffers.

        Args:
            model (torch.nn.Module): The PyTorch model whose state should be
                captured during checkpointing.
            is_peft (bool): Whether the model is PEFT.
            is_init_step (bool): Whether the model is being initialized.
            skip_task_head_prefixes (list[str] | None): List of parameter name prefixes to skip when loading from base model. If None or empty, loads all parameters.
                Common examples:
                - ["classifier."] for sequence/token classification
                - ["qa_outputs."] for question answering
                - ["score."] for some classification heads
        """
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.is_tied_lm_head = is_tied_word_embeddings(self.model[0])

        if self.is_tied_lm_head:
            _, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            self.lm_head_param_name = lm_head_param_name
        self.is_peft = is_peft
        self.is_init_step = is_init_step
        self.skip_task_head_prefixes = skip_task_head_prefixes or []

    def state_dict(self) -> dict[str, Any]:
        """
        Get the model's state dictionary.

        Returns:
            dict: Dictionary containing the model's state dict with CPU offloading enabled.
        """
        if self.is_init_step:
            return self._get_base_model_state_dict()

        # For PEFT (LoRA/QLoRA/DoRA), only save trainable adapter weights. We bypass
        # get_model_state_dict() because: (1) with quantized params (QLoRA) DCP fails
        # traversing Params4bit; (2) with expert parallelism (MoE+EP) DCP expects
        # full-model FQNs and raises KeyError when popping expert params that are
        # sharded across ranks. Direct collection ensures we save only adapter weights,
        # matching dense LLaMA/Qwen LoRA behavior and fixing MoE LoRA checkpointing.
        if self.is_peft:
            model_state_dict = {k: v for sd in map(_get_peft_state_dict, self.model) for k, v in sd.items()}
        else:
            options = None
            func = partial(get_model_state_dict, options=options)
            model_state_dict = {k: v for sd in map(func, self.model) for k, v in sd.items()}

        if self.is_tied_lm_head:
            # PP models don't have tied embeddings. Safe to pass in model[0] here.
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft and not _has_quantized_params(self.model[0]):
            # HF PEFT models are saved with a "base.model." prefix. This is so they can be loaded
            # correctly with the HF PEFT API.
            _add_outer_prefix(model_state_dict, "base_model.model.")
            # DoRA: rename lora_magnitude to match HF PEFT's expected key format
            _rename_dora_keys_to_hf(model_state_dict)

        return model_state_dict

    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> None:
        """
        Load the state dictionary into the model.

        Args:
            state_dict (dict): State dictionary to load.
        """
        if self.is_init_step:
            self._set_base_model_state_dict(state_dict)
            return

        # Multi-stage PP models have different state dicts for each stage.
        options = StateDictOptions(strict=strict)
        if self.is_peft:
            _drop_outer_prefix(state_dict, "base_model.model.")
            # DoRA: reverse the HF PEFT key rename so DCP can match model params
            _rename_dora_keys_from_hf(state_dict)
            
            options = StateDictOptions(strict=False, full_state_dict=True)

        # If we intentionally skipped saving "lm_head.weight" (tied embeddings)
        # PyTorch will complain during load even with strict=False.
        # To be fully compatible we inject a reference tensor so the key exists.
        if self.is_tied_lm_head and not self.is_peft:
            # PP models don't have tied embeddings. Safe to pass in model[0] here.
            lm_head_weight, lm_head_param_name = _get_lm_head_weight_and_name(self.model[0])
            # Skip for Biencoder models as it doesn't have a lm_head at the top level
            if lm_head_weight is not None and lm_head_param_name not in state_dict:
                # weight tying guarantees this is identical to the embedding weight
                state_dict[lm_head_param_name] = lm_head_weight.detach()

        func = partial(set_model_state_dict, model_state_dict=state_dict, options=options)
        list(map(func, self.model))

    def _get_base_model_state_dict(self) -> dict[str, Any]:
        model_state_dict = {k: v for sd in map(get_model_state_dict, self.model) for k, v in sd.items()}

        if self.is_tied_lm_head:
            # PP models don't have tied embeddings. Safe to pass in model[0] here.
            model_state_dict.pop(self.lm_head_param_name, None)

        if self.is_peft:
            keys_to_remove = [k for k in model_state_dict.keys() if "lora" in k]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        if self.skip_task_head_prefixes:
            # Remove task-specific heads when loading base model for fine-tuning
            # These layers don't exist in base pretrained models and will be randomly initialized
            keys_to_remove = [
                k
                for k in model_state_dict.keys()
                if any(k.startswith(prefix) for prefix in self.skip_task_head_prefixes)
            ]
            for k in keys_to_remove:
                model_state_dict.pop(k)

        return model_state_dict

    def _set_base_model_state_dict(self, state_dict: dict[str, Any]) -> None:
        func = partial(set_model_state_dict, model_state_dict=state_dict, options=StateDictOptions(strict=False))
        list(map(func, self.model))


class OptimizerState:
    """
    Helper class for tracking optimizer state in distributed checkpointing.

    This class is compliant with the Stateful protocol, allowing DCP to automatically
    call state_dict/load_state_dict as needed in the dcp.save/load APIs.

    Args:
        model: The PyTorch model associated with the optimizer.
        optimizer: The optimizer to track.
        scheduler: Optional learning rate scheduler.
    """

    def __init__(
        self,
        model: torch.nn.Module | list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any] = None,
        is_peft: bool = False,
    ):
        """
        Initialize an OptimizerState instance.

        The constructor simply stores references to the model, optimizer, and
        (optionally) learning-rate scheduler so that their state can be captured
        and restored by the Distributed Checkpointing (DCP) framework.

        Args:
            model (torch.nn.Module): The neural-network model whose parameters the
                optimizer updates. Keeping the reference allows DCP to re-establish
                the model–optimizer relationship when loading a checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer whose internal buffers
                (e.g., momentum, Adam moments, step counters) need to be saved and
                restored.
            scheduler (Optional[Any], optional): Learning-rate scheduler to track
                alongside the optimizer. Pass ``None`` if no scheduler is used.
            is_peft (bool): Whether the model uses PEFT adapters (e.g. LoRA/QLoRA).
        """
        self.model = [model] if isinstance(model, torch.nn.Module) else model
        self.optimizer = [optimizer] if isinstance(optimizer, torch.optim.Optimizer) else optimizer
        self.scheduler = [scheduler] if isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler) else scheduler
        self.is_peft = is_peft

    def state_dict(self) -> dict[str, Any]:
        """
        Get the optimizer and scheduler state dictionaries.

        Returns:
            dict: Dictionary containing the optimizer and scheduler state dicts with CPU offloading enabled.
        """
        # For PEFT (LoRA/QLoRA/DoRA), bypass DCP's get_optimizer_state_dict() which
        # builds a parameter-ID-to-FQN mapping from the full model and then fails
        # (KeyError) when the optimizer only has state for trainable params and the
        # model is sharded (e.g. MoE+EP). Use native optimizer state_dict instead.
        if self.is_peft:
            optimizer_state_dict = self.optimizer[0].state_dict()
        else:
            # this line automatically manages FSDP FQN's, as well as sets the default state dict type
            # to FSDP.SHARDED_STATE_DICT
            func = partial(
                get_optimizer_state_dict,
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            optimizer_state_dict = {k: v for sd in map(func, self.model, self.optimizer) for k, v in sd.items()}

        state_dict = {
            "optim": optimizer_state_dict,
        }
        if self.scheduler is not None:
            state_dict["sched"] = self.scheduler[0].state_dict()

        return state_dict

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """
        Load the state dictionaries into the optimizer and scheduler.

        Args:
            state_dict (dict): State dictionary containing optimizer and scheduler states to load.
        """
        # For PEFT, use native load to match the native save path.
        if self.is_peft:
            self.optimizer[0].load_state_dict(state_dict["optim"])
        else:
            # sets our state dicts on the optimizer, now that we've loaded
            func = partial(
                set_optimizer_state_dict,
                optim_state_dict=state_dict["optim"],
                options=StateDictOptions(flatten_optimizer_state_dict=True),
            )
            list(map(func, self.model, self.optimizer))

        # load the scheduler state if it exists
        if "sched" in state_dict and self.scheduler is not None:
            list(map(lambda x: x.load_state_dict(state_dict["sched"]), self.scheduler))
