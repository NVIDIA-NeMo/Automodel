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

import logging
import re
from dataclasses import dataclass, field
from types import MethodType

import torch
from torch.nn.parallel import DistributedDataParallel

from nemo_lm.utils.import_utils import safe_import_from

te, HAVE_TE = safe_import_from("transformer_engine", "pytorch")

ALL_MODULE_WRAPPER_CLASSNAMES = (DistributedDataParallel,)

logger = logging.getLogger(__name__)


def unwrap_model(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return_list = True
    if not isinstance(model, list):
        model = [model]
        return_list = False
    unwrapped_model = []
    for model_module in model:
        while isinstance(model_module, module_instances):
            model_module = model_module.module
        unwrapped_model.append(model_module)
    if not return_list:
        return unwrapped_model[0]
    return unwrapped_model


# -----------------------------------------------------------------------------
# TE utils
# -----------------------------------------------------------------------------


@dataclass
class TEConfig:
    """Config POD for Transformer Enginer config
    Options:
    - fp8_autocast (bool): indicated whether to autocast to FP8 or not.
    """

    fp8_autocast: bool = False


def te_accelerate(model, fp8_autocast=False):
    """
    Replaces original model layers with TE's accelerated layers
    Args:
        model: HF model
        fp8_autocast (bool): apply autocast or not
    """

    if not HAVE_TE:
        logger.warning("Transformer Engine is not available and the module replacements " "will not be applied.")
    else:
        _apply_basic_module_replacement(model)
        if fp8_autocast:
            apply_fp8_autocast(model)


@torch.no_grad
def _apply_basic_module_replacement(model):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            has_bias = module.bias is not None
            if any(p % 16 != 0 for p in module.weight.shape):
                continue
            te_module = te.Linear(
                module.in_features, module.out_features, bias=has_bias, params_dtype=module.weight.dtype
            )
            te_module.weight.copy_(module.weight)
            if has_bias:
                te_module.bias.copy_(module.bias)

            setattr(model, name, te_module)
        elif isinstance(module, torch.nn.LayerNorm):
            te_module = te.LayerNorm(module.normalized_shape[0], eps=module.eps, params_dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)
            setattr(model, name, te_module)
        elif isinstance(module, torch.nn.RMSNorm):
            te_module = te.RMSNorm(module.normalized_shape[0], eps=module.eps, dtype=module.weight.dtype)
            te_module.weight.copy_(module.weight)
            te_module.bias.copy_(module.bias)
            setattr(model, name, te_module)
        else:
            _apply_basic_module_replacement(module)


def is_te_accelerated(model):
    """
    Checks whether model has TE layers or not
    Args:
        model: HF model
    """

    if not HAVE_TE:
        logging.warning("Transformer Engine is not available.")
        return False
    else:
        for name, module in model.named_modules():
            if isinstance(module, (te.LayerNorm, te.Linear, te.TransformerLayer)):
                return True

        return False


def apply_fp8_autocast(model, fp8_recipe_handler=None):
    """
    Applies TE's autocast
    Args:
        model: HF model
        fp8_recipe_handler: fpt handler
    """

    if not HAVE_TE:
        logging.warning("Transformer Engine is not available and the FP8 autocast " "will not be applied.")
    else:
        import transformer_engine.common.recipe as te_recipe

        kwargs = fp8_recipe_handler.to_kwargs() if fp8_recipe_handler is not None else {}
        if "fp8_format" in kwargs:
            kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
        use_during_eval = kwargs.pop("use_autocast_during_eval", False)
        fp8_recipe = te_recipe.DelayedScaling(**kwargs)
        new_forward = _contextual_fp8_autocast(model.forward, fp8_recipe, use_during_eval)

        if hasattr(model.forward, "__func__"):
            model.forward = MethodType(new_forward, model)
        else:
            model.forward = new_forward


def _contextual_fp8_autocast(model_forward, fp8_recipe, use_during_eval=False):
    from transformer_engine.pytorch import fp8_autocast

    def forward(self, *args, **kwargs):
        enabled = use_during_eval or self.training
        with fp8_autocast(enabled=enabled, fp8_recipe=fp8_recipe):
            return model_forward(*args, **kwargs)

    forward.__wrapped__ = model_forward

    return forward


# -----------------------------------------------------------------------------
# Jit utils
# -----------------------------------------------------------------------------


def listify(x):
    """Wraps input in a list, if not already a list.

    Args:
        x (Anything): the input, can be anything.

    Returns:
        Anything | list(Anything): Anything (if it's already a list) o/w list(Anything)
    """
    if not isinstance(x, list):
        return [x]
    return x


def get_modules_from_selector(model, module_selector):
    """Iterator over model's modules whose FQN match the module_selector.

    Args:
        model (nn.Module): the model to iterate over.
        module_selector (str): module selector, if empty or '*' will return the whole model. If
        there's an asterisk in the name will match it as a regexp.

    Raises:
        AttributeError: if the user provides an invalid selector.
        AttributeError: if user's selector selects a non-nn.Module attribute.

    Yields:
        Iterator(nn.Module): iterator over modules whose FQN matches module_selector
    """
    if module_selector is None or module_selector == "" or module_selector == "*":
        yield model
        return

    assert isinstance(module_selector, str), module_selector
    atoms: list[str] = module_selector.split(".")
    tmp = model

    for i, item in enumerate(atoms):
        if "*" in item:
            # handle wildcard selector
            # TODO(@akoumparouli): support more complex selectors e.g. net_b.*.net_c.*.conv
            for name, module in tmp.named_children():
                if re.match(item.replace("*", ".*"), name):
                    yield module
            return

        if not hasattr(tmp, item):
            raise AttributeError(tmp._get_name() + " has no " "attribute `" + item + "`")
        tmp = getattr(tmp, item)

        if not isinstance(tmp, torch.nn.Module):
            raise AttributeError("`" + item + "` is not " "an nn.Module")

    yield tmp


def compile_module(config, module):
    """Jit-compiles an nn.Module

    Args:
        config (JitConfig): jit config
        module (nn.Module): the module to be compiled

    Returns:
        nn.Module: the (potentially) compiled module
    """
    if config.use_torch:
        module.compile(**config.torch_kwargs)
        return True
    elif config.use_thunder:
        import thunder
        import thunder.dynamo
        from thunder.dev_utils.nvtx_profile_transform import NvtxProfileTransform

        # With this setting, Dynamo Graphs inline all the modules (so Dynamo FXGraph just
        # consists of `call_function` nodes only and no `call_module` node.
        # This is the default setting in PyTorch 2.5 onwards
        # (see https://github.com/pytorch/pytorch/pull/131275)
        torch._dynamo.config.inline_inbuilt_nn_modules = True

        xforms: list = [NvtxProfileTransform()] if config.profile_thunder else []
        module.compile(backend=thunder.dynamo.ThunderCompiler(transforms=xforms))
        return True
    else:
        return False


@dataclass
class JitConfig:
    """Config POD for Jit transforms (e.g. torch.compile or thunder)
    Options:
    - module_selector (str): reg-exp to match modules to apply JitTransform to, useful for multi-trunk
      models where you want to apply it on one of them only. If empty will apply transform to root
      module.
    - use_torch (bool): whether to use torch.compile or not.
    - torch_kwargs (dict): kwargs to pass to torch.compile.
    - use_thunder (bool): whether to use thunder or not.
    - profile_thunder (bool): toggle for thunder's profiler.
    """

    module_selector: str = ""
    use_torch: bool = False
    torch_kwargs: dict = field(default_factory=dict)
    use_thunder: bool = False
    profile_thunder: bool = False

    def __post_init__(self):
        assert not (self.use_torch and self.use_thunder), "use_torch cannot be used at the same time with use_thunder"


def jit_compile_model(model: torch.nn.Module, jit_config: JitConfig):
    """Jit-compiles the model at the start of the epoch.
    While other events such as on_train_start are more suitable, we use on_train_epoch_start
    since that is what is used in peft (we want to jit after adding the adapters).

    Args:
        trainer (pl.Trainer): PTL trainer
        pl_module (pl.LightningModule): PTL module
    """
    if jit_config is None:
        return
    if not jit_config.use_thunder and not jit_config.use_torch:
        return

    if getattr(model, "_compiled", False):
        return

    # TODO(@akoumparouli): you want to concatenate (via regex OR-operator) all expressions
    # and trigger the compile if anyone matches, instead of iterating over all O(N^2).
    compiled = False
    for config in listify(jit_config):
        for module in get_modules_from_selector(model, config.module_selector):
            compiled |= compile_module(config, module)

    setattr(model, "_compiled", compiled)
