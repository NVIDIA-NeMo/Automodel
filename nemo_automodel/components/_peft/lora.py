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
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nemo_automodel.components._peft.lora_kernel import (
    lora_da_dx_update_wrapper,
    lora_db_update_wrapper,
    lora_forward_wrapper,
)
from nemo_automodel.components._peft.lora_moe import GroupedExpertsDeepEPLoRA, GroupedExpertsLoRA
from nemo_automodel.components._peft.module_matcher import ModuleMatcher
from nemo_automodel.components.moe.layers import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.shared.import_utils import safe_import, safe_import_te
from nemo_automodel.shared.utils import dtype_from_str

HAS_BNB, bitsandbytes = safe_import("bitsandbytes")
HAS_TE, transformer_engine = safe_import_te()

logger = logging.getLogger(__name__)


@dataclass
class PeftConfig:
    target_modules: list = field(default_factory=list)
    exclude_modules: list = field(default_factory=list)
    match_all_linear: bool = False
    dim: int = 8
    alpha: int = 32
    # Note: we currently support DoRA for nn.Linear only.
    use_dora: bool = False
    dropout: float = 0.0
    dropout_position: Literal["pre", "post"] = "post"
    lora_A_init: str = "xavier"
    lora_dtype: Optional[torch.dtype] = None
    use_triton: bool = False

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]):
        return cls(
            target_modules=d.get("target_modules", []),
            exclude_modules=d.get("exclude_modules", []),
            match_all_linear=d.get("match_all_linear", False),
            dim=d.get("dim", 8),
            alpha=d.get("alpha", 32),
            use_dora=d.get("use_dora", False),
            dropout=d.get("dropout", 0.0),
            dropout_position=d.get("dropout_position", "post"),
            lora_A_init=d.get("lora_A_init", "xavier"),
            lora_dtype=d.get("lora_dtype", None),
            use_triton=d.get("use_triton", False),
        )


class LinearLoRA(nn.Linear):
    """
    Linear + LoRA, maintains ckpts structure (i.e. Linear's weight/bias remain at the same FQN).

    The _init_wrapper and _forward methods provide the LoRA functionality. We want to be able to
    use those inside LinearLoRA but also for monkey-patching modules, without repeating the
    same code -> therefore those are decorated with @staticmethod.
    """

    def __init__(
        self,
        orig_linear,
        dim=8,
        alpha=32,
        use_dora: bool = False,
        dropout=0.0,
        dropout_position="post",
        lora_A_init_method="xavier",
        lora_dtype=None,
    ):
        """
        LinearLora constructor.

        Args:
            orig_linear (nn.Module): the linear module to augment.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        assert isinstance(orig_linear, nn.Linear)
        super(LinearLoRA, self).__init__(
            in_features=orig_linear.in_features,
            out_features=orig_linear.out_features,
            bias=orig_linear.bias is not None,
            device=orig_linear.weight.device,
            dtype=orig_linear.weight.dtype,
        )
        # copy weights
        self.weight.data.copy_(orig_linear.weight.data)
        if orig_linear.bias is not None:
            self.bias.data.copy_(orig_linear.bias.data)
        # initialize the adapte
        LinearLoRA._init_adapter(
            self,
            dim=dim,
            alpha=alpha,
            use_dora=use_dora,
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @torch.no_grad
    def init_lora_weights(self, init_method: str):
        """
        Initialize the LoRA weights.

        Args:
            init_method (str): Method to initialize the LoRA weights.
        """
        if init_method == "xavier":
            nn.init.xavier_normal_(self.lora_A.weight.data)
        else:
            nn.init.kaiming_uniform_(self.lora_A.weight.data, a=math.sqrt(5))
        self.lora_B.weight.data.fill_(0)

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj,
        dim=8,
        alpha=32,
        use_dora: bool = False,
        dropout=0.0,
        dropout_position="post",
        lora_A_init_method="xavier",
        lora_dtype=None,
    ):
        """
        Adds LoRA weights to obj. Obj is either a LinearLoRA or an nn.Module (when monkey-patching).

        Args:
            obj (LinearLoRA | nn.Module): input module to adapt.
            dim (int): lora's dim in_features -> dim -> out_features.
            alpha (int): lora's scaling alpha.
            dropout (float): dropout prob (default: 0.0).
            dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
            lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
            lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
            are quantized weights (e.g. 4bit) needs to be specified explicitly.
        """
        obj.dim = dim
        obj.scale = alpha / dim
        obj.use_dora = bool(use_dora)

        # Freezer
        device = obj.weight.device
        obj.weight.requires_grad = False
        if obj.bias is not None:
            obj.bias.requires_grad = False

        in_features = obj.in_features
        out_features = obj.out_features
        if isinstance(lora_dtype, str):
            lora_dtype = dtype_from_str(lora_dtype)
        assert lora_dtype is None or isinstance(lora_dtype, torch.dtype)
        dtype = lora_dtype or obj.weight.dtype

        obj.lora_A = nn.Linear(in_features, dim, bias=False, dtype=dtype, device=device)
        obj.lora_B = nn.Linear(dim, out_features, bias=False, dtype=dtype, device=device)
        LinearLoRA.init_lora_weights(obj, lora_A_init_method)
        obj.dropout_p = dropout
        assert dropout_position in ["pre", "post"], ("dropout position can only be pre/post", dropout_position)
        obj.dropout_position = dropout_position

        if obj.use_dora:
            # initialize DoRA magnitude vector to ||W|| (row-wise L2 norm).
            with torch.no_grad():
                weight_norm = torch.linalg.norm(obj.weight.data, dim=1).to(dtype=dtype, device=device)
            obj.lora_magnitude = nn.Parameter(weight_norm, requires_grad=True)

    def _dora_weight_norm(self) -> torch.Tensor:
        """
        Compute the detached weight norm used by DoRA.
        """
        # ΔW = B @ A, shapes: [out, dim] @ [dim, in] => [out, in]
        delta_w = (self.lora_B.weight @ self.lora_A.weight).detach().to(self.weight.dtype)
        weight = self.weight.to(self.weight.dtype)
        weight_norm = torch.linalg.norm(weight + self.scale * delta_w, dim=1).to(weight.dtype)
        return weight_norm.detach()

    def forward(self, x):
        """
        Forward pass through the original linear layer augmented with the LoRA pathway.

        Applies LoRA either before or after the dropout, depending on the configuration.
        The result of the original linear transformation is combined with the LoRA output.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: Output tensor of shape (batch_size, out_features).
        """
        # pylint: disable=C0115,C0116
        # If LinearLoRA is used to monkey-patch a nn.Linear module, we want to use nn.Linear's
        # forward in the case where it uses quantized weights. We store a reference to nn.Linear's
        # forward in `super_fwd` attribute. If the attribute does not exist we do the usual linear.
        if (fwd := getattr(self, "super_fwd", None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            # TE Linear can expose an empty .bias tensor (numel()==0) when bias=False; treat as no bias.
            bias = self.bias
            if bias is not None and bias.numel() == 0:
                bias = None
            res = F.linear(x, self.weight, bias)

        if not self.use_dora:
            if self.dropout_position == "pre":
                x = F.dropout(x, p=self.dropout_p, training=self.training)

            # Apply scale before lora_B to keep lora_res as a Partial tensor.
            # This allows both res and lora_res to remain Partial, so only one reduce-scatter is needed after addition.
            # Multiplying after lora_B would convert Partial to Replicate, causing an extra reduce-scatter operation.
            lora_res = self.lora_B(self.lora_A(x) * self.scale)
            if self.dropout_position == "post":
                lora_res = F.dropout(lora_res, p=self.dropout_p, training=self.training)
            return res + lora_res

        if getattr(self, "lora_magnitude", None) is None:
            raise RuntimeError("use_dora=True but lora_magnitude was not initialized")

        if self.dropout_position == "pre" and self.training and self.dropout_p > 0.0:
            x_lora = F.dropout(x, p=self.dropout_p, training=True)
            base_result = None
        else:
            x_lora = x
            base_result = res

        lora_result = self.lora_B(self.lora_A(x_lora))
        if self.dropout_position == "post":
            lora_result = F.dropout(lora_result, p=self.dropout_p, training=self.training)

        # Compute DoRA scaling factor.
        weight_norm = self._dora_weight_norm()
        mag = self.lora_magnitude.to(x.dtype)
        weight_norm = weight_norm.to(x.dtype)

        # Broadcast magnitude scaling across batch/sequence dimensions.
        mag_norm_scale = mag / weight_norm
        if res.dim() == 3:
            mag_norm_scale = mag_norm_scale.view(1, 1, -1)
        else:
            mag_norm_scale = mag_norm_scale.view(1, -1)

        # HF PEFT subtracts bias from base_result before applying scaling terms.
        if base_result is not None:
            bias = self.bias
            if bias is not None and bias.numel() > 0:
                base_no_bias = base_result - bias
            else:
                base_no_bias = base_result
        else:
            # Recompute base linear output without bias on x_lora (see HF PEFT DoraLinearLayer.forward).
            base_no_bias = F.linear(x_lora, self.weight, None)

        dora_extra = (mag_norm_scale - 1) * base_no_bias + mag_norm_scale * lora_result * self.scale
        return res + dora_extra


class TritonLinearLoRA(LinearLoRA):
    """
    Subclass of LinearLoRA that uses triton kernels for forward and backward passes.

    Args:
        orig_linear (nn.Module): the linear module to augment.
        dim (int): lora's dim in_features -> dim -> out_features.
        alpha (int): lora's scaling alpha.
        dropout (float): dropout prob (default: 0.0).
        dropout_position (str): where to apply dropout rel. to lora (choices= ['pre', 'post'], default=post)
        lora_A_init_method (str): init method for lora_A (choices= ['xavier', 'uniform'])
        lora_dtype (torch.dtype): weight's dtype, by default will use orig_linear's but if they
        are quantized weights (e.g. 4bit) needs to be specified explicitly.
    """

    def forward(self, x):
        """
        Forward function for LoRA with triton kernels.

        Args:
            x (torch.Tensor): the input tensor.

        Returns:
            torch.Tensor: the output tensor.
        """
        # If LinearLoRA is used to monkey-patch a nn.Linear module, we want to use nn.Linear's
        # forward in the case where it uses quantized weights. We store a reference to nn.Linear's
        # forward in `super_fwd` attribute. If the attribute does not exist we do the usual linear.
        if (fwd := getattr(self, "super_fwd", None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if self.dropout_position == "pre":
            x = F.dropout(x, p=self.dropout_p, training=self.training)
        lora_res = LoRATritonFunction.apply(x, self.lora_A.weight, self.lora_B.weight, self.scale, x.dtype)
        if self.dropout_position == "post":
            lora_res = F.dropout(lora_res, p=self.dropout_p, training=self.training)

        return res + lora_res


def patch_linear_module(
    orig_linear,
    dim=8,
    alpha=32,
    use_dora: bool = False,
    dropout=0.0,
    dropout_position="post",
    lora_A_init_method="xavier",
    lora_dtype=None,
    use_triton=True,
    layer_name=None,
):
    """
    Monkey-patches a nn.Linear (orig_linear param) to be a LinearLoRA.

    The orig_linear might not contain valid weights, for example, the given orig_linear was
    initialized within a context-manager that uses a "meta" device. Therefore, we cannot copy
    the weight/bias from the orig_linear to the LinearLoRA, since those have not been allocated,

    To circumvent this scenario, LinearLoRA's additional functionality (_init_adapter, _forward)
    is based on static functions, so that we can use them for patching or when allocating a
    new LinearLoRA object.

    Args:
        orig_linear (nn.Linear): the module we add adapter to.
        dim (int, optional): Lora dim. Defaults to 8.
        alpha (int, optional): Lora alpha scale. Defaults to 32.
        dropout (float, optional): dropout prob. Defaults to 0.0.
        dropout_position (str, optional): location to apply dropout wrt lora.
            Defaults to 'post' (choices: 'pre', 'post').
        lora_A_init_method (str, optional): lora_a init method. Defaults to 'xavier'.
        lora_dtype (_type_, optional): Lora weights' dtype. By default will use orig_linear's dtype
        but orig_linear might use non-trainable dtype (e.g., 4bit), in which case the user must
        specify the dtype manually. Defaults to None.
        use_triton (bool, optional): By default we use the triton kernel LoRA implementation.

    Returns:
        (nn.Module): the monkey-patched (nn.Linear + LoRA) nn.Module
    """
    linear_types = [nn.Linear]
    if HAS_TE:
        linear_types.append(transformer_engine.pytorch.Linear)
        use_triton = False
    if not isinstance(orig_linear, tuple(linear_types)):
        raise NotImplementedError("Expected isinstance(orig_linear, nn.Linear)")
    assert not hasattr(orig_linear, "super_fwd"), orig_linear.super_fwd

    if use_dora:
        if HAS_TE and isinstance(orig_linear, transformer_engine.pytorch.Linear):
            raise ValueError("DoRA is not supported for transformer_engine.pytorch.Linear layers.")
        if getattr(orig_linear, "quant_state", None) is not None:
            raise ValueError("DoRA is not supported for quantized linear layers (e.g., BitsAndBytes).")
        use_triton = False

    linear_lora_cls = TritonLinearLoRA if use_triton else LinearLoRA
    linear_lora_cls._init_adapter(
        orig_linear,
        dim=dim,
        alpha=alpha,
        use_dora=use_dora,
        dropout=dropout,
        dropout_position=dropout_position,
        lora_A_init_method=lora_A_init_method,
        lora_dtype=lora_dtype,
    )
    cls = orig_linear.__class__
    new_cls = type("PatchedLinearLoRA", (linear_lora_cls, cls), {})

    # If the model uses quantized weights, we want to use orig_linear's forward
    if (
        getattr(orig_linear, "quant_state", None) is not None
        and orig_linear.quant_state.__class__ == bitsandbytes.functional.QuantState
    ):
        if HAS_TE:
            assert not isinstance(orig_linear, transformer_engine.pytorch.Linear), (
                "quant_state is not supported with transformer_engine.pytorch.Linear"
            )
        orig_linear.super_fwd = orig_linear.forward

    orig_linear.__class__ = new_cls
    if layer_name is not None:
        orig_linear._layer_name = layer_name
    return orig_linear


def patch_moe_module(
    orig_module,
    dim=8,
    alpha=32,
    lora_A_init_method="xavier",
    lora_dtype=None,
):
    """
    Patches a custom MoE module (GroupedExperts or GroupedExpertsDeepEP) with LoRA.

    Args:
        orig_module (nn.Module): The original MoE module to be patched.
        dim (int, optional): LoRA rank (dimension). Defaults to 8.
        alpha (int, optional): LoRA scaling factor. Defaults to 32.
        lora_A_init_method (str, optional): Initialization method for LoRA A matrix. Defaults to "xavier".
        lora_dtype (torch.dtype or str, optional): Data type for LoRA weights. Defaults to None.

    Returns:
        nn.Module: The LoRA-wrapped MoE module (GroupedExpertsLoRA or GroupedExpertsDeepEPLoRA).
    """
    if isinstance(orig_module, GroupedExpertsDeepEP):
        new_module = GroupedExpertsDeepEPLoRA(
            orig_module,
            lora_dim=dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
    elif isinstance(orig_module, GroupedExperts):
        new_module = GroupedExpertsLoRA(
            orig_module,
            lora_dim=dim,
            alpha=alpha,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )
    else:
        raise NotImplementedError(f"Unsupported MoE module type: {type(orig_module)}")

    return new_module


# patch a model in-place
def apply_lora_to_linear_modules(
    model: nn.Module,
    peft_config: PeftConfig,
    quantization_config=None,
    skip_freeze: bool = False,
) -> int:
    """
    Replace selected nn.Linear layers with LinearLoRA layers (in-place).

    Args:
        model: The model to apply LoRA to.
        peft_config: PEFT configuration for LoRA parameters.
        quantization_config: Optional separate QLoRA quantization configuration.
        skip_freeze: If True, skip the global parameter freeze (caller will handle it later).

    Returns:
        Number of modules that were modified with LoRA.

    Note:
        target_modules accepts wildcard fragments, e.g. ["q_proj", "k_proj", ".*fc.*"].
    """
    # Freeze base model parameters
    if not skip_freeze:
        for w in model.parameters():
            w.requires_grad_(False)

    is_causal_lm = False
    try:
        if (
            hasattr(model, "config")
            and model.config.architectures is not None
            and len(model.config.architectures) > 0
            and "CausalLM" in model.config.architectures[0]
        ):
            # for example, LlamaForCausalLM
            is_causal_lm = True
    except (AttributeError, TypeError):
        is_causal_lm = False

    matcher = ModuleMatcher(
        peft_config.target_modules, peft_config.exclude_modules, peft_config.match_all_linear, is_causal_lm
    )
    num_modules_matched = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, (GroupedExperts, GroupedExpertsDeepEP)):
            if matcher.match(module, name):
                if peft_config.use_dora:
                    raise NotImplementedError("DoRA is not supported for MoE expert modules in Automodel yet.")
                num_modules_matched += 1
                lora_dtype = peft_config.lora_dtype
                if quantization_config is not None and lora_dtype is None:
                    lora_dtype = quantization_config.bnb_4bit_compute_dtype or torch.bfloat16

                # Replace the module in the model
                new_module = patch_moe_module(
                    module,
                    dim=peft_config.dim,
                    alpha=peft_config.alpha,
                    lora_A_init_method=peft_config.lora_A_init,
                    lora_dtype=lora_dtype,
                )

                # Find parent and replace
                if "." not in name:
                    setattr(model, name, new_module)
                else:
                    parent_name, child_name = name.rsplit(".", 1)
                    parent = model.get_submodule(parent_name)
                    setattr(parent, child_name, new_module)
        else:
            # Standard Linear patching
            linear_types = [nn.Linear] + ([transformer_engine.pytorch.Linear] if HAS_TE else [])
            if isinstance(module, tuple(linear_types)) and matcher.match(module, name):
                num_modules_matched += 1
                # For QLora, set lora_dtype to float16/bfloat16 since base weights are quantized
                lora_dtype = peft_config.lora_dtype
                if quantization_config is not None and lora_dtype is None:
                    lora_dtype = quantization_config.bnb_4bit_compute_dtype or torch.bfloat16

                patch_linear_module(
                    module,
                    dim=peft_config.dim,
                    alpha=peft_config.alpha,
                    use_dora=peft_config.use_dora,
                    dropout=peft_config.dropout,
                    dropout_position=peft_config.dropout_position,
                    lora_A_init_method=peft_config.lora_A_init,
                    lora_dtype=lora_dtype,
                    use_triton=peft_config.use_triton,
                    layer_name=name,
                )

    return num_modules_matched


class LoRATritonFunction(torch.autograd.Function):
    """
    Autograd function that calls the triton kernel wrappers for the LoRA forward and backward passes.
    """

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Stores context for LoRA backward pass.
        """
        x, lora_A, lora_B, scale, _ = inputs
        ctx.save_for_backward(x, lora_A, lora_B)
        ctx.scale = scale

    @staticmethod
    def forward(x, lora_A, lora_B, scale, dtype):
        """
        Forward method for LoRATriton.

        Reshapes 3D tensors into 2D and then calls the triton kernel.
        """
        reshape = x.dim() == 3
        if reshape:
            bs, seq_len, d = x.shape
            x = x.reshape(-1, d)

        lora_res = lora_forward_wrapper(x, lora_A.t(), lora_B.t(), res=None, scale=scale, dtype=dtype)

        if reshape:
            return lora_res.view(bs, seq_len, -1)
        else:
            return lora_res

    @staticmethod
    def backward(ctx, d_y):
        """
        Backward method for LoRATriton.

        Reshapes 3D tensors into 2D and then calls the kernels to update d_lora_a, d_lora_b, and dx.
        """
        x, lora_A, lora_B = ctx.saved_tensors
        scale = ctx.scale
        dtype = x.dtype

        reshape = x.dim() == 3
        if reshape:
            bs, seq_len, d = x.shape
            d_y = d_y.reshape(-1, d_y.shape[-1])
            x = x.reshape(-1, d)

        d_lora_A, d_x = lora_da_dx_update_wrapper(x.t(), d_y, lora_B, lora_A, scale, dtype=dtype)
        d_lora_B = lora_db_update_wrapper(lora_A, x.t(), d_y, scale, dtype)

        if reshape:
            d_x = d_x.view(bs, seq_len, d)
        return d_x, d_lora_A.t(), d_lora_B, None, None
