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

import math
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torch import nn

from nemo_automodel._peft.lora_kernel import (
    lora_da_dx_update_wrapper,
    lora_db_update_wrapper,
    lora_forward_wrapper,
)
from nemo_automodel._peft.module_matcher import ModuleMatcher
from nemo_automodel.shared.import_utils import safe_import


HAS_BNB, bitsandbytes = safe_import("bitsandbytes")

MODEL_TYPE_TO_PEFT_TASK_TYPE = {
    "SequenceClassification": "SEQ_CLS",
    "Seq2SeqLM": "SEQ_2_SEQ_LM",
    "CausalLM": "CAUSAL_LM",
    "TokenClassification": "TOKEN_CLS",
    "QuestionAnswering": "QUESTION_ANS",
    "FeatureExtraction": "FEATURE_EXTRACTION",
    "ConditionalGeneration": "CONDITIONAL_GENERATION",
}

def dtype_from_str(val):
    """
    Translate a str val of a dtype into the corresponding torch.dtype
    Args:
        val (str): the dotted path of the dtype (e.g., "torch.bfloat16").

    Returns:
        torch.dtype: the actual dtype (e.g., torch.bfloat16)
    """
    if isinstance(val, torch.dtype):
        return val
    lut = {
        'torch.float': torch.float,
        'torch.float32': torch.float,
        'torch.float64': torch.float64,
        'torch.double': torch.float64,
        'torch.complex64': torch.complex,
        'torch.cfloat': torch.complex,
        'torch.float16': torch.float16,
        'torch.half': torch.float16,
        'torch.bfloat16': torch.bfloat16,
        'torch.uint8': torch.uint8,
        'torch.int8': torch.int8,
        'torch.int16': torch.int16,
        'torch.short': torch.short,
        'torch.int32': torch.int32,
        'torch.int': torch.int,
        'torch.int64': torch.int64,
        'torch.long': torch.long,
        'torch.bool': torch.bool,
    }
    return lut[val.lower()]


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
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
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
            dropout=dropout,
            dropout_position=dropout_position,
            lora_A_init_method=lora_A_init_method,
            lora_dtype=lora_dtype,
        )

    @torch.no_grad
    @staticmethod
    def _init_adapter(
        obj,
        dim=8,
        alpha=32,
        dropout=0.0,
        dropout_position='post',
        lora_A_init_method='xavier',
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
        if lora_A_init_method == 'xavier':
            torch.nn.init.uniform_(obj.lora_A.weight.data)
        else:
            nn.init.kaiming_uniform_(obj.lora_A.weight.data, a=math.sqrt(5))
        obj.lora_B.weight.data.fill_(0)
        obj.dropout = nn.Dropout(p=dropout)
        assert dropout_position in ['pre', 'post'], ('dropout position can only be pre/post', dropout_position)
        obj.dropout_position = dropout_position

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
        if (fwd := getattr(self, 'super_fwd', None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if self.dropout_position == 'pre':
            x = self.dropout(x)
        lora_res = self.lora_B(self.lora_A(x))
        lora_res = lora_res * self.scale
        if self.dropout_position == 'post':
            lora_res = self.dropout(lora_res)
        return res + lora_res


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
        if (fwd := getattr(self, 'super_fwd', None)) is not None:
            assert fwd != self.forward
            res = fwd(x)
        else:
            res = F.linear(x, self.weight, self.bias)

        if self.dropout_position == 'pre':
            x = self.dropout(x)
        lora_res = LoRATritonFunction.apply(x, self.lora_A.weight, self.lora_B.weight, self.scale, x.dtype)
        if self.dropout_position == "post":
            lora_res = self.dropout(lora_res)

        return res + lora_res


def patch_linear_module(
    orig_linear,
    dim=8,
    alpha=32,
    dropout=0.0,
    dropout_position='post',
    lora_A_init_method='xavier',
    lora_dtype=None,
    use_triton=True
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
    assert isinstance(orig_linear, nn.Linear), type(orig_linear)
    assert not hasattr(orig_linear, 'super_fwd'), orig_linear.super_fwd

    if isinstance(orig_linear, nn.Linear):
        linear_lora_cls = TritonLinearLoRA if use_triton else LinearLoRA
        linear_lora_cls._init_adapter(
            orig_linear, dim, alpha, dropout, dropout_position, lora_A_init_method,
            lora_dtype)
        cls = orig_linear.__class__
        new_cls = type('PatchedLinearLoRA', (linear_lora_cls, cls), {})
    else:
        raise NotImplementedError("Expected isinstance(orig_linear, nn.Linear)")

    # If the model uses quantized weights, we want to use orig_linear's forward
    if (
        getattr(orig_linear, 'quant_state', None) is not None
        and orig_linear.quant_state.__class__ == bitsandbytes.functional.QuantState
    ):
        orig_linear.super_fwd = orig_linear.forward

    orig_linear.__class__ = new_cls
    return orig_linear


# -----------------------------------------------------------------------------#
# 2.  Convenience: patch a model in-place                                      #
# -----------------------------------------------------------------------------#
def apply_lora_to_linear_modules(
    model: nn.Module,
    target_modules = [],
    exclude_modules = [],
    match_all_linear = False,
    dim: int = 8,
    alpha: int = 32,
    dropout: float = 0.0,
    dropout_position: Literal["pre", "post"] = "post",
    lora_A_init: str = "xavier",
    lora_dtype: Optional[torch.dtype] = None,
    use_triton: bool = False
):
    """
    Replace selected nn.Linear layers with LinearLoRA layers (in-place).

    target_modules accepts wildcard fragments, e.g. ["q_proj", "k_proj", ".*fc.*"].
    """
    # To make our PeftConfig compatible with HF, we need to keep track of the
    # final target modules, without the wildcard fragments.
    final_target_modules = set()

    # Freeze base model parameters
    for w in model.parameters():
        w.requires_grad_(False)

    is_causal_lm = False
    try:
        if hasattr(model, "config") and "CausalLM" in model.config.architectures[0]:
            # for example, LlamaForCausalLM
            is_causal_lm = True
    except AttributeError:
        is_causal_lm = False

    matcher = ModuleMatcher(target_modules, exclude_modules, match_all_linear, is_causal_lm)
    num_modules_matched = 0
    for name, module in list(model.named_modules()):
        if matcher.match(module, name):
            final_target_modules.add(name.split(".")[-1])

            num_modules_matched += 1
            patch_linear_module(
                module,
                dim=dim,
                alpha=alpha,
                dropout=dropout,
                dropout_position=dropout_position,
                lora_A_init_method=lora_A_init,
                lora_dtype=lora_dtype,
                use_triton=use_triton
           )

    # finalize the peft config
    try:
        model_task = model.config.architectures[0].split("For")[-1]
    except AttributeError:
        model_task = "N/A"
    try:
        name_or_path = model.config.name_or_path
        task_type = MODEL_TYPE_TO_PEFT_TASK_TYPE[model_task]
    except AttributeError:
        name_or_path = "N/A"
        task_type = "CAUSAL_LM"

    model._automodel_peft_config = {
        "task_type": task_type,
        "peft_type": "LORA",
        "r": dim,
        "lora_alpha": alpha,
        "target_modules": list(final_target_modules),
        "bias": "none",
        "base_model_name_or_path": name_or_path,
        "lora_dropout": dropout,
    }

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
        return d_x, d_lora_A.t(), d_lora_B, None, None, None