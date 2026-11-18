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

"""Shared model traversal and LM-head projection utilities."""

import abc
import math
import sys
from collections.abc import Iterator

import torch
from torch import nn
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from transformers.modeling_outputs import CausalLMOutputWithPast

from nemo_automodel.shared.import_utils import safe_import_from

_TEXT_MODULE_ATTRS = ("language_model", "text_model", "text_decoder")


def iter_transformer_and_mtp_blocks(model: nn.Module) -> Iterator[tuple[nn.Module, str, nn.Module]]:
    """Yield transformer and MTP blocks without depending on a recipe or component.

    Args:
        model: Model root containing a transformer layer collection and optional
            multi-token-prediction layers.

    Yields:
        Tuples containing the parent layer collection, child name, and block.
    """
    inner_model = getattr(model, "model", None)
    inner = inner_model if inner_model is not None else model
    text_model = inner
    for attribute in _TEXT_MODULE_ATTRS:
        candidate = getattr(inner, attribute, None)
        if candidate is not None:
            text_model = candidate
            break

    layers = getattr(text_model, "layers", None)
    if layers is not None:
        for layer_id, block in layers.named_children():
            yield layers, layer_id, block

    mtp_layers = getattr(getattr(model, "mtp", None), "layers", None)
    if mtp_layers is not None:
        for layer_id, block in mtp_layers.named_children():
            yield mtp_layers, layer_id, block


def compute_lm_head_logits(
    lm_head: nn.Module | torch.Tensor | None,
    hidden_states: torch.Tensor,
    logits_to_keep: int | torch.Tensor = 0,
    is_thd: bool = False,
    output_hidden_states: bool = False,
    *,
    bias: torch.Tensor | None = None,
) -> CausalLMOutputWithPast:
    """Project hidden states through ``lm_head`` and wrap the result.

    Centralizes the lm_head projection and output packaging shared by every
    custom ``*ForCausalLM`` / ``*ForConditionalGeneration`` ``forward()``. The
    returned ``CausalLMOutputWithPast`` carries the projected ``logits`` and,
    when requested, the final ``hidden_states``; callers that also need ``loss``,
    ``past_key_values``, etc. read ``.logits`` and build their own output.

    - ``lm_head is None`` (e.g. a non-final pipeline-parallel stage that does not
      own the head): ``hidden_states`` is passed through as ``logits`` so the
      next stage receives it.
    - ``logits_to_keep == 0`` (training default): every position is projected.
      The full range is deliberately *not* sliced, because ``slice(0, None)`` on
      a DTensor is unsupported (it raises on the ``aten.alias`` op under tensor
      parallel with sequence parallelism).
    - ``logits_to_keep`` as a positive int or a tensor of indices: only the
      requested positions are projected. Both 2D ``[T, H]`` (THD/packed) and 3D
      ``[B, S, H]`` (BSHD) hidden states are handled.
    - ``is_thd``: THD/packed inputs yield 2D ``[T, V]`` logits; the leading batch
      dim is restored (``unsqueeze(0)`` -> ``[1, T, V]``) so downstream code sees
      a uniform ``[B, S, V]`` layout. The same restoration is applied to the
      ``hidden_states`` field. Only applied while the tensor is still 2D, so an
      ``inputs_embeds`` path that already produced ``[1, T, *]`` is left
      untouched.
    - The sliced input is cast to the head's compute dtype (``head_compute_dtype``):
      a standalone FSDP2 head's mixed-precision ``param_dtype``, else its weight
      dtype. The projection goes through the ``lm_head`` module (DTensor-aware
      under FSDP2) rather than ``F.linear`` so its FSDP hooks and DTensor
      redistribution run, and its output is returned uncast: a
      ``Float32LogitsLinear`` head (``logits_dtype=float32``, or a head that is
      FP32 by design) writes FP32 logits, any other head its compute dtype.
    - A bare weight tensor is projected in its compute dtype with FP32 logits.
      This path is used by callers that already own or have materialized the weight.
    - ``output_hidden_states``: when set, the (full-sequence, THD-restored)
      ``hidden_states`` are attached to the output so the fused cross-entropy
      path can recompute logits over every position; otherwise the field is
      ``None``.

    Args:
        lm_head: The language-model head module, or ``None`` on a pipeline stage
            that does not own it.
        hidden_states: Final hidden states, shaped ``[T, H]`` or ``[B, S, H]``.
        logits_to_keep: ``0`` to project every position; a positive int to keep
            the last ``N`` positions; or a tensor of position indices.
        is_thd: Whether the inputs were THD/packed; if so, a 2D logits (and
            hidden-states) result is unsqueezed back to a leading batch dim of 1.
        output_hidden_states: Attach the final hidden states to the output.
        bias: Optional bias [vocab] when passing a local weight tensor.

    Returns:
        A ``CausalLMOutputWithPast`` whose ``logits`` are the projected logits
        (or ``hidden_states`` unchanged when ``lm_head`` is ``None``) and whose
        ``hidden_states`` are the final hidden states when ``output_hidden_states``
        is set, else ``None``.
    """
    if lm_head is None:
        logits = hidden_states
    else:
        if isinstance(logits_to_keep, int) and logits_to_keep == 0:
            sliced = hidden_states
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            if hidden_states.dim() == 2:
                sliced = hidden_states[slice_indices, :]
            else:
                sliced = hidden_states[:, slice_indices, :]
        if isinstance(lm_head, torch.Tensor):
            logits = _linear_with_fp32_logits(sliced, lm_head, bias)
        else:
            logits = lm_head(sliced.to(head_compute_dtype(lm_head)))
    if is_thd and logits.dim() == 2:
        logits = logits.unsqueeze(0)

    hidden_out = None
    if output_hidden_states:
        hidden_out = hidden_states.unsqueeze(0) if is_thd and hidden_states.dim() == 2 else hidden_states

    return CausalLMOutputWithPast(logits=logits, hidden_states=hidden_out)


def _fp32_output_mm(input_2d: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Multiply local hidden states by the transposed head weight into FP32 logits.

    Args:
        input_2d: Tensor of shape [tokens, hidden] in ``weight.dtype``.
        weight: Tensor of shape [vocab, hidden] in BF16, FP16 or FP32.

    Returns:
        New FP32 tensor of shape [tokens, vocab].
    """
    if weight.dtype == torch.float32:
        return input_2d @ weight.t()
    if input_2d.is_cuda:
        return torch.mm(input_2d, weight.t(), out_dtype=torch.float32)
    # aten::mm.dtype has no CPU kernel. Products of two BF16/FP16 values are
    # exact in FP32, so upcasting the operands changes only the summation order.
    return input_2d.float() @ weight.float().t()


class _Float32LogitsMatmul(torch.autograd.Function):
    """``input @ weight.T`` with operands in ``weight.dtype`` and an FP32 result."""

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Project local hidden states into FP32 logits.

        Args:
            ctx: Autograd context; saves ``input`` and ``weight`` for backward.
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions,
                in ``weight.dtype``.
            weight: Tensor of shape [vocab, hidden].

        Returns:
            New FP32 tensor of shape [..., vocab]; neither input is aliased or mutated.
        """
        ctx.save_for_backward(input, weight)
        logits = _fp32_output_mm(input.reshape(-1, input.shape[-1]), weight)
        return logits.view(*input.shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_logits: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Cast the logits gradient to the operand dtype, then run the head's usual GEMMs.

        Args:
            ctx: Autograd context holding ``input`` of shape [..., hidden] and
                ``weight`` of shape [vocab, hidden].
            grad_logits: FP32 tensor of shape [..., vocab], with the forward's leading dimensions.

        Returns:
            Gradient of ``input`` (shape [..., hidden]) and of ``weight`` (shape
            [vocab, hidden]) in the operand dtype, each ``None`` when not required.
        """
        input, weight = ctx.saved_tensors
        grad = grad_logits.reshape(-1, grad_logits.shape[-1]).to(weight.dtype)
        grad_input = grad_weight = None
        with torch.autocast(device_type=weight.device.type, enabled=False):
            if ctx.needs_input_grad[0]:
                grad_input = (grad @ weight).view(input.shape)
            if ctx.needs_input_grad[1]:
                grad_weight = grad.t() @ input.reshape(-1, input.shape[-1])
        return grad_input, grad_weight


def _vocab_parallel_fp32_logits(input: torch.Tensor, weight: DTensor) -> DTensor:
    """Project through a column-parallel head whose weight is sharded over the vocabulary.

    Args:
        input: Tensor of shape [..., hidden], with arbitrary leading dimensions.
            A DTensor on ``weight``'s mesh with any placement is redistributed to
            ``Replicate``; a regular tensor is taken as the replicated value.
        weight: DTensor of global shape [vocab, hidden] with placements
            ``(Shard(0),)`` on a 1-D tensor-parallel mesh; each rank holds a
            [local_vocab, hidden] slice.

    Returns:
        FP32 DTensor of global shape [..., vocab] with placement ``Shard(-1)``;
        each rank's local tensor is [..., local_vocab].
    """
    mesh = weight.device_mesh
    if mesh.ndim != 1 or weight.placements != (Shard(0),):
        raise NotImplementedError(
            "FP32 logits support a vocabulary-sharded (column-parallel) LM head, whose weight has placements "
            f"(Shard(0),) on a 1-D tensor-parallel mesh; got {weight.placements} on a {mesh.ndim}-D mesh."
        )
    if not isinstance(input, DTensor):
        input = DTensor.from_local(input, mesh, [Replicate()], run_check=False)
    input = input.redistribute(mesh, [Replicate()])
    # Every rank holds a vocabulary slice of the weight, so its input gradient is a
    # partial sum; ``Partial`` makes DTensor autograd reduce it, as it does for aten.mm.
    local_logits = _Float32LogitsMatmul.apply(
        input.to_local(grad_placements=[Partial()]).to(weight.dtype),
        weight.to_local(grad_placements=[Shard(0)]),
    )
    # Pass the global shape so a vocabulary that the mesh does not divide evenly is kept exact.
    shape = (*input.shape[:-1], weight.shape[0])
    stride = tuple(math.prod(shape[dim + 1 :]) for dim in range(len(shape)))
    return DTensor.from_local(
        local_logits, mesh, [Shard(local_logits.ndim - 1)], run_check=False, shape=shape, stride=stride
    )


def _linear_with_fp32_logits(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Project hidden states through an LM-head weight into FP32 logits.

    The GEMM reads ``input`` cast to ``weight.dtype`` and ``weight`` unchanged,
    accumulates in FP32 and stores FP32 logits; a bias is added in FP32. Autocast
    is disabled inside so it cannot lower the output precision.

    Args:
        input: Tensor of shape [..., hidden], with arbitrary leading dimensions, of
            any floating dtype. With a DTensor ``weight`` it may be a DTensor on the
            same mesh with any placement.
        weight: Tensor of shape [vocab, hidden] in BF16, FP16 or FP32. A DTensor must
            have global shape [vocab, hidden] and placements ``(Shard(0),)`` on a 1-D
            tensor-parallel mesh.
        bias: Optional tensor of shape [vocab] in any floating dtype; not supported
            with a DTensor ``weight``.

    Returns:
        New FP32 tensor of shape [..., vocab]. For a DTensor ``weight`` it is a
        DTensor of that global shape with placement ``Shard(-1)``. No input is
        mutated.
    """
    with torch.autocast(device_type=input.device.type, enabled=False):
        if isinstance(weight, DTensor):
            if bias is not None:
                raise NotImplementedError("FP32 logits with tensor parallelism need a bias-free LM head.")
            return _vocab_parallel_fp32_logits(input, weight)
        logits = _Float32LogitsMatmul.apply(input.to(weight.dtype), weight)
        return logits if bias is None else logits + bias.to(torch.float32)


def _linear_backend_classes() -> tuple[type[nn.Module], ...]:
    """Return the linear classes that ``initialize_linear_module`` builds.

    Each projects with its ``weight`` and ``bias`` exactly as ``nn.Linear`` does, so a
    head of one of these classes can hand its parameters to a :class:`Float32LogitsLinear`.
    A head of a kernel backend's class implies that its module is already imported,
    so these imports never load a backend that the model does not use.

    Returns:
        ``nn.Linear`` and the imported Transformer Engine and quack ``Linear`` classes.
    """
    classes: list[type[nn.Module]] = [nn.Linear]
    for module_name in ("transformer_engine.pytorch.module.linear", "quack.linear"):
        if module_name in sys.modules:
            available, linear = safe_import_from(module_name, "Linear")
            if available:
                classes.append(linear)
    return tuple(classes)


class Float32LogitsConvertible(abc.ABC):
    """Mixin for a model-owned LM head whose projection is not a plain linear map.

    :meth:`Float32LogitsLinear.from_head` asks such a head for its FP32-logits
    variant instead of rejecting it, so the model package keeps owning the projection.
    """

    @abc.abstractmethod
    def to_fp32_logits(self) -> "Float32LogitsLinear":
        """Return a head that owns this head's parameters and whose projection writes FP32 logits.

        Returns:
            A :class:`Float32LogitsLinear` subclass instance sharing this head's Parameters.
        """


class Float32LogitsLinear(nn.Linear):
    """LM head whose GEMM writes FP32 logits from operands in its weight's compute dtype.

    Models whose head is FP32 by design construct it directly; :meth:`from_head`
    wraps another head's parameters for ``logits_dtype: float32``. Only the
    projection's output dtype differs from ``nn.Linear``.
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Project hidden states into FP32 logits.

        Args:
            input: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            FP32 tensor of shape [..., vocab], sharded over the vocabulary under TP.
        """
        return compute_lm_head_logits(self.weight, input, bias=self.bias).logits

    @classmethod
    def from_head(cls, head: nn.Module) -> "Float32LogitsLinear":
        """Return an FP32-logits head that owns ``head``'s parameters (no copy, identity preserved).

        An FP32-logits head is returned unchanged; a :class:`Float32LogitsConvertible`
        head converts itself; a head of a linear kernel backend (``nn.Linear``,
        Transformer Engine or quack ``Linear``) becomes a :class:`Float32LogitsLinear`
        owning the same ``weight`` and ``bias`` Parameters, so parameter names, tied
        embeddings, optimizer state, checkpoints and rollout weight sync are
        unchanged. A Transformer Engine head under FP8 autocast then projects in its
        compute dtype instead.

        Args:
            head: LM head whose weight has shape [vocab, hidden] and optional bias shape [vocab].

        Returns:
            A head whose projection writes FP32 logits.

        Raises:
            ValueError: If ``head`` owns a projection that cannot write FP32 logits,
                such as LoRA, QAT or quantized linears.
        """
        if isinstance(head, Float32LogitsLinear):
            return head
        if isinstance(head, Float32LogitsConvertible):
            return head.to_fp32_logits()
        if type(head) not in _linear_backend_classes():
            raise ValueError(
                "FP32 logits require an LM head built from a linear kernel backend (torch, te or quack) or one "
                f"that converts itself (Float32LogitsConvertible); got {type(head).__name__}. LoRA, QAT and "
                "quantized linears own their projection; keep them off the LM head."
            )
        fp32_head = cls(head.in_features, head.out_features, bias=False, device="meta")
        fp32_head.weight = head.weight
        # Transformer Engine keeps an empty, unregistered tensor for a disabled bias.
        if isinstance(head.bias, nn.Parameter):
            fp32_head.bias = head.bias
        fp32_head.train(head.training)
        return fp32_head


def head_compute_dtype(head: nn.Module) -> torch.dtype | None:
    """Return the dtype in which ``head``'s GEMM reads its weight.

    A head that is its own FSDP unit computes in its mixed-precision policy's
    ``param_dtype``; its sharded weight may still expose the storage dtype before
    the pre-forward hook runs. Any other head computes in its weight's dtype.

    Args:
        head: LM-head module.

    Returns:
        The compute dtype, or None when the head has no weight tensor.
    """
    if isinstance(head, FSDPModule):
        param_dtype = head._get_fsdp_state()._mp_policy.param_dtype
        if param_dtype is not None:
            return param_dtype
    weight = getattr(head, "weight", None)
    return weight.dtype if isinstance(weight, torch.Tensor) else None
