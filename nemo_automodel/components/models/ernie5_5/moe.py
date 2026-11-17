# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""ERNIE 5.5 latent mixture-of-experts: router, gate and MoE block.

* ``route``: the dual-projection sigmoid router. It returns the selected routing weights and expert IDs for local
  token states. It is a pure function of its inputs: it does no load tracking, no correction-bias update and no
  communication between ranks.
* ``Ernie5_5Gate``: ``route`` behind AutoModel's MoE ``Gate`` interface. It owns the two FP32 gate matrices, the
  learned per-expert scale and the selection-only correction bias, and records expert loads for the bias update.
* ``Ernie5_5SharedExpert``: the shared expert with the SiTU activation.
* ``Ernie5_5MoE``: the MoE block: FP32 latent projections around the routed experts, a latent RMSNorm and the shared
  expert.

Supported execution: the torch linear backend with one of two expert and dispatcher pairs.

* ``experts="torch"`` with ``dispatcher="torch"``: the routed experts run in ``GroupedExperts``, which gathers the
  tokens of the expert-parallel group, loops over the local experts and sums the selected expert outputs of a
  token in FP32.
* ``experts="torch_mm"`` with ``dispatcher="hybridep"``: with more than one process the routed experts run in
  ``GroupedExpertsDeepEP``, which exchanges the tokens through HybridEP and runs two grouped GEMMs. The HybridEP
  combine also sums the selected expert outputs of a token in FP32, but it rounds the partial sum of every
  expert-parallel rank to the activation dtype before it adds the partial sums. The result differs from the
  first pair by BF16 rounding and depends on ``ep_size``. Only BF16 activations are supported there. In a single
  process the shared ``MoE`` builds ``GroupedExperts`` on its grouped-GEMM path.

The two pairs store the expert weights in the same parameters with the same placements. In a single process there
is no placement requirement. With more than one process the expert weights must be ``DTensor`` objects on an
``("ep",)`` mesh with placements ``(Shard(0),)`` or on an ``("ep_shard", "ep")`` mesh with placements
``(Shard(1), Shard(0))``; ``Ernie5_5MoE.forward`` raises ``RuntimeError`` for any other layout and for context
parallelism.

Provenance: the expert-selection and routing-weight contracts of ``route`` are those of the ERNIE 5.5 implementation
in Megatron-Bridge (Apache License 2.0) at revision ``a38833622b88969564fe8bfa053c5e4e4989c6cd``:
https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/a38833622b88969564fe8bfa053c5e4e4989c6cd/src/megatron/bridge/models/ernie/modeling_ernie55/moe.py
"""

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from nemo_automodel.components.distributed.init_utils import get_world_size_safe
from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.ernie5_5.layers import ernie_situ
from nemo_automodel.components.moe.config import MoEConfig
from nemo_automodel.components.moe.experts import GroupedExperts, GroupedExpertsDeepEP
from nemo_automodel.components.moe.layers import MLP, Gate, MoE

# The supported pairs of ``BackendConfig.experts`` and ``BackendConfig.dispatcher``.
_EXPERT_BACKENDS = (("torch", "torch"), ("torch_mm", "hybridep"))


def route(
    hidden: torch.Tensor,
    gate_a: torch.Tensor,
    gate_b: torch.Tensor,
    correction: torch.Tensor,
    expert_scale: torch.Tensor,
    top_k: int,
    *,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return selected routing weights and expert IDs.

    Args:
        hidden: Local token states of shape [tokens, hidden].
        gate_a: First FP32 gate matrix of shape [experts, hidden].
        gate_b: Second FP32 gate matrix of shape [experts, hidden].
        correction: FP32 selection-only bias of shape [experts].
        expert_scale: Learnable FP32 per-expert scale of shape [experts].
        top_k: Number of distinct experts selected per token.
        normalize: Normalize selected unbiased scores when top_k exceeds one.

    Returns:
        FP32 weights and integer expert IDs, each of shape [tokens, top_k].
        The result is newly allocated and does not mutate inputs.

    Raises:
        ValueError: If the tensor shapes do not agree, ``top_k`` is not between 1 and the number of experts, or a
            gate matrix, ``correction`` or ``expert_scale`` is not FP32.
    """
    if hidden.ndim != 2 or gate_a.ndim != 2 or gate_b.shape != gate_a.shape:
        raise ValueError("hidden and both equal-shaped gates must be 2-D")
    experts, hidden_size = gate_a.shape
    if hidden.shape[1] != hidden_size or correction.shape != (experts,) or expert_scale.shape != (experts,):
        raise ValueError("router dimensions do not agree")
    if not 1 <= top_k <= experts:
        raise ValueError("top_k must lie between 1 and the number of experts")
    if any(t.dtype != torch.float32 for t in (gate_a, gate_b, correction, expert_scale)):
        raise ValueError("router gates, correction and expert scales must be FP32")

    with torch.autocast(device_type=hidden.device.type, enabled=False):
        inputs = hidden.float()
        scores = torch.sigmoid(F.linear(inputs, gate_a)) + torch.sigmoid(F.linear(inputs, gate_b))
        indices = torch.topk(scores + correction, top_k, dim=-1).indices
        weights = scores.gather(-1, indices)
        if normalize and top_k > 1:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        weights = weights * expert_scale[indices]
    return weights, indices


class Ernie5_5Gate(Gate):
    """ERNIE 5.5 gate using AutoModel's MoE gate interface.

    The gate supports one expert group, sigmoid scores with a selection-only correction bias and FP32 routing
    weights. The auxiliary loss, a gate bias and routing replay are not supported; the constructor raises
    ``ValueError`` for such a configuration. Routing uses the local token states of the calling process. Expert
    loads for the correction-bias update are accumulated only in training mode with gradients enabled, so a
    no-grad forward followed by its recomputation with gradients is counted once.

    The parameters ``weight`` and ``weight_1`` of shape [experts, hidden] and ``routed_scaling_factor_param`` of
    shape [experts] are FP32, and ``forward`` raises if any of them has another dtype; preserving FP32 under FSDP
    mixed precision is the responsibility of the model that owns the gate.
    """

    def __init__(self, config: MoEConfig) -> None:
        """Validate the configuration and create the second gate matrix and the per-expert scales.

        Args:
            config: MoE configuration. It must select FP32 gate storage, one expert group, sigmoid scores with a
                selection-only correction bias and FP32 routing weights, and must create the correction-bias
                buffer.

        Raises:
            ValueError: If ``config`` does not meet these requirements or enables the auxiliary loss, a gate bias
                or routing replay.
        """
        if config.gate_dtype != torch.float32:
            raise ValueError("ERNIE 5.5 requires FP32 gate storage")
        if config.n_expert_groups != 1 or config.n_limited_groups != 1:
            raise ValueError("ERNIE 5.5 currently supports one expert group")
        if config.score_func != "sigmoid" or config.router_weight_uses_score_correction_bias:
            raise ValueError("ERNIE 5.5 requires sigmoid scores and selection-only correction bias")
        if not config.router_weights_fp32:
            raise ValueError("ERNIE 5.5 requires FP32 routing weights")
        if config.aux_loss_coeff != 0 or config.router_bias or config.enable_routing_replay:
            raise ValueError("Auxiliary loss, gate bias and routing replay are not supported")
        if not (config.force_e_score_correction_bias or config.gate_bias_update_factor > 0):
            raise ValueError("ERNIE 5.5 requires the selection correction-bias buffer")
        super().__init__(config, gate_precision=torch.float32)
        self.weight_1 = torch.nn.Parameter(torch.empty_like(self.weight), requires_grad=config.train_gate)
        self.routed_scaling_factor_param = torch.nn.Parameter(
            torch.full((self.n_experts,), config.route_scale, dtype=torch.float32, device=self.weight.device),
            requires_grad=config.train_gate,
        )
        torch.nn.init.normal_(self.weight, std=0.02)
        torch.nn.init.normal_(self.weight_1, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        token_mask: torch.Tensor,
        cp_mesh: DeviceMesh | None,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        """Route local token states and record valid-token loads.

        Args:
            x: Local token states of shape [tokens, hidden].
            token_mask: Boolean valid-token mask of shape [tokens].
            cp_mesh: Context mesh; auxiliary loss is unsupported and it is unused.

        Returns:
            Tuple of FP32 routing weights [tokens, topk], integer expert IDs
            [tokens, topk], and None for the unsupported auxiliary loss.

        Raises:
            ValueError: If ``token_mask`` does not have one entry per token, the correction bias is not on the
                device of ``x``, or ``route`` rejects the shapes or dtypes.
            RuntimeError: If no correction bias is available.
        """
        del cp_mesh
        if token_mask.shape != (x.shape[0],):
            raise ValueError("token_mask must have one entry per token")
        correction = self._local_score_correction_bias()
        if correction is None:
            raise RuntimeError("ERNIE 5.5 correction bias is missing")
        if correction.device != x.device:
            raise ValueError("Correction bias must be on the input device")
        weights, indices = route(
            x,
            self.weight,
            self.weight_1,
            correction,
            self.routed_scaling_factor_param,
            self.topk,
            normalize=self.norm_topk_prob,
        )
        # Loads are accumulated for the bias update only in training with gradients enabled, so that a forward
        # under no_grad is not counted.
        accumulate = self.bias_update_factor > 0 and self.training and torch.is_grad_enabled()
        if self._track_load_balance or accumulate:
            expert_load = self._compute_expert_load(indices, token_mask).detach()
            if self._track_load_balance:
                self._last_expert_load = expert_load
            if accumulate:
                if self._cumulative_expert_load is None:
                    self._cumulative_expert_load = expert_load
                else:
                    self._cumulative_expert_load = self._cumulative_expert_load + expert_load
        return weights, indices, None

    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize both gates and learned expert scales after meta materialization.

        Args:
            buffer_device: Device passed to the parent initializer.
            init_std: Standard deviation of the normal initialization of both gate matrices.
        """
        super().init_weights(buffer_device, init_std)
        torch.nn.init.normal_(self.weight_1, mean=0.0, std=init_std)
        torch.nn.init.constant_(self.routed_scaling_factor_param, self.route_scale)


def _routed_situ(x: torch.Tensor, weights: torch.Tensor, *, beta: float, linear_beta: float | None) -> torch.Tensor:
    """Apply SiTU and the routing weight to the packed gate/up output of one routed expert.

    Args:
        x: Tensor of shape [routed_tokens, 2 * intermediate], with the gate followed by the up projection on the
            final axis. ``routed_tokens`` is the number of (token, top-k slot) pairs routed to the expert.
        weights: Routing weights of shape [routed_tokens, 1].
        beta: Gate-clamp parameter of SiTU.
        linear_beta: Optional up-projection clamp parameter of SiTU.

    Returns:
        Tensor of shape [routed_tokens, intermediate] in the dtype of ``x``.
    """
    return ernie_situ(x, beta=beta, linear_beta=linear_beta, routing_weight=weights)


class Ernie5_5SharedExpert(MLP):
    """Preserve ERNIE shared-expert arithmetic at a callable FSDP boundary."""

    def __init__(self, old: MLP, *, beta: float, linear_beta: float | None) -> None:
        """Wrap an existing shared-expert MLP.

        Args:
            old: Gated MLP with torch linear projections. Its ``gate_proj``, ``up_proj`` and ``down_proj`` modules
                are reused, so the parameters are shared with ``old`` and not copied.
            beta: Gate-clamp parameter of SiTU.
            linear_beta: Optional up-projection clamp parameter of SiTU.

        Raises:
            TypeError: If the gate or up projection of ``old`` is not an ``nn.Linear``.
        """
        if not isinstance(old.gate_proj, nn.Linear) or not isinstance(old.up_proj, nn.Linear):
            raise TypeError("ERNIE shared experts require torch gated linear projections")
        with torch.device("meta"):
            super().__init__(
                old.gate_proj.in_features,
                old.gate_proj.out_features,
                "torch",
                dtype=old.gate_proj.weight.dtype,
                activation=old.activation,
                bias=False,
                swiglu_limit=old.swiglu_limit,
            )
        self.gate_proj, self.up_proj, self.down_proj = old.gate_proj, old.up_proj, old.down_proj
        self.beta, self.linear_beta = beta, linear_beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SiTU and the output projection inside the FSDP-wrapped module.

        Args:
            x: Tensor of shape [..., hidden], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., hidden].
        """
        inputs = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(ernie_situ(inputs, beta=self.beta, linear_beta=self.linear_beta))


class _FP32LatentLinear(nn.Linear):
    """Project in the activation dtype while preserving an FP32 master weight."""

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """Project [..., in_features] to [..., out_features].

        Args:
            hidden: Tensor of shape [..., in_features], with arbitrary leading dimensions.

        Returns:
            Tensor of shape [..., out_features] in the input dtype; the FP32 weight is converted to that dtype
            for the projection.

        Raises:
            RuntimeError: If the weight is not FP32.
        """
        if self.weight.dtype != torch.float32:
            raise RuntimeError("ERNIE latent projection parameters must remain FP32")
        return F.linear(hidden, self.weight.to(hidden.dtype))


def _make_callable_projection(projection: nn.Linear) -> _FP32LatentLinear:
    """Preserve the initialized parameter while adding a callable FSDP boundary.

    Args:
        projection: Bias-free torch linear module.

    Returns:
        Projection module whose ``weight`` is the same parameter object as ``projection.weight``.

    Raises:
        TypeError: If ``projection`` is not an ``nn.Linear`` or has a bias.
    """
    if not isinstance(projection, nn.Linear) or projection.bias is not None:
        raise TypeError("ERNIE latent projections must be bias-free torch linear modules")
    result = _FP32LatentLinear(projection.in_features, projection.out_features, bias=False, device="meta")
    result.weight = projection.weight
    return result


class Ernie5_5MoE(MoE):
    """ERNIE dual-router latent experts and shared SiTU expert.

    The routed experts are the shared ``GroupedExperts`` (torch dispatcher, or one process) or the shared
    ``GroupedExpertsDeepEP`` (HybridEP dispatcher in a job of several processes), with SiTU installed as their
    activation. Distributed execution is restricted to the expert-parallel layouts that ``forward`` checks, and
    with the HybridEP dispatcher to BF16 activations. Context parallelism is not supported.
    """

    def __init__(
        self,
        config: MoEConfig,
        backend: BackendConfig,
        *,
        beta: float = 4.0,
        linear_beta: float | None = 25.0,
        norm_eps: float = 1e-6,
    ) -> None:
        """Build the latent MoE on the torch linear backend with a supported expert and dispatcher pair.

        Args:
            config: MoE configuration with a latent width, exactly one shared expert, routing weights applied
                before the down projection, and bias-free, ungated experts.
            backend: Backend configuration. ``experts`` and ``dispatcher`` must be ``"torch"`` and ``"torch"``, or
                ``"torch_mm"`` and ``"hybridep"``; ``linear`` must be ``"torch"``, with the fake balanced gate and
                the shared-expert overlap disabled.
            beta: Gate-clamp parameter of SiTU.
            linear_beta: Optional up-projection clamp parameter of SiTU.
            norm_eps: Positive epsilon of the latent RMSNorm.

        Raises:
            ValueError: If ``config``, ``backend`` or ``norm_eps`` does not meet these requirements.
            TypeError: If the parent class built neither ``GroupedExperts`` nor ``GroupedExpertsDeepEP``, or no
                torch linear projections.
        """
        if (backend.experts, backend.dispatcher) not in _EXPERT_BACKENDS:
            raise ValueError(
                "ERNIE 5.5 MoE supports experts='torch' with dispatcher='torch' and experts='torch_mm' with "
                f"dispatcher='hybridep'; got experts={backend.experts!r} with dispatcher={backend.dispatcher!r}"
            )
        if backend.linear != "torch" or backend.fake_balanced_gate or backend.shared_expert_overlap:
            raise ValueError(
                "ERNIE 5.5 MoE requires the torch linear backend, no fake balanced gate and no shared-expert overlap"
            )
        if config.moe_latent_size is None or config.n_shared_experts != 1:
            raise ValueError("ERNIE MoE requires a latent width and exactly one shared expert")
        if config.apply_router_weight_after_down or config.expert_bias or config.shared_expert_gate:
            raise ValueError("ERNIE MoE requires pre-down routing and bias-free, ungated experts")
        if norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        super().__init__(config, backend)
        self.shared_experts = Ernie5_5SharedExpert(self.shared_experts, beta=beta, linear_beta=linear_beta)
        # The shared constructor builds GroupedExpertsDeepEP for the HybridEP dispatcher in a job of several
        # processes and GroupedExperts otherwise. The two classes take the same activation callable under
        # different attribute names and share no method to set it, so the class is looked at here, once;
        # ``forward`` reads the recorded result.
        activation = partial(_routed_situ, beta=beta, linear_beta=linear_beta)
        self._dispatches_through_hybridep = isinstance(self.experts, GroupedExpertsDeepEP)
        if self._dispatches_through_hybridep:
            self.experts.expert_activation = activation
        elif isinstance(self.experts, GroupedExperts):
            self.experts.expert_activation_grouped = activation
        else:
            raise TypeError("Expected unsharded AutoModel GroupedExperts or GroupedExpertsDeepEP")
        self.gate = Ernie5_5Gate(config)
        self.fc1_latent_proj.to(dtype=torch.float32)
        self.fc2_latent_proj.to(dtype=torch.float32)
        self.fc1_latent_proj = _make_callable_projection(self.fc1_latent_proj)
        self.fc2_latent_proj = _make_callable_projection(self.fc2_latent_proj)
        self.latent_norm = nn.RMSNorm(config.moe_latent_size, eps=norm_eps, dtype=config.dtype)
        self.beta, self.linear_beta = beta, linear_beta

    def forward(
        self,
        hidden: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        cp_mesh: DeviceMesh | None = None,
    ) -> torch.Tensor:
        """Run the MoE.

        Args:
            hidden: Tensor of shape [..., hidden] with arbitrary leading dimensions.
            padding_mask: Boolean tensor of shape [...] where true marks padding.
            cp_mesh: Context-parallel mesh; only None is supported.

        Returns:
            Output of shape [..., hidden].

        Raises:
            RuntimeError: If a context-parallel mesh is passed or set on the module; if, with more than one
                process, the expert weights are not ``DTensor`` objects with the same mesh and placements in a
                supported layout (mesh ``("ep",)`` with placements ``(Shard(0),)``, or mesh ``("ep_shard", "ep")``
                with placements ``(Shard(1), Shard(0))``, with more than one expert-parallel rank); if, with more
                than one process, the HybridEP dispatcher was requested but the module was built for one process;
                if a latent projection weight is not FP32; or if the routed experts dispatch through HybridEP and
                the latent activations are not BF16.
            ValueError: If ``hidden`` is empty, has fewer than two dimensions or does not end in the configured
                hidden size, or ``padding_mask`` does not have the shape of ``hidden`` without its final axis.
        """
        if cp_mesh is not None or self.cp_mesh is not None:
            raise RuntimeError("ERNIE 5.5 MoE does not support context parallelism")
        if get_world_size_safe() != 1:
            if self.backend.dispatcher == "hybridep" and not self._dispatches_through_hybridep:
                raise RuntimeError(
                    "The HybridEP dispatcher was requested, but this ERNIE MoE was built for one process and "
                    "holds experts without a token dispatcher: build the model after the distributed job is "
                    "initialized"
                )
            gate_up, down = self.experts.gate_and_up_projs, self.experts.down_projs
            if not isinstance(gate_up, DTensor) or not isinstance(down, DTensor):
                raise RuntimeError("Distributed ERNIE MoE requires sharded expert DTensors")
            mesh = gate_up.device_mesh
            if mesh != down.device_mesh or gate_up.placements != down.placements:
                raise RuntimeError("Expert tensors must have matching mesh and placements")
            names, placements = mesh.mesh_dim_names, gate_up.placements
            ep_only = names == ("ep",) and placements == (Shard(0),) and mesh.size() > 1
            ep_fsdp = names == ("ep_shard", "ep") and placements == (Shard(1), Shard(0)) and mesh["ep"].size() > 1
            if not (ep_only or ep_fsdp):
                raise RuntimeError("Unsupported distributed ERNIE expert mesh layout")
        if hidden.ndim < 2 or hidden.shape[-1] != self.dim or hidden.numel() == 0:
            raise ValueError("Expected nonempty hidden states ending in the configured hidden size")
        if padding_mask is not None and padding_mask.shape != hidden.shape[:-1]:
            raise ValueError("padding_mask must match hidden batch and sequence dimensions")
        if self.fc1_latent_proj.weight.dtype != torch.float32 or self.fc2_latent_proj.weight.dtype != torch.float32:
            raise RuntimeError("ERNIE latent projection parameters must remain FP32")
        shape = hidden.shape
        x = hidden.reshape(-1, self.dim)
        token_mask = (
            ~padding_mask.bool().reshape(-1)
            if padding_mask is not None
            else torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        )
        weights, indices, _ = self.gate(x, token_mask, None)
        latent = self.fc1_latent_proj(x)
        if self._dispatches_through_hybridep and latent.dtype != torch.bfloat16:
            # HybridEP exchanges 16-bit activations only. Its dispatch copies the tensor as 16-bit elements without
            # a check, and its combine fails a C++ assertion for another width, which aborts the process, so the
            # dtype is checked before the dispatch.
            raise RuntimeError(
                "ERNIE 5.5 with the HybridEP dispatcher requires BF16 activations at the routed experts, got "
                f"{latent.dtype}: use model.dtype bfloat16 with an FSDP param_dtype of null or bfloat16, or the "
                "torch expert and dispatcher backends"
            )
        routed = self.experts(latent, token_mask, weights, indices)
        routed = self.latent_norm(routed)
        routed = self.fc2_latent_proj(routed)
        shared = self.shared_experts(x)
        return (routed + shared).reshape(shape)

    @torch.no_grad()
    def init_weights(self, buffer_device: torch.device, init_std: float = 0.02) -> None:
        """Initialize standard MoE modules plus both ERNIE router projections.

        Args:
            buffer_device: Device passed to the parent initializer.
            init_std: Standard deviation of the normal initializations.
        """
        super().init_weights(buffer_device, init_std)
        self.gate.init_weights(buffer_device, init_std)
        self.latent_norm.reset_parameters()
