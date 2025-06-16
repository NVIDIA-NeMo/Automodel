# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

import contextlib
import torch
import logging
import operator
from functools import reduce
from torch import _C
from torch.cuda import _lazy_call, _lazy_init
from torch.cuda import device as device_ctx_manager
from typing import List, Optional, Union
from packaging.version import Version as PkgVersion
from importlib.metadata import version

try:
    import transformer_engine  # pylint: disable=W0611

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    # Transformer Engine not found
    HAVE_TE = False


def get_te_version():
    """Get TE version from __version__; if not available use pip's. Use caching."""
    if not HAVE_TE:
        # No TE installed, so return None.
        return None

    def get_te_version_str():
        import transformer_engine as te

        if hasattr(te, "__version__"):
            return str(te.__version__)
        else:
            return version("transformer-engine")

    return PkgVersion(get_te_version_str())


def is_te_min_version(vers, check_equality=True):
    """Check if minimum version of `transformer-engine` is installed."""
    te_version = get_te_version()
    if not isinstance(te_version, PkgVersion):
        # No TE installed, so cannot satisfy any version requirement.
        return False

    if check_equality:
        return te_version >= PkgVersion(vers)
    return te_version > PkgVersion(vers)


# Check if Transformer Engine has class for fp8 tensors.
try:
    if is_te_min_version("2.0"):
        # In TE2.x, QuantizedTensor is the base class for all different type of fp8 tensors,
        # including fp8 tensor for delayed scaling, current scaling and mxfp8, etc.
        from transformer_engine.pytorch.tensor import (
            QuantizedTensor as FP8_TENSOR_CLASS,
        )
    else:
        from transformer_engine.pytorch.float8_tensor import (
            Float8Tensor as FP8_TENSOR_CLASS,
        )

    HAVE_TE_FP8_TENSOR_CLASS = True
except (ImportError, ModuleNotFoundError):
    # FP8 tensor class not found
    HAVE_TE_FP8_TENSOR_CLASS = False

try:
    from transformer_engine.pytorch.optimizers import (
        multi_tensor_applier,
        multi_tensor_scale,
    )

    multi_tensor_scale_impl = multi_tensor_scale
except ImportError:
    try:
        import amp_C
        from apex.multi_tensor_apply import multi_tensor_applier

        multi_tensor_scale_impl = amp_C.multi_tensor_scale
    except ImportError:
        import warnings

        warnings.warn(
            "Transformer Engine and Apex are not installed. "
            "Falling back to local implementations of "
            "multi_tensor_applier and multi_tensor_scale"
        )

        def local_multi_tensor_applier(op, noop_flag_buffer, tensor_lists, *args):
            """Multi tensor op applier"""
            return op(2048 * 32, noop_flag_buffer, tensor_lists, *args)

        def local_multi_tensor_scale(chunk_size, noop_flag, tensor_lists, scale):
            """Works as a drop-in replacement for amp_C.multi_tensor_scale."""
            for src, dst in zip(tensor_lists[0], tensor_lists[1]):
                dst.copy_(src * scale)

        multi_tensor_applier = local_multi_tensor_applier
        multi_tensor_scale_impl = local_multi_tensor_scale


def is_submodule(module, parent_module, strict=True):
    """
    Check if a module is a submodule of another module.
    """
    if strict:
        if module is parent_module:
            return False
    for m in parent_module.modules():
        if m is module:
            return True
    return False


def is_float8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine Float8Tensor.

    Note that in TE2.x, in order to support more recipes, the design of the fp8 tensor class has
    changed. Now Float8Tensor is only used for current scaling and delayed scaling. And mxfp8
    and blockwise scaling have their own fp8 tensor classes. These different fp8 tensor classes
    are both inherited from QuantizedTensor. So, for TE1.x, FP8_TENSOR_CLASS is Float8Tensor,
    and for TE2.x, FP8_TENSOR_CLASS is QuantizedTensor.
    """
    return HAVE_TE_FP8_TENSOR_CLASS and isinstance(tensor, FP8_TENSOR_CLASS)


def _multi_tensor_copy_this_to_that(
    this: List[torch.Tensor],
    that: List[torch.Tensor],
    overflow_buf: Optional[torch.Tensor] = None,
):
    """
    Use multi-tensor-applier to copy values from one list to another.
    We don't have a bfloat16 implementation so for now if the overflow_buf
    is not provided, we default back to simple loop copy to be compatible
    with bfloat16.
    """
    if overflow_buf is not None:
        overflow_buf.fill_(0)
        # Scaling with factor `1.0` is equivalent to copy.
        multi_tensor_applier(multi_tensor_scale_impl, overflow_buf, [this, that], 1.0)
    else:
        for this_, that_ in zip(this, that):
            that_.copy_(this_)


"""
The code below abstracts the functionalities needed for implementing "--fp8-param-gather" into
several functions. It provides different implementations for each function based on different
versions of TE, ensuring compatibility across various TE versions.

Currently, there are three functions:
    - modify_underlying_storage
        This function is used in DDP to place all parameters into a contiguous buffer. For
        non-fp8 tensors, replacing their data is simple, just using code like
        "tensor.data = new_data". However, for fp8 tensors, their raw data is not stored in the
        ".data" attribute, and it varies with different TE versions and different recipes. This
        function provides a unified interface to replace the underlying storage of a fp8 tensor.
    - quantize_param_shard
        This function is used in dist-opt to cast fp32 main params to fp8 params. For non-fp8
        params, this casting is as simple as "bf16_params.copy_(fp32_main_params)"; but for fp8
        params, the casting logic varies with different TE versions and different recipes. This
        function provides a unified interface to cast fp32 main params to fp8 params, and also
        updates the necessary attributes (like amax, scale, scale_inv or transpose cache) of the
        fp8 model params.
    - correct_amax_history_if_needed
        This function is used to correct the amax history of fp8 tensors. In TE1.x, some inplace
        copy operations will write unwanted values to the amax_history of fp8 tensors. This function
        corrects the amax_history back. For TE2.x, it's an empty function.
        Only useful for delayed scaling.
"""
if HAVE_TE and is_te_min_version("2.2"):
    # Supported TE versions: 2.2+
    from transformer_engine.pytorch.tensor import QuantizedTensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        from transformer_engine.pytorch.tensor.utils import replace_raw_data

        replace_raw_data(fp8_tensor, new_raw_data)

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:
        if len(model_params) == 0:
            return

        from transformer_engine.pytorch.tensor.utils import cast_master_weights_to_fp8

        args = [model_params, main_params, start_offsets, data_parallel_group]
        if fsdp_shard_model_params is not None:
            if get_te_version() == PkgVersion(
                "2.3.0.dev0+5fdd7bb"
            ) or is_te_min_version("2.3.0"):
                args.append(fsdp_shard_model_params)
            else:
                raise NotImplementedError(
                    f"FSDP with --fp8-param-gather is not supported in TE v{get_te_version()}"
                )
        cast_master_weights_to_fp8(*args)

elif HAVE_TE and is_te_min_version("2.0"):
    # Supported TE versions: 2.0
    from transformer_engine.pytorch.tensor import QuantizedTensor
    from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(
        fp8_tensor: QuantizedTensor, new_raw_data: torch.Tensor
    ) -> None:
        old_raw_data = fp8_tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        fp8_tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[QuantizedTensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            quantizer = model_param._quantizer
            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            out = Float8Tensor(
                shape=main_param.size(),
                dtype=model_param.dtype,
                requires_grad=False,
                data=shard_model_param,
                fp8_scale_inv=model_param._scale_inv,
                fp8_dtype=model_param._fp8_dtype,
                quantizer=quantizer,
            )
            quantizer.update_quantized(main_param, out)

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            quantizer = model_param._quantizer
            amaxes.append(quantizer.amax.view(1))
            scales.append(quantizer.scale.view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(
            len(scales), dtype=torch.float32, device=scales[0].device
        )
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(
            packed_scale_views, scale_invs, dummy_overflow_buf
        )

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(
            len(amaxes), dtype=torch.float32, device=amaxes[0].device
        )
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

elif HAVE_TE and is_te_min_version("1.0"):
    # Supported TE versions: 1.0 - 1.14
    from transformer_engine.pytorch.cpp_extensions import cast_to_fp8
    from transformer_engine.pytorch.float8_tensor import Float8Tensor

    def _modify_underlying_storage_impl(
        tensor: Float8Tensor, new_raw_data: torch.Tensor
    ) -> None:
        old_raw_data = tensor._data
        assert old_raw_data.dtype == new_raw_data.dtype
        new_raw_data.detach().copy_(old_raw_data)
        tensor._data = new_raw_data
        del old_raw_data

    def _quantize_param_shard_impl(
        model_params: List[Float8Tensor],
        main_params: List[torch.Tensor],
        start_offsets: List[int],
        data_parallel_group: torch.distributed.ProcessGroup,
        fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
    ) -> None:

        if len(model_params) == 0:
            return

        if fsdp_shard_model_params is None:
            fsdp_shard_model_params = [None] * len(model_params)

        for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
            model_params, main_params, start_offsets, fsdp_shard_model_params
        ):
            if main_param is None:
                continue

            if fsdp_shard_model_param is not None:
                shard_model_param = fsdp_shard_model_param
            else:
                shard_model_param = model_param._data.view(-1)[
                    start_offset : start_offset + main_param.numel()
                ]

            # When not using --fp8-param-gather, the main_param (fp32) is first cast to bf16/fp16,
            # and then cast to fp8 during forward.
            # Although it's not necessary when --fp8-param-gather is enabled, we still keep this
            # logic to keep numerical consistency. So here cast the main_param to model_param.dtype.
            main_param = main_param.to(model_param.dtype)
            cast_to_fp8(
                main_param.view(1, -1),
                model_param._fp8_meta["scaling_fwd"],
                model_param._fp8_meta_index,
                model_param._fp8_dtype,
                out=shard_model_param.view(1, -1),
            )

        amaxes = []
        scales = []
        scale_invs = []
        for model_param in model_params:
            fp8_meta = model_param._fp8_meta["scaling_fwd"]
            fp8_meta_index = model_param._fp8_meta_index
            amaxes.append(fp8_meta.amax_history[0][fp8_meta_index].view(1))
            scales.append(fp8_meta.scale[fp8_meta_index].view(1))
            scale_invs.append(model_param._scale_inv.view(1))
            model_param._reset_caches()

        dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device="cuda")

        # Update scaling factors.
        packed_scales = torch.empty(
            len(scales), dtype=torch.float32, device=scales[0].device
        )
        packed_scale_views = [packed_scales[i].view(1) for i in range(len(scales))]
        _multi_tensor_copy_this_to_that(scales, packed_scale_views, dummy_overflow_buf)
        torch.reciprocal(packed_scales, out=packed_scales)
        _multi_tensor_copy_this_to_that(
            packed_scale_views, scale_invs, dummy_overflow_buf
        )

        # Reduce amaxes.
        # Note: Assume each param has a separate amax.
        packed_amaxes = torch.empty(
            len(amaxes), dtype=torch.float32, device=amaxes[0].device
        )
        packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
        _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
        torch.distributed.all_reduce(
            packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
        )
        _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)

else:
    # Fallback impl if TE version is invalid or TE is not installed.
    def _modify_underlying_storage_impl(*args, **kwargs):
        raise RuntimeError(
            "Invalid Transformer Engine version for FP8 distributed optimizer"
        )

    def _quantize_param_shard_impl(*args, **kwargs):
        raise RuntimeError(
            "Invalid Transformer Engine version for FP8 distributed optimizer"
        )


def modify_underlying_storage(tensor: torch.Tensor, new_raw_data: torch.Tensor):
    """Replace the underlying raw data of a tensor with new data."""
    _modify_underlying_storage_impl(tensor, new_raw_data)


def quantize_param_shard(
    model_params,
    main_params,
    start_offsets,
    data_parallel_group,
    fsdp_shard_model_params=None,
):
    """Cast shard fp32 main params to fp8 model params."""
    _quantize_param_shard_impl(
        model_params,
        main_params,
        start_offsets,
        data_parallel_group,
        fsdp_shard_model_params,
    )


def _get_cuda_rng_state(
    device: Union[int, str, torch.device] = "cuda",
    clone: bool = False,
    graph_safe: bool = False,
) -> torch.Tensor:
    """Return the random number generator state of the specified GPU.

    Arguments:
        device (int): The gpu to retrieve the rng state
        clone (bool): Whether to also clone the retrieved RNG state
        graph_safe (bool): Get the rng state in a graph safe manner.

    This function is adapted from torch.cuda.random.get_rng_state()"""

    # if not using cuda graphs, just use the builtin pytorch function
    if not graph_safe:
        return torch.cuda.random.get_rng_state(device=device)

    _lazy_init()
    if isinstance(device, str):
        device = torch.device(device)
    elif isinstance(device, int):
        device = torch.device("cuda", device)
    idx = device.index
    if idx is None:
        idx = torch.cuda.current_device()

    default_generator = torch.cuda.default_generators[idx]
    if clone:
        return default_generator.clone_state()
    return default_generator.graphsafe_get_state()


def _set_cuda_rng_state(
    new_state: torch.Tensor, device: int = -1, graph_safe: bool = False
):
    """Sets the random number generator state of the current GPU.

    Arguments:
        new_state (torch.ByteTensor): The desired state
        device (int): The gpu to retrieve the rng state
        graph_safe (bool): Set the rng state in a graph safe manner.

    This function is adapted from PyTorch repo (torch.cuda.set_rng_state)
    with a single change: the input state is not cloned. Cloning caused
    major performance issues for +4 GPU cases.
    """
    if hasattr(_C, "_cuda_setRNGState") and callable(_C._cuda_setRNGState):
        # older PyTorch
        def cb():
            with device_ctx_manager(device):
                _C._cuda_setRNGState(new_state)

    else:
        # newer PyTorch
        if device == -1:
            device = torch.device("cuda")
        elif isinstance(device, str):
            device = torch.device(device)
        elif isinstance(device, int):
            device = torch.device("cuda", device)

        def cb():
            idx = device.index
            if idx is None:
                idx = torch.cuda.current_device()
            default_generator = torch.cuda.default_generators[idx]

            # if graph capturing, set the rng state in a cudagraphable way
            if graph_safe:
                default_generator.graphsafe_set_state(new_state)
            else:
                default_generator.set_state(new_state)

    _lazy_call(cb)


def initialize_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
    force_reset: bool = False,
):
    """Create the RNG tracker. 'use_te_rng_tracker' determines whether to use
    Megatron or TransformerEngine's implementation.
    In particular, TransformerEngine's implementation is cudagraphable and supports FP8.
    """
    global _CUDA_RNG_STATE_TRACKER
    global _CUDA_RNG_STATE_TRACKER_INITIALIZED
    if force_reset:
        _CUDA_RNG_STATE_TRACKER = None
        _CUDA_RNG_STATE_TRACKER_INITIALIZED = False

    if (
        "_CUDA_RNG_STATE_TRACKER_INITIALIZED" in globals()
        and _CUDA_RNG_STATE_TRACKER_INITIALIZED
    ):
        return

    _MODEL_PARALLEL_RNG_TRACKER_NAME = "model-parallel-rng"

    # Get the base tracker class
    base_tracker = None
    if HAVE_TE and use_te_rng_tracker:
        if not is_te_min_version("1.5.0"):
            raise RuntimeError(
                "use_te_rng_tracker requires TransformerEngine version >= 1.5"
            )

        class TECudaRNGStatesTracker(
            transformer_engine.pytorch.distributed.CudaRNGStatesTracker
        ):
            """Wraps TransformerEngine's CudaRNGStatesTracker so that it is
            interchangeable with Megatron's RNG tracker"""

            def __init__(self, is_inference_rng_tracker=False):
                super().__init__()
                self.reset()
                self.is_inference_rng_tracker = is_inference_rng_tracker

            def is_initialized(self):
                """Checks if the internal RNG state has been set with set_states()."""
                return self._is_initialized

            def reset(self):
                """Reset the internal RNG state."""
                super().reset()
                self._is_initialized = False

            def set_states(self, states):
                """Set the internal RNG state."""
                super().set_states(states)
                self._is_initialized = True

            def add(self, name, seed):
                """Track the rng state."""
                super().add(name, seed)
                self._is_initialized = True

        base_tracker = TECudaRNGStatesTracker
        tracker_kwargs = {"is_inference_rng_tracker": inference_rng_tracker}
    else:

        class CudaRNGStatesTracker:
            """Tracker for the cuda RNG states.

            Using the `add` method, a cuda rng state is initialized based on
            the input `seed` and is assigned to `name`. Later, by forking the
            rng state, we can perform operations and return to our starting
            cuda state.
            """

            def __init__(
                self, use_cudagraphable_rng=False, is_inference_rng_tracker=False
            ):
                self.reset()
                self.use_cudagraphable_rng = use_cudagraphable_rng
                self.is_inference_rng_tracker = is_inference_rng_tracker

                if self.use_cudagraphable_rng:
                    assert (
                        hasattr(torch.cuda.CUDAGraph, "register_generator_state")
                        and hasattr(torch.Generator, "graphsafe_set_state")
                        and hasattr(torch.Generator, "graphsafe_get_state")
                        and hasattr(torch.Generator, "clone_state")
                    ), "Tried using cudagraphs with RNG, however not detected in pytorch!"

            def is_initialized(self):
                """Checks if the internal RNG state has been set wirth set_states()."""
                return self._is_initialized

            def reset(self):
                """Set to the initial state (no tracker)."""

                # Track if initialized.
                self._is_initialized = False

                # Map from a string name to the cuda rng state.
                self.states_ = {}

                # Seeds are just for book keeping and ensure no seed is set twice.
                self.seeds_ = set()

            def get_states(self):
                """Get rng states. Copy the dictionary so we have direct
                pointers to the states, not just a pointer to the dictionary."""
                states = {}
                for name in self.states_:
                    states[name] = self.states_[name]
                return states

            def set_states(self, states):
                """Set the rng states. For efficiency purposes, we do not check
                the size of seed for compatibility."""
                self._is_initialized = True
                self.states_ = states

            def add(self, name, seed):
                """Track the rng state."""
                self._is_initialized = True
                # Check seed is not already used.
                if seed in self.seeds_:
                    raise Exception("seed {} already exists".format(seed))
                self.seeds_.add(seed)
                # Check that state is not already defined.
                if name in self.states_:
                    raise Exception("cuda rng state {} already exists".format(name))

                # If available, create the state in a graph safe manner
                if self.use_cudagraphable_rng:
                    new_state = _get_cuda_rng_state(clone=True, graph_safe=True)
                    new_state.manual_seed(seed)
                    self.states_[name] = new_state
                else:
                    # Get the current rng state.
                    orig_rng_state = torch.cuda.get_rng_state()
                    # Set the new state and store it.
                    torch.cuda.manual_seed(seed)
                    self.states_[name] = torch.cuda.get_rng_state()
                    # Reset rng state to what it was.
                    _set_cuda_rng_state(orig_rng_state)

            @contextlib.contextmanager
            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Fork the cuda rng state, perform operations, and exit with
                the original state."""
                # Check if we have added the state
                if name not in self.states_:
                    raise Exception("cuda rng state {} is not added".format(name))
                # Store current rng state.
                orig_cuda_rng_state = _get_cuda_rng_state(
                    graph_safe=self.use_cudagraphable_rng
                )
                # Set rng state to the desired one
                _set_cuda_rng_state(
                    self.states_[name], graph_safe=self.use_cudagraphable_rng
                )
                # Record cpu RNG state
                cpu_rng_state = torch.get_rng_state()
                # Do the stuff we wanted to do.
                try:
                    yield
                finally:
                    # Throw a warning if cpu RNG state changed
                    if not torch.all(cpu_rng_state == torch.get_rng_state()).item():
                        logging.getLogger(__name__).warning(
                            "CPU RNG state changed within GPU RNG context"
                        )
                    # Update the current rng state for later use.
                    self.states_[name] = _get_cuda_rng_state(
                        graph_safe=self.use_cudagraphable_rng
                    )
                    # And set the state to the original state we started with.
                    _set_cuda_rng_state(
                        orig_cuda_rng_state, graph_safe=self.use_cudagraphable_rng
                    )

        base_tracker = CudaRNGStatesTracker
        tracker_kwargs = {
            "use_cudagraphable_rng": use_cudagraphable_rng,
            "is_inference_rng_tracker": inference_rng_tracker,
        }

    if inference_rng_tracker:

        class InferenceCudaRNGStatesTracker(base_tracker):  # type: ignore[valid-type, misc]
            """RNG tracker for inference."""

            def add(self, name, seed):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def set_states(self, states):
                """Mirrors the interface from the training RNG tracker."""
                pass

            def fork(self, name=_MODEL_PARALLEL_RNG_TRACKER_NAME):
                """Mirrors the interface from the training RNG tracker."""
                return contextlib.nullcontext()

        tracker_class = InferenceCudaRNGStatesTracker
    else:
        tracker_class = base_tracker

    _CUDA_RNG_STATE_TRACKER = tracker_class(**tracker_kwargs)
    _CUDA_RNG_STATE_TRACKER_INITIALIZED = True


def get_cuda_rng_tracker(
    use_te_rng_tracker: bool = False,
    inference_rng_tracker: bool = False,
    use_cudagraphable_rng: bool = False,
):
    """Get cuda rng tracker."""
    initialize_rng_tracker(
        use_te_rng_tracker, inference_rng_tracker, use_cudagraphable_rng
    )
    return _CUDA_RNG_STATE_TRACKER


class GlobalMemoryBuffer:
    """Global buffer to avoid dynamic memory allocations.
    Caller should ensure that buffers of the same name
    are not used concurrently."""

    def __init__(self):
        self.buffer = {}

    def get_tensor(self, tensor_shape, dtype, name):
        """
        Returns (potentially) a sub-tensor from the self.buffer for the given shape.
        """
        required_len = reduce(operator.mul, tensor_shape, 1)
        if (
            self.buffer.get((name, dtype), None) is None
            or self.buffer[(name, dtype)].numel() < required_len
        ):
            self.buffer[(name, dtype)] = torch.empty(
                required_len,
                dtype=dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

        return self.buffer[(name, dtype)][0:required_len].view(*tensor_shape)


def get_global_memory_buffer():
    """Return the global GlobalMemoryBuffer object"""
    global _GLOBAL_MEMORY_BUFFER
    if "_GLOBAL_MEMORY_BUFFER" not in globals() or _GLOBAL_MEMORY_BUFFER is None:
        _GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
    return _GLOBAL_MEMORY_BUFFER
