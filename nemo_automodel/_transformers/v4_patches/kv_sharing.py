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

"""Runtime fix for HF cross-layer KV sharing under FSDP2.

Some HF models (e.g. gemma3n) implement cross-layer KV sharing by threading a
single mutable ``shared_kv_states`` dict through every decoder layer as a forward
kwarg: the last full-length layer of each attention type writes its K/V into the
dict, and the later "shared" layers read them back.

Under FSDP2 with ``MixedPrecisionPolicy(cast_forward_inputs=True)``, the per-layer
``fully_shard`` pre-forward casts forward inputs by running ``tree_map`` over
``(args, kwargs)``. ``tree_map`` *reconstructs* every container it understands —
including a plain ``dict`` — so each wrapped layer receives a fresh, empty copy of
``shared_kv_states``. The writing layer fills its throwaway copy and the reading
layer sees an empty one, raising ``KeyError`` (e.g. ``KeyError: 18``) on the first
forward. This only triggers once ``fully_shard`` is active (dp_shard > 1).

The fix swaps the dict for a :class:`_SharedKVStates` instance: it behaves like the
dict the model expects but, because it is not a type pytree knows how to flatten,
``tree_map`` treats it as a leaf and passes the *same* instance to every layer, so
in-place writes are visible to the readers. See AM-454.
"""

import logging
from functools import partial

logger = logging.getLogger(__name__)


class _SharedKVStates:
    """Pytree-opaque, dict-like holder for HF cross-layer KV sharing.

    Implements just enough of the mapping protocol for HF attention
    (``__getitem__`` / ``__setitem__`` / ``__contains__``). Crucially it is *not*
    a ``dict``/``list``/``tuple``, so ``torch.utils._pytree`` treats it as a leaf
    and FSDP2's forward-input cast does not reconstruct it.

    Mirrors ``gemma4_moe._FSDPSafeSharedKVStates`` (#2566), which solves the same
    problem in-model for gemma4; gemma3n is native HF so we inject the holder via
    a forward pre-hook instead (see ``install_kv_sharing_holder``).
    """

    __slots__ = ("_d",)

    def __init__(self):
        self._d = {}

    def __getitem__(self, key):
        return self._d[key]

    def __setitem__(self, key, value):
        self._d[key] = value

    def __contains__(self, key):
        return key in self._d

    def __len__(self):
        return len(self._d)

    def get(self, key, default=None):
        return self._d.get(key, default)

    def keys(self):
        return self._d.keys()

    def clear(self):
        self._d.clear()


def _kv_sharing_stacks(model):
    """Yield (module, layers) for each gemma3n decoder stack that uses KV sharing.

    Scoped to gemma3n on purpose. gemma3n is a native HF model whose
    ``Gemma3nTextModel.forward`` builds a plain ``shared_kv_states`` dict that we
    cannot edit in place, so we swap in an FSDP2-safe holder via a forward
    pre-hook. Other kv-sharing models (e.g. gemma4) already make their own
    shared store FSDP2-safe in-model (``gemma4_moe`` model.py, #2566) and may
    thread a caller-supplied store (the speculative drafter via
    ``gemma4_drafter`` composite); injecting our holder onto their layers would
    clobber it and break that path, so we leave them alone.
    """
    for module in model.modules():
        layers = getattr(module, "layers", None)
        cfg = getattr(module, "config", None)
        text_cfg = getattr(cfg, "text_config", None) or cfg
        if layers is None or getattr(text_cfg, "num_kv_shared_layers", 0) <= 0:
            continue
        if not str(getattr(text_cfg, "model_type", "")).startswith("gemma3n"):
            continue
        try:
            layer_list = list(layers)
        except TypeError:
            continue
        if layer_list:
            yield module, layer_list


def should_install_kv_sharing_holder(model_parts):
    """True if any model part has a decoder stack that uses HF KV sharing."""
    for mp in model_parts:
        for _ in _kv_sharing_stacks(mp):
            return True
    return False


def install_kv_sharing_holder(model_parts):
    """Inject a pytree-opaque ``shared_kv_states`` holder so KV sharing survives FSDP2.

    Registers a per-layer ``forward_pre_hook`` that swaps the ``shared_kv_states``
    kwarg for a single :class:`_SharedKVStates` instance shared across the stack.
    The holder is reset at the first layer of every forward so stale tensors are
    not retained across steps. No-op for models without KV sharing.
    """
    hooked = 0
    for mp in model_parts:
        for _module, layer_list in _kv_sharing_stacks(mp):
            # A wrapper may re-expose the same ``.layers``; only hook a stack once.
            if getattr(layer_list[0], "_kv_sharing_holder_installed", False):
                continue

            holder = _SharedKVStates()

            def _pre_hook(_mod, args, kwargs, _holder=holder, _is_first=False):
                # Reset on the first layer so we don't carry K/V across forwards.
                if _is_first:
                    _holder.clear()
                kwargs["shared_kv_states"] = _holder
                return args, kwargs

            for idx, layer in enumerate(layer_list):
                layer.register_forward_pre_hook(
                    partial(_pre_hook, _is_first=(idx == 0)),
                    with_kwargs=True,
                )
                layer._kv_sharing_holder_installed = True
                hooked += 1

    if hooked:
        logger.info("Installed shared-KV-states holder on %d decoder layers (FSDP2 KV-sharing fix).", hooked)
    return hooked
