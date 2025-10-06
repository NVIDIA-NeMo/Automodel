# coding=utf-8
# Copyright 2024 Nvidia Corporation. All rights reserved.
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

from copy import deepcopy
from typing import Optional, Dict, Any, Tuple

import torch
from transformers.cache_utils import Cache  # used to let GenerationMixin know that we use a Cache object

from .configuration_decilm import DeciLMConfig
from .transformers_4_44_2__cache_utils import Cache as Cache_4_44_2, SinkCache, StaticCache, SlidingWindowCache


class VariableCache(Cache_4_44_2, Cache):
    """
    A Cache object that supports a different Cache implementation for every layer,
    including layers without any kv-cache.
    Implemented using a list of Cache objects, each represents a "model" with 1 layer.
    The default implementation for the layer caches is StaticCache.
    The cache of each layer is allocated to the same gpu as the layer itself.
    """

    def __init__(
            self,
            *,  # key-word only, no positional args allowed to avoid mix-ups with newer transformers versions
            config: DeciLMConfig,
            batch_size: int = None,
            max_cache_len: int = None,
            dtype: torch.dtype = torch.float32,
            max_batch_size: Optional[int] = None,
            **kwargs,
    ) -> None:
        Cache_4_44_2.__init__(self)

        self.config = deepcopy(config)
        self.max_batch_size = batch_size or max_batch_size
        self.batch_size = self.max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.dtype = dtype

        self.layer_caches: list[Cache_4_44_2 | None] = [None] * config.num_hidden_layers
        self.layer_devices: list[torch.device | None] = [None] * config.num_hidden_layers

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.layer_caches[layer_idx] is None:
            self.layer_devices[layer_idx] = key_states.device
            self._init_layer_cache(layer_idx)

        layer_cache = self.layer_caches[layer_idx]
        assert layer_cache is not None, f"Trying to update the cache of a cache-less layer: {layer_idx=}"

        k_out, v_out = layer_cache.update(key_states=key_states,
                                          value_states=value_states,
                                          layer_idx=0,
                                          cache_kwargs=cache_kwargs)
        seq_len = self.get_seq_length(layer_idx)
        k_out = k_out[:, :, :seq_len, :]
        v_out = v_out[:, :, :seq_len, :]
        return k_out, v_out

    def _init_layer_cache(self, layer_idx: int) -> None:
        block_config = self.config.block_configs[layer_idx]
        attention_config = block_config.attention

        if attention_config.no_op or attention_config.replace_with_linear:
            return None

        device = self.layer_devices[layer_idx]
        assert device is not None, f"Trying to init layer cache for {layer_idx=} without device"

        config = deepcopy(self.config)
        config.num_hidden_layers = 1
        config.num_key_value_heads = self.config.num_attention_heads // attention_config.n_heads_in_group

        if attention_config.window_length is not None:
            if not attention_config.is_sink:
                config.sliding_window = attention_config.window_length
                self.layer_caches[layer_idx] = SlidingWindowCache(config=config,
                                                                  max_batch_size=self.max_batch_size,
                                                                  max_cache_len=self.max_cache_len,
                                                                  device=device,
                                                                  dtype=self.dtype)
                return
            elif not attention_config.unshifted_sink:
                self.layer_caches[layer_idx] = SinkCache(window_length=attention_config.window_length,
                                                         num_sink_tokens=attention_config.num_sink_tokens)
                return

        self.layer_caches[layer_idx] = StaticCache(config=config,
                                                   max_batch_size=self.max_batch_size,
                                                   max_cache_len=self.max_cache_len,
                                                   device=device,
                                                   dtype=self.dtype)

    def _get_first_real_cache(self) -> Cache:
        for layer_cache in self.layer_caches:
            if layer_cache is not None:
                return layer_cache
        raise ValueError(f"No real cache found, all layer caches are None.")

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if layer_idx == 0 and self.layer_caches[0] is None:
            try:
                layer_cache = self._get_first_real_cache()
            except ValueError:
                return 0
        else:
            layer_cache = self.layer_caches[layer_idx]
        return layer_cache.get_seq_length()

    def get_max_length(self) -> Optional[int]:
        """Returns the maximum sequence length of the cached states."""
        return self.max_cache_len

    def reset(self):
        for layer_idx in range(len(self.layer_caches)):
            layer_cache = self.layer_caches[layer_idx]
            if hasattr(layer_cache, "reset"):
                layer_cache.reset()
            else:
                self._init_layer_cache(layer_idx)
