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
#
# Includes Apache-2.0 code adapted from ByteDance-Seed/Bagel. Upstream references:
#   https://github.com/bytedance-seed/BAGEL
#   data/dataset_base.py (``SimpleCustomBatch`` + ``collate_wrapper``)
# Upstream copyright: Copyright 2025 Bytedance Ltd. and/or its affiliates.
# Attribute and key names match the packed-batch schema consumed by the BAGEL
# model forward path.

"""Multimodal collate functions.

BAGEL uses **packed** sequences (samples concatenated along the sequence
axis with a cumulative-seqlens index), not left/right padding. The collate
function is essentially a pass-through that wraps the single packed dict
produced by :class:`PackedDataset` in a ``SimpleCustomBatch`` with
``pin_memory`` / ``cuda`` helpers.
"""

from __future__ import annotations


class SimpleCustomBatch:
    """Pass-through wrapper around one packed batch from :class:`PackedDataset`."""

    def __init__(self, batch):
        data = batch[0]
        self.batch_data_indexes = data["batch_data_indexes"]
        self.sequence_length = data["sequence_length"]
        self.sample_lens = data["sample_lens"]
        self.packed_text_ids = data["packed_text_ids"]
        self.packed_text_indexes = data["packed_text_indexes"]
        self.packed_position_ids = data["packed_position_ids"]

        self.use_flex = "nested_attention_masks" not in data.keys()

        if self.use_flex:
            self.split_lens = data["split_lens"]
            self.attn_modes = data["attn_modes"]
        else:
            self.nested_attention_masks = data["nested_attention_masks"]

        if "padded_images" in data.keys():
            self.padded_images = data["padded_images"]
            self.patchified_vae_latent_shapes = data["patchified_vae_latent_shapes"]
            self.packed_latent_position_ids = data["packed_latent_position_ids"]
            self.packed_vae_token_indexes = data["packed_vae_token_indexes"]

        if "packed_vit_tokens" in data.keys():
            self.packed_vit_tokens = data["packed_vit_tokens"]
            self.packed_vit_position_ids = data["packed_vit_position_ids"]
            self.packed_vit_token_indexes = data["packed_vit_token_indexes"]
            self.vit_token_seqlens = data["vit_token_seqlens"]

        if "packed_timesteps" in data.keys():
            self.packed_timesteps = data["packed_timesteps"]
            self.mse_loss_indexes = data["mse_loss_indexes"]

        if "packed_label_ids" in data.keys():
            self.packed_label_ids = data["packed_label_ids"]
            self.ce_loss_indexes = data["ce_loss_indexes"]
            self.ce_loss_weights = data["ce_loss_weights"]

    def pin_memory(self):
        self.packed_text_ids = self.packed_text_ids.pin_memory()
        self.packed_text_indexes = self.packed_text_indexes.pin_memory()
        self.packed_position_ids = self.packed_position_ids.pin_memory()

        if not self.use_flex:
            self.nested_attention_masks = [item.pin_memory() for item in self.nested_attention_masks]

        if hasattr(self, "padded_images"):
            self.padded_images = self.padded_images.pin_memory()
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.pin_memory()
            self.packed_latent_position_ids = self.packed_latent_position_ids.pin_memory()

        if hasattr(self, "packed_timesteps"):
            self.packed_timesteps = self.packed_timesteps.pin_memory()
            self.mse_loss_indexes = self.mse_loss_indexes.pin_memory()

        if hasattr(self, "packed_vit_tokens"):
            self.packed_vit_tokens = self.packed_vit_tokens.pin_memory()
            self.packed_vit_position_ids = self.packed_vit_position_ids.pin_memory()
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.pin_memory()
            self.vit_token_seqlens = self.vit_token_seqlens.pin_memory()

        if hasattr(self, "packed_label_ids"):
            self.packed_label_ids = self.packed_label_ids.pin_memory()
            self.ce_loss_indexes = self.ce_loss_indexes.pin_memory()
            self.ce_loss_weights = self.ce_loss_weights.pin_memory()

        return self

    def cuda(self, device):
        self.packed_text_ids = self.packed_text_ids.to(device)
        self.packed_text_indexes = self.packed_text_indexes.to(device)
        self.packed_position_ids = self.packed_position_ids.to(device)

        if not self.use_flex:
            self.nested_attention_masks = [item.to(device) for item in self.nested_attention_masks]

        if hasattr(self, "padded_images"):
            self.padded_images = self.padded_images.to(device)
            self.packed_vae_token_indexes = self.packed_vae_token_indexes.to(device)
            self.packed_latent_position_ids = self.packed_latent_position_ids.to(device)

        if hasattr(self, "packed_timesteps"):
            self.packed_timesteps = self.packed_timesteps.to(device)
            self.mse_loss_indexes = self.mse_loss_indexes.to(device)

        if hasattr(self, "packed_vit_tokens"):
            self.packed_vit_tokens = self.packed_vit_tokens.to(device)
            self.packed_vit_position_ids = self.packed_vit_position_ids.to(device)
            self.packed_vit_token_indexes = self.packed_vit_token_indexes.to(device)
            self.vit_token_seqlens = self.vit_token_seqlens.to(device)

        if hasattr(self, "packed_label_ids"):
            self.packed_label_ids = self.packed_label_ids.to(device)
            self.ce_loss_indexes = self.ce_loss_indexes.to(device)
            self.ce_loss_weights = self.ce_loss_weights.to(device)

        return self

    def to_dict(self):
        data = dict(
            sequence_length=self.sequence_length,
            sample_lens=self.sample_lens,
            packed_text_ids=self.packed_text_ids,
            packed_text_indexes=self.packed_text_indexes,
            packed_position_ids=self.packed_position_ids,
            batch_data_indexes=self.batch_data_indexes,
        )

        if not self.use_flex:
            data["nested_attention_masks"] = self.nested_attention_masks
        else:
            data["split_lens"] = self.split_lens
            data["attn_modes"] = self.attn_modes

        if hasattr(self, "padded_images"):
            data["padded_images"] = self.padded_images
            data["patchified_vae_latent_shapes"] = self.patchified_vae_latent_shapes
            data["packed_latent_position_ids"] = self.packed_latent_position_ids
            data["packed_vae_token_indexes"] = self.packed_vae_token_indexes

        if hasattr(self, "packed_vit_tokens"):
            data["packed_vit_tokens"] = self.packed_vit_tokens
            data["packed_vit_position_ids"] = self.packed_vit_position_ids
            data["packed_vit_token_indexes"] = self.packed_vit_token_indexes
            data["vit_token_seqlens"] = self.vit_token_seqlens

        if hasattr(self, "packed_timesteps"):
            data["packed_timesteps"] = self.packed_timesteps
            data["mse_loss_indexes"] = self.mse_loss_indexes

        if hasattr(self, "packed_label_ids"):
            data["packed_label_ids"] = self.packed_label_ids
            data["ce_loss_indexes"] = self.ce_loss_indexes
            data["ce_loss_weights"] = self.ce_loss_weights

        return data


def collate_wrapper():
    """Return the BAGEL-style identity collate (wraps a single packed dict)."""

    def collate_fn(batch):
        return SimpleCustomBatch(batch)

    return collate_fn


def bagel_packed_collate_fn(batch):
    """Canonical name in AM's collate-fn registry."""
    return SimpleCustomBatch(batch)
