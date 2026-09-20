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

"""Tests for the TE packed-mask format and the post-label hook.

Two contracts are covered:

1. TransformerEngine derives per-document ``cu_seqlens`` from the compact document
   map, so the packed collater must not expand the dense ``[B, 1, S, S]`` mask for
   it, and the shared TE preprocessing must reject a non-2-D mask outright.
2. ``PreTokenizedDatasetWrapper`` builds labels itself, after its only existing
   hook runs. ``label_post_hook`` is the seam that lets a caller restrict the
   supervised span under packing, where the dataloader's ``collate_fn`` is bypassed.
"""

import pytest
import torch


def _packed_sample() -> dict:
    """One pack holding two documents: ids 1,1 then 2,2,2."""
    return {
        "input_ids": torch.tensor([10, 20, 30, 40, 50]),
        "labels": torch.tensor([-100, 20, 30, 40, 50]),
        "attention_mask": torch.tensor([1, 1, 2, 2, 2]),
        "position_ids": torch.tensor([0, 1, 0, 1, 2]),
        "n_images": 0,
        "n_videos": 0,
    }


class TestPackedMaskFormatForTE:
    """TE must receive the compact map, never the quadratic mask."""

    def test_te_keeps_the_compact_document_map(self):
        """attn_implementation='te' returns the indexed [B, S] map."""
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        result = neat_packed_vlm_collater([_packed_sample()], attn_implementation="te")

        assert result["attention_mask"].ndim == 2
        assert result["attention_mask"].shape == (1, 5)
        assert result["attention_mask"][0].tolist() == [1, 1, 2, 2, 2]

    def test_te_emits_packed_seq_ids(self):
        """The document ids are exposed so the model can build cu_seqlens."""
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        result = neat_packed_vlm_collater([_packed_sample()], attn_implementation="te")

        assert "_packed_seq_ids" in result
        assert result["_packed_seq_ids"][0].tolist() == [1, 1, 2, 2, 2]

    def test_sdpa_still_gets_the_dense_mask(self):
        """SDPA consumes an explicit mask, so its behavior must not regress."""
        from nemo_automodel.components.datasets.vlm.collate_fns import neat_packed_vlm_collater

        result = neat_packed_vlm_collater([_packed_sample()], attn_implementation="sdpa")

        assert result["attention_mask"].ndim == 4
        assert result["attention_mask"].shape == (1, 1, 5, 5)
        # No cross-document attention: the first token of document 2 (index 2)
        # must not see the last token of document 1 (index 1).
        assert not bool(result["attention_mask"][0, 0, 2, 1])


class TestTEMaskGuard:
    """A packed mask handed to TE used to be reshaped into nonsense."""

    def test_four_dimensional_mask_raises(self):
        """The 4-D block-causal mask has no TE mask route and must not be reshaped."""
        from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn

        q = torch.zeros(1, 4, 2, 8)
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()

        with pytest.raises(ValueError, match="2-D"):
            preprocess_args_and_kwargs_for_attn(q, q.clone(), q.clone(), mask, "te")

    def test_two_dimensional_padding_mask_still_works(self):
        """The ordinary unpacked path is unaffected."""
        from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn

        q = torch.zeros(1, 4, 2, 8)
        mask = torch.ones(1, 4, dtype=torch.long)

        _, _, _, kwargs = preprocess_args_and_kwargs_for_attn(q, q.clone(), q.clone(), mask, "te")

        assert kwargs["attn_mask_type"] == "padding_causal"
        assert kwargs["attention_mask"].shape == (1, 1, 1, 4)

    def test_cu_seqlens_route_is_selected_without_a_mask(self):
        """No mask plus cu_seqlens selects TE's native THD format."""
        from nemo_automodel.components.attention.utils import preprocess_args_and_kwargs_for_attn

        q = torch.zeros(5, 2, 8)
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)

        _, _, _, kwargs = preprocess_args_and_kwargs_for_attn(
            q, q.clone(), q.clone(), None, "te", cu_seqlens=cu_seqlens
        )

        assert kwargs["qkv_format"] == "thd"
        assert kwargs["attn_mask_type"] == "padding_causal"
        assert torch.equal(kwargs["cu_seqlens_q"], cu_seqlens)


class TestLabelPostHook:
    """The hook runs after labels exist, unlike post_tokenize_hook."""

    def _wrapper(self, label_post_hook):
        from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapper

        wrapper = PreTokenizedDatasetWrapper.__new__(PreTokenizedDatasetWrapper)
        wrapper.label_post_hook = label_post_hook
        return wrapper

    def test_hook_can_narrow_the_supervised_span(self):
        """A hook that masks all but the final label is honored."""
        wrapper = self._wrapper(None)
        sample = {"input_ids": torch.tensor([1, 2, 3]), "labels": torch.tensor([1, 2, 3])}

        def _keep_last(out, processor):
            out["labels"] = torch.tensor([-100, -100, 3])
            return out

        wrapper.label_post_hook = _keep_last
        hooked = wrapper.label_post_hook(sample, None)
        assert hooked["labels"].tolist() == [-100, -100, 3]

    def test_config_forwards_the_hook(self):
        """PreTokenizedDatasetWrapperConfig.build threads the hook onto the wrapper."""
        from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapperConfig

        def _hook(sample, processor):
            return sample

        config = PreTokenizedDatasetWrapperConfig(label_post_hook=_hook)
        wrapper = config.build(dataset=[], processor=None)

        assert wrapper.label_post_hook is _hook

    def test_config_default_is_none(self):
        """The field is backward compatible: absent means no hook."""
        from nemo_automodel.components.datasets.vlm.datasets import PreTokenizedDatasetWrapperConfig

        assert PreTokenizedDatasetWrapperConfig().label_post_hook is None


class TestLabelPostHookConfigValidation:
    """The YAML key must be validated where the other hook is."""

    def test_non_callable_is_rejected(self):
        """A stringly-typed hook should fail at config resolution, not at runtime."""
        from nemo_automodel.components.datasets.vlm.datasets import make_cord_v2_dataset
        from nemo_automodel.recipes._typed_config import RecipeConfig

        with pytest.raises(TypeError, match="label_post_hook_fn"):
            RecipeConfig.resolve_vlm_dataloader(
                dataset_node={"_target_": make_cord_v2_dataset},
                dataloader_node=None,
                packed_sequence_node={"pack_size": 128, "label_post_hook_fn": "not-callable"},
            )

    def test_callable_hook_is_accepted(self):
        """A resolved callable reaches the pretokenization config."""
        from nemo_automodel.components.datasets.vlm.datasets import make_cord_v2_dataset
        from nemo_automodel.recipes._typed_config import RecipeConfig

        def _hook(sample, processor):
            return sample

        resolved = RecipeConfig.resolve_vlm_dataloader(
            dataset_node={"_target_": make_cord_v2_dataset},
            dataloader_node=None,
            packed_sequence_node={"pack_size": 128, "label_post_hook_fn": _hook},
        )

        assert resolved.pretokenization.label_post_hook is _hook
