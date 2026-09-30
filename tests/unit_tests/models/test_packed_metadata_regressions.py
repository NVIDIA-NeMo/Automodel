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

"""CPU coverage of packed metadata at hybrid-model forward boundaries."""

import importlib
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from nemo_automodel.components.datasets.packing import build_packed_sequence_metadata


@pytest.mark.parametrize("family,class_name", [("kimi_linear", "KimiLinear48BModel"), ("kimi_k3", "KimiK3TextModel")])
@pytest.mark.parametrize("mask_kind", ["indexed", "sdpa", "explicit", "metadata_only"])
def test_kimi_packed_forward_preserves_padding_for_every_layer(family, class_name, mask_kind):
    module = importlib.import_module(f"nemo_automodel.components.models.{family}.model")
    cls = getattr(module, class_name)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.norm = nn.Identity()
    model.use_attn_residuals = False

    def capture_layer(hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        """Keep activations unchanged while the mock records the layer contract.

        Args:
            hidden_states: Tensor of shape [batch, sequence, hidden].
            **kwargs: Layer arguments including attention_mask of shape [batch,
                sequence] or [batch, 1, sequence, sequence], padding_mask of shape
                [batch, sequence], and packed metadata from the dataset.

        Returns:
            The input tensor of shape [batch, sequence, hidden], without copying.
        """
        return hidden_states

    layers = [nn.Identity(), nn.Identity()]
    for is_linear, layer in zip((True, False), layers):
        layer.is_linear_attn = is_linear
        layer.forward = Mock(side_effect=capture_layer)
    model.layers = nn.ModuleDict({str(i): layer for i, layer in enumerate(layers)})
    doc_ids = torch.tensor([[1, 1, 2, 2, 0, 0]])
    metadata = build_packed_sequence_metadata(doc_ids)
    kwargs = dict(metadata)
    if mask_kind == "sdpa":
        kwargs["_packed_seq_ids"] = doc_ids
        kwargs["attention_mask"] = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    elif mask_kind != "metadata_only":
        kwargs["attention_mask"] = doc_ids
    expected = doc_ids == 0
    if mask_kind == "explicit":
        expected = torch.tensor([[False, True, False, False, True, True]])
        kwargs["padding_mask"] = expected
    model(inputs_embeds=torch.randn(1, 6, 4), **kwargs)
    assert layers[0].forward.call_args.kwargs["attention_mask"] is None
    for layer in layers:
        torch.testing.assert_close(layer.forward.call_args.kwargs["padding_mask"], expected)


@pytest.mark.parametrize(
    "family,class_name", [("qwen3_5", "Qwen3_5DenseTextBackbone"), ("qwen3_5_moe", "Qwen3_5MoeTextModelBackend")]
)
def test_qwen_cp_rejects_unconverted_neat_metadata(family, class_name):
    from nemo_automodel.components.distributed.context_parallel.sharder import shard_batch_load_balanced

    class Mesh:
        def size(self):
            return 2

        def get_local_rank(self):
            return 0

    cls = getattr(importlib.import_module(f"nemo_automodel.components.models.{family}.model"), class_name)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model._cp_enabled = True
    model.embed_tokens = nn.Embedding(16, 8)
    doc_ids = torch.tensor([[1, 1, 2, 2, 0, 0, 0, 0], [1, 1, 2, 2, 3, 3, 0, 0]])
    batch = dict(
        input_ids=torch.ones(2, 8, dtype=torch.long),
        labels=torch.ones(2, 8, dtype=torch.long),
        attention_mask=doc_ids,
        **build_packed_sequence_metadata(doc_ids),
    )
    _, local_batch, _ = shard_batch_load_balanced(Mesh(), None, batch)
    assert "attention_mask" not in local_batch
    assert local_batch["cu_seqlens"].tolist() == [[0, 2, 4, -1], [0, 2, 4, 6]]
    with pytest.raises(ValueError, match="unsupported with load-balanced context parallelism"):
        model(**local_batch)


@pytest.mark.parametrize("family,class_name", [("kimi_linear", "KimiLinear48BModel"), ("kimi_k3", "KimiK3TextModel")])
def test_kimi_explicit_metadata_reaches_kda_without_private_document_ids(family, class_name):
    module = importlib.import_module(f"nemo_automodel.components.models.{family}.model")
    cls = getattr(module, class_name)
    model = cls.__new__(cls)
    nn.Module.__init__(model)
    model.norm = nn.Identity()
    model.use_attn_residuals = False
    kda = module.KimiDeltaAttention.__new__(module.KimiDeltaAttention)
    nn.Module.__init__(kda)
    kda.is_linear_attn = True

    def recurrent_core(hidden_states: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Use a segmented prefix sum to expose boundary and padding mistakes.

        Args:
            hidden_states: Unpadded tensor of shape [1, tokens, hidden].
            cu_seqlens: Flat document boundaries of shape [documents + 1].

        Returns:
            Per-document prefix sums of shape [1, tokens, hidden].
        """
        assert hidden_states.shape == (1, 8, 4)
        assert cu_seqlens.tolist() == [0, 2, 4, 7, 8]
        cuts = cu_seqlens.tolist()
        return torch.cat([hidden_states[:, start:end].cumsum(1) for start, end in zip(cuts, cuts[1:])], dim=1)

    kda._kda_core = recurrent_core
    model.layers = nn.ModuleDict({"0": kda})
    doc_ids = torch.tensor([[1, 1, 2, 2, 0, 0], [1, 1, 1, 2, 0, 0]])
    hidden = torch.randn(2, 6, 4, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_()
    output = model(inputs_embeds=hidden, attention_mask=doc_ids, **build_packed_sequence_metadata(doc_ids))
    expected = torch.zeros_like(ref_hidden)
    for row, segments in enumerate([[(0, 2), (2, 4)], [(0, 3), (3, 4)]]):
        for start, end in segments:
            expected[row, start:end] = ref_hidden[row, start:end].cumsum(0)
    torch.testing.assert_close(output, expected)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    expected.backward(upstream)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)


@pytest.mark.parametrize("family", ["kimi_linear", "kimi_k3"])
def test_kimi_kda_preserves_flat_cu_seqlens_only_inputs(family):
    module = importlib.import_module(f"nemo_automodel.components.models.{family}.model")
    layer = module.KimiDeltaAttention.__new__(module.KimiDeltaAttention)
    nn.Module.__init__(layer)

    def core(hidden_states: torch.Tensor, *, cu_seqlens: torch.Tensor) -> torch.Tensor:
        """Stand in for the GPU kernel while checking the existing THD contract.

        Args:
            hidden_states: Unpadded tensor of shape [1, tokens, hidden].
            cu_seqlens: Flat boundaries of shape [documents + 1].

        Returns:
            A tensor of shape [1, tokens, hidden].
        """
        assert cu_seqlens.tolist() == [0, 2, 5]
        return hidden_states * 2

    layer._kda_core = core
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    output = layer(hidden, cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32))
    torch.testing.assert_close(output, hidden * 2)
    upstream = torch.randn_like(output)
    output.backward(upstream)
    torch.testing.assert_close(hidden.grad, upstream * 2)
