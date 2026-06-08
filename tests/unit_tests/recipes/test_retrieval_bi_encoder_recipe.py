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

import torch

from nemo_automodel.recipes.retrieval.train_bi_encoder import _unwrap_model_for_attrs, _uses_multi_vector_scoring


class _RetrieverAttrs(torch.nn.Module):
    pooling = "multi_vector"
    l2_normalize = True
    do_distributed_inbatch_negative = True
    detach_distributed_inbatch_negatives = False


class _DDPLikeWrapper(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module


def test_retrieval_attrs_unwrap_ddp_like_wrapper():
    inner = _RetrieverAttrs()
    wrapped = _DDPLikeWrapper(inner)

    attr_model = _unwrap_model_for_attrs(wrapped)

    assert attr_model is inner
    assert attr_model.l2_normalize is True
    assert attr_model.do_distributed_inbatch_negative is True
    assert attr_model.detach_distributed_inbatch_negatives is False
    assert _uses_multi_vector_scoring(wrapped) is True


def test_retrieval_attrs_accept_unwrapped_model():
    inner = _RetrieverAttrs()

    assert _unwrap_model_for_attrs(inner) is inner
    assert _uses_multi_vector_scoring(inner) is True
