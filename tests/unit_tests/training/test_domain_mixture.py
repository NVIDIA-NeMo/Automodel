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

import pytest
import torch

from nemo_automodel.components.training.domain_mixture import (
    DomainMixture,
    DomainMixtureConfig,
    DomainWeightConfig,
)


def _mixture():
    return DomainMixtureConfig(
        domains=(
            DomainWeightConfig(name="web", sampling_weight=3.0, objective_weight=1.0),
            DomainWeightConfig(name="code", sampling_weight=1.0, objective_weight=1.0),
        )
    ).build()


def test_domain_mixture_builds_normalized_importance_weights():
    mixture = _mixture()

    assert mixture.names == ("web", "code")
    assert mixture.sampling_weights == pytest.approx((0.75, 0.25))
    assert mixture.objective_weights == pytest.approx((0.5, 0.5))
    assert mixture.loss_multipliers == pytest.approx((2.0 / 3.0, 2.0))


def test_domain_mixture_expands_weights_and_counts_supervised_tokens():
    mixture = _mixture()
    dataset_ids = torch.tensor([0, 1], dtype=torch.int16)
    labels = torch.tensor([[1, 2, -100], [3, 4, 5]])

    weights = mixture.loss_weights(dataset_ids, labels)
    expected = torch.tensor([[2.0 / 3.0] * 3, [2.0] * 3])

    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(mixture.label_counts(dataset_ids, labels), torch.tensor([2, 3]))


def test_domain_mixture_combines_named_validation_losses():
    mixture = _mixture()

    assert mixture.weighted_validation_loss({"web": 2.0, "code": 4.0}) == pytest.approx(3.0)
    with pytest.raises(ValueError, match="Missing validation losses.*code"):
        mixture.weighted_validation_loss({"web": 2.0})


@pytest.mark.parametrize(
    "domains",
    [
        (),
        (DomainWeightConfig(name="web", sampling_weight=1.0, objective_weight=1.0),),
        (
            DomainWeightConfig(name="web", sampling_weight=1.0, objective_weight=1.0),
            DomainWeightConfig(name="web", sampling_weight=1.0, objective_weight=1.0),
        ),
        (
            DomainWeightConfig(name="web", sampling_weight=1.0, objective_weight=0.0),
            DomainWeightConfig(name="code", sampling_weight=1.0, objective_weight=0.0),
        ),
    ],
)
def test_domain_mixture_rejects_invalid_domain_sets(domains):
    with pytest.raises(ValueError):
        DomainMixtureConfig(domains=domains)


@pytest.mark.parametrize("path", ["", "  ", 42, ["/data/web"]])
def test_domain_mixture_rejects_invalid_corpus_path(path):
    with pytest.raises(ValueError, match="path must be a non-empty string.*web"):
        DomainWeightConfig(name="web", sampling_weight=1, objective_weight=1, path=path)


@pytest.mark.parametrize("provide_cache", [False, True])
def test_domain_mixture_preserves_standalone_positional_constructor(provide_cache):
    args = (("web", "code"), (0.5, 0.5), (0.25, 0.75), (0.5, 1.5))
    cache = {}
    mixture = DomainMixture(*args, cache) if provide_cache else DomainMixture(*args)

    weights = mixture.loss_weights(torch.tensor([0, 1]), torch.tensor([[1, 2], [3, 4]]))

    torch.testing.assert_close(weights, torch.tensor([[0.5, 0.5], [1.5, 1.5]]))
    assert mixture.weighted_validation_loss({"web": 2.0, "code": 4.0}) == pytest.approx(3.5)
    assert mixture.paths == ()
    if provide_cache:
        # The fifth positional argument must still receive the runtime cache.
        torch.testing.assert_close(cache[torch.device("cpu")], torch.tensor([0.5, 1.5]))
