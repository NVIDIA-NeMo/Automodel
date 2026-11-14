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

"""Image ownership and differentiable residual transport through shared PP hooks."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp, stage_vlm_media_for_pp
from nemo_automodel.components.distributed.pipelining.functional import split_model_into_stages
from nemo_automodel.components.models.inkling2.config import Inkling2Config
from nemo_automodel.components.models.inkling2.multimodal import Inkling2ForConditionalGeneration
from nemo_automodel.shared.parameter_names import canonical_parameter_fqn

from .test_multimodal import vision_config
from .test_precision import initialize_nonuniform_queries
from .test_reference import tiny_config

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("ac", [False, True])
def test_image_chunks_pipeline_outputs_and_all_decoder_gradients(monkeypatch: pytest.MonkeyPatch, ac: bool) -> None:
    """Unequal image ownership, two in-flight graphs and AC retain exact gradients."""
    torch.set_num_threads(1)
    torch.manual_seed(817)
    text = tiny_config()
    text.vocab_size = text.unpadded_vocab_size = 200058
    reference = Inkling2ForConditionalGeneration(Inkling2Config(text_config=text, vision_config=vision_config()))
    reference.visual.requires_grad_(False)
    initialize_nonuniform_queries(reference)

    def stage_stub(
        submod: nn.Module, stage_index: int, num_stages: int, device: torch.device, group: None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(submod=submod, stage_index=stage_index)

    # Real splitter and pruning, with communication tested by the CUDA suite.
    monkeypatch.setattr("nemo_automodel.components.distributed.pipelining.functional.PipelineStage", stage_stub)
    parts = []
    for rank in range(2):
        mesh = SimpleNamespace(size=lambda: 2, get_local_rank=lambda: rank, get_group=lambda _: None)
        stages, local = split_model_into_stages(reference, mesh, "pp", "interleaved1f1b", torch.device("cpu"))
        parts.extend(zip(stages, local))
    parts = [part for _, part in sorted(parts, key=lambda pair: pair[0].stage_index)]
    assert parts[0].visual is not None
    assert all(part.visual is None for part in parts[1:])
    assert len(parts) == 4
    owners = {}
    for part in parts:
        if ac:
            for key, layer in part.layers.items():
                part.layers[key] = checkpoint_wrapper(layer)
        names = {canonical_parameter_fqn(n): p for n, p in part.named_parameters()}
        assert not owners.keys() & names.keys()
        owners.update(names)
    assert owners.keys() == dict(reference.named_parameters()).keys()

    ids = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 200005, 200005, 200005, 5, 6]])
    mask = torch.tensor([[0, 0, 0, 0, 0, 0], [0, 1, 1, 1, 0, 0]])
    patches = torch.randn(3, 2, 40, 40, 3, dtype=torch.bfloat16)
    grid = torch.tensor([[1, 1, 1], [1, 1, 2]])
    # Two samples own zero and two images respectively, with different patch counts.
    batch = prepare_vlm_media_for_pp(
        {"pixel_values": patches, "image_grid_thw": grid, "n_images_per_sample": torch.tensor([0, 2])},
        batch_size=2,
        n_microbatches=2,
    )
    pp = SimpleNamespace(info=SimpleNamespace(has_first_stage=True, schedule=None, stages=None))
    actual, expected = [], []
    with stage_vlm_media_for_pp(pp, parts, batch):
        for row in range(2):
            media = {"pixel_values": patches, "image_grid_thw": grid} if row else {}
            expected.append(reference(ids[row : row + 1], mm_token_type_ids=mask[row : row + 1], **media))
            value = parts[0](ids[row : row + 1], mm_token_type_ids=mask[row : row + 1])
            for part in parts[1:]:
                value = part(*value, mm_token_type_ids=mask[row : row + 1])
            actual.append(value)
            torch.testing.assert_close(value, expected[-1], rtol=0, atol=0)
        with pytest.raises(ValueError, match="exhausted"):
            parts[0](ids[:1])
    assert parts[0]._vlm_pixel_values_chunks is None and parts[0]._vlm_chunk_idx is None
    for got, wanted in zip(reversed(actual), reversed(expected)):
        # Restrict scalar objective, retaining all decoder owners including the output head.
        got[..., :37].square().mean().backward()
        wanted[..., :37].square().mean().backward()
    for name, parameter in reference.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None and owners[name].grad is None
        else:
            assert parameter.grad is not None and owners[name].grad is not None, name
            torch.testing.assert_close(owners[name].grad, parameter.grad, rtol=0, atol=0, msg=name)
    assert owners["layers.5.attn.relative_table"].grad.norm() > 0
