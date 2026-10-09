import pytest
import torch
from torch import nn

from nemo_automodel.components.utils.parameter_config import ParameterConfig, apply_parameter_config


def test_surface_precision_precedes_fp32_checkpoint_load():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1)).to(torch.bfloat16)
    frozen_before = model[1].weight.detach().clone()
    config = ParameterConfig(trainable_names=("0.weight", "0.bias"))
    apply_parameter_config(model, config)
    assert model[0].weight.dtype == torch.float32
    assert model[0].bias.requires_grad
    assert not model[1].weight.requires_grad
    assert model[1].weight.dtype == torch.bfloat16
    assert torch.equal(model[1].weight, frozen_before)
    source = model.state_dict()
    source["0.weight"] = torch.full((2, 2), 0.1234567, dtype=torch.float32)
    model.load_state_dict(source)
    assert torch.equal(model[0].weight, source["0.weight"])


def test_surface_on_meta_before_materialization():
    model = nn.Linear(2, 2, device="meta", dtype=torch.bfloat16)
    apply_parameter_config(model, ParameterConfig(("weight",)))
    assert model.weight.dtype == torch.float32
    assert model.bias.dtype == torch.bfloat16
    assert model.weight.is_meta
    assert not model.bias.requires_grad


@pytest.mark.parametrize("names", [(), ("weight", "weight"), ("missing",)])
def test_surface_rejects_bad_names_before_mutation(names):
    model = nn.Linear(2, 2, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        apply_parameter_config(model, ParameterConfig(names))
    assert all(p.requires_grad and p.dtype == torch.bfloat16 for p in model.parameters())
