import math

import pytest
import torch

from nemo_automodel.components.optim.gradient_sketch.sketch import (
    SketchSettings,
    fill_projection_,
    local_weight_rows,
    make_projection,
    projection_seed,
)


def test_settings_defaults_and_validation():
    settings = SketchSettings()
    assert settings.rank == 256 and settings.optimizer == "row_adam"
    assert settings.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    with pytest.raises(ValueError, match="rank"):
        SketchSettings(rank=0)
    with pytest.raises(ValueError, match="distribution"):
        SketchSettings(distribution="uniform")
    with pytest.raises(ValueError, match="optimizer"):
        SketchSettings(optimizer="adam")
    with pytest.raises(ValueError, match="beta2"):
        SketchSettings(beta2=1.0)
    with pytest.raises(ValueError, match="epsilon"):
        SketchSettings(epsilon=0.0)


def test_projection_seed_ignores_wrapper_names():
    plain = projection_seed(42, 3, "model.layers.0.self_attn.q_proj")
    wrapped = projection_seed(42, 3, "model._fsdp_wrapped_module.layers.0._checkpoint_wrapped_module.self_attn.q_proj")
    assert plain == wrapped
    assert projection_seed(42, 4, "model.layers.0.self_attn.q_proj") != plain


@pytest.mark.parametrize("distribution", ["rademacher", "gaussian"])
def test_make_projection_is_deterministic_and_unit_column_norm(distribution):
    a = make_projection(64, 512, 7, device="cpu", dtype=torch.float32, distribution=distribution)
    b = make_projection(64, 512, 7, device="cpu", dtype=torch.float32, distribution=distribution)
    c = make_projection(64, 512, 8, device="cpu", dtype=torch.float32, distribution=distribution)
    assert torch.equal(a, b) and not torch.equal(a, c)
    if distribution == "rademacher":
        assert torch.equal(a.abs(), torch.full_like(a, 1 / math.sqrt(64)))
    assert (a * a).sum(dim=0).mean().item() == pytest.approx(1.0, abs=0.1)


def test_fill_projection_rejects_bad_inputs():
    with pytest.raises(ValueError, match="distribution"):
        fill_projection_(torch.empty(4, 4), 1, distribution="uniform")
    with pytest.raises(ValueError, match="positive"):
        make_projection(0, 4, 1, device="cpu", dtype=torch.float32, distribution="rademacher")


def test_local_weight_rows_plain_and_3d():
    w2 = torch.zeros(6, 4)
    local, offset = local_weight_rows(w2)
    assert local is w2 and offset == 0
    w3 = torch.zeros(2, 3, 4)
    local, offset = local_weight_rows(w3)
    assert local.shape == (6, 4) and offset == 0
