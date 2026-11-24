import torch

from nemo_automodel.components.models.hy_v4_flash.model import HY4FlashForCausalLM

from .tiny import CPU_BACKEND, tiny_config


def test_tiny_forward_backward_is_finite():
    torch.manual_seed(0)
    model = HY4FlashForCausalLM(tiny_config(), backend=CPU_BACKEND)
    model.initialize_weights(dtype=torch.float32)
    ids = torch.randint(3, 280, (2, 24))
    out = model(ids, labels=ids)
    assert torch.isfinite(out.logits).all()
    out.loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
