import torch
from torchao.optim import AdamW8bit
from nemo_automodel.components.optim.optimizer import LRSchedulerConfig  # noqa: F401  (import check only)
from nemo_automodel.components.optim.scheduler import OptimizerParamScheduler
import inspect
p = torch.nn.Parameter(torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16))
opt = AdamW8bit([p], lr=1e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
sig = inspect.signature(OptimizerParamScheduler.__init__)
print("RESULT sched params:", list(sig.parameters)[:14])
kw = dict(optimizer=opt, init_lr=0.0, max_lr=1e-5, min_lr=1e-6, lr_warmup_steps=2, lr_decay_steps=6, lr_decay_style="cosine",
          start_wd=0.1, end_wd=0.1, wd_incr_steps=6, wd_incr_style="constant")
kw = {k: v for k, v in kw.items() if k in sig.parameters}
s = OptimizerParamScheduler(**kw)
for i in range(6):
    p.grad = torch.randn_like(p); opt.step(); opt.zero_grad(); s.step(increment=1)
    lr = opt.param_groups[0]["lr"]
    print(f"RESULT step {i} lr={float(lr):.3e} type={type(lr).__name__} finite={bool(torch.isfinite(p).all())}")
