import os, torch, torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torchao.optim import AdamW8bit
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29611")
dist.init_process_group("nccl", rank=0, world_size=1); torch.cuda.set_device(0)
mesh = init_device_mesh("cuda", (1,))
plain = torch.nn.Parameter(torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16))
dt = torch.nn.Parameter(distribute_tensor(torch.randn(4096, 2048, device="cuda", dtype=torch.bfloat16), mesh, [Shard(0)]))
small = torch.nn.Parameter(torch.randn(2048, device="cuda", dtype=torch.bfloat16))
opt = AdamW8bit([plain, dt, small], lr=1e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
before = [p.detach().clone() for p in (plain, dt, small)]
for _ in range(3):
    for p in (plain, dt, small): p.grad = torch.randn_like(p)
    opt.step(); opt.zero_grad()
for name, p, b in zip(("plain", "dtensor", "small"), (plain, dt, small), before):
    d = (p.detach() - b)
    d = d.full_tensor() if hasattr(d, "full_tensor") else d
    print(f"RESULT {name}: finite={bool(torch.isfinite(d).all())} changed={bool(d.abs().sum() > 0)}")
st = opt.state[plain]
print("RESULT state types:", {k: type(v).__name__ for k, v in st.items()})
dist.destroy_process_group()
