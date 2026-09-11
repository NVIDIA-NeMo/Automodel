import os, sys, torch, torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torchao.optim import AdamW8bit
os.environ.setdefault("MASTER_ADDR", "127.0.0.1"); os.environ.setdefault("MASTER_PORT", "29612")
dist.init_process_group("nccl", rank=0, world_size=1); torch.cuda.set_device(0)
mesh = init_device_mesh("cuda", (1,)); ckpt = sys.argv[1]
def make():
    torch.manual_seed(0)
    m = torch.nn.Linear(2048, 4096, bias=False, device="cuda", dtype=torch.bfloat16)
    m.weight = torch.nn.Parameter(distribute_tensor(m.weight.detach(), mesh, [Shard(0)]))
    return m, AdamW8bit(m.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
m, opt = make()
for _ in range(2):
    m.weight.grad = torch.ones_like(m.weight) * 0.01; opt.step(); opt.zero_grad()
try:
    dcp.save({"optim": get_optimizer_state_dict(m, opt)}, checkpoint_id=ckpt); print("RESULT save ok")
    m2, opt2 = make()
    m2.weight.grad = torch.zeros_like(m2.weight); opt2.step(); opt2.zero_grad()  # materialize state
    sd = {"optim": get_optimizer_state_dict(m2, opt2)}; dcp.load(sd, checkpoint_id=ckpt)
    set_optimizer_state_dict(m2, opt2, sd["optim"]); print("RESULT load ok")
    a = opt.state[m.weight]["exp_avg"]; b = opt2.state[m2.weight]["exp_avg"]
    fa = a.to_local() if hasattr(a, "to_local") else a; fb = b.to_local() if hasattr(b, "to_local") else b
    print("RESULT exp_avg types:", type(a).__name__, type(fa).__name__)
    print("RESULT exp_avg roundtrip equal:", torch.equal(fa.dequantize() if hasattr(fa,'dequantize') else fa, fb.dequantize() if hasattr(fb,'dequantize') else fb))
except Exception as e:
    print("RESULT FAIL", type(e).__name__, str(e)[:300])
dist.destroy_process_group()
