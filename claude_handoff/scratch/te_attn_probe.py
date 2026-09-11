import os, torch
from transformer_engine.pytorch import DotProductAttention
torch.cuda.set_device(0)
H, KV, D = 16, 2, 256
for mask in ("causal", "padding_causal"):
    dpa = DotProductAttention(H, D, num_gqa_groups=KV, attn_mask_type=mask, qkv_format="bshd").cuda()
    for S in (4096, 8192):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        q = torch.randn(1, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, S, KV, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        kw = {}
        if mask == "padding_causal":
            cu = torch.tensor([0, S], device="cuda", dtype=torch.int32)
            kw = dict(cu_seqlens_q=cu, cu_seqlens_kv=cu)
        base = torch.cuda.memory_allocated()
        out = dpa(q, k, v, **kw); out.backward(torch.randn_like(out))
        torch.cuda.synchronize()
        print(f"RESULT mask={mask} S={S} peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:.2f} GiB")
