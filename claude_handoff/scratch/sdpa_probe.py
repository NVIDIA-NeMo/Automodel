import time, torch, torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
H, KV, D = 16, 2, 256
for name, be in (("flash", SDPBackend.FLASH_ATTENTION), ("efficient", SDPBackend.EFFICIENT_ATTENTION), ("cudnn", SDPBackend.CUDNN_ATTENTION)):
    for S in (4096, 8192, 32768):
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        q = torch.randn(1, H, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(1, KV, S, D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        base = torch.cuda.memory_allocated()
        try:
            with sdpa_kernel([be]):
                for i in range(2):
                    torch.cuda.synchronize(); t = time.time()
                    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
                    out.backward(torch.randn_like(out)); torch.cuda.synchronize(); dt = time.time() - t
            print(f"RESULT {name:9s} S={S:6d} ok  peak_extra={(torch.cuda.max_memory_allocated()-base)/2**30:6.2f} GiB  fwd+bwd={dt*1e3:8.1f} ms")
        except Exception as e:
            print(f"RESULT {name:9s} S={S:6d} FAIL {type(e).__name__}: {str(e)[:120]}")
