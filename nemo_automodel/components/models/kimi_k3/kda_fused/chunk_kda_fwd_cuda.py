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

"""Source text of the fused chunked Kimi-K3 KDA forward (CUDA) and its torch binding.

``kda_fused._forward_ext`` hands these strings to ``torch.utils.cpp_extension.load_inline`` as
``cpp_sources = [HEADER, BINDING]`` and ``cuda_sources = [HEADER, KERNEL]``; torch writes them to ``main.cpp`` and
``cuda.cu`` in the build directory and builds the extension once, so the wheel ships Python files only. The header is
inlined into both translation units (there is no file to ``#include``), which is why it carries no ``#pragma once``.
"""

HEADER = r"""// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Fused chunked Kimi Delta Attention forward for the Kimi-K3 call. One CTA per (head, document)
// walks the 64-token chunks with the [K, V] state on-chip (mma.sync + ldmatrix, sm_80+). Specialised to head dims
// K = V = 128, a packed layout (batch 1 with int32 cu_seqlens), q/k l2-normalised in kernel, a bounded log gate, no
// initial/final state. Measured on GB200 against FLA 0.4.2 chunk_kda at the Kimi-K3 layer-microbatch shape
// (8192 tokens = 4 documents, 96 heads): 0.78-0.80 ms vs 2.07-2.19 ms, max |diff| 1-2 bf16 ulp.

#include <cuda_runtime.h>
// Returns the status of the one-time cudaFuncSetAttribute (dynamic shared memory); the launch itself is checked by
// the caller (C10_CUDA_KERNEL_LAUNCH_CHECK in the binding).
cudaError_t kda_launch(const void* q, const void* k, const void* v, const void* g, const void* beta,
                       const void* cu, void* o, int H, int ndoc, cudaStream_t stream);
"""

KERNEL = r"""// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Fused chunked Kimi Delta Attention forward for the Kimi-K3 call. One CTA per (head, document)
// walks the 64-token chunks with the [K, V] state on-chip (mma.sync + ldmatrix, sm_80+). Specialised to head dims
// K = V = 128, a packed layout (batch 1 with int32 cu_seqlens), q/k l2-normalised in kernel, a bounded log gate, no
// initial/final state. Measured on GB200 against FLA 0.4.2 chunk_kda at the Kimi-K3 layer-microbatch shape
// (8192 tokens = 4 documents, 96 heads): 0.78-0.80 ms vs 2.07-2.19 ms, max |diff| 1-2 bf16 ulp.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

#define BT   64
#define DK   128
#define DV   128
#define LDB  136   // bf16 row stride for 128-wide tiles
#define LDS  72    // bf16 row stride for 64-wide tiles
#define LDA  72    // bf16 row stride for the 64-wide WY matrices
#define LDW  24    // bf16 row stride for 16x16 WY scratch
#define LDG  132    // float row stride for 64-wide tiles
#define NTHREADS 512
#define NW   (NTHREADS/32)
#define L2E  1.44269504088896340736f

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float ex2(float x){ float r; asm("ex2.approx.f32 %0,%1;":"=f"(r):"f"(x)); return r; }
__device__ __forceinline__ uint32_t sm(const void* p){ return (uint32_t)__cvta_generic_to_shared(p); }

// A-fragment (16x16) from row-major [M][K]
__device__ __forceinline__ void ld_a(uint32_t (&r)[4], const bf16* P, int LD, int m0, int k0, int lane){
  int row = m0 + (lane&7) + 8*((lane>>3)&1);
  int col = k0 + 8*((lane>>3)>>1);
  uint32_t a = sm(&P[row*LD+col]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
// A-fragment = Z^T, Z row-major [K][M]
__device__ __forceinline__ void ld_a_t(uint32_t (&r)[4], const bf16* P, int LD, int m0, int k0, int lane){
  int row = k0 + (lane&7) + 8*((lane>>3)>>1);
  int col = m0 + 8*((lane>>3)&1);
  uint32_t a = sm(&P[row*LD+col]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
// two B-fragments (n0, n0+8) from X row-major [N][K], B = X^T
__device__ __forceinline__ void ld_b2(uint32_t (&r)[4], const bf16* P, int LD, int n0, int k0, int lane){
  int row = n0 + (lane&7) + 8*((lane>>3)>>1);
  int col = k0 + 8*((lane>>3)&1);
  uint32_t a = sm(&P[row*LD+col]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
// two B-fragments (n0, n0+8) from Y row-major [K][N], B = Y
__device__ __forceinline__ void ld_b2_t(uint32_t (&r)[4], const bf16* P, int LD, int n0, int k0, int lane){
  int row = k0 + (lane&7) + 8*((lane>>3)&1);
  int col = n0 + 8*((lane>>3)>>1);
  uint32_t a = sm(&P[row*LD+col]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];\n"
    : "=r"(r[0]),"=r"(r[1]),"=r"(r[2]),"=r"(r[3]) : "r"(a));
}
// single B-fragment from X row-major [N][K], B = X^T
__device__ __forceinline__ void ld_b1(uint32_t (&r)[2], const bf16* P, int LD, int n0, int k0, int lane){
  int row = n0 + (lane&7);
  int col = k0 + 8*((lane>>3)&1);
  uint32_t a = sm(&P[row*LD+col]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1},[%2];\n"
    : "=r"(r[0]),"=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void mma1(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]){
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
    : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
    : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

// ---- bf16 16x16x16 machinery for the WY inverse ----
__device__ __forceinline__ void zero16(float (&acc)[2][4]){
  #pragma unroll
  for (int t=0;t<2;t++)
  #pragma unroll
  for (int e=0;e<4;e++) acc[t][e]=0.f;
}
// acc += P(16x16 row-major [M][K]) * Q(16x16 row-major [K][N])
__device__ __forceinline__ void mm16b(float (&acc)[2][4], const bf16* P, int LDp,
                                      const bf16* Q, int LDq, int lane){
  uint32_t af[4], b4[4];
  ld_a(af, P, LDp, 0, 0, lane);
  ld_b2_t(b4, Q, LDq, 0, 0, lane);
  uint32_t b0[2]={b4[0],b4[1]}, b1[2]={b4[2],b4[3]};
  mma1(acc[0], af, b0);
  mma1(acc[1], af, b1);
}
__device__ __forceinline__ void st16b(const float (&acc)[2][4], bf16* D, int LD, int lane){
  int r0 = lane>>2, c0 = (lane&3)*2;
  #pragma unroll
  for (int t=0;t<2;t++){
    *(__nv_bfloat162*)(&D[r0*LD + t*8 + c0])     = __floats2bfloat162_rn(acc[t][0],acc[t][1]);
    *(__nv_bfloat162*)(&D[(r0+8)*LD + t*8 + c0]) = __floats2bfloat162_rn(acc[t][2],acc[t][3]);
  }
}

#define SZ128 (BT*LDB*2)
#define SZS   (DK*LDB*2)
#define SZA   (BT*LDA*2)
#define SZ64  (BT*LDS*2)

__global__ __launch_bounds__(NTHREADS,1)
void kda_fused(const bf16* __restrict__ q, const bf16* __restrict__ k, const bf16* __restrict__ v,
               const float* __restrict__ g, const float* __restrict__ beta,
               const int* __restrict__ cu, bf16* __restrict__ o, int H)
{
  extern __shared__ __align__(16) char smem[];
  int off = 0;
  bf16*  sQ  = (bf16*)(smem+off); off += SZ128;
  bf16*  sK  = (bf16*)(smem+off); off += SZ128;
  bf16*  sKE = (bf16*)(smem+off); off += SZ128;
  bf16*  sV  = (bf16*)(smem+off); off += SZ128;   // KH[0:64] / v / vnew
  bf16*  sW  = (bf16*)(smem+off); off += SZ128;   // KH[128:192] / sX / w
  bf16*  sS  = (bf16*)(smem+off); off += SZS;
  bf16*  sA  = (bf16*)(smem+off); off += SZA;
  bf16*  sAQ = (bf16*)(smem+off); off += SZ64;
  bf16*  sCr = (bf16*)(smem+off); off += 4*4*DK*2;
  bf16*  sCq = (bf16*)(smem+off); off += 4*DK*2;
  bf16*  sCk = (bf16*)(smem+off); off += 4*DK*2;
  float* sGL = (float*)(smem+off); off += DK*4;
  float* sSb = (float*)(smem+off); off += 4*DK*4;
  float* sGf = (float*)(smem+off); off += 4*DK*4;
  float* sBe = (float*)(smem+off); off += BT*4;
  bf16*  sWY = (bf16*)(smem+off); off += 4*2*16*LDW*2;
  float* sG  = (float*)sV;   // [64][LDG] fp32, alive only until cross
  bf16*  sX  = (bf16*)sW;       // WY inverse T, [64][LDA] bf16; upper blocks stay zero

  const int h   = blockIdx.x;
  const int doc = blockIdx.y;
  const int s0 = cu[doc], s1 = cu[doc+1];
  const int L = s1 - s0;
  if (L <= 0) return;
  const int NT = (L + BT - 1)/BT;
  const int tid = threadIdx.x;
  const int warp = tid>>5, lane = tid&31;
  const long rstride = (long)H*DK;

  float Sacc[2][4][4];
  #pragma unroll
  for (int mi=0;mi<2;mi++)
    #pragma unroll
    for (int i=0;i<4;i++)
      #pragma unroll
      for (int e=0;e<4;e++) Sacc[mi][i][e]=0.f;
  const int sm_row0 = 32*(warp&3);
  const int sm_col0 = 32*(warp>>2);
  for (int i=tid; i<DK*LDB/2; i+=NTHREADS) ((uint32_t*)sS)[i]=0u;
  for (int i=tid; i<BT*LDA/2; i+=NTHREADS) ((uint32_t*)sX)[i]=0u;   // upper blocks of T are never written

  const int lrow = tid>>4, lc8 = (tid&15)*8;   // rows lrow and lrow+32 of [64][128]
  const int cb = tid>>7, cd = tid&127;          // cumsum mapping

  for (int ic=0; ic<NT; ++ic) {
    const int t0 = s0 + ic*BT;
    const int nrows = min(BT, s1-t0);
    __syncthreads();
    // ---- load q,k ; prefetch v ----
    uint4 vpf[2], qreg[2], kreg[2];   // l2-normalized q,k stay in registers across the cumsum barrier
    uint4 qo2[2], eo2[2], ko2[2];     // q*e, k*e, k*ie stay in registers across the A-tile barrier
    {
      const bf16* qp = q + (long)(t0)*rstride + (long)h*DK;
      const bf16* kp = k + (long)(t0)*rstride + (long)h*DK;
      const bf16* vp = v + (long)(t0)*rstride + (long)h*DK;
      const float* gp = g + (long)(t0)*rstride + (long)h*DK;
      const int gr0 = tid>>5, gc0 = (tid&31)*4;
      const float scale = 0.08838834764831845f;
      // Issue every global load for this chunk up front, before any dependent math.
      // Previously the l2norm reduction (11 dependent ops incl. 8 shuffles) sat
      // between the q/k loads and the g/beta loads, serialising two memory round
      // trips; now all 11 loads are in flight concurrently and only the l2norm
      // waits on q/k.
      uint4 qa[2], kb[2], gv[4];
      float bev = 0.f;
      #pragma unroll
      for (int rr=0; rr<2; ++rr){
        const int row = lrow + rr*32;
        qa[rr] = make_uint4(0,0,0,0); kb[rr] = make_uint4(0,0,0,0);
        vpf[rr] = make_uint4(0,0,0,0);
        // q/k/v/g are each read exactly once (NCU: L1 hit rate 2.5%), so pull them
        // with .cg -- cache in L2 only, never allocate an L1 line. L1/TEX is the top
        // limiter at 64%, and any line these loads install evicts shared/ldmatrix
        // capacity from the same unified L1 for no reuse benefit.
        if (row<nrows){ qa[rr]  = __ldcg((const uint4*)(qp + (long)row*rstride + lc8));
                        kb[rr]  = __ldcg((const uint4*)(kp + (long)row*rstride + lc8));
                        vpf[rr] = __ldcg((const uint4*)(vp + (long)row*rstride + lc8)); }
      }
      #pragma unroll
      for (int j=0;j<4;j++){
        int gr = gr0 + j*16;
        gv[j] = make_uint4(0,0,0,0);
        if (gr<nrows) gv[j] = __ldcg((const uint4*)(gp + (long)gr*rstride + gc0));
      }
      if (tid<BT) bev = (tid<nrows)? __ldcg(&beta[(long)(t0+tid)*H + h]) : 0.f;
      #pragma unroll
      for (int j=0;j<4;j++) *(uint4*)(&sG[(gr0+j*16)*LDG+gc0]) = gv[j];
      if (tid<BT) sBe[tid] = bev;
      #pragma unroll
      for (int rr=0; rr<2; ++rr){
        uint4 a = qa[rr], b = kb[rr];
        // ---- l2 normalize q,k in registers (same thread owns these 8 dims) ----
        __nv_bfloat162* pq = (__nv_bfloat162*)&a;
        __nv_bfloat162* pk = (__nv_bfloat162*)&b;
        float aq=0.f, ak=0.f;
        #pragma unroll
        for (int j=0;j<4;j++){
          float2 x = __bfloat1622float2(pq[j]); aq += x.x*x.x + x.y*x.y;
          float2 y = __bfloat1622float2(pk[j]); ak += y.x*y.x + y.y*y.y;
        }
        #pragma unroll
        for (int off2=8; off2>0; off2>>=1){
          aq += __shfl_xor_sync(0xffffffff, aq, off2);
          ak += __shfl_xor_sync(0xffffffff, ak, off2);
        }
        __nv_bfloat162 nq2 = __float2bfloat162_rn(rsqrtf(aq+1e-6f)*scale);
        __nv_bfloat162 nk2 = __float2bfloat162_rn(rsqrtf(ak+1e-6f));
        #pragma unroll
        for (int j=0;j<4;j++){ pq[j]=__hmul2(pq[j],nq2); pk[j]=__hmul2(pk[j],nk2); }
        qreg[rr] = a; kreg[rr] = b;
      }
    }
    __syncthreads();
    // ---- in-sub-block exclusive cumsum of g, written back over sG (threads 0..511) ----
    {
      float acc = 0.f, gfirst = 0.f;
      #pragma unroll
      for (int t=0;t<16;t++){
        int r = cb*16+t;
        float gv = sG[r*LDG+cd];
        if (t==0) gfirst = gv; else acc += gv;
        sG[r*LDG+cd] = acc;
      }
      sSb[cb*DK+cd] = acc + gfirst;
      sGf[cb*DK+cd] = gfirst;
    }
    __syncthreads();
    // ---- vectorized qe/ke/kh scaling (all threads) + block scalars + cross ----
    {
      #pragma unroll
      for (int rr=0; rr<2; ++rr){
        const int row = lrow + rr*32;
        uint4 qa = qreg[rr];
        uint4 ka = kreg[rr];
        uint4 ga = *(const uint4*)(&sG[row*LDG+lc8]);
        uint4 gb = *(const uint4*)(&sG[row*LDG+lc8+4]);
        const float* gf = (const float*)&ga; const float* gh = (const float*)&gb;
        __nv_bfloat162 qo[4], eo[4], ko[4];
        const __nv_bfloat162* pq=(const __nv_bfloat162*)&qa;
        const __nv_bfloat162* pk=(const __nv_bfloat162*)&ka;
        #pragma unroll
        for (int j=0;j<4;j++){
          float y0 = (j<2? gf[2*j] : gh[2*j-4])*L2E;
          float y1 = (j<2? gf[2*j+1] : gh[2*j-3])*L2E;
          __nv_bfloat162 e  = __floats2bfloat162_rn(ex2(y0), ex2(y1));
          __nv_bfloat162 ie = __floats2bfloat162_rn(ex2(-y0), ex2(-y1));
          qo[j] = __hmul2(pq[j], e);
          eo[j] = __hmul2(pk[j], e);
          ko[j] = __hmul2(pk[j], ie);
        }
        qo2[rr] = *(const uint4*)qo; eo2[rr] = *(const uint4*)eo; ko2[rr] = *(const uint4*)ko;
        *(uint4*)(&sQ [row*LDB+lc8]) = qo2[rr];
        *(uint4*)(&sKE[row*LDB+lc8]) = eo2[rr];
        *(uint4*)(&sK [row*LDB+lc8]) = ko2[rr];
      }

      int d = tid & (DK-1);
      float pp = 0.f, base[4];
      #pragma unroll
      for (int b=0;b<4;b++){ base[b] = pp + sGf[b*DK+d]; pp += sSb[b*DK+d]; }
      // All 512 threads already computed base[]/pp for their own d = tid&127, so the four
      // threads sharing a d are holding identical values. Previously only threads 0..127
      // wrote sCq/sCk -- 8 serialized ex2+cvt+STS while the other 384 threads idled at
      // the barrier. Split the four b-values across those four threads (b = tid>>7) so
      // each writes one sCq and one sCk: the phase's dependent chain drops 8 -> 2.
      {
        const int b = tid>>7;
        if (b==0) sGL[d] = ex2(pp*L2E);
        sCq[b*DK+d] = __float2bfloat16(ex2(base[b]*L2E));
        sCk[b*DK+d] = __float2bfloat16(ex2((pp-base[b])*L2E));
      }
      #pragma unroll
      for (int n=0;n<(4*4*DK)/NTHREADS;n++){
        int i = tid + n*NTHREADS;
        int bj=(i>>7)&3, bc=i>>9;
        sCr[i] = __float2bfloat16((bc>=bj)? ex2((base[bc]-base[bj])*L2E) : 0.f);
      }
    }
    __syncthreads();
    // ---- Aqk / Akk : 6 tiles, each producing both matrices from one shared B fragment ----
    for (int t=warp; t<6; t+=NW){
      const int bc = (t==0)?0:((t==1)?1:((t<4)?2:3));
      const int nb = (t==3||t==5)?1:0;
      const int cx = (lane&3)*2;
      float acc[2][4][4];
      #pragma unroll
      for (int q2=0;q2<2;q2++)
        #pragma unroll
        for (int i=0;i<4;i++)
          #pragma unroll
          for (int e=0;e<4;e++) acc[q2][i][e]=0.f;
      // Hoist the cross-block decay factors out of the k-loop. Each warp touches only the
      // fixed slice sCr[bc][np][*] and its lane-constant column offset cx, so the 32
      // LDS.32 these did across the 8 k-steps are loop-invariant per (hh, kk) pair --
      // 16 scalar shared loads per warp per chunk that L1 was serving for nothing.
      // Holding them as 16 registers turns the inner loop into pure ldmatrix + mma.
      uint32_t crlo[2][8], crhi[2][8];
      #pragma unroll
      for (int hh=0; hh<2; hh++){
        const bf16* cr = sCr + bc*512 + (nb*2+hh)*DK;
        #pragma unroll
        for (int s=0; s<8; s++){
          crlo[hh][s] = *(const uint32_t*)(&cr[s*16 + cx]);
          crhi[hh][s] = *(const uint32_t*)(&cr[s*16 + 8 + cx]);
        }
      }
      for (int kk=0; kk<DK; kk+=16){
        uint32_t aq[4], ak[4];
        ld_a(aq, sQ,  LDB, bc*16, kk, lane);
        ld_a(ak, sKE, LDB, bc*16, kk, lane);
        #pragma unroll
        for (int hh=0; hh<2; hh++){
          const int np = nb*2 + hh;
          uint32_t b4[4];
          ld_b2(b4, sK, LDB, np*16, kk, lane);
          uint32_t clo = crlo[hh][kk>>4];
          uint32_t chi = crhi[hh][kk>>4];
          __nv_bfloat162 c0b = *(__nv_bfloat162*)&clo, c1b = *(__nv_bfloat162*)&chi;
          *(__nv_bfloat162*)&b4[0] = __hmul2(*(__nv_bfloat162*)&b4[0], c0b);
          *(__nv_bfloat162*)&b4[1] = __hmul2(*(__nv_bfloat162*)&b4[1], c1b);
          *(__nv_bfloat162*)&b4[2] = __hmul2(*(__nv_bfloat162*)&b4[2], c0b);
          *(__nv_bfloat162*)&b4[3] = __hmul2(*(__nv_bfloat162*)&b4[3], c1b);
          uint32_t b0[2]={b4[0],b4[1]}, b1[2]={b4[2],b4[3]};
          mma1(acc[1][hh*2],   aq, b0);
          mma1(acc[1][hh*2+1], aq, b1);
          mma1(acc[0][hh*2],   ak, b0);
          mma1(acc[0][hh*2+1], ak, b1);
        }
      }
      // -beta depends only on the row, i.e. only on h2, so the fully-unrolled 4x2
      // epilogue was issuing 8 LDS.32 for 2 distinct values. Hoist them.
      const float nbe[2] = { -sBe[bc*16 + (lane>>2)], -sBe[bc*16 + (lane>>2) + 8] };
      #pragma unroll
      for (int j=0;j<4;j++)
      #pragma unroll
      for (int h2=0;h2<2;h2++){
        int rr = bc*16 + (lane>>2) + 8*h2;
        int cc = nb*32 + j*8 + (lane&3)*2;
        float q0 = acc[1][j][2*h2], q1 = acc[1][j][2*h2+1];
        *(__nv_bfloat162*)(&sAQ[rr*LDS+cc]) =
            __floats2bfloat162_rn((rr>=cc)? q0:0.f, (rr>=cc+1)? q1:0.f);
        float nb2 = nbe[h2];
        float k0 = acc[0][j][2*h2], k1 = acc[0][j][2*h2+1];
        *(__nv_bfloat162*)(&sA[rr*LDA+cc]) =
            __floats2bfloat162_rn((rr>cc)? nb2*k0:0.f, (rr>cc+1)? nb2*k1:0.f);
      }
    }
    __syncthreads();
    // ---- store v ----
    *(uint4*)(&sV[lrow*LDB+lc8]) = vpf[0];
    *(uint4*)(&sV[(lrow+32)*LDB+lc8]) = vpf[1];
    // ---- qg, kgc, kg ----
    {
      const int c8 = lc8;
      #pragma unroll
      for (int rr=0; rr<2; ++rr){
        int row = lrow + rr*32, b = row>>4;
        uint4 cqv = *(const uint4*)(&sCq[b*DK+c8]);
        uint4 ckv = *(const uint4*)(&sCk[b*DK+c8]);
        const __nv_bfloat162* pcq=(const __nv_bfloat162*)&cqv;
        const __nv_bfloat162* pck=(const __nv_bfloat162*)&ckv;
        uint4 a = qo2[rr], b2 = eo2[rr], c2 = ko2[rr];
        __nv_bfloat162* pa=(__nv_bfloat162*)&a; __nv_bfloat162* pb=(__nv_bfloat162*)&b2; __nv_bfloat162* pc=(__nv_bfloat162*)&c2;
        #pragma unroll
        for (int z=0;z<4;z++){ pa[z]=__hmul2(pa[z],pcq[z]); pb[z]=__hmul2(pb[z],pcq[z]); pc[z]=__hmul2(pc[z],pck[z]); }
        *(uint4*)(&sQ [row*LDB+c8]) = a;
        *(uint4*)(&sKE[row*LDB+c8]) = b2;
        *(uint4*)(&sK [row*LDB+c8]) = c2;
      }
    }
    __syncthreads();
    // ---- warps 8-15: P1 (Z = v - kgc@S into sV, and o1 = qg@S) ; warps 0-3: WY inverse ----
    const int pmat = (warp>=8);
    const int prm  = ((warp>>2)&1)*32;     // row block of the 64-row chunk
    const int pcn  = (warp&3)*32;          // column block of DV
    float acc[2][4][4];                    // o accumulator, lives until the sU write
    #pragma unroll
    for (int mi=0;mi<2;mi++)
      #pragma unroll
      for (int p=0;p<4;p++)
        #pragma unroll
        for (int e=0;e<4;e++) acc[mi][p][e]=0.f;
    if (pmat){
      float az[2][4][4];
      #pragma unroll
      for (int mi=0;mi<2;mi++)
        #pragma unroll
        for (int p=0;p<4;p++)
          #pragma unroll
          for (int e=0;e<4;e++) az[mi][p][e]=0.f;
      for (int kk=0; kk<DK; kk+=16){
        uint32_t ba[4], bb[4];
        ld_b2_t(ba, sS, LDB, pcn,    kk, lane);
        ld_b2_t(bb, sS, LDB, pcn+16, kk, lane);
        uint32_t b0[2]={ba[0],ba[1]}, b1[2]={ba[2],ba[3]};
        uint32_t b2[2]={bb[0],bb[1]}, b3[2]={bb[2],bb[3]};
        #pragma unroll
        for (int mi=0;mi<2;mi++){
          uint32_t ae[4]; ld_a(ae, sKE, LDB, prm+mi*16, kk, lane);
          mma1(az[mi][0], ae, b0);
          mma1(az[mi][1], ae, b1);
          mma1(az[mi][2], ae, b2);
          mma1(az[mi][3], ae, b3);
          uint32_t aq[4]; ld_a(aq, sQ,  LDB, prm+mi*16, kk, lane);
          mma1(acc[mi][0], aq, b0);
          mma1(acc[mi][1], aq, b1);
          mma1(acc[mi][2], aq, b2);
          mma1(acc[mi][3], aq, b3);
        }
      }
      #pragma unroll
      for (int mi=0;mi<2;mi++){
        int rb = prm + mi*16 + (lane>>2);
        float be0 = sBe[rb], be1 = sBe[rb+8];
        #pragma unroll
        for (int p=0;p<4;p++){
          int c0 = pcn + p*8 + (lane&3)*2;
          float2 v0 = __bfloat1622float2(*(const __nv_bfloat162*)(&sV[rb*LDB+c0]));
          float2 v1 = __bfloat1622float2(*(const __nv_bfloat162*)(&sV[(rb+8)*LDB+c0]));
          *(__nv_bfloat162*)(&sV[rb*LDB+c0])     = __floats2bfloat162_rn(be0*(v0.x-az[mi][p][0]), be0*(v0.y-az[mi][p][1]));
          *(__nv_bfloat162*)(&sV[(rb+8)*LDB+c0]) = __floats2bfloat162_rn(be1*(v1.x-az[mi][p][2]), be1*(v1.y-az[mi][p][3]));
        }
      }
    } else if (warp<4){
    // ---- WY: 4 diagonal 16x16 inversions via tf32 mma ----
    // L_ii = (I-N)^-1 = (I+N)(I+N^2)(I+N^4)(I+N^8), N = A_ii strictly lower (N^16=0)
    if (warp<4){
      int base = warp*16;
      bf16* Pb = sWY + warp*(2*16*LDW);
      bf16* Qb = Pb + 16*LDW;
      #pragma unroll
      for (int idx=lane; idx<128; idx+=32){
        int r=idx>>3, c=(idx&7)*2;
        __nv_bfloat162 n = *(const __nv_bfloat162*)(&sA[(base+r)*LDA+base+c]);
        *(__nv_bfloat162*)(&Qb[r*LDW+c]) = n;
        float2 f = __bfloat1622float2(n);
        *(__nv_bfloat162*)(&Pb[r*LDW+c]) =
            __floats2bfloat162_rn(f.x + ((r==c)?1.f:0.f), f.y + ((r==c+1)?1.f:0.f));
      }
      __syncwarp();
      #pragma unroll
      for (int st=0; st<3; ++st){
        float q2[2][4]; zero16(q2);
        mm16b(q2, Qb, LDW, Qb, LDW, lane);
        __syncwarp();
        st16b(q2, Qb, LDW, lane);
        __syncwarp();
        float pq[2][4]; zero16(pq);
        mm16b(pq, Pb, LDW, Qb, LDW, lane);
        __syncwarp();
        int r0 = lane>>2, c0 = (lane&3)*2;
        #pragma unroll
        for (int t=0;t<2;t++){
          float2 a0 = __bfloat1622float2(*(const __nv_bfloat162*)(&Pb[r0*LDW+t*8+c0]));
          float2 a1 = __bfloat1622float2(*(const __nv_bfloat162*)(&Pb[(r0+8)*LDW+t*8+c0]));
          *(__nv_bfloat162*)(&Pb[r0*LDW+t*8+c0])     = __floats2bfloat162_rn(a0.x+pq[t][0], a0.y+pq[t][1]);
          *(__nv_bfloat162*)(&Pb[(r0+8)*LDW+t*8+c0]) = __floats2bfloat162_rn(a1.x+pq[t][2], a1.y+pq[t][3]);
        }
        __syncwarp();
      }
      #pragma unroll
      for (int idx=lane; idx<128; idx+=32){
        int r=idx>>3, c=(idx&7)*2;
        *(__nv_bfloat162*)(&sX[(base+r)*LDA+base+c]) = *(const __nv_bfloat162*)(&Pb[r*LDW+c]);
      }
    }
      asm volatile("bar.sync 1, 128;");
    // ---- WY: block forward substitution, 3 stages, tf32 mma ----
    // X_ij = L_ii * ( sum_{k=j}^{i-1} A_ik * L_kj ),  L_jj on the diagonal
    {
      bf16* Yb = sWY + (warp&3)*(2*16*LDW);
      #pragma unroll
      for (int st=0; st<3; st++){
        int bj = warp, bi = warp + st + 1;
        if (warp < 3-st){
          float acc[2][4]; zero16(acc);
          for (int kb=bj; kb<bi; ++kb)
            mm16b(acc, &sA[(bi*16)*LDA + kb*16], LDA, &sX[(kb*16)*LDA + bj*16], LDA, lane);
          __syncwarp();
          st16b(acc, Yb, LDW, lane);
          __syncwarp();
          float acc2[2][4]; zero16(acc2);
          mm16b(acc2, &sX[(bi*16)*LDA + bi*16], LDA, Yb, LDW, lane);
          st16b(acc2, &sX[(bi*16)*LDA + bj*16], LDA, lane);
        }
        if (st == 0 && warp < 3) asm volatile("bar.sync 1, 96;");
        if (st == 1 && warp < 2) asm volatile("bar.sync 1, 64;");
        if (st == 2 && warp < 1) asm volatile("bar.sync 1, 32;");
      }
    }
    }
    __syncthreads();
    // ---- P2: vnew = twy @ Z -> sKE (all 16 warps, 16x32 output tiles) ----
    {
      // sX (the WY inverse T) is block-lower-triangular: its upper 16x16 blocks are
      // zeroed once before the chunk loop and never written, so k-blocks past m0
      // contribute nothing. Stop at kk<=m0 (10 of 16 block-mmas instead of 16).
      // Row block from warp>>2 so the four warps sharing a scheduler (selected by
      // warp&3) get the four different row costs 1,2,3,4 rather than four copies of
      // one cost -- that balances the triangular work across the four schedulers.
      int m0 = (warp>>2)*16, n0 = (warp&3)*32;
      float a2[4][4];
      #pragma unroll
      for (int i=0;i<4;i++)
        #pragma unroll
        for (int e=0;e<4;e++) a2[i][e]=0.f;
      for (int kk=0; kk<=m0; kk+=16){
        uint32_t af[4]; ld_a(af, sX, LDA, m0, kk, lane);
        uint32_t ba[4], bb[4];
        ld_b2_t(ba, sV, LDB, n0,    kk, lane);
        ld_b2_t(bb, sV, LDB, n0+16, kk, lane);
        uint32_t b0[2]={ba[0],ba[1]}, b1[2]={ba[2],ba[3]};
        uint32_t b2[2]={bb[0],bb[1]}, b3[2]={bb[2],bb[3]};
        mma1(a2[0], af, b0); mma1(a2[1], af, b1);
        mma1(a2[2], af, b2); mma1(a2[3], af, b3);
      }
      #pragma unroll
      for (int p=0;p<4;p++){
        int r0 = m0 + (lane>>2), c0 = n0 + p*8 + (lane&3)*2;
        *(__nv_bfloat162*)(&sKE[r0*LDB+c0])     = __floats2bfloat162_rn(a2[p][0],a2[p][1]);
        *(__nv_bfloat162*)(&sKE[(r0+8)*LDB+c0]) = __floats2bfloat162_rn(a2[p][2],a2[p][3]);
      }
    }
    __syncthreads();
    // ---- o += Aqk @ vnew (lower-triangular) on warps 8-15 ----
    if (pmat){
      for (int kk=0; kk<=prm+16; kk+=16){
        uint32_t ba[4], bb[4];
        ld_b2_t(ba, sKE, LDB, pcn,    kk, lane);
        ld_b2_t(bb, sKE, LDB, pcn+16, kk, lane);
        uint32_t b0[2]={ba[0],ba[1]}, b1[2]={ba[2],ba[3]};
        uint32_t b2[2]={bb[0],bb[1]}, b3[2]={bb[2],bb[3]};
        #pragma unroll
        for (int mi=0;mi<2;mi++){
          int m0 = prm + mi*16;
          if (kk<=m0){
            uint32_t af[4]; ld_a(af, sAQ, LDS, m0, kk, lane);
            mma1(acc[mi][0], af, b0);
            mma1(acc[mi][1], af, b1);
            mma1(acc[mi][2], af, b2);
            mma1(acc[mi][3], af, b3);
          }
        }
      }
    }
    // ---- S = S*glast + kg^T @ vnew ----
    {
      #pragma unroll
      for (int mi=0;mi<2;mi++){
        float gl0 = sGL[sm_row0 + mi*16 + (lane>>2)], gl1 = sGL[sm_row0 + mi*16 + (lane>>2) + 8];
        #pragma unroll
        for (int i=0;i<4;i++){
          Sacc[mi][i][0] *= gl0; Sacc[mi][i][1] *= gl0;
          Sacc[mi][i][2] *= gl1; Sacc[mi][i][3] *= gl1;
        }
      }
      for (int kk=0; kk<BT; kk+=16){
        uint32_t ba[4], bb[4];
        ld_b2_t(ba, sKE, LDB, sm_col0,    kk, lane);
        ld_b2_t(bb, sKE, LDB, sm_col0+16, kk, lane);
        uint32_t b0[2]={ba[0],ba[1]}, b1[2]={ba[2],ba[3]};
        uint32_t b2[2]={bb[0],bb[1]}, b3[2]={bb[2],bb[3]};
        #pragma unroll
        for (int mi=0;mi<2;mi++){
          uint32_t af[4]; ld_a_t(af, sK, LDB, sm_row0+mi*16, kk, lane);
          mma1(Sacc[mi][0], af, b0);
          mma1(Sacc[mi][1], af, b1);
          mma1(Sacc[mi][2], af, b2);
          mma1(Sacc[mi][3], af, b3);
        }
      }
    }
    // ---- write sS and o staging ----
    {
      #pragma unroll
      for (int mi=0;mi<2;mi++){
        int sr0 = sm_row0 + mi*16 + (lane>>2);
        #pragma unroll
        for (int p=0;p<4;p++){
          int c0 = sm_col0 + p*8 + (lane&3)*2;
          *(__nv_bfloat162*)(&sS[sr0*LDB+c0])     = __floats2bfloat162_rn(Sacc[mi][p][0],Sacc[mi][p][1]);
          *(__nv_bfloat162*)(&sS[(sr0+8)*LDB+c0]) = __floats2bfloat162_rn(Sacc[mi][p][2],Sacc[mi][p][3]);
        }
      }
      // o goes straight from the mma accumulators to global, but first the four
      // accumulator fragments of a row are transposed across each quad of lanes.
      // The m16n8k16 layout hands lane L only 4B (one bf162) per fragment at column
      // p*8+(L&3)*2, so a naive store covers 8 rows x 16B -- half of every 32B sector.
      // NCU measured 6.98M excessive sectors (16% of all traffic) from exactly this.
      // A 4x4 transpose over (fragment index p, lane&3) -- two shfl_xor butterfly
      // stages, entirely within the quad so the row (lane>>2) is untouched -- gives
      // each lane 8 consecutive columns. Stores drop 16 -> 4 per warp and every
      // sector is fully written.
      if (pmat){
        bf16* opd = o + (long)(t0)*rstride + (long)h*DK;
        const unsigned FM = 0xffffffffu;
        #pragma unroll
        for (int mi=0;mi<2;mi++)
        #pragma unroll
        for (int hf=0; hf<2; hf++){
          uint32_t a0,a1,a2,a3;
          { __nv_bfloat162 t;
            t = __floats2bfloat162_rn(acc[mi][0][2*hf],acc[mi][0][2*hf+1]); a0 = *(uint32_t*)&t;
            t = __floats2bfloat162_rn(acc[mi][1][2*hf],acc[mi][1][2*hf+1]); a1 = *(uint32_t*)&t;
            t = __floats2bfloat162_rn(acc[mi][2][2*hf],acc[mi][2][2*hf+1]); a2 = *(uint32_t*)&t;
            t = __floats2bfloat162_rn(acc[mi][3][2*hf],acc[mi][3][2*hf+1]); a3 = *(uint32_t*)&t; }
          // stage 1: swap lane bit0 with fragment bit0
          uint32_t s0 = (lane&1)? a0 : a1, s1 = (lane&1)? a2 : a3;
          s0 = __shfl_xor_sync(FM, s0, 1);
          s1 = __shfl_xor_sync(FM, s1, 1);
          if (lane&1){ a0 = s0; a2 = s1; } else { a1 = s0; a3 = s1; }
          // stage 2: swap lane bit1 with fragment bit1
          uint32_t s2 = (lane&2)? a0 : a2, s3 = (lane&2)? a1 : a3;
          s2 = __shfl_xor_sync(FM, s2, 2);
          s3 = __shfl_xor_sync(FM, s3, 2);
          if (lane&2){ a0 = s2; a1 = s3; } else { a2 = s2; a3 = s3; }
          int r0 = prm + mi*16 + (lane>>2) + 8*hf, c0 = pcn + (lane&3)*8;
          if (r0 < nrows) *(uint4*)(opd + (long)r0*rstride + c0) = make_uint4(a0,a1,a2,a3);
        }
      }
    }
  }
}

cudaError_t kda_launch(const void* q, const void* k, const void* v, const void* g, const void* beta,
                       const void* cu, void* o, int H, int ndoc, cudaStream_t stream)
{
  int off = 5*SZ128 + SZS + SZA + SZ64 + (16+4+4)*DK*2 + (1+4+4)*DK*4 + BT*4 + 4*2*16*LDW*2;
  static bool init = false;
  static cudaError_t init_err = cudaSuccess;
  if (!init){ init_err = cudaFuncSetAttribute(kda_fused, cudaFuncAttributeMaxDynamicSharedMemorySize, off); init = true; }
  if (init_err != cudaSuccess) return init_err;
  dim3 grid(H, ndoc);
  kda_fused<<<grid, NTHREADS, off, stream>>>((const bf16*)q,(const bf16*)k,(const bf16*)v,
      (const float*)g,(const float*)beta,(const int*)cu,(bf16*)o,H);
  return cudaSuccess;
}
"""

BINDING = r"""// Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Fused chunked Kimi Delta Attention forward for the Kimi-K3 call. One CTA per (head, document)
// walks the 64-token chunks with the [K, V] state on-chip (mma.sync + ldmatrix, sm_80+). Specialised to head dims
// K = V = 128, a packed layout (batch 1 with int32 cu_seqlens), q/k l2-normalised in kernel, a bounded log gate, no
// initial/final state. Measured on GB200 against FLA 0.4.2 chunk_kda at the Kimi-K3 layer-microbatch shape
// (8192 tokens = 4 documents, 96 heads): 0.78-0.80 ms vs 2.07-2.19 ms, max |diff| 1-2 bf16 ulp.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>

namespace {
constexpr int64_t kHeadDim = 128;

void check_inputs(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& g,
                  const torch::Tensor& beta, const torch::Tensor& cu_seqlens, const torch::Tensor& o) {
  TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda() && g.is_cuda() && beta.is_cuda() && cu_seqlens.is_cuda() &&
                  o.is_cuda(),
              "chunk_kda_fwd: all tensors must live on the GPU");
  TORCH_CHECK(q.dim() == 4 && q.size(0) == 1, "chunk_kda_fwd: q must be [1, T, H, K] (packed batch of 1), got ",
              q.sizes());
  TORCH_CHECK(k.sizes() == q.sizes() && v.sizes() == q.sizes() && g.sizes() == q.sizes(),
              "chunk_kda_fwd: k, v, g must match q's shape [1, T, H, 128]; got k ", k.sizes(), " v ", v.sizes(), " g ",
              g.sizes());
  TORCH_CHECK(q.size(3) == kHeadDim, "chunk_kda_fwd: head dims must be K = V = 128, got ", q.size(3));
  TORCH_CHECK(beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == q.size(1) && beta.size(2) == q.size(2),
              "chunk_kda_fwd: beta must be [1, T, H], got ", beta.sizes());
  TORCH_CHECK(o.sizes() == v.sizes(), "chunk_kda_fwd: o must match v's shape, got ", o.sizes());
  TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                  v.scalar_type() == at::kBFloat16 && o.scalar_type() == at::kBFloat16,
              "chunk_kda_fwd: q, k, v, o must be bfloat16");
  TORCH_CHECK(g.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat,
              "chunk_kda_fwd: g and beta must be float32");
  TORCH_CHECK(cu_seqlens.scalar_type() == at::kInt && cu_seqlens.dim() == 1 && cu_seqlens.numel() >= 2,
              "chunk_kda_fwd: cu_seqlens must be a 1-D int32 tensor of at least two offsets");
  TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous() && g.is_contiguous() &&
                  beta.is_contiguous() && cu_seqlens.is_contiguous() && o.is_contiguous(),
              "chunk_kda_fwd: all tensors must be contiguous");
}
}  // namespace

void run(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
         const torch::Tensor& g, const torch::Tensor& beta, const torch::Tensor& cu_seqlens,
         torch::Tensor& o) {
  check_inputs(q, k, v, g, beta, cu_seqlens, o);
  const c10::cuda::OptionalCUDAGuard guard(q.device());
  int H = (int)q.size(2);
  int ndoc = (int)cu_seqlens.size(0) - 1;
  cudaError_t err = kda_launch(q.data_ptr(), k.data_ptr(), v.data_ptr(), g.data_ptr(), beta.data_ptr(),
                               cu_seqlens.data_ptr(), o.data_ptr(), H, ndoc,
                               at::cuda::getCurrentCUDAStream().stream());
  TORCH_CHECK(err == cudaSuccess, "chunk_kda_fwd: cudaFuncSetAttribute(MaxDynamicSharedMemorySize) failed: ",
              cudaGetErrorString(err));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "chunk_kda forward (CUDA, K = V = 128, packed batch of 1)");
}
"""

CUDA_SOURCES = (HEADER, KERNEL)
CPP_SOURCES = (HEADER, BINDING)
