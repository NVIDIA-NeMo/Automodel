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

"""Two-GPU TileLang CP parity against the legacy dense-mask path.

Run with torchrun --standalone --nproc-per-node=2. Covers packed documents
crossing rank boundaries, padding keys, SWA/SCA/HCA, outputs and gradients.
"""

import copy
import json
import os

import torch
import torch.distributed as dist

from nemo_automodel.components.models.common import BackendConfig
from nemo_automodel.components.models.deepseek_v4.config import DeepseekV4Config
from nemo_automodel.components.models.deepseek_v4.cp import (
    build_dsv4_cp_causal_padding_mask,
    build_dsv4_cp_packed_causal_padding_mask,
)
from nemo_automodel.components.models.deepseek_v4.layers import DeepseekV4Attention, DeepseekV4RotaryEmbedding
from nemo_automodel.components.models.deepseek_v4.sparse_mask import build_dsv4_sparse_mask


def main() -> None:
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 2
    for ratio in (0, 4, 128):
        for packed in (False, True):
            torch.manual_seed(119)
            cfg = DeepseekV4Config(
                hidden_size=64,
                num_hidden_layers=1,
                num_attention_heads=16,
                head_dim=128,
                qk_rope_head_dim=32,
                q_lora_rank=32,
                o_lora_rank=32,
                o_groups=2,
                compress_ratios=[ratio],
                sliding_window=32,
                index_head_dim=128,
                index_n_heads=16,
                index_topk=8,
                max_position_embeddings=512,
                torch_dtype="bfloat16",
            )
            backend = BackendConfig(
                attn="tilelang",
                linear="torch",
                rms_norm="torch",
                rope_fusion=False,
                experts="torch",
                dispatcher="torch",
            )
            module = DeepseekV4Attention(cfg, 0, backend).cuda()
            module.init_weights(torch.device("cuda"))
            module.bfloat16().train()
            module.sinks_param.float()
            if module.compressor is not None:
                for c in (module.compressor, module.compressor.indexer):
                    if c is not None:
                        c.wkv.float()
                        c.wgate.float()
                        c.ape_param.float()
                if ratio == 4:
                    module.compressor.indexer.backend = BackendConfig(attn="sdpa")
            reference = copy.deepcopy(module)
            full_x = torch.randn(1, 512, 64, device="cuda", dtype=torch.bfloat16)
            x = full_x[:, rank * 256 : (rank + 1) * 256].clone().requires_grad_()
            xr = x.detach().clone().requires_grad_()
            full_pos = torch.arange(512, device="cuda")[None]
            full_docs = None
            if packed:
                full_docs = torch.ones_like(full_pos)
                full_docs[:, 128:] = 2
                full_pos[:, 128:] -= 128
            positions = full_pos[:, rank * 256 : (rank + 1) * 256]
            docs = full_docs[:, rank * 256 : (rank + 1) * 256] if packed else None
            padding = torch.zeros(1, 256, device="cuda", dtype=torch.bool)
            padding[:, 30:33] = True
            if packed:
                dense = build_dsv4_cp_packed_causal_padding_mask(
                    position_ids=positions,
                    packed_seq_ids=docs,
                    dtype=x.dtype,
                    device=x.device,
                    cp_group=dist.group.WORLD,
                    padding_mask=padding,
                    sliding_window=32,
                )
            else:
                dense = build_dsv4_cp_causal_padding_mask(
                    position_ids=positions,
                    key_len=512,
                    dtype=x.dtype,
                    device=x.device,
                    cp_group=dist.group.WORLD,
                    padding_mask=padding,
                    sliding_window=32,
                )
            compact = build_dsv4_sparse_mask(
                positions,
                key_len=512,
                packed_seq_ids=docs,
                padding_mask=padding,
                sliding_window=32,
                cp_group=dist.group.WORLD,
            )
            rotary = DeepseekV4RotaryEmbedding(rope_theta=10000.0, head_dim=128, partial_rotary_factor=0.25).cuda()
            rotary_c = DeepseekV4RotaryEmbedding(rope_theta=160000.0, head_dim=128, partial_rotary_factor=0.25).cuda()
            kwargs = dict(
                position_embeddings=rotary(x, positions),
                position_embeddings_compress=rotary_c(x, positions),
                rotary_compress=rotary_c,
                position_ids=positions,
                packed_seq_ids=docs,
                _dsv4_cp_group=dist.group.WORLD,
            )
            actual = module(x, sparse_mask=compact, **kwargs)[0]
            expected = reference(xr, attention_mask=dense, **kwargs)[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            dy = torch.randn_like(actual)
            actual.backward(dy)
            expected.backward(dy)
            pairs = [("input", x.grad, xr.grad)]
            pairs += [
                (name, a.grad, b.grad)
                for (name, a), (_, b) in zip(module.named_parameters(), reference.named_parameters())
            ]
            # Establish the same kernel's A/A backward envelope: its KV
            # accumulation uses atomics, then rounds the result to BF16.
            reference.zero_grad(set_to_none=True)
            xr.grad = None
            repeated = reference(xr, attention_mask=dense, **kwargs)[0]
            repeated.backward(dy)
            repeat_grads = {"input": xr.grad}
            repeat_grads.update({name: p.grad for name, p in reference.named_parameters()})
            worst = aa_worst = 0.0
            for name, a, b in pairs:
                assert (a is None) == (b is None), name
                if a is not None:
                    scale = b.float().norm().clamp_min(1e-12)
                    relative = ((a.float() - b.float()).norm() / scale).item()
                    aa = ((repeat_grads[name].float() - b.float()).norm() / scale).item()
                    assert torch.isfinite(a).all() and torch.isfinite(b).all(), name
                    assert relative < 0.004 and aa < 0.004, (name, relative, aa)
                    worst = max(worst, relative)
                    aa_worst = max(aa_worst, aa)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "ratio": ratio,
                        "packed": packed,
                        "max_grad_relative_l2": worst,
                        "aa_grad_relative_l2": aa_worst,
                    }
                ),
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
