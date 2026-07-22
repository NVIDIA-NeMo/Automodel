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

"""GPU CP forward-equivalence for the MiniMax M3 VL in-forward pre-embed shard (L1, 2 GPUs).

Exercises the sunk CP path end to end: ``ContextParallelSharder`` invokes the
sharder-only ``prepare_model_inputs_for_cp`` hook (``shard_batch_aux_only``
round-robin shards labels/position_ids and installs the ring-SDPA context), and
``MiniMaxM3SparseForConditionalGeneration.forward`` embeds the full sequence then
``shard_sequence_for_cp_round_robin`` shards ``inputs_embeds`` per rank. The unsharded cp2
logits must match the cp1 eager forward.

  cp1 eager (cp_mesh unset)  ==  cp2 (apply_cp + ContextParallelSharder + in-forward shard)

Text-only batch, dense layers (sparse disabled) so torch ``context_parallel`` ring
SDPA is the transport. bf16 (the dense CP SDPA kernel has no fp32 path); per-token
argmax agreement is the leakage-robust signal.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_minimax_m3_vl_cp_sink.py
"""

import os
import sys

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    device = torch.device(f"cuda:{lr}")
    cp_size = world

    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor.experimental._attention import context_parallel_unshard

    from nemo_automodel.components.distributed.cp_sharder import ContextParallelSharder
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration
    from nemo_automodel.components.moe.parallelizer import apply_cp

    tiny = dict(
        hidden_size=64, intermediate_size=32, dense_intermediate_size=48, shared_intermediate_size=32,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16, rotary_dim=8,
        partial_rotary_factor=0.5, vocab_size=128, max_position_embeddings=512, rms_norm_eps=1e-6,
        rope_theta=10000.0, num_local_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        moe_layer_freq=[0, 1, 1], use_gemma_norm=True, use_qk_norm=True, qk_norm_type="per_head",
        scoring_func="sigmoid", use_routing_bias=True, routed_scaling_factor=2.0, swiglu_alpha=1.702,
        swiglu_limit=7.0, num_mtp_modules=0, sparse_attention_config=dict(use_sparse_attention=False),
    )
    vision = dict(
        hidden_size=32, num_attention_heads=4, num_hidden_layers=2, intermediate_size=64, patch_size=2,
        num_channels=3, rope_theta=10000.0, hidden_act="gelu", layer_norm_eps=1e-5,
        img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2},
    )
    backend = BackendConfig(linear="torch", attn="sdpa", rms_norm="torch", rope_fusion=False,
                            dispatcher="torch", fake_balanced_gate=False, enable_hf_state_dict_adapter=False)

    torch.manual_seed(0)
    cfg = MiniMaxM3VLConfig(vision_config=dict(vision), text_config={**tiny, "torch_dtype": "bfloat16"},
                            image_token_index=100, video_token_index=101, projector_hidden_size=tiny["hidden_size"])
    model = MiniMaxM3SparseForConditionalGeneration(cfg, backend=backend).eval().to(device)
    model.initialize_weights(dtype=torch.bfloat16)
    model.to(torch.bfloat16)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    seqlen = 32 * cp_size  # divisible by 2*cp -> no pad
    input_ids = torch.randint(2, cfg.text_config.vocab_size, (1, seqlen), device=device)
    dist.broadcast(input_ids, src=0)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)

    # cp1 eager (no CP mesh installed): full-sequence forward.
    with torch.no_grad():
        logits_eager = model(input_ids=input_ids, position_ids=position_ids.clone()).float()

    # cp2: the recipe path -- apply_cp installs model.cp_mesh, ContextParallelSharder runs
    # the sharder-only hook, forward embeds then in-forward shards inputs_embeds.
    device_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    apply_cp(model, device_mesh["cp"])
    batch = {"input_ids": input_ids.clone(), "labels": input_ids.clone(), "position_ids": position_ids.clone()}
    cp_sharder = ContextParallelSharder(model, device_mesh, batch)
    train_ctx, batch = cp_sharder.shard(batch)
    batch.pop("labels", None)
    with torch.no_grad(), train_ctx():
        logits_local = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"]).float()
    (logits_cp_full,) = context_parallel_unshard(device_mesh["cp"], [logits_local], seq_dims=[1])
    logits_cp_full = logits_cp_full[:, :seqlen, :]

    rc = 0
    if rank == 0:
        diff = (logits_cp_full - logits_eager).abs()
        top1 = (logits_cp_full.argmax(-1) == logits_eager.argmax(-1)).float().mean().item()
        ce = torch.nn.functional.cross_entropy
        loss_eager = ce(logits_eager[0, :-1], input_ids[0, 1:]).item()
        loss_cp = ce(logits_cp_full[0, :-1], input_ids[0, 1:]).item()
        print(f"\n{'=' * 72}\nMiniMax M3 VL in-forward CP sink  cp_size={cp_size}  seqlen={seqlen}\n{'=' * 72}")
        print(f"logits mean_abs={diff.mean().item():.3e} max_abs={diff.max().item():.3e} top1={top1:.4f}")
        print(f"loss eager={loss_eager:.6f} cp={loss_cp:.6f} |dloss|={abs(loss_eager - loss_cp):.3e}")
        ok = top1 > 0.98 and abs(loss_eager - loss_cp) < 5e-2
        print("RESULT:", "PASS (in-forward CP == eager)" if ok else "FAIL")
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
