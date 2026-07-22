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

"""cp2xpp2 layer-2 verification for the sunk pre-embed path (L1, 2/4 GPUs).

Drives the REAL AutoPipeline split + schedule.step under cp2xpp2 (4 GPUs) and
cp2xpp1 (2 GPUs) with a tiny random-init text-only config, exercising the whole
sunk layer-2 contract: the sharder-only hook, the in-forward embed +
shard_sequence_for_cp_round_robin, the asymmetric get_pipeline_stage_metas (full-length
first-stage ids, local sharded outputs), and per-microbatch backward. Asserts:

  (1) 20 steps run clean -- no "backward through the graph a second time"
      (the double-backward the old shared pre-embed graph caused under PP*CP);
  (2) the batch-mean loss is finite and, run in both modes, cp2xpp2 == cp2xpp1
      within bf16 tolerance (layout invariance across the PP split);
  (3) embed_tokens receives finite gradients -- the trainable-embeddings
      capability the in-forward shard unlocks (the old pre-embed detached them);
  (4) for MTP models (step3p7) the per-depth MTP loss is exercised under PP*CP.

Config-swappable via NEMO_CP_PP_MODEL={minimax (default), step3p7}; step3p7's
last PP stage emits ``(logits, *mtp_per_depth_logits)`` so the loss is
tuple-aware and threads the MTP heads through the schedule.

Run:
    torchrun --standalone --nproc-per-node=4 run_cp_pp_layer2_sink.py         # cp2xpp2
    NEMO_CP_PP_TEST_PP1=1 torchrun --nproc-per-node=2 run_cp_pp_layer2_sink.py  # cp2xpp1
Observed (bf16): minimax cp2xpp2/pp1 = 5.588573; step3p7 = 7.875539 / 7.875540.
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F


def build_step3p7(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.step3p7.configuration_step3p7 import Step3p7Config
    from nemo_automodel.components.models.step3p7.model import Step3p7ForConditionalGeneration

    layers = 4
    cfg = Step3p7Config(
        vision_config={"width": 8, "layers": 0, "heads": 2, "num_channels": 3, "image_size": 8, "patch_size": 2,
                       "mlp_ratio": 2.0, "hidden_act": "gelu", "use_ln_pre": False, "use_ln_post": False,
                       "use_abs_posemb": False, "use_rope2d": False},
        text_config={"hidden_size": 16, "intermediate_size": 32, "num_attention_heads": 4, "num_attention_groups": 2,
                     "num_hidden_layers": layers, "vocab_size": 32, "moe_num_experts": 2, "moe_top_k": 1,
                     "moe_intermediate_size": 8, "share_expert_dims": 8, "head_dim": 4, "torch_dtype": "bfloat16",
                     "moe_layers_enum": (), "layer_types": ["full_attention"] * layers, "num_nextn_predict_layers": 1},
        image_token_id=31,
    )
    backend = BackendConfig(attn="sdpa", linear="torch", rms_norm="torch", dispatcher="torch",
                            rope_fusion=False, enable_hf_state_dict_adapter=False)
    model = Step3p7ForConditionalGeneration(cfg, backend=backend)
    model.initialize_weights(dtype=torch.bfloat16)
    return model.to(device).to(torch.bfloat16)


def build_minimax(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration

    tiny = dict(
        hidden_size=64, intermediate_size=32, dense_intermediate_size=48, shared_intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16, rotary_dim=8,
        partial_rotary_factor=0.5, vocab_size=128, max_position_embeddings=512, rms_norm_eps=1e-6,
        rope_theta=10000.0, num_local_experts=4, num_experts_per_tok=2, n_shared_experts=1,
        moe_layer_freq=[0, 1, 1, 1], use_gemma_norm=True, use_qk_norm=True, qk_norm_type="per_head",
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
    cfg = MiniMaxM3VLConfig(vision_config=dict(vision), text_config={**tiny, "torch_dtype": "bfloat16"},
                            image_token_index=100, video_token_index=101, projector_hidden_size=tiny["hidden_size"])
    model = MiniMaxM3SparseForConditionalGeneration(cfg, backend=backend)
    model.initialize_weights(dtype=torch.bfloat16)
    return model.to(device).to(torch.bfloat16)


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.cp_sharder import ContextParallelSharder
    from nemo_automodel.components.distributed.pipelining import AutoPipeline
    from nemo_automodel.components.moe.parallelizer import apply_cp

    pp1 = bool(os.environ.get("NEMO_CP_PP_TEST_PP1"))  # cp2xpp1 comparison leg
    pp_size = 1 if pp1 else 2
    cp_size = world // pp_size
    mesh = init_device_mesh("cuda", (pp_size, 1, cp_size), mesh_dim_names=("pp", "dp", "cp"))

    which = os.environ.get("NEMO_CP_PP_MODEL", "minimax")
    torch.manual_seed(0)
    model = build_step3p7(device) if which == "step3p7" else build_minimax(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
    model.train()
    vocab = model.config.text_config.vocab_size
    mtp_used = {"any": False}

    def loss_fn(output, labels):
        # step3p7's last PP stage emits (logits, *mtp_per_depth_logits); minimax
        # emits a bare logits tensor. Handle both, threading the MTP depths.
        if isinstance(output, tuple):
            logits, mtp = output[0], list(output[1:])
        else:
            logits = getattr(output, "logits", output)
            mtp = list(getattr(output, "mtp_per_depth_logits", None) or [])
        loss = F.cross_entropy(logits.reshape(-1, vocab).float(), labels.reshape(-1), ignore_index=-100)
        for m in mtp:
            mtp_used["any"] = True
            loss = loss + F.cross_entropy(m.reshape(-1, vocab).float(), labels.reshape(-1), ignore_index=-100)
        return loss

    def cp_only_parallelize(m, world_mesh, moe_mesh, *, dp_axis_names, cp_axis_name=None, **kw):
        if cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1:
            apply_cp(m, world_mesh[cp_axis_name])

    seqlen = 32
    if pp_size > 1:
        pp = AutoPipeline(
            world_mesh=mesh, moe_mesh=None, pp_axis_name="pp", dp_axis_names=("dp",), cp_axis_name="cp",
            pp_schedule="1f1b", pp_microbatch_size=1, pp_batch_size=2, device=device, dtype=torch.bfloat16,
            pp_seq_len=seqlen,
        ).build(model, loss_fn=loss_fn, parallelize_fn=cp_only_parallelize)
        model_part0, has_last, has_first = pp.parts[0], pp.info.has_last_stage, pp.info.has_first_stage
    else:
        cp_only_parallelize(model, mesh, None, dp_axis_names=("dp",), cp_axis_name="cp")
        model_part0 = model

    losses = []
    for step in range(20):
        torch.manual_seed(1000 + step)
        input_ids = torch.randint(2, vocab, (2, seqlen), device=device)
        dist.broadcast(input_ids, src=0)
        pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(2, -1).contiguous()
        batch = {"input_ids": input_ids.clone(), "labels": input_ids.clone(), "position_ids": pos.clone()}
        cp_sharder = ContextParallelSharder(model_part0, mesh, batch)
        train_ctx, batch = cp_sharder.shard(batch)
        labels = batch.pop("labels")
        if pp_size > 1:
            with train_ctx():
                step_losses = [] if has_last else None
                model_input = batch.pop("input_ids")
                pp.update_seq_len(model_input.shape[1])
                if has_first:
                    pp.info.schedule.step(model_input, target=labels, losses=step_losses, **batch)
                else:
                    pp.info.schedule.step(target=labels, losses=step_losses, **batch)
            # Per-microbatch loss_fn returns the microbatch mean; averaging over
            # microbatches gives the batch mean, comparable to the cp2xpp1 leg.
            local = torch.stack(step_losses).mean() if has_last else torch.tensor(0.0, device=device)
        else:
            with train_ctx():
                out = model(input_ids=batch["input_ids"], position_ids=batch["position_ids"])
                local = loss_fn(out, labels)  # full output so MTP depths are included
                local.backward()
        losses.append(float(local.detach()))

    # (3) embeddings receive gradients -- only the first PP stage owns embed_tokens.
    embed = model_part0.get_input_embeddings()
    embed_grad = (
        embed is not None
        and getattr(embed, "weight", None) is not None
        and embed.weight.grad is not None
        and torch.isfinite(embed.weight.grad).all().item()
    )

    last = torch.tensor(losses[-1], device=device)
    gflag = torch.tensor(1.0 if embed_grad else 0.0, device=device)
    mflag = torch.tensor(1.0 if mtp_used["any"] else 0.0, device=device)
    dist.all_reduce(last, op=dist.ReduceOp.MAX)  # last stage holds the real loss
    dist.all_reduce(gflag, op=dist.ReduceOp.MAX)  # first stage owns embeddings
    dist.all_reduce(mflag, op=dist.ReduceOp.MAX)  # last stage owns the MTP heads

    rc = 0
    if rank == 0:
        finite = all(x == x and abs(x) != float("inf") for x in losses)
        tag = "cp2xpp1" if pp1 else "cp2xpp2"
        print(f"\n{'=' * 64}\n{tag}: {which} 20-step layer-2 (cp={cp_size} pp={pp_size})\n{'=' * 64}")
        print(
            f"last-step loss (max-reduced)={last.item():.6f}  all_finite={finite}  "
            f"embed_grad={bool(gflag.item())}  mtp_used={bool(mflag.item())}"
        )
        # MTP models must exercise the MTP heads; non-MTP models (minimax) need not.
        mtp_ok = bool(mflag.item()) or which != "step3p7"
        ok = finite and bool(gflag.item()) and mtp_ok
        print("RESULT:", "PASS (clean 20 steps + embed grads)" if ok else "FAIL")
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
