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

"""images x cp2xpp2 for minimax: the in-forward vision splice under CP+PP (L1, 2/4 GPU).

Regression cover for images x context-parallelism x pipeline-parallelism after the
pre-embed sink. Media rides the existing per-microbatch side channel
(prepare_vlm_media_for_pp -> stage_vlm_media_for_pp -> stage-0 chunk pull); the
in-forward embed + vision splice runs on the microbatch's full sequence with the
CP ring dispatcher suspended around the (non-causal, unsharded) vision tower
(cp_dispatcher_suspended), then shard_sequence_for_cp_round_robin shards the result. Two
images of different sizes across two samples exercise the sample-aware chunker.

Asserts: 20 clean steps (no double-backward), finite loss, cp2xpp2 == cp2xpp1
within bf16, embed_tokens grads, and vision-tower grads (trainable vision under CP).
Observed (bf16): cp2xpp2 == cp2xpp1 == 5.609686.

    torchrun --standalone --nproc-per-node=4 run_cp_pp_image_sink.py         # cp2xpp2
    NEMO_CP_PP_TEST_PP1=1 torchrun --nproc-per-node=2 run_cp_pp_image_sink.py  # cp2xpp1
"""

import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F

IMG = 100


def build_minimax(device):
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.minimax_m3_vl.config import MiniMaxM3VLConfig
    from nemo_automodel.components.models.minimax_m3_vl.model import MiniMaxM3SparseForConditionalGeneration

    tiny = dict(
        hidden_size=64,
        intermediate_size=32,
        dense_intermediate_size=48,
        shared_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rotary_dim=8,
        partial_rotary_factor=0.5,
        vocab_size=128,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        num_local_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_layer_freq=[0, 1, 1, 1],
        use_gemma_norm=True,
        use_qk_norm=True,
        qk_norm_type="per_head",
        scoring_func="sigmoid",
        use_routing_bias=True,
        routed_scaling_factor=2.0,
        swiglu_alpha=1.702,
        swiglu_limit=7.0,
        num_mtp_modules=0,
        sparse_attention_config=dict(use_sparse_attention=False),
    )
    vision = dict(
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        patch_size=2,
        num_channels=3,
        rope_theta=10000.0,
        hidden_act="gelu",
        layer_norm_eps=1e-5,
        img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2},
    )
    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        rope_fusion=False,
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    cfg = MiniMaxM3VLConfig(
        vision_config=dict(vision),
        text_config={**tiny, "torch_dtype": "bfloat16"},
        image_token_index=IMG,
        video_token_index=101,
        projector_hidden_size=tiny["hidden_size"],
    )
    model = MiniMaxM3SparseForConditionalGeneration(cfg, backend=backend)
    model.initialize_weights(dtype=torch.bfloat16)
    return model.to(device).to(torch.bfloat16)


def make_image_batch(device, seqlen):
    # 2 samples, 2 images of different sizes: A grid (1,4,4)->16 patches->4 tokens; B (1,2,2)->4 patches->1 token.
    patch_dim = 3 * 2 * 2**2  # 24
    grids = [[1, 4, 4], [1, 2, 2]]
    npatch = [g[0] * g[1] * g[2] for g in grids]  # [16, 4]
    ntok = [n // 4 for n in npatch]  # [4, 1]
    torch.manual_seed(7)
    pixel_values = torch.randn(sum(npatch), patch_dim, device=device, dtype=torch.bfloat16)
    image_grid_thw = torch.tensor(grids, device=device, dtype=torch.long)
    n_images_per_sample = torch.tensor([1, 1], device=device, dtype=torch.long)
    ids = torch.randint(2, 100, (2, seqlen), device=device)  # text ids avoid IMG=100
    ids[0, 5 : 5 + ntok[0]] = IMG  # sample 0: 4 image tokens
    ids[1, 5 : 5 + ntok[1]] = IMG  # sample 1: 1 image token
    for t in (pixel_values, image_grid_thw, n_images_per_sample, ids):
        dist.broadcast(t, src=0)
    return ids, pixel_values, image_grid_thw, n_images_per_sample


def main():
    dist.init_process_group(backend="nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device(f"cuda:{os.environ['LOCAL_RANK']}")

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.datasets.vlm.pp_media import prepare_vlm_media_for_pp, stage_vlm_media_for_pp
    from nemo_automodel.components.distributed import ContextParallelRuntime
    from nemo_automodel.components.distributed.pipelining import AutoPipeline
    from nemo_automodel.components.moe.parallelizer import apply_cp

    pp1 = bool(os.environ.get("NEMO_CP_PP_TEST_PP1"))
    pp_size = 1 if pp1 else 2
    cp_size = world // pp_size
    mesh = init_device_mesh("cuda", (pp_size, 1, cp_size), mesh_dim_names=("pp", "dp", "cp"))
    cp_runtime = ContextParallelRuntime(device_mesh=mesh)

    torch.manual_seed(0)
    model = build_minimax(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)
    model.train()
    vocab = model.config.text_config.vocab_size

    def loss_fn(output, labels):
        logits = output[0] if isinstance(output, tuple) else getattr(output, "logits", output)
        return F.cross_entropy(logits.reshape(-1, vocab).float(), labels.reshape(-1), ignore_index=-100)

    def cp_only(m, world_mesh, moe_mesh, *, dp_axis_names, cp_axis_name=None, **kw):
        if cp_axis_name is not None and world_mesh[cp_axis_name].size() > 1:
            apply_cp(m, world_mesh[cp_axis_name])

    seqlen = 32
    if pp_size > 1:
        pp = AutoPipeline(
            world_mesh=mesh,
            moe_mesh=None,
            pp_axis_name="pp",
            dp_axis_names=("dp",),
            cp_axis_name="cp",
            pp_schedule="1f1b",
            pp_microbatch_size=1,
            pp_batch_size=2,
            device=device,
            dtype=torch.bfloat16,
            pp_seq_len=seqlen,
        ).build(model, loss_fn=loss_fn, parallelize_fn=cp_only)
        model_part0, has_last, has_first = pp.parts[0], pp.info.has_last_stage, pp.info.has_first_stage
    else:
        cp_only(model, mesh, None, dp_axis_names=("dp",), cp_axis_name="cp")
        model_part0 = model

    losses = []
    for step in range(20):
        ids, pv, grid, nimg = make_image_batch(device, seqlen)
        pos = torch.arange(seqlen, device=device).unsqueeze(0).expand(2, -1).contiguous()
        batch = {"input_ids": ids.clone(), "labels": ids.clone(), "position_ids": pos.clone()}
        if pp_size > 1:
            batch.update({"pixel_values": pv, "image_grid_thw": grid, "n_images_per_sample": nimg})
            batch = prepare_vlm_media_for_pp(batch, batch_size=2, n_microbatches=2)
        else:
            batch.update({"pixel_values": pv, "image_grid_thw": grid})
        prepared_cp = cp_runtime.prepare_forward(model_part0, batch)
        train_ctx, batch = prepared_cp.context, prepared_cp.batch
        labels = batch.pop("labels")
        if pp_size > 1:
            with train_ctx(), stage_vlm_media_for_pp(pp, pp.parts, batch):
                sl = [] if has_last else None
                mi = batch.pop("input_ids")
                pp.update_seq_len(mi.shape[1])
                (
                    pp.info.schedule.step(mi, target=labels, losses=sl, **batch)
                    if has_first
                    else pp.info.schedule.step(target=labels, losses=sl, **batch)
                )
            local = torch.stack(sl).mean() if has_last else torch.tensor(0.0, device=device)
        else:
            with train_ctx():
                out = model(
                    input_ids=batch["input_ids"],
                    pixel_values=batch["pixel_values"],
                    image_grid_thw=batch["image_grid_thw"],
                    position_ids=batch["position_ids"],
                )
                local = loss_fn(out, labels)
                local.backward()
        losses.append(float(local.detach()))

    embed = model_part0.get_input_embeddings()
    egrad = (
        embed is not None
        and getattr(embed, "weight", None) is not None
        and embed.weight.grad is not None
        and torch.isfinite(embed.weight.grad).all().item()
    )
    # vision tower lives on stage 0 (the first PP part).
    vt = getattr(model_part0, "vision_tower", None)
    vgrad = False
    if vt is not None:
        gs = [p.grad for p in vt.parameters() if p.grad is not None]
        vgrad = len(gs) > 0 and all(torch.isfinite(g).all().item() for g in gs)

    last = torch.tensor(losses[-1], device=device)
    gflag = torch.tensor([1.0 if egrad else 0.0, 1.0 if vgrad else 0.0], device=device)
    dist.all_reduce(last, op=dist.ReduceOp.MAX)
    dist.all_reduce(gflag, op=dist.ReduceOp.MAX)
    rc = 0
    if rank == 0:
        finite = all(x == x and abs(x) != float("inf") for x in losses)
        tag = "cp2xpp1" if pp1 else "cp2xpp2"
        print(
            f"\n{'=' * 60}\nIMAGE {tag}: last={last.item():.6f} finite={finite} "
            f"embed_grad={bool(gflag[0].item())} vision_grad={bool(gflag[1].item())}"
        )
        ok = finite and gflag[0].item() and gflag[1].item()
        print("RESULT:", "PASS" if ok else "FAIL")
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
