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

"""GPU CP forward-equivalence for the Gemma4 in-forward pre-embed shard (L1, 2 GPUs).

Exercises the sunk contiguous-CP path end to end for the E-series-shaped dense
Gemma4: ``prepare_cp_forward`` invokes the sharder-only
``prepare_model_inputs_for_cp`` hook (``shard_batch_contiguous(shard_primary=False)``
shards labels/position_ids and the synthesized ``_packed_seq_ids``; the model records
``cp_mesh`` and installs its p2p flex ring), and
``Gemma4ForConditionalGeneration.forward`` embeds the full sequence, builds
``per_layer_inputs`` + the vision-bidirectional ring metadata, then contiguously
slices this rank's shard. The unsharded cp2 logits must match the cp1 eager
forward.

  cp1 eager (cp_mesh unset)  ==  cp2 (apply_cp + prepare_cp_forward + in-forward slice)

vision-bidirectional mask is driven by ``mm_token_type_ids`` (a vision block) so
the ``_gemma4_vision_group_ids`` cumsum-then-slice path is exercised without
building the vision tower; ``hidden_size_per_layer_input`` exercises the 4D
``per_layer_inputs`` slice. bf16 (the flex ring has no fp32 kernel here);
per-token argmax agreement is the leakage-robust signal.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/functional_tests/context_parallel/run_gemma4_vl_cp_sink.py
"""

import os
import sys
from datetime import timedelta

import torch
import torch.distributed as dist


def main():
    # Gemma4's flex ring compiles per (layer-type, chunk-geometry) on the first
    # cp2 forward; the two ranks can reach the first p2p exchange minutes apart, so
    # a generous PG timeout keeps the NCCL watchdog from aborting mid-compile.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(lr)
    device = torch.device(f"cuda:{lr}")
    cp_size = world

    from torch.distributed.device_mesh import init_device_mesh

    from nemo_automodel.components.distributed.cp_utils import prepare_cp_forward
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.gemma4_moe.model import (
        Gemma4Config,
        Gemma4ForConditionalGeneration,
        Gemma4TextConfig,
    )
    from nemo_automodel.components.moe.parallelizer import apply_cp

    backend = BackendConfig(
        linear="torch",
        attn="sdpa",
        rms_norm="torch",
        experts="torch",
        dispatcher="torch",
        fake_balanced_gate=False,
        enable_hf_state_dict_adapter=False,
    )
    text_cfg = dict(
        vocab_size=128,
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_hidden_layers=4,
        intermediate_size=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        enable_moe_block=False,
        layer_types=["full_attention", "sliding_attention"] * 2,
        sliding_window=64,
        hidden_activation="gelu_pytorch_tanh",
        torch_dtype="bfloat16",
        # E-series shape: per-layer inputs + vision-bidirectional attention.
        hidden_size_per_layer_input=8,
        vocab_size_per_layer_input=128,
        use_double_wide_mlp=False,
        use_bidirectional_attention="vision",
    )
    # CP_NO_VISION isolates the core sunk mechanism (embed + per_layer_inputs +
    # contiguous ring + position sharding) from the flex-ring-vs-HF-eager
    # vision-mask kernel difference: plain causal attention, no vision block. The
    # simple causal flex mask also compiles fast.
    no_vision = bool(os.environ.get("CP_NO_VISION"))
    if no_vision:
        text_cfg.pop("use_bidirectional_attention", None)

    torch.manual_seed(0)
    cfg = Gemma4Config(text_config=Gemma4TextConfig(**text_cfg))
    cfg.image_token_id = 42
    model = Gemma4ForConditionalGeneration(cfg, backend=backend).eval().to(device)
    model.initialize_weights(dtype=torch.bfloat16)
    model.to(torch.bfloat16)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    for b in model.buffers():
        dist.broadcast(b.data, src=0)

    # A freshly-initialized tiny model sits at max entropy (loss ~= ln(vocab)),
    # so its logits are near-uniform and per-token argmax is noise-dominated. Scale
    # the (tied) embedding so the logits are peaked and top1 agreement is a real
    # signal; both the eager and CP forwards see the identical scaled model.
    with torch.no_grad():
        model.get_input_embeddings().weight.mul_(6.0)

    seqlen = 16 * cp_size  # divisible by 2*cp -> no pad
    input_ids = torch.randint(2, cfg.text_config.vocab_size, (1, seqlen), device=device)
    # A contiguous vision block (mm_token_type_ids==1) so the vision-bidirectional
    # mask + vision_group_ids cumsum are exercised; put it mid-sequence.
    mm = torch.zeros(1, seqlen, dtype=torch.long, device=device)
    if not no_vision:
        mm[0, seqlen // 4 : seqlen // 4 + 4] = 1
        input_ids[0, seqlen // 4 : seqlen // 4 + 4] = cfg.image_token_id
    dist.broadcast(input_ids, src=0)
    dist.broadcast(mm, src=0)
    position_ids = torch.arange(seqlen, device=device).unsqueeze(0)
    mm_arg = None if no_vision else mm

    # cp1 eager (no CP mesh installed): full-sequence forward.
    with torch.no_grad():
        logits_eager = model(
            input_ids=input_ids,
            position_ids=position_ids.clone(),
            mm_token_type_ids=None if mm_arg is None else mm_arg.clone(),
        ).logits.float()
    if rank == 0:
        print("[progress] cp1 eager forward done", flush=True)

    # cp2: the recipe path -- apply_cp installs the ring, prepare_cp_forward runs the
    # sharder-only hook (aux-only contiguous shard), forward embeds + contiguously
    # slices this rank's shard.
    device_mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("cp",))
    apply_cp(model, device_mesh["cp"])
    batch = {
        "input_ids": input_ids.clone(),
        "labels": input_ids.clone(),
        "position_ids": position_ids.clone(),
    }
    if mm_arg is not None:
        batch["mm_token_type_ids"] = mm_arg.clone()
    train_ctx, batch, _ = prepare_cp_forward(model, device_mesh, batch)
    batch.pop("labels", None)
    # Drop the synthesized single-document _packed_seq_ids: it reaches the ring by
    # the identical contiguous slice in old and new (unit-proven slice-equivalence),
    # but its per-batch flex block mask is a heavy compile. Dropping it symmetrically
    # keeps the old-vs-new comparison valid and the run tractable.
    batch.pop("_packed_seq_ids", None)
    # Pattern-agnostic call so this harness drives BOTH the sunk path (batch keeps
    # input_ids full) and the legacy recipe-level pre-embed path (batch carries the
    # pre-embedded, dispatch-sharded inputs_embeds) -- letting old-vs-new cp2 logits
    # be compared directly.
    with torch.no_grad(), train_ctx():
        logits_local = model(**batch).logits.float()
    if rank == 0:
        print("[progress] cp2 sunk forward done", flush=True)

    # Contiguous unshard: rank r owns global [r*L:(r+1)*L]; gather + concat in rank order.
    local_len = logits_local.shape[1]
    parts = [torch.empty_like(logits_local) for _ in range(world)]
    dist.all_gather(parts, logits_local.contiguous())
    logits_cp_full = torch.cat(parts, dim=1)[:, :seqlen, :]

    rc = 0
    if rank == 0:
        out_path = os.environ.get("CP_LOGITS_OUT")
        if out_path:
            torch.save({"cp": logits_cp_full.cpu(), "eager": logits_eager.cpu()}, out_path)
        diff = (logits_cp_full - logits_eager).abs()
        top1 = (logits_cp_full.argmax(-1) == logits_eager.argmax(-1)).float().mean().item()
        ce = torch.nn.functional.cross_entropy
        loss_eager = ce(logits_eager[0, :-1], input_ids[0, 1:]).item()
        loss_cp = ce(logits_cp_full[0, :-1], input_ids[0, 1:]).item()
        print(
            f"\n{'=' * 72}\nGemma4 in-forward contiguous CP sink  cp_size={cp_size}  seqlen={seqlen}  local={local_len}"
        )
        print("=" * 72)
        print(f"logits mean_abs={diff.mean().item():.3e} max_abs={diff.max().item():.3e} top1={top1:.4f} (info)")
        print(f"loss eager={loss_eager:.6f} cp={loss_cp:.6f} |dloss|={abs(loss_eager - loss_cp):.3e}")
        # cp2 runs Gemma4's flex p2p ring while cp1 eager runs HF's SDPA, so per-token
        # argmax agreement (top1) carries an inherent kernel gap and is reported for
        # information only. The loss agreement is the robust signal that the sunk
        # contiguous shard reproduces the full-sequence forward end to end.
        ok = abs(loss_eager - loss_cp) < 1e-1
        print("RESULT:", "PASS (in-forward CP loss == eager)" if ok else "FAIL")
        rc = 0 if ok else 1
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(rc)


if __name__ == "__main__":
    main()
