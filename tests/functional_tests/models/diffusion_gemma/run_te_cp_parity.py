# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Multi-node TE-CP forward parity probe for DiffusionGemma."""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh


def main():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    rank = dist.get_rank()

    from transformers.models.diffusion_gemma.configuration_diffusion_gemma import DiffusionGemmaConfig

    from nemo_automodel.components.distributed.context_parallel import ContextParallelSharder
    from nemo_automodel.components.distributed.context_parallel.utils import attach_te_context_parallel
    from nemo_automodel.components.loss.linear_ce import FusedLinearCrossEntropy
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.components.models.diffusion_gemma.attention_mask import build_block_diffusion_training_mask
    from nemo_automodel.components.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusion

    text_cfg = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        global_head_dim=16,
        num_global_key_value_heads=2,
        sliding_window=8,
        layer_types=["sliding_attention", "full_attention"],
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=32,
        torch_dtype="bfloat16",
    )
    config = DiffusionGemmaConfig(text_config=text_cfg, vision_config=None, canvas_length=5)
    backend = BackendConfig(attn="te", linear="torch", rms_norm="torch_fp32", experts="torch_mm", dispatcher="torch")

    def make_model():
        torch.manual_seed(1234)
        model = DiffusionGemmaForBlockDiffusion(
            config, backend=backend, self_conditioning=False, freeze_router=False
        ).to(device=device, dtype=torch.bfloat16)
        for layer in model.model.layers.values():
            torch.nn.init.normal_(layer.moe.experts.gate_and_up_projs, std=0.02)
            torch.nn.init.normal_(layer.moe.experts.down_projs, std=0.02)
        return model.train()

    cp_model = make_model()
    reference = make_model()
    reference.load_state_dict(cp_model.state_dict())

    batch_size, encoder_len, canvas_len = 1, 13, 5
    generator = torch.Generator(device=device).manual_seed(7)
    clean = torch.randint(0, 128, (batch_size, encoder_len), generator=generator, device=device)
    canvas = torch.randint(0, 128, (batch_size, canvas_len), generator=generator, device=device)
    full, sliding = build_block_diffusion_training_mask(
        prefix_lengths=8,
        response_length=canvas_len,
        enc_len=encoder_len,
        block_size=5,
        sliding_window=8,
        batch_size=batch_size,
        device=device,
        dtype=torch.bfloat16,
    )
    batch = {
        "input_ids": clean,
        "canvas_ids": canvas,
        "encoder_position_ids": torch.arange(encoder_len, device=device)[None],
        "decoder_position_ids": torch.arange(8, 13, device=device)[None],
        "encoder_padding_mask": torch.zeros_like(clean, dtype=torch.bool),
        "decoder_padding_mask": torch.zeros_like(canvas, dtype=torch.bool),
        "encoder_labels": torch.cat((clean[:, 1:], torch.full_like(clean[:, :1], -100)), dim=1),
        "decoder_attention_mask": {"full_attention": full, "sliding_attention": sliding},
        "do_self_conditioning": False,
    }

    mesh = init_device_mesh("cuda", (dist.get_world_size(),), mesh_dim_names=("cp",))
    attach_te_context_parallel(cp_model, mesh["cp"])
    sharder = ContextParallelSharder(cp_model, mesh, batch)
    _, local_batch = sharder.shard(batch)
    local_labels = local_batch.pop("encoder_labels")
    cp_out = cp_model(**local_batch)
    local_out = cp_out.logits.float()

    reference_out = reference(
        input_ids=clean,
        canvas_ids=canvas,
        encoder_position_ids=torch.arange(encoder_len, device=device)[None],
        decoder_position_ids=torch.arange(8, 13, device=device)[None],
        encoder_padding_mask=torch.zeros_like(clean, dtype=torch.bool),
        decoder_padding_mask=torch.zeros_like(canvas, dtype=torch.bool),
        decoder_attention_mask={"full_attention": full, "sliding_attention": sliding},
        do_self_conditioning=False,
    ).logits.float()
    valid = local_batch["cp_canvas_indices"] < canvas_len
    expected = reference_out.index_select(1, local_batch["cp_canvas_indices"][valid])
    local_valid = local_out[:, valid]
    difference = (local_valid - expected).abs()
    relative_mean = difference.mean() / expected.abs().mean().clamp_min(torch.finfo(torch.float32).eps)
    cosine = torch.nn.functional.cosine_similarity(local_valid.flatten(), expected.flatten(), dim=0)
    assert cosine.item() > 0.999, cosine.item()
    assert relative_mean.item() < 0.02, relative_mean.item()

    # Exercise the long-context AR path and TE CP backward. This is deliberately
    # the same fused linear CE used by the recipe; no [sequence, vocab] encoder
    # logits are materialized.
    ar_loss = FusedLinearCrossEntropy(logit_softcapping=30.0, reduction="sum")(
        cp_out.encoder_hidden_states,
        local_labels,
        cp_model.lm_head.weight,
        num_label_tokens=encoder_len - 1,
    )
    (ar_loss + local_valid.square().mean()).backward()
    grad = cp_model.model.layers["0"].self_attn.q_proj.weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    if rank == 0:
        print(
            f"PASS world={dist.get_world_size()} nodes={os.environ.get('SLURM_NNODES')} "
            f"local_logits={tuple(local_out.shape)} max_error={difference.max().item():.6f} "
            f"relative_mean={relative_mean.item():.6f} cosine={cosine.item():.8f} ar_loss={ar_loss.item():.6f}"
        )

    del cp_model, reference, cp_out, local_out, reference_out
    torch.cuda.empty_cache()
    dist.barrier()

    # A real 100K-token forward/backward exercises padding, RoPE, TE CP
    # transport, the non-causal decoder bias, and fused AR CE without requiring
    # the 26B checkpoint. Keep this probe sliding-only so validation finishes
    # quickly; the dependent job runs the released model/config.
    long_len, long_canvas = 100_000, 16
    long_text_cfg = dict(text_cfg)
    long_text_cfg.update(
        num_hidden_layers=1,
        layer_types=["sliding_attention"],
        sliding_window=64,
    )
    long_config = DiffusionGemmaConfig(text_config=long_text_cfg, vision_config=None, canvas_length=long_canvas)
    torch.manual_seed(5678)
    long_model = DiffusionGemmaForBlockDiffusion(
        long_config, backend=backend, self_conditioning=False, freeze_router=False
    ).to(device=device, dtype=torch.bfloat16)
    for layer in long_model.model.layers.values():
        torch.nn.init.normal_(layer.moe.experts.gate_and_up_projs, std=0.02)
        torch.nn.init.normal_(layer.moe.experts.down_projs, std=0.02)
    long_model.train()
    attach_te_context_parallel(long_model, mesh["cp"])

    long_clean = torch.randint(0, 128, (1, long_len), generator=generator, device=device)
    long_canvas_ids = torch.randint(0, 128, (1, long_canvas), generator=generator, device=device)
    long_full, long_sliding = build_block_diffusion_training_mask(
        prefix_lengths=long_len - long_canvas,
        response_length=long_canvas,
        enc_len=long_len,
        block_size=long_canvas,
        sliding_window=64,
        batch_size=1,
        device=device,
        dtype=torch.bfloat16,
    )
    long_batch = {
        "input_ids": long_clean,
        "canvas_ids": long_canvas_ids,
        "encoder_position_ids": torch.arange(long_len, device=device)[None],
        "decoder_position_ids": torch.arange(long_len - long_canvas, long_len, device=device)[None],
        "encoder_padding_mask": torch.zeros_like(long_clean, dtype=torch.bool),
        "decoder_padding_mask": torch.zeros_like(long_canvas_ids, dtype=torch.bool),
        "encoder_labels": torch.cat((long_clean[:, 1:], torch.full_like(long_clean[:, :1], -100)), dim=1),
        "decoder_attention_mask": {"full_attention": long_full, "sliding_attention": long_sliding},
        "do_self_conditioning": False,
    }
    long_sharder = ContextParallelSharder(long_model, mesh, long_batch)
    _, long_local = long_sharder.shard(long_batch)
    long_labels = long_local.pop("encoder_labels")
    long_out = long_model(**long_local)
    long_ar_loss = FusedLinearCrossEntropy(logit_softcapping=30.0, reduction="sum")(
        long_out.encoder_hidden_states,
        long_labels,
        long_model.lm_head.weight,
        num_label_tokens=long_len - 1,
    )
    (long_ar_loss + long_out.logits.float().square().mean()).backward()
    long_grad = long_model.model.layers["0"].self_attn.q_proj.weight.grad
    assert long_grad is not None and torch.isfinite(long_grad).all()
    if rank == 0:
        print(
            f"PASS_LONG seq={long_len} local_encoder={long_local['input_ids'].shape[1]} "
            f"local_canvas={long_local['canvas_ids'].shape[1]} ar_loss={long_ar_loss.item():.6f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
