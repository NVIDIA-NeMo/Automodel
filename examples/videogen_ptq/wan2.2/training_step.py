from typing import Dict

import torch

from .conditioning import prepare_i2v_conditioning


def boundary_from_ratio(pipe, ratio: float):
    sch = pipe.scheduler
    num_train = getattr(sch, "num_train_timesteps", None)
    if num_train is None and hasattr(sch, "config") and hasattr(sch.config, "num_train_timesteps"):
        num_train = int(sch.config.num_train_timesteps)
    if num_train is None:
        num_train = 1000
    return max(0, int(ratio * num_train)), num_train


def step_dual_transformer(
    pipe,
    model_map,
    transformer_names,
    batch: Dict,
    device,
    bf16,
    boundary_ratio: float,
) -> torch.Tensor:
    text_embeddings = batch["text_embeddings"].to(device, dtype=bf16)
    video_latents = batch["video_latents"].to(device, dtype=torch.float32)
    if video_latents.ndim == 6:
        video_latents = video_latents.squeeze(1)
    if text_embeddings.ndim == 4:
        text_embeddings = text_embeddings.squeeze(1)

    boundary_ts, num_train = boundary_from_ratio(pipe, boundary_ratio)

    B = video_latents.shape[0]
    timesteps = torch.randint(0, num_train, (B,), device=device, dtype=torch.long)

    with torch.no_grad():
        cond, noise, cond_mask = prepare_i2v_conditioning(pipe, video_latents, timesteps, bf16)

    use_t1 = timesteps >= boundary_ts
    use_t2 = ~use_t1

    total_loss = None
    n = 0
    with torch.autocast(device_type="cuda", dtype=bf16):
        if use_t1.any():
            idx = use_t1.nonzero(as_tuple=True)[0]
            m = model_map["transformer"]["fsdp"]
            out = m(
                hidden_states=cond[idx],
                timestep=timesteps[idx],
                encoder_hidden_states=text_embeddings[idx],
                return_dict=False,
            )
            pred = out[0] if isinstance(out, tuple) else out
            mask = 1 - cond_mask[idx]
            l = torch.nn.functional.mse_loss(pred * mask, noise[idx] * mask)
            s = l * len(idx)
            total_loss = s if total_loss is None else total_loss + s
            n += len(idx)

        if use_t2.any() and "transformer_2" in model_map:
            idx = use_t2.nonzero(as_tuple=True)[0]
            m = model_map["transformer_2"]["fsdp"]
            out = m(
                hidden_states=cond[idx],
                timestep=timesteps[idx],
                encoder_hidden_states=text_embeddings[idx],
                return_dict=False,
            )
            pred = out[0] if isinstance(out, tuple) else out
            mask = 1 - cond_mask[idx]
            l = torch.nn.functional.mse_loss(pred * mask, noise[idx] * mask)
            s = l * len(idx)
            total_loss = s if total_loss is None else total_loss + s
            n += len(idx)

    if total_loss is None:
        raise ValueError("No samples processed in training step")
    final = total_loss / n
    if torch.isnan(final) or torch.isinf(final):
        raise ValueError(f"Invalid loss: {final.item()}")
    return final
