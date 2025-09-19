import torch

def prepare_i2v_conditioning(pipe, video_latents: torch.Tensor, timesteps: torch.Tensor, bf16) :
    """
    Keep the first frame clean (condition), noise the rest; returns
    (conditioned_latents[bf16], noise[bf16], condition_mask[bf16])
    """
    noise = torch.randn_like(video_latents)
    noisy = pipe.scheduler.add_noise(video_latents, noise, timesteps)

    condition_mask = torch.zeros_like(video_latents, dtype=bf16)
    condition_mask[:, :, 0] = 1.0

    conditioned = condition_mask * video_latents + (1 - condition_mask) * noisy
    return conditioned.to(bf16), noise.to(bf16), condition_mask.to(bf16)