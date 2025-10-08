# Basic usage
from dataloader import create_dataloader

# Create a simple dataloader
dataloader = create_dataloader(
    meta_folder="/code/hdvilla_sample/processed_meta",
    batch_size=1,
    shuffle=True,
    device="cuda"
)

# Use in training loop
iter = 0
for batch in dataloader:
    text_embeddings = batch['text_embeddings']  # Shape: (batch_size, seq_len, embed_dim)
    video_latents = batch['video_latents']      # Shape: (batch_size, channels, frames, h, w)
    metadata = batch['metadata']                # List of metadata dicts
    file_info = batch['file_info']              # List of file info dicts
    iter += 1
    print("@", iter, text_embeddings.shape, video_latents.shape)