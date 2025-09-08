import os
import json
import pickle
import torch
from pathlib import Path
from typing import Dict, List
import cv2
import numpy as np
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import AutoencoderKLWan
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoPreprocessor:
    def __init__(self, 
                 video_folder: str = "clipped_video",
                 wan22_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                 output_folder: str = "processed_meta",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 deterministic_latents: bool = True,
                 enable_memory_optimization: bool = True):
        """
        Initialize the video preprocessor for Wan2.2 fine-tuning.
        
        Args:
            video_folder: Path to folder containing videos and meta.json
            wan22_model_id: Hugging Face model ID for Wan2.2
            output_folder: Path to folder where .meta files will be saved
            device: Device to run inference on
            deterministic_latents: If True, use posterior mean instead of sampling (recommended for clean reconstructions)
            enable_memory_optimization: Enable Wan's built-in slicing and tiling
        """
        self.video_folder = Path(video_folder)
        self.output_folder = Path(output_folder)
        self.device = device
        self.wan22_model_id = wan22_model_id
        self.deterministic_latents = deterministic_latents
        self.enable_memory_optimization = enable_memory_optimization
        
        # Log the encoding mode
        if self.deterministic_latents:
            logger.info("Using DETERMINISTIC latents (posterior mean) - no flares expected")
        else:
            logger.info("Using STOCHASTIC latents (sampling) - may cause temporal flares")
        
        # Log memory optimization setting
        if self.enable_memory_optimization:
            logger.info("Using Wan's built-in memory optimization (slicing + tiling)")
        else:
            logger.info("Memory optimization disabled - using full tensors")
        
        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output folder created/verified: {self.output_folder}")
        
        # Load Wan2.2 components
        logger.info(f"Loading Wan2.2 components from {wan22_model_id}...")
        self.text_encoder = self._load_text_encoder()
        self.vae = self._load_vae()
        self.tokenizer = self._load_tokenizer()
        
        # Load metadata
        self.metadata = self._load_metadata()
        
    def _load_text_encoder(self):
        """Load Wan2.2 UMT5 text encoder from Hugging Face."""
        logger.info("Loading UMT5 text encoder...")
        text_encoder = UMT5EncoderModel.from_pretrained(
            self.wan22_model_id, 
            subfolder="text_encoder",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        text_encoder.to(self.device)
        text_encoder.eval()
        logger.info("UMT5 text encoder loaded successfully")
        return text_encoder
    
    def _load_vae(self):
        """Load Wan2.2 VAE from Hugging Face with memory optimization."""
        logger.info("Loading Wan VAE...")
        
        # CRITICAL: Use the correct VAE subfolder for your model
        # For Wan2.2-TI2V-5B, make sure you're using the 5B-compatible VAE
        try:
            vae = AutoencoderKLWan.from_pretrained(
                self.wan22_model_id,
                subfolder="vae",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
        except Exception as e:
            logger.error(f"Failed to load VAE from {self.wan22_model_id}/{self.vae_subfolder}")
            logger.error(f"Error: {e}")
            logger.info("Make sure you're using the correct VAE for your model:")
            logger.info("- For Wan2.2-TI2V-5B: use the Wan2.2 VAE (not Wan2.1)")
            logger.info("- For Wan2.1 models: use the Wan2.1 VAE")
            raise
        
        vae.to(self.device)
        vae.eval()
        
        # Enable Wan's built-in memory optimization
        if self.enable_memory_optimization:
            logger.info("Enabling Wan VAE memory optimization...")
            vae.enable_slicing()   # Reduce peak memory by slicing batch
            vae.disable_tiling()    # Tile H/W during encode+decode
            logger.info("✅ Enabled slicing and tiling for memory efficiency")
        else:
            logger.info("Memory optimization disabled - using full tensors")
        
        # Debug: Print VAE config to understand available attributes
        logger.info("Wan VAE loaded successfully")
        logger.info(f"VAE config type: {type(vae.config)}")
        
        # Log the input/output channels to verify correctness
        if hasattr(vae.config, 'in_channels'):
            logger.info(f"VAE in_channels: {vae.config.in_channels}")
        if hasattr(vae.config, 'out_channels'):  
            logger.info(f"VAE out_channels: {vae.config.out_channels}")
        
        # Check for scaling factor in different places
        if hasattr(vae.config, 'scaling_factor'):
            logger.info(f"Found scaling_factor in config: {vae.config.scaling_factor}")
        elif hasattr(vae, 'scaling_factor'):
            logger.info(f"Found scaling_factor as VAE attribute: {vae.scaling_factor}")
        else:
            logger.warning("No scaling_factor found - will use default")
            
        return vae
    
    def _load_tokenizer(self):
        """Load UMT5 tokenizer from Hugging Face."""
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.wan22_model_id,
            subfolder="tokenizer"
        )
        logger.info("Tokenizer loaded successfully")
        return tokenizer
    
    def _load_metadata(self) -> List[Dict]:
        """Load video metadata from meta.json."""
        meta_path = self.video_folder / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"meta.json not found in {self.video_folder}")
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Loaded metadata for {len(metadata)} videos")
        return metadata
    
    def load_video_frames(self, video_path: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """
        Load video frames and convert to tensor for Wan VAE.
        
        Args:
            video_path: Path to video file
            start_frame: Starting frame index
            end_frame: Ending frame index
            
        Returns:
            Video tensor of shape (batch, channels, num_frames, height, width)
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        # Set to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and normalize to [0, 1]
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames loaded from {video_path}")
        
        logger.info(f"Loaded {len(frames)} frames from {video_path}")
        
        # Convert to numpy array: (num_frames, height, width, channels)
        video_array = np.array(frames)
        logger.info(f"Video array shape: {video_array.shape}")
        
        # Convert to tensor and rearrange to: (batch, channels, num_frames, height, width)
        video_tensor = torch.from_numpy(video_array)
        video_tensor = video_tensor.permute(3, 0, 1, 2)  # (channels, num_frames, height, width)
        video_tensor = video_tensor.unsqueeze(0)  # (batch, channels, num_frames, height, width)
        
        # Convert to the same dtype as VAE (float16 for GPU, float32 for CPU)
        target_dtype = torch.float16 if self.device == "cuda" else torch.float32
        video_tensor = video_tensor.to(dtype=target_dtype)
        
        logger.info(f"Final video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        return video_tensor.to(self.device)
    
    def encode_text(self, caption: str) -> torch.Tensor:
        """
        Encode text caption using Wan2.2 UMT5 text encoder.
        
        Args:
            caption: Text description of the video
            
        Returns:
            Text embedding tensor
        """
        # Tokenize text with UMT5 settings
        inputs = self.tokenizer(
            caption,
            max_length=256,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Encode text using UMT5 encoder
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state
        
        return text_embeddings
    
    def encode_video(self, video_tensor: torch.Tensor) -> torch.Tensor:
        """
        Encode video using Wan2.2 VAE with built-in memory optimization.
        Uses deterministic posterior mean instead of random sampling to prevent flares.
        
        Args:
            video_tensor: Video tensor of shape (batch, channels, num_frames, height, width)
            
        Returns:
            Video latent tensor (normalized)
        """
        logger.info(f"Input video tensor shape: {video_tensor.shape}, dtype: {video_tensor.dtype}")
        
        B, C, T, H, W = video_tensor.shape
        
        # Ensure tensor is on correct device and dtype
        video_tensor = video_tensor.to(device=self.device, dtype=self.vae.dtype)
        
        # Convert to [-1, 1] range for VAE
        video_tensor = video_tensor * 2.0 - 1.0
        
        # Get normalization parameters for Wan VAE
        if hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
            # Wan VAE uses per-channel normalization
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device, dtype=self.vae.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std, device=self.device, dtype=self.vae.dtype)
            
            # Reshape for broadcasting: (1, C, 1, 1, 1) for 5D tensors
            latents_mean = latents_mean.view(1, -1, 1, 1, 1)
            latents_std = latents_std.view(1, -1, 1, 1, 1)
            
            logger.info(f"Using Wan VAE per-channel normalization")
            logger.info(f"latents_mean shape: {latents_mean.shape}, latents_std shape: {latents_std.shape}")
            use_wan_normalization = True
        else:
            # Fallback to standard scaling factor
            scaling_factor = getattr(self.vae.config, 'scaling_factor', 0.18215)
            logger.warning(f"No latents_mean/latents_std found, using scaling_factor: {scaling_factor}")
            use_wan_normalization = False
        
        # 🔥 SIMPLIFIED: Use Wan's built-in memory optimization
        # No manual chunking needed - VAE handles it internally with slicing/tiling
        with torch.no_grad():
            logger.info("Encoding with Wan VAE (using built-in memory optimization)")
            latent_dist = self.vae.encode(video_tensor)
            
            if self.deterministic_latents:
                # Use posterior mean for deterministic, flare-free encoding
                video_latents = latent_dist.latent_dist.mean
                logger.info("Using deterministic posterior mean (no flares)")
            else:
                # Use random sampling (training-style, but causes flares in reconstruction)
                video_latents = latent_dist.latent_dist.sample()
                logger.info("Using stochastic sampling (may cause flares)")
            
            # Apply normalization (Wan style or standard)
            if use_wan_normalization:
                # Wan VAE: normalize per channel (z - mean) / std
                video_latents = (video_latents - latents_mean) / latents_std
                logger.info("Applied Wan VAE per-channel normalization")
            else:
                # Standard VAE: apply scaling factor
                video_latents = video_latents * scaling_factor
                logger.info(f"Applied standard scaling factor: {scaling_factor}")
        
        logger.info(f"Output video latents shape: {video_latents.shape}, dtype: {video_latents.dtype}")
        logger.info(f"Encoding mode: {'deterministic' if self.deterministic_latents else 'stochastic'}")
        logger.info(f"Memory optimization: {'enabled' if self.enable_memory_optimization else 'disabled'}")
        return video_latents
    
    def save_processed_data(self, video_name: str, text_embeddings: torch.Tensor, 
                          video_latents: torch.Tensor, metadata: Dict):
        """
        Save processed text embeddings and video latents to binary file.
        
        Args:
            video_name: Original video filename
            text_embeddings: Encoded text embeddings
            video_latents: Encoded video latents
            metadata: Original metadata for the video
        """
        # Create output filename in the output folder
        video_stem = Path(video_name).stem
        output_path = self.output_folder / f"{video_stem}.meta"
        
        # Prepare data for saving
        processed_data = {
            'text_embeddings': text_embeddings.cpu(),
            'video_latents': video_latents.cpu(),
            'metadata': metadata,
            'original_filename': video_name,
            'original_video_path': str(self.video_folder / video_name),
            'deterministic_latents': self.deterministic_latents,  # Save encoding mode
            'memory_optimization': self.enable_memory_optimization  # Save memory setting
        }
        
        # Save as pickle file
        with open(output_path, 'wb') as f:
            pickle.dump(processed_data, f)
        
        logger.info(f"Saved processed data to {output_path}")
        logger.info(f"Encoding mode saved: {'deterministic' if self.deterministic_latents else 'stochastic'}")
        logger.info(f"Memory optimization: {'enabled' if self.enable_memory_optimization else 'disabled'}")
        
        # Log file sizes for reference
        video_path = self.video_folder / video_name
        if video_path.exists():
            original_size = video_path.stat().st_size / (1024*1024)  # MB
            meta_size = output_path.stat().st_size / (1024*1024)  # MB
            logger.info(f"Compression: {original_size:.1f}MB → {meta_size:.1f}MB ({meta_size/original_size:.2%})")
    
    def process_single_video(self, video_metadata: Dict):
        """Process a single video and save the results."""
        video_name = video_metadata['file_name']
        video_path = self.video_folder / video_name
        
        if not video_path.exists():
            logger.warning(f"Video file {video_path} not found, skipping...")
            return
        
        logger.info(f"Processing {video_name}...")
        
        try:
            # Load video frames
            logger.info(f"Step 1: Loading video frames...")
            video_tensor = self.load_video_frames(
                str(video_path),
                video_metadata['start_frame'],
                video_metadata['end_frame']
            )
            logger.info(f"Step 1 completed: video_tensor shape = {video_tensor.shape}")
            
            # Encode text caption
            logger.info(f"Step 2: Encoding text caption...")
            text_embeddings = self.encode_text(video_metadata['vila_caption'])
            logger.info(f"Step 2 completed: text_embeddings shape = {text_embeddings.shape}")
            
            # Encode video
            logger.info(f"Step 3: Encoding video with VAE...")
            video_latents = self.encode_video(video_tensor)
            logger.info(f"Step 3 completed: video_latents shape = {video_latents.shape}")
            
            # Save processed data
            logger.info(f"Step 4: Saving processed data...")
            self.save_processed_data(
                video_name,
                text_embeddings,
                video_latents,
                video_metadata
            )
            logger.info(f"Step 4 completed")
            
            logger.info(f"Successfully processed {video_name}")
            
        except Exception as e:
            import traceback
            logger.error(f"Error processing {video_name}: {e}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
    
    def process_all_videos(self):
        """Process all videos in the folder."""
        logger.info(f"Starting to process {len(self.metadata)} videos...")
        logger.info(f"Encoding mode: {'deterministic (flare-free)' if self.deterministic_latents else 'stochastic (may have flares)'}")
        logger.info(f"Memory optimization: {'enabled (slicing + tiling)' if self.enable_memory_optimization else 'disabled'}")
        
        for i, video_metadata in enumerate(self.metadata):
            logger.info(f"Progress: {i+1}/{len(self.metadata)}")
            self.process_single_video(video_metadata)
        
        logger.info("Finished processing all videos!")
    
    def load_processed_data(self, meta_file: str) -> Dict:
        """
        Load processed data from .meta file.
        
        Args:
            meta_file: Path to .meta file (can be relative to output_folder or absolute path)
            
        Returns:
            Dictionary containing text_embeddings, video_latents, and metadata
        """
        meta_path = Path(meta_file)
        
        # If it's not an absolute path, assume it's in the output folder
        if not meta_path.is_absolute():
            meta_path = self.output_folder / meta_file
            
        with open(meta_path, 'rb') as f:
            data = pickle.load(f)
            
        # Check encoding mode and memory optimization of loaded data
        encoding_mode = data.get('deterministic_latents', 'unknown')
        memory_opt = data.get('memory_optimization', 'unknown')
        logger.info(f"Loaded .meta file with encoding mode: {encoding_mode}, memory optimization: {memory_opt}")
        
        return data
    
    def list_processed_files(self) -> List[str]:
        """
        List all .meta files in the output folder.
        
        Returns:
            List of .meta filenames
        """
        meta_files = list(self.output_folder.glob("*.meta"))
        return [f.name for f in meta_files]


def main():
    """Main function to run the preprocessing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Preprocess videos for Wan2.2 fine-tuning")
    parser.add_argument("--video_folder", default="clipped_video",
                        help="Path to folder containing videos and meta.json")
    parser.add_argument("--output_folder", default="processed_meta",
                        help="Path to folder where .meta files will be saved")
    parser.add_argument("--model", default="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
                        help="Wan2.2 model ID")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic encoding (sampling) instead of deterministic (may cause flares)")
    parser.add_argument("--no-memory-optimization", action="store_true",
                        help="Disable Wan's built-in memory optimization")
    
    args = parser.parse_args()
    
    # Initialize preprocessor
    preprocessor = VideoPreprocessor(
        video_folder=args.video_folder,
        wan22_model_id=args.model,
        output_folder=args.output_folder,
        device=args.device,
        deterministic_latents=not args.stochastic,  # Default to deterministic
        enable_memory_optimization=not args.no_memory_optimization  # Default to enabled
    )
    
    # Process all videos
    preprocessor.process_all_videos()


if __name__ == "__main__":
    main()


# Example usage:
"""
# Process with deterministic encoding and memory optimization (recommended)
python tokenizer.py

# Disable memory optimization if you have plenty of VRAM
python tokenizer.py --no-memory-optimization

# Use stochastic encoding (not recommended for reconstruction quality)
python tokenizer.py --stochastic

# Custom settings
python tokenizer.py --video_folder clipped_video --output_folder processed_meta_clean --model Wan-AI/Wan2.2-TI2V-5B-Diffusers

# Programmatic usage
preprocessor = VideoPreprocessor(
    video_folder="clipped_video", 
    wan22_model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    output_folder="processed_meta",
    deterministic_latents=True,      # Flare-free encoding
    enable_memory_optimization=True  # Use Wan's built-in optimization
)

preprocessor.process_all_videos()
"""
