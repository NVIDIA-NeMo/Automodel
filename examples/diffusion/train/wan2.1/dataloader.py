import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

import torch
from dist_utils import print0
from torch.utils.data import DataLoader, Dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetaFilesDataset(Dataset):
    """
    PyTorch Dataset for loading .meta files created by VideoPreprocessor.

    Each .meta file contains:
    - text_embeddings: Encoded text embeddings from UMT5
    - video_latents: Encoded video latents from Wan VAE
    - metadata: Original video metadata
    - original_filename: Original video filename
    - original_video_path: Path to original video
    - deterministic_latents: Whether deterministic encoding was used
    - memory_optimization: Whether memory optimization was enabled
    """

    def __init__(
        self,
        meta_folder: str,
        transform_text: Optional[callable] = None,
        transform_video: Optional[callable] = None,
        filter_fn: Optional[callable] = None,
        device: str = "cpu",
        max_files: Optional[int] = None,
    ):
        """
        Initialize the dataset.

        Args:
            meta_folder: Path to folder containing .meta files
            transform_text: Optional transform to apply to text embeddings
            transform_video: Optional transform to apply to video latents
            filter_fn: Optional function to filter files (takes metadata dict, returns bool)
            device: Device to load tensors on
            max_files: Optional limit on number of files to load (for testing)
        """
        self.meta_folder = Path(meta_folder)
        self.transform_text = transform_text
        self.transform_video = transform_video
        self.filter_fn = filter_fn
        self.device = device

        # Find all .meta files
        self.meta_files = sorted(list(self.meta_folder.glob("*.meta")))

        # Apply file limit if specified (via parameter or environment variable)
        if max_files is None:
            max_files_env = os.environ.get("MAX_META_FILES", None)
            if max_files_env is not None:
                max_files = int(max_files_env)
        
        if max_files is not None and max_files > 0:
            self.meta_files = self.meta_files[:max_files]
            logger.info(f"Limited to {max_files} files for testing")

        if not self.meta_files:
            raise ValueError(f"No .meta files found in {meta_folder}")

        # Filter files if filter function is provided
        if self.filter_fn:
            logger.info("Applying filter function to .meta files...")
            filtered_files = []
            for meta_file in self.meta_files:
                try:
                    with open(meta_file, "rb") as f:
                        data = pickle.load(f)
                    if self.filter_fn(data["metadata"]):
                        filtered_files.append(meta_file)
                except Exception as e:
                    logger.warning(f"Error loading {meta_file} for filtering: {e}")

            self.meta_files = filtered_files
            logger.info(f"Filtered to {len(self.meta_files)} files")

        logger.info(f"Found {len(self.meta_files)} .meta files in {meta_folder}")

        # Log some statistics about the dataset
        self._log_dataset_stats()

    def _log_dataset_stats(self):
        """Log statistics about the dataset."""
        if len(self.meta_files) == 0:
            return

        # Sample a few files to get statistics
        sample_size = min(5, len(self.meta_files))
        text_shapes = []
        video_shapes = []
        encoding_modes = []

        for i in range(sample_size):
            try:
                with open(self.meta_files[i], "rb") as f:
                    data = pickle.load(f)
                text_shapes.append(data["text_embeddings"].shape)
                video_shapes.append(data["video_latents"].shape)
                encoding_modes.append(data.get("deterministic_latents", "unknown"))
            except Exception as e:
                logger.warning(f"Error sampling {self.meta_files[i]}: {e}")

        if text_shapes:
            logger.info(f"Sample text embedding shapes: {text_shapes[:3]}")
            logger.info(f"Sample video latent shapes: {video_shapes[:3]}")
            logger.info(f"Sample encoding modes: {set(encoding_modes)}")

    def __len__(self) -> int:
        """Return the number of .meta files."""
        return len(self.meta_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and return data from a .meta file.

        Args:
            idx: Index of the file to load

        Returns:
            Dictionary containing:
            - text_embeddings: Text embeddings tensor
            - video_latents: Video latents tensor
            - metadata: Original metadata dict
            - file_info: Additional file information
        """
        meta_file = self.meta_files[idx]

        try:
            with open(meta_file, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            logger.error(f"Error loading {meta_file}: {e}")
            raise

        # Extract tensors and move to device
        text_embeddings = data["text_embeddings"].to(self.device)
        video_latents = data["video_latents"].to(self.device)

        # Apply transforms if provided
        if self.transform_text:
            text_embeddings = self.transform_text(text_embeddings)

        if self.transform_video:
            video_latents = self.transform_video(video_latents)

        # Prepare file info
        file_info = {
            "meta_filename": meta_file.name,
            "original_filename": data.get("original_filename", "unknown"),
            "original_video_path": data.get("original_video_path", "unknown"),
            "deterministic_latents": data.get("deterministic_latents", "unknown"),
            "memory_optimization": data.get("memory_optimization", "unknown"),
            "num_frames": data.get("num_frames", "unknown"),
        }

        return {
            "text_embeddings": text_embeddings,
            "video_latents": video_latents,
            "metadata": data["metadata"],
            "file_info": file_info,
        }

    def get_file_info(self, idx: int) -> Dict:
        """Get file information without loading the full data."""
        meta_file = self.meta_files[idx]

        try:
            with open(meta_file, "rb") as f:
                data = pickle.load(f)

            return {
                "meta_filename": meta_file.name,
                "original_filename": data.get("original_filename", "unknown"),
                "vila_caption": data["metadata"].get("vila_caption", "No caption"),
                "start_frame": data["metadata"].get("start_frame", "unknown"),
                "end_frame": data["metadata"].get("end_frame", "unknown"),
                "text_shape": data["text_embeddings"].shape,
                "video_shape": data["video_latents"].shape,
                "deterministic_latents": data.get("deterministic_latents", "unknown"),
                "memory_optimization": data.get("memory_optimization", "unknown"),
            }
        except Exception as e:
            logger.error(f"Error getting info for {meta_file}: {e}")
            return {"error": str(e)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Custom collate function for batching .meta file data.

    Args:
        batch: List of data dictionaries from __getitem__

    Returns:
        Batched data dictionary
    """
    # Stack text embeddings and video latents
    text_embeddings = torch.stack([item["text_embeddings"] for item in batch])
    video_latents = torch.stack([item["video_latents"] for item in batch])

    # Collect metadata and file info as lists
    metadata_list = [item["metadata"] for item in batch]
    file_info_list = [item["file_info"] for item in batch]

    return {
        "text_embeddings": text_embeddings,
        "video_latents": video_latents,
        "metadata": metadata_list,
        "file_info": file_info_list,
    }


def create_dataloader(
    meta_folder: str,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 0,
    device: str = "cpu",
    transform_text: Optional[callable] = None,
    transform_video: Optional[callable] = None,
    filter_fn: Optional[callable] = None,
    max_files: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for .meta files.

    Args:
        meta_folder: Path to folder containing .meta files
        batch_size: Batch size for the DataLoader
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        device: Device to load tensors on
        transform_text: Optional transform for text embeddings
        transform_video: Optional transform for video latents
        filter_fn: Optional function to filter files
        max_files: Optional limit on number of files to load

    Returns:
        DataLoader instance
    """
    dataset = MetaFilesDataset(
        meta_folder=meta_folder,
        transform_text=transform_text,
        transform_video=transform_video,
        filter_fn=filter_fn,
        device=device,
        max_files=max_files,
    )

    # Only use pin_memory when device is CPU (tensors will be moved to GPU later)
    # When device is CUDA, tensors are already on GPU so pin_memory should be False
    use_pin_memory = device == "cpu"

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=use_pin_memory,
    )

    return dataloader


# Example filter functions
def filter_by_duration(min_frames: int = 10, max_frames: int = 100):
    """Create a filter function based on video duration."""

    def filter_fn(metadata):
        duration = metadata["end_frame"] - metadata["start_frame"] + 1
        return min_frames <= duration <= max_frames

    return filter_fn


def filter_by_caption_length(min_length: int = 10, max_length: int = 200):
    """Create a filter function based on caption length."""

    def filter_fn(metadata):
        caption = metadata.get("vila_caption", "")
        return min_length <= len(caption) <= max_length

    return filter_fn


# Example transforms
def normalize_text_embeddings(embeddings):
    """Normalize text embeddings to unit length."""
    return torch.nn.functional.normalize(embeddings, dim=-1)


def add_noise_to_video_latents(noise_scale: float = 0.01):
    """Add small amount of noise to video latents."""

    def transform_fn(latents):
        noise = torch.randn_like(latents) * noise_scale
        return latents + noise

    return transform_fn