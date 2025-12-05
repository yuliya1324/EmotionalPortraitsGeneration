"""
Data utilities for Classifier Guidance training.

Implements on-disk latent caching to speed up training by 20x.
Pre-encodes images with VAE and caches latents to disk.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
from diffusers import AutoencoderKL

# Add shared directory to path
import sys
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
SHARED_DIR = REPO_ROOT / "shared" / "src"
sys.path.insert(0, str(SHARED_DIR))

from dataset import EmoSetLocalDataset


class CachedLatentsDataset(Dataset):
    """
    Dataset wrapper that caches VAE-encoded latents on disk.
    
    On first access, encodes images with VAE and saves latents to disk.
    On subsequent accesses, loads cached latents from disk.
    This allows the first epoch to build the cache, and subsequent epochs
    to run much faster (20x speedup).
    """
    
    def __init__(self, root_dir: str, cache_dir: str, vae: AutoencoderKL, device: torch.device):
        """
        Initialize cached latents dataset.
        
        Args:
            root_dir: Root directory of the dataset (passed to EmoSetLocalDataset)
            cache_dir: Directory to store cached latents
            vae: VAE model for encoding (must be in eval mode and on device)
            device: Device to run VAE encoding on
        """
        # Initialize base dataset
        self.base_dataset = EmoSetLocalDataset(data_dir=root_dir, image_size=512)
        self.root_dir = root_dir
        
        self.vae = vae
        self.cache_dir = Path(cache_dir)
        self.device = device
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure VAE is in eval mode and frozen
        self.vae.eval()
        self.vae.requires_grad_(False)
        
        print(f"Initialized CachedLatentsDataset:")
        print(f"  Dataset size: {len(self.base_dataset)}")
        print(f"  Cache directory: {cache_dir}")
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """
        Get item with on-disk caching.
        
        Args:
            idx: Index of the item
            
        Returns:
            Tuple of (latent, emotion_idx)
            - latent: Pre-encoded latent tensor [4, 64, 64] (scaled by 0.18215)
            - emotion_idx: Emotion index (0-7)
        """
        # Construct cache filename using image_id if available, otherwise use idx
        # Try to get image_id from dataset if available
        try:
            example = self.base_dataset.dataset[idx]
            image_id = example.get('image_id', idx)
            cache_file = self.cache_dir / f"{image_id}.pt"
        except:
            # Fallback to idx if image_id not available
            cache_file = self.cache_dir / f"{idx}.pt"
        
        # Check if cached
        if cache_file.exists():
            # Load from cache
            latent = torch.load(cache_file, map_location="cpu")
            # Get emotion from base dataset
            _, _, emotion = self.base_dataset[idx]
            emotion_idx = self._emotion_to_idx(emotion)
            return latent, emotion_idx
        else:
            # Cache miss: encode with VAE
            # Get image from base dataset
            image, _, emotion = self.base_dataset[idx]
            
            # Move image to device and add batch dimension
            image_batch = image.unsqueeze(0).to(self.device)  # [1, 3, 512, 512]
            
            # Encode with VAE
            with torch.no_grad():
                # CRITICAL: VAE expects images in [0, 1] range, but dataset provides [-1, 1]
                # Convert from [-1, 1] to [0, 1]
                image_batch = (image_batch + 1.0) / 2.0
                
                # VAE encode: [1, 3, 512, 512] -> [1, 4, 64, 64]
                dist = self.vae.encode(image_batch)
                latent = dist.latent_dist.sample()
                # CRITICAL: Scale by VAE scaling factor (0.18215 for SD v1.5)
                latent = latent * self.vae.config.scaling_factor
            # Move to CPU and remove batch dimension
            latent = latent.squeeze(0).cpu()  # [4, 64, 64]
            
            # Save to cache
            torch.save(latent, cache_file)
            
            # Get emotion index
            emotion_idx = self._emotion_to_idx(emotion)
            return latent, emotion_idx
    
    def _emotion_to_idx(self, emotion: str) -> int:
        """Convert emotion string to index."""
        emotion = emotion.lower()
        emotion_map = {
            'amusement': 0,
            'anger': 1,
            'awe': 2,
            'contentment': 3,
            'disgust': 4,
            'excitement': 5,
            'fear': 6,
            'sadness': 7,
        }
        return emotion_map.get(emotion, 0)  # Default to amusement if not found

