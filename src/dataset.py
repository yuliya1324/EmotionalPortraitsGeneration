"""
Dataset class for loading EmoSet-118K with generated captions.
Handles loading from disk and preprocessing images for Stable Diffusion training.
"""

import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from PIL import Image
import torchvision.transforms as transforms


class EmoSetLocalDataset(Dataset):
    """
    PyTorch Dataset for EmoSet-118K with BLIP-generated captions.
    
    Loads images, captions, and emotions from a HuggingFace dataset saved to disk.
    Preprocesses images for Stable Diffusion (512x512, normalized to [-1, 1]).
    Constructs prompts by appending emotion tokens to captions.
    """
    
    def __init__(self, data_dir: str, image_size: int = 512):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Path to the dataset directory (saved with save_to_disk)
            image_size: Target image size for resizing (default: 512)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Load dataset from disk
        print(f"Loading dataset from {data_dir}...")
        self.dataset = load_from_disk(data_dir)
        print(f"Loaded {len(self.dataset)} examples")
        
        # Define image transforms for Stable Diffusion
        # Images need to be resized to 512x512 and normalized to [-1, 1]
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # Converts to [0, 1] range
            transforms.Normalize([0.5], [0.5])  # Normalizes to [-1, 1] range
        ])
        
        # Map emotion names to emotion tokens
        self.emotion_to_token = {
            'amusement': '<amusement>',
            'awe': '<awe>',
            'contentment': '<contentment>',
            'excitement': '<excitement>',
            'anger': '<anger>',
            'disgust': '<disgust>',
            'fear': '<fear>',
            'sadness': '<sadness>',
        }
        
        # Validate dataset structure
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate that the dataset has required columns."""
        required_columns = ['image', 'emotion']
        missing_columns = [col for col in required_columns if col not in self.dataset.column_names]
        
        if missing_columns:
            raise ValueError(
                f"Dataset missing required columns: {missing_columns}. "
                f"Available columns: {self.dataset.column_names}"
            )
        
        # Check if we have captions (either 'generated_caption' or 'caption')
        if 'generated_caption' not in self.dataset.column_names and 'caption' not in self.dataset.column_names:
            print("Warning: No caption column found. Will use empty captions with only emotion tokens.")
    
    def __len__(self):
        """Return the number of examples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Tuple of (image_tensor, prompt_string, emotion_string)
        """
        example = self.dataset[idx]
        
        # Load and preprocess image
        image = example['image']
        if not isinstance(image, Image.Image):
            # If it's a path, load it
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            else:
                # Try to convert if it's already an array
                image = Image.fromarray(image).convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        # Get emotion
        emotion = example.get('emotion', 'amusement').lower()
        
        # Get caption (prefer 'generated_caption', fallback to 'caption')
        caption = example.get('generated_caption') or example.get('caption', '')
        
        # Construct prompt: caption + emotion token
        emotion_token = self.emotion_to_token.get(emotion, f'<{emotion}>')
        if caption:
            prompt = f"{caption} {emotion_token}"
        else:
            prompt = emotion_token
        
        return image_tensor, prompt, emotion
