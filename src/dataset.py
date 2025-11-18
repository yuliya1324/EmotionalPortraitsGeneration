"""
Dataset module for loading and processing locally saved EmoSet-118K dataset.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from datasets import load_from_disk
from typing import Tuple


class EmoSetLocalDataset(Dataset):
    """
    Dataset class for locally saved EmoSet-118K with generated captions.
    
    Maps integer emotion labels to special tokens and processes images for training.
    """
    
    # Integer emotion label to token mapping
    # Assuming standard emotion label mapping (adjust if needed)
    EMOTION_MAP = {
        0: '<amusement>',
        1: '<awe>',
        2: '<contentment>',
        3: '<excitement>',
        4: '<anger>',
        5: '<disgust>',
        6: '<fear>',
        7: '<sadness>',
    }
    
    # All emotion tokens
    EMOTION_TOKENS = list(EMOTION_MAP.values())
    
    def __init__(
        self,
        data_dir: str = "./data/emoset_captioned_10k",
        image_size: int = 512,
    ):
        """
        Initialize the EmoSet dataset from local disk.
        
        Args:
            data_dir: Path to local dataset directory
            image_size: Target image size for resizing
        """
        self.data_dir = data_dir
        self.image_size = image_size
        
        # Load dataset from local disk
        print(f"Loading dataset from {data_dir}...")
        try:
            self.dataset = load_from_disk(data_dir)
            print(f"Loaded {len(self.dataset)} examples")
        except Exception as e:
            raise ValueError(
                f"Failed to load dataset from {data_dir}. "
                f"Make sure you've run preprocess.py first. Error: {e}"
            )
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
        ])
        
        # Validate dataset structure
        self._validate_dataset()
        
        # Print emotion distribution
        self._print_emotion_distribution()
    
    def _validate_dataset(self):
        """Validate that the dataset has required columns."""
        if len(self.dataset) == 0:
            raise ValueError("Dataset is empty")
        
        example = self.dataset[0]
        required_cols = ['image', 'emotion', 'generated_caption']
        
        for col in required_cols:
            if col not in example:
                raise ValueError(
                    f"Dataset missing required column: {col}. "
                    f"Available columns: {list(example.keys())}"
                )
    
    def _print_emotion_distribution(self):
        """Print distribution of emotion labels."""
        emotion_counts = {}
        for example in self.dataset:
            emotion = example.get('emotion', 0)
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\nEmotion distribution:")
        for emotion_id, count in sorted(emotion_counts.items()):
            token = self.EMOTION_MAP.get(emotion_id, f"<unknown_{emotion_id}>")
            print(f"  {token} (id={emotion_id}): {count} examples")
    
    def _get_emotion_token(self, emotion: int) -> str:
        """
        Get the special token for a given integer emotion label.
        
        Args:
            emotion: Integer emotion label
            
        Returns:
            Special token string
        """
        token = self.EMOTION_MAP.get(emotion)
        if token is None:
            print(f"Warning: Unknown emotion ID {emotion}, defaulting to <amusement>")
            return self.EMOTION_MAP[0]  # Default to amusement
        return token
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, int]:
        """
        Get a single dataset item.
        
        Args:
            idx: Item index
            
        Returns:
            Tuple of (image_tensor, prompt, emotion_label)
        """
        example = self.dataset[idx]
        
        # Load and transform image
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        
        image_tensor = self.transform(image)
        
        # Get generated caption
        generated_caption = example.get('generated_caption', 'A photo of a scene')
        if not generated_caption or generated_caption.strip() == '':
            generated_caption = 'A photo of a scene'
        
        # Get emotion token
        emotion = example.get('emotion', 0)
        emotion_token = self._get_emotion_token(emotion)
        
        # Construct prompt: "caption emotion_token"
        prompt = f"{generated_caption.strip()} {emotion_token}"
        
        return image_tensor, prompt, emotion


if __name__ == "__main__":
    """Test dataset loading and mapping logic."""
    import sys
    
    try:
        # Initialize dataset
        dataset = EmoSetLocalDataset()
        
        # Print dataset info
        print(f"\nDataset loaded successfully!")
        print(f"Total examples: {len(dataset)}")
        print(f"Emotion tokens: {dataset.EMOTION_TOKENS}")
        
        # Get first example
        if len(dataset) > 0:
            image, prompt, emotion = dataset[0]
            print(f"\nFirst example:")
            print(f"  Emotion label (int): {emotion}")
            print(f"  Emotion token: {dataset._get_emotion_token(emotion)}")
            print(f"  Prompt: {prompt}")
            print(f"  Image shape: {image.shape}")
            print(f"  Image dtype: {image.dtype}")
            print(f"  Image range: [{image.min():.2f}, {image.max():.2f}]")
        
        # Test emotion mapping
        print(f"\nTesting emotion mapping:")
        test_emotions = [0, 1, 2, 3, 4, 5, 6, 7, 99]  # Include invalid ID
        for emo_id in test_emotions:
            token = dataset._get_emotion_token(emo_id)
            print(f"  Emotion ID {emo_id} -> {token}")
        
        print("\nDataset validation passed!")
        
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
