"""
Validation script to check Clean Accuracy of the trained Latent Classifier.

Tests the classifier on clean latents (timestep=0) to verify performance
on non-noisy data, which is what the model will see during inference guidance.
"""

import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict

# Add shared directory to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
SHARED_DIR = REPO_ROOT / "shared" / "src"
sys.path.insert(0, str(SHARED_DIR))

# Add classifier_guidance src to path
CLASSIFIER_DIR = REPO_ROOT / "approaches" / "classifier_guidance" / "src"
sys.path.insert(0, str(CLASSIFIER_DIR))

from dataset import EmoSetLocalDataset
from model import EmotionLatentClassifier

# Set HuggingFace cache directory
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"
CACHE_DIR = os.path.join(STORAGE_BASE, "cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "hub")

# Emotion mapping (8 emotions) - must match training
EMOTIONS = [
    'amusement',
    'anger',
    'awe',
    'contentment',
    'disgust',
    'excitement',
    'fear',
    'sadness',
]

EMOTION_TO_IDX = {emotion: idx for idx, emotion in enumerate(EMOTIONS)}


def load_vae(model_id: str = "runwayml/stable-diffusion-v1-5", device=None):
    """Load and return the VAE from Stable Diffusion v1.5."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        cache_dir=CACHE_DIR,
    )
    vae = vae.to(device)
    vae.eval()
    vae.requires_grad_(False)
    print(f"VAE loaded on {device}")
    return vae


def load_classifier(weights_path: str, device=None):
    """Load the trained EmotionLatentClassifier."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading classifier from {weights_path}...")
    classifier = EmotionLatentClassifier(num_emotions=8)
    
    # Load weights
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and 'classifier_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['classifier_state_dict'])
    else:
        # Assume the checkpoint is the state dict directly
        classifier.load_state_dict(checkpoint)
    
    classifier = classifier.to(device)
    classifier.eval()
    classifier.requires_grad_(False)
    
    print(f"Classifier loaded on {device}")
    return classifier


def validate_clean_accuracy(
    classifier,
    vae,
    dataloader,
    device,
    num_batches=None,
):
    """
    Validate classifier accuracy on clean latents (timestep=0).
    
    Args:
        classifier: Trained EmotionLatentClassifier
        vae: VAE model for encoding
        dataloader: DataLoader with images and labels
        device: Device to run on
        num_batches: Optional limit on number of batches to process
        
    Returns:
        Dictionary with accuracy metrics
    """
    classifier.eval()
    vae.eval()
    
    total_correct = 0
    total_samples = 0
    
    # Per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    # Process batches
    num_batches_processed = 0
    with torch.no_grad():
        for batch_idx, (images, prompts, emotions) in enumerate(tqdm(dataloader, desc="Validating")):
            if num_batches is not None and batch_idx >= num_batches:
                break
            
            batch_size = images.shape[0]
            images = images.to(device)  # [B, 3, 512, 512]
            
            # CRITICAL: VAE expects images in [0, 1] range, but dataset provides [-1, 1]
            # Convert from [-1, 1] to [0, 1]
            images_normalized = (images + 1.0) / 2.0
            
            # Encode with VAE: [B, 3, 512, 512] -> [B, 4, 64, 64]
            # Process in smaller sub-batches if needed to avoid OOM
            try:
                posterior = vae.encode(images_normalized).latent_dist
                latents = posterior.sample()
            except torch.cuda.OutOfMemoryError:
                # Fallback: process one image at a time
                print(f"\nWarning: OOM with batch_size={batch_size}, processing one at a time...")
                torch.cuda.empty_cache()
                latents_list = []
                for i in range(batch_size):
                    single_image = images_normalized[i:i+1]
                    posterior = vae.encode(single_image).latent_dist
                    single_latent = posterior.sample()
                    latents_list.append(single_latent)
                    torch.cuda.empty_cache()
                latents = torch.cat(latents_list, dim=0)
            
            # CRITICAL: Scale by VAE scaling factor (0.18215 for SD v1.5)
            # Use config.scaling_factor to match training
            latents = latents * vae.config.scaling_factor
            
            # Create timestep tensor of zeros (clean latents, no noise)
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            # Forward pass through classifier
            logits = classifier(latents, timesteps)  # [B, 8]
            
            # Get predictions
            pred_indices = logits.argmax(dim=1)  # [B]
            
            # Get ground truth labels
            emotion_indices = torch.tensor(
                [EMOTION_TO_IDX.get(emotion.lower(), 0) for emotion in emotions],
                device=device
            )
            
            # Calculate accuracy
            correct = (pred_indices == emotion_indices).sum().item()
            total_correct += correct
            total_samples += batch_size
            
            # Clear cache periodically to avoid memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Per-class accuracy
            for i in range(batch_size):
                emotion_idx = emotion_indices[i].item()
                class_total[emotion_idx] += 1
                if pred_indices[i].item() == emotion_idx:
                    class_correct[emotion_idx] += 1
            
            num_batches_processed += 1
    
    # Calculate metrics
    total_accuracy = (total_correct / total_samples * 100) if total_samples > 0 else 0.0
    
    # Per-class accuracy
    per_class_accuracy = {}
    for emotion_idx in range(len(EMOTIONS)):
        if class_total[emotion_idx] > 0:
            acc = class_correct[emotion_idx] / class_total[emotion_idx] * 100
            per_class_accuracy[EMOTIONS[emotion_idx]] = acc
        else:
            per_class_accuracy[EMOTIONS[emotion_idx]] = 0.0
    
    return {
        'total_accuracy': total_accuracy,
        'total_correct': total_correct,
        'total_samples': total_samples,
        'per_class_accuracy': per_class_accuracy,
        'per_class_correct': dict(class_correct),
        'per_class_total': dict(class_total),
        'num_batches': num_batches_processed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate Clean Accuracy of trained Latent Classifier"
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to dataset directory",
    )
    
    parser.add_argument(
        "--weights_path",
        type=str,
        default=str(Path(STORAGE_BASE) / "Weights" / "classifier_guidance" / "classifier.pt"),
        help="Path to classifier checkpoint (default: /Data/yash.bhardwaj/.../Weights/classifier_guidance/classifier.pt)",
    )
    
    parser.add_argument(
        "--num_batches",
        type=int,
        default=50,
        help="Optional limit on number of batches to process (default: 50)",
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for validation (default: 8, reduced for memory efficiency)",
    )
    
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="Stable Diffusion model ID (default: runwayml/stable-diffusion-v1-5)",
    )
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load VAE
    print("\n" + "="*60)
    print("Step 1: Loading VAE")
    print("="*60)
    vae = load_vae(args.model_id, device=device)
    
    # Load classifier
    print("\n" + "="*60)
    print("Step 2: Loading Classifier")
    print("="*60)
    classifier = load_classifier(args.weights_path, device=device)
    
    # Load dataset
    print("\n" + "="*60)
    print("Step 3: Loading Dataset")
    print("="*60)
    print(f"Loading dataset from {args.data_dir}...")
    dataset = EmoSetLocalDataset(data_dir=args.data_dir, image_size=512)
    print(f"Loaded {len(dataset)} examples")
    
    # Limit dataset size if needed (for faster validation)
    if args.num_batches is not None:
        max_samples = args.num_batches * args.batch_size
        if len(dataset) > max_samples:
            print(f"Limiting to first {max_samples} samples for faster validation")
            dataset = torch.utils.data.Subset(dataset, range(max_samples))
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    print(f"Created DataLoader with {len(dataloader)} batches")
    if args.num_batches is not None:
        print(f"Will process {min(args.num_batches, len(dataloader))} batches")
    
    # Run validation
    print("\n" + "="*60)
    print("Step 4: Running Validation (Clean Accuracy @ timestep=0)")
    print("="*60)
    metrics = validate_clean_accuracy(
        classifier=classifier,
        vae=vae,
        dataloader=dataloader,
        device=device,
        num_batches=args.num_batches,
    )
    
    # Print results
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    print(f"\nTotal Accuracy: {metrics['total_accuracy']:.2f}%")
    print(f"Correct: {metrics['total_correct']} / {metrics['total_samples']}")
    print(f"Batches processed: {metrics['num_batches']}")
    
    print("\n" + "-"*60)
    print("Per-Class Accuracy:")
    print("-"*60)
    for emotion in EMOTIONS:
        emotion_idx = EMOTION_TO_IDX[emotion]
        acc = metrics['per_class_accuracy'][emotion]
        correct = metrics['per_class_correct'].get(emotion_idx, 0)
        total = metrics['per_class_total'].get(emotion_idx, 0)
        print(f"  {emotion.capitalize():12s}: {acc:6.2f}% ({correct:4d}/{total:4d})")
    
    print("\n" + "="*60)
    
    # Interpretation
    print("\nInterpretation:")
    if metrics['total_accuracy'] >= 80:
        print("  ✓ Excellent clean accuracy! Model performs well on clean latents.")
    elif metrics['total_accuracy'] >= 60:
        print("  ⚠ Moderate clean accuracy. Model may benefit from more training.")
    else:
        print("  ✗ Low clean accuracy. Model may need more training or architecture changes.")
    
    # Check for class imbalance
    min_class_acc = min(metrics['per_class_accuracy'].values())
    max_class_acc = max(metrics['per_class_accuracy'].values())
    if max_class_acc - min_class_acc > 30:
        print("  ⚠ Significant class imbalance detected. Consider class weighting or more balanced training.")
    else:
        print("  ✓ Class accuracies are relatively balanced.")


if __name__ == "__main__":
    main()

