"""
Training script for the Noise-Aware Latent Classifier.

Trains a CNN classifier on noisy latents to predict emotion labels.
Uses pretrained VAE to encode images to latents, adds noise at random timesteps,
and trains the classifier to predict the emotion from noisy latents.
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from diffusers import DDPMScheduler, AutoencoderKL
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
from pathlib import Path

# Add shared directory to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
SHARED_DIR = REPO_ROOT / "shared" / "src"
sys.path.insert(0, str(SHARED_DIR))

from dataset import EmoSetLocalDataset
from model import EmotionLatentClassifier
from data_utils import CachedLatentsDataset

# Set HuggingFace cache directory
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"
CACHE_DIR = os.path.join(STORAGE_BASE, "cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "hub")

# Emotion mapping (8 emotions)
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


def setup_device():
    """Detect and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def load_vae(model_id: str = "runwayml/stable-diffusion-v1-5", device=None):
    """
    Load pretrained VAE and freeze it.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        
    Returns:
        VAE model (frozen)
    """
    print(f"Loading VAE from {model_id}...")
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        cache_dir=CACHE_DIR,
    )
    vae.requires_grad_(False)
    vae.eval()
    if device:
        vae = vae.to(device)
    print("VAE loaded and frozen.")
    return vae


def main():
    parser = argparse.ArgumentParser(description="Train Noise-Aware Latent Classifier")
    
    # Dataset arguments
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Path to dataset directory")
    parser.add_argument("--cache_dir", type=str, default=None,
                       help="Directory for caching VAE latents (default: /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/latents_cache_classifier_guidance)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for classifier weights (default: /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/classifier_guidance)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Training batch size (default: 64)")
    parser.add_argument("--num_epochs", type=int, default=15,
                       help="Number of training epochs (default: 15)")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint to resume from (optional)")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="HuggingFace model ID for VAE (default: runwayml/stable-diffusion-v1-5)")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
    )
    
    # Setup device
    device = accelerator.device
    
    # Determine output directory
    if args.output_dir is None:
        # Default: Use STORAGE_BASE/Weights/classifier_guidance
        weights_dir = Path(STORAGE_BASE) / "Weights" / "classifier_guidance"
    else:
        weights_dir = Path(args.output_dir)
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine cache directory
    if args.cache_dir is None:
        # Default: Use STORAGE_BASE/Datasets/latents_cache_classifier_guidance
        cache_dir = Path(STORAGE_BASE) / "Datasets" / "latents_cache_classifier_guidance"
    else:
        cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("Training Noise-Aware Latent Classifier")
    print("="*70)
    print(f"Dataset: {args.data_dir}")
    print(f"Cache directory: {cache_dir}")
    print(f"Output: {weights_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Learning rate: {args.lr}")
    print("="*70)
    
    # Load VAE (frozen) - MUST be loaded before dataset wrapper
    print("\nLoading VAE...")
    vae = load_vae(args.model_id, device=device)
    
    # Initialize cached latents dataset (handles base dataset internally)
    print(f"\nInitializing cached latents dataset from {args.data_dir}...")
    dataset = CachedLatentsDataset(
        root_dir=args.data_dir,
        cache_dir=str(cache_dir),
        vae=vae,
        device=device,
    )
    
    # Count cached vs uncached
    cached_count = sum(1 for idx in range(len(dataset)) 
                       if (cache_dir / f"latent_{idx}.pt").exists())
    uncached_count = len(dataset) - cached_count
    print(f"  Cached latents: {cached_count}/{len(dataset)}")
    print(f"  Uncached (will encode on first access): {uncached_count}")
    if uncached_count > 0:
        print(f"  Note: First epoch will be slower as it builds the cache")
        print(f"        Subsequent epochs will be much faster!")
    
    # CRITICAL: num_workers must be 0 because CachedLatentsDataset uses VAE
    # VAE models cannot be pickled/sent to worker processes
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Must be 0 - VAE cannot be pickled for multiprocessing
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    # Initialize classifier
    print("\nInitializing classifier...")
    classifier = EmotionLatentClassifier(num_emotions=8)
    classifier = classifier.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)
    
    # Resume from checkpoint if provided
    start_epoch = 0
    best_loss = float('inf')
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if resume_path.exists():
            print(f"\nResuming from checkpoint: {resume_path}")
            checkpoint = torch.load(resume_path, map_location=device)
            
            # Load model state
            if 'model_state_dict' in checkpoint:
                classifier.load_state_dict(checkpoint['model_state_dict'])
            else:
                classifier.load_state_dict(checkpoint)
            
            # Load optimizer state if available
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Get starting epoch and best loss
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                print(f"  Resuming from epoch {start_epoch}")
            if 'loss' in checkpoint:
                best_loss = checkpoint['loss']
                print(f"  Previous best loss: {best_loss:.4f}")
            if 'accuracy' in checkpoint:
                print(f"  Previous accuracy: {checkpoint['accuracy']:.4f}")
        else:
            print(f"Warning: Resume checkpoint not found: {resume_path}")
            print("  Starting training from scratch")
    
    # Scheduler for noise
    scheduler = DDPMScheduler.from_pretrained(
        args.model_id,
        subfolder="scheduler",
        cache_dir=CACHE_DIR,
    )
    
    # Prepare with accelerator
    classifier, optimizer, dataloader = accelerator.prepare(
        classifier, optimizer, dataloader
    )
    
    # Create learning rate scheduler (OneCycleLR)
    # Calculate total steps now that dataloader is prepared
    num_steps_per_epoch = len(dataloader)
    total_steps = num_steps_per_epoch * args.num_epochs
    
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% of steps for warmup
        anneal_strategy='cos',
    )
    
    # If resuming, step the scheduler to the correct position
    if start_epoch > 0:
        steps_to_skip = start_epoch * num_steps_per_epoch
        print(f"\nResuming LR scheduler: stepping {steps_to_skip} steps to match epoch {start_epoch + 1}")
        for _ in range(steps_to_skip):
            lr_scheduler.step()
    
    print(f"\nLearning Rate Scheduler: OneCycleLR")
    print(f"  Max LR: {args.lr}")
    print(f"  Total steps: {total_steps} ({num_steps_per_epoch} steps/epoch × {args.num_epochs} epochs)")
    if start_epoch > 0:
        print(f"  Current step: {lr_scheduler.last_epoch + 1} (resumed from epoch {start_epoch + 1})")
    
    # Training loop
    
    print("\nStarting training...")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}/{args.num_epochs}")
    else:
        print(f"Starting from epoch 1/{args.num_epochs}")
    
    for epoch in range(start_epoch, args.num_epochs):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process,
        )
        
        for batch_idx, (latents, emotion_indices) in enumerate(progress_bar):
            # Latents are already pre-encoded and cached (from CachedLatentsDataset)
            # Move to device
            latents = latents.to(device)  # [B, 4, 64, 64]
            
            # Emotion indices are already provided as tensors from CachedLatentsDataset
            emotion_indices = emotion_indices.to(device)  # [B]
            
            # Sample random timesteps (0-1000)
            batch_size = latents.shape[0]
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps,
                (batch_size,), device=device
            )
            
            # Sample noise
            noise = torch.randn_like(latents)
            
            # Add noise to latents
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)
            
            # Forward pass
            optimizer.zero_grad()
            logits = classifier(noisy_latents, timesteps)  # [B, 8]
            
            # Compute loss
            loss = criterion(logits, emotion_indices)
            
            # Backward pass
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()  # Step learning rate scheduler
            
            # Get current learning rate for logging
            current_lr = lr_scheduler.get_last_lr()[0]
            
            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == emotion_indices).sum().item()
            total += batch_size
            
            # Update progress bar (include learning rate)
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = correct / total
            progress_bar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'acc': f'{accuracy:.4f}',
                'lr': f'{current_lr:.2e}',
            })
        
        # Epoch summary
        avg_loss = total_loss / num_steps_per_epoch
        accuracy = correct / total
        final_lr = lr_scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Learning Rate: {final_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if accelerator.is_main_process:
                checkpoint_path = weights_dir / "classifier_large.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(classifier).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy,
                }, checkpoint_path)
                print(f"  ✓ Saved best model to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model saved to: {weights_dir / 'classifier_large.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()


        final_lr = lr_scheduler.get_last_lr()[0]
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs}:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        print(f"  Learning Rate: {final_lr:.2e}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            if accelerator.is_main_process:
                checkpoint_path = weights_dir / "classifier_large.pt"
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': accelerator.unwrap_model(classifier).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy,
                }, checkpoint_path)
                print(f"  ✓ Saved best model to {checkpoint_path}")
    
    print("\n" + "="*70)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model saved to: {weights_dir / 'classifier_large.pt'}")
    print("="*70)


if __name__ == "__main__":
    main()

