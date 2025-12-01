#!/usr/bin/env python3
"""
Experiment runner for different approaches.
Manages training and inference across different approaches and dataset sizes.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Base storage paths
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"
WEIGHTS_BASE = os.path.join(STORAGE_BASE, "Weights")
LOGS_BASE = os.path.join(STORAGE_BASE, "Logs")
DATASETS_BASE = os.path.join(STORAGE_BASE, "Datasets")

# Repository root
REPO_ROOT = Path(__file__).parent.absolute()
APPROACHES_DIR = REPO_ROOT / "approaches"
SHARED_DIR = REPO_ROOT / "shared"


def get_available_approaches():
    """List all available approaches."""
    if not APPROACHES_DIR.exists():
        return []
    approaches = [d.name for d in APPROACHES_DIR.iterdir() 
                  if d.is_dir() and (d / "src" / "train.py").exists()]
    return sorted(approaches)


def get_dataset_path(dataset_size: str) -> str:
    """
    Get the path to a dataset variant.
    
    Args:
        dataset_size: Dataset size identifier (e.g., "10K", "30K", "25K")
        
    Returns:
        Path to the dataset directory
    """
    # Normalize dataset size (handle "10K", "10k", "10000")
    size_normalized = dataset_size.upper()
    if not size_normalized.endswith("K"):
        # Assume it's a number, convert to K format
        try:
            num = int(size_normalized)
            size_normalized = f"{num // 1000}K"
        except ValueError:
            pass
    
    dataset_name = f"emoset_captioned_{size_normalized.lower()}"
    dataset_path = os.path.join(DATASETS_BASE, dataset_name)
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            f"Please generate it first using:\n"
            f"  python shared/src/preprocessing.py \\\n"
            f"    --subset_size {size_normalized.replace('K', '000')} \\\n"
            f"    --output_dir {dataset_path}"
        )
    
    return dataset_path


def get_output_paths(dataset_size: str, approach: str):
    """
    Get output paths for weights and logs.
    
    Args:
        dataset_size: Dataset size identifier (e.g., "10K", "30K")
        approach: Approach name (e.g., "baseline_lora")
        
    Returns:
        Tuple of (weights_dir, logs_dir)
    """
    size_normalized = dataset_size.upper()
    if not size_normalized.endswith("K"):
        try:
            num = int(size_normalized)
            size_normalized = f"{num // 1000}K"
        except ValueError:
            pass
    
    weights_dir = os.path.join(WEIGHTS_BASE, size_normalized, approach)
    logs_dir = os.path.join(LOGS_BASE, size_normalized, approach)
    
    # Create directories if they don't exist
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return weights_dir, logs_dir


def run_training(approach: str, dataset_size: str, **kwargs):
    """
    Run training for a specific approach and dataset size.
    
    Args:
        approach: Approach name
        dataset_size: Dataset size identifier
        **kwargs: Additional training arguments
    """
    # Validate approach exists
    available = get_available_approaches()
    if approach not in available:
        raise ValueError(
            f"Approach '{approach}' not found. Available: {available}"
        )
    
    # Get paths
    dataset_path = get_dataset_path(dataset_size)
    weights_dir, logs_dir = get_output_paths(dataset_size, approach)
    
    # Get approach train script
    train_script = APPROACHES_DIR / approach / "src" / "train.py"
    if not train_script.exists():
        raise FileNotFoundError(f"Training script not found: {train_script}")
    
    # Build command
    cmd = [
        "accelerate", "launch",
        str(train_script),
        "--data_dir", dataset_path,
        "--output_dir", weights_dir,
        "--log_dir", logs_dir,
    ]
    
    # Add additional arguments
    # Note: train.py uses underscores in argument names (e.g., --batch_size)
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print("="*70)
    print(f"Running Training")
    print("="*70)
    print(f"Approach: {approach}")
    print(f"Dataset Size: {dataset_size}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Weights Dir: {weights_dir}")
    print(f"Logs Dir: {logs_dir}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("="*70)
    print()
    
    # Run training
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def run_inference(approach: str, dataset_size: str, prompt: str = "A photo of a park", **kwargs):
    """
    Run inference for a specific approach.
    
    Args:
        approach: Approach name
        dataset_size: Dataset size identifier
        prompt: Base prompt for generation
        **kwargs: Additional inference arguments
    """
    # Validate approach exists
    available = get_available_approaches()
    if approach not in available:
        raise ValueError(
            f"Approach '{approach}' not found. Available: {available}"
        )
    
    # Get paths
    weights_dir, logs_dir = get_output_paths(dataset_size, approach)
    
    # Get approach inference script
    inference_script = APPROACHES_DIR / approach / "src" / "inference.py"
    if not inference_script.exists():
        raise FileNotFoundError(f"Inference script not found: {inference_script}")
    
    # Create inference output directory
    inference_output_dir = os.path.join(logs_dir, "inference")
    os.makedirs(inference_output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        "python", str(inference_script),
        "--prompt", prompt,
        "--lora_path", weights_dir,
        "--learned_embeds_path", os.path.join(weights_dir, "learned_embeds.bin"),
        "--tokenizer_info_path", os.path.join(weights_dir, "tokenizer_info.json"),
        "--output_path", os.path.join(inference_output_dir, f"{prompt.replace(' ', '_').replace('/', '_')}.png"),
    ]
    
    # Add additional arguments
    # Note: inference.py uses underscores in argument names (e.g., --num_inference_steps)
    for key, value in kwargs.items():
        if value is not None:
            cmd.extend([f"--{key}", str(value)])
    
    print("="*70)
    print(f"Running Inference")
    print("="*70)
    print(f"Approach: {approach}")
    print(f"Dataset Size: {dataset_size}")
    print(f"Prompt: {prompt}")
    print(f"Weights Dir: {weights_dir}")
    print("="*70)
    print(f"Command: {' '.join(cmd)}")
    print("="*70)
    print()
    
    # Run inference
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments for different approaches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available approaches
  python run_experiment.py --list-approaches
  
  # Train baseline_lora on 30K dataset
  python run_experiment.py train --approach baseline_lora --dataset-size 30K
  
  # Train with custom parameters
  python run_experiment.py train --approach baseline_lora --dataset-size 30K \\
      --batch-size 8 --num-epochs 7 --lora-r 32
  
  # Run inference
  python run_experiment.py inference --approach baseline_lora --dataset-size 30K \\
      --prompt "A living room"
  
  # Evaluate model
  python run_experiment.py evaluate --approach baseline_lora --dataset-size 30K
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # List approaches command
    list_parser = subparsers.add_parser('list-approaches', help='List available approaches')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--approach', type=str, required=True,
                             help='Approach name (e.g., baseline_lora)')
    train_parser.add_argument('--dataset-size', type=str, default='25K',
                             help='Dataset size (e.g., 10K, 30K, 25K). Default: 25K')
    
    # Training arguments (pass through to train.py)
    train_parser.add_argument('--batch-size', type=int, default=4)
    train_parser.add_argument('--num-epochs', type=int, default=10)
    train_parser.add_argument('--lr-lora', type=float, default=1e-4)
    train_parser.add_argument('--lr-embeddings', type=float, default=5e-3)
    train_parser.add_argument('--lora-r', type=int, default=16)
    train_parser.add_argument('--lora-alpha', type=int, default=32)
    train_parser.add_argument('--save-steps', type=int, default=500)
    train_parser.add_argument('--validation-steps', type=int, default=1000)
    train_parser.add_argument('--gradient-accumulation-steps', type=int, default=3)
    train_parser.add_argument('--seed', type=int, default=42)
    train_parser.add_argument('--init-word', type=str, default=None,
                             help='Fallback word for token initialization (default: use emotion words)')
    train_parser.add_argument('--emotion-reg-weight', type=float, default=0.05,
                             help='Weight for emotion regularization loss (default: 0.05)')
    train_parser.add_argument('--early-stopping-patience', type=int, default=5,
                             help='Early stopping patience in epochs (default: 5)')
    train_parser.add_argument('--resume-from', type=str, default=None)
    
    # Inference command
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--approach', type=str, required=True,
                                 help='Approach name')
    inference_parser.add_argument('--dataset-size', type=str, required=True,
                                 help='Dataset size')
    inference_parser.add_argument('--prompt', type=str, default='A photo of a park',
                                 help='Base prompt for generation')
    inference_parser.add_argument('--seed', type=int, default=42)
    inference_parser.add_argument('--num-inference-steps', type=int, default=50)
    inference_parser.add_argument('--guidance-scale', type=float, default=7.5)
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Evaluate model with CLIP')
    evaluate_parser.add_argument('--approach', type=str, required=True,
                                help='Approach name')
    evaluate_parser.add_argument('--dataset-size', type=str, required=True,
                                help='Dataset size')
    evaluate_parser.add_argument('--skip-generation', action='store_true',
                                help='Skip image generation, use existing images')
    evaluate_parser.add_argument('--seed', type=int, default=42)
    evaluate_parser.add_argument('--num-inference-steps', type=int, default=50)
    evaluate_parser.add_argument('--guidance-scale', type=float, default=7.5)
    
    args = parser.parse_args()
    
    if args.command == 'list-approaches' or (not args.command and 'list' in sys.argv):
        approaches = get_available_approaches()
        if approaches:
            print("Available approaches:")
            for app in approaches:
                print(f"  - {app}")
        else:
            print("No approaches found. Create an approach directory in approaches/")
        return 0
    
    elif args.command == 'train':
        # Convert args to dict, excluding command and approach/dataset-size
        train_kwargs = {
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'lr_lora': args.lr_lora,
            'lr_embeddings': args.lr_embeddings,
            'lora_r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'save_steps': args.save_steps,
            'validation_steps': args.validation_steps,
            'gradient_accumulation_steps': args.gradient_accumulation_steps,
            'seed': args.seed,
            'init_word': args.init_word,
            'emotion_reg_weight': args.emotion_reg_weight,
            'early_stopping_patience': args.early_stopping_patience,
        }
        if args.resume_from:
            train_kwargs['resume_from'] = args.resume_from
        
        return run_training(args.approach, args.dataset_size, **train_kwargs)
    
    elif args.command == 'inference':
        inference_kwargs = {
            'seed': args.seed,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
        }
        return run_inference(args.approach, args.dataset_size, args.prompt, **inference_kwargs)
    
    elif args.command == 'evaluate':
        # Run evaluation script
        evaluate_script = REPO_ROOT / "evaluate.py"
        cmd = [
            "python", str(evaluate_script),
            "--approach", args.approach,
            "--dataset-size", args.dataset_size,
            "--seed", str(args.seed),
            "--num-inference-steps", str(args.num_inference_steps),
            "--guidance-scale", str(args.guidance_scale),
        ]
        if args.skip_generation:
            cmd.append("--skip-generation")
        
        print("="*70)
        print(f"Running Evaluation")
        print("="*70)
        print(f"Approach: {args.approach}")
        print(f"Dataset Size: {args.dataset_size}")
        print("="*70)
        
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        return result.returncode
    
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
