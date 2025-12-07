#!/usr/bin/env python3
"""
Evaluation script for emotion-conditioned image generation.

Uses EmotionCLIP (https://huggingface.co/jiangchengchengNLP/EmotionCLIP) to measure:
1. Emotion Accuracy: argmax(similarity) == target emotion → "correct prediction rate"
2. Emotion Strength: softmax[target] → "degree of emotional match"

Generates images for 25 prompts × 8 emotions = 200 images, then evaluates each.
"""

import os
import sys
import argparse
import json
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import subprocess
from transformers import ViTImageProcessor, ViTForImageClassification

from train_text_to_image_lora import generate_with_emotion

# Add paths for imports
REPO_ROOT = Path(__file__).parent.absolute()

# Storage paths
STORAGE_BASE = "/Data/iuliia.korotkova/output/"

# Storage paths
CACHE_DIR = "/Data/iuliia.korotkova/cache"

# EmotionCLIP path
EMOTIONCLIP_PATH = "/users/eleves-a/2025/iuliia.korotkova/EmotionalPortraitsGeneration/EmotionCLIP"

# ============================================================================
# Evaluation Prompts (25 diverse scene prompts)
# ============================================================================
EVALUATION_PROMPTS = [
    "A quiet residential street with parked cars and trees",
    "A narrow alley between old buildings with a few posters on the walls",
    "A wide city square with a fountain and surrounding cafés",
    "A subway platform with signs, benches, and overhead lighting",
    "The inside of an empty train carriage with seats and windows",
    "A small convenience store aisle with shelves of snacks and drinks",
    "An outdoor market with stalls of fruits, vegetables, and people walking",
    "A calm beach with gentle waves and a cloudy sky",
    "A rocky shoreline with tide pools and scattered seaweed",
    "A forest path with tall trees and sunlight filtering through leaves",
    "A grassy field with a lone tree under an open sky",
    "A mountain lake with reflections of peaks in the water",
    "A wooden cabin in the snow with smoke coming from a chimney",
    "A desert road stretching into the distance with a few shrubs",
    "A bridge over a river with boats passing underneath",
    "A harbor with docked boats, ropes, and shipping containers",
    "A rooftop view of a city skyline at dusk",
    "A park with a playground, swings, and a bench nearby",
    "A museum gallery with paintings and soft lighting",
    "A library aisle with tall bookshelves and reading tables",
    "A classroom with desks, a whiteboard, and a window",
    "A kitchen counter with utensils, a cutting board, and a bowl of fruit",
    "A living room with a sofa, a lamp, and curtains by a window",
    "A long hallway in a large building with doors and ceiling lights",
    "A stairwell with metal railings and concrete walls",
]

# EmoSet emotions
EMOTIONS_EMOSET = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]
EMOTIONS_RAFDB = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]


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


def setup_emotionclip():
    """Clone EmotionCLIP repository if not present."""
    if not os.path.exists(EMOTIONCLIP_PATH):
        print(f"Cloning EmotionCLIP to {EMOTIONCLIP_PATH}...")
        os.makedirs(os.path.dirname(EMOTIONCLIP_PATH), exist_ok=True)
        result = subprocess.run(
            ["git", "clone", "https://huggingface.co/jiangchengchengNLP/EmotionCLIP", EMOTIONCLIP_PATH],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error cloning EmotionCLIP: {result.stderr}")
            raise RuntimeError("Failed to clone EmotionCLIP")
        print("EmotionCLIP cloned successfully")
    else:
        print(f"EmotionCLIP found at {EMOTIONCLIP_PATH}")


def load_emotionclip_model(device):
    """Load EmotionCLIP model for emotion evaluation."""
    # Setup EmotionCLIP if needed
    setup_emotionclip()
    
    # Add EmotionCLIP to path
    sys.path.insert(0, EMOTIONCLIP_PATH)
    
    # Change to EmotionCLIP directory temporarily (needed for relative path in EmotionCLIP.py)
    original_cwd = os.getcwd()
    try:
        os.chdir(EMOTIONCLIP_PATH)
        # Import EmotionCLIP components
        from EmotionCLIP import model, preprocess, tokenizer
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    print(f"Loaded EmotionCLIP model (device: {model.device}, dtype: {model.dtype})")
    
    return model, preprocess, tokenizer


def load_vit_model(device):
    processor = ViTImageProcessor.from_pretrained(
        'abhilash88/face-emotion-detection', 
        cache_dir=CACHE_DIR
        )
    model = ViTForImageClassification.from_pretrained(
        'abhilash88/face-emotion-detection', 
        cache_dir=CACHE_DIR
        )
    model = model.to(device)
    return processor, model


def load_generation_pipeline(weights_dir: str, device):
    """Load the trained Stable Diffusion pipeline with LoRA and embeddings."""
    from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
    
    model_base = "CompVis/stable-diffusion-v1-4"
    
    print(f"\nLoading generation pipeline from {weights_dir}...")
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_base, 
        torch_dtype=torch.float16, 
        cache_dir=CACHE_DIR
        )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    # Load LoRA weights
    pipe.unet.load_attn_procs(weights_dir)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def generate_images(
    pipe,
    task: str,
    approach: str,
    prompts: list,
    emotions: list,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
):
    """
    Generate images for all prompt-emotion combinations.
    
    Args:
        pipe: Stable Diffusion pipeline
        prompts: List of scene prompts
        emotions: List of emotions
        output_dir: Directory to save generated images
        seed: Random seed for reproducibility
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        prompt_template: Template for combining prompt and emotion
        
    Returns:
        List of dicts with image_path, prompt, emotion, full_prompt
    """
    os.makedirs(output_dir, exist_ok=True)
    device = pipe.device

    if task == "multimodal":
        emotion_embedding = nn.Embedding(8, pipe.text_encoder.config.hidden_size)
        emotion_embedding.load_state_dict(
            os.path.join(STORAGE_BASE, approach, "emotion_embedding.pth")
            )
        emotion_embedding.eval()
        emotion_embedding = emotion_embedding.to(device)
    
    results = []
    total = len(prompts) * len(emotions)
    
    print(f"\nGenerating {total} images ({len(prompts)} prompts × {len(emotions)} emotions)...")
    
    pbar = tqdm(total=total, desc="Generating images")
    
    for prompt_idx, prompt in enumerate(prompts):
        for emotion_idx, emotion in enumerate(emotions):
            
            # Generate with fixed seed for reproducibility across emotions
            generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
            
            try:
                if task == "label2image":
                    image = pipe(
                        emotion, 
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]

                elif task == "multimodal":
                    image = generate_with_emotion(
                        pipe, 
                        emotion_embedding, 
                        prompt, 
                        emotion_idx, 
                        num_inference_steps=25, 
                        generator=generator
                    )
                
                # Save image
                if not image.getbbox():
                    print(f"\n  Image is empty. Skipping...")
                else:
                    filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                    image_path = os.path.join(output_dir, filename)
                    image.save(image_path)
                    
                    results.append({
                        "image_path": image_path,
                        "prompt": prompt,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "prompt_idx": prompt_idx,
                    })
                
            except Exception as e:
                print(f"\n  Error generating {filename}: {e}")
            
            pbar.update(1)
    
    pbar.close()
    print(f"Generated {len(results)} images to {output_dir}")
    
    return results


def evaluate_emotions_emoclip(
    emotionclip_model,
    emotionclip_preprocess,
    emotionclip_tokenizer,
    results: list,
    emotions: list,
):
    """
    Evaluate generated images using EmotionCLIP for emotion classification.
    
    For each image:
    1. Compute EmotionCLIP similarity with all 8 emotion labels
    2. Apply softmax to get probabilities
    3. Calculate accuracy (argmax == target) and strength (softmax[target])
    
    Args:
        emotionclip_model: EmotionCLIP model
        emotionclip_preprocess: EmotionCLIP preprocessing function
        emotionclip_tokenizer: EmotionCLIP tokenizer
        results: List of generation results
        
    Returns:
        Updated results with evaluation metrics, and aggregate metrics
    """
    print("\nEvaluating generated images with EmotionCLIP...")
    
    # Prepare text inputs using EmotionCLIP's expected format
    text_list = [f"This picture conveys a sense of {emotion}" for emotion in emotions]
    text_input = emotionclip_tokenizer(text_list)
    
    # Move text input to model device
    text_input = text_input.to(device=emotionclip_model.device)
    
    # Evaluate each image
    correct_predictions = 0
    total_strength = 0.0
    
    per_emotion_correct = {emotion: 0 for emotion in emotions}
    per_emotion_total = {emotion: 0 for emotion in emotions}
    per_emotion_strength = {emotion: [] for emotion in emotions}
    
    for result in tqdm(results, desc="Evaluating"):
        # Load and preprocess image
        image = Image.open(result["image_path"]).convert("RGB")
        img_input = emotionclip_preprocess(image)
        
        # Run EmotionCLIP inference
        with torch.no_grad():
            logits_per_image, _ = emotionclip_model(
                img_input.unsqueeze(0).to(device=emotionclip_model.device, dtype=emotionclip_model.dtype),
                text_input
            )
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(logits_per_image, dim=-1).squeeze(0)
        
        # Get prediction and target
        predicted_idx = probabilities.argmax().item()
        target_idx = result["emotion_idx"]
        target_emotion = result["emotion"]
        
        # Calculate metrics
        is_correct = predicted_idx == target_idx
        target_strength = probabilities[target_idx].item()
        
        # Update result
        result["predicted_emotion"] = emotions[predicted_idx] if predicted_idx < len(emotions) else "unknown"
        result["predicted_idx"] = predicted_idx
        result["is_correct"] = is_correct
        result["target_strength"] = target_strength
        result["all_probabilities"] = probabilities.cpu().numpy().tolist()
        result["all_logits"] = logits_per_image.squeeze(0).cpu().numpy().tolist()
        
        # Update aggregates
        if is_correct:
            correct_predictions += 1
            per_emotion_correct[target_emotion] += 1
        
        total_strength += target_strength
        per_emotion_total[target_emotion] += 1
        per_emotion_strength[target_emotion].append(target_strength)
    
    # Calculate aggregate metrics
    n_total = len(results)
    overall_accuracy = correct_predictions / n_total if n_total > 0 else 0
    overall_strength = total_strength / n_total if n_total > 0 else 0
    
    # Per-emotion metrics
    per_emotion_metrics = {}
    for emotion in emotions:
        total = per_emotion_total[emotion]
        correct = per_emotion_correct[emotion]
        strengths = per_emotion_strength[emotion]
        
        per_emotion_metrics[emotion] = {
            "accuracy": correct / total if total > 0 else 0,
            "avg_strength": np.mean(strengths) if strengths else 0,
            "std_strength": np.std(strengths) if strengths else 0,
            "correct": correct,
            "total": total,
        }
    
    metrics = {
        "overall": {
            "accuracy": overall_accuracy,
            "avg_strength": overall_strength,
            "correct_predictions": correct_predictions,
            "total_samples": n_total,
        },
        "per_emotion": per_emotion_metrics,
    }
    
    return results, metrics


def evaluate_emotions_vit(
    processor,
    model,
    results: list,
    emotions: list,
    ):
    print("\nEvaluating generated images with ViT model...")

    # Evaluate each image
    correct_predictions = 0
    total_strength = 0.0
    
    per_emotion_correct = {emotion: 0 for emotion in emotions}
    per_emotion_total = {emotion: 0 for emotion in emotions}
    per_emotion_strength = {emotion: [] for emotion in emotions}

    for result in tqdm(results, desc="Evaluating"):
        # Load and preprocess image
        image = Image.open(result["image_path"])
        inputs = processor(image, return_tensors="pt")

        # Make prediction
        with torch.no_grad():
            outputs = model(**inputs.to(model.device))
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_idx = torch.argmax(probabilities, dim=-1).item()

        target_idx = result["emotion_idx"]
        target_emotion = result["emotion"]
        
        # Calculate metrics
        is_correct = predicted_idx == target_idx
        target_strength = probabilities[0][predicted_idx].item()
        
        # Update result
        result["predicted_emotion"] = emotions[predicted_idx] if predicted_idx < len(emotions) else "unknown"
        result["predicted_idx"] = predicted_idx
        result["is_correct"] = is_correct
        result["target_strength"] = target_strength
        result["all_probabilities"] = probabilities.cpu().numpy().tolist()
        
        # Update aggregates
        if is_correct:
            correct_predictions += 1
            per_emotion_correct[target_emotion] += 1
        
        total_strength += target_strength
        per_emotion_total[target_emotion] += 1
        per_emotion_strength[target_emotion].append(target_strength)

    # Calculate aggregate metrics
    n_total = len(results)
    overall_accuracy = correct_predictions / n_total if n_total > 0 else 0
    overall_strength = total_strength / n_total if n_total > 0 else 0
    
    # Per-emotion metrics
    per_emotion_metrics = {}
    for emotion in emotions:
        total = per_emotion_total[emotion]
        correct = per_emotion_correct[emotion]
        strengths = per_emotion_strength[emotion]
        
        per_emotion_metrics[emotion] = {
            "accuracy": correct / total if total > 0 else 0,
            "avg_strength": np.mean(strengths) if strengths else 0,
            "std_strength": np.std(strengths) if strengths else 0,
            "correct": correct,
            "total": total,
        }
    
    metrics = {
        "overall": {
            "accuracy": overall_accuracy,
            "avg_strength": overall_strength,
            "correct_predictions": correct_predictions,
            "total_samples": n_total,
        },
        "per_emotion": per_emotion_metrics,
    }
    
    return results, metrics


def create_evaluation_report(results: list, metrics: dict, output_dir: str, emotions: list):
    """Create evaluation report with visualizations and summary."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    print("\nCreating evaluation report...")
    
    # 1. Save detailed results as JSON
    results_path = os.path.join(report_dir, "detailed_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": results,
            "metrics": metrics,
            "emotions": emotions,
            "prompts": EVALUATION_PROMPTS,
        }, f, indent=2)
    print(f"  Saved detailed results to {results_path}")
    
    # 2. Create confusion matrix
    confusion = np.zeros((len(emotions), len(emotions)))
    for result in results:
        true_idx = result["emotion_idx"]
        pred_idx = result["predicted_idx"]
        confusion[true_idx, pred_idx] += 1
    
    # Normalize by row (true class)
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_norm, cmap='Blues')
    
    ax.set_xticks(range(len(emotions)))
    ax.set_yticks(range(len(emotions)))
    ax.set_xticklabels(emotions, rotation=45, ha='right')
    ax.set_yticklabels(emotions)
    ax.set_xlabel('Predicted Emotion')
    ax.set_ylabel('True Emotion')
    ax.set_title('Emotion Classification Confusion Matrix (Normalized)')
    
    # Add text annotations
    for i in range(len(emotions)):
        for j in range(len(emotions)):
            text = ax.text(j, i, f'{confusion_norm[i, j]:.2f}',
                          ha='center', va='center',
                          color='white' if confusion_norm[i, j] > 0.5 else 'black')
    
    plt.colorbar(im)
    plt.tight_layout()
    confusion_path = os.path.join(report_dir, "confusion_matrix.png")
    plt.savefig(confusion_path, dpi=150)
    plt.close()
    print(f"  Saved confusion matrix to {confusion_path}")
    
    # 3. Per-emotion accuracy bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    accuracies = [metrics["per_emotion"][e]["accuracy"] for e in emotions]
    colors = plt.cm.tab10(np.linspace(0, 1, len(emotions)))
    axes[0].bar(emotions, accuracies, color=colors)
    axes[0].axhline(y=metrics["overall"]["accuracy"], color='red', linestyle='--', 
                    label=f'Overall: {metrics["overall"]["accuracy"]:.2%}')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Emotion Accuracy by Class')
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.1%}', ha='center', fontsize=9)
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Strength
    strengths = [metrics["per_emotion"][e]["avg_strength"] for e in emotions]
    axes[1].bar(emotions, strengths, color=colors)
    axes[1].axhline(y=metrics["overall"]["avg_strength"], color='red', linestyle='--',
                    label=f'Overall: {metrics["overall"]["avg_strength"]:.3f}')
    axes[1].set_ylabel('Average Strength (softmax probability)')
    axes[1].set_title('Emotion Strength by Class')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    for i, v in enumerate(strengths):
        axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    metrics_path = os.path.join(report_dir, "per_emotion_metrics.png")
    plt.savefig(metrics_path, dpi=150)
    plt.close()
    print(f"  Saved per-emotion metrics to {metrics_path}")
    
    # 4. Create summary text report
    summary_path = os.path.join(report_dir, "summary.txt")
    with open(summary_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EMOTION EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Emotion Accuracy: {metrics['overall']['accuracy']:.2%}\n")
        f.write(f"  Emotion Strength: {metrics['overall']['avg_strength']:.4f}\n")
        f.write(f"  Correct Predictions: {metrics['overall']['correct_predictions']}/{metrics['overall']['total_samples']}\n")
        f.write("\n")
        
        f.write("PER-EMOTION METRICS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"{'Emotion':<12} {'Accuracy':>10} {'Strength':>10} {'Correct':>10}\n")
        f.write("-" * 40 + "\n")
        for emotion in emotions:
            em = metrics["per_emotion"][emotion]
            f.write(f"{emotion:<12} {em['accuracy']:>10.1%} {em['avg_strength']:>10.4f} {em['correct']:>7}/{em['total']}\n")
        f.write("\n")
        
        f.write("PROMPTS USED:\n")
        f.write("-" * 40 + "\n")
        for i, prompt in enumerate(EVALUATION_PROMPTS):
            f.write(f"  {i+1:2d}. {prompt}\n")
        f.write("\n")
        
        f.write("EMOTIONS:\n")
        f.write("-" * 40 + "\n")
        for emotion in emotions:
            f.write(f"  - {emotion}\n")
    
    print(f"  Saved summary report to {summary_path}")
    
    # 5. Create image grid for each prompt showing all emotions
    grid_dir = os.path.join(report_dir, "prompt_grids")
    os.makedirs(grid_dir, exist_ok=True)
    
    for prompt_idx in range(len(EVALUATION_PROMPTS)):
        prompt_results = [r for r in results if r["prompt_idx"] == prompt_idx]
        if len(prompt_results) != len(emotions):
            continue
        
        # Sort by emotion order
        prompt_results.sort(key=lambda x: x["emotion_idx"])
        
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        for i, result in enumerate(prompt_results):
            img = Image.open(result["image_path"])
            axes[i].imshow(img)
            
            # Title with emotion and metrics
            emotion = result["emotion"]
            strength = result["target_strength"]
            is_correct = result["is_correct"]
            predicted = result["predicted_emotion"]
            
            title_color = 'green' if is_correct else 'red'
            title = f"{emotion}\nStrength: {strength:.3f}"
            if not is_correct:
                title += f"\n(Pred: {predicted})"
            
            axes[i].set_title(title, fontsize=10, color=title_color)
            axes[i].axis('off')
        
        # Add prompt as figure title
        prompt = EVALUATION_PROMPTS[prompt_idx]
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        fig.suptitle(f"Prompt {prompt_idx+1}: {prompt}", fontsize=12, y=1.02)
        
        plt.tight_layout()
        grid_path = os.path.join(grid_dir, f"prompt_{prompt_idx:02d}.png")
        plt.savefig(grid_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(EVALUATION_PROMPTS)} prompt grids to {grid_dir}")
    
    return report_dir


def print_summary(metrics: dict, emotions: list):
    """Print evaluation summary to console."""
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    
    print(f"\n📊 OVERALL METRICS:")
    print(f"   Emotion Accuracy: {metrics['overall']['accuracy']:.2%}")
    print(f"   Emotion Strength: {metrics['overall']['avg_strength']:.4f}")
    print(f"   Correct: {metrics['overall']['correct_predictions']}/{metrics['overall']['total_samples']}")
    
    print(f"\n📈 PER-EMOTION BREAKDOWN:")
    print(f"   {'Emotion':<12} {'Accuracy':>10} {'Strength':>10}")
    print("   " + "-" * 35)
    
    for emotion in emotions:
        em = metrics["per_emotion"][emotion]
        acc_bar = "█" * int(em['accuracy'] * 10)
        print(f"   {emotion:<12} {em['accuracy']:>10.1%} {em['avg_strength']:>10.4f}  {acc_bar}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate emotion-conditioned image generation using CLIP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate baseline_lora on 30K dataset
  python evaluate.py --approach baseline_lora --dataset-size 30K
  
  # Skip generation (use existing images)
  python evaluate.py --approach baseline_lora --dataset-size 30K --skip-generation
  
  # Custom output directory
  python evaluate.py --approach baseline_lora --dataset-size 30K --output-dir /path/to/output
        """
    )
    
    parser.add_argument('--approach', type=str, required=True,
                       help='Approach name (e.g., baseline_lora)')
    parser.add_argument('--dataset-size', type=str, required=True,
                       help='Dataset size (e.g., 10K, 30K)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory for evaluation results')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Directory with images')
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip image generation, use existing images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for generation')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Guidance scale for generation')
    parser.add_argument('--dataset', type=str, default="emoset",
                       help='Type of the dataset used for training (either "emoset" or "rafdb")')
    parser.add_argument('--task', type=str, default="label2image",
                       help='Type of the task (either "label2image" or "multimodal")')
    
    args = parser.parse_args()

    if args.dataset == "emoset":
        emotions = EMOTIONS_EMOSET
    elif args.dataset == "rafdb":
        emotions = EMOTIONS_RAFDB
    else:
        raise ValueError("Unknown dataset type")
    
    # Normalize dataset size
    size_normalized = args.dataset_size.upper()
    if not size_normalized.endswith("K"):
        try:
            num = int(size_normalized)
            size_normalized = f"{num // 1000}K"
        except ValueError:
            pass
    
    # Setup paths
    weights_dir = os.path.join(STORAGE_BASE, args.approach, "pytorch_lora_weights.safetensors")
    
    # Always save evaluation images in validation_images/{size}/{approach} in repository root
    if args.images_dir:
        images_dir = os.path.join(args.images_dir, size_normalized, args.approach)
    else:
        images_dir = os.path.join(STORAGE_BASE, "validation_images", size_normalized, args.approach)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Evaluation reports go to Evaluations directory
        output_dir = os.path.join(STORAGE_BASE, "Evaluations", size_normalized, args.approach)
    
    # Check weights exist
    if not os.path.exists(weights_dir):
        print(f"Error: Weights not found at {weights_dir}")
        print("Please train the model first or check the path.")
        return 1
    
    print("=" * 60)
    print("EMOTION EVALUATION")
    print("=" * 60)
    print(f"Approach: {args.approach}")
    print(f"Dataset Size: {size_normalized}")
    print(f"Weights: {weights_dir}")
    print(f"Images: {images_dir}")
    print(f"Reports: {output_dir}")
    print(f"Prompts: {len(EVALUATION_PROMPTS)}")
    print(f"Emotions: {len(emotions)}")
    print(f"Total Images: {len(EVALUATION_PROMPTS) * len(emotions)}")
    print("=" * 60)
    
    # Setup device (needed for both generation and evaluation)
    device = setup_device()
    
    # Check if images already exist
    expected_total = len(EVALUATION_PROMPTS) * len(emotions)
    existing_count = 0
    missing_images = []
    
    if os.path.exists(images_dir):
        for prompt_idx, prompt in enumerate(EVALUATION_PROMPTS):
            for emotion_idx, emotion in enumerate(emotions):
                filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    existing_count += 1
                else:
                    missing_images.append((prompt_idx, emotion_idx, emotion))
    
    # Determine if we should skip generation
    should_skip = args.skip_generation or (existing_count == expected_total and existing_count > 0)
    
    if should_skip:
        if existing_count == expected_total:
            print(f"\n✓ All {existing_count} images already exist. Skipping generation.")
        else:
            print(f"\n⚠ Skipping generation (--skip-generation flag). Found {existing_count}/{expected_total} images.")
        
        # Load existing results
        print(f"Loading existing images from {images_dir}...")
        results = []
        for prompt_idx, prompt in enumerate(EVALUATION_PROMPTS):
            for emotion_idx, emotion in enumerate(emotions):
                filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    
                    results.append({
                        "image_path": image_path,
                        "prompt": prompt,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "prompt_idx": prompt_idx,
                    })
        print(f"Loaded {len(results)} existing images")
    else:
        # Load generation pipeline
        pipe = load_generation_pipeline(weights_dir, device)
        
        if existing_count > 0:
            print(f"\n⚠ Found {existing_count}/{expected_total} existing images.")
            if missing_images:
                print(f"  Missing {len(missing_images)} images. Regenerating all to ensure consistency.")
        
        # Generate all images
        results = generate_images(
            pipe,
            args.task,
            args.approach,
            EVALUATION_PROMPTS,
            emotions,
            images_dir,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        
        # Free GPU memory
        del pipe
        torch.cuda.empty_cache()

    if not results:
        print("Error: No images to evaluate")
        return 1
    
    if args.dataset == "emoset":
        # Load EmotionCLIP for evaluation
        emotionclip_model, emotionclip_preprocess, emotionclip_tokenizer = load_emotionclip_model(device)
            # Evaluate with EmotionCLIP
        results, metrics = evaluate_emotions_emoclip(
            emotionclip_model,
            emotionclip_preprocess,
            emotionclip_tokenizer,
            results,
            emotions,
        )
    elif args.dataset == "rafdb":
        vit_processor, vit_model = load_vit_model(device)
        results, metrics = evaluate_emotions_vit(
            vit_processor,
            vit_model,
            results,
            emotions,
        )

    
    # Create evaluation report
    report_dir = create_evaluation_report(results, metrics, output_dir, emotions)
    
    # Print summary
    print_summary(metrics, emotions)
    
    print(f"\n📁 Full evaluation saved to: {output_dir}")
    print(f"📊 Report available at: {report_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())