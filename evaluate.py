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
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
import subprocess

# Add paths for imports
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "approaches" / "baseline_lora" / "src"))
sys.path.insert(0, str(REPO_ROOT / "shared" / "src"))

# Storage paths
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"

# EmotionCLIP path
EMOTIONCLIP_PATH = "/Data/yash.bhardwaj/EmotionCLIP"

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
EMOTIONS = [
    "amusement",
    "anger",
    "awe",
    "contentment",
    "disgust",
    "excitement",
    "fear",
    "sadness",
]

# Emotion tokens used in training
EMOTION_TOKENS = [f"<{emotion}>" for emotion in EMOTIONS]


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


def load_generation_pipeline(weights_dir: str, device):
    """Load the trained Stable Diffusion pipeline with LoRA and embeddings."""
    from diffusers import StableDiffusionPipeline
    from peft import PeftModel
    import json
    
    model_id = "runwayml/stable-diffusion-v1-5"
    cache_dir = os.path.join(STORAGE_BASE, "cache", "huggingface")
    
    print(f"\nLoading generation pipeline from {weights_dir}...")
    
    # Load base pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=cache_dir
    )
    
    # Load tokenizer info
    tokenizer_info_path = os.path.join(weights_dir, "tokenizer_info.json")
    if os.path.exists(tokenizer_info_path):
        with open(tokenizer_info_path, 'r') as f:
            tokenizer_info = json.load(f)
        emotion_tokens = tokenizer_info.get("emotion_tokens", EMOTION_TOKENS)
    else:
        emotion_tokens = EMOTION_TOKENS
    
    # Add tokens to tokenizer
    num_added = pipe.tokenizer.add_tokens(emotion_tokens)
    if num_added > 0:
        pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))
    
    # Load learned embeddings
    learned_embeds_path = os.path.join(weights_dir, "learned_embeds.bin")
    if os.path.exists(learned_embeds_path):
        learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        for token, embedding in learned_embeds.items():
            token_id = pipe.tokenizer.convert_tokens_to_ids(token)
            if token_id != pipe.tokenizer.unk_token_id:
                with torch.no_grad():
                    pipe.text_encoder.get_input_embeddings().weight[token_id] = embedding.to(
                        pipe.text_encoder.get_input_embeddings().weight.device
                    )
        print(f"  Loaded {len(learned_embeds)} learned embeddings")
    
    # Load LoRA weights
    lora_path = weights_dir
    if os.path.exists(os.path.join(lora_path, "adapter_model.safetensors")):
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet = pipe.unet.merge_and_unload()
            print("  Loaded and merged LoRA weights")
        except Exception as e:
            print(f"  Warning: Could not load LoRA: {e}")
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    return pipe, emotion_tokens


def generate_images(
    pipe,
    prompts: list,
    emotions: list,
    output_dir: str,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    prompt_template: str = "{prompt}, conveying {emotion}"
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
    
    results = []
    total = len(prompts) * len(emotions)
    
    print(f"\nGenerating {total} images ({len(prompts)} prompts × {len(emotions)} emotions)...")
    
    pbar = tqdm(total=total, desc="Generating images")
    
    for prompt_idx, prompt in enumerate(prompts):
        for emotion_idx, emotion in enumerate(emotions):
            # Use emotion token in the prompt (matches training format)
            emotion_token = f"<{emotion}>"
            
            # Full prompt with emotion token at the start (matches training)
            full_prompt = f"{emotion_token} {prompt}"
            
            # Also create a natural language version for CLIP evaluation
            natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
            
            # Generate with fixed seed for reproducibility across emotions
            generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
            
            try:
                image = pipe(
                    prompt=full_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                ).images[0]
                
                # Save image
                filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                image_path = os.path.join(output_dir, filename)
                image.save(image_path)
                
                results.append({
                    "image_path": image_path,
                    "prompt": prompt,
                    "emotion": emotion,
                    "emotion_idx": emotion_idx,
                    "prompt_idx": prompt_idx,
                    "full_prompt": full_prompt,
                    "natural_prompt": natural_prompt,
                })
                
            except Exception as e:
                print(f"\n  Error generating {filename}: {e}")
            
            pbar.update(1)
    
    pbar.close()
    print(f"Generated {len(results)} images to {output_dir}")
    
    return results


def evaluate_emotions(
    emotionclip_model,
    emotionclip_preprocess,
    emotionclip_tokenizer,
    results: list,
    device,
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
        device: Torch device (not used, EmotionCLIP uses its own device)
        
    Returns:
        Updated results with evaluation metrics, and aggregate metrics
    """
    print("\nEvaluating generated images with EmotionCLIP...")
    
    # EmotionCLIP emotion label mapping (matches EmoSet emotions)
    emotion_to_idx = {
        'amusement': 0,
        'anger': 1,
        'awe': 2,
        'contentment': 3,
        'disgust': 4,
        'excitement': 5,
        'fear': 6,
        'sadness': 7,
        # 'neutral': 8  # EmotionCLIP also has neutral, but we don't use it
    }
    idx_to_emotion = {v: k for k, v in emotion_to_idx.items()}
    
    # Prepare text inputs using EmotionCLIP's expected format
    text_list = [f"This picture conveys a sense of {emotion}" for emotion in EMOTIONS]
    text_input = emotionclip_tokenizer(text_list)
    
    # Move text input to model device
    text_input = text_input.to(device=emotionclip_model.device)
    
    # Evaluate each image
    correct_predictions = 0
    total_strength = 0.0
    
    per_emotion_correct = {emotion: 0 for emotion in EMOTIONS}
    per_emotion_total = {emotion: 0 for emotion in EMOTIONS}
    per_emotion_strength = {emotion: [] for emotion in EMOTIONS}
    
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
        result["predicted_emotion"] = EMOTIONS[predicted_idx] if predicted_idx < len(EMOTIONS) else "unknown"
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
    for emotion in EMOTIONS:
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


def create_evaluation_report(results: list, metrics: dict, output_dir: str):
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
            "emotions": EMOTIONS,
            "prompts": EVALUATION_PROMPTS,
        }, f, indent=2)
    print(f"  Saved detailed results to {results_path}")
    
    # 2. Create confusion matrix
    confusion = np.zeros((len(EMOTIONS), len(EMOTIONS)))
    for result in results:
        true_idx = result["emotion_idx"]
        pred_idx = result["predicted_idx"]
        confusion[true_idx, pred_idx] += 1
    
    # Normalize by row (true class)
    confusion_norm = confusion / confusion.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(confusion_norm, cmap='Blues')
    
    ax.set_xticks(range(len(EMOTIONS)))
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_xticklabels(EMOTIONS, rotation=45, ha='right')
    ax.set_yticklabels(EMOTIONS)
    ax.set_xlabel('Predicted Emotion')
    ax.set_ylabel('True Emotion')
    ax.set_title('Emotion Classification Confusion Matrix (Normalized)')
    
    # Add text annotations
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
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
    accuracies = [metrics["per_emotion"][e]["accuracy"] for e in EMOTIONS]
    colors = plt.cm.tab10(np.linspace(0, 1, len(EMOTIONS)))
    axes[0].bar(EMOTIONS, accuracies, color=colors)
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
    strengths = [metrics["per_emotion"][e]["avg_strength"] for e in EMOTIONS]
    axes[1].bar(EMOTIONS, strengths, color=colors)
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
        for emotion in EMOTIONS:
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
        for emotion in EMOTIONS:
            f.write(f"  - {emotion}\n")
    
    print(f"  Saved summary report to {summary_path}")
    
    # 5. Create image grid for each prompt showing all emotions
    grid_dir = os.path.join(report_dir, "prompt_grids")
    os.makedirs(grid_dir, exist_ok=True)
    
    for prompt_idx in range(len(EVALUATION_PROMPTS)):
        prompt_results = [r for r in results if r["prompt_idx"] == prompt_idx]
        if len(prompt_results) != len(EMOTIONS):
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


def print_summary(metrics: dict):
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
    
    for emotion in EMOTIONS:
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
    parser.add_argument('--skip-generation', action='store_true',
                       help='Skip image generation, use existing images')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for generation')
    parser.add_argument('--num-inference-steps', type=int, default=50,
                       help='Number of diffusion steps')
    parser.add_argument('--guidance-scale', type=float, default=7.5,
                       help='Guidance scale for generation')
    
    args = parser.parse_args()
    
    # Normalize dataset size
    size_normalized = args.dataset_size.upper()
    if not size_normalized.endswith("K"):
        try:
            num = int(size_normalized)
            size_normalized = f"{num // 1000}K"
        except ValueError:
            pass
    
    # Setup paths
    weights_dir = os.path.join(STORAGE_BASE, "Weights", size_normalized, args.approach)
    
    # Always save evaluation images in validation_images/{size}/{approach} in repository root
    images_dir = os.path.join(REPO_ROOT, "validation_images", size_normalized, args.approach)
    
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
    print(f"Emotions: {len(EMOTIONS)}")
    print(f"Total Images: {len(EVALUATION_PROMPTS) * len(EMOTIONS)}")
    print("=" * 60)
    
    # Setup device (needed for both generation and evaluation)
    device = setup_device()
    
    # Check if images already exist
    expected_total = len(EVALUATION_PROMPTS) * len(EMOTIONS)
    existing_count = 0
    missing_images = []
    
    if os.path.exists(images_dir):
        for prompt_idx, prompt in enumerate(EVALUATION_PROMPTS):
            for emotion_idx, emotion in enumerate(EMOTIONS):
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
            for emotion_idx, emotion in enumerate(EMOTIONS):
                filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                image_path = os.path.join(images_dir, filename)
                if os.path.exists(image_path):
                    # Use emotion token in the prompt (matches training format)
                    emotion_token = f"<{emotion}>"
                    full_prompt = f"{emotion_token} {prompt}"
                    natural_prompt = f"{prompt}, conveying {emotion}"
                    
                    results.append({
                        "image_path": image_path,
                        "prompt": prompt,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "prompt_idx": prompt_idx,
                        "full_prompt": full_prompt,
                        "natural_prompt": natural_prompt,
                    })
        print(f"Loaded {len(results)} existing images")
    else:
        # Load generation pipeline
        pipe, emotion_tokens = load_generation_pipeline(weights_dir, device)
        
        if existing_count > 0:
            print(f"\n⚠ Found {existing_count}/{expected_total} existing images.")
            if missing_images:
                print(f"  Missing {len(missing_images)} images. Regenerating all to ensure consistency.")
        
        # Generate all images
        results = generate_images(
            pipe,
            EVALUATION_PROMPTS,
            EMOTIONS,
            images_dir,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
        
        # Free GPU memory
        del pipe
        torch.cuda.empty_cache()
    
    # Load EmotionCLIP for evaluation
    emotionclip_model, emotionclip_preprocess, emotionclip_tokenizer = load_emotionclip_model(device)
    
    if not results:
        print("Error: No images to evaluate")
        return 1
    
    # Evaluate with EmotionCLIP
    results, metrics = evaluate_emotions(
        emotionclip_model,
        emotionclip_preprocess,
        emotionclip_tokenizer,
        results,
        device,
    )
    
    # Create evaluation report
    report_dir = create_evaluation_report(results, metrics, output_dir)
    
    # Print summary
    print_summary(metrics)
    
    print(f"\n📁 Full evaluation saved to: {output_dir}")
    print(f"📊 Report available at: {report_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
