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
import gc

# Add paths for imports
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "approaches" / "baseline_lora" / "src"))
sys.path.insert(0, str(REPO_ROOT / "shared" / "src"))

# Storage paths - use environment variables or default to current directory
STORAGE_BASE = os.getenv("EMOTIONAL_PORTRAITS_BASE", str(REPO_ROOT))

# EmotionCLIP path - use environment variable or default
EMOTIONCLIP_PATH = os.getenv("EMOTIONCLIP_PATH", None)
if EMOTIONCLIP_PATH is None:
    # Try common locations
    possible_paths = [
        os.path.join(STORAGE_BASE, "EmotionCLIP"),
        os.path.join(os.path.expanduser("~"), "EmotionCLIP"),
        "/EmotionCLIP",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            EMOTIONCLIP_PATH = path
            break
    # Don't raise error here - EmotionCLIP is only needed for EmoSet evaluation
    # It will be checked when actually needed in load_emotionclip_model()

# ============================================================================
# Evaluation Prompts
# ============================================================================
# Scene prompts for EmoSet (scene emotion evaluation)
EVALUATION_PROMPTS_SCENES = [
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

# Portrait prompt template for RAFDB (facial emotion evaluation)
# For RAFDB, we use a single portrait prompt template that will be formatted with each emotion
PORTRAIT_PROMPT_TEMPLATE = "a photorealistic human portrait depicting {emotion}, head and shoulders, looking at camera"

# Number of images to generate per emotion for RAFDB portrait evaluation
# This allows multiple samples per emotion for better evaluation statistics
RAFDB_IMAGES_PER_EMOTION = 10

# Default to scene prompts for backward compatibility
EVALUATION_PROMPTS = EVALUATION_PROMPTS_SCENES


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

# RAFDB emotions
EMOTIONS_RAFDB = [
    "anger",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
]

# Default to EmoSet for backward compatibility
EMOTIONS = EMOTIONS_EMOSET

# Emotion tokens used in training (for EmoSet)
EMOTION_TOKENS = [f"<{emotion}>" for emotion in EMOTIONS_EMOSET]


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
    if EMOTIONCLIP_PATH is None:
        raise ValueError(
            "EmotionCLIP is required for EmoSet evaluation. "
            "Please set EMOTIONCLIP_PATH environment variable or install EmotionCLIP. "
            "See README.md for installation instructions."
        )
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
    
    # Change to EmotionCLIP directory temporarily (needed for relative path in EmotionCLIP.py)
    original_cwd = os.getcwd()
    try:
        os.chdir(EMOTIONCLIP_PATH)
        # Dynamically load EmotionCLIP from its script to avoid import path issues
        import importlib.util
        emotionclip_file = os.path.join(EMOTIONCLIP_PATH, "EmotionCLIP.py")
        spec = importlib.util.spec_from_file_location("emotionclip", emotionclip_file)
        emotionclip = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(emotionclip)
        model = emotionclip.model
        preprocess = emotionclip.preprocess
        tokenizer = emotionclip.tokenizer
    finally:
        # Restore original working directory
        os.chdir(original_cwd)
    
    print(f"Loaded EmotionCLIP model (device: {model.device}, dtype: {model.dtype})")
    
    return model, preprocess, tokenizer


def load_vit_model(device):
    """Load ViT model for RAFDB emotion evaluation."""
    from transformers import ViTImageProcessor, ViTForImageClassification
    
    cache_dir = os.path.join(STORAGE_BASE, "cache", "huggingface")
    processor = ViTImageProcessor.from_pretrained(
        'abhilash88/face-emotion-detection', 
        cache_dir=cache_dir
    )
    model = ViTForImageClassification.from_pretrained(
        'abhilash88/face-emotion-detection', 
        cache_dir=cache_dir
    )
    model = model.to(device)
    print(f"Loaded ViT emotion detection model (device: {model.device})")
    return processor, model


def load_baseline_pipeline(device):
    """Load vanilla Stable Diffusion 1.4 pipeline without any fine-tuning."""
    from diffusers import StableDiffusionPipeline
    
    model_id = "CompVis/stable-diffusion-v1-4"
    cache_dir = os.path.join(STORAGE_BASE, "cache", "huggingface")
    
    print(f"\nLoading baseline (vanilla) Stable Diffusion 1.4 pipeline...")
    
    # Load base pipeline without any modifications
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        cache_dir=cache_dir
    )
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    print("  Loaded vanilla Stable Diffusion 1.4 (no fine-tuning)")
    
    return pipe


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
    
    # Check if this is scripts format (pytorch_lora_weights.safetensors) or baseline_lora format (adapter_model.safetensors)
    pytorch_lora_path = os.path.join(weights_dir, "pytorch_lora_weights.safetensors")
    adapter_lora_path = os.path.join(weights_dir, "adapter_model.safetensors")
    
    if os.path.exists(pytorch_lora_path):
        # Scripts format: use diffusers load_lora_weights
        print("  Detected scripts format (pytorch_lora_weights.safetensors)")
        try:
            pipe.load_lora_weights(weights_dir)
            print("  Loaded LoRA weights using diffusers format")
        except Exception as e:
            print(f"  Warning: Could not load LoRA weights: {e}")
        emotion_tokens = []  # Scripts format doesn't use emotion tokens in prompt
    elif os.path.exists(adapter_lora_path):
        # weights format: use PEFT format
        print("  Detected weights format (adapter_model.safetensors)")
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
        try:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, weights_dir)
            pipe.unet = pipe.unet.merge_and_unload()
            print("  Loaded and merged LoRA weights")
        except Exception as e:
            print(f"  Warning: Could not load LoRA: {e}")
    else:
        print(f"  Warning: No LoRA weights found in {weights_dir}")
        emotion_tokens = []
    
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
    prompt_template: str = "{prompt}, conveying {emotion}",
    approach: str = "baseline_lora",
    classifier=None,
    classifier_scale: float = 20.0,
    task: str = "label2image",
    emotion_embedding=None,
    dataset: str = "emoset",
    is_rafdb_portrait_mode: bool = False,
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
    
    if is_rafdb_portrait_mode:
        # For RAFDB: multiple images per emotion
        total = len(prompts) * RAFDB_IMAGES_PER_EMOTION
        print(f"\nGenerating {total} images ({len(prompts)} emotions × {RAFDB_IMAGES_PER_EMOTION} images per emotion)...")
    else:
        # For EmoSet: images for each prompt × emotion combination
        total = len(prompts) * len(emotions)
        print(f"\nGenerating {total} images ({len(prompts)} prompts × {len(emotions)} emotions)...")
    
    pbar = tqdm(total=total, desc="Generating images")
    
    if is_rafdb_portrait_mode:
        # For RAFDB: generate multiple images per emotion with different seeds
        for prompt_idx, prompt in enumerate(prompts):
            emotion = emotions[prompt_idx]  # Prompts are in same order as emotions
            emotion_idx = prompt_idx
            
            # Generate multiple images per emotion with different seeds
            for img_idx in range(RAFDB_IMAGES_PER_EMOTION):
                filename = f"prompt{prompt_idx:02d}_{emotion}_{img_idx:02d}.png"
                image_path = os.path.join(output_dir, filename)
                
                try:
                    # Generate with different seed for each image (seed + prompt_idx * 100 + img_idx)
                    generator = torch.Generator(device=device).manual_seed(seed + prompt_idx * 100 + img_idx)
                    
                    if approach == "baseline":
                        # Portrait prompt already includes the emotion
                        full_prompt = prompt
                        natural_prompt = prompt
                        
                        # Log the prompt being used for verification (only once)
                        if prompt_idx == 0 and img_idx == 0:
                            print(f"\n[DEBUG] Baseline RAFDB portrait prompt format:")
                            print(f"  Portrait prompt: '{prompt}'")
                            print(f"  Full prompt sent to model: '{full_prompt}'")
                            print(f"  Generating {RAFDB_IMAGES_PER_EMOTION} images per emotion")
                            print()
                        
                        image = pipe(
                            prompt=full_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                    else:
                        # For other approaches, use the prompt as-is (it already contains emotion)
                        full_prompt = prompt
                        natural_prompt = prompt
                        image = pipe(
                            prompt=full_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                    
                    image.save(image_path)
                    
                    results.append({
                        "image_path": image_path,
                        "prompt": prompt,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "prompt_idx": prompt_idx,
                        "img_idx": img_idx,  # Track which image this is for this emotion
                        "full_prompt": full_prompt,
                        "natural_prompt": natural_prompt,
                    })
                    
                    pbar.update(1)
                except Exception as e:
                    print(f"\nError generating image for prompt {prompt_idx}, emotion {emotion}, image {img_idx}: {e}")
                    pbar.update(1)
    else:
        # For EmoSet: iterate over all prompt × emotion combinations
        for prompt_idx, prompt in enumerate(prompts):
            for emotion_idx, emotion in enumerate(emotions):
                filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                image_path = os.path.join(output_dir, filename)
            
            try:
                # Generate with fixed seed for reproducibility across emotions
                generator = torch.Generator(device=device).manual_seed(seed + prompt_idx)
                
                if approach == "baseline" and task == "label2image":
                    # Baseline label2image: use only emotion label, no scene prompt
                    full_prompt = emotion
                    natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    
                    image = pipe(
                        prompt=full_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                elif approach == "baseline":
                    # Baseline: vanilla SD 1.4 with emotion label in prompt
                    # For RAFDB portraits: prompt already contains emotion (e.g., "a photorealistic human portrait depicting anger, ...")
                    # For EmoSet scenes: Format: "{prompt}, {emotion}" to combine scene and emotion
                    if dataset == "rafdb":
                        # Portrait prompt already includes the emotion
                        full_prompt = prompt
                        natural_prompt = prompt
                    else:
                        # Scene prompt + emotion
                        full_prompt = f"{prompt}, {emotion}"
                        natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    
                    # Log the prompt being used for verification
                    if prompt_idx == 0 and emotion_idx == 0:
                        print(f"\n[DEBUG] Baseline prompt format ({dataset}):")
                        if dataset == "rafdb":
                            print(f"  Portrait prompt: '{prompt}'")
                        else:
                            print(f"  Scene prompt: '{prompt}'")
                            print(f"  Emotion: '{emotion}'")
                        print(f"  Full prompt sent to model: '{full_prompt}'")
                        print()
                    
                    image = pipe(
                        prompt=full_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                elif approach == "classifier_guidance":
                    # Classifier guidance: use plain prompt, emotion is guided via classifier
                    full_prompt = prompt
                    natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    
                    # Import classifier guidance function
                    import sys
                    cg_inference_path = os.path.join(REPO_ROOT, "approaches", "classifier_guidance", "src")
                    if cg_inference_path not in sys.path:
                        sys.path.insert(0, cg_inference_path)
                    from inference import generate_with_classifier_guidance
                    
                    image, _ = generate_with_classifier_guidance(
                        pipe=pipe,
                        classifier=classifier,
                        prompt=full_prompt,
                        target_emotion_idx=emotion_idx,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        classifier_scale=classifier_scale,
                        seed=seed + prompt_idx,
                        device=device,
                        track_metrics=False,
                        use_wandb=False,
                    )
                elif task == "label2image" or approach in ["emoset_label2image", "portraits"]:
                    # Label-to-image: use emotion as the prompt
                    # For portraits approach, use portrait prompt template
                    if approach == "portraits":
                        full_prompt = PORTRAIT_PROMPT_TEMPLATE.format(emotion=emotion)
                        natural_prompt = full_prompt
                    else:
                        full_prompt = emotion
                        natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    image = pipe(
                        prompt=full_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                elif task == "multimodal" or approach in ["emoset_multicond", "emoset_multicond_classifier_001", "emoset_multicond_classifier_01"]:
                    # Multimodal: use emotion embedding with prompt (E_cond = [e_emo; E_text])
                    if emotion_embedding is None:
                        raise ValueError("emotion_embedding is required for multimodal task")
                    # Import generate_with_emotion if available
                    try:
                        import sys
                        scripts_path = os.path.join(REPO_ROOT, "scripts")
                        if scripts_path not in sys.path:
                            sys.path.insert(0, scripts_path)
                        from train_text_to_image_lora import generate_with_emotion
                        image = generate_with_emotion(
                            pipe, 
                            emotion_embedding, 
                            prompt, 
                            emotion_idx, 
                            num_inference_steps=num_inference_steps, 
                            generator=generator
                        )
                        full_prompt = f"{prompt} <{emotion}>"
                        natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    except ImportError:
                        # Fallback: use emotion token if generate_with_emotion not available
                        emotion_token = f"<{emotion}>"
                        full_prompt = f"{emotion_token} {prompt}"
                        natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                        image = pipe(
                            prompt=full_prompt,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            generator=generator,
                        ).images[0]
                else:
                    # Baseline LoRA: use emotion token in the prompt
                    emotion_token = f"<{emotion}>"
                    full_prompt = f"{emotion_token} {prompt}"
                    natural_prompt = prompt_template.format(prompt=prompt, emotion=emotion)
                    
                    image = pipe(
                        prompt=full_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        generator=generator,
                    ).images[0]
                
                # Save image
                if not image.getbbox():
                    print(f"\n  Image is empty. Skipping...")
                else:
                    image.save(image_path)
                    
                    results.append({
                        "image_path": image_path,
                        "prompt": prompt,
                        "emotion": emotion,
                        "emotion_idx": emotion_idx,
                        "prompt_idx": prompt_idx,
                        "full_prompt": full_prompt if 'full_prompt' in locals() else prompt,
                        "natural_prompt": natural_prompt if 'natural_prompt' in locals() else prompt_template.format(prompt=prompt, emotion=emotion),
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
    1. Compute EmotionCLIP similarity with all emotion labels
    2. Apply softmax to get probabilities
    3. Calculate accuracy (argmax == target) and strength (softmax[target])
    
    Args:
        emotionclip_model: EmotionCLIP model
        emotionclip_preprocess: EmotionCLIP preprocessing function
        emotionclip_tokenizer: EmotionCLIP tokenizer
        results: List of generation results
        emotions: List of emotions to evaluate against
        
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
        target_strength = probabilities[target_idx].item() if target_idx < len(probabilities) else 0.0
        
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
    """
    Evaluate generated images using ViT model for emotion classification (for RAFDB).
    
    Args:
        processor: ViT image processor
        model: ViT emotion detection model
        results: List of generation results
        emotions: List of emotions to evaluate against
        
    Returns:
        Updated results with evaluation metrics, and aggregate metrics
    """
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
        target_strength = probabilities[0][target_idx].item() if target_idx < probabilities.shape[1] else 0.0
        
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


def create_evaluation_report(results: list, metrics: dict, output_dir: str, emotions: list, prompts: list = None):
    """Create evaluation report with visualizations and summary."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    report_dir = os.path.join(output_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    
    # Use provided prompts or fall back to EVALUATION_PROMPTS for backward compatibility
    prompts_to_use = prompts if prompts is not None else EVALUATION_PROMPTS
    
    print("\nCreating evaluation report...")
    
    # 1. Save detailed results as JSON
    results_path = os.path.join(report_dir, "detailed_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "results": results,
            "metrics": metrics,
            "emotions": emotions,
            "prompts": prompts if prompts is not None else EVALUATION_PROMPTS,
        }, f, indent=2)
    print(f"  Saved detailed results to {results_path}")
    
    # 2. Create confusion matrix
    confusion = np.zeros((len(emotions), len(emotions)))
    for result in results:
        true_idx = result["emotion_idx"]
        pred_idx = result["predicted_idx"]
        if true_idx < len(emotions) and pred_idx < len(emotions):
            confusion[true_idx, pred_idx] += 1
    
    # Normalize by row (true class)
    confusion_norm = confusion / (confusion.sum(axis=1, keepdims=True) + 1e-8)
    
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
        for i, prompt in enumerate(prompts_to_use):
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
    
    for prompt_idx in range(len(prompts_to_use)):
        prompt_results = [r for r in results if r["prompt_idx"] == prompt_idx]
        if len(prompt_results) != len(emotions):
            continue
        
        # Sort by emotion order
        prompt_results.sort(key=lambda x: x["emotion_idx"])
        
        # Calculate grid dimensions based on number of emotions
        n_emotions = len(emotions)
        n_cols = 4
        n_rows = (n_emotions + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
        if n_emotions == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, result in enumerate(prompt_results):
            if i < len(axes):
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
        
        # Hide unused subplots
        for i in range(len(prompt_results), len(axes)):
            axes[i].axis('off')
        
        # Add prompt as figure title
        prompt = prompts_to_use[prompt_idx]
        if len(prompt) > 80:
            prompt = prompt[:77] + "..."
        fig.suptitle(f"Prompt {prompt_idx+1}: {prompt}", fontsize=12, y=1.02)
        
        plt.tight_layout()
        grid_path = os.path.join(grid_dir, f"prompt_{prompt_idx:02d}.png")
        plt.savefig(grid_path, dpi=100, bbox_inches='tight')
        plt.close()
    
    print(f"  Saved {len(prompts_to_use)} prompt grids to {grid_dir}")
    
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
  # Evaluate baseline (vanilla SD 1.4) - no fine-tuning needed
  python evaluate.py --approach baseline --dataset-size FULL --dataset emoset
  
  # Evaluate baseline_lora on 30K dataset
  python evaluate.py --approach baseline_lora --dataset-size 30K
  
  # Evaluate classifier_guidance
  python evaluate.py --approach classifier_guidance --dataset-size FULL --dataset emoset
  
  # Skip generation (use existing images)
  python evaluate.py --approach baseline --dataset-size FULL --skip-generation
  
  # Custom output directory
  python evaluate.py --approach baseline --dataset-size FULL --output-dir /path/to/output
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
    parser.add_argument('--classifier-scale', type=float, default=20.0,
                       help='Classifier guidance strength (for classifier_guidance approach)')
    parser.add_argument('--images-dir', type=str, default=None,
                       help='Explicit path for saving images (overrides default)')
    parser.add_argument('--dataset', type=str, default="emoset",
                       help='Type of the dataset used for training (either "emoset" or "rafdb")')
    parser.add_argument('--task', type=str, default="label2image",
                       help='Type of the task (either "label2image", "multimodal", or "scene_emotion" for baseline scene+emotion)')
    
    args = parser.parse_args()
    
    # Auto-detect task for scripts approaches
    scripts_approaches = [
        "portraits", "emoset_label2image", "emoset_multicond",
        "emoset_multicond_classifier_001", "emoset_multicond_classifier_01"
    ]
    if args.approach in scripts_approaches:
        if args.approach == "portraits":
            args.dataset = "rafdb"
            args.task = "label2image"
        elif args.approach == "emoset_label2image":
            args.dataset = "emoset"
            args.task = "label2image"
        elif args.approach in ["emoset_multicond", "emoset_multicond_classifier_001", "emoset_multicond_classifier_01"]:
            args.dataset = "emoset"
            args.task = "multimodal"
    
    # Select emotions and prompts based on dataset
    if args.dataset == "emoset":
        emotions = EMOTIONS_EMOSET
        # Use scene prompts for EmoSet
        evaluation_prompts = EVALUATION_PROMPTS_SCENES
        # For EmoSet, we generate images for each prompt × emotion combination
        is_rafdb_portrait_mode = False
    elif args.dataset == "rafdb":
        emotions = EMOTIONS_RAFDB
        # For RAFDB, generate portrait prompts using the template for each emotion
        # This creates one prompt per emotion (portrait generation)
        # Each prompt already contains the emotion, so we generate one image per prompt
        evaluation_prompts = [PORTRAIT_PROMPT_TEMPLATE.format(emotion=emotion) for emotion in emotions]
        is_rafdb_portrait_mode = True
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}. Must be 'emoset' or 'rafdb'")
    
    # Normalize dataset size
    size_normalized = args.dataset_size.upper()
    if not size_normalized.endswith("K"):
        try:
            num = int(size_normalized)
            size_normalized = f"{num // 1000}K"
        except ValueError:
            pass
    
    # Setup paths
    # Check if this is a scripts approach (from scripts/train_text_to_image_lora.py)
    scripts_approaches = [
        "portraits", "emoset_label2image", "emoset_multicond",
        "emoset_multicond_classifier_001", "emoset_multicond_classifier_01"
    ]
    
    if args.approach in scripts_approaches:
        # Scripts format: located in weights/ directory
        weights_dir = os.path.join(REPO_ROOT, "weights", args.approach)
    elif args.approach == "classifier_guidance":
        # Classifier guidance uses different path
        weights_dir = os.path.join(STORAGE_BASE, "Weights", "classifier_guidance")
    elif args.approach == "baseline":
        # Baseline doesn't need weights
        weights_dir = None
    else:
        # Standard weights format: in Weights/{size}/{approach}
        weights_dir = os.path.join(STORAGE_BASE, "Weights", size_normalized, args.approach)
    
    # Always save evaluation images in validation_images/{size}/{approach} in repository root
    if args.images_dir:
        images_dir = args.images_dir
    else:
        images_dir = os.path.join(REPO_ROOT, "validation_images", size_normalized, args.approach)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Evaluation reports go to Evaluations directory in repository root
        output_dir = os.path.join(REPO_ROOT, "Evaluations", size_normalized, args.approach)
    
    # Check weights exist (skip for classifier_guidance and baseline as they use different paths)
    if args.approach not in ["classifier_guidance", "baseline"]:
        if weights_dir and not os.path.exists(weights_dir):
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
    print(f"Dataset: {args.dataset}")
    print(f"Task: {args.task}")
    print(f"Prompts: {len(evaluation_prompts)}")
    print(f"Emotions: {len(emotions)}")
    if is_rafdb_portrait_mode:
        # For RAFDB portraits, multiple images per emotion
        total_images = len(evaluation_prompts) * RAFDB_IMAGES_PER_EMOTION
        print(f"Images per emotion: {RAFDB_IMAGES_PER_EMOTION}")
    else:
        # For EmoSet scenes, images for each prompt × emotion combination
        total_images = len(evaluation_prompts) * len(emotions)
    print(f"Total Images: {total_images}")
    print("=" * 60)
    
    # Setup device (needed for both generation and evaluation)
    device = setup_device()
    
    # Check if images already exist
    if is_rafdb_portrait_mode:
        # For RAFDB: multiple images per emotion
        expected_total = len(evaluation_prompts) * RAFDB_IMAGES_PER_EMOTION
    else:
        # For EmoSet: images for each prompt × emotion combination
        expected_total = len(evaluation_prompts) * len(emotions)
    
    existing_count = 0
    missing_images = []
    
    if os.path.exists(images_dir):
        if is_rafdb_portrait_mode:
            # For RAFDB: multiple images per emotion
            for prompt_idx, prompt in enumerate(evaluation_prompts):
                emotion = emotions[prompt_idx]  # Prompts are in same order as emotions
                for img_idx in range(RAFDB_IMAGES_PER_EMOTION):
                    filename = f"prompt{prompt_idx:02d}_{emotion}_{img_idx:02d}.png"
                    image_path = os.path.join(images_dir, filename)
                    if os.path.exists(image_path):
                        existing_count += 1
                    else:
                        missing_images.append((prompt_idx, prompt_idx, emotion, img_idx))
        else:
            # For EmoSet: iterate over all prompt × emotion combinations
            for prompt_idx, prompt in enumerate(evaluation_prompts):
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
        if is_rafdb_portrait_mode:
            # For RAFDB: multiple images per emotion
            for prompt_idx, prompt in enumerate(evaluation_prompts):
                emotion = emotions[prompt_idx]  # Prompts are in same order as emotions
                emotion_idx = prompt_idx
                for img_idx in range(RAFDB_IMAGES_PER_EMOTION):
                    filename = f"prompt{prompt_idx:02d}_{emotion}_{img_idx:02d}.png"
                    image_path = os.path.join(images_dir, filename)
                    if os.path.exists(image_path):
                        # Portrait prompt already includes emotion
                        full_prompt = prompt
                        natural_prompt = prompt
                        
                        results.append({
                            "image_path": image_path,
                            "prompt": prompt,
                            "emotion": emotion,
                            "emotion_idx": emotion_idx,
                            "prompt_idx": prompt_idx,
                            "img_idx": img_idx,
                            "full_prompt": full_prompt,
                            "natural_prompt": natural_prompt,
                        })
        else:
            # For EmoSet: iterate over all prompt × emotion combinations
            for prompt_idx, prompt in enumerate(evaluation_prompts):
                for emotion_idx, emotion in enumerate(emotions):
                    filename = f"prompt{prompt_idx:02d}_{emotion}.png"
                    image_path = os.path.join(images_dir, filename)
                    if os.path.exists(image_path):
                        # Determine prompt format based on task and approach
                        if args.approach == "baseline":
                            # Baseline: use "{prompt}, {emotion}" format
                            full_prompt = f"{prompt}, {emotion}"
                        elif args.task == "label2image":
                            full_prompt = emotion
                        elif args.task == "multimodal":
                            full_prompt = f"{prompt} <{emotion}>"
                        else:
                            # Default: emotion token in prompt
                            emotion_token = f"<{emotion}>" if emotion in EMOTIONS_EMOSET else emotion
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
        if args.approach == "baseline":
            # Baseline: load vanilla SD 1.4 without any fine-tuning
            pipe = load_baseline_pipeline(device)
            classifier = None
            emotion_tokens = []
            emotion_embedding = None
        elif args.approach == "classifier_guidance":
            # For classifier_guidance, weights are in a different location
            classifier_weights_dir = os.path.join(STORAGE_BASE, "Weights", "classifier_guidance")
            # Import from classifier_guidance module
            import sys
            cg_inference_path = os.path.join(REPO_ROOT, "approaches", "classifier_guidance", "src")
            if cg_inference_path not in sys.path:
                sys.path.insert(0, cg_inference_path)
            from inference import load_pipeline, load_classifier
            
            pipe = load_pipeline(device=device)
            classifier_path = os.path.join(classifier_weights_dir, "classifier.pt")
            if not os.path.exists(classifier_path):
                classifier_path = os.path.join(classifier_weights_dir, "classifier_large.pt")
            classifier = load_classifier(classifier_path, device=device)
            # CRITICAL: Ensure classifier stays in float32 and never gets converted
            classifier = classifier.float()  # Convert entire model to float32
            # Register hook to prevent dtype conversion
            def keep_float32(module, input):
                for param in module.parameters():
                    if param.dtype != torch.float32:
                        param.data = param.data.to(dtype=torch.float32)
                for buffer in module.buffers():
                    if buffer.dtype != torch.int64 and buffer.dtype != torch.float32:
                        buffer.data = buffer.data.to(dtype=torch.float32)
            for m in classifier.modules():
                m.register_forward_pre_hook(keep_float32)
            emotion_tokens = []
            emotion_embedding = None
        else:
            pipe, emotion_tokens = load_generation_pipeline(weights_dir, device)
            classifier = None
            
            # Load emotion embedding for multimodal task if needed
            # Check if this is a scripts multimodal approach
            multimodal_approaches = ["emoset_multicond", "emoset_multicond_classifier_001", "emoset_multicond_classifier_01"]
            emotion_embedding = None
            if args.approach in multimodal_approaches or args.task == "multimodal":
                from torch import nn
                emotion_embedding_path = os.path.join(weights_dir, "emotion_embedding.pth")
                if os.path.exists(emotion_embedding_path):
                    emotion_embedding = nn.Embedding(len(emotions), pipe.text_encoder.config.hidden_size)
                    emotion_embedding.load_state_dict(torch.load(emotion_embedding_path, map_location=device))
                    emotion_embedding.eval()
                    emotion_embedding = emotion_embedding.to(device)
                    print("  Loaded emotion embedding for multimodal task")
                else:
                    print(f"  Warning: emotion_embedding.pth not found at {emotion_embedding_path}")
        
        if existing_count > 0:
            print(f"\n⚠ Found {existing_count}/{expected_total} existing images.")
            if missing_images:
                print(f"  Missing {len(missing_images)} images. Regenerating all to ensure consistency.")
        
        # Generate all images
        generate_kwargs = {
            "pipe": pipe,
            "prompts": evaluation_prompts,
            "emotions": emotions,
            "output_dir": images_dir,
            "seed": args.seed,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "approach": args.approach,
            "task": args.task,
            "dataset": args.dataset,  # Pass dataset to handle RAFDB portrait prompts
            "is_rafdb_portrait_mode": is_rafdb_portrait_mode,  # Pass flag for RAFDB mode
        }
        
        if args.approach == "classifier_guidance":
            generate_kwargs["classifier"] = classifier
            generate_kwargs["classifier_scale"] = args.classifier_scale
        
        if emotion_embedding is not None:
            generate_kwargs["emotion_embedding"] = emotion_embedding
        
        results = generate_images(**generate_kwargs)
        
        # Free GPU memory
        del pipe
        if classifier is not None:
            del classifier
        if emotion_embedding is not None:
            del emotion_embedding
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"💾 Freed generation models. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    if not results:
        print("Error: No images to evaluate")
        return 1
    
    # Evaluate based on dataset type
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
        # Free memory
        del emotionclip_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif args.dataset == "rafdb":
        # Load ViT model for evaluation
        vit_processor, vit_model = load_vit_model(device)
        results, metrics = evaluate_emotions_vit(
            vit_processor,
            vit_model,
            results,
            emotions,
        )
        # Free memory
        del vit_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset}")
    
    # Create evaluation report
    report_dir = create_evaluation_report(results, metrics, output_dir, emotions, evaluation_prompts)
    
    # Print summary
    print_summary(metrics, emotions)
    
    print(f"\n📁 Full evaluation saved to: {output_dir}")
    print(f"📊 Report available at: {report_dir}")
    
    # Final cleanup
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\n💾 Memory cleanup complete. GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
