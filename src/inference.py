"""
Inference script for emotion-conditioned image generation.
Visualizes the same prompt with different emotion tokens.
"""

import os
import argparse
import torch
from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel
from PIL import Image
import json
import numpy as np

# Set HuggingFace cache directory
DATA_DIR = "/Data/yash.bhardwaj"
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(DATA_DIR, "cache", "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(DATA_DIR, "cache", "huggingface", "transformers")


# Emotion tokens
EMOTION_TOKENS = [
    '<amusement>',
    '<awe>',
    '<contentment>',
    '<excitement>',
    '<anger>',
    '<disgust>',
    '<fear>',
    '<sadness>',
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


def load_model_and_adapters(
    model_id: str = "runwayml/stable-diffusion-v1-5",
    lora_path: str = None,
    learned_embeds_path: str = None,
    tokenizer_info_path: str = None
):
    """
    Load Stable Diffusion model with LoRA adapters and learned embeddings.
    
    Args:
        model_id: Base Stable Diffusion model ID
        lora_path: Path to LoRA checkpoint directory
        learned_embeds_path: Path to learned embeddings file
        tokenizer_info_path: Path to tokenizer info JSON file
        
    Returns:
        Tuple of (pipeline, tokenizer, emotion_tokens)
    """
    device = setup_device()
    
    # Set default paths if not provided
    if lora_path is None:
        lora_path = os.path.join(DATA_DIR, "outputs", "final_model")
    if learned_embeds_path is None:
        learned_embeds_path = os.path.join(DATA_DIR, "outputs", "final_model", "learned_embeds.bin")
    if tokenizer_info_path is None:
        tokenizer_info_path = os.path.join(DATA_DIR, "outputs", "final_model", "tokenizer_info.json")
    
    print(f"\nLoading base model: {model_id}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
    except Exception as e:
        print(f"Warning: Could not load with local_files_only: {e}")
        print("Attempting to load without local_files_only...")
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
    
    # Load tokenizer info
    if os.path.exists(tokenizer_info_path):
        with open(tokenizer_info_path, 'r') as f:
            tokenizer_info = json.load(f)
        emotion_tokens = tokenizer_info.get("emotion_tokens", EMOTION_TOKENS)
        print(f"Loaded tokenizer info: {len(emotion_tokens)} emotion tokens")
    else:
        emotion_tokens = EMOTION_TOKENS
        print(f"Warning: {tokenizer_info_path} not found, using default tokens")
    
    # Add tokens to tokenizer if not already present
    tokenizer = pipe.tokenizer
    num_added = tokenizer.add_tokens(emotion_tokens)
    if num_added > 0:
        print(f"Added {num_added} tokens to tokenizer")
        pipe.text_encoder.resize_token_embeddings(len(tokenizer))
    
    # Load learned embeddings
    if os.path.exists(learned_embeds_path):
        print(f"\nLoading learned embeddings from {learned_embeds_path}")
        learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        
        # Set embeddings in text encoder
        for token, embedding in learned_embeds.items():
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                with torch.no_grad():
                    pipe.text_encoder.get_input_embeddings().weight[token_id] = embedding.to(
                        pipe.text_encoder.get_input_embeddings().weight.device
                    )
                print(f"  Loaded embedding for {token} (id: {token_id})")
            else:
                print(f"  Warning: Could not find token ID for {token}")
    else:
        print(f"Warning: {learned_embeds_path} not found, using default embeddings")
    
    # Load LoRA weights
    if os.path.exists(lora_path):
        print(f"\nLoading LoRA weights from {lora_path}")
        try:
            # Try to load as PEFT model
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            pipe.unet = pipe.unet.merge_and_unload()
            print("  LoRA weights loaded and merged")
        except Exception as e:
            print(f"  Warning: Could not load LoRA as PEFT model: {e}")
            print("  Attempting to load directly...")
            try:
                pipe.unet.load_attn_procs(lora_path)
                print("  LoRA weights loaded directly")
            except Exception as e2:
                print(f"  Error loading LoRA: {e2}")
    else:
        print(f"Warning: {lora_path} not found, using base model")
    
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    
    return pipe, tokenizer, emotion_tokens


def visualize_scene(
    pipeline,
    prompt: str,
    emotion_tokens: list,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    output_path: str = "emotion_comparison.png",
    grid_cols: int = 4
):
    """
    Generate the same scene with different emotion tokens and create a grid.
    Uses the same seed for all generations to preserve scene structure.
    
    Args:
        pipeline: Stable Diffusion pipeline
        prompt: Base prompt (e.g., "A photo of a park")
        emotion_tokens: List of emotion token strings
        seed: Random seed for reproducibility (same seed used for all emotions)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Guidance scale for classifier-free guidance
        output_path: Path to save the grid image
        grid_cols: Number of columns in the grid
        
    Returns:
        PIL Image of the grid
    """
    device = pipeline.device
    
    print(f"\nGenerating images for prompt: '{prompt}'")
    print(f"Using fixed seed: {seed} (same for all emotions to preserve structure)")
    print(f"Emotion tokens: {emotion_tokens}")
    
    images = []
    prompts = []
    
    for i, emotion_token in enumerate(emotion_tokens):
        full_prompt = f"{prompt} {emotion_token}"
        prompts.append(full_prompt)
        
        print(f"\n[{i+1}/{len(emotion_tokens)}] Generating: {full_prompt}")
        
        try:
            # Create a new generator with the SAME seed for each emotion
            # This ensures identical initial noise, preserving scene structure
            generator = torch.Generator(device=device).manual_seed(seed)
            
            image = pipeline(
                prompt=full_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            images.append(image)
        except Exception as e:
            print(f"  Error generating image: {e}")
            # Create a blank image as placeholder
            images.append(Image.new('RGB', (512, 512), color='gray'))
    
    # Create grid
    print(f"\nCreating grid image ({len(images)} images)...")
    grid_image = create_image_grid(images, prompts, cols=grid_cols)
    
    # Save grid
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    grid_image.save(output_path)
    print(f"Saved grid to {output_path}")
    
    return grid_image


def create_image_grid(images: list, labels: list = None, cols: int = 4):
    """
    Create a grid image from a list of PIL Images.
    
    Args:
        images: List of PIL Images
        labels: Optional list of labels for each image
        cols: Number of columns in the grid
        
    Returns:
        PIL Image of the grid
    """
    if not images:
        raise ValueError("No images provided")
    
    num_images = len(images)
    rows = (num_images + cols - 1) // cols
    
    # Get image dimensions
    img_width, img_height = images[0].size
    
    # Create grid canvas
    grid_width = cols * img_width
    grid_height = rows * img_height
    
    # Add space for labels if provided
    label_height = 30 if labels else 0
    grid_height += rows * label_height
    
    grid_image = Image.new('RGB', (grid_width, grid_height), color='white')
    
    # Paste images into grid
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = col * img_width
        y = row * (img_height + label_height)
        
        grid_image.paste(img, (x, y))
        
        # Add label if provided
        if labels and idx < len(labels):
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(grid_image)
            
            # Try to use a nice font, fallback to default
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
            except:
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
                except:
                    font = ImageFont.load_default()
            
            # Extract emotion token from label
            label = labels[idx]
            if ' <' in label:
                emotion_part = label.split(' <')[1].split('>')[0]
                label_text = f"<{emotion_part}>"
            else:
                label_text = label
            
            # Draw text with background
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = x + (img_width - text_width) // 2
            text_y = y + img_height + (label_height - text_height) // 2
            
            # Draw background rectangle
            padding = 5
            draw.rectangle(
                [
                    text_x - padding,
                    text_y - padding,
                    text_x + text_width + padding,
                    text_y + text_height + padding
                ],
                fill='black'
            )
            
            # Draw text
            draw.text((text_x, text_y), label_text, fill='white', font=font)
    
    return grid_image


def main():
    parser = argparse.ArgumentParser(
        description="Generate emotion-conditioned images"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A photo of a park",
        help="Base prompt to generate with different emotions"
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        default=os.path.join(DATA_DIR, "outputs", "final_model"),
        help="Path to LoRA checkpoint directory"
    )
    parser.add_argument(
        "--learned_embeds_path",
        type=str,
        default=os.path.join(DATA_DIR, "outputs", "final_model", "learned_embeds.bin"),
        help="Path to learned embeddings file"
    )
    parser.add_argument(
        "--tokenizer_info_path",
        type=str,
        default=os.path.join(DATA_DIR, "outputs", "final_model", "tokenizer_info.json"),
        help="Path to tokenizer info JSON file"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=os.path.join(DATA_DIR, "outputs", "emotion_comparison.png"),
        help="Output path for grid image"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of diffusion steps"
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale"
    )
    parser.add_argument(
        "--grid_cols",
        type=int,
        default=4,
        help="Number of columns in grid"
    )
    
    args = parser.parse_args()
    
    # Load model and adapters
    print("="*50)
    print("Loading model and adapters...")
    print("="*50)
    pipeline, tokenizer, emotion_tokens = load_model_and_adapters(
        lora_path=args.lora_path,
        learned_embeds_path=args.learned_embeds_path,
        tokenizer_info_path=args.tokenizer_info_path
    )
    
    # Generate visualization
    print("\n" + "="*50)
    print("Generating emotion variations...")
    print("="*50)
    grid_image = visualize_scene(
        pipeline,
        args.prompt,
        emotion_tokens,
        seed=args.seed,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        output_path=args.output_path,
        grid_cols=args.grid_cols
    )
    
    print("\n" + "="*50)
    print("Inference completed!")
    print("="*50)
    print(f"Grid image saved to: {args.output_path}")


if __name__ == "__main__":
    main()

