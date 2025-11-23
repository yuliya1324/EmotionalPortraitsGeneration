"""
Simple script to generate specific test images with emotion tokens.
Loads model directly from Data cache to avoid downloads.
"""

import os
import sys
import torch
import json
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import PeftModel

# Set HuggingFace cache directory
DATA_DIR = "/Data/yash.bhardwaj"
os.environ["HF_HOME"] = os.path.join(DATA_DIR, "cache", "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(DATA_DIR, "cache", "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(DATA_DIR, "cache", "huggingface", "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(DATA_DIR, "cache", "huggingface", "hub")


def setup_device():
    """Detect and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def load_model_from_cache():
    """Load model components directly from cache."""
    print("[DEBUG] Entering load_model_from_cache()", flush=True)
    device = setup_device()
    model_id = "runwayml/stable-diffusion-v1-5"
    
    print(f"\n[DEBUG] Loading model components from cache: {model_id}", flush=True)
    
    # Load components individually
    print("[DEBUG] Step 1: Loading tokenizer...", flush=True)
    try:
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id,
            subfolder="tokenizer",
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface")
        )
        print("[DEBUG] ✓ Tokenizer loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading tokenizer: {e}", flush=True)
        raise
    
    print("[DEBUG] Step 2: Loading text encoder...", flush=True)
    try:
        text_encoder = CLIPTextModel.from_pretrained(
            model_id,
            subfolder="text_encoder",
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("[DEBUG] ✓ Text encoder loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading text encoder: {e}", flush=True)
        raise
    
    print("[DEBUG] Step 3: Loading UNet...", flush=True)
    try:
        unet = UNet2DConditionModel.from_pretrained(
            model_id,
            subfolder="unet",
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("[DEBUG] ✓ UNet loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading UNet: {e}", flush=True)
        raise
    
    print("[DEBUG] Step 4: Loading VAE...", flush=True)
    try:
        vae = AutoencoderKL.from_pretrained(
            model_id,
            subfolder="vae",
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        print("[DEBUG] ✓ VAE loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading VAE: {e}", flush=True)
        raise
    
    print("[DEBUG] Step 5: Loading scheduler...", flush=True)
    try:
        scheduler = DDPMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface")
        )
        print("[DEBUG] ✓ Scheduler loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading scheduler: {e}", flush=True)
        raise
    
    # Create pipeline using from_pretrained with components
    print("[DEBUG] Step 6: Creating pipeline...", flush=True)
    try:
        # Try loading the full pipeline from cache
        print("[DEBUG] Attempting to load full pipeline from cache...", flush=True)
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=os.path.join(DATA_DIR, "cache", "huggingface"),
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            local_files_only=True
        )
        print("[DEBUG] ✓ Pipeline loaded from cache", flush=True)
    except Exception as e:
        print(f"[DEBUG] Could not load full pipeline, using fallback: {e}", flush=True)
        # Fallback: create pipeline with loaded components
        from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
        print("[DEBUG] Creating pipeline from individual components...", flush=True)
        pipe = StableDiffusionPipeline(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            safety_checker=None,
            requires_safety_checker=False,
            feature_extractor=None
        )
        print("[DEBUG] ✓ Pipeline created from components", flush=True)
    
    print("[DEBUG] Returning pipeline and tokenizer", flush=True)
    return pipe, tokenizer


def generate_image(pipeline, prompt, output_path, seed=42, num_inference_steps=50, guidance_scale=7.5):
    """Generate a single image with the given prompt."""
    print(f"[DEBUG] Entering generate_image() for prompt: '{prompt}'", flush=True)
    try:
        device = pipeline.device
        print(f"[DEBUG] Pipeline device: {device}", flush=True)
        print(f"[DEBUG] Output path: {output_path}", flush=True)
        
        print(f"[DEBUG] Creating generator with seed {seed}...", flush=True)
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[DEBUG] ✓ Generator created", flush=True)
        
        print(f"[DEBUG] Calling pipeline.generate() with {num_inference_steps} steps, guidance={guidance_scale}...", flush=True)
        print(f"[DEBUG] This may take a while (30-60 seconds)...", flush=True)
        result = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )
        print(f"[DEBUG] ✓ Pipeline generation completed", flush=True)
        
        print(f"[DEBUG] Extracting image from result...", flush=True)
        image = result.images[0]
        print(f"[DEBUG] ✓ Image extracted, size: {image.size}", flush=True)
        
        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            print(f"[DEBUG] Creating output directory: {output_dir}", flush=True)
            os.makedirs(output_dir, exist_ok=True)
            print(f"[DEBUG] ✓ Output directory ready", flush=True)
        
        print(f"[DEBUG] Saving image to {output_path}...", flush=True)
        image.save(output_path)
        print(f"[DEBUG] ✓ Saved to {output_path}", flush=True)
        
        return image
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR in generate_image: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def main():
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    print("="*80, flush=True)
    print("STARTING IMAGE GENERATION SCRIPT", flush=True)
    print("="*80, flush=True)
    print("Starting image generation script...", flush=True)
    device = setup_device()
    print(f"[DEBUG] Device set to: {device}", flush=True)
    
    # Paths
    lora_path = os.path.join(DATA_DIR, "outputs", "final_model")
    learned_embeds_path = os.path.join(DATA_DIR, "outputs", "final_model", "learned_embeds.bin")
    tokenizer_info_path = os.path.join(DATA_DIR, "outputs", "final_model", "tokenizer_info.json")
    print(f"Paths configured: lora={lora_path}, embeds={learned_embeds_path}", flush=True)
    
    # Load model
    print("="*50, flush=True)
    print("Loading model from cache...", flush=True)
    print("="*50, flush=True)
    print("[DEBUG] Calling load_model_from_cache()...", flush=True)
    pipeline, tokenizer = load_model_from_cache()
    print("[DEBUG] ✓ Model loaded from cache", flush=True)
    
    # Load tokenizer info
    print("\n[DEBUG] Loading tokenizer info...", flush=True)
    try:
        with open(tokenizer_info_path, 'r') as f:
            tokenizer_info = json.load(f)
        emotion_tokens = tokenizer_info.get("emotion_tokens", [
            '<amusement>', '<awe>', '<contentment>', '<excitement>',
            '<anger>', '<disgust>', '<fear>', '<sadness>'
        ])
        print(f"[DEBUG] ✓ Loaded {len(emotion_tokens)} emotion tokens", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading tokenizer info: {e}", flush=True)
        raise
    
    # Add tokens to tokenizer FIRST, before loading embeddings
    print("\n[DEBUG] Adding emotion tokens to tokenizer...", flush=True)
    try:
        num_added = tokenizer.add_tokens(emotion_tokens)
        print(f"[DEBUG] Added {num_added} tokens", flush=True)
        if num_added > 0:
            print(f"[DEBUG] Resizing embeddings (this may take a moment)...", flush=True)
            # Resize embeddings - this creates new random embeddings for new tokens
            pipeline.text_encoder.resize_token_embeddings(len(tokenizer))
            print(f"[DEBUG] ✓ Embeddings resized to {len(tokenizer)} tokens", flush=True)
        else:
            print(f"[DEBUG] No new tokens added (already present)", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR adding tokens: {e}", flush=True)
        raise
    
    # Load learned embeddings AFTER resizing
    print("\n[DEBUG] Loading learned embeddings...", flush=True)
    try:
        print(f"[DEBUG] Loading from {learned_embeds_path}...", flush=True)
        learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
        print(f"[DEBUG] ✓ Loaded {len(learned_embeds)} embeddings from file", flush=True)
        
        device_emb = pipeline.text_encoder.get_input_embeddings().weight.device
        dtype_emb = pipeline.text_encoder.get_input_embeddings().weight.dtype
        print(f"[DEBUG] Embedding device: {device_emb}, dtype: {dtype_emb}", flush=True)
        
        print(f"[DEBUG] Setting embeddings for {len(learned_embeds)} tokens...", flush=True)
        for i, (token, embedding) in enumerate(learned_embeds.items()):
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                with torch.no_grad():
                    # Ensure embedding is on correct device and dtype
                    embedding_tensor = embedding.to(device=device_emb, dtype=dtype_emb)
                    pipeline.text_encoder.get_input_embeddings().weight[token_id] = embedding_tensor
                if i < 3 or i >= len(learned_embeds) - 1:  # Log first 3 and last
                    print(f"[DEBUG]   Loaded embedding for {token} (id: {token_id})", flush=True)
            else:
                print(f"[DEBUG]   ERROR: Token {token} has unk_token_id!", flush=True)
        print(f"[DEBUG] ✓ All embeddings loaded", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR loading embeddings: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Verify embeddings were loaded
    print("\n[DEBUG] Verifying embeddings...", flush=True)
    try:
        for token in ['<anger>', '<amusement>']:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                loaded_emb = pipeline.text_encoder.get_input_embeddings().weight[token_id].cpu()
                saved_emb = learned_embeds[token]
                diff = torch.norm(loaded_emb - saved_emb).item()
                print(f"[DEBUG]   {token}: embedding difference = {diff:.6f} (should be ~0 if loaded correctly)", flush=True)
        print(f"[DEBUG] ✓ Embedding verification complete", flush=True)
    except Exception as e:
        print(f"[DEBUG] ⚠ Warning during verification: {e}", flush=True)
    
    # Load LoRA weights and merge them
    print("\n[DEBUG] Loading LoRA weights...", flush=True)
    print(f"[DEBUG] LoRA path: {lora_path}", flush=True)
    print(f"[DEBUG] Checking if path exists: {os.path.exists(lora_path)}", flush=True)
    try:
        # Load as PEFT model and merge
        print(f"[DEBUG] Attempting to load LoRA as PEFT model...", flush=True)
        print(f"[DEBUG] This may take 30-60 seconds to merge...", flush=True)
        pipeline.unet = PeftModel.from_pretrained(pipeline.unet, lora_path)
        print(f"[DEBUG] ✓ LoRA loaded, now merging (this may take a while)...", flush=True)
        pipeline.unet = pipeline.unet.merge_and_unload()
        print(f"[DEBUG] ✓ LoRA weights loaded and merged successfully", flush=True)
    except Exception as e:
        print(f"[DEBUG] ⚠ Warning: Could not load LoRA as PEFT: {e}", flush=True)
        import traceback
        traceback.print_exc()
        try:
            # Try loading as attention processors
            print(f"[DEBUG] Attempting to load as attention processors...", flush=True)
            pipeline.unet.load_attn_procs(lora_path)
            print(f"[DEBUG] ✓ LoRA weights loaded as attention processors", flush=True)
        except Exception as e2:
            print(f"[DEBUG] ✗ ERROR loading LoRA: {e2}", flush=True)
            import traceback
            traceback.print_exc()
            print(f"[DEBUG] ⚠ WARNING: LoRA weights not loaded - using base model only!", flush=True)
    
    # Move to device
    print(f"\n[DEBUG] Moving pipeline to device: {device}", flush=True)
    try:
        if torch.cuda.is_available():
            print(f"[DEBUG] Moving to CUDA device...", flush=True)
            print(f"[DEBUG] This may take 10-30 seconds...", flush=True)
            pipeline = pipeline.to(device)
            print(f"[DEBUG] ✓ Pipeline moved to {device}", flush=True)
        else:
            print(f"[DEBUG] CUDA not available, keeping on CPU", flush=True)
        print(f"[DEBUG] Disabling progress bar...", flush=True)
        pipeline.set_progress_bar_config(disable=True)
        print(f"[DEBUG] ✓ Progress bar disabled", flush=True)
    except Exception as e:
        print(f"[DEBUG] ✗ ERROR moving to device: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    # Output directory in current directory
    output_dir = os.path.join(os.getcwd(), "test_images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"[DEBUG] Output directory: {output_dir}", flush=True)
    
    # Generate test images
    print("\n" + "="*50, flush=True)
    print("Generating test images...", flush=True)
    print("="*50, flush=True)
    
    # Base prompt - a landscape scene that will clearly show emotional contrast
    # Landscape scenes show emotions through atmosphere, weather, colors, and mood
    base_prompt = "A beautiful landscape, rolling hills, dramatic sky, photorealistic, high quality"
    
    # Generate 3 images with different emotions: amusement, sadness, and anger
    emotions = [
        ("<amusement>", "landscape_amusement.png"),
        ("<sadness>", "landscape_sadness.png"),
        ("<anger>", "landscape_anger.png"),
    ]
    
    generated_files = []
    for i, (emotion_token, filename) in enumerate(emotions, 1):
        print(f"\n[DEBUG] ===== Generating Image {i}/3 =====", flush=True)
        prompt = f"{base_prompt} {emotion_token}"
        output_path = os.path.join(output_dir, filename)
        try:
            generate_image(pipeline, prompt, output_path, seed=42, guidance_scale=9.0)
            print(f"[DEBUG] ✓ Image {i} completed", flush=True)
            generated_files.append(filename)
        except Exception as e:
            print(f"[DEBUG] ✗ ERROR generating image {i}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            raise
    
    print("\n" + "="*50, flush=True)
    print("Test image generation completed!", flush=True)
    print("="*50, flush=True)
    print(f"Images saved to: {output_dir}", flush=True)
    for filename in generated_files:
        print(f"  - {filename}", flush=True)


if __name__ == "__main__":
    main()
