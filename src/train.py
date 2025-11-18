"""
Training script for fine-tuning Stable Diffusion v1.5 with emotion-conditioned generation.
Uses LoRA for UNet adaptation and learned embeddings for emotion tokens.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import json
from PIL import Image

from dataset import EmoSetLocalDataset


# Emotion tokens to add
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


def add_tokens_to_tokenizer(tokenizer: CLIPTokenizer, tokens: list) -> int:
    """
    Add special tokens to the tokenizer.
    
    Args:
        tokenizer: CLIP tokenizer
        tokens: List of token strings to add
        
    Returns:
        Number of tokens added
    """
    num_added = tokenizer.add_tokens(tokens)
    print(f"Added {num_added} tokens to tokenizer")
    return num_added


def resize_text_encoder_embeddings(text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer):
    """
    Resize text encoder embeddings to accommodate new tokens.
    
    Args:
        text_encoder: CLIP text encoder model
        tokenizer: CLIP tokenizer
    """
    text_encoder.resize_token_embeddings(len(tokenizer))
    print(f"Resized text encoder embeddings to {len(tokenizer)} tokens")


def initialize_token_embeddings(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    emotion_tokens: list,
    init_word: str = "style"
):
    """
    Initialize new token embeddings with the embedding of a neutral word.
    
    Args:
        text_encoder: CLIP text encoder model
        tokenizer: CLIP tokenizer
        emotion_tokens: List of emotion token strings
        init_word: Neutral word to use for initialization (default: "style")
    """
    # Get the embedding of the initialization word
    init_token_id = tokenizer.encode(init_word, add_special_tokens=False)[0]
    init_embedding = text_encoder.get_input_embeddings().weight[init_token_id].clone()
    
    # Initialize each emotion token with the init embedding
    for token in emotion_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            with torch.no_grad():
                text_encoder.get_input_embeddings().weight[token_id] = init_embedding.clone()
            print(f"Initialized {token} (id: {token_id}) with embedding from '{init_word}'")
        else:
            print(f"Warning: Could not find token ID for {token}")


def setup_lora_unet(unet: UNet2DConditionModel, r: int = 16, lora_alpha: int = 32):
    """
    Apply LoRA to UNet cross-attention layers.
    
    Args:
        unet: UNet model
        r: LoRA rank
        lora_alpha: LoRA alpha scaling parameter
        
    Returns:
        PEFT model with LoRA adapters
    """
    # LoRA configuration for cross-attention layers
    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
    )
    
    # Apply LoRA
    unet = get_peft_model(unet, lora_config)
    print(f"Applied LoRA to UNet (r={r}, alpha={lora_alpha})")
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in unet.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
    
    return unet


def get_optimizer_groups(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    emotion_tokens: list,
    lr_lora: float = 1e-4,
    lr_embeddings: float = 1e-3  # Higher LR for embeddings
):
    """
    Create separate optimizer parameter groups for LoRA and embeddings.
    
    Args:
        unet: UNet model with LoRA
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        emotion_tokens: List of emotion token strings
        lr_lora: Learning rate for LoRA parameters (default: 1e-4)
        lr_embeddings: Learning rate for token embeddings (default: 1e-3)
        
    Returns:
        List of parameter groups for optimizer
    """
    # Get LoRA parameters
    lora_params = []
    for name, param in unet.named_parameters():
        if param.requires_grad:
            lora_params.append(param)
    
    # Get embedding parameters for emotion tokens
    embedding_params = []
    emotion_token_ids = [
        tokenizer.convert_tokens_to_ids(token)
        for token in emotion_tokens
    ]
    
    for token_id in emotion_token_ids:
        if token_id != tokenizer.unk_token_id:
            embedding_params.append(
                text_encoder.get_input_embeddings().weight[token_id]
            )
    
    # Create parameter groups
    param_groups = [
        {"params": lora_params, "lr": lr_lora, "name": "lora"},
        {"params": embedding_params, "lr": lr_embeddings, "name": "embeddings"},
    ]
    
    print(f"Optimizer groups:")
    print(f"  LoRA: {len(lora_params)} parameters, lr={lr_lora}")
    print(f"  Embeddings: {len(embedding_params)} parameters, lr={lr_embeddings}")
    
    return param_groups


def compute_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor
) -> torch.Tensor:
    """
    Compute MSE loss between predicted and actual noise.
    
    Args:
        noise_pred: Predicted noise from model
        noise: Ground truth noise
        timesteps: Diffusion timesteps
        
    Returns:
        Loss tensor
    """
    return F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")


def generate_validation_images(
    pipeline,
    test_prompts: list,
    output_dir: str,
    step: int,
    seed: int = 42,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
):
    """
    Generate validation images for monitoring training progress.
    
    Args:
        pipeline: Stable Diffusion pipeline
        test_prompts: List of test prompts to generate
        output_dir: Directory to save images
        step: Current training step
        seed: Random seed
        num_inference_steps: Number of diffusion steps
        guidance_scale: Guidance scale
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = pipeline.device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipeline.set_progress_bar_config(disable=True)
    
    for i, prompt in enumerate(test_prompts):
        try:
            image = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]
            
            # Save image
            filename = f"step_{step}_prompt_{i}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"  Generated validation image: {filepath} (prompt: {prompt})")
        except Exception as e:
            print(f"  Warning: Failed to generate validation image for '{prompt}': {e}")


def save_checkpoint(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    emotion_tokens: list,
    output_dir: str,
    epoch: int,
    step: int
):
    """
    Save LoRA weights and learned token embeddings.
    
    Args:
        unet: UNet model with LoRA
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        emotion_tokens: List of emotion token strings
        output_dir: Output directory
        epoch: Current epoch
        step: Current step
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save LoRA weights
    unet.save_pretrained(output_dir)
    print(f"Saved LoRA weights to {output_dir}")
    
    # Save learned embeddings
    learned_embeds = {}
    for token in emotion_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            learned_embeds[token] = text_encoder.get_input_embeddings().weight[token_id].cpu()
    
    embed_path = os.path.join(output_dir, "learned_embeds.bin")
    torch.save(learned_embeds, embed_path)
    print(f"Saved learned embeddings to {embed_path}")
    
    # Save tokenizer info
    tokenizer_info = {
        "emotion_tokens": emotion_tokens,
        "token_ids": {
            token: tokenizer.convert_tokens_to_ids(token)
            for token in emotion_tokens
        }
    }
    info_path = os.path.join(output_dir, "tokenizer_info.json")
    with open(info_path, 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    print(f"Saved tokenizer info to {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Train emotion-conditioned Stable Diffusion")
    parser.add_argument("--data_dir", type=str, default="./data/emoset_captioned_10k",
                       help="Path to local dataset directory")
    parser.add_argument("--output_dir", type=str, default="output/final_model",
                       help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default="output/logs",
                       help="Directory for validation images")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr_lora", type=float, default=1e-4,
                       help="Learning rate for LoRA parameters")
    parser.add_argument("--lr_embeddings", type=float, default=1e-3,
                       help="Learning rate for token embeddings")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--validation_steps", type=int, default=500,
                       help="Generate validation images every N steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--init_word", type=str, default="style",
                       help="Word to use for initializing emotion token embeddings")
    parser.add_argument("--test_prompt_1", type=str, default="A living room <fear>",
                       help="First test prompt for validation")
    parser.add_argument("--test_prompt_2", type=str, default="A living room <excitement>",
                       help="Second test prompt for validation")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no"
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Setup device
    device = setup_device()
    
    # Load dataset
    print("\n" + "="*50)
    print("Loading dataset...")
    print("="*50)
    dataset = EmoSetLocalDataset(data_dir=args.data_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load Stable Diffusion pipeline
    print("\n" + "="*50)
    print("Loading Stable Diffusion v1.5...")
    print("="*50)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer"
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet"
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler"
    )
    
    # Freeze text encoder (except new token embeddings)
    text_encoder.requires_grad_(False)
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    # Add emotion tokens
    print("\n" + "="*50)
    print("Adding emotion tokens...")
    print("="*50)
    num_added = add_tokens_to_tokenizer(tokenizer, EMOTION_TOKENS)
    resize_text_encoder_embeddings(text_encoder, tokenizer)
    
    # Enable gradients for new token embeddings
    emotion_token_ids = [
        tokenizer.convert_tokens_to_ids(token)
        for token in EMOTION_TOKENS
    ]
    for token_id in emotion_token_ids:
        if token_id != tokenizer.unk_token_id:
            text_encoder.get_input_embeddings().weight[token_id].requires_grad = True
    
    # Initialize token embeddings
    initialize_token_embeddings(
        text_encoder,
        tokenizer,
        EMOTION_TOKENS,
        init_word=args.init_word
    )
    
    # Setup LoRA for UNet
    print("\n" + "="*50)
    print("Setting up LoRA...")
    print("="*50)
    unet = setup_lora_unet(unet, r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # Move models to device
    text_encoder.to(device)
    unet.to(device)
    
    # Setup optimizer
    print("\n" + "="*50)
    print("Setting up optimizer...")
    print("="*50)
    param_groups = get_optimizer_groups(
        unet,
        text_encoder,
        tokenizer,
        EMOTION_TOKENS,
        lr_lora=args.lr_lora,
        lr_embeddings=args.lr_embeddings
    )
    optimizer = torch.optim.AdamW(param_groups)
    
    # Prepare with accelerator
    unet, text_encoder, optimizer, dataloader = accelerator.prepare(
        unet, text_encoder, optimizer, dataloader
    )
    
    # Create validation pipeline (only on main process)
    # Note: We'll update it with unwrapped models during validation
    validation_pipeline = None
    if accelerator.is_main_process:
        # Create a temporary pipeline for validation (will be updated with latest weights)
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        )
        validation_pipeline = validation_pipeline.to(device)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    global_step = 0
    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    test_prompts = [args.test_prompt_1, args.test_prompt_2]
    
    for epoch in range(args.num_epochs):
        unet.train()
        text_encoder.train()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet, text_encoder):
                # Get batch data
                images, prompts, emotions = batch
                images = images.to(device)
                
                # Tokenize prompts
                text_inputs = tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=77,
                    truncation=True,
                    return_tensors="pt"
                )
                text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
                
                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = text_encoder(text_inputs.input_ids).last_hidden_state
                
                # Sample noise
                noise = torch.randn_like(images)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (images.shape[0],),
                    device=device
                ).long()
                
                # Add noise to images
                noisy_images = noise_scheduler.add_noise(images, noise, timesteps)
                
                # Predict noise
                noise_pred = unet(
                    noisy_images,
                    timesteps,
                    encoder_hidden_states=text_embeddings
                ).sample
                
                # Compute loss
                loss = compute_loss(noise_pred, noise, timesteps)
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Validation logging
            if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                print(f"\n[Step {global_step}] Generating validation images...")
                # Update pipeline with latest weights
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                validation_pipeline.unet = unwrapped_unet.to(device)
                validation_pipeline.text_encoder = unwrapped_text_encoder.to(device)
                validation_pipeline.unet.eval()
                validation_pipeline.text_encoder.eval()
                generate_validation_images(
                    validation_pipeline,
                    test_prompts,
                    args.log_dir,
                    global_step,
                    seed=args.seed
                )
                # Set back to train mode
                unwrapped_unet.train()
                unwrapped_text_encoder.train()
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(
                        accelerator.unwrap_model(unet),
                        accelerator.unwrap_model(text_encoder),
                        tokenizer,
                        EMOTION_TOKENS,
                        args.output_dir,
                        epoch,
                        global_step
                    )
        
        # Validation at end of epoch
        if accelerator.is_main_process:
            print(f"\n[End of Epoch {epoch+1}] Generating validation images...")
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            validation_pipeline.unet = unwrapped_unet.to(device)
            validation_pipeline.text_encoder = unwrapped_text_encoder.to(device)
            validation_pipeline.unet.eval()
            validation_pipeline.text_encoder.eval()
            generate_validation_images(
                validation_pipeline,
                test_prompts,
                args.log_dir,
                global_step,
                seed=args.seed
            )
            # Set back to train mode for next epoch
            unwrapped_unet.train()
            unwrapped_text_encoder.train()
        
        # Save at end of epoch
        if accelerator.is_main_process:
            save_checkpoint(
                accelerator.unwrap_model(unet),
                accelerator.unwrap_model(text_encoder),
                tokenizer,
                EMOTION_TOKENS,
                args.output_dir,
                epoch,
                global_step
            )
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Final checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
