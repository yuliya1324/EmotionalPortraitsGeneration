"""
Training script for fine-tuning Stable Diffusion v1.5 with emotion-conditioned generation.
Uses LoRA for UNet adaptation and learned embeddings for emotion tokens.
"""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from diffusers import StableDiffusionPipeline, DDPMScheduler, UNet2DConditionModel, AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm
import json
import numpy as np
from PIL import Image

import sys
from pathlib import Path

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠ wandb not available. Install with: pip install wandb")

# Add shared directory to path
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
SHARED_DIR = REPO_ROOT / "shared" / "src"
sys.path.insert(0, str(SHARED_DIR))

from dataset import EmoSetLocalDataset

# Set HuggingFace cache directory
# Use environment variable or default to repository root
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
STORAGE_BASE = os.getenv("EMOTIONAL_PORTRAITS_BASE", str(REPO_ROOT))
CACHE_DIR = os.getenv("HF_CACHE_DIR", os.path.join(STORAGE_BASE, "cache"))
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "transformers")
os.environ["HF_HUB_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "hub")


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


def get_word_embedding(text_encoder: CLIPTextModel, tokenizer: CLIPTokenizer, word: str):
    """Get embedding for a word (handles multi-token words by averaging)."""
    tokens = tokenizer(word, add_special_tokens=False, return_tensors="pt")
    token_ids = tokens["input_ids"][0]
    embeddings = text_encoder.get_input_embeddings().weight[token_ids]
    if len(embeddings) > 1:
        return embeddings.mean(dim=0).clone()
    return embeddings[0].clone()


def initialize_token_embeddings(
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    emotion_tokens: list,
    emotion_words: list = None,
    init_word: str = None
):
    """
    Initialize new token embeddings with emotion word embeddings.
    
    Args:
        text_encoder: CLIP text encoder model
        tokenizer: CLIP tokenizer
        emotion_tokens: List of emotion token strings (e.g., ["<amusement>", ...])
        emotion_words: List of corresponding emotion words (e.g., ["amusement", ...])
        init_word: Fallback word if emotion_words not provided (default: "style")
    """
    # Default emotion words if not provided
    if emotion_words is None:
        emotion_words = [
            "amusement", "anger", "awe", "contentment",
            "disgust", "excitement", "fear", "sadness"
        ]
    
    # Initialize each emotion token with its corresponding emotion word embedding
    for token, emotion_word in zip(emotion_tokens, emotion_words):
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id != tokenizer.unk_token_id:
            try:
                # Get emotion word embedding
                word_emb = get_word_embedding(text_encoder, tokenizer, emotion_word)
                with torch.no_grad():
                    text_encoder.get_input_embeddings().weight[token_id] = word_emb
                print(f"Initialized {token} (id: {token_id}) with embedding from '{emotion_word}'")
            except Exception as e:
                # Fallback to init_word if emotion word fails
                if init_word:
                    init_token_id = tokenizer.encode(init_word, add_special_tokens=False)[0]
                    init_embedding = text_encoder.get_input_embeddings().weight[init_token_id].clone()
                    with torch.no_grad():
                        text_encoder.get_input_embeddings().weight[token_id] = init_embedding
                    print(f"Initialized {token} (id: {token_id}) with fallback '{init_word}' (error: {e})")
                else:
                    print(f"Warning: Could not initialize {token}: {e}")
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
    # IMPORTANT: Access embeddings through the model's embedding layer
    embedding_params = []
    emotion_token_ids = [
        tokenizer.convert_tokens_to_ids(token)
        for token in emotion_tokens
    ]
    
    # Get the embedding layer
    embedding_layer = text_encoder.get_input_embeddings()
    full_weight = embedding_layer.weight
    
    # CRITICAL FIX: Add the FULL embedding weight tensor to optimizer
    # The key insight: PyTorch can track gradients for specific rows of a 2D tensor
    # by setting requires_grad=True on the full tensor and then selectively on rows
    
    # Enable gradients on the full weight tensor
    # This is necessary for PyTorch to track gradients on specific rows
    full_weight.requires_grad = True
    
    # Now selectively disable gradients for non-emotion tokens
    # We'll create a mask to zero out gradients for non-emotion tokens during backward
    # But first, let's try a simpler approach: just add the full tensor with requires_grad=True
    # and rely on the fact that only emotion tokens are used in the loss
    
    # Actually, we need to be more careful. Let's add individual rows but ensure they're
    # properly connected. The issue is that views might not work with accelerator.
    # So let's add the full tensor and use a custom approach.
    
    # For now, let's try adding the full weight tensor
    # The optimizer should only update rows that have requires_grad=True
    # But we need to ensure all rows can have gradients computed
    embedding_params.append(full_weight)
    
    # Verify emotion tokens have requires_grad=True
    emotion_token_count = 0
    for token_id in emotion_token_ids:
        if token_id != tokenizer.unk_token_id:
            if full_weight[token_id].requires_grad:
                emotion_token_count += 1
    
    print(f"  Added full embedding weight tensor ({full_weight.shape[0]} x {full_weight.shape[1]})")
    print(f"  Emotion tokens with requires_grad=True: {emotion_token_count}/{len([t for t in emotion_token_ids if t != tokenizer.unk_token_id])}")
    
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
        
    Returns:
        List of tuples (prompt, image_path) for logging
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = pipeline.device
    generator = torch.Generator(device=device).manual_seed(seed)
    
    pipeline.set_progress_bar_config(disable=True)
    
    validation_results = []
    
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
            validation_results.append((prompt, filepath))
        except Exception as e:
            print(f"  Warning: Failed to generate validation image for '{prompt}': {e}")
    
    return validation_results


def save_checkpoint(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    emotion_tokens: list,
    output_dir: str,
    epoch: int,
    step: int,
    is_best: bool = False
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
        is_best: If True, save as best model checkpoint
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


def load_checkpoint(
    unet: UNet2DConditionModel,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    checkpoint_dir: str
):
    """
    Load LoRA weights and learned token embeddings from checkpoint.
    
    Args:
        unet: UNet model (will be loaded with LoRA)
        text_encoder: CLIP text encoder
        tokenizer: CLIP tokenizer
        checkpoint_dir: Directory containing checkpoint files
        
    Returns:
        Tuple of (loaded_unet, loaded_text_encoder, resume_step)
        resume_step is estimated from checkpoint or None if unknown
    """
    print(f"\nLoading checkpoint from {checkpoint_dir}...")
    
    # Load LoRA weights
    adapter_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    
    if os.path.exists(adapter_path) or os.path.exists(config_path):
        from peft import PeftModel as PEFTModel
        from peft.utils import set_peft_model_state_dict
        from safetensors.torch import load_file
        
        if isinstance(unet, PEFTModel):
            # If already a PEFT model, load adapter state dict
            if os.path.exists(adapter_path):
                adapter_state_dict = load_file(adapter_path)
                set_peft_model_state_dict(unet, adapter_state_dict)
                print(f"✓ Loaded LoRA weights from {checkpoint_dir}")
            else:
                print(f"⚠ Warning: Adapter weights file not found")
        else:
            # If not a PEFT model, we need to set up LoRA first, then load weights
            # Load the adapter config to get the LoRA parameters
            import json
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    adapter_config = json.load(f)
                # Apply LoRA with the same config as saved
                from peft import LoraConfig, get_peft_model, TaskType
                lora_config = LoraConfig(
                    r=adapter_config.get('r', 16),
                    lora_alpha=adapter_config.get('lora_alpha', 32),
                    target_modules=adapter_config.get('target_modules', ["to_k", "to_q", "to_v", "to_out.0"]),
                    task_type=TaskType.FEATURE_EXTRACTION,
                    inference_mode=False,
                )
                unet = get_peft_model(unet, lora_config)
                # Now load the weights
                if os.path.exists(adapter_path):
                    adapter_state_dict = load_file(adapter_path)
                    set_peft_model_state_dict(unet, adapter_state_dict)
                print(f"✓ Loaded LoRA weights from {checkpoint_dir}")
            else:
                # Fallback to PeftModel.from_pretrained
                unet = PeftModel.from_pretrained(unet, checkpoint_dir)
                print(f"✓ Loaded LoRA weights from {checkpoint_dir} (using from_pretrained)")
    else:
        print(f"⚠ Warning: No LoRA weights found in {checkpoint_dir}")
    
    # Load learned embeddings
    embed_path = os.path.join(checkpoint_dir, "learned_embeds.bin")
    if os.path.exists(embed_path):
        learned_embeds = torch.load(embed_path, map_location="cpu")
        with torch.no_grad():
            for token, embedding in learned_embeds.items():
                token_id = tokenizer.convert_tokens_to_ids(token)
                if token_id != tokenizer.unk_token_id:
                    text_encoder.get_input_embeddings().weight[token_id] = embedding.to(
                        text_encoder.get_input_embeddings().weight.device
                    )
        print(f"✓ Loaded learned embeddings from {embed_path}")
    else:
        print(f"⚠ Warning: No learned embeddings found in {checkpoint_dir}")
    
    # Try to estimate resume step from checkpoint directory name or files
    resume_step = None
    # Check if there's a loss history file that might tell us the step
    # For now, we'll estimate based on save_steps (checkpoint was saved at step 500)
    # This will be handled by the caller
    
    return unet, text_encoder, resume_step


def main():
    parser = argparse.ArgumentParser(description="Train emotion-conditioned Stable Diffusion")
    parser.add_argument("--data_dir", type=str, default=os.path.join(STORAGE_BASE, "Datasets", "emoset_captioned_10k"),
                       help="Path to local dataset directory")
    parser.add_argument("--output_dir", type=str, default=os.path.join(STORAGE_BASE, "Weights", "10K", "baseline_lora"),
                       help="Output directory for checkpoints")
    parser.add_argument("--log_dir", type=str, default=os.path.join(STORAGE_BASE, "Logs", "10K", "baseline_lora"),
                       help="Directory for validation images and logs")
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=10,
                       help="Number of training epochs")
    parser.add_argument("--lr_lora", type=float, default=1e-4,
                       help="Learning rate for LoRA parameters")
    parser.add_argument("--lr_embeddings", type=float, default=5e-3,
                       help="Learning rate for token embeddings")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha")
    parser.add_argument("--save_steps", type=int, default=500,
                       help="Save checkpoint every N steps")
    parser.add_argument("--validation_steps", type=int, default=1000,
                       help="Generate validation images every N steps")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=3,
                       help="Gradient accumulation steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--init_word", type=str, default=None,
                       help="Fallback word for initializing emotion token embeddings (default: use emotion words)")
    parser.add_argument("--emotion_reg_weight", type=float, default=0.05,
                       help="Weight for emotion regularization loss (0.0 to disable)")
    parser.add_argument("--warmup_steps", type=int, default=500,
                       help="Number of warmup steps for learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay for optimizer")
    parser.add_argument("--early_stopping_patience", type=int, default=5,
                       help="Early stopping patience (epochs)")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                       help="Minimum LR as ratio of initial LR")
    parser.add_argument("--test_prompt_1", type=str, default="A living room <fear>",
                       help="First test prompt for validation")
    parser.add_argument("--test_prompt_2", type=str, default="A living room <excitement>",
                       help="Second test prompt for validation")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Path to checkpoint directory to resume training from")
    parser.add_argument("--resume_step", type=int, default=None,
                       help="Step number to resume from (if not specified, will try to infer from checkpoint)")
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use Weights & Biases for experiment tracking")
    parser.add_argument("--wandb_project", type=str, default="emotional-portraits",
                       help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/team name (optional)")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Wandb run name (optional, auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
        log_with="wandb" if (args.use_wandb and WANDB_AVAILABLE) else None,
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Initialize wandb (only on main process)
    if args.use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
        # Generate run name if not provided
        if args.wandb_name is None:
            dataset_name = os.path.basename(args.data_dir).replace("emoset_captioned_", "")
            args.wandb_name = f"baseline_lora_{dataset_name}_r{args.lora_r}_lr{args.lr_lora}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "approach": "baseline_lora",
                "dataset": os.path.basename(args.data_dir),
                "batch_size": args.batch_size,
                "num_epochs": args.num_epochs,
                "lr_lora": args.lr_lora,
                "lr_embeddings": args.lr_embeddings,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
                "save_steps": args.save_steps,
                "validation_steps": args.validation_steps,
                "seed": args.seed,
                "init_word": args.init_word,
                "warmup_steps": args.warmup_steps,
                "weight_decay": args.weight_decay,
            },
            tags=["baseline_lora", os.path.basename(args.data_dir)],
        )
        print(f"✓ Initialized wandb: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        print("⚠ wandb requested but not available. Install with: pip install wandb")
    
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
        num_workers=8,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Load Stable Diffusion pipeline
    print("\n" + "="*50)
    print("Loading Stable Diffusion v1.5...")
    print("="*50)
    model_id = "runwayml/stable-diffusion-v1-5"
    
    # Use cache_dir to load from Data folder cache (avoids disk quota issues)
    cache_dir = os.path.join(CACHE_DIR, "huggingface")
    tokenizer = CLIPTokenizer.from_pretrained(
        model_id,
        subfolder="tokenizer",
        cache_dir=cache_dir
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_id,
        subfolder="text_encoder",
        cache_dir=cache_dir
    )
    unet = UNet2DConditionModel.from_pretrained(
        model_id,
        subfolder="unet",
        cache_dir=cache_dir
    )
    # Enable gradient checkpointing to save memory
    if hasattr(unet, 'enable_gradient_checkpointing'):
        unet.enable_gradient_checkpointing()
        print("✓ Enabled gradient checkpointing for UNet (memory optimization)")
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_id,
        subfolder="scheduler",
        cache_dir=cache_dir
    )
    vae = AutoencoderKL.from_pretrained(
        model_id,
        subfolder="vae",
        cache_dir=cache_dir
    )
    
    # Freeze VAE and text encoder (except new token embeddings)
    vae.requires_grad_(False)
    vae.eval()
    
    # CRITICAL FIX: Don't freeze the entire text_encoder, as that prevents
    # us from selectively enabling gradients for emotion token embeddings
    # Instead, freeze individual components EXCEPT the embedding layer
    text_encoder.text_model.encoder.requires_grad_(False)
    text_encoder.text_model.final_layer_norm.requires_grad_(False)
    text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)
    
    # Keep embedding layer trainable (we'll use a hook to zero gradients for non-emotion tokens)
    # This allows gradients to flow, but we'll mask out non-emotion token gradients
    embedding_layer = text_encoder.get_input_embeddings()
    embedding_layer.weight.requires_grad = True
    
    # Add emotion tokens
    print("\n" + "="*50)
    print("Adding emotion tokens...")
    print("="*50)
    num_added = add_tokens_to_tokenizer(tokenizer, EMOTION_TOKENS)
    resize_text_encoder_embeddings(text_encoder, tokenizer)
    
    # NOTE: Embedding layer is already set to requires_grad=True above
    # We'll use a gradient hook to mask out non-emotion token gradients
    # No need to set requires_grad on individual rows
    
    # Initialize token embeddings with emotion word embeddings
    # IMPORTANT: Order must match EMOTION_TOKENS exactly!
    EMOTION_WORDS = [
        "amusement",  # matches '<amusement>'
        "awe",        # matches '<awe>'
        "contentment", # matches '<contentment>'
        "excitement", # matches '<excitement>'
        "anger",      # matches '<anger>'
        "disgust",    # matches '<disgust>'
        "fear",       # matches '<fear>'
        "sadness"     # matches '<sadness>'
    ]
    initialize_token_embeddings(
        text_encoder,
        tokenizer,
        EMOTION_TOKENS,
        emotion_words=EMOTION_WORDS,
        init_word=args.init_word
    )
    
    # Load checkpoint if resuming (BEFORE setting up LoRA to avoid structure conflicts)
    resume_step = args.resume_step
    checkpoint_step = None
    if args.resume_from:
        # Load checkpoint - this will set up LoRA if needed
        unet, text_encoder, checkpoint_step = load_checkpoint(
            unet, text_encoder, tokenizer, args.resume_from
        )
        # Check if LoRA was loaded from checkpoint
        from peft import PeftModel
        has_peft = isinstance(unet, PeftModel)
    else:
        has_peft = False
    
    # Setup LoRA for UNet if not already loaded from checkpoint
    if not has_peft:
        print("\n" + "="*50)
        print("Setting up LoRA...")
        print("="*50)
        unet = setup_lora_unet(unet, r=args.lora_r, lora_alpha=args.lora_alpha)
    
    # Set resume step if resuming
    if args.resume_from:
        if resume_step is None:
            resume_step = checkpoint_step
        if resume_step is None:
            # Estimate from save_steps - checkpoint was likely saved at a multiple of save_steps
            # Since we know it was saved at step 500, use that
            resume_step = 500
            print(f"⚠ Could not determine resume step from checkpoint, using estimated step: {resume_step}")
        else:
            print(f"✓ Resuming from step {resume_step}")
    
    # Move models to device
    vae.to(device)
    text_encoder.to(device)
    unet.to(device)
    
    # Prepare models with accelerator FIRST
    # This is critical: we need to prepare models before creating optimizer
    # so that optimizer references the wrapped model parameters
    print("\n" + "="*50)
    print("Preparing models with accelerator...")
    print("="*50)
    unet, text_encoder, dataloader = accelerator.prepare(
        unet, text_encoder, dataloader
    )
    
    # Setup optimizer AFTER prepare() to get correct parameter references
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
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay
    )
    
    # Prepare optimizer with accelerator
    optimizer = accelerator.prepare(optimizer)
    
    # CRITICAL: Register a hook to zero out gradients for non-emotion token embeddings
    # This ensures only emotion tokens get updated, even though the full weight tensor is trainable
    # Must be done AFTER prepare() to access the wrapped model
    emotion_token_ids_set = set([
        tokenizer.convert_tokens_to_ids(token)
        for token in EMOTION_TOKENS
        if tokenizer.convert_tokens_to_ids(token) != tokenizer.unk_token_id
    ])
    
    def zero_non_emotion_grads(grad):
        """Zero out gradients for non-emotion token embeddings"""
        if grad is not None:
            # Create a mask: 1 for emotion tokens, 0 for others
            mask = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)
            for token_id in emotion_token_ids_set:
                if token_id < grad.shape[0]:
                    mask[token_id] = 1.0
            # Apply mask to gradients (broadcast across embedding dimension)
            # Use out-of-place operation to avoid issues with mixed precision scaler
            grad = grad * mask.unsqueeze(1)
        return grad
    
    # Register the hook on the embedding weight tensor
    # Access unwrapped model to register hook on the actual parameter
    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
    embedding_weight = unwrapped_text_encoder.get_input_embeddings().weight
    embedding_weight.register_hook(zero_non_emotion_grads)
    print(f"  Registered gradient masking hook for {len(emotion_token_ids_set)} emotion tokens")
    
    # Setup learning rate scheduler
    print("\n" + "="*50)
    print("Setting up learning rate scheduler...")
    print("="*50)
    # Calculate total training steps
    total_training_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps
    
    # NOTE: We'll manually update learning rates to avoid scheduler wrapper conflicts with mixed precision
    # The scheduler wrapper intercepts optimizer.step() and can cause "Attempting to unscale FP16 gradients" errors
    # Instead, we'll calculate and set learning rates manually in the training loop
    lr_scheduler = None  # We'll update LR manually
    
    print(f"Total training steps: {total_training_steps}")
    print(f"Warmup steps: {args.warmup_steps}")
    print(f"Cosine annealing steps: {total_training_steps - args.warmup_steps}")
    print("Using manual LR updates to avoid mixed precision conflicts")
    
    # Create validation pipeline (only on main process)
    # Note: We'll update it with unwrapped models during validation
    validation_pipeline = None
    if accelerator.is_main_process:
        # Create a temporary pipeline for validation (will be updated with latest weights)
        # Use cache_dir to load from Data folder cache
        cache_dir = os.path.join(CACHE_DIR, "huggingface")
        validation_pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            tokenizer=tokenizer,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            safety_checker=None,
            requires_safety_checker=False,
            cache_dir=cache_dir
        )
        validation_pipeline = validation_pipeline.to(device)
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    # Initialize resume variables (resume_step was already set above if resuming)
    # If resume_step wasn't set above, check args
    if 'resume_step' not in locals():
        resume_step = args.resume_step if args.resume_from else None
    global_step = resume_step if resume_step else 0
    start_epoch = 0
    
    # Calculate starting epoch from resume step
    if resume_step:
        num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
        start_epoch = resume_step // num_update_steps_per_epoch
        # No scheduler to update - we use manual LR updates
        print(f"Resuming from step {resume_step}, epoch {start_epoch + 1}")
    
    num_update_steps_per_epoch = len(dataloader) // args.gradient_accumulation_steps
    test_prompts = [args.test_prompt_1, args.test_prompt_2]
    
    # Loss tracking variables
    loss_history = []
    best_loss = float('inf')
    no_improve_count = 0
    ema_loss = None  # Exponential moving average
    ema_alpha = 0.99
    # Store loss components for logging
    last_reconstruction_loss = 0.0
    last_emotion_reg_loss = 0.0
    
    # Try to load loss history if resuming
    if args.resume_from:
        loss_file = os.path.join(args.log_dir, "loss_history.json")
        if os.path.exists(loss_file):
            try:
                with open(loss_file, 'r') as f:
                    loss_data = json.load(f)
                    if 'loss_history' in loss_data:
                        loss_history = loss_data['loss_history'][:resume_step] if resume_step else loss_data['loss_history']
                    if 'best_loss' in loss_data:
                        best_loss = loss_data['best_loss']
                    if 'ema_loss' in loss_data and loss_history:
                        ema_loss = loss_data.get('ema_loss', None)
                print(f"✓ Loaded loss history from {loss_file}")
            except Exception as e:
                print(f"⚠ Could not load loss history: {e}")
    
    for epoch in range(start_epoch, args.num_epochs):
        unet.train()
        text_encoder.train()
        
        progress_bar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(progress_bar):
            # Skip batches if resuming until we reach the resume step
            if resume_step and global_step < resume_step:
                # Only increment global_step when we would do an optimizer step
                # (every gradient_accumulation_steps batches)
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                    global_step += 1
                continue
            
            with accelerator.accumulate(unet, text_encoder):
                # Get batch data
                images, prompts, emotions = batch
                images = images.to(device)
                
                # Encode images to latents using VAE
                with torch.no_grad():
                    # VAE expects images in [0, 1] range, but we have [-1, 1], so convert back
                    images_for_vae = (images + 1.0) / 2.0
                    latents = vae.encode(images_for_vae).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
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
                # NOTE: We need gradients for embedding updates, so NO torch.no_grad() here
                # Only the embedding weights for emotion tokens have requires_grad=True
                text_embeddings = text_encoder(text_inputs["input_ids"]).last_hidden_state
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (latents.shape[0],),
                    device=device
                ).long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Ensure all inputs match UNet dtype to avoid dtype mismatch errors
                unet_dtype = next(unet.parameters()).dtype
                noisy_latents = noisy_latents.to(unet_dtype)
                text_embeddings = text_embeddings.to(unet_dtype)
                
                # Predict noise
                # PEFT wraps the model - access base model through base_model attribute
                # The LoRA adapters are automatically applied by PEFT
                if hasattr(unet, 'base_model') and hasattr(unet.base_model, 'model'):
                    # PEFT model structure
                    model_output = unet.base_model.model(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=text_embeddings,
                        return_dict=True
                    )
                else:
                    # Direct UNet (shouldn't happen but just in case)
                    model_output = unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=text_embeddings,
                        return_dict=True
                    )
                noise_pred = model_output.sample
                
                # Compute reconstruction loss
                reconstruction_loss = compute_loss(noise_pred, noise, timesteps)
                
                # Compute emotion regularization loss (if enabled)
                emotion_reg_loss = torch.tensor(0.0, device=device, dtype=reconstruction_loss.dtype)
                if args.emotion_reg_weight > 0:
                    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                    embedding_layer = unwrapped_text_encoder.get_input_embeddings()
                    emotion_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in EMOTION_TOKENS]
                    
                    # Get emotion word embeddings for regularization
                    EMOTION_WORDS = [
                        "amusement", "anger", "awe", "contentment",
                        "disgust", "excitement", "fear", "sadness"
                    ]
                    
                    reg_losses = []
                    for token_id, emotion_word in zip(emotion_token_ids, EMOTION_WORDS):
                        if token_id != tokenizer.unk_token_id:
                            token_emb = embedding_layer.weight[token_id]
                            word_emb = get_word_embedding(unwrapped_text_encoder, tokenizer, emotion_word).to(
                                device=device, dtype=token_emb.dtype
                            )
                            # MSE loss to push token toward emotion word
                            reg_losses.append(F.mse_loss(token_emb, word_emb))
                    
                    if reg_losses:
                        emotion_reg_loss = torch.stack(reg_losses).mean()
                
                # Total loss
                loss = reconstruction_loss + args.emotion_reg_weight * emotion_reg_loss
                
                # Store loss components for logging
                last_reconstruction_loss = reconstruction_loss.item() if isinstance(reconstruction_loss, torch.Tensor) else float(reconstruction_loss)
                last_emotion_reg_loss = emotion_reg_loss.item() if isinstance(emotion_reg_loss, torch.Tensor) else float(emotion_reg_loss)
                
                # Backward pass
                # DEBUG: Log detailed state before backward (especially after validation to catch issues)
                is_post_validation = (global_step > 0 and (global_step - 1) % args.validation_steps == 0) or (global_step % args.validation_steps == 0)
                if is_post_validation:
                    print(f"\n[Step {global_step}] Pre-backward state:", flush=True)
                    if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                        state_count = len(accelerator.scaler._per_optimizer_states)
                        print(f"  Scaler state: {state_count} optimizer(s) tracked", flush=True)
                        if state_count > 0:
                            for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                                stage = opt_state.get('stage', 'unknown')
                                print(f"    Optimizer stage: {stage}", flush=True)
                    else:
                        print(f"  ⚠ No scaler found", flush=True)
                    
                    # Log gradient states before backward
                    unet_has_grads = any(p.grad is not None for p in unet.parameters() if p.requires_grad)
                    text_encoder_has_grads = any(p.grad is not None for p in text_encoder.parameters() if p.requires_grad)
                    print(f"  UNet has gradients before backward: {unet_has_grads}", flush=True)
                    print(f"  Text encoder has gradients before backward: {text_encoder_has_grads}", flush=True)
                    if unet_has_grads:
                        # Check gradient dtype
                        for p in unet.parameters():
                            if p.requires_grad and p.grad is not None:
                                print(f"  UNet gradient dtype: {p.grad.dtype}", flush=True)
                                break
                    if text_encoder_has_grads:
                        # Check gradient dtype
                        for p in text_encoder.parameters():
                            if p.requires_grad and p.grad is not None:
                                print(f"  Text encoder gradient dtype: {p.grad.dtype}", flush=True)
                                break
                
                accelerator.backward(loss)
                
                # DEBUG: Log detailed state after backward (especially after validation to catch issues)
                if is_post_validation:
                    print(f"[Step {global_step}] Post-backward state:", flush=True)
                    if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                        state_count = len(accelerator.scaler._per_optimizer_states)
                        print(f"  Scaler state: {state_count} optimizer(s) tracked", flush=True)
                        if state_count > 0:
                            for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                                stage = opt_state.get('stage', 'unknown')
                                found_inf_per_device = opt_state.get('found_inf_per_device', {})
                                print(f"    Optimizer stage: {stage}", flush=True)
                                if found_inf_per_device:
                                    print(f"    Found inf per device: {found_inf_per_device}", flush=True)
                    else:
                        print(f"  ⚠ No scaler found", flush=True)
                    
                    # Log gradient states after backward
                    unet_has_grads = any(p.grad is not None for p in unet.parameters() if p.requires_grad)
                    text_encoder_has_grads = any(p.grad is not None for p in text_encoder.parameters() if p.requires_grad)
                    print(f"  UNet has gradients after backward: {unet_has_grads}", flush=True)
                    print(f"  Text encoder has gradients after backward: {text_encoder_has_grads}", flush=True)
                    if unet_has_grads:
                        # Check gradient dtype
                        for p in unet.parameters():
                            if p.requires_grad and p.grad is not None:
                                print(f"  UNet gradient dtype: {p.grad.dtype}, norm: {p.grad.norm().item():.6f}", flush=True)
                                break
                    if text_encoder_has_grads:
                        # Check gradient dtype
                        for p in text_encoder.parameters():
                            if p.requires_grad and p.grad is not None:
                                print(f"  Text encoder gradient dtype: {p.grad.dtype}, norm: {p.grad.norm().item():.6f}", flush=True)
                                break
                
                # Monitor embedding gradients BEFORE optimizer.step() (every 100 steps)
                embedding_values_before = None
                if global_step % 100 == 0 and accelerator.is_main_process:
                    # Get unwrapped text encoder to access embeddings
                    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                    embedding_layer = unwrapped_text_encoder.get_input_embeddings()
                    emotion_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in EMOTION_TOKENS]
                    
                    # Check gradients BEFORE optimizer step
                    embedding_grads = []
                    embedding_values_before = []
                    embedding_in_optimizer = False
                    
                    for param_group in optimizer.param_groups:
                        if param_group.get('name') == 'embeddings':
                            embedding_in_optimizer = True
                            for param in param_group['params']:
                                embedding_values_before.append(param.data.norm().item())
                                if param.grad is not None:
                                    embedding_grads.append(param.grad.norm().item())
                    
                    # Also check embedding layer directly for gradients
                    for token_id in emotion_token_ids:
                        if token_id != tokenizer.unk_token_id:
                            emb_weight = embedding_layer.weight[token_id]
                            if emb_weight.grad is not None:
                                embedding_grads.append(emb_weight.grad.norm().item())
                    
                    print(f"\n[Step {global_step}] Embedding Update Verification (BEFORE optimizer.step()):", flush=True)
                    print(f"  Embeddings in optimizer: {'✓' if embedding_in_optimizer else '✗'}", flush=True)
                    if embedding_values_before:
                        avg_emb_norm = np.mean(embedding_values_before)
                        print(f"  Avg embedding value norm (before update): {avg_emb_norm:.6f}", flush=True)
                    if embedding_grads:
                        avg_grad_norm = np.mean(embedding_grads)
                        print(f"  Avg embedding grad norm: {avg_grad_norm:.6f}", flush=True)
                        print(f"  ✓ Gradients are flowing to embeddings!", flush=True)
                    else:
                        print(f"  ✗ WARNING: No gradients found for embeddings!", flush=True)
                    print(flush=True)
                
                # Manually update learning rate BEFORE optimizer step
                # Calculate LR schedule (warmup + cosine annealing)
                current_scheduler_step = global_step
                if current_scheduler_step < args.warmup_steps:
                    # Warmup phase: linear from 0.1 to 1.0
                    lr_scale = 0.1 + (1.0 - 0.1) * (current_scheduler_step / args.warmup_steps)
                else:
                    # Cosine annealing phase
                    cosine_step = current_scheduler_step - args.warmup_steps
                    cosine_max = total_training_steps - args.warmup_steps
                    lr_scale = args.min_lr_ratio + (1.0 - args.min_lr_ratio) * 0.5 * (1 + np.cos(np.pi * cosine_step / cosine_max))
                
                # Update learning rates manually
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'lora':
                        param_group['lr'] = args.lr_lora * lr_scale
                    elif param_group.get('name') == 'embeddings':
                        param_group['lr'] = args.lr_embeddings * lr_scale
                
                # Optimizer step (no scheduler wrapper to interfere)
                # DEBUG: Log detailed state before optimizer step (especially after validation to catch issues)
                if is_post_validation:
                    print(f"\n[Step {global_step}] Pre-optimizer.step() state:", flush=True)
                    if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                        state_count = len(accelerator.scaler._per_optimizer_states)
                        optimizer_in_state = optimizer in accelerator.scaler._per_optimizer_states
                        underlying_in_state = (hasattr(optimizer, 'optimizer') and 
                                              optimizer.optimizer in accelerator.scaler._per_optimizer_states)
                        print(f"  Scaler state count: {state_count}", flush=True)
                        print(f"  Wrapped optimizer in state: {optimizer_in_state}", flush=True)
                        print(f"  Underlying optimizer in state: {underlying_in_state}", flush=True)
                        if optimizer_in_state or underlying_in_state:
                            opt_key = optimizer if optimizer_in_state else optimizer.optimizer
                            opt_state = accelerator.scaler._per_optimizer_states[opt_key]
                            stage = opt_state.get('stage', 'unknown')
                            found_inf_per_device = opt_state.get('found_inf_per_device', {})
                            print(f"  Optimizer stage: {stage}", flush=True)
                            if found_inf_per_device:
                                print(f"  Found inf per device: {found_inf_per_device}", flush=True)
                        else:
                            print(f"  ⚠ WARNING: Optimizer not found in scaler state!", flush=True)
                    else:
                        print(f"  ⚠ No scaler found", flush=True)
                    
                    # Log gradient states before optimizer step
                    unet_grad_info = []
                    for p in unet.parameters():
                        if p.requires_grad and p.grad is not None:
                            unet_grad_info.append((p.grad.dtype, p.grad.norm().item()))
                            break
                    text_encoder_grad_info = []
                    for p in text_encoder.parameters():
                        if p.requires_grad and p.grad is not None:
                            text_encoder_grad_info.append((p.grad.dtype, p.grad.norm().item()))
                            break
                    if unet_grad_info:
                        dtype, norm = unet_grad_info[0]
                        print(f"  UNet gradient dtype: {dtype}, norm: {norm:.6f}", flush=True)
                    if text_encoder_grad_info:
                        dtype, norm = text_encoder_grad_info[0]
                        print(f"  Text encoder gradient dtype: {dtype}, norm: {norm:.6f}", flush=True)
                
                try:
                    optimizer.step()
                    
                    # DEBUG: Log if optimizer step succeeded (especially after validation to catch issues)
                    if is_post_validation:
                        print(f"[Step {global_step}] ✓ Optimizer.step() completed successfully", flush=True)
                        
                        # Log post-optimizer state
                        if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                            state_count = len(accelerator.scaler._per_optimizer_states)
                            print(f"  Post-optimizer scaler state: {state_count} optimizer(s) tracked", flush=True)
                except Exception as e:
                    # DEBUG: Log detailed error information
                    print(f"\n[Step {global_step}] ✗✗✗ OPTIMIZER.STEP() FAILED ✗✗✗", flush=True)
                    print(f"  Error type: {type(e).__name__}", flush=True)
                    print(f"  Error message: {str(e)}", flush=True)
                    
                    # Log scaler state at failure
                    if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                        state_count = len(accelerator.scaler._per_optimizer_states)
                        print(f"  Scaler state at failure: {state_count} optimizer(s) tracked", flush=True)
                        if state_count > 0:
                            for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                                stage = opt_state.get('stage', 'unknown')
                                found_inf_per_device = opt_state.get('found_inf_per_device', {})
                                print(f"    Optimizer stage: {stage}", flush=True)
                                if found_inf_per_device:
                                    print(f"    Found inf per device: {found_inf_per_device}", flush=True)
                    
                    # Log gradient states at failure
                    unet_grad_info = []
                    for p in unet.parameters():
                        if p.requires_grad and p.grad is not None:
                            unet_grad_info.append((p.grad.dtype, p.grad.norm().item()))
                            break
                    text_encoder_grad_info = []
                    for p in text_encoder.parameters():
                        if p.requires_grad and p.grad is not None:
                            text_encoder_grad_info.append((p.grad.dtype, p.grad.norm().item()))
                            break
                    if unet_grad_info:
                        dtype, norm = unet_grad_info[0]
                        print(f"  UNet gradient dtype at failure: {dtype}, norm: {norm:.6f}", flush=True)
                    if text_encoder_grad_info:
                        dtype, norm = text_encoder_grad_info[0]
                        print(f"  Text encoder gradient dtype at failure: {dtype}, norm: {norm:.6f}", flush=True)
                    
                    # Log model states at failure
                    unwrapped_unet_err = accelerator.unwrap_model(unet)
                    unwrapped_text_encoder_err = accelerator.unwrap_model(text_encoder)
                    if hasattr(unwrapped_unet_err, 'base_model') and hasattr(unwrapped_unet_err.base_model, 'model'):
                        base_unet_err = unwrapped_unet_err.base_model.model
                    else:
                        base_unet_err = unwrapped_unet_err
                    print(f"  UNet training mode at failure: {base_unet_err.training}", flush=True)
                    print(f"  Text encoder training mode at failure: {unwrapped_text_encoder_err.training}", flush=True)
                    print(f"  Base UNet dtype at failure: {next(base_unet_err.parameters()).dtype}", flush=True)
                    print(f"  Text encoder dtype at failure: {next(unwrapped_text_encoder_err.parameters()).dtype}", flush=True)
                    
                    raise  # Re-raise the exception
                
                # Monitor embedding values AFTER optimizer.step() (every 100 steps)
                if global_step % 100 == 0 and accelerator.is_main_process:
                    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                    embedding_layer = unwrapped_text_encoder.get_input_embeddings()
                    emotion_token_ids = [tokenizer.convert_tokens_to_ids(token) for token in EMOTION_TOKENS]
                    
                    embedding_values_after = []
                    for param_group in optimizer.param_groups:
                        if param_group.get('name') == 'embeddings':
                            for param in param_group['params']:
                                embedding_values_after.append(param.data.norm().item())
                    
                    # Also check embedding layer directly
                    for token_id in emotion_token_ids:
                        if token_id != tokenizer.unk_token_id:
                            emb_weight = embedding_layer.weight[token_id]
                            embedding_values_after.append(emb_weight.data.norm().item())
                    
                    if embedding_values_after:
                        avg_emb_norm_after = np.mean(embedding_values_after)
                        print(f"[Step {global_step}] Embedding Update Verification (AFTER optimizer.step()):", flush=True)
                        print(f"  Avg embedding value norm (after update): {avg_emb_norm_after:.6f}", flush=True)
                        
                        # Compare with before if we have it
                        if embedding_values_before:
                            avg_emb_norm_before = np.mean(embedding_values_before)
                            diff = avg_emb_norm_after - avg_emb_norm_before
                            print(f"  Change in norm: {diff:.8f}", flush=True)
                            if abs(diff) < 1e-6:
                                print(f"  ✗ WARNING: Embeddings NOT changing! Norm difference is {diff:.8f}", flush=True)
                            else:
                                print(f"  ✓ Embeddings ARE changing!", flush=True)
                        print(flush=True)
                
                optimizer.zero_grad()
            
            global_step += 1
            
            # Loss tracking
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Update EMA
            if ema_loss is None:
                ema_loss = current_loss
            else:
                ema_loss = ema_alpha * ema_loss + (1 - ema_alpha) * current_loss
            
            # Update progress bar with enhanced metrics
            current_lr = optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{current_loss:.4f}',
                'ema_loss': f'{ema_loss:.4f}',
                'lr': f'{current_lr:.2e}'
            })
            
            # Log to wandb (every step for real-time monitoring)
            if args.use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
                log_dict = {
                    "train/loss": current_loss,
                    "train/reconstruction_loss": last_reconstruction_loss,
                    "train/emotion_reg_loss": last_emotion_reg_loss,
                    "train/ema_loss": ema_loss,
                    "train/learning_rate": current_lr,
                    "train/global_step": global_step,
                    "train/epoch": epoch + 1,
                }
                
                # Log learning rates for each parameter group
                for i, param_group in enumerate(optimizer.param_groups):
                    group_name = param_group.get('name', f'group_{i}')
                    log_dict[f"train/lr_{group_name}"] = param_group['lr']
                
                # Log GPU metrics if available
                if torch.cuda.is_available():
                    log_dict["system/gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024**2
                    log_dict["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                
                wandb.log(log_dict, step=global_step)
                
                wandb.log(log_dict, step=global_step)
            
            # Validation logging
            if global_step % args.validation_steps == 0 and accelerator.is_main_process:
                print(f"\n[Step {global_step}] ========== VALIDATION START ==========", flush=True)
                
                # DEBUG: Log pre-validation state
                print(f"[Step {global_step}] Pre-validation state:", flush=True)
                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    state_count = len(accelerator.scaler._per_optimizer_states)
                    print(f"  Scaler state: {state_count} optimizer(s) tracked", flush=True)
                    if state_count > 0:
                        for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                            stage = opt_state.get('stage', 'unknown')
                            print(f"    Optimizer stage: {stage}", flush=True)
                else:
                    print(f"  ⚠ No scaler found", flush=True)
                
                # Log model states
                unwrapped_unet_temp = accelerator.unwrap_model(unet)
                unwrapped_text_encoder_temp = accelerator.unwrap_model(text_encoder)
                print(f"  UNet training mode: {unwrapped_unet_temp.training}", flush=True)
                print(f"  Text encoder training mode: {unwrapped_text_encoder_temp.training}", flush=True)
                if hasattr(unwrapped_unet_temp, 'base_model') and hasattr(unwrapped_unet_temp.base_model, 'model'):
                    base_unet_temp = unwrapped_unet_temp.base_model.model
                else:
                    base_unet_temp = unwrapped_unet_temp
                print(f"  Base UNet dtype: {next(base_unet_temp.parameters()).dtype}", flush=True)
                print(f"  Text encoder dtype: {next(unwrapped_text_encoder_temp.parameters()).dtype}", flush=True)
                
                # Log gradient states
                has_grads = False
                for param in unet.parameters():
                    if param.requires_grad and param.grad is not None:
                        has_grads = True
                        print(f"  UNet has gradients: ✓ (dtype: {param.grad.dtype})", flush=True)
                        break
                if not has_grads:
                    print(f"  UNet has gradients: ✗", flush=True)
                
                has_grads = False
                for param in text_encoder.parameters():
                    if param.requires_grad and param.grad is not None:
                        has_grads = True
                        print(f"  Text encoder has gradients: ✓ (dtype: {param.grad.dtype})", flush=True)
                        break
                if not has_grads:
                    print(f"  Text encoder has gradients: ✗", flush=True)
                
                print(f"[Step {global_step}] Generating validation images...", flush=True)
                # CRITICAL: Use torch.no_grad() context to prevent any gradient computation during validation
                # This ensures validation doesn't interfere with the training gradient computation graph
                with torch.no_grad():
                    # Update pipeline with latest weights
                    unwrapped_unet = accelerator.unwrap_model(unet)
                    unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
                    
                    # Unwrap PEFT model to get base UNet for validation pipeline
                    if hasattr(unwrapped_unet, 'base_model') and hasattr(unwrapped_unet.base_model, 'model'):
                        base_unet = unwrapped_unet.base_model.model
                    else:
                        base_unet = unwrapped_unet
                    
                    # CRITICAL: Don't modify the unwrapped models directly - this corrupts the gradient computation graph
                    # Instead, use the models as-is without dtype/device changes
                    # The validation pipeline should handle dtype conversion internally during inference
                    validation_pipeline.unet = base_unet
                    validation_pipeline.text_encoder = unwrapped_text_encoder
                    
                    # Store original training mode
                    unet_was_training = base_unet.training
                    text_encoder_was_training = unwrapped_text_encoder.training
                    
                    # Temporarily set to eval mode for validation (this is safe as long as we restore it)
                    base_unet.eval()
                    unwrapped_text_encoder.eval()
                    
                    try:
                        validation_images = generate_validation_images(
                            validation_pipeline,
                            test_prompts,
                            args.log_dir,
                            global_step,
                            seed=args.seed
                        )
                        
                        # Log validation images to wandb
                        if args.use_wandb and WANDB_AVAILABLE and accelerator.is_main_process:
                            wandb_images = []
                            for prompt, image_path in validation_images:
                                img = Image.open(image_path)
                                wandb_images.append(wandb.Image(img, caption=prompt))
                            wandb.log({
                                "validation/images": wandb_images,
                                "validation/step": global_step,
                            }, step=global_step)
                    finally:
                        # CRITICAL: Restore original training mode immediately after validation
                        # This ensures the models are in the correct state for training
                        if unet_was_training:
                            base_unet.train()
                        if text_encoder_was_training:
                            unwrapped_text_encoder.train()
                # CRITICAL: After validation, we need to ensure the gradient scaler state is reset
                # Validation manipulates models which can corrupt the scaler's internal state
                # The scaler tracks whether gradients have been unscaled, and validation can break this
                
                # CRITICAL: Don't modify the unwrapped models after validation
                # The wrapped models (unet, text_encoder) used for training should remain untouched
                # We'll set them back to train mode below, but we shouldn't modify their dtype/device
                # The unwrapped models are just references, so modifying them can affect training
                
                # CRITICAL FIX: Reset the scaler's per-optimizer state to prevent "Attempting to unscale FP16 gradients" error
                # After validation, the scaler may think gradients are already unscaled when they're not
                print(f"\n[Step {global_step}] Post-validation state (BEFORE cleanup):", flush=True)
                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    initial_state_count = len(accelerator.scaler._per_optimizer_states)
                    print(f"  Scaler state: {initial_state_count} optimizer(s) tracked", flush=True)
                    if initial_state_count > 0:
                        for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                            stage = opt_state.get('stage', 'unknown')
                            found_inf_per_device = opt_state.get('found_inf_per_device', {})
                            print(f"    Optimizer stage: {stage}", flush=True)
                            if found_inf_per_device:
                                print(f"    Found inf per device: {found_inf_per_device}", flush=True)
                else:
                    print(f"  ⚠ No scaler found", flush=True)
                
                # Log model states after validation
                unwrapped_unet_after = accelerator.unwrap_model(unet)
                unwrapped_text_encoder_after = accelerator.unwrap_model(text_encoder)
                if hasattr(unwrapped_unet_after, 'base_model') and hasattr(unwrapped_unet_after.base_model, 'model'):
                    base_unet_after = unwrapped_unet_after.base_model.model
                else:
                    base_unet_after = unwrapped_unet_after
                print(f"  UNet training mode: {base_unet_after.training}", flush=True)
                print(f"  Text encoder training mode: {unwrapped_text_encoder_after.training}", flush=True)
                print(f"  Base UNet dtype: {next(base_unet_after.parameters()).dtype}", flush=True)
                print(f"  Text encoder dtype: {next(unwrapped_text_encoder_after.parameters()).dtype}", flush=True)
                
                print(f"\n[Step {global_step}] Post-validation cleanup:", flush=True)
                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    initial_state_count = len(accelerator.scaler._per_optimizer_states)
                    print(f"  Scaler state before reset: {initial_state_count} optimizer(s) tracked", flush=True)
                    
                    # Clear the scaler's internal state for this optimizer
                    # The scaler uses the optimizer object itself as a key, not its ID
                    # Try both the wrapped optimizer and the underlying optimizer
                    optimizer_found = False
                    if optimizer in accelerator.scaler._per_optimizer_states:
                        del accelerator.scaler._per_optimizer_states[optimizer]
                        optimizer_found = True
                        print(f"  ✓ Removed scaler state for wrapped optimizer", flush=True)
                    elif hasattr(optimizer, 'optimizer') and optimizer.optimizer in accelerator.scaler._per_optimizer_states:
                        del accelerator.scaler._per_optimizer_states[optimizer.optimizer]
                        optimizer_found = True
                        print(f"  ✓ Removed scaler state for underlying optimizer", flush=True)
                    
                    # As a fallback, clear all optimizer states (should only be one anyway)
                    remaining_states = len(accelerator.scaler._per_optimizer_states)
                    if remaining_states > 0:
                        print(f"  ⚠ Warning: {remaining_states} optimizer state(s) still remain, clearing all", flush=True)
                        accelerator.scaler._per_optimizer_states.clear()
                        print(f"  ✓ Cleared all scaler optimizer states", flush=True)
                    elif optimizer_found:
                        print(f"  ✓ Scaler state successfully reset", flush=True)
                    else:
                        print(f"  ℹ No scaler state found to reset (this is normal if no gradients were computed yet)", flush=True)
                else:
                    print(f"  ⚠ Warning: No scaler found in accelerator (mixed precision may not be enabled)", flush=True)
                
                # Clear any gradients that might have been left in an inconsistent state
                optimizer.zero_grad(set_to_none=True)
                print(f"  ✓ Cleared optimizer gradients", flush=True)
                
                # CRITICAL: After validation, ensure the scaler is in a clean state
                # The scaler may have internal state beyond just the per-optimizer states dictionary
                # Force the scaler to reset by ensuring it doesn't think gradients are already unscaled
                # We do this by ensuring the next backward() call will create fresh state
                
                # Clear GPU cache
                torch.cuda.empty_cache()
                print(f"  ✓ Cleared GPU cache", flush=True)
                
                # IMPORTANT: After validation, we need to ensure the next training step starts fresh
                # The issue is that validation may have corrupted the gradient computation graph
                # We'll add a dummy backward/step cycle to reset the scaler state properly
                # But actually, that might cause issues. Instead, let's ensure models are back in training state
                
                # Models should already be in train mode (restored in the try/finally block above)
                # But ensure the wrapped models are explicitly in train mode as a safeguard
                unet.train()
                text_encoder.train()
                
                print(f"  ✓ Ensured models are in train mode", flush=True)
                
                # DEBUG: Log post-cleanup state
                print(f"\n[Step {global_step}] Post-validation cleanup state (AFTER cleanup):", flush=True)
                if hasattr(accelerator, 'scaler') and accelerator.scaler is not None:
                    final_state_count = len(accelerator.scaler._per_optimizer_states)
                    print(f"  Scaler state: {final_state_count} optimizer(s) tracked", flush=True)
                    if final_state_count > 0:
                        for opt_key, opt_state in accelerator.scaler._per_optimizer_states.items():
                            stage = opt_state.get('stage', 'unknown')
                            print(f"    Optimizer stage: {stage}", flush=True)
                else:
                    print(f"  ⚠ No scaler found", flush=True)
                
                # Verify model states
                unwrapped_unet_final = accelerator.unwrap_model(unet)
                unwrapped_text_encoder_final = accelerator.unwrap_model(text_encoder)
                if hasattr(unwrapped_unet_final, 'base_model') and hasattr(unwrapped_unet_final.base_model, 'model'):
                    base_unet_final = unwrapped_unet_final.base_model.model
                else:
                    base_unet_final = unwrapped_unet_final
                print(f"  UNet training mode: {base_unet_final.training}", flush=True)
                print(f"  Text encoder training mode: {unwrapped_text_encoder_final.training}", flush=True)
                print(f"  Base UNet dtype: {next(base_unet_final.parameters()).dtype}", flush=True)
                print(f"  Text encoder dtype: {next(unwrapped_text_encoder_final.parameters()).dtype}", flush=True)
                
                print(f"  Post-validation cleanup complete", flush=True)
                print(f"[Step {global_step}] ========== VALIDATION END ==========\n", flush=True)
            
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
        
        # Validation and early stopping check at end of epoch
        if accelerator.is_main_process:
            # Calculate epoch average loss
            epoch_avg_loss = np.mean(loss_history[-len(dataloader):])
            
            print(f"\n[End of Epoch {epoch+1}]")
            print(f"  Average loss: {epoch_avg_loss:.4f}")
            print(f"  EMA loss: {ema_loss:.4f}")
            
            # Log epoch metrics to wandb
            if args.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch/avg_loss": epoch_avg_loss,
                    "epoch/ema_loss": ema_loss,
                    "epoch/epoch": epoch + 1,
                }, step=global_step)
            
            # Early stopping check
            if epoch_avg_loss < best_loss:
                best_loss = epoch_avg_loss
                no_improve_count = 0
                print(f"  ✓ New best loss: {best_loss:.4f}")
                
                # Log best loss to wandb
                if args.use_wandb and WANDB_AVAILABLE:
                    wandb.log({"best_loss": best_loss}, step=global_step)
                # Save best model checkpoint
                save_checkpoint(
                    accelerator.unwrap_model(unet),
                    accelerator.unwrap_model(text_encoder),
                    tokenizer,
                    EMOTION_TOKENS,
                    args.output_dir,
                    epoch,
                    global_step,
                    is_best=True
                )
            else:
                no_improve_count += 1
                print(f"  ⚠ No improvement for {no_improve_count}/{args.early_stopping_patience} epochs")
                
                if no_improve_count >= args.early_stopping_patience:
                    print("\n🛑 Early stopping triggered!")
                    break
            
            # Generate validation images
            print(f"  Generating validation images...")
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_text_encoder = accelerator.unwrap_model(text_encoder)
            
            # Unwrap PEFT model to get base UNet for validation pipeline
            if hasattr(unwrapped_unet, 'base_model') and hasattr(unwrapped_unet.base_model, 'model'):
                base_unet = unwrapped_unet.base_model.model
            else:
                base_unet = unwrapped_unet
            
            # Ensure consistent dtype for validation (match pipeline dtype)
            pipeline_dtype = next(validation_pipeline.unet.parameters()).dtype
            base_unet = base_unet.to(device).to(pipeline_dtype)
            unwrapped_text_encoder = unwrapped_text_encoder.to(device).to(pipeline_dtype)
            
            # Ensure input embeddings are also in the correct dtype
            with torch.no_grad():
                embedding_weight = unwrapped_text_encoder.get_input_embeddings().weight
                if embedding_weight.dtype != pipeline_dtype:
                    unwrapped_text_encoder.get_input_embeddings().weight.data = embedding_weight.to(pipeline_dtype)
            
            validation_pipeline.unet = base_unet
            validation_pipeline.text_encoder = unwrapped_text_encoder
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
        
        # Save checkpoint at end of epoch
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
    
    # Save loss history
    if accelerator.is_main_process:
        loss_file = os.path.join(args.log_dir, "loss_history.json")
        os.makedirs(args.log_dir, exist_ok=True)
        with open(loss_file, 'w') as f:
            json.dump({
                'loss_history': loss_history,
                'best_loss': best_loss,
                'final_epoch': epoch + 1,
                'total_steps': global_step
            }, f, indent=2)
        print(f"\n✓ Saved loss history to {loss_file}")
    
    print("\n" + "="*50)
    print("Training completed!")
    print("="*50)
    print(f"Final checkpoint saved to {args.output_dir}")
    if accelerator.is_main_process:
        print(f"Best loss: {best_loss:.4f}")
        
        # Final wandb logging
        if args.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "final/best_loss": best_loss,
                "final/total_steps": global_step,
                "final/total_epochs": epoch + 1,
            })
            wandb.finish()
            print(f"✓ Wandb run completed")


if __name__ == "__main__":
    main()
