"""
Inference script with Classifier Guidance for emotion-conditioned image generation.

Uses a trained Noise-Aware Latent Classifier to guide the diffusion process
toward the target emotion by adding gradients at each denoising step.
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from PIL import Image
from tqdm import tqdm

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add paths for imports
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "approaches" / "classifier_guidance" / "src"))

from model import EmotionLatentClassifier

# Set HuggingFace cache directory and storage base
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


def setup_logging(log_dir=None, log_level=logging.INFO, verbose=False):
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory to save log files (optional)
        log_level: Logging level (default: INFO)
        verbose: If True, set level to DEBUG
        
    Returns:
        Logger instance
    """
    if verbose:
        log_level = logging.DEBUG
    
    # Create logger
    logger = logging.getLogger('classifier_guidance_inference')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (if log_dir provided)
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'inference_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
    
    return logger


def setup_device(logger=None):
    """Detect and return the appropriate device."""
    log = logger if logger else logging.getLogger('classifier_guidance_inference')
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        log.info(f"Using CUDA device: {device_name}")
        log.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using MPS device")
    else:
        device = torch.device("cpu")
        log.info("Using CPU device")
    return device


def load_pipeline(model_id: str = "runwayml/stable-diffusion-v1-5", device=None, logger=None):
    """
    Load Stable Diffusion pipeline.
    
    Args:
        model_id: HuggingFace model ID
        device: Device to load model on
        logger: Logger instance (optional)
        
    Returns:
        Stable Diffusion pipeline
    """
    log = logger if logger else logging.getLogger('classifier_guidance_inference')
    log.info(f"Loading Stable Diffusion pipeline from {model_id}...")
    
    import time
    start_time = time.time()
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)  # We'll use our own progress bar
    
    load_time = time.time() - start_time
    log.info(f"Pipeline loaded in {load_time:.2f} seconds")
    
    if device.type == "cuda":
        memory_used = torch.cuda.memory_allocated() / 1024**3
        log.debug(f"GPU memory used: {memory_used:.2f} GB")
    
    return pipe


def load_classifier(checkpoint_path: str, device=None, logger=None):
    """
    Load trained classifier.
    
    Args:
        checkpoint_path: Path to classifier checkpoint
        device: Device to load model on
        logger: Logger instance (optional)
        
    Returns:
        Trained classifier model
    """
    log = logger if logger else logging.getLogger('classifier_guidance_inference')
    log.info(f"Loading classifier from {checkpoint_path}...")
    
    import time
    start_time = time.time()
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")  # Load to CPU first
    
    classifier = EmotionLatentClassifier(num_emotions=8)
    
    # Load state dict and convert all tensors to float32 immediately
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if 'epoch' in checkpoint:
            log.info(f"Checkpoint from epoch {checkpoint['epoch']}")
        if 'loss' in checkpoint:
            log.info(f"Checkpoint loss: {checkpoint['loss']:.4f}")
    else:
        state_dict = checkpoint
    
    # Convert all parameters in state dict to float32 before loading
    state_dict_fp32 = {}
    converted_count = 0
    for key, value in state_dict.items():
        if isinstance(value, torch.Tensor) and value.dtype != torch.int64:
            if value.dtype != torch.float32:
                log.debug(f"Converting {key} from {value.dtype} to float32")
                converted_count += 1
            state_dict_fp32[key] = value.to(dtype=torch.float32)
        else:
            state_dict_fp32[key] = value
    
    if converted_count > 0:
        log.info(f"Converted {converted_count} tensors to float32 in state dict")
    
    classifier.load_state_dict(state_dict_fp32)
    
    # Move to device
    classifier = classifier.to(device=device)
    
    # CRITICAL: Convert to float32 and ensure it STAYS float32
    # Use .float() which converts all parameters and buffers
    classifier = classifier.float()
    
    # Double-check: explicitly convert every parameter and buffer
    for param in classifier.parameters():
        if param.dtype != torch.float32:
            param.data = param.data.to(dtype=torch.float32)
    for buffer in classifier.buffers():
        if buffer.dtype != torch.int64 and buffer.dtype != torch.float32:
            buffer.data = buffer.data.to(dtype=torch.float32)
    
    classifier.eval()
    classifier.requires_grad_(False)
    
    # CRITICAL: Register hook to enforce float32 before EVERY forward pass
    def enforce_float32(module, input):
        # Convert all parameters to float32 right before forward
        for param in module.parameters():
            if param.dtype != torch.float32:
                param.data = param.data.to(dtype=torch.float32)
        for buffer in module.buffers():
            if buffer.dtype != torch.int64 and buffer.dtype != torch.float32:
                buffer.data = buffer.data.to(dtype=torch.float32)
    
    # Register on all modules
    for m in classifier.modules():
        m.register_forward_pre_hook(enforce_float32)
    
    log.info("Classifier loaded (dtype: torch.float32)")
    
    load_time = time.time() - start_time
    log.info(f"Classifier loaded in {load_time:.2f} seconds (dtype: float32)")
    
    # Count parameters
    num_params = sum(p.numel() for p in classifier.parameters())
    log.debug(f"Classifier parameters: {num_params:,}")
    
    return classifier


def generate_with_classifier_guidance(
    pipe,
    classifier,
    prompt: str,
    target_emotion_idx: int,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    classifier_scale: float = 20.0,
    seed: int = 42,
    device=None,
    logger=None,
    track_metrics: bool = True,
    use_wandb: bool = False,
) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Generate image with classifier-based guidance.
    
    Args:
        pipe: Stable Diffusion pipeline
        classifier: Trained emotion classifier
        prompt: Text prompt
        target_emotion_idx: Target emotion index (0-7)
        num_inference_steps: Number of diffusion steps
        guidance_scale: Classifier-free guidance scale
        classifier_scale: Classifier guidance strength
        seed: Random seed
        device: Device to run on
        logger: Logger instance (optional)
        track_metrics: If True, track and return metrics during generation
        use_wandb: If True, log metrics to wandb during generation
        
    Returns:
        Tuple of (Generated PIL Image, metrics dictionary)
    """
    log = logger if logger else logging.getLogger('classifier_guidance_inference')
    
    if device is None:
        device = pipe.device
    
    # Initialize metrics tracking
    metrics = {
        'step_metrics': [],
        'final_confidence': None,
        'final_emotion_probs': None,
        'generation_time': None,
    }
    
    import time
    generation_start = time.time()
    
    # Set seed
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
        log.debug(f"Random seed set to {seed}")
    
    # Prepare prompt for CFG
    # CFG requires both conditional and unconditional prompts
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_inputs = text_inputs.to(device)
    
    # Get text embeddings
    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_inputs.input_ids)[0]
    
    # For CFG, we need unconditional embeddings (empty prompt)
    uncond_inputs = pipe.tokenizer(
        [""],
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    uncond_inputs = uncond_inputs.to(device)
    
    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_inputs.input_ids)[0]
    
    # Concatenate for CFG: [uncond, cond]
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    
    # Prepare latents
    # Use float32 for latents to match classifier dtype (gradient computation is more stable)
    latents_shape = (1, 4, 64, 64)  # SD v1.5 latent shape
    latents = torch.randn(
        latents_shape,
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    
    # Scale latents
    latents = latents * pipe.scheduler.init_noise_sigma
    
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = pipe.scheduler.timesteps
    
    # Denoising loop with classifier guidance
    log.info(f"Starting generation with classifier guidance (scale={classifier_scale})...")
    log.info(f"Total steps: {num_inference_steps}, Target emotion: {EMOTIONS[target_emotion_idx]}")
    
    progress_bar = tqdm(timesteps, desc="Denoising", unit="step")
    for i, t in enumerate(progress_bar):
        # Expand latents for CFG: [1, 4, 64, 64] -> [2, 4, 64, 64]
        # UNet expects float16, so convert latents for UNet inference
        latents_for_unet = latents.to(dtype=text_embeddings.dtype)
        latent_model_input = torch.cat([latents_for_unet] * 2)
        latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)
        
        # Predict noise with UNet (no grad needed for UNet)
        with torch.no_grad():
            noise_pred = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample
        
        # Perform CFG
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        # CRITICAL: Enable gradients for classifier guidance
        # We need to compute gradients w.r.t. the latents
        # Create a copy that requires grad for gradient computation
        # Convert to float32 to match classifier dtype (pipeline uses float16)
        latents_for_classifier = latents.detach().to(dtype=torch.float32).requires_grad_(True)
        
        # Classifier guidance step
        # Get classifier prediction on the current noisy latents
        # Note: timestep t is a tensor, we need to expand it to match batch size
        # Ensure timestep is in float32 to match classifier
        t_expanded = t.expand(1).to(dtype=torch.float32)  # [1]
        
        # Forward pass through classifier (with gradients enabled)
        # CRITICAL: Force all parameters to float32 right before forward pass
        # This ensures nothing has been converted to float16
        with torch.no_grad():
            for name, param in classifier.named_parameters():
                if param.dtype != torch.float32:
                    param.data = param.data.to(dtype=torch.float32)
            for name, buffer in classifier.named_buffers():
                if buffer.dtype != torch.int64 and buffer.dtype != torch.float32:
                    buffer.data = buffer.data.to(dtype=torch.float32)
        
        try:
            # Log input dtypes for debugging
            if log.isEnabledFor(logging.DEBUG) and i == 0:
                log.debug(f"Classifier input - latents: {latents_for_classifier.dtype}, timesteps: {t_expanded.dtype}")
                # Check first layer dtype
                first_conv = classifier.conv1
                log.debug(f"First conv weight dtype: {first_conv.weight.dtype}, bias dtype: {first_conv.bias.dtype if first_conv.bias is not None else 'None'}")
            
            # CRITICAL: Ensure classifier is in float32 mode before calling
            # Convert all parameters one more time right before forward
            with torch.no_grad():
                for param in classifier.parameters():
                    if param.dtype != torch.float32:
                        param.data = param.data.to(dtype=torch.float32)
            
            logits = classifier(latents_for_classifier, t_expanded)  # [1, 8]
        except RuntimeError as e:
            if "dtype" in str(e).lower() or "Half" in str(e) or "Float" in str(e):
                # Debug: check classifier dtype
                log.error(f"Dtype mismatch error at step {i+1}: {e}")
                log.error(f"Latents dtype: {latents_for_classifier.dtype}, shape: {latents_for_classifier.shape}")
                log.error(f"Timesteps dtype: {t_expanded.dtype}, shape: {t_expanded.shape}")
                
                # Check all layer dtypes
                log.error("Checking classifier layer dtypes:")
                for name, module in classifier.named_modules():
                    if hasattr(module, 'weight') and module.weight is not None:
                        log.error(f"  {name}.weight: {module.weight.dtype}")
                    if hasattr(module, 'bias') and module.bias is not None:
                        log.error(f"  {name}.bias: {module.bias.dtype}")
                    if isinstance(module, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
                        if hasattr(module, 'running_mean') and module.running_mean is not None:
                            log.error(f"  {name}.running_mean: {module.running_mean.dtype}")
                        if hasattr(module, 'running_var') and module.running_var is not None:
                            log.error(f"  {name}.running_var: {module.running_var.dtype}")
                
                raise
            else:
                raise
        
        # Get probabilities and log probabilities
        probs = F.softmax(logits, dim=1)  # [1, 8]
        log_probs = F.log_softmax(logits, dim=1)  # [1, 8]
        
        # Target loss: maximize probability of target emotion
        # Negative because we want to maximize (gradient ascent)
        target_loss = -log_probs[:, target_emotion_idx].sum()
        
        # Track metrics at this step
        if track_metrics:
            step_metric = {
                'step': i + 1,
                'timestep': t.item() if isinstance(t, torch.Tensor) else t,
                'target_emotion': EMOTIONS[target_emotion_idx],
                'target_confidence': probs[0, target_emotion_idx].item(),
                'target_log_prob': log_probs[0, target_emotion_idx].item(),
                'target_loss': target_loss.item(),
                'predicted_emotion': EMOTIONS[probs.argmax(dim=1).item()],
                'emotion_probs': {EMOTIONS[j]: probs[0, j].item() for j in range(8)},
            }
            metrics['step_metrics'].append(step_metric)
            
            # Update progress bar with current confidence
            progress_bar.set_postfix({
                'conf': f"{step_metric['target_confidence']:.3f}",
                'pred': step_metric['predicted_emotion'][:4]
            })
            
            # Log to wandb if enabled
            if use_wandb and WANDB_AVAILABLE:
                log_dict = {
                    "generation/step": i + 1,
                    "generation/timestep": step_metric['timestep'],
                    "generation/target_confidence": step_metric['target_confidence'],
                    "generation/target_log_prob": step_metric['target_log_prob'],
                    "generation/target_loss": step_metric['target_loss'],
                    "generation/predicted_emotion_idx": EMOTIONS.index(step_metric['predicted_emotion']),
                }
                
                # Log individual emotion probabilities
                emotion_probs = step_metric['emotion_probs']
                for emotion, prob in emotion_probs.items():
                    log_dict[f"generation/prob_{emotion}"] = prob
                
                # Log GPU memory if available
                if torch.cuda.is_available():
                    log_dict["system/gpu_memory_used_mb"] = torch.cuda.memory_allocated() / 1024**2
                    log_dict["system/gpu_memory_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
                
                wandb.log(log_dict, step=i + 1)
        
        # Compute gradient w.r.t. latents
        grad = torch.autograd.grad(
            target_loss,
            latents_for_classifier,
            create_graph=False,
            retain_graph=False,
        )[0]
        
        # Track gradient magnitude for debugging
        if track_metrics:
            grad_norm = grad.norm().item()
            if log.isEnabledFor(logging.DEBUG):
                log.debug(f"Step {i+1}/{num_inference_steps}: grad_norm={grad_norm:.6f}, "
                         f"target_conf={probs[0, target_emotion_idx].item():.4f}")
            
            # Log gradient magnitude to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "generation/grad_norm": grad_norm,
                }, step=i + 1)
        
        # Update noise prediction with classifier guidance
        # Subtract gradient to move toward target emotion
        # The gradient points in the direction to increase the target emotion probability
        # Convert grad to match noise_pred dtype (float16)
        grad = grad.to(dtype=noise_pred.dtype)
        noise_pred = noise_pred - classifier_scale * grad
        
        # Step scheduler (no grad needed)
        # Note: scheduler expects latents in the same dtype as noise_pred (float16)
        # So we need to convert latents to float16 for the scheduler step
        with torch.no_grad():
            latents_for_scheduler = latents.to(dtype=noise_pred.dtype)
            latents = pipe.scheduler.step(noise_pred, t, latents_for_scheduler, return_dict=False)[0]
            # Convert back to float32 for classifier in next iteration
            latents = latents.to(dtype=torch.float32)
    
    # Final classifier prediction on clean latents
    if track_metrics:
        with torch.no_grad():
            final_logits = classifier(latents, torch.tensor([0], device=device))
            final_probs = F.softmax(final_logits, dim=1)
            metrics['final_confidence'] = final_probs[0, target_emotion_idx].item()
            metrics['final_emotion_probs'] = {
                EMOTIONS[j]: final_probs[0, j].item() for j in range(8)
            }
            metrics['final_predicted_emotion'] = EMOTIONS[final_probs.argmax(dim=1).item()]
            
            log.info(f"Final prediction: {metrics['final_predicted_emotion']} "
                    f"(confidence: {metrics['final_confidence']:.4f})")
    
    # Decode latents to image
    # Scale back using VAE scaling factor (more robust than hardcoded value)
    latents = 1 / pipe.vae.config.scaling_factor * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    
    # Post-process image
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    image = Image.fromarray(image[0])
    
    # Record generation time
    metrics['generation_time'] = time.time() - generation_start
    log.info(f"Generation completed in {metrics['generation_time']:.2f} seconds")
    
    # Log final metrics to wandb
    if use_wandb and WANDB_AVAILABLE and track_metrics:
        final_log_dict = {
            "final/target_emotion": EMOTIONS[target_emotion_idx],
            "final/predicted_emotion": metrics.get('final_predicted_emotion', 'N/A'),
            "final/target_confidence": metrics['final_confidence'],
            "final/generation_time": metrics['generation_time'],
            "final/num_steps": num_inference_steps,
        }
        
        # Log final emotion probabilities
        if metrics['final_emotion_probs']:
            final_probs = metrics['final_emotion_probs']
            for emotion, prob in final_probs.items():
                final_log_dict[f"final/prob_{emotion}"] = prob
        
        wandb.log(final_log_dict)
    
    return image, metrics


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with classifier-based guidance"
    )
    
    # Generation arguments
    parser.add_argument("--prompt", type=str, required=True,
                       help="Text prompt for generation")
    parser.add_argument("--emotion_idx", type=int, required=True,
                       help="Target emotion index (0-7): 0=amusement, 1=anger, 2=awe, 3=contentment, 4=disgust, 5=excitement, 6=fear, 7=sadness")
    parser.add_argument("--classifier_path", type=str, default=None,
                       help="Path to trained classifier checkpoint (default: /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/classifier_guidance/classifier_large.pt)")
    parser.add_argument("--output_path", type=str, default="output.png",
                       help="Output image path (default: output.png)")
    
    # Generation parameters
    parser.add_argument("--num_inference_steps", type=int, default=50,
                       help="Number of diffusion steps (default: 50)")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                       help="Classifier-free guidance scale (default: 7.5)")
    parser.add_argument("--classifier_scale", type=float, default=20.0,
                       help="Classifer guidance strength (default: 20.0)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    
    # Model arguments
    parser.add_argument("--model_id", type=str, default="runwayml/stable-diffusion-v1-5",
                       help="HuggingFace model ID (default: runwayml/stable-diffusion-v1-5)")
    
    # Logging arguments
    parser.add_argument("--log_dir", type=str, default=None,
                       help="Directory to save log files (optional)")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose (DEBUG) logging")
    parser.add_argument("--save_metrics", action="store_true",
                       help="Save generation metrics to JSON file")
    parser.add_argument("--no_track_metrics", action="store_true",
                       help="Disable metrics tracking during generation (faster)")
    
    # Wandb arguments
    parser.add_argument("--use_wandb", action="store_true", default=False,
                       help="Enable wandb logging (default: False)")
    parser.add_argument("--wandb_project", type=str, default="emotional-portraits",
                       help="Wandb project name (default: emotional-portraits)")
    parser.add_argument("--wandb_entity", type=str, default=None,
                       help="Wandb entity/team name (optional)")
    parser.add_argument("--wandb_name", type=str, default=None,
                       help="Wandb run name (optional, auto-generated if not provided)")
    
    args = parser.parse_args()
    
    # Validate emotion index
    if args.emotion_idx < 0 or args.emotion_idx >= len(EMOTIONS):
        raise ValueError(f"Emotion index must be 0-7, got {args.emotion_idx}")
    
    # Set default classifier path if not provided
    if args.classifier_path is None:
        args.classifier_path = str(Path(STORAGE_BASE) / "Weights" / "classifier_guidance" / "classifier_large.pt")
    
    # Set default log directory if save_metrics is enabled
    if args.save_metrics and args.log_dir is None:
        args.log_dir = str(Path(STORAGE_BASE) / "Logs" / "classifier_guidance")
    
    # Setup logging
    logger = setup_logging(
        log_dir=args.log_dir,
        log_level=logging.DEBUG if args.verbose else logging.INFO,
        verbose=args.verbose
    )
    
    # Setup device
    device = setup_device(logger=logger)
    
    # Load pipeline
    pipe = load_pipeline(args.model_id, device=device, logger=logger)
    
    # Load classifier
    classifier = load_classifier(args.classifier_path, device=device, logger=logger)
    
    # Get emotion name
    emotion_name = EMOTIONS[args.emotion_idx]
    
    # Initialize wandb if requested
    if args.use_wandb and WANDB_AVAILABLE:
        # Generate run name if not provided
        if args.wandb_name is None:
            prompt_short = args.prompt[:30].replace(' ', '_').replace('/', '_')
            args.wandb_name = f"classifier_guidance_{emotion_name}_{prompt_short}_s{args.seed}"
        
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            config={
                "approach": "classifier_guidance",
                "prompt": args.prompt,
                "target_emotion": emotion_name,
                "target_emotion_idx": args.emotion_idx,
                "num_inference_steps": args.num_inference_steps,
                "guidance_scale": args.guidance_scale,
                "classifier_scale": args.classifier_scale,
                "seed": args.seed,
                "model_id": args.model_id,
                "classifier_path": args.classifier_path,
                "track_metrics": not args.no_track_metrics,
            },
            tags=["classifier_guidance", "inference", emotion_name],
        )
        logger.info(f"✓ Initialized wandb: {wandb.run.url}")
    elif args.use_wandb and not WANDB_AVAILABLE:
        logger.warning("⚠ wandb requested but not available. Install with: pip install wandb")
    
    logger.info("="*70)
    logger.info("Classifier-Guided Image Generation")
    logger.info("="*70)
    logger.info(f"Prompt: {args.prompt}")
    logger.info(f"Target Emotion: {emotion_name} (index {args.emotion_idx})")
    logger.info(f"CFG Scale: {args.guidance_scale}")
    logger.info(f"Classifier Scale: {args.classifier_scale}")
    logger.info(f"Inference Steps: {args.num_inference_steps}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Track Metrics: {not args.no_track_metrics}")
    logger.info(f"Wandb Logging: {args.use_wandb and WANDB_AVAILABLE}")
    logger.info("="*70)
    
    # Generate image
    image, metrics = generate_with_classifier_guidance(
        pipe=pipe,
        classifier=classifier,
        prompt=args.prompt,
        target_emotion_idx=args.emotion_idx,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        classifier_scale=args.classifier_scale,
        seed=args.seed,
        device=device,
        logger=logger,
        track_metrics=not args.no_track_metrics,
        use_wandb=args.use_wandb and WANDB_AVAILABLE,
    )
    
    # Save image
    image.save(args.output_path)
    logger.info(f"✓ Image saved to: {args.output_path}")
    
    # Log generated image to wandb
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.log({
            "generated_image": wandb.Image(
                image,
                caption=f"Prompt: {args.prompt}\nTarget: {emotion_name}\nPredicted: {metrics.get('final_predicted_emotion', 'N/A')}"
            )
        })
    
    # Save metrics if requested
    if args.save_metrics and not args.no_track_metrics:
        metrics_file = Path(args.output_path).with_suffix('.metrics.json')
        # Prepare metrics for JSON serialization
        metrics_json = {
            'prompt': args.prompt,
            'target_emotion': emotion_name,
            'target_emotion_idx': args.emotion_idx,
            'parameters': {
                'num_inference_steps': args.num_inference_steps,
                'guidance_scale': args.guidance_scale,
                'classifier_scale': args.classifier_scale,
                'seed': args.seed,
            },
            'final_results': {
                'final_confidence': metrics['final_confidence'],
                'final_predicted_emotion': metrics.get('final_predicted_emotion'),
                'final_emotion_probs': metrics['final_emotion_probs'],
            },
            'generation_time': metrics['generation_time'],
            'step_metrics': metrics['step_metrics'],
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        
        logger.info(f"✓ Metrics saved to: {metrics_file}")
        
        # Print summary
        logger.info("\n" + "="*70)
        logger.info("Generation Summary")
        logger.info("="*70)
        logger.info(f"Target Emotion: {emotion_name}")
        logger.info(f"Final Prediction: {metrics.get('final_predicted_emotion', 'N/A')}")
        logger.info(f"Target Confidence: {metrics['final_confidence']:.4f}")
        logger.info(f"Generation Time: {metrics['generation_time']:.2f}s")
        logger.info("="*70)
    
    # Finish wandb run
    if args.use_wandb and WANDB_AVAILABLE:
        wandb.finish()
        logger.info("✓ Wandb run completed")


# if __name__ == "__main__":
#     main()


#             "approach": "classifier_guidance",
#             "prompt": args.prompt,
#             "target_emotion": emotion_name,
#             "target_emotion_idx": args.emotion_idx,
#             "num_inference_steps": args.num_inference_steps,
#             "guidance_scale": args.guidance_scale,
#             "classifier_scale": args.classifier_scale,
#             "seed": args.seed,
#             "model_id": args.model_id,
#             "classifier_path": args.classifier_path,
#             "track_metrics": not args.no_track_metrics,
#         },
#         tags=["classifier_guidance", "inference", emotion_name],
#     )
#     logger.info(f"✓ Initialized wandb: {wandb.run.url}")
#     elif args.use_wandb and not WANDB_AVAILABLE:
#         logger.warning("⚠ wandb requested but not available. Install with: pip install wandb")
    
#     logger.info("="*70)
#     logger.info("Classifier-Guided Image Generation")
#     logger.info("="*70)
#     logger.info(f"Prompt: {args.prompt}")
#     logger.info(f"Target Emotion: {emotion_name} (index {args.emotion_idx})")
#     logger.info(f"CFG Scale: {args.guidance_scale}")
#     logger.info(f"Classifier Scale: {args.classifier_scale}")
#     logger.info(f"Inference Steps: {args.num_inference_steps}")
#     logger.info(f"Seed: {args.seed}")
#     logger.info(f"Track Metrics: {not args.no_track_metrics}")
#     logger.info(f"Wandb Logging: {args.use_wandb and WANDB_AVAILABLE}")
#     logger.info("="*70)
    
#     # Generate image
#     image, metrics = generate_with_classifier_guidance(
#         pipe=pipe,
#         classifier=classifier,
#         prompt=args.prompt,
#         target_emotion_idx=args.emotion_idx,
#         num_inference_steps=args.num_inference_steps,
#         guidance_scale=args.guidance_scale,
#         classifier_scale=args.classifier_scale,
#         seed=args.seed,
#         device=device,
#         logger=logger,
#         track_metrics=not args.no_track_metrics,
#         use_wandb=args.use_wandb and WANDB_AVAILABLE,
#     )
    
#     # Save image
#     image.save(args.output_path)
#     logger.info(f"✓ Image saved to: {args.output_path}")
    
#     # Log generated image to wandb
#     if args.use_wandb and WANDB_AVAILABLE:
#         wandb.log({
#             "generated_image": wandb.Image(
#                 image,
#                 caption=f"Prompt: {args.prompt}\nTarget: {emotion_name}\nPredicted: {metrics.get('final_predicted_emotion', 'N/A')}"
#             )
#         })
    
#     # Save metrics if requested
#     if args.save_metrics and not args.no_track_metrics:
#         metrics_file = Path(args.output_path).with_suffix('.metrics.json')
#         # Prepare metrics for JSON serialization
#         metrics_json = {
#             'prompt': args.prompt,
#             'target_emotion': emotion_name,
#             'target_emotion_idx': args.emotion_idx,
#             'parameters': {
#                 'num_inference_steps': args.num_inference_steps,
#                 'guidance_scale': args.guidance_scale,
#                 'classifier_scale': args.classifier_scale,
#                 'seed': args.seed,
#             },
#             'final_results': {
#                 'final_confidence': metrics['final_confidence'],
#                 'final_predicted_emotion': metrics.get('final_predicted_emotion'),
#                 'final_emotion_probs': metrics['final_emotion_probs'],
#             },
#             'generation_time': metrics['generation_time'],
#             'step_metrics': metrics['step_metrics'],
#         }
        
#         with open(metrics_file, 'w') as f:
#             json.dump(metrics_json, f, indent=2)
        
#         logger.info(f"✓ Metrics saved to: {metrics_file}")
        
#         # Print summary
#         logger.info("\n" + "="*70)
#         logger.info("Generation Summary")
#         logger.info("="*70)
#         logger.info(f"Target Emotion: {emotion_name}")
#         logger.info(f"Final Prediction: {metrics.get('final_predicted_emotion', 'N/A')}")
#         logger.info(f"Target Confidence: {metrics['final_confidence']:.4f}")
#         logger.info(f"Generation Time: {metrics['generation_time']:.2f}s")
#         logger.info("="*70)
    
#     # Finish wandb run
#     if args.use_wandb and WANDB_AVAILABLE:
#         wandb.finish()
#         logger.info("✓ Wandb run completed")


if __name__ == "__main__":
    main()

