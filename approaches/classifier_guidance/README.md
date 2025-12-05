# Classifier Guidance Approach

This approach uses a **Noise-Aware Latent Classifier** to guide Stable Diffusion v1.5 image generation, rather than fine-tuning the UNet or using LoRA.

## Overview

Instead of training the diffusion model itself, we train a lightweight CNN classifier that predicts emotions from noisy latents at different timesteps. During inference, we use the classifier's gradients to guide the denoising process toward the target emotion.

### Key Features

- **No UNet/LoRA Training**: Only trains a small classifier (~2M parameters)
- **Gradient-Based Guidance**: Uses classifier gradients to steer generation
- **Noise-Aware**: Classifier is trained on noisy latents at various timesteps
- **On-Disk Latent Caching**: Pre-encodes images to latents for faster training

## Architecture

### EmotionLatentClassifier

A lightweight CNN that takes:
- **Input**: Noisy latents `[Batch, 4, 64, 64]` and timesteps `[Batch]`
- **Output**: Emotion logits `[Batch, 8]`

**Architecture**:
- 4 Conv blocks (Conv2d → BatchNorm → SiLU)
- Time embedding injection after Block 4
- Global Average Pooling → Linear classifier

## Usage

### 1. Train the Classifier

```bash
cd /users/eleves-a/2025/yash.bhardwaj/EmotionalPortraitsGeneration

python approaches/classifier_guidance/src/train_classifier.py \
    --data_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_full \
    --batch_size 64 \
    --num_epochs 10 \
    --lr 1e-3 \
    --seed 42
```

**Arguments**:
- `--data_dir`: Path to dataset directory (can use `emoset_full` or `emoset_captioned_25k`)
- `--cache_dir`: Directory for caching VAE latents (default: `/Data/yash.bhardwaj/.../Datasets/latents_cache_classifier_guidance`)
- `--output_dir`: Output directory for classifier weights (default: `/Data/yash.bhardwaj/.../Weights/classifier_guidance`)
- `--batch_size`: Training batch size (default: 64)
- `--num_epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 1e-3)
- `--seed`: Random seed (default: 42)
- `--resume_from`: Path to checkpoint to resume from (optional)

**Note**: The first epoch will be slower as it builds the latent cache. Subsequent epochs will be much faster!

### 2. Generate Images with Classifier Guidance

```bash
python approaches/classifier_guidance/src/inference.py \
    --prompt "A living room" \
    --emotion_idx 6 \
    --output_path output.png \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --classifier_scale 20.0 \
    --seed 42
```

**Arguments**:
- `--prompt`: Text prompt for generation
- `--emotion_idx`: Target emotion index (0-7)
  - 0: amusement, 1: anger, 2: awe, 3: contentment
  - 4: disgust, 5: excitement, 6: fear, 7: sadness
- `--classifier_path`: Path to trained classifier (default: `/Data/yash.bhardwaj/.../Weights/classifier_guidance/classifier.pt`)
- `--output_path`: Output image path
- `--num_inference_steps`: Number of denoising steps (default: 50)
- `--guidance_scale`: CFG scale (default: 7.5)
- `--classifier_scale`: Classifier guidance strength (default: 20.0)
- `--seed`: Random seed (default: 42)

## How It Works

### Training

1. **VAE Encoding**: Images are encoded to latents using Stable Diffusion's VAE
2. **Noise Addition**: Random timesteps are sampled, and noise is added to latents
3. **Classification**: The classifier predicts emotion from noisy latents
4. **Loss**: CrossEntropyLoss between predictions and ground-truth emotions

### Inference

1. **Standard CFG**: Compute noise prediction with classifier-free guidance
2. **Classifier Guidance**: 
   - Detach latents and enable gradients
   - Forward pass through classifier
   - Compute gradient w.r.t. latents
   - Update noise prediction: `noise_pred = noise_pred - classifier_scale * grad`
3. **Denoising**: Step the scheduler with the updated noise prediction

## Hyperparameters

### Training
- **Learning Rate**: 1e-3
- **Batch Size**: 64
- **Epochs**: 10
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss

### Inference
- **Classifier Scale**: 20.0 (controls guidance strength)
- **CFG Scale**: 7.5 (standard classifier-free guidance)
- **Inference Steps**: 50

## File Structure

```
approaches/classifier_guidance/
├── README.md                    # This file
└── src/
    ├── __init__.py             # Package exports
    ├── model.py                # EmotionLatentClassifier architecture
    ├── train_classifier.py     # Training script
    └── inference.py            # Inference with classifier guidance
```

## Outputs

- **Classifier Weights**: Saved to `--output_dir/classifier.pt`
- **Best Model**: Saved when validation loss improves
- **Latent Cache**: Stored in `--cache_dir/` for faster subsequent training

## Notes

- The classifier is trained on the entire dataset (94K images from `emoset_full`)
- Latents are cached on disk to avoid redundant VAE encoding
- The classifier doesn't require captions - only images and emotions
- First epoch is slower due to cache building, subsequent epochs are faster

