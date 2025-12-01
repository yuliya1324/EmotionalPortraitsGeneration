# Quick Start Guide

Complete guide from data preprocessing to running an approach.

## Overview

This repository supports multiple approaches for emotion-conditioned image generation. The workflow is:

1. **Preprocess Dataset** - Generate captions for EmoSet-118K
2. **Train Model** - Train a specific approach on the dataset
3. **Run Inference** - Generate emotion-conditioned images

---

## Step 1: Preprocess Dataset

Generate captions for a dataset variant using BLIP.

### Basic Usage

```bash
python shared/src/preprocessing.py \
    --subset_size 30000 \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_30k \
    --batch_size 8 \
    --seed 42
```

### Full Options

```bash
python shared/src/preprocessing.py \
    --dataset_name Woleek/EmoSet-118K \
    --split train \
    --subset_size 30000 \
    --full_dataset_dir /Data/yash.bhardwaj/datasets/emoset_full \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_30k \
    --batch_size 8 \
    --seed 42 \
    --model_name Salesforce/blip-image-captioning-base
```

### What This Does

1. Downloads full EmoSet-118K dataset (or loads from cache if exists)
2. Samples `subset_size` images (e.g., 30,000)
3. Generates captions using BLIP model
4. Saves processed dataset to `output_dir`

**Expected time**: ~2-4 hours for 30K images on GPU

### Common Dataset Sizes

- **10K**: `--subset_size 10000` → `emoset_captioned_10k`
- **25K**: `--subset_size 25000` → `emoset_captioned_25k`
- **30K**: `--subset_size 30000` → `emoset_captioned_30k`

---

## Step 2: List Available Approaches

Check what approaches are available:

```bash
python run_experiment.py list-approaches
```

Output:
```
Available approaches:
  - baseline_lora
```

---

## Step 3: Train a Model

Train a specific approach on your dataset.

### Basic Training

```bash
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K
```

This uses default parameters:
- Batch size: 4
- Epochs: 10
- LoRA rank: 16
- LoRA alpha: 32
- Learning rates: LoRA=1e-4, Embeddings=1e-3

### Custom Training Parameters

```bash
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --batch-size 8 \
    --num-epochs 7 \
    --lora-r 32 \
    --lora-alpha 64 \
    --gradient-accumulation-steps 2 \
    --save-steps 1000 \
    --validation-steps 1000
```

### All Training Parameters

```bash
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --batch-size 8 \                    # Training batch size
    --num-epochs 7 \                    # Number of epochs
    --lr-lora 1e-4 \                    # LoRA learning rate
    --lr-embeddings 1e-3 \              # Embeddings learning rate
    --lora-r 32 \                       # LoRA rank
    --lora-alpha 64 \                   # LoRA alpha
    --save-steps 1000 \                 # Save checkpoint every N steps
    --validation-steps 1000 \           # Generate validation images every N steps
    --gradient-accumulation-steps 2 \   # Gradient accumulation
    --seed 42 \                         # Random seed
    --init-word style                    # Word to initialize emotion tokens
```

### What Gets Saved

- **Weights**: `/Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora/`
  - `adapter_model.safetensors` - LoRA weights
  - `adapter_config.json` - LoRA configuration
  - `learned_embeds.bin` - Learned emotion token embeddings
  - `tokenizer_info.json` - Tokenizer information

- **Logs**: `/Data/yash.bhardwaj/EmotionalPortraitsGeneration/Logs/30K/baseline_lora/`
  - Validation images (every `validation_steps`)
  - Training logs
  - Loss history

**Expected time**: ~8-12 hours for 30K dataset, 7 epochs on single GPU

---

## Step 4: Run Inference

Generate emotion-conditioned images with your trained model.

### Basic Inference

```bash
python run_experiment.py inference \
    --approach baseline_lora \
    --dataset-size 30K \
    --prompt "A living room"
```

### Custom Inference Parameters

```bash
python run_experiment.py inference \
    --approach baseline_lora \
    --dataset-size 30K \
    --prompt "A photo of a park" \
    --seed 42 \
    --num-inference-steps 50 \
    --guidance-scale 7.5
```

### What This Does

1. Loads trained model from `Weights/30K/baseline_lora/`
2. Generates the same prompt with all 8 emotion tokens
3. Creates a grid image showing all emotions
4. Saves to `Logs/30K/baseline_lora/inference/`

### Emotion Tokens

The model generates images for these 8 emotions:
- `<amusement>` - Light-hearted, playful
- `<awe>` - Inspiring, majestic
- `<contentment>` - Peaceful, satisfied
- `<excitement>` - Energetic, dynamic
- `<anger>` - Intense, aggressive
- `<disgust>` - Repulsive, unpleasant
- `<fear>` - Tense, frightening
- `<sadness>` - Melancholic, somber

---

## Complete Example Workflow

### For 30K Dataset

```bash
# 1. Preprocess dataset (one-time, takes ~2-4 hours)
python shared/src/preprocessing.py \
    --subset_size 30000 \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_30k \
    --batch_size 8 \
    --seed 42

# 2. Train model (takes ~8-12 hours)
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --batch-size 8 \
    --num-epochs 7 \
    --lora-r 32 \
    --lora-alpha 64

# 3. Generate images
python run_experiment.py inference \
    --approach baseline_lora \
    --dataset-size 30K \
    --prompt "A living room"
```

### For 10K Dataset (Faster)

```bash
# 1. Preprocess
python shared/src/preprocessing.py \
    --subset_size 10000 \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_10k

# 2. Train
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 10K \
    --batch-size 4 \
    --num-epochs 10

# 3. Inference
python run_experiment.py inference \
    --approach baseline_lora \
    --dataset-size 10K \
    --prompt "A photo of a park"
```

---

## Direct Script Usage (Alternative)

You can also run scripts directly without `run_experiment.py`:

### Direct Training

```bash
accelerate launch approaches/baseline_lora/src/train.py \
    --data_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_30k \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora \
    --log_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Logs/30K/baseline_lora \
    --batch_size 8 \
    --num_epochs 7
```

### Direct Inference

```bash
python approaches/baseline_lora/src/inference.py \
    --prompt "A living room" \
    --lora_path /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora \
    --learned_embeds_path /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora/learned_embeds.bin \
    --tokenizer_info_path /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora/tokenizer_info.json
```

---

## Troubleshooting

### Dataset Not Found

If you get `FileNotFoundError` for dataset:
```bash
# Check if dataset exists
ls /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/

# If not, run preprocessing first
python shared/src/preprocessing.py --subset_size 30000 --output_dir /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/emoset_captioned_30k
```

### Out of Memory

Reduce batch size:
```bash
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --batch-size 2 \              # Reduce from 8
    --gradient-accumulation-steps 4  # Increase to maintain effective batch size
```

### Check GPU

```bash
nvidia-smi
```

### Resume Training

If training was interrupted:
```bash
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --resume-from /Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/30K/baseline_lora
```

---

## Storage Locations

All outputs are organized in:
- **Datasets**: `/Data/yash.bhardwaj/EmotionalPortraitsGeneration/Datasets/`
- **Weights**: `/Data/yash.bhardwaj/EmotionalPortraitsGeneration/Weights/{DatasetSize}/{Approach}/`
- **Logs**: `/Data/yash.bhardwaj/EmotionalPortraitsGeneration/Logs/{DatasetSize}/{Approach}/`

---

## Next Steps

- See `EXPERIMENT_STRUCTURE.md` for detailed structure documentation
- See `approaches/README.md` for adding new approaches
- See `EMOTION_IMPROVEMENTS.md` for strategies to improve emotion conditioning

