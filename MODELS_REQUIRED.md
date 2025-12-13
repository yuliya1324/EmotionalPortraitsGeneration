# Models Required for Replication

This document lists all pre-trained models that need to be downloaded to replicate the results.

## Summary

**Minimum Required Size:** ~9.8 GB  
**Total Size (with optional):** ~10.1 GB

---

## Required Models
Base
### 1. Stable Diffusion v1.5
- **Model ID:** `runwayml/stable-diffusion-v1-5`
- **Size:** ~4.0 GB
- **Used for:**
  - Baseline LoRA training and inference
  - Classifier Guidance inference (base pipeline)
- **Download Method:** 
  ```python
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
  ```
- **Components:** UNet, VAE, Text Encoder, Tokenizer, Scheduler

### 2. Stable Diffusion v1.4
- **Model ID:** `CompVis/stable-diffusion-v1-4`
- **Size:** ~4.0 GB
- **Used for:**
  - Baseline evaluation (label-only approach)
  - Baseline evaluation (scene+emotion approach)
- **Download Method:**
  ```python
  from diffusers import StableDiffusionPipeline
  pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")
  ```
- **Components:** UNet, VAE, Text Encoder, Tokenizer, Scheduler

### 3. EmotionCLIP
- **Model ID:** `jiangchengchengNLP/EmotionCLIP`
- **Size:** ~1.8 GB
- **Used for:**
  - EmoSet emotion evaluation (all scene-based evaluations)
- **Download Method:**
  ```bash
  git clone https://huggingface.co/jiangchengchengNLP/EmotionCLIP /path/to/EmotionCLIP
  ```
- **Note:** This is a git repository, not a standard HuggingFace model

---

## Optional Models

### 4. Face Emotion Detection (ViT)
- **Model ID:** `abhilash88/face-emotion-detection`
- **Size:** ~300 MB
- **Used for:**
  - RAFDB emotion evaluation (portrait-based evaluations only)
- **Download Method:**
  ```python
  from transformers import ViTImageProcessor, ViTForImageClassification
  processor = ViTImageProcessor.from_pretrained('abhilash88/face-emotion-detection')
  model = ViTForImageClassification.from_pretrained('abhilash88/face-emotion-detection')
  ```
- **Required:** Only if evaluating on RAFDB dataset (portrait generation)

---

## Download Instructions

### Option 1: Automatic Download (Recommended)
Models will be automatically downloaded on first use when running:
- `evaluate.py` (for evaluation)
- `approaches/baseline_lora/src/train.py` (for training)
- `approaches/classifier_guidance/src/train_classifier.py` (for classifier training)

### Option 2: Manual Pre-download
To pre-download all models:

```python
# Download Stable Diffusion models
from diffusers import StableDiffusionPipeline

# SD v1.5 (required)
pipe_v15 = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    cache_dir="/path/to/cache"
)

# SD v1.4 (required)
pipe_v14 = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    cache_dir="/path/to/cache"
)

# Download ViT model (optional, for RAFDB)
from transformers import ViTImageProcessor, ViTForImageClassification
processor = ViTImageProcessor.from_pretrained(
    'abhilash88/face-emotion-detection',
    cache_dir="/path/to/cache"
)
model = ViTForImageClassification.from_pretrained(
    'abhilash88/face-emotion-detection',
    cache_dir="/path/to/cache"
)
```

```bash
# Download EmotionCLIP (required)
git clone https://huggingface.co/jiangchengchengNLP/EmotionCLIP /path/to/EmotionCLIP
```

---

## Cache Directory Structure

Models are cached in:
```
<repository_root>/cache/huggingface/
├── models--runwayml--stable-diffusion-v1-5/     (~4.0 GB)
├── models--CompVis--stable-diffusion-v1-4/      (~4.0 GB)
└── models--abhilash88--face-emotion-detection/  (~300 MB)

/path/to/EmotionCLIP/                            (~1.8 GB)
```

---

## Size Breakdown

| Model | Size | Required |
|-------|------|----------|
| Stable Diffusion v1.5 | ~4.0 GB | ✅ Yes |
| Stable Diffusion v1.4 | ~4.0 GB | ✅ Yes |
| EmotionCLIP | ~1.8 GB | ✅ Yes |
| Face Emotion Detection (ViT) | ~300 MB | ⚠️ Optional |
| **Total (minimum)** | **~9.8 GB** | |
| **Total (with optional)** | **~10.1 GB** | |

---

## Notes

1. **Stable Diffusion v1.4 vs v1.5:** Both are needed because:
   - v1.4 is used for baseline comparisons
   - v1.5 is used for fine-tuned approaches (LoRA, Classifier Guidance)

2. **EmotionCLIP:** This is a git repository, not a standard HuggingFace model. It must be cloned separately.

3. **ViT Model:** Only needed if you plan to evaluate on RAFDB dataset (portrait generation). For EmoSet evaluations (scene generation), only EmotionCLIP is needed.

4. **Disk Space:** Ensure you have at least **10-11 GB** free space for model downloads and caching.

5. **Network:** First-time download may take time depending on your internet connection. Models are cached locally after first download.


