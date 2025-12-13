# Trained Model Weights and Checkpoints

This document lists all trained/fine-tuned model weights required to replicate the results.

## Summary

**Total Size:** ~159 MB

---

## Baseline LoRA Weights

### 1. 10K Dataset - Baseline LoRA
- **Path:** `Weights/10K/baseline_lora/`
- **Total Size:** ~25 MB
- **Files:**
  - `adapter_model.safetensors` (25 MB) - LoRA adapter weights for UNet
  - `learned_embeds.bin` (28 KB) - Learned emotion token embeddings
  - `adapter_config.json` (4 KB) - LoRA configuration
  - `tokenizer_info.json` (4 KB) - Tokenizer configuration with emotion tokens
  - `README.md` (8 KB) - Model documentation

**Configuration:**
- **LoRA Rank (r):** 32
- **LoRA Alpha:** 64
- **Target Modules:** `to_v`, `to_q`, `to_k`, `to_out.0` (cross-attention layers)
- **Emotion Tokens:** 8 tokens (`<amusement>`, `<anger>`, `<awe>`, `<contentment>`, `<disgust>`, `<excitement>`, `<fear>`, `<sadness>`)
- **Base Model:** `runwayml/stable-diffusion-v1-5`

**Usage:**
```python
from diffusers import StableDiffusionPipeline
from peft import PeftModel

# Load base model
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

# Load LoRA weights
pipe.unet = PeftModel.from_pretrained(pipe.unet, "Weights/10K/baseline_lora")
pipe.unet = pipe.unet.merge_and_unload()

# Load learned embeddings
learned_embeds = torch.load("Weights/10K/baseline_lora/learned_embeds.bin")
# ... (see evaluate.py for full loading code)
```

---

### 2. 25K Dataset - Baseline LoRA
- **Path:** `Weights/25K/baseline_lora/`
- **Total Size:** ~25 MB
- **Files:** Same structure as 10K version
  - `adapter_model.safetensors` (25 MB)
  - `learned_embeds.bin` (28 KB)
  - `adapter_config.json` (4 KB)
  - `tokenizer_info.json` (4 KB)
  - `README.md` (8 KB)

**Configuration:**
- **LoRA Rank (r):** 32
- **LoRA Alpha:** 64
- **Target Modules:** Same as 10K version
- **Emotion Tokens:** Same 8 tokens
- **Base Model:** `runwayml/stable-diffusion-v1-5`

**Note:** Same architecture as 10K, but trained on larger dataset (25K vs 10K images).

---

## Classifier Guidance Weights

### 3. Classifier Guidance - Standard Classifier
- **Path:** `Weights/classifier_guidance/classifier.pt`
- **Size:** ~23 MB
- **Type:** PyTorch checkpoint
- **Model:** `EmotionLatentClassifier` (standard architecture)
- **Training Info:**
  - **Final Epoch:** 10
  - **Final Loss:** 1.795
  - **Final Accuracy:** 32.07%
  - **Parameters:** ~1.95M

**Architecture:**
- 4 Conv blocks (4→64→128→256→512 channels)
- Time embedding (256→512 dim)
- Global Average Pooling
- Linear classifier (512→8 emotions)

**Usage:**
```python
from approaches.classifier_guidance.src.inference import load_classifier
classifier = load_classifier("Weights/classifier_guidance/classifier.pt", device=device)
```

---

### 4. Classifier Guidance - Large Classifier
- **Path:** `Weights/classifier_guidance/classifier_large.pt`
- **Size:** ~87 MB
- **Type:** PyTorch checkpoint
- **Model:** `EmotionLatentClassifier` (larger architecture)
- **Training Info:**
  - **Final Epoch:** 15
  - **Final Loss:** 1.600
  - **Final Accuracy:** 40.65%
  - **Parameters:** ~7.8M (estimated, based on size)

**Architecture:**
- Larger version with more channels (4→128→256→512→1024)
- Time embedding (256→1024 dim)
- Global Average Pooling
- Linear classifier (1024→8 emotions)

**Note:** This is the model currently used in evaluation (better accuracy than standard classifier).

**Usage:**
```python
from approaches.classifier_guidance.src.inference import load_classifier
classifier = load_classifier("Weights/classifier_guidance/classifier_large.pt", device=device)
```

---

## File Structure

```
Weights/
├── 10K/
│   └── baseline_lora/
│       ├── adapter_model.safetensors      (25 MB)
│       ├── learned_embeds.bin             (28 KB)
│       ├── adapter_config.json            (4 KB)
│       ├── tokenizer_info.json            (4 KB)
│       └── README.md                      (8 KB)
│
├── 25K/
│   └── baseline_lora/
│       ├── adapter_model.safetensors      (25 MB)
│       ├── learned_embeds.bin             (28 KB)
│       ├── adapter_config.json            (4 KB)
│       ├── tokenizer_info.json            (4 KB)
│       └── README.md                      (8 KB)
│
└── classifier_guidance/
    ├── classifier.pt                      (23 MB)
    └── classifier_large.pt                (87 MB)
```

---

## Size Breakdown

| Model | Size | Description |
|-------|------|-------------|
| 10K/baseline_lora | ~25 MB | LoRA weights + embeddings for 10K dataset |
| 25K/baseline_lora | ~25 MB | LoRA weights + embeddings for 25K dataset |
| classifier_guidance/classifier.pt | ~23 MB | Standard classifier (32% accuracy) |
| classifier_guidance/classifier_large.pt | ~87 MB | Large classifier (40.65% accuracy) |
| **Total** | **~160 MB** | All trained weights |

---

## Required Files for Replication

### Minimum (for evaluation only):
1. `10K/baseline_lora/` - All 5 files (~25 MB)
2. `25K/baseline_lora/` - All 5 files (~25 MB)
3. `classifier_guidance/classifier_large.pt` (~87 MB)

**Minimum Total:** ~137 MB

### Complete (all trained models):
- All files listed above (~160 MB)

---

## Loading Instructions

### Baseline LoRA Models
The evaluation script (`evaluate.py`) automatically loads these weights. For manual loading, see `evaluate.py` function `load_generation_pipeline()`.

### Classifier Guidance
The evaluation script automatically loads the classifier. For manual loading:
```python
from approaches.classifier_guidance.src.inference import load_classifier
classifier = load_classifier("Weights/classifier_guidance/classifier_large.pt", device=device)
```

---

## Training Configuration Summary

### Baseline LoRA Training
- **Base Model:** `runwayml/stable-diffusion-v1-5`
- **LoRA Rank:** 32
- **LoRA Alpha:** 64
- **Target Modules:** Cross-attention layers (`to_q`, `to_k`, `to_v`, `to_out.0`)
- **Learning Rate (LoRA):** 1e-4
- **Learning Rate (Embeddings):** 5e-3
- **Batch Size:** 4
- **Gradient Accumulation:** 3
- **Epochs:** 10

### Classifier Guidance Training
- **Base Model:** Uses VAE from `runwayml/stable-diffusion-v1-5` (for encoding)
- **Classifier Architecture:** CNN with time embedding
- **Learning Rate:** 1e-3
- **Batch Size:** 64
- **Epochs:** 10 (standard) / 15 (large)
- **Dataset:** EmoSet (full dataset, ~94K images)

---

## Notes

1. **LoRA Weights:** The `adapter_model.safetensors` files contain only the trainable LoRA parameters, not the full UNet. They must be loaded onto the base Stable Diffusion v1.5 model.

2. **Learned Embeddings:** The `learned_embeds.bin` files contain the learned emotion token embeddings that must be loaded into the text encoder.

3. **Classifier Models:** Both classifier models are standalone and don't require the base Stable Diffusion model weights (only for inference, they use the base pipeline's VAE).

4. **Model Selection:** The evaluation script automatically selects `classifier_large.pt` if available, otherwise falls back to `classifier.pt`.

5. **Compatibility:** All weights are compatible with the codebase as of the evaluation date. Ensure you're using the same base models (`runwayml/stable-diffusion-v1-5` for LoRA, `CompVis/stable-diffusion-v1-4` for baselines).


