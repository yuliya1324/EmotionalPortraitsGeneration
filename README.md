# Emotional Portraits Generation

A research repository for emotion-conditioned image generation using Stable Diffusion with multiple approaches: baseline LoRA, classifier guidance, and multimodal conditioning.

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

**No configuration needed by default!** The repository automatically:
- Uses the repository root for all paths
- Looks for EmotionCLIP in common locations: `<repo_root>/EmotionCLIP`, `~/EmotionCLIP`, `/EmotionCLIP`

### When is EmotionCLIP needed?

- **Required for:** Evaluation (`evaluate.py`) with EmoSet dataset
- **Optional for:** Training multimodal approaches (falls back gracefully if missing)
- **Not needed for:** Baseline approaches, inference-only usage, RAFDB evaluation (uses ViT instead)

**To install EmotionCLIP:**
```bash
git clone https://huggingface.co/jiangchengchengNLP/EmotionCLIP <repo_root>/EmotionCLIP
```

The repository will automatically find it in `<repo_root>/EmotionCLIP`, `~/EmotionCLIP`, or `/EmotionCLIP`.

**Or set a custom path:**
```bash
export EMOTIONCLIP_PATH="/path/to/EmotionCLIP"
```

**Other optional settings:**
```bash
export EMOTIONAL_PORTRAITS_BASE="/path/to/base"
export HF_CACHE_DIR="/path/to/cache"
```

## Demo and Inference

### Interactive Demo Notebook

Open `demo.ipynb` to interactively test all approaches:
- Portrait Generation Baseline (Label-Only)
- Scene Generation Baseline (Label-Only / Label+Caption)
- Portrait Generation fine tuned on RafDB (Label-Only)
- EmoSet Label-to-Image (Label-Only)
- Emotion-Token Conditioning with LoRA
- Multimodal Conditioning (Caption + Emotion Embedding)
- Multimodal Conditioning with Emotion Classifier Reinforcement
- Classifier Guidance (Noise-Aware Latent Classifier)

### Quick Inference

```bash
# Generate images with a specific approach
# Note: --dataset and --task are optional (auto-detected for most approaches)
python evaluate.py \
    --approach baseline_lora \
    --dataset-size 25K
```

**Available approaches:**
- `baseline` - Scene Generation Baseline (Label-Only)
- `baseline_scene_emotion` - Scene Generation Baseline (Label+Caption)
- `baseline_rafdb` - Portrait Generation Baseline (Label-Only)
- `baseline_lora` - Emotion-Token Conditioning with LoRA (10K/25K datasets)
- `portraits` - Portrait Generation (Label-Only Prompting)
- `emoset_label2image` - General Image Generation (Label-Only Prompting)
- `emoset_multicond` - Multimodal Conditioning (Caption + Emotion Embedding)
- `emoset_multicond_classifier_001` - Multimodal Conditioning with Classifier Reinforcement (α=0.01)
- `emoset_multicond_classifier_01` - Multimodal Conditioning with Classifier Reinforcement (α=0.1)
- `classifier_guidance` - Classifier Guidance (Noise-Aware Latent Classifier)

## Training

Scripts to train individual models:

### 1. Emotion-Token Conditioning with LoRA

```bash
python approaches/baseline_lora/src/train.py \
    --data_dir /path/to/dataset \
    --output_dir weights/25K/baseline_lora \
    --batch_size 4 \
    --num_epochs 10 \
    --lr_lora 1e-4 \
    --lr_embeddings 5e-3
```

### 2. Classifier Guidance (Noise-Aware Latent Classifier)

```bash
python approaches/classifier_guidance/src/train_classifier.py \
    --data_dir /path/to/dataset \
    --output_dir weights/classifier_guidance \
    --batch_size 64 \
    --num_epochs 15
```

### 3. Portrait and Scene Generation Approaches

```bash
# Portrait Generation (Label-Only Prompting) - RafDB
# Note: --emotion_condition is a flag (no value needed)
python scripts/train_text_to_image_lora.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_name /path/to/rafdb \
    --output_dir weights/portraits \
    --dataset rafdb

# General Image Generation (Label-Only Prompting) - EmoSet
python scripts/train_text_to_image_lora.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_name /path/to/emoset \
    --output_dir weights/emoset_label2image \
    --dataset emoset

# Multimodal Conditioning (Caption + Emotion Embedding)
# Note: --emotion_condition enables emotion embedding, captions loaded from EMOSET_CAPTIONS_PATH
python scripts/train_text_to_image_lora.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_name /path/to/emoset \
    --output_dir weights/emoset_multicond \
    --dataset emoset \
    --emotion_condition

# Multimodal Conditioning with Emotion Classifier Reinforcement (α=0.1)
# Note: Classifier weight is hardcoded to 0.1 in the script
# For α=0.01, modify emo_weight in the script or use a different training run
python scripts/train_text_to_image_lora.py \
    --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
    --dataset_name /path/to/emoset \
    --output_dir weights/emoset_multicond_classifier_01 \
    --dataset emoset \
    --emotion_condition \
    --emo_classifier
```

**Note:** The classifier weight (α) is currently hardcoded to 0.1 in the script. To train with α=0.01, modify line 637 in `scripts/train_text_to_image_lora.py` to set `emo_weight = 0.01`.

## Deep Evaluation

Comprehensive evaluation with EmotionCLIP or ViT:

```bash
# Evaluate with EmotionCLIP (EmoSet)
# --dataset and --task are auto-detected for most approaches
python evaluate.py \
    --approach baseline_lora \
    --dataset-size 25K

# Evaluate with ViT (RAFDB portraits)
# --dataset and --task are auto-detected for portraits approach
python evaluate.py \
    --approach portraits \
    --dataset-size FULL

# Skip generation (use existing images)
python evaluate.py \
    --approach baseline_lora \
    --dataset-size 25K \
    --skip-generation
```

**Evaluation outputs:**
- Generated images: `validation_images/{size}/{approach}/`
- Evaluation reports: `Evaluations/{size}/{approach}/report/`
  - Confusion matrix
  - Per-emotion metrics
  - Summary statistics

## Weights

All trained weights are stored in `weights/`:
- `10K/baseline_lora/` - Emotion-Token Conditioning with LoRA (10K dataset)
- `25K/baseline_lora/` - Emotion-Token Conditioning with LoRA (25K dataset)
- `classifier_guidance/` - Classifier Guidance (Noise-Aware Latent Classifier)
- `portraits/` - Portrait Generation (Label-Only Prompting)
- `emoset_label2image/` - General Image Generation (Label-Only Prompting)
- `emoset_multicond/` - Multimodal Conditioning (Caption + Emotion Embedding)
- `emoset_multicond_classifier_001/` - Multimodal Conditioning with Classifier Reinforcement (α=0.01)
- `emoset_multicond_classifier_01/` - Multimodal Conditioning with Classifier Reinforcement (α=0.1)

See [TRAINED_WEIGHTS.md](TRAINED_WEIGHTS.md) for details.

## Models Required

Download pretrained models from HuggingFace:
- `runwayml/stable-diffusion-v1-5` (~4.0 GB)
- `CompVis/stable-diffusion-v1-4` (~4.0 GB)
- `jiangchengchengNLP/EmotionCLIP` (~1.8 GB)
- `abhilash88/face-emotion-detection` (~300 MB, optional)

See [MODELS_REQUIRED.md](MODELS_REQUIRED.md) for details.
