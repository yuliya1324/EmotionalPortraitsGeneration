# Experiment Structure

This repository is organized to support multiple approaches for emotion-conditioned image generation, with a clean evaluation framework.

## Directory Structure

```
EmotionalPortraitsGeneration/
├── approaches/              # Different approaches/experiments
│   ├── baseline_lora/       # Baseline: LoRA + learned embeddings
│   │   └── src/
│   │       ├── train.py     # Training script for this approach
│   │       └── inference.py # Inference script for this approach
│   ├── approach_2/          # Future approaches...
│   └── ...
├── shared/                  # Shared utilities
│   └── src/
│       ├── dataset.py       # Dataset loading (used by all approaches)
│       └── preprocessing.py # Dataset preprocessing (BLIP captioning)
├── run_experiment.py        # Main experiment runner
└── ...
```

## Storage Structure

All outputs are stored in `/Data/yash.bhardwaj/EmotionalPortraitGeneration/`:

```
/Data/yash.bhardwaj/EmotionalPortraitGeneration/
├── Weights/                 # Model weights/checkpoints
│   ├── 10K/                 # 10K dataset experiments
│   │   ├── baseline_lora/
│   │   └── approach_2/
│   ├── 25K/                 # 25K dataset experiments
│   │   ├── baseline_lora/
│   │   └── approach_2/
│   └── 30K/                 # 30K dataset experiments
│       ├── baseline_lora/
│       └── approach_2/
├── Logs/                    # Training logs and validation images
│   ├── 10K/
│   │   ├── baseline_lora/
│   │   └── approach_2/
│   ├── 25K/
│   └── 30K/
└── Datasets/                # Processed datasets
    ├── emoset_captioned_10k/
    ├── emoset_captioned_25k/
    └── emoset_captioned_30k/
```

## Usage

### 1. Preprocess a Dataset

Generate captions for a dataset variant:

```bash
python shared/src/preprocessing.py \
    --subset_size 30000 \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitGeneration/Datasets/emoset_captioned_30k \
    --batch_size 8 \
    --seed 42
```

### 2. List Available Approaches

```bash
python run_experiment.py list-approaches
```

### 3. Train a Model

Train using the experiment runner:

```bash
# Basic training
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K

# With custom parameters
python run_experiment.py train \
    --approach baseline_lora \
    --dataset-size 30K \
    --batch-size 8 \
    --num-epochs 7 \
    --lora-r 32 \
    --lora-alpha 64 \
    --gradient-accumulation-steps 2
```

This will:
- Load dataset from `/Data/yash.bhardwaj/EmotionalPortraitGeneration/Datasets/emoset_captioned_30k`
- Save weights to `/Data/yash.bhardwaj/EmotionalPortraitGeneration/Weights/30K/baseline_lora/`
- Save logs to `/Data/yash.bhardwaj/EmotionalPortraitGeneration/Logs/30K/baseline_lora/`

### 4. Run Inference

```bash
python run_experiment.py inference \
    --approach baseline_lora \
    --dataset-size 30K \
    --prompt "A living room" \
    --guidance-scale 7.5
```

### 5. Direct Training (Alternative)

You can also run training directly if you need more control:

```bash
accelerate launch approaches/baseline_lora/src/train.py \
    --data_dir /Data/yash.bhardwaj/EmotionalPortraitGeneration/Datasets/emoset_captioned_30k \
    --output_dir /Data/yash.bhardwaj/EmotionalPortraitGeneration/Weights/30K/baseline_lora \
    --log_dir /Data/yash.bhardwaj/EmotionalPortraitGeneration/Logs/30K/baseline_lora \
    --batch_size 8 \
    --num_epochs 7
```

## Adding a New Approach

1. Create a new directory in `approaches/`:
   ```bash
   mkdir -p approaches/my_approach/src
   ```

2. Create training and inference scripts:
   - `approaches/my_approach/src/train.py`
   - `approaches/my_approach/src/inference.py`

3. Use shared utilities:
   ```python
   import sys
   from pathlib import Path
   
   REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
   SHARED_DIR = REPO_ROOT / "shared" / "src"
   sys.path.insert(0, str(SHARED_DIR))
   
   from dataset import EmoSetLocalDataset
   ```

4. Your approach will automatically appear in `list-approaches` and can be run with `run_experiment.py`

## Training Parameters

Common training parameters (passed via `run_experiment.py`):

- `--batch-size`: Training batch size (default: 4)
- `--num-epochs`: Number of epochs (default: 10)
- `--lr-lora`: Learning rate for LoRA (default: 1e-4)
- `--lr-embeddings`: Learning rate for embeddings (default: 1e-3)
- `--lora-r`: LoRA rank (default: 16)
- `--lora-alpha`: LoRA alpha (default: 32)
- `--save-steps`: Save checkpoint every N steps (default: 500)
- `--validation-steps`: Generate validation images every N steps (default: 500)
- `--gradient-accumulation-steps`: Gradient accumulation (default: 4)
- `--seed`: Random seed (default: 42)

## Dataset Sizes

Supported dataset sizes:
- `10K` - 10,000 images
- `25K` - 25,000 images
- `30K` - 30,000 images
- Any size can be specified (e.g., `50K`)

The dataset size is used to:
1. Load the correct dataset from `Datasets/emoset_captioned_{size}/`
2. Organize weights in `Weights/{size}/`
3. Organize logs in `Logs/{size}/`

## Comparing Approaches

To compare different approaches:

1. Train each approach on the same dataset size
2. Checkpoints are saved in `Weights/{size}/{approach}/`
3. Logs and validation images in `Logs/{size}/{approach}/`
4. Run inference with the same prompts to compare outputs

## Notes

- All paths use the centralized storage in `/Data/yash.bhardwaj/EmotionalPortraitGeneration/`
- The experiment runner automatically creates necessary directories
- Each approach is self-contained but can use shared utilities
- Dataset preprocessing is shared across all approaches

