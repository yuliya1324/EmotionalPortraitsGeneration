# Controllable Emotional Scene Generation

A research repository for fine-tuning Stable Diffusion v1.5 to generate scenes conditioned on specific emotion labels using the **EmoSet-118K** dataset. This project enables generating the same scene with different emotional variants (e.g., "A living room <fear>" vs "A living room <excitement>") while preserving structural composition.

## Architecture

The approach combines **LoRA (Low-Rank Adaptation)** with **learned token embeddings**:

1. **Preprocessing**: Generate captions for EmoSet-118K using BLIP (since the dataset lacks captions)
2. **Tokenizer**: Adds 8 special emotion tokens: `<amusement>`, `<awe>`, `<contentment>`, `<excitement>`, `<anger>`, `<disgust>`, `<fear>`, `<sadness>`
3. **Text Encoder**: Resized to accommodate new tokens, with embeddings optimized during training (higher learning rate: 1e-3)
4. **UNet**: LoRA adapters applied to cross-attention layers for style adaptation (learning rate: 1e-4)
5. **Dataset**: EmoSet-118K subset (10,000 images) with BLIP-generated captions

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended) or MPS (Apple Silicon) or CPU
- 16GB+ RAM recommended
- 20GB+ free disk space for model weights, dataset, and generated data

### Setup

1. Clone this repository:
```bash
cd EmotionalPortraitsGeneration
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')"
python -c "from transformers import BlipProcessor; print('BLIP available')"
```

## Workflow

The project follows a three-step workflow:

1. **Preprocess**: Generate captions using BLIP
2. **Train**: Fine-tune Stable Diffusion with LoRA + learned embeddings
3. **Inference**: Generate emotion-conditioned images

---

## Step 1: Preprocessing

Since EmoSet-118K lacks captions, we generate them using BLIP on a subset of 10,000 images.

### Run Preprocessing

```bash
python src/preprocess.py
```

### Preprocessing Options

```bash
python src/preprocess.py \
    --dataset_name Woleek/EmoSet-118K \
    --split train \
    --subset_size 10000 \
    --output_dir ./data/emoset_captioned_10k \
    --batch_size 8 \
    --seed 42 \
    --model_name Salesforce/blip-image-captioning-base
```

### Arguments

- `--dataset_name`: HuggingFace dataset name (default: `Woleek/EmoSet-118K`)
- `--split`: Dataset split (default: `train`)
- `--subset_size`: Number of images to process (default: `10000`)
- `--output_dir`: Output directory (default: `./data/emoset_captioned_10k`)
- `--batch_size`: Batch size for caption generation (default: `8`)
- `--seed`: Random seed (default: `42`)
- `--model_name`: BLIP model name (default: `Salesforce/blip-image-captioning-base`)

### Output

The preprocessing script will:
- Download EmoSet-118K from HuggingFace
- Shuffle and select 10,000 images
- Generate captions using BLIP
- Save the processed dataset to `./data/emoset_captioned_10k`

**Expected time**: ~2-4 hours on a GPU (depends on GPU and batch size)

---

## Step 2: Training

Train the model with LoRA adapters and learned token embeddings.

### Basic Training

```bash
accelerate launch src/train.py
```

### Advanced Training Options

```bash
accelerate launch src/train.py \
    --data_dir ./data/emoset_captioned_10k \
    --output_dir output/final_model \
    --log_dir output/logs \
    --batch_size 4 \
    --num_epochs 10 \
    --lr_lora 1e-4 \
    --lr_embeddings 1e-3 \
    --lora_r 16 \
    --lora_alpha 32 \
    --save_steps 500 \
    --validation_steps 500 \
    --seed 42 \
    --init_word style \
    --test_prompt_1 "A living room <fear>" \
    --test_prompt_2 "A living room <excitement>"
```

### Training Arguments

- `--data_dir`: Path to local dataset directory (default: `./data/emoset_captioned_10k`)
- `--output_dir`: Output directory for checkpoints (default: `output/final_model`)
- `--log_dir`: Directory for validation images (default: `output/logs`)
- `--batch_size`: Training batch size (default: `4`)
- `--num_epochs`: Number of training epochs (default: `10`)
- `--lr_lora`: Learning rate for LoRA parameters (default: `1e-4`)
- `--lr_embeddings`: Learning rate for token embeddings (default: `1e-3`)
- `--lora_r`: LoRA rank (default: `16`)
- `--lora_alpha`: LoRA alpha scaling (default: `32`)
- `--save_steps`: Save checkpoint every N steps (default: `500`)
- `--validation_steps`: Generate validation images every N steps (default: `500`)
- `--gradient_accumulation_steps`: Gradient accumulation (default: `1`)
- `--seed`: Random seed (default: `42`)
- `--init_word`: Word to initialize emotion tokens (default: `style`)
- `--test_prompt_1`: First validation prompt (default: `A living room <fear>`)
- `--test_prompt_2`: Second validation prompt (default: `A living room <excitement>`)

### Accelerate Configuration

First-time setup for Accelerate:
```bash
accelerate config
```

Recommended settings:
- Mixed precision: `fp16` (if CUDA) or `no` (if CPU/MPS)
- Multi-GPU: Configure if available
- DeepSpeed: Optional for large-scale training

### Training Output

The training script saves:
- **LoRA weights**: `output/final_model/` (PEFT adapter format)
- **Learned embeddings**: `output/final_model/learned_embeds.bin`
- **Tokenizer info**: `output/final_model/tokenizer_info.json`
- **Validation images**: `output/logs/step_*.png` (generated every 500 steps and at end of each epoch)

### Validation Logging

During training, the script automatically generates validation images every 500 steps (or 1 epoch) to visually track progress. These images are saved to `output/logs/` and show how the model learns to generate different emotions for the same scene.

**Expected time**: ~8-12 hours on a single GPU for 10 epochs (depends on GPU and dataset size)

---

## Step 3: Inference

Generate emotion-conditioned images using the trained model.

### Basic Inference

```bash
python src/inference.py --prompt "A photo of a park"
```

### Advanced Inference Options

```bash
python src/inference.py \
    --prompt "A photo of a park" \
    --lora_path output/final_model \
    --learned_embeds_path output/final_model/learned_embeds.bin \
    --tokenizer_info_path output/final_model/tokenizer_info.json \
    --output_path emotion_comparison.png \
    --seed 42 \
    --num_inference_steps 50 \
    --guidance_scale 7.5 \
    --grid_cols 4
```

### Inference Arguments

- `--prompt`: Base prompt (e.g., "A photo of a park", "A living room")
- `--lora_path`: Path to LoRA checkpoint directory (default: `output/final_model`)
- `--learned_embeds_path`: Path to learned embeddings (default: `output/final_model/learned_embeds.bin`)
- `--tokenizer_info_path`: Path to tokenizer info (default: `output/final_model/tokenizer_info.json`)
- `--output_path`: Output image path (default: `emotion_comparison.png`)
- `--seed`: Random seed for reproducibility (default: `42`)
- `--num_inference_steps`: Number of diffusion steps (default: `50`)
- `--guidance_scale`: Guidance scale (default: `7.5`)
- `--grid_cols`: Number of columns in grid (default: `4`)

### Example Prompts

```bash
# Outdoor scenes
python src/inference.py --prompt "A photo of a park"
python src/inference.py --prompt "A mountain landscape at sunset"

# Indoor scenes
python src/inference.py --prompt "A living room"
python src/inference.py --prompt "A cozy bedroom"

# Abstract concepts
python src/inference.py --prompt "A futuristic city"
```

### Output

The inference script generates a grid image showing the same scene with all 8 emotion tokens:
- Uses the **same seed** for all generations to preserve scene structure
- Creates a 4x2 grid (or custom layout) with labels
- Saves to the specified output path

---

## Project Structure

```
EmotionalPortraitsGeneration/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── preprocess.py        # BLIP caption generation
│   ├── dataset.py           # Local dataset loading
│   ├── train.py             # Training script
│   └── inference.py         # Inference and visualization
├── data/
│   └── emoset_captioned_10k/  # Processed dataset (created by preprocess.py)
├── output/
│   ├── final_model/         # Training outputs (created during training)
│   │   ├── learned_embeds.bin
│   │   ├── tokenizer_info.json
│   │   └── adapter_model.safetensors  # LoRA weights
│   └── logs/                # Validation images (created during training)
│       └── step_*.png
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── .gitignore              # Git ignore rules
```

## Emotion Tokens

The model supports 8 emotion tokens:

- `<amusement>` - Light-hearted, playful scenes
- `<awe>` - Inspiring, majestic scenes
- `<contentment>` - Peaceful, satisfied scenes
- `<excitement>` - Energetic, dynamic scenes
- `<anger>` - Intense, aggressive scenes
- `<disgust>` - Repulsive, unpleasant scenes
- `<fear>` - Tense, frightening scenes
- `<sadness>` - Melancholic, somber scenes

## Device Support

The code automatically detects and uses the best available device:

- **CUDA**: Full support with mixed precision (fp16)
- **MPS**: Apple Silicon GPU support
- **CPU**: Fallback support (slower)

## Troubleshooting

### Out of Memory Errors

- Reduce `--batch_size` in preprocessing (e.g., `--batch_size 4`)
- Reduce `--batch_size` in training (e.g., `--batch_size 2` or `1`)
- Increase `--gradient_accumulation_steps` to maintain effective batch size
- Use `--lora_r 8` to reduce LoRA rank

### Slow Training/Preprocessing

- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use mixed precision: `accelerate config` → select `fp16`
- Increase batch size if memory allows
- For preprocessing, use larger `--batch_size` (e.g., 16 or 32) if GPU memory allows

### Dataset Loading Issues

- Ensure preprocessing completed successfully
- Check that `./data/emoset_captioned_10k` exists
- Verify dataset has `generated_caption` column: `python src/dataset.py`

### Inference Errors

- Ensure training completed successfully
- Check that all checkpoint files exist in `output/final_model/`
- Verify tokenizer info matches emotion tokens
- Try loading with explicit paths: `--lora_path output/final_model`

### Validation Images Not Generating

- Check that `output/logs/` directory is writable
- Verify GPU memory is sufficient for inference during training
- Reduce `--num_inference_steps` in validation if needed (modify code)

## Citation

If you use this code in your research, please cite:

```bibtex
@software{emotional_scene_generation,
  title = {Controllable Emotional Scene Generation},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/EmotionalPortraitsGeneration}
}
```

## License

This project is provided for research purposes. Please ensure compliance with:
- Stable Diffusion v1.5 license (CreativeML Open RAIL-M)
- EmoSet-118K dataset license
- BLIP model license
- HuggingFace model licenses

## Acknowledgments

- Stable Diffusion by Stability AI
- EmoSet-118K dataset by Woleek
- BLIP by Salesforce Research
- HuggingFace Diffusers and Transformers libraries
- PEFT library for LoRA support

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For questions or issues, please open a GitHub issue.
