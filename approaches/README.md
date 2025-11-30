# Approaches

This directory contains different approaches for emotion-conditioned image generation.

## Current Approaches

- **baseline_lora**: Baseline approach using LoRA adapters + learned emotion token embeddings

## Adding a New Approach

1. Create a new directory:
   ```bash
   mkdir -p approaches/my_approach/src
   ```

2. Create `train.py` and `inference.py` in the `src/` directory:
   - `train.py`: Training script for your approach
   - `inference.py`: Inference script for your approach

3. Use shared utilities:
   ```python
   import sys
   from pathlib import Path
   
   # Add shared directory to path
   REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
   SHARED_DIR = REPO_ROOT / "shared" / "src"
   sys.path.insert(0, str(SHARED_DIR))
   
   from dataset import EmoSetLocalDataset
   ```

4. Your approach should accept these standard arguments:
   - `--data_dir`: Path to dataset directory
   - `--output_dir`: Path to save model weights
   - `--log_dir`: Path to save logs and validation images

5. The approach will automatically appear when you run:
   ```bash
   python run_experiment.py list-approaches
   ```

## Approach Structure

Each approach directory should have:
```
my_approach/
└── src/
    ├── train.py      # Training script
    └── inference.py  # Inference script
```

Optional:
- `README.md`: Documentation for the approach
- `config.py`: Configuration file
- Other supporting files

## Example: Minimal Approach

**approaches/my_approach/src/train.py**:
```python
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
SHARED_DIR = REPO_ROOT / "shared" / "src"
sys.path.insert(0, str(SHARED_DIR))

from dataset import EmoSetLocalDataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    # ... your approach-specific arguments
    
    args = parser.parse_args()
    
    # Your training code here
    dataset = EmoSetLocalDataset(data_dir=args.data_dir)
    # ...

if __name__ == "__main__":
    main()
```

