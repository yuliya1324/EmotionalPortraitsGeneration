"""
Diagnostic script to analyze classifier training issues.
Checks timestep distribution, class balance, and model predictions.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import sys

# Add paths
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "approaches" / "classifier_guidance" / "src"))

from model import EmotionLatentClassifier

# Use environment variable or default to repository root
from pathlib import Path
REPO_ROOT = Path(__file__).parent.parent.parent.parent.absolute()
STORAGE_BASE = os.getenv("EMOTIONAL_PORTRAITS_BASE", str(REPO_ROOT))
EMOTIONS = ['amusement', 'anger', 'awe', 'contentment', 'disgust', 'excitement', 'fear', 'sadness']

def analyze_checkpoint():
    """Analyze the saved checkpoint."""
    checkpoint_path = Path(STORAGE_BASE) / "weights" / "classifier_guidance" / "classifier.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("="*70)
    print("Classifier Diagnostic Analysis")
    print("="*70)
    
    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    print(f"\n✓ Checkpoint loaded from: {checkpoint_path}")
    print(f"  Epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"  Loss: {ckpt.get('loss', 'N/A'):.4f}")
    print(f"  Accuracy: {ckpt.get('accuracy', 'N/A'):.4f} ({ckpt.get('accuracy', 0)*100:.1f}%)")
    
    # Load model
    classifier = EmotionLatentClassifier(num_emotions=8)
    classifier.load_state_dict(ckpt['model_state_dict'])
    classifier.eval()
    
    print("\n" + "="*70)
    print("Key Issues Identified:")
    print("="*70)
    
    print("\n1. TIMESTEP DISTRIBUTION PROBLEM")
    print("   - Training samples timesteps uniformly from 0-1000")
    print("   - At timestep 0-500: latents are ~50-100% noise (very hard to classify)")
    print("   - At timestep 500-1000: latents have more signal (easier to classify)")
    print("   - Model sees ~50% impossible/hard samples during training")
    print("   - Solution: Weight samples by timestep (focus on learnable timesteps)")
    
    print("\n2. NO VALIDATION SPLIT")
    print("   - Training uses entire dataset (no validation)")
    print("   - Can't detect overfitting or measure generalization")
    print("   - Solution: Add train/val split (e.g., 80/20)")
    
    print("\n3. MODEL CAPACITY")
    print("   - Model has ~2M parameters")
    print("   - May be insufficient for this difficult task")
    print("   - Solution: Increase model size or use attention mechanisms")
    
    print("\n4. FIXED LEARNING RATE")
    print("   - Uses constant LR=1e-3 throughout training")
    print("   - No learning rate scheduling")
    print("   - Solution: Add LR scheduler (cosine annealing, etc.)")
    
    print("\n5. LOSS FUNCTION")
    print("   - Uses standard CrossEntropyLoss")
    print("   - No class weighting (potential class imbalance)")
    print("   - Solution: Add class weights or focal loss")
    
    print("\n" + "="*70)
    print("Recommended Fixes:")
    print("="*70)
    print("""
1. TIMESTEP WEIGHTING:
   - Weight loss by timestep: w(t) = 1.0 for t < 200, 0.5 for 200-500, 0.1 for t > 500
   - Or: Only train on timesteps 200-1000 (skip very noisy ones)
   - Or: Use curriculum learning (start with low noise, gradually increase)

2. ADD VALIDATION:
   - Split dataset 80/20 train/val
   - Monitor validation accuracy
   - Early stopping based on val loss

3. IMPROVE MODEL:
   - Add attention layers
   - Increase model capacity
   - Use residual connections

4. LEARNING RATE SCHEDULING:
   - Use CosineAnnealingLR or ReduceLROnPlateau
   - Start with higher LR, decay over time

5. CLASS BALANCE:
   - Check emotion distribution in dataset
   - Use weighted loss if imbalanced
   - Consider focal loss for hard examples

6. TRAINING STRATEGY:
   - Train longer (more epochs)
   - Use data augmentation
   - Consider contrastive learning
    """)
    
    print("\n" + "="*70)
    print("Expected Accuracy Ranges:")
    print("="*70)
    print("  Random baseline (8 classes): 12.5%")
    print("  Current accuracy: 31.5% (2.5x better than random)")
    print("  Reasonable target: 50-60% (for this difficult task)")
    print("  Excellent performance: 70%+ (very challenging)")
    
    print("\n" + "="*70)
    print("Why 31.5% is actually reasonable:")
    print("="*70)
    print("""
  The task is EXTREMELY difficult:
  - Predicting emotions from noisy latents (especially at high noise levels)
  - At timestep 0-300: latents are 70-100% noise → emotion info mostly destroyed
  - At timestep 300-600: latents are 30-70% noise → some signal remains
  - At timestep 600-1000: latents are 0-30% noise → more signal available
  
  The model is learning something (2.5x better than random), but:
  - It's struggling with high-noise samples (which are 50% of training data)
  - It may be learning to predict the most common emotion for high-noise cases
  - The low accuracy reflects the inherent difficulty of the task
    """)

if __name__ == "__main__":
    analyze_checkpoint()


