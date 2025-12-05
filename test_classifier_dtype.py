#!/usr/bin/env python
"""
Quick test to verify classifier dtype handling works correctly.
"""
import os
import sys
import torch
from pathlib import Path

# Add paths
REPO_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(REPO_ROOT / "approaches" / "classifier_guidance" / "src"))

# Set cache
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"
CACHE_DIR = os.path.join(STORAGE_BASE, "cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")

from model import EmotionLatentClassifier

def test_classifier_dtype():
    """Test that classifier handles float32 inputs correctly."""
    print("=" * 60)
    print("CLASSIFIER DTYPE TEST")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create classifier
    classifier = EmotionLatentClassifier(num_emotions=8)
    classifier = classifier.to(device)
    classifier = classifier.float()
    classifier.eval()
    
    print("\n1. Checking classifier parameters...")
    for name, param in classifier.named_parameters():
        print(f"  {name}: {param.dtype}")
        assert param.dtype == torch.float32, f"{name} is not float32!"
    
    print("\n2. Checking classifier buffers...")
    for name, buf in classifier.named_buffers():
        print(f"  {name}: {buf.dtype}")
        if buf.dtype not in [torch.int64, torch.float32]:
            print(f"    WARNING: {name} is {buf.dtype}, not float32!")
    
    print("\n3. Testing forward pass with float32 inputs...")
    x = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=device)
    t = torch.tensor([10], dtype=torch.float32, device=device)
    
    try:
        with torch.no_grad():
            logits = classifier(x, t)
        print(f"  ✓ Forward pass successful!")
        print(f"  Output shape: {logits.shape}")
        print(f"  Output dtype: {logits.dtype}")
        print(f"  Output values: {logits}")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        return False
    
    print("\n4. Testing backward pass for classifier guidance...")
    x = torch.randn(1, 4, 64, 64, dtype=torch.float32, device=device, requires_grad=True)
    t = torch.tensor([10], dtype=torch.float32, device=device)
    
    try:
        logits = classifier(x, t)
        loss = -logits[0, 0]  # Target emotion 0
        grad = torch.autograd.grad(loss, x, create_graph=False)[0]
        print(f"  ✓ Gradient computation successful!")
        print(f"  Gradient shape: {grad.shape}")
        print(f"  Gradient dtype: {grad.dtype}")
        print(f"  Gradient norm: {grad.norm():.6f}")
    except Exception as e:
        print(f"  ✗ Gradient computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_classifier_dtype()
    sys.exit(0 if success else 1)
