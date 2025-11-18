"""
Preprocessing script to generate captions for EmoSet-118K using BLIP.
"""

import os
import argparse
import torch
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import random


def setup_device():
    """Detect and return the appropriate device."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
    else:
        device = torch.device("cpu")
        print("Using CPU device")
    return device


def generate_captions_batch(
    processor,
    model,
    images,
    device,
    batch_size=8,
    max_length=50,
    num_beams=3
):
    """
    Generate captions for a batch of images.
    
    Args:
        processor: BLIP processor
        model: BLIP model
        images: List of PIL Images
        device: Device to run on
        batch_size: Batch size for processing
        max_length: Maximum caption length
        num_beams: Number of beams for beam search
        
    Returns:
        List of generated captions
    """
    captions = []
    
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        
        # Process images
        inputs = processor(
            images=batch_images,
            return_tensors="pt"
        )
        # Move pixel_values to device
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(device)
        
        # Generate captions
        with torch.no_grad():
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=max_length,
                num_beams=num_beams,
                do_sample=False
            )
        
        # Decode captions
        batch_captions = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        captions.extend(batch_captions)
    
    return captions


def main():
    parser = argparse.ArgumentParser(
        description="Generate captions for EmoSet-118K using BLIP"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Woleek/EmoSet-118K",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subset_size",
        type=int,
        default=10000,
        help="Number of images to process"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/emoset_captioned_10k",
        help="Output directory for processed dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for caption generation"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Salesforce/blip-image-captioning-base",
        help="BLIP model name"
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device()
    
    # Load dataset
    print("\n" + "="*50)
    print("Loading EmoSet-118K dataset...")
    print("="*50)
    dataset = load_dataset(args.dataset_name, split=args.split)
    print(f"Loaded {len(dataset)} examples")
    
    # Shuffle and select subset
    print(f"\nShuffling and selecting {args.subset_size} examples...")
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    selected_indices = indices[:args.subset_size]
    subset = dataset.select(selected_indices)
    print(f"Selected {len(subset)} examples")
    
    # Load BLIP model
    print("\n" + "="*50)
    print(f"Loading BLIP model: {args.model_name}")
    print("="*50)
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    print("BLIP model loaded successfully")
    
    # Generate captions
    print("\n" + "="*50)
    print("Generating captions...")
    print("="*50)
    
    generated_captions = []
    images_list = []
    
    # Collect all images first
    print("Loading images...")
    for example in tqdm(subset, desc="Loading images"):
        image = example['image']
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert('RGB')
        images_list.append(image)
    
    # Generate captions in batches
    print("\nGenerating captions with BLIP...")
    captions = generate_captions_batch(
        processor,
        model,
        images_list,
        device,
        batch_size=args.batch_size
    )
    
    # Add captions to dataset
    print("\nAdding captions to dataset...")
    subset = subset.add_column("generated_caption", captions)
    
    # Validate
    print("\n" + "="*50)
    print("Validation")
    print("="*50)
    print(f"Total examples: {len(subset)}")
    print(f"Columns: {subset.column_names}")
    
    # Show sample
    if len(subset) > 0:
        sample = subset[0]
        print(f"\nSample example:")
        print(f"  Image: {type(sample['image'])}")
        print(f"  Emotion: {sample.get('emotion', 'N/A')}")
        print(f"  Generated Caption: {sample['generated_caption']}")
    
    # Save dataset
    print("\n" + "="*50)
    print(f"Saving dataset to {args.output_dir}...")
    print("="*50)
    os.makedirs(args.output_dir, exist_ok=True)
    subset.save_to_disk(args.output_dir)
    print(f"Dataset saved successfully!")
    
    # Also save as JSON for easy inspection
    json_path = os.path.join(args.output_dir, "sample.json")
    if len(subset) > 0:
        sample_dict = {
            "total_examples": len(subset),
            "columns": subset.column_names,
            "sample": {
                "emotion": str(subset[0].get('emotion', 'N/A')),
                "caption": subset[0]['generated_caption']
            }
        }
        import json
        with open(json_path, 'w') as f:
            json.dump(sample_dict, f, indent=2)
        print(f"Sample info saved to {json_path}")


if __name__ == "__main__":
    main()

