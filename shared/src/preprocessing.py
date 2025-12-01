"""
Preprocessing script to generate captions for EmoSet-118K using BLIP.
Downloads full dataset first, then samples a subset and generates captions.
"""

import os
import argparse
import torch
from datasets import load_dataset, Dataset, load_from_disk
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from tqdm import tqdm
import random

# Set HuggingFace cache directory
STORAGE_BASE = "/Data/yash.bhardwaj/EmotionalPortraitsGeneration"
CACHE_DIR = os.path.join(STORAGE_BASE, "cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["HF_DATASETS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "datasets")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(CACHE_DIR, "huggingface", "transformers")


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
    
    for i in tqdm(range(0, len(images), batch_size), desc="  Generating captions", leave=False):
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
        description="Generate captions for EmoSet-118K using BLIP. Downloads full dataset first, then samples subset."
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
        help="Number of images to sample and process"
    )
    parser.add_argument(
        "--full_dataset_dir",
        type=str,
        default=os.path.join(STORAGE_BASE, "Datasets", "emoset_full"),
        help="Directory to save/load full dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(STORAGE_BASE, "Datasets", "emoset_captioned_10k"),
        help="Output directory for processed subset dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for caption generation (images processed per GPU batch)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=None,
        help="Number of images to process in each chunk (default: batch_size * 10). Lower if memory issues occur."
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
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Force re-download even if full dataset exists (default: auto-load if exists)"
    )
    
    args = parser.parse_args()
    
    # Set seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Setup device
    device = setup_device()
    
    # Step 1: Download full dataset (or load if exists)
    print("\n" + "="*50)
    print("Step 1: Loading full EmoSet-118K dataset...")
    print("="*50)
    
    full_dataset_path = args.full_dataset_dir
    
    # Check if full dataset already exists on disk
    if os.path.exists(full_dataset_path) and not args.skip_download:
        print(f"Full dataset found at {full_dataset_path}, loading from disk...")
        try:
            full_dataset = load_from_disk(full_dataset_path)
            print(f"Loaded {len(full_dataset)} examples from disk")
        except Exception as e:
            print(f"Failed to load from disk: {e}")
            print("Downloading full dataset...")
            full_dataset = load_dataset(args.dataset_name, split=args.split)
            print(f"Downloaded {len(full_dataset)} examples")
            os.makedirs(full_dataset_path, exist_ok=True)
            full_dataset.save_to_disk(full_dataset_path)
            print(f"Saved full dataset to {full_dataset_path}")
    else:
        if args.skip_download:
            print("--skip_download flag set, but dataset not found. Downloading anyway...")
        print(f"Downloading full dataset (this may take a while)...")
        full_dataset = load_dataset(args.dataset_name, split=args.split)
        print(f"Downloaded {len(full_dataset)} examples")
        os.makedirs(full_dataset_path, exist_ok=True)
        full_dataset.save_to_disk(full_dataset_path)
        print(f"Saved full dataset to {full_dataset_path}")
    
    # Step 2: Sample subset from full dataset
    print("\n" + "="*50)
    print(f"Step 2: Sampling {args.subset_size} images from {len(full_dataset)} total images...")
    print("="*50)
    
    # Shuffle indices and select subset
    indices = list(range(len(full_dataset)))
    random.shuffle(indices)
    selected_indices = indices[:args.subset_size]
    
    # Create subset dataset
    subset = full_dataset.select(selected_indices)
    print(f"Selected {len(subset)} examples")
    
    # Print emotion distribution in subset
    if len(subset) > 0:
        emotion_counts = {}
        for example in subset:
            emotion = example.get('emotion', 'amusement')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        print("\nEmotion distribution in subset:")
        for emotion, count in sorted(emotion_counts.items()):
            print(f"  {emotion}: {count} examples")
    
    # Step 3: Load BLIP model
    print("\n" + "="*50)
    print(f"Step 3: Loading BLIP model: {args.model_name}")
    print("="*50)
    processor = BlipProcessor.from_pretrained(args.model_name)
    model = BlipForConditionalGeneration.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    print("BLIP model loaded successfully")
    
    # Step 4: Generate captions
    print("\n" + "="*50)
    print("Step 4: Generating captions...")
    print("="*50)
    print(f"Processing {len(subset)} images in chunks to avoid memory issues...")
    
    generated_captions = []
    
    # Process images in chunks to avoid loading all into memory
    # Process in chunks of chunk_size images at a time
    if args.chunk_size is None:
        chunk_size = args.batch_size * 10  # Process 10 batches worth at a time
    else:
        chunk_size = args.chunk_size
    total_chunks = (len(subset) + chunk_size - 1) // chunk_size
    
    print(f"Processing in {total_chunks} chunks of ~{chunk_size} images each")
    print(f"Using batch size: {args.batch_size} for BLIP caption generation\n")
    
    for chunk_idx in range(total_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, len(subset))
        chunk_subset = subset.select(range(start_idx, end_idx))
        
        print(f"Processing chunk {chunk_idx + 1}/{total_chunks} (images {start_idx}-{end_idx-1})...")
        
        # Load images for this chunk only
        images_list = []
        for example in tqdm(chunk_subset, desc=f"  Loading images", leave=False):
            image = example['image']
            if not isinstance(image, Image.Image):
                if isinstance(image, str):
                    image = Image.open(image).convert('RGB')
                else:
                    # Try to convert if it's already an array
                    image = Image.fromarray(image).convert('RGB')
            images_list.append(image)
        
        # Generate captions for this chunk
        chunk_captions = generate_captions_batch(
            processor,
            model,
            images_list,
            device,
            batch_size=args.batch_size
        )
        
        generated_captions.extend(chunk_captions)
        num_processed = len(chunk_captions)
        
        # Clear memory
        del images_list
        del chunk_captions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"  ✓ Processed {num_processed} images in chunk {chunk_idx + 1}")
    
    # Add captions to dataset
    print(f"\nAdding {len(generated_captions)} captions to dataset...")
    subset = subset.add_column("generated_caption", generated_captions)
    
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
    
    # Step 5: Save final dataset
    print("\n" + "="*50)
    print(f"Step 5: Saving final dataset to {args.output_dir}...")
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

