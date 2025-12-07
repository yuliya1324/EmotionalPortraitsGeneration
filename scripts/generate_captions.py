"""
Generate image captions for EmoSet-118K dataset using open-source vision-language models.
Supports multiple captioning models: BLIP-2, LLaVA, and InstructBLIP.
"""

import torch
from datasets import load_dataset
from transformers import (
    Blip2Processor, Blip2ForConditionalGeneration,
    InstructBlipProcessor, InstructBlipForConditionalGeneration,
    AutoProcessor, LlavaForConditionalGeneration
)
from PIL import Image
from tqdm import tqdm
import argparse
import json
import os


def load_captioning_model(model_name="blip2", device="cuda"):
    """
    Load a vision-language model for image captioning.
    
    Args:
        model_name: One of ["blip2", "instructblip", "llava"]
        device: Device to load model on
    
    Returns:
        processor, model
    """
    print(f"Loading {model_name} model...")
    
    if model_name == "blip2":
        # BLIP-2: Fast and accurate
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
        model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            torch_dtype=torch.float16,
            device_map="auto",
        )
    
    elif model_name == "instructblip":
        # InstructBLIP: Follows instructions well
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
        model = InstructBlipForConditionalGeneration.from_pretrained(
            "Salesforce/instructblip-vicuna-7b",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    elif model_name == "llava":
        # LLaVA: Best quality but slower
        processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/llava-1.5-7b-hf",
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    print(f"✓ Model loaded on {device}")
    return processor, model


def generate_caption(image, processor, model, model_name="blip2", prompt=None):
    """
    Generate a caption for an image.
    
    Args:
        image: PIL Image
        processor: Model processor
        model: Captioning model
        model_name: Name of the model being used
        prompt: Optional instruction/prompt for the model
    
    Returns:
        Generated caption string
    """
    # Default prompts for different models
    if prompt is None:
        if model_name == "blip2":
            prompt = "a photo of"
        elif model_name == "instructblip":
            prompt = "Describe this image in detail."
        elif model_name == "llava":
            prompt = "USER: <image>\nDescribe this image in detail.\nASSISTANT:"
    
    # Process inputs
    if model_name == "llava":
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    else:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device, torch.float16)
    
    # Generate caption
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            num_beams=5,
        )
    
    # Decode caption
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    # Clean up LLaVA output
    if model_name == "llava":
        caption = caption.split("ASSISTANT:")[-1].strip()
    
    return caption


def add_emotion_context(caption, emotion_label):
    """
    Add emotion context to the caption.
    
    Args:
        caption: Original caption
        emotion_label: Emotion name (e.g., "happiness", "sadness")
    
    Returns:
        Caption with emotion context
    """
    emotion_map = {
        0: "amusement", 1: "anger", 2: "awe", 3: "contentment",
        4: "disgust", 5: "excitement", 6: "fear", 7: "sadness"
    }
    
    # Convert numeric label to text if needed
    if isinstance(emotion_label, int):
        emotion_label = emotion_map.get(emotion_label, "neutral")
    
    # Add emotion context to caption
    return f"{caption}, showing {emotion_label}"


def process_dataset(
    dataset_name="Woleek/EmoSet-118K",
    output_dir="./data/emoset_with_captions",
    cache_dir="./cache",
    model_name="blip2",
    add_emotion=True,
    split="train",
    max_samples=None,
    batch_size=1
):
    """
    Process EmoSet dataset and generate captions.
    
    Args:
        dataset_name: HuggingFace dataset name
        output_dir: Where to save the captioned dataset
        model_name: Captioning model to use
        add_emotion: Whether to add emotion context to captions
        split: Dataset split to process
        max_samples: Limit number of samples (for testing)
        batch_size: Processing batch size
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading {dataset_name}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    
    if split not in dataset:
        print(f"Available splits: {list(dataset.keys())}")
        split = "train"
    
    data = dataset[split]
    
    # Limit samples if specified
    if max_samples:
        data = data.select(range(min(max_samples, len(data))))
    
    print(f"Processing {len(data)} samples from {split} split...")
    
    # Load captioning model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor, model = load_captioning_model(model_name, device)
    
    # Process images
    captions = []
    errors = []
    
    for idx in tqdm(range(0, len(data), batch_size), desc="Generating captions"):
        try:
            # Get batch
            batch_end = min(idx + batch_size, len(data))
            
            for i in range(idx, batch_end):
                item = data[i]
                image = item["image"]
                
                # Handle different possible emotion column names
                emotion = item.get("emotion", item.get("label", 0))
                
                # Generate caption
                caption = generate_caption(image, processor, model, model_name)
                
                # Add emotion context if requested
                if add_emotion:
                    caption = add_emotion_context(caption, emotion)
                
                captions.append({
                    "idx": i,
                    "caption": caption,
                    "emotion": emotion,
                    "original_caption": item.get("caption", "")
                })
        
        except Exception as e:
            print(f"\nError processing sample {idx}: {e}")
            errors.append({"idx": idx, "error": str(e)})
            captions.append({
                "idx": idx,
                "caption": "",
                "emotion": data[idx].get("emotion", 0),
                "error": str(e)
            })
    
    # Save captions
    output_file = os.path.join(output_dir, f"{split}_captions.json")
    with open(output_file, "w") as f:
        json.dump(captions, f, indent=2)
    
    print(f"\n✓ Captions saved to {output_file}")
    print(f"Successfully processed: {len(captions) - len(errors)}/{len(data)}")
    
    if errors:
        error_file = os.path.join(output_dir, f"{split}_errors.json")
        with open(error_file, "w") as f:
            json.dump(errors, f, indent=2)
        print(f"Errors saved to {error_file}")
    
    # Save sample captions for inspection
    print("\nSample captions:")
    for i in range(min(5, len(captions))):
        if captions[i].get("caption"):
            print(f"\n{i+1}. Emotion: {captions[i]['emotion']}")
            print(f"   Caption: {captions[i]['caption']}")
    
    return captions


def merge_captions_with_dataset(
    dataset_name="Woleek/EmoSet-118K",
    captions_dir="./data/emoset_with_captions",
    output_dir="./data/emoset_captioned",
    split="train",
    cache_dir="./cache"
):
    """
    Merge generated captions back into the dataset.
    
    Args:
        dataset_name: Original dataset name
        captions_dir: Directory containing caption JSON files
        output_dir: Where to save the merged dataset
        split: Dataset split to merge
    """
    print(f"Merging captions with {split} split...")
    
    # Load dataset
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    data = dataset[split]
    
    # Load captions
    captions_file = os.path.join(captions_dir, f"{split}_captions.json")
    with open(captions_file, "r") as f:
        captions = json.load(f)
    
    # Create caption mapping
    caption_map = {item["idx"]: item["caption"] for item in captions}
    
    # Add captions to dataset
    def add_caption(example, idx):
        example["generated_caption"] = caption_map.get(idx, "")
        return example
    
    data = data.map(add_caption, with_indices=True)
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    data.save_to_disk(os.path.join(output_dir, split))
    
    print(f"✓ Merged dataset saved to {output_dir}/{split}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for EmoSet dataset")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="Woleek/EmoSet-118K",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/emoset_with_captions",
        help="Output directory for captions"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="blip2",
        choices=["blip2", "instructblip", "llava"],
        help="Captioning model to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to process"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Processing batch size"
    )
    parser.add_argument(
        "--no_emotion",
        action="store_true",
        help="Don't add emotion context to captions"
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge captions back into dataset after generation"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Cache directory"
    )
    
    args = parser.parse_args()
    
    # Generate captions
    captions = process_dataset(
        dataset_name=args.dataset_name,
        output_dir=args.output_dir,
        model_name=args.model,
        add_emotion=not args.no_emotion,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        cache_dir=args.cache_dir,
    )
    
    # Optionally merge with dataset
    if args.merge:
        merge_captions_with_dataset(
            dataset_name=args.dataset_name,
            captions_dir=args.output_dir,
            output_dir=args.output_dir,
            split=args.split,
            cache_dir=args.cache_dir,
        )