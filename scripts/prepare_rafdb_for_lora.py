import os
import pandas as pd
import shutil
import json
import kagglehub

def convert_to_imagefolder(root_dir):
    """Convert RAF-DB to imagefolder format with metadata"""
    
    emotions = {
        1: "surprise", 2: "fear", 3: "disgust",
        4: "happy", 5: "sad", 6: "anger", 7: "neutral"
    }
    
    output_dir = "./data/rafdb_imagefolder"
    
    for split in ["train", "test"]:
        labels_path = os.path.join(root_dir, f"{split}_labels.csv")
        dataset_dir = os.path.join(root_dir, "DATASET", split)
        output_split_dir = os.path.join(output_dir, split)
        
        # Read labels
        df = pd.read_csv(labels_path)
        df.columns = ["filename", "label"]
        
        # Create metadata
        metadata = []
        
        for idx, row in df.iterrows():
            emotion = emotions[row["label"]]
            src_path = os.path.join(dataset_dir, str(row["label"]), row["filename"])
            
            # Create emotion subdirectory
            emotion_dir = os.path.join(output_split_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Copy image
            dst_path = os.path.join(emotion_dir, row["filename"])
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
            
            # Add metadata entry
            metadata.append({
                "file_name": os.path.join(emotion, row["filename"]),
                "prompt": f"{emotion}"
            })
        
        # Save metadata
        metadata_path = os.path.join(output_split_dir, "metadata.jsonl")
        with open(metadata_path, "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")
    
    print(f"Conversion complete! Dataset saved to {output_dir}")


cache_path = kagglehub.dataset_download("shuvoalok/raf-db-dataset")
# Run conversion
convert_to_imagefolder(cache_path)