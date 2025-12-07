# EmotionalPortraitsGeneration

## 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Dataset Preparation

Preprocess RafDB dataset

```bash
python prepare_rafdb_for_lora.py
```

Get captions for Emoset

```bash
python generate_captions.py \
  --dataset_name "Woleek/EmoSet-118K" \
  --model blip2 \
  --split train \
  --output_dir path_to_output_dir \
  --no_emotion \
  --cache_dir path_to_cache
```

## 3. Run fine-tuning

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="output/exp_emoset" # put your exp name
export CACHE_DIR="cache"
```

### 3.1. Portrait generation

```bash
accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir="./data/rafdb_imagefolder" \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir=${OUTPUT_DIR} \
  --validation_prompt="happiness" \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --report_to=wandb \
  --dataset="rafdb" \
```

### 3.2. EmoSet images generation

```bash
accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=Woleek/EmoSet-118K \
  --cache_dir=$CACHE_DIR \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --report_to=wandb \
  --caption_column=emotion \
  --validation_prompt="amusement" \
  --dataset="emoset" \
```

### 3.3. EmoSet images generation with Captions and Emotion Embedding

```bash
accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=Woleek/EmoSet-118K \
  --cache_dir=$CACHE_DIR \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --report_to=wandb \
  --caption_column=prompt \
  --validation_prompt='soldiers walking on the street' \
  --dataset="emoset" \
  --emotion_condition \
```

### 3.3. EmoSet images generation with Captions and Emotion Embedding + Classifier

```bash
accelerate launch --mixed_precision="bf16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=Woleek/EmoSet-118K \
  --cache_dir=$CACHE_DIR \
  --dataloader_num_workers=4 \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=2 \
  --max_train_steps=5000 \
  --learning_rate=1e-04 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=500 \
  --output_dir=$OUTPUT_DIR \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --report_to=wandb \
  --caption_column=prompt \
  --validation_prompt='soldiers walking on the street' \
  --dataset="emoset" \
  --emotion_condition \
  --emo_classifier
```

## 4. Evaluation

## 4. Inference the model

Got to `inference.ipynb`, load the base model and add fine-tuned parameters, generate pictures.