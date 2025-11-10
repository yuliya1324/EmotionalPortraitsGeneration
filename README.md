# EmotionalPortraitsGeneration

## 1. Setup environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 2. Preprocess dataset

```bash
python prepare_rafdb_for_lora.py
```

You can check and play with the data in `data_exp.ipynb`

## 3. Run fine-tuning

```bash
export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export OUTPUT_DIR="./output"
export HUB_MODEL_ID="portraits-lora"
export DATASET_DIR="./data/rafdb_imagefolder"

accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_DIR \
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
  --validation_prompt="sadness" \
  --validation_prompt="anger" \
  --num_validation_images=4 \
  --validation_epochs=5 \
  --checkpointing_steps=500 \
  --seed=1337 \
  --report_to=wandb
```

## 4. Inference the model

Got to `inference.ipynb`, load the base model and add fine-tuned parameters, generate pictures.