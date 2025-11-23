#!/bin/bash
# Script to download full dataset, sample 10K images, and train the model

set -e  # Exit on error

echo "=========================================="
echo "Emotional Scene Generation - Full Pipeline"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="/Data/yash.bhardwaj"
FULL_DATASET_DIR="${DATA_DIR}/datasets/emoset_full"
OUTPUT_DATASET_DIR="${DATA_DIR}/datasets/emoset_captioned_10k"
SUBSET_SIZE=10000

# Step 1: Preprocessing - Download full dataset and sample 10K
echo "Step 1: Preprocessing (Download full dataset + Sample 10K + Generate captions)"
echo "-------------------------------------------------------------------"
echo "This will:"
echo "  1. Download the full EmoSet-118K dataset to ${FULL_DATASET_DIR}"
echo "  2. Sample ${SUBSET_SIZE} images from the full dataset"
echo "  3. Generate captions using BLIP"
echo "  4. Save the processed dataset to ${OUTPUT_DATASET_DIR}"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

python src/preprocessing.py \
    --subset_size ${SUBSET_SIZE} \
    --full_dataset_dir "${FULL_DATASET_DIR}" \
    --output_dir "${OUTPUT_DATASET_DIR}" \
    --batch_size 8 \
    --seed 42

echo ""
echo "✓ Preprocessing completed!"
echo ""

# Step 2: Training
echo "Step 2: Training"
echo "-------------------------------------------------------------------"
echo "This will train the model with the 10K sampled images."
echo ""
read -p "Continue to training? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training skipped. You can run it later with:"
    echo "  python src/train.py --data_dir ${OUTPUT_DATASET_DIR}"
    exit 0
fi

python src/train.py \
    --data_dir "${OUTPUT_DATASET_DIR}" \
    --batch_size 4 \
    --num_epochs 10 \
    --save_steps 500 \
    --validation_steps 500

echo ""
echo "=========================================="
echo "✓ Full pipeline completed!"
echo "=========================================="

