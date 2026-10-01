#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 4. Prediction ---
echo "Step 4: Running prediction to generate submission files..."

# Find the best checkpoint for the detection model
# The name is based on the 'save_best' key in the config's CheckpointHook
DETECTION_CHECKPOINT=$(find work_dirs/dino_swinl_vacantlot_detection -name "best_coco_bbox*.pth" | head -n 1)
if [ -z "$DETECTION_CHECKPOINT" ]; then
    echo "Error: Could not find best detection checkpoint in work_dirs/dino_swinl_vacantlot_detection/"
    exit 1
fi
echo "Found detection checkpoint: $DETECTION_CHECKPOINT"

# Find the best checkpoint for the segmentation model
# The name is based on the 'save_best' key in the config's CheckpointHook
SEGMENTATION_CHECKPOINT=$(find work_dirs/mask2former_swinl_vacantlot_seg -name "best_val_mDice*.pth" | head -n 1)
if [ -z "$SEGMENTATION_CHECKPOINT" ]; then
    echo "Error: Could not find best segmentation checkpoint in work_dirs/mask2former_swinl_vacantlot_seg/"
    exit 1
fi
echo "Found segmentation checkpoint: $SEGMENTATION_CHECKPOINT"

# Run prediction scripts
python3.11 src/predict_detection.py configs/dino_swinl_finetune_vacantlot.py $DETECTION_CHECKPOINT
python3.11 src/predict_segmentation.py configs/mask2former_swinl_finetune_vacantlot_seg.py $SEGMENTATION_CHECKPOINT
echo "Prediction complete. Output files are in the 'outputs' directory."

# --- 5. Packaging ---
echo "Step 5: Creating submission.zip..."
# Ensure the outputs directory exists before trying to zip
if [ -d "outputs" ] && [ -f "outputs/bbox.json" ] && [ -f "outputs/segmentation.json" ]; then
    zip -j submission.zip outputs/bbox.json outputs/segmentation.json
    echo "submission.zip created successfully."
else
    echo "Error: Could not find bbox.json or segmentation.json in the 'outputs' directory."
    exit 1
fi

echo "--- Pipeline Finished ---"
