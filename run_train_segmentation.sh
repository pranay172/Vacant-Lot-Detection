#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 3. Training Segmentation Model (Mask2Former Swin-L) ---
echo "Step 3.2: Training Segmentation Model (Mask2Former Swin-L)..."
python3.11 src/train_segmentation.py configs/mask2former_swinl_finetune_vacantlot_seg.py --work-dir work_dirs/mask2former_swinl_vacantlot_seg
echo "Segmentation model training complete."