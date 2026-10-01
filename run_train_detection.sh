#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- 3. Training Detection Model (DINO Swin-L) ---
echo "Step 3.1: Training Detection Model (DINO Swin-L)..."
python3.11 src/train_detection.py configs/dino_swinl_finetune_vacantlot.py --work-dir work_dirs/dino_swinl_vacantlot_detection
echo "Detection model training complete."
