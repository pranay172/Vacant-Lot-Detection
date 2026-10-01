#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting Vacant Lot Detection Pipeline ---"

# --- 1. Preprocessing ---
echo "Step 1: Preprocessing training data..."
python3.11 src/preprocess_train_detection.py
python3.11 src/preprocess_train_segmentation.py
echo "Preprocessing complete."

# --- 2. Offline Augmentation ---
echo "Step 2: Performing offline data augmentation..."
python3.11 src/augment_detection_offline.py
python3.11 src/augment_segmentation_offline.py
echo "Augmentation complete."