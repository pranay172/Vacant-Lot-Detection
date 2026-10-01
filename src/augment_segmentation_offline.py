# augment_segmentation_offline.py
import json
import argparse
import cv2
import shutil
import albumentations as A
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os


def main(args):
    # --- Setup Paths ---
    # Input directories (your current preprocessed data)
    original_images_dir = Path(args.images_dir)
    original_masks_dir_train = Path(args.masks_dir_train) # e.g., data/preprocessed_data/seg_masks_png/train/
    original_masks_dir_val = Path(args.masks_dir_val)     # e.g., data/preprocessed_data/seg_masks_png/val/
    
    original_list_file_train = Path(args.list_file_train) # e.g., data/preprocessed_data/train_seg_list.txt
    original_list_file_val = Path(args.list_file_val)     # e.g., data/preprocessed_data/val_seg_list.txt

    # Output directories
    output_base_dir = Path(args.output_dir)
    output_images_dir_train = output_base_dir / "seg_images_png_aug" / "train"
    output_masks_dir_train = output_base_dir / "seg_masks_png_aug" / "train"
    output_images_dir_val = output_base_dir / "seg_images_png_aug" / "val"
    output_masks_dir_val = output_base_dir / "seg_masks_png_aug" / "val"
    
    output_list_file_train = output_base_dir / "augmented_train_seg_list.txt"
    output_list_file_val = output_base_dir / "augmented_val_seg_list.txt"

    output_images_dir_train.mkdir(parents=True, exist_ok=True)
    output_masks_dir_train.mkdir(parents=True, exist_ok=True)
    output_images_dir_val.mkdir(parents=True, exist_ok=True)
    output_masks_dir_val.mkdir(parents=True, exist_ok=True)
    
    # --- Define the Augmentation Pipeline using Albumentations ---
    # Minimal geometric, focus on flips and color/photometric
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        # Color/Photometric Augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=3, p=0.5),
        A.OneOf([
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.3), p=1.0),
        ], p=0.5),
        # Optional: Slight geometric changes if desired
        A.ShiftScaleRotate(shift_limit=0.03, scale_limit=0.1, rotate_limit=5, 
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT,
                           mask_interpolation=cv2.INTER_NEAREST, fill=255, p=0.3), # fill=255 for ignore index

    ]) # No bbox_params needed for segmentation masks directly with image/mask pairs

    # --- Function to process a single dataset split (train or val) ---
    def augment_split(image_filenames, input_img_dir, input_mask_dir, 
                      output_img_dir, output_mask_dir, num_augs_per_image, is_training_set=True):
        augmented_file_entries = []
        
        print(f"Augmenting {len(image_filenames)} images in {input_img_dir}, creating {num_augs_per_image} versions of each...")

        for img_stem_from_list in tqdm(image_filenames):
            img_name_png = f"{img_stem_from_list}.png" # Reconstruct .png name from stem
            
            image_path = input_img_dir / img_name_png
            mask_path = input_mask_dir / img_name_png # Mask has the same .png name

            if not image_path.exists():
                print(f"Warning: Image not found {image_path}, skipping.")
                continue
            if not mask_path.exists():
                print(f"Warning: Mask not found {mask_path}, skipping.")
                continue
            
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"Warning: Could not read image {image_path}, skipping.")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Albu prefers RGB

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE) # Load mask as grayscale
            if mask is None:
                print(f"Warning: Could not read mask {mask_path}, skipping.")
                continue

            # Original image/mask pair (if num_augs_per_image is set to include it, or handle separately)
            # For simplicity, let's always save original if is_training_set (or could be an option)
            if is_training_set or args.augment_validation_too: # Control if validation set also gets augmented versions
                original_img_save_path = output_img_dir / img_name_png
                original_mask_save_path = output_mask_dir / img_name_png
                cv2.imwrite(str(original_img_save_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                cv2.imwrite(str(original_mask_save_path), mask)
                augmented_file_entries.append(f"{img_stem_from_list}")


            # --- Loop to create N augmented versions (N-1 if original is already counted) ---
            actual_augmentations_to_generate = num_augs_per_image
            if args.include_original_in_aug_count and (is_training_set or args.augment_validation_too):
                actual_augmentations_to_generate = num_augs_per_image -1 
            
            for i in range(actual_augmentations_to_generate):
                try:
                    # Apply the transform
                    # For masks, Albumentations uses cv2.INTER_NEAREST by default if it detects 'mask' key
                    transformed = transform(image=image, mask=mask) 
                    
                    transformed_image = transformed['image']
                    transformed_mask = transformed['mask']

                    # --- Save the new image and mask ---
                    new_img_stem = f"{img_stem_from_list}_aug_{i}"
                    img_save_path = output_img_dir / f"{new_img_stem}.png"
                    mask_save_path = output_mask_dir / f"{new_img_stem}.png"
                    
                    cv2.imwrite(str(img_save_path), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(str(mask_save_path), transformed_mask)

                    augmented_file_entries.append(f"{new_img_stem}")

                except Exception as e:
                    print(f"Error augmenting {img_name_png} (aug #{i}): {e}")
        
        return augmented_file_entries

    # --- Load original file lists ---
    def read_list_file(file_path):
        stems = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts: # Ensure line is not empty
                    stems.append(parts[0]) # First part is the stem
        return stems

    train_stems = read_list_file(original_list_file_train)
    val_stems = read_list_file(original_list_file_val)

    # --- Process Training Data ---
    print("\n--- Processing Training Set ---")
    train_augmented_entries = augment_split(
        train_stems, original_images_dir, original_masks_dir_train,
        output_images_dir_train, output_masks_dir_train, 
        args.num_augmentations, is_training_set=True
    )
    with open(output_list_file_train, 'w') as f:
        for entry in train_augmented_entries:
            f.write(entry + '\n')
    print(f"Saved augmented training list to {output_list_file_train} ({len(train_augmented_entries)} entries)")

    # --- Process Validation Data ---
    # Typically, we don't augment the validation set in the same way, or only apply minimal/deterministic ones.
    # For offline, you might choose to not augment it, or create only one "test-time like" version if any.
    # Here, we provide an option to augment it like training, or just copy.
    print("\n--- Processing Validation Set ---")
    if args.augment_validation_too:
        val_augmented_entries = augment_split(
            val_stems, original_images_dir, original_masks_dir_val,
            output_images_dir_val, output_masks_dir_val, 
            args.num_augmentations, is_training_set=False # Pass False here
        )
    else: # Just copy original validation files to new structure and create list
        print("Copying original validation files...")
        val_augmented_entries = []
        for stem in tqdm(val_stems):
            img_name_png = f"{stem}.png"
            shutil.copy2(original_images_dir / img_name_png, output_images_dir_val / img_name_png)
            shutil.copy2(original_masks_dir_val / img_name_png, output_masks_dir_val / img_name_png)
            val_augmented_entries.append(f"{stem}")
            
    with open(output_list_file_val, 'w') as f:
        for entry in val_augmented_entries:
            f.write(entry + '\n')
    print(f"Saved augmented validation list to {output_list_file_val} ({len(val_augmented_entries)} entries)")

    print("\nOffline augmentation for segmentation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform offline data augmentation for segmentation.")
    # Input paths
    parser.add_argument("--images_dir", type=str, default="data/preprocessed_data/seg_images_png/", help="Path to the directory with original PNG images for segmentation.")
    parser.add_argument("--masks_dir_train", type=str, default="data/preprocessed_data/seg_masks_png/train/", help="Path to the directory with original training PNG masks.")
    parser.add_argument("--masks_dir_val", type=str, default="data/preprocessed_data/seg_masks_png/val/", help="Path to the directory with original validation PNG masks.")
    parser.add_argument("--list_file_train", type=str, default="data/preprocessed_data/train_seg_list.txt", help="Path to the original training list file.")
    parser.add_argument("--list_file_val", type=str, default="data/preprocessed_data/val_seg_list.txt", help="Path to the original validation list file.")
    # Output paths
    parser.add_argument("--output_dir", type=str, default="data/preprocessed_data_seg_augmented/", help="Base directory to save augmented images, masks, and new list files.")
    # Augmentation parameters
    parser.add_argument("--num_augmentations", type=int, default=5, help="Number of augmented versions to create per original image (includes original if --include_original_in_aug_count is True).")
    parser.add_argument("--include_original_in_aug_count", action='store_true', help="If set, num_augmentations includes the original image. So N-1 actual new versions are made.")
    parser.add_argument("--augment_validation_too", action='store_true', help="If set, apply the same augmentation process to the validation set. Otherwise, validation files are just copied.")

    args = parser.parse_args()
    main(args)