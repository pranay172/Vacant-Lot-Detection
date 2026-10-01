# src/preprocess_train_segmentation.py
import json
import os
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import argparse
from tqdm import tqdm

def convert_tiff_to_png_seg(input_dir: Path, output_dir: Path):
    """Converts all TIFF images in input_dir to PNG format in output_dir."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No .tif files found in {input_dir}")
        return

    print(f"Converting {len(tiff_files)} TIFF images to PNG for segmentation...")
    for tiff_file in tqdm(tiff_files):
        try:
            img = Image.open(tiff_file)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            png_file = output_dir / (tiff_file.stem + ".png")
            img.save(png_file, "PNG")
        except Exception as e:
            print(f"Error converting {tiff_file}: {e}")

def create_segmentation_masks(
    original_annotations_path: Path, 
    output_mask_dir: Path, 
    image_filenames_tif: list,
    all_image_dims: dict
    ):
    """
    Creates binary segmentation masks from polygon annotations.
    Masks will have 0 for background and 1 for 'vacant_lot'.
    """
    if not output_mask_dir.exists():
        output_mask_dir.mkdir(parents=True, exist_ok=True)

    with open(original_annotations_path, 'r') as f:
        data = json.load(f)

    image_name_to_annotations = {
        img_entry["file_name"]: img_entry for img_entry in data["images"]
    }
    
    print(f"Creating masks for {len(image_filenames_tif)} images in {output_mask_dir}...")
    for original_tif_filename in tqdm(image_filenames_tif):
        if original_tif_filename not in image_name_to_annotations:
            print(f"Warning: Annotations for {original_tif_filename} not found. Skipping mask creation.")
            continue
        
        img_entry = image_name_to_annotations[original_tif_filename]
        
        if original_tif_filename not in all_image_dims:
            print(f"Warning: Dimensions for {original_tif_filename} not found. Cannot create mask. Skipping.")
            continue
        
        width, height = all_image_dims[original_tif_filename]
        mask_pil = Image.new('L', (width, height), 0) # 'L' for 8-bit grayscale, 0 for background
        draw = ImageDraw.Draw(mask_pil)

        for ann in img_entry["annotations"]:
            if ann["class"] == "vacant_lot":
                polygon = ann["segmentation"]
                # Ensure polygon is a flat list of (x1, y1, x2, y2, ...)
                if not polygon or len(polygon) < 6: # Need at least 3 points for a polygon
                    print(f"Warning: Invalid polygon for {original_tif_filename}. Skipping this annotation.")
                    continue
                
                # Convert to list of tuples [(x1,y1), (x2,y2), ...] for PIL
                polygon_tuples = []
                for i in range(0, len(polygon), 2):
                    try:
                        polygon_tuples.append((int(polygon[i]), int(polygon[i+1])))
                    except (ValueError, IndexError) as e:
                        print(f"Error processing polygon point for {original_tif_filename}: {e}. Point data: {polygon[i:i+2]}")
                        break # Skip this polygon if points are malformed
                else: # if no break occurred in the loop
                    try:
                        draw.polygon(polygon_tuples, outline=1, fill=1) # Fill with 1 for vacant_lot
                    except Exception as e:
                        print(f"Error drawing polygon for {original_tif_filename}: {e}. Polygon: {polygon_tuples}")


        mask_png_filename = original_tif_filename.replace(".tif", ".png")
        mask_pil.save(output_mask_dir / mask_png_filename)


def get_image_dimensions(image_dir: Path):
    """Reads dimensions of all PNG images in a directory."""
    all_dims = {}
    png_files = list(image_dir.glob("*.png"))
    print(f"Reading dimensions for {len(png_files)} PNG images...")
    for png_file in tqdm(png_files):
        try:
            with Image.open(png_file) as img:
                # Store dimensions using the original .tif filename as key for consistency
                original_tif_name = png_file.name.replace(".png", ".tif")
                all_dims[original_tif_name] = img.size
        except Exception as e:
            print(f"Could not read dimensions for {png_file}: {e}")
    return all_dims


def create_file_list(output_file_path: Path, image_filenames_tif: list):
    """
    Creates a file list for MMSegmentation.
    """
    with open(output_file_path, 'w') as f:
        for tif_filename in image_filenames_tif:
            stem = Path(tif_filename).stem
            # Write the stem for both image and mask
            f.write(f"{stem}\n")
    print(f"File list saved to {output_file_path}")


def main(args):
    base_data_dir = Path(args.base_data_dir)
    original_images_dir_seg = base_data_dir / "train_segmentation_images"
    original_annotations_path_seg = base_data_dir / "train_segmentation_annotations.json"
    
    preprocessed_seg_dir = base_data_dir / "preprocessed_data"
    images_png_dir_seg = preprocessed_seg_dir / "seg_images_png"
    masks_png_dir = preprocessed_seg_dir / "seg_masks_png" # Base for train/val masks
    train_masks_dir = masks_png_dir / "train"
    val_masks_dir = masks_png_dir / "val"
    
    # Create directories
    for p_dir in [preprocessed_seg_dir, images_png_dir_seg, masks_png_dir, train_masks_dir, val_masks_dir]:
        p_dir.mkdir(parents=True, exist_ok=True)

    # 1. Convert all TIFF images to PNG
    print("--- Converting Segmentation TIFFs to PNG ---")
    convert_tiff_to_png_seg(original_images_dir_seg, images_png_dir_seg)

    # 2. Get dimensions of converted PNG images (needed for mask creation)
    # The keys in all_image_dims will be the original .tif filenames
    all_image_dims = get_image_dimensions(images_png_dir_seg)
    if not all_image_dims:
        print("Error: Could not read dimensions for any image. Masks cannot be created. Exiting.")
        return

    # 3. Filter out images with invalid shapes (where a dimension is 1)
    valid_dims = {}
    invalid_count = 0
    for fname, (w, h) in all_image_dims.items():
        if w <= 1 or h <= 1:
            print(f"Excluding image {fname} due to invalid shape: ({w}, {h})")
            invalid_count += 1
        else:
            valid_dims[fname] = (w, h)
    
    if invalid_count > 0:
        print(f"Excluded {invalid_count} images with invalid shapes.")
    all_image_dims = valid_dims # Continue with only valid images

    # 4. Load original annotations to get list of all .tif filenames involved
    with open(original_annotations_path_seg, 'r') as f:
        all_seg_data_info = json.load(f)
    all_image_filenames_tif_seg = [img_entry["file_name"] for img_entry in all_seg_data_info["images"]]
    
    # Filter out filenames for which dimensions couldn't be read or were invalid
    valid_image_filenames_tif = [f_name for f_name in all_image_filenames_tif_seg if f_name in all_image_dims]
    if len(valid_image_filenames_tif) != len(all_image_filenames_tif_seg):
        print(f"Warning: Some images mentioned in annotations could not be read or their dimensions are missing/invalid. "
              f"Original count: {len(all_image_filenames_tif_seg)}, Valid for mask creation: {len(valid_image_filenames_tif)}")


    # 5. Split image filenames into train and validation
    train_filenames_tif, val_filenames_tif = train_test_split(
        valid_image_filenames_tif, test_size=args.val_split, random_state=args.random_seed
    )
    print(f"Training images for segmentation: {len(train_filenames_tif)}")
    print(f"Validation images for segmentation: {len(val_filenames_tif)}")

    # 6. Create segmentation masks for training and validation sets
    print("\n--- Creating Segmentation Masks for Training set ---")
    create_segmentation_masks(original_annotations_path_seg, train_masks_dir, train_filenames_tif, all_image_dims)

    print("\n--- Creating Segmentation Masks for Validation set ---")
    create_segmentation_masks(original_annotations_path_seg, val_masks_dir, val_filenames_tif, all_image_dims)
    
    # 7. Create file lists for MMSegmentation
    print("\n--- Creating File Lists for MMSegmentation ---")
    create_file_list(preprocessed_seg_dir / "train_seg_list.txt", train_filenames_tif)
    create_file_list(preprocessed_seg_dir / "val_seg_list.txt", val_filenames_tif)
    
    print("\nPreprocessing for segmentation complete.")
    print(f"Converted segmentation images in: {images_png_dir_seg}")
    print(f"Training masks in: {train_masks_dir}")
    print(f"Validation masks in: {val_masks_dir}")
    print(f"Training file list: {preprocessed_seg_dir / 'train_seg_list.txt'}")
    print(f"Validation file list: {preprocessed_seg_dir / 'val_seg_list.txt'}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess vacant lot segmentation data.")
    parser.add_argument(
        "--base_data_dir", 
        type=str, 
        default="data", 
        help="Base directory where 'train_segmentation_images' and 'train_segmentation_annotations.json' are located."
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=0.1, 
        help="Fraction of data to use for validation."
    )
    parser.add_argument(
        "--random_seed", 
        type=int, 
        default=42, 
        help="Random seed for train/val split."
    )
    args = parser.parse_args()
    main(args)
