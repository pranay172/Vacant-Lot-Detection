# src/preprocess_train_detection.py
import json
import os
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil
import argparse
from tqdm import tqdm

def convert_tiff_to_png(input_dir: Path, output_dir: Path):
    """Converts all TIFF images in input_dir to PNG format in output_dir."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No .tif files found in {input_dir}")
        return

    print(f"Converting {len(tiff_files)} TIFF images to PNG...")
    for tiff_file in tqdm(tiff_files):
        try:
            img = Image.open(tiff_file)
            # Ensure it's RGB if not already
            if img.mode != 'RGB':
                img = img.convert('RGB')
            png_file = output_dir / (tiff_file.stem + ".png")
            img.save(png_file, "PNG")
        except Exception as e:
            print(f"Error converting {tiff_file}: {e}")

def competition_to_coco(original_json_path: Path, image_base_path: Path, output_coco_path: Path, image_filenames: list):
    """
    Converts the competition's JSON annotation format to COCO format.
    Only includes images specified in image_filenames.
    Assumes images are already converted to PNG and dimensions can be read.
    """
    with open(original_json_path, 'r') as f:
        data = json.load(f)

    coco_output = {
        "info": {
            "description": "Vacant Lot Dataset",
            "year": 2025,
            "contributor": "Solafune",
            "date_created": "2025/07/04"
        },
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "vacant_lot", "supercategory": "land"}] # MMDetection often uses 0-indexed categories
    }

    annotation_id = 0
    image_id_map = {} # To map original filename to new COCO image_id

    print(f"Processing annotations for {len(image_filenames)} images...")
    current_image_id = 0
    for entry in tqdm(data["images"]):
        original_file_name = entry["file_name"]
        # Check if this image is in the desired subset (train or val)
        if original_file_name not in image_filenames:
            continue

        # Assumes images are converted to PNG for easier dimension reading
        # If using original TIFFs, adjust path and reading method
        png_file_name = original_file_name.replace(".tif", ".png")
        img_path = image_base_path / png_file_name
        
        if not img_path.exists():
            print(f"Warning: Image file {img_path} not found. Skipping.")
            continue

        # Get image dimensions (use PIL to be safe with various image formats)
        try:
            with Image.open(img_path) as img:
                width, height = img.size
        except Exception as e:
            print(f"Warning: Could not open or read dimensions for {img_path}. Error: {e}. Skipping.")
            continue
        
        # Use a consistent image ID for COCO
        coco_img_id = current_image_id
        image_id_map[original_file_name] = coco_img_id
        current_image_id += 1

        coco_output["images"].append({
            "id": coco_img_id,
            "file_name": png_file_name, # Use PNG filename
            "width": width,
            "height": height
        })

        for ann in entry["annotations"]:
            if ann["class"] == "vacant_lot":
                x_min, y_min, w, h = ann["bbox"]
                coco_output["annotations"].append({
                    "id": annotation_id,
                    "image_id": coco_img_id,
                    "category_id": 0, # "vacant_lot"
                    "bbox": [float(x_min), float(y_min), float(w), float(h)],
                    "area": float(w * h),
                    "iscrowd": 0
                })
                annotation_id += 1
    
    if not coco_output["images"]:
        print("Warning: No images were processed into the COCO format. Check image_filenames and paths.")

    with open(output_coco_path, 'w') as f:
        json.dump(coco_output, f, indent=4)
    print(f"COCO annotation saved to {output_coco_path}")


def main(args):
    base_data_dir = Path(args.base_data_dir)
    original_images_dir = base_data_dir / "train_bbox_images"
    original_annotations_path = base_data_dir / "train_bbox_annotations.json"
    
    preprocessed_dir = base_data_dir / "preprocessed_data"
    coco_annotations_dir = preprocessed_dir / "coco_annotations"
    train_images_png_dir = preprocessed_dir / "train_images_png" # All images converted
    
    # Create directories
    for p in [preprocessed_dir, coco_annotations_dir, train_images_png_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # 1. Convert all TIFF images to PNG
    print("--- Converting TIFF to PNG ---")
    convert_tiff_to_png(original_images_dir, train_images_png_dir)

    # 2. Load original annotations and prepare for split
    with open(original_annotations_path, 'r') as f:
        all_data_info = json.load(f)
    
    all_image_filenames = [img_entry["file_name"] for img_entry in all_data_info["images"]]
    
    # 3. Split image filenames into train and validation
    # Ensure you use .tif filenames for splitting as they are in the original JSON
    train_filenames_tif, val_filenames_tif = train_test_split(
        all_image_filenames, test_size=args.val_split, random_state=args.random_seed
    )
    print(f"Training images: {len(train_filenames_tif)}, Validation images: {len(val_filenames_tif)}")

    # 4. Create COCO annotations for training and validation sets
    print("\n--- Creating COCO annotations for Training set ---")
    competition_to_coco(
        original_annotations_path, 
        train_images_png_dir, # Base path where PNGs are (or will be)
        coco_annotations_dir / "train_bbox_coco.json",
        train_filenames_tif
    )

    print("\n--- Creating COCO annotations for Validation set ---")
    competition_to_coco(
        original_annotations_path,
        train_images_png_dir, # Base path where PNGs are (or will be)
        coco_annotations_dir / "val_bbox_coco.json",
        val_filenames_tif
    )
            
    print("\nPreprocessing complete.")
    print(f"Converted images in: {train_images_png_dir}")
    print(f"COCO annotations in: {coco_annotations_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess vacant lot detection data.")
    parser.add_argument(
        "--base_data_dir", 
        type=str, 
        default="data", 
        help="Base directory where 'train_bbox_images' and 'train_bbox_annotations.json' are located."
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
