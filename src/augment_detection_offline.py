import json
import argparse
import cv2
import albumentations as A
from pathlib import Path
from tqdm import tqdm

def main(args):
    # --- Setup Paths ---
    original_coco_path = Path(args.coco_path)
    original_images_dir = Path(args.images_dir)
    
    output_dir = Path(args.output_dir)
    output_images_dir = output_dir / "augmented_images_png"
    output_coco_path = output_dir / "augmented_coco.json"
    
    output_images_dir.mkdir(parents=True, exist_ok=True)
    
    # --- Define the Augmentation Pipeline using Albumentations ---
    transform = A.Compose([
        # Geometric Augmentations
        A.HorizontalFlip(p=0.5),
        # Using a safer resize and crop strategy for offline generation
        A.RandomSizedBBoxSafeCrop(width=1012, height=1012, erosion_rate=0.2, p=0.5),
        A.Resize(width=1333, height=800, p=1.0), # Resize all to a standard size
        # Photometric/Color Augmentations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
        A.OneOf([
            A.GaussNoise(std_range=(0.1, 0.3), p=1.0),
            A.Blur(blur_limit=3, p=1.0)
        ], p=0.5),
    ], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

    # --- Load Original COCO Data ---
    print(f"Loading original COCO annotations from {original_coco_path}")
    with open(original_coco_path, 'r') as f:
        coco_data = json.load(f)
        
    # --- Prepare New COCO structure ---
    augmented_coco = {
        "images": [],
        "annotations": [],
        "categories": coco_data['categories']
    }
    
    new_image_id = 0
    new_annotation_id = 0

    print(f"Augmenting {len(coco_data['images'])} images, creating {args.num_augmentations} versions of each...")
    
    # Process each image from the original COCO file
    for image_info in tqdm(coco_data['images']):
        image_path = original_images_dir / image_info['file_name']
        if not image_path.exists():
            print(f"Warning: Image not found {image_path}, skipping.")
            continue
            
        # Read the image. Albumentations works best with NumPy arrays (loaded via OpenCV).
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Convert to RGB
        
        # Get all annotations for the current image
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_info['id']]
        bboxes = [ann['bbox'] for ann in annotations]
        # 'category_ids' is required by the bbox_params
        category_ids = [ann['category_id'] for ann in annotations]

        # --- Loop to create N augmented versions ---
        for i in range(args.num_augmentations):
            try:
                # Apply the transform
                transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                
                transformed_image = transformed['image']
                transformed_bboxes = transformed['bboxes']
                transformed_category_ids = transformed['category_ids']

                if not transformed_bboxes: # Skip if all bboxes were cropped out
                    continue

                # --- Save the new image ---
                new_file_name = f"{Path(image_info['file_name']).stem}_aug_{i}.png"
                save_path = output_images_dir / new_file_name
                # Convert back to BGR for OpenCV saving
                cv2.imwrite(str(save_path), cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR))

                # --- Add entry to the new COCO file ---
                h, w, _ = transformed_image.shape
                augmented_coco['images'].append({
                    "id": new_image_id,
                    "file_name": new_file_name,
                    "width": w,
                    "height": h
                })
                
                # Add corresponding annotations
                for bbox, cat_id in zip(transformed_bboxes, transformed_category_ids):
                    x_min, y_min, width, height = bbox
                    augmented_coco['annotations'].append({
                        "id": new_annotation_id,
                        "image_id": new_image_id,
                        "category_id": cat_id,
                        "bbox": [float(x_min), float(y_min), float(width), float(height)],
                        "area": float(width * height),
                        "iscrowd": 0
                    })
                    new_annotation_id += 1
                
                new_image_id += 1

            except Exception as e:
                print(f"Error augmenting {image_info['file_name']} (aug #{i}): {e}")

    # --- Save the final augmented COCO JSON ---
    print(f"Saving new COCO annotations to {output_coco_path}")
    with open(output_coco_path, 'w') as f:
        json.dump(augmented_coco, f, indent=4)
        
    print("\nOffline augmentation complete!")
    print(f"Generated {len(augmented_coco['images'])} new images in {output_images_dir}")
    print(f"Generated {len(augmented_coco['annotations'])} new annotations in {output_coco_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform offline data augmentation for object detection.")
    parser.add_argument("--coco_path", type=str, default="data/preprocessed_data/coco_annotations/train_bbox_coco.json", help="Path to the original COCO training JSON file.")
    parser.add_argument("--images_dir", type=str, default="data/preprocessed_data/train_images_png/", help="Path to the directory with original PNG images.")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed_data_augmented/", help="Directory to save augmented images and new COCO file.")
    parser.add_argument("--num_augmentations", type=int, default=5, help="Number of augmented versions to create per original image.")
    args = parser.parse_args()
    main(args)
