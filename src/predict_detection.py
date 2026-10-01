# src/predict_detection.py
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import mmcv # For MMDetection < v3.0
from PIL import Image # For image conversion if needed

# For MMDetection v3.x
from mmdet.apis import DetInferencer # New, simplified inference API

def convert_evaluation_images(input_dir: Path, output_dir: Path):
    """Converts evaluation TIFF images to PNG, if necessary."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tif"))
    if not tiff_files:
        print(f"No .tif files found in evaluation directory {input_dir}")
        return False # Indicate no files processed

    converted_any = False
    print(f"Converting {len(tiff_files)} evaluation TIFF images to PNG...")
    for tiff_file in tqdm(tiff_files):
        try:
            img_pil = Image.open(tiff_file)
            if img_pil.mode != 'RGB':
                img_pil = img_pil.convert('RGB')
            png_file = output_dir / (tiff_file.stem + ".png")
            img_pil.save(png_file, "PNG")
            converted_any = True
        except Exception as e:
            print(f"Error converting {tiff_file}: {e}")
    return converted_any


def main(args):
    config_file = args.config
    checkpoint_file = args.checkpoint
    eval_images_dir_orig = Path(args.eval_images_dir)
    output_json_path = Path(args.output_json)
    device = args.device
    score_thr = args.score_thr

    # It's good practice to convert evaluation TIFFs to PNG first, like training data
    # Let's assume evaluation images are also TIFFs and need conversion
    preprocessed_eval_dir = eval_images_dir_orig.parent / f"{eval_images_dir_orig.name}_png"
    
    print(f"Checking for TIFFs in {eval_images_dir_orig} for conversion to {preprocessed_eval_dir}...")
    convert_evaluation_images(eval_images_dir_orig, preprocessed_eval_dir)
    
    # Use the directory with PNGs for inference
    eval_images_dir_for_inference = preprocessed_eval_dir \
        if preprocessed_eval_dir.exists() and any(preprocessed_eval_dir.iterdir()) \
        else eval_images_dir_orig # Fallback to original if conversion failed or no PNGs

    print(f"Using images from: {eval_images_dir_for_inference}")


    # Initialize the DetInferencer
    inferencer = DetInferencer(
        model=config_file,
        weights=checkpoint_file,
        device=device
    )

    submission_results = {"images": []}
    
    # Collect all image file paths (PNG or TIF)
    image_paths = list(eval_images_dir_for_inference.glob("*.png"))
    if not image_paths: # If no PNGs, try TIFs from original
        image_paths = list(eval_images_dir_orig.glob("*.tif"))
    
    if not image_paths:
        print(f"No images found in {eval_images_dir_for_inference} or {eval_images_dir_orig}. Exiting.")
        return

    print(f"Running inference on {len(image_paths)} images...")
    for img_path in tqdm(image_paths):
        # DetInferencer can take a path, ndarray, or list of paths/ndarrays
        result = inferencer(str(img_path), return_vis=False, no_save_pred=True) 
        
        # Structure for submission
        image_entry = {
            "file_name": img_path.name.replace(".png", ".tif"), # Use the actual file name being processed
            "width": 0, # Placeholder
            "height": 0, # Placeholder
            "annotations": []
        }
        
        # Get image dimensions (optional, might not be needed for submission JSON)
        try:
            with Image.open(img_path) as img_pil:
                w, h = img_pil.size
                image_entry["width"] = w
                image_entry["height"] = h
        except Exception:
            pass # If it fails, width/height remain 0

        # Process predictions
        pred_instances = result['predictions'][0] 
        
        for i in range(len(pred_instances['labels'])):
            label = pred_instances['labels'][i]
            score = pred_instances['scores'][i]
            bbox = pred_instances['bboxes'][i] # [xmin, ymin, xmax, ymax]

            # Assuming class "vacant_lot" corresponds to label 0 (as per our COCO conversion)
            if label == 0 and score >= score_thr:
                # Convert [xmin, ymin, xmax, ymax] to [xmin, ymin, width, height]
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                
                image_entry["annotations"].append({
                    "class": "vacant_lot",
                    "bbox": [float(x_min), float(y_min), float(width), float(height)]
                })
        submission_results["images"].append(image_entry)

    # Save the submission JSON
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, 'w') as f:
        json.dump(submission_results, f, indent=4)
    
    print(f"Prediction results saved to {output_json_path}")
    print(f"Remember: The F-Beta (0.5) score heavily prioritizes PRECISION. The chosen score_thr ({score_thr}) is critical.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vacant lots using a trained MMDetection model.")
    parser.add_argument(
        "config", 
        help="Path to the MMDetection config file used for training."
    )
    parser.add_argument(
        "checkpoint", 
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--eval_images_dir", 
        type=str, 
        default="data/evaluation_bbox_images", 
        help="Directory containing evaluation images (TIFFs or PNGs)."
    )
    parser.add_argument(
        "--output_json", 
        type=str, 
        default="outputs/bbox.json", 
        help="Path to save the output submission JSON file."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda:0", 
        help="Device to use for inference (e.g., 'cpu', 'cuda:0')."
    )
    parser.add_argument(
        "--score_thr", 
        type=float, 
        default=0.35, # <<< CRITICAL for F-Beta 0.5: Prioritize Precision
        help="Score threshold for detections. Higher means more precision, less recall."
    )
    args = parser.parse_args()

    # --- PyTorch load patch ---
    import torch
    import pickle

    _original_torch_load_seg = torch.load
    def _patched_torch_load_seg(*args_patch, **kwargs_patch):
        # PyTorch 2.6+ makes weights_only=True the default.
        # For older versions that warn or default to False, this ensures False.
        if 'weights_only' not in kwargs_patch or kwargs_patch.get('weights_only') is not False:
            # print("Patching torch.load to use weights_only=False for this call.")
            kwargs_patch['weights_only'] = False
        try:
            return _original_torch_load_seg(*args_patch, **kwargs_patch)
        except pickle.UnpicklingError as e_patch:
            # Specifically catch errors related to unpickling non-weight objects
            print(f"Pickle error encountered ('{str(e_patch)}'). Forcing weights_only=False.")
            kwargs_patch['weights_only'] = False
            return _original_torch_load_seg(*args_patch, **kwargs_patch)
        # It's better to let other UnpicklingErrors raise if they are not related to weights_only
        except Exception as e_patch_other: # Catch any other exception during load
           print(f"Another error during torch.load: {e_patch_other}")
           raise e_patch_other

    torch.load = _patched_torch_load_seg
    # --- End of PyTorch load patch ---
    
    main(args)

    # --- Restore original torch.load ---
    torch.load = _original_torch_load_seg
    # --- End of restore ---