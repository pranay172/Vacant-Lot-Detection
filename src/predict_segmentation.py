# src/predict_segmentation.py
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import mmcv # For older MMDetection/MMSegmentation utilities if needed
import cv2 # For findContours
import numpy as np
from PIL import Image

# For MMSegmentation v1.x
from mmseg.apis import MMSegInferencer # New, simplified inference API
from mmseg.utils import register_all_modules

def convert_evaluation_images_seg(input_dir: Path, output_dir: Path):
    """Converts evaluation TIFF images to PNG for segmentation, if necessary."""
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    tiff_files = list(input_dir.glob("*.tif"))
    if not tiff_files:
        # print(f"No .tif files found in evaluation directory {input_dir} for segmentation")
        return False

    converted_any = False
    # print(f"Converting {len(tiff_files)} evaluation TIFF images to PNG for segmentation...")
    for tiff_file in tqdm(tiff_files, desc="Converting eval seg images"):
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

def mask_to_polygons(mask_array: np.ndarray, min_area_pixels: int = 10):
    """
    Converts a binary mask array (0s and 1s) to a list of polygons.
    Each polygon is a flat list of [x1, y1, x2, y2, ...].
    Filters out very small polygons.
    """
    polygons = []
    # Ensure mask is 8-bit unsigned integer
    if mask_array.dtype != np.uint8:
        mask_array = mask_array.astype(np.uint8)
    
    # Find contours. RETR_EXTERNAL retrieves only the external contours.
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments, leaving only their end points.
    contours, hierarchy = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area_pixels:
            continue # Skip small artifacts
        
        # Simplify polygon to reduce number of points
        epsilon = 0.002 * cv2.arcLength(contour, True) # Adjust epsilon for more/less simplification
        approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
        
        # Polygon should have at least 3 points
        if len(approx_polygon) < 3:
            continue

        # Flatten the polygon to [x1, y1, x2, y2, ...]
        poly_list = approx_polygon.flatten().tolist()
        polygons.append(poly_list)
        
    return polygons


def main(args):
    config_file = args.config
    checkpoint_file = args.checkpoint
    eval_images_dir_orig_seg = Path(args.eval_images_dir)
    output_json_path_seg = Path(args.output_json)
    device = args.device

    # Register MMSeg modules
    register_all_modules(init_default_scope=True)

    # Convert evaluation TIFFs to PNG
    preprocessed_eval_seg_dir = eval_images_dir_orig_seg.parent / f"{eval_images_dir_orig_seg.name}_png"
    # print(f"Checking for TIFFs in {eval_images_dir_orig_seg} for conversion to {preprocessed_eval_seg_dir}...")
    convert_evaluation_images_seg(eval_images_dir_orig_seg, preprocessed_eval_seg_dir)
    
    eval_images_dir_for_inference_seg = preprocessed_eval_seg_dir \
        if preprocessed_eval_seg_dir.exists() and any(preprocessed_eval_seg_dir.iterdir()) \
        else eval_images_dir_orig_seg
        
    # print(f"Using segmentation images from: {eval_images_dir_for_inference_seg}")

    # Initialize the MMSegInferencer
    # print("Initializing MMSegInferencer...")
    inferencer = MMSegInferencer(
        model=config_file,
        weights=checkpoint_file,
        device=device,
        # Set classes and palette if your model/config doesn't have it or for visualization consistency
        # classes=('background', 'vacant_lot'), 
        # palette=[[0,0,0], [220,20,60]] 
    )
    # print("MMSegInferencer initialized.")

    submission_results_seg = {"images": []}
    
    image_paths_seg = list(eval_images_dir_for_inference_seg.glob("*.png"))
    if not image_paths_seg:
         image_paths_seg = list(eval_images_dir_orig_seg.glob("*.tif"))

    if not image_paths_seg:
        print(f"No images found in {eval_images_dir_for_inference_seg} or {eval_images_dir_orig_seg}. Exiting.")
        return

    print(f"Running segmentation inference on {len(image_paths_seg)} images...")
    for img_path in tqdm(image_paths_seg, desc="Segmenting"):
        # Inferencer returns a dict, 'pred_sem_seg' contains the predicted mask array
        result = inferencer(str(img_path), return_vis=False, out_dir='') # out_dir=None to prevent saving images
        if 'predictions' not in result:
            print(f"ERROR: 'predictions' key not found in inferencer result for {img_path.name}.")
            print(f"Available keys: {result.keys()}")
            continue # Skip this image or handle error

        # The value of result['predictions'] IS the mask tensor/array based on postprocess logic
        # when a single image is processed and return_datasamples=False.
        pred_mask_tensor_or_array = result['predictions']

        # Convert to NumPy array and ensure it's on CPU
        if hasattr(pred_mask_tensor_or_array, 'cpu'): # If it's a PyTorch Tensor
            pred_mask_array = pred_mask_tensor_or_array.cpu().numpy()
        elif isinstance(pred_mask_tensor_or_array, np.ndarray):
            pred_mask_array = pred_mask_tensor_or_array
        else:
            print(f"ERROR: result['predictions'] is of unexpected type: {type(pred_mask_tensor_or_array)} for {img_path.name}")
            continue

        # Squeeze if there's an unnecessary batch or channel dimension (e.g., from .data[0])
        pred_mask_array = pred_mask_array.squeeze()
        if pred_mask_array.ndim != 2:
            print(f"ERROR: pred_mask_array for {img_path.name} is not 2D after squeeze. Shape: {pred_mask_array.shape}")
            continue
        # Get original image dimensions for the submission JSON (Optional if not strictly needed by submission format)
        width, height = 0, 0
        try:
            with Image.open(img_path) as img_pil_orig:
                width, height = img_pil_orig.size
        except Exception:
            print(f"Warning: Could not read dimensions for {img_path.name}")
            if pred_mask_array is not None:
                 height, width = pred_mask_array.shape[:2]


        # Convert predicted mask (where class 1 is vacant_lot) to polygons
        # Ensure we are only considering the 'vacant_lot' class pixels for polygonization
        binary_mask_for_lot = (pred_mask_array == 1).astype(np.uint8)
        processed_mask = binary_mask_for_lot
        polygons_for_image = mask_to_polygons(processed_mask)
        
        image_entry_seg = {
            "file_name": img_path.name.replace(".png", ".tif"), # Use the actual file name (e.g., cropped_eval_X.png or .tif)
            "width": width,
            "height": height,
            "annotations": []
        }

        for poly in polygons_for_image:
            image_entry_seg["annotations"].append({
                "class": "vacant_lot",
                "segmentation": poly
            })
        
        submission_results_seg["images"].append(image_entry_seg)

    # Save the submission JSON
    output_json_path_seg.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path_seg, 'w') as f:
        json.dump(submission_results_seg, f, indent=4)
    
    print(f"Segmentation prediction results saved to {output_json_path_seg}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict vacant lot segmentations using a trained MMSegmentation model.")
    parser.add_argument(
        "config",
        help="Path to the MMSegmentation config file used for training."
    )
    parser.add_argument(
        "checkpoint",
        help="Path to the trained model checkpoint (.pth file)."
    )
    parser.add_argument(
        "--eval_images_dir",
        type=str,
        default="data/evaluation_segmentation_images",
        help="Directory containing evaluation images for segmentation (TIFFs or PNGs)."
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="outputs/segmentation.json",
        help="Path to save the output segmentation submission JSON file."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use for inference (e.g., 'cpu', 'cuda:0')."
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