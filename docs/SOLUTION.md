# Solution Write-up

**Competition:** Solafune — Vacant Lot Detection (2025)
**Final result:** 7th / 467 on the private leaderboard (score 0.7341), code verified

I trained two models, one for object detection and one for semantic segmentation. Both are transformer-based architectures fine-tuned from public pretrained checkpoints, and both were trained **only on the data provided in this competition** (no external datasets).

---

## 1. Problem recap

- **Imagery:** RGB TIFFs of Japan at varying resolutions.
- **Detection:** predict `vacant_lot` bounding boxes on 600 train / 400 eval images. Scored with **F-beta, β = 0.5**, so precision counts more than recall.
- **Segmentation:** predict `vacant_lot` polygons on 2,653 train / 2,500 eval **small crops** (some only a few tens of pixels across). Scored with **F1**.
- **Final score:** harmonic mean of the two, so a weak model on either task drags the total down.

---

## 2. Detection — DINO with Swin-L

### Preprocessing ([`src/preprocess_train_detection.py`](../src/preprocess_train_detection.py))

1. Convert every TIFF to RGB PNG.
2. Convert the competition JSON (`[x, y, w, h]` boxes) to COCO format with a single category `vacant_lot`.
3. Split images 90 / 10 into train / val (`random_state=42`), giving 540 / 60 images.

### Offline augmentation ([`src/augment_detection_offline.py`](../src/augment_detection_offline.py))

Each training image gets **5 augmented copies**, written to disk once with Albumentations:

| Transform | Parameters | p |
|---|---|---|
| HorizontalFlip | — | 0.5 |
| RandomSizedBBoxSafeCrop | 1012 × 1012, erosion 0.2 | 0.5 |
| Resize | 1333 × 800 | 1.0 |
| RandomBrightnessContrast | ±0.2 / ±0.2 | 0.7 |
| HueSaturationValue | hue 10, sat 15, val 10 | 0.5 |
| OneOf(GaussNoise σ∈[0.1, 0.3], Blur k≤3) | — | 0.5 |

The training set consists of these augmented copies only (≈2,700 images). The MMDetection training pipeline itself does no further augmentation. Validation uses the original images.

### Model ([`configs/dino_swinl_finetune_vacantlot.py`](../configs/dino_swinl_finetune_vacantlot.py))

- **DINO**, 5 feature levels, 900 queries, two-stage, box refinement, contrastive denoising (100 dn queries).
- **Backbone:** Swin-L (window 12, ImageNet-22k 384 pretrain), first 2 stages frozen, gradient checkpointing on to fit a 24 GB GPU.
- **Initialization:** full DINO-5scale Swin-L COCO checkpoint from MMDetection; only the classification head changes (1 class).
- **Losses:** Focal loss (α 0.25, γ 2) + L1 (×5) + GIoU (×2), Hungarian matching with the same weights.

### Training

- AdamW, lr 3e-5, weight decay 1e-4, backbone lr ×0.1, gradient clipping 0.1.
- Linear warmup over 500 iterations, then cosine annealing to 1e-6 over **5 epochs**.
- Batch size 1. Validation every epoch, and I kept the checkpoint with the best COCO bbox mAP.
- About 4 hours on one RTX 3090.

### Inference ([`src/predict_detection.py`](../src/predict_detection.py))

- Resize to 1333 × 800 (aspect ratio kept) and run `DetInferencer`.
- Keep detections with **score ≥ 0.35**. I chose this threshold empirically from public-LB feedback. Because the metric is F-beta with β = 0.5, a threshold that trades some recall for precision pays off.

---

## 3. Segmentation — Mask2Former with Swin-L

### Preprocessing ([`src/preprocess_train_segmentation.py`](../src/preprocess_train_segmentation.py))

1. Convert every TIFF to RGB PNG.
2. Drop degenerate images whose width or height is ≤ 1 px.
3. Rasterize the polygon annotations into binary masks (0 = background, 1 = vacant lot).
4. Split 90 / 10 into train / val (`random_state=42`).

### Offline augmentation ([`src/augment_segmentation_offline.py`](../src/augment_segmentation_offline.py))

Each training image is kept **plus 5 augmented copies** (6× data):

| Transform | Parameters | p |
|---|---|---|
| HorizontalFlip | — | 0.5 |
| VerticalFlip | — | 0.3 |
| RandomBrightnessContrast | ±0.2 / ±0.2 | 0.7 |
| HueSaturationValue | hue 3, sat 5, val 3 | 0.5 |
| ISONoise | color shift 0.01–0.05, intensity 0.1–0.3 | 0.5 |
| ShiftScaleRotate | shift 0.03, scale 0.1, rotate 5°, pad mask with 255 (ignore) | 0.3 |

I kept the colour jitter deliberately mild. Vacant lots are mostly told apart by subtle texture and colour differences (bare soil, gravel, grass), and strong colour shifts blur those differences. The validation set is copied without augmentation.

Online, MMSegmentation also applies `RandomResize` (ratio 0.75–1.5 around 256) and a `RandomCrop` of 256 × 256 (cat_max_ratio 0.75).

### Model ([`configs/mask2former_swinl_finetune_vacantlot_seg.py`](../configs/mask2former_swinl_finetune_vacantlot_seg.py))

- **Mask2Former** with an MSDeformAttn pixel decoder (6 layers) and a 9-layer transformer decoder, 150 queries.
- **Backbone:** Swin-L (ImageNet-22k 384 pretrain), only the first stage frozen.
- **Initialization:** full Mask2Former Swin-L ADE20K (640 × 640, 160k) checkpoint.
- **Classes:** background + vacant_lot.
- **Losses:** classification cross-entropy (×2, class weights 0.8 / 1.2 / 0.1 for background / vacant lot / no-object), mask BCE (×5) and Dice (×5).

### Training

- AdamW, lr 6e-5, weight decay 0.05, backbone lr ×0.1, no decay on norms, embeddings or position tables.
- Poly LR (power 0.9) over **15,000 iterations**, batch size 1.
- Validation every 1,000 iterations, and I kept the checkpoint with the best **val mDice**. Dice is the pixel-level F1, which matches the competition metric directly.
- About 75 minutes on one RTX 3090.

### Inference and polygon extraction ([`src/predict_segmentation.py`](../src/predict_segmentation.py))

1. Resize to 256 (aspect ratio kept) and run `MMSegInferencer`. The predicted mask comes back at the original resolution.
2. Take the class-1 pixels as a binary mask.
3. `cv2.findContours` (external contours only), discard contours with area < 10 px.
4. Simplify with `cv2.approxPolyDP` (ε = 0.2 % of the perimeter) and keep polygons with ≥ 3 points.

---

## 4. What I tried that didn't help

- **Test-time augmentation** for both tasks: horizontal flip plus multi-scale (detection at 1000×600 / 1333×800 / 1600×1000; segmentation via `SegTTAModel` at 512 / 768 / 1024). It never improved the public-LB score over single-pass inference, so the final submission doesn't use it.
- I didn't try any other architectures. DINO and Mask2Former with Swin-L were the only models I experimented with.

## 5. Challenges and reflection

The hardest part was the **environment**, not the modelling. Getting MMCV with CUDA ops, MMDetection, MMSegmentation and PyTorch 2.6 to agree inside a container took a lot of effort. Examples: the `weights_only` default change in `torch.load`, MMCV wheel and arch-list issues, and conda Terms-of-Service prompts during the Docker build.

I got busy with other work partway through, so the experiment budget stayed small (23 submissions in total). Ideas I would explore with more time:

- **Detection train/test resize mismatch.** Training images are resized offline to 1333 × 800 *without* keeping the aspect ratio, while inference keeps it. Matching the two, or using multi-scale training, is the first thing I'd try.
- Keep the un-augmented originals in the detection training set as well, and train for more epochs.
- Tune the detection score threshold on a local F-beta(0.5) validation score instead of relying on public-LB feedback.
- Tune the minimum polygon area and the simplification ε on the val split against the real F1 metric.
- Ensembles (e.g. WBF across detection checkpoints) and k-fold training instead of a single 90/10 split.

Overall I'm happy with 7th place for the time I could put in. Thanks to the Solafune team for a well-run competition.
