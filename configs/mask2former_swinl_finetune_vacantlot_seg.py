# configs/mask2former_swinl_finetune_vacantlot_seg.py

# Inherit from the Swin-Large Mask2Former config for ADE20K (which itself inherits from Swin-Base and others)
# The paths below assume your 'mmsegmentation' directory is a sibling to your 'configs' directory,
# or that MMSegmentation's configs are in your PYTHONPATH. Adjust if needed.
_base_ = [
    '../mmsegmentation/configs/_base_/default_runtime.py',
    # We will define our own dataset configuration directly instead of inheriting ade20k_640x640.py
    # We will also define our own schedule.
    # The model structure comes from mask2former_swin-b and then is overridden by mask2former_swin-l.
]


# --- Model Definition (Adapted from mask2former_swin-l-in22k...ade20k-640x640.py) ---
# Backbone pretrained weights
pretrained_swin_l_22k = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # Swin-L ImageNet-22k pretrain

# Image size for our task. The "cropped_train_0.tif" example was 35x45.
# Mask2Former / Swin often use larger inputs, e.g., 512x512 or 640x640.
# Given the small nature of the example, we might need to pad or use a smaller crop_size.
# Let's choose a moderate crop_size, assuming images will be padded or are somewhat varied.
# This needs careful tuning based on actual image size distribution.
crop_size = (256, 256) # <<< MODIFIED: Start with a smaller crop size, adjust based on EDA of segmentation images

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53], # Standard ImageNet means
    std=[58.395, 57.12, 57.375],  # Standard ImageNet stds
    bgr_to_rgb=True,
    pad_val=0,          # Value for image padding
    seg_pad_val=255,    # Value for segmentation map padding (ignored class)
    size=crop_size      # Pad images to this size if smaller, or crop if larger during training
)

num_classes = 2 # <<< MODIFIED: 0 for background, 1 for vacant_lot

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384, # From Swin-L pretraining
        embed_dims=192,
        depths=[2, 2, 18, 2], # Swin-Large architecture
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3, # Standard for Swin-L
        patch_norm=True,
        out_indices=(0, 1, 2, 3), # Stages for FPN-like features
        with_cp=False, # Gradient checkpointing, set True if OOM, but can slow training
        frozen_stages=1, # Unfreeze all stages for fine-tuning except the first one
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_swin_l_22k)
    ),
    decode_head=dict(
        type='Mask2FormerHead',
        in_channels=[192, 384, 768, 1536], # From Swin-L out_indices
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_classes=num_classes,
        num_queries=150, # Standard Mask2Former, might be reducible for single class focus
        num_transformer_feat_level=3,
        align_corners=False,
        pixel_decoder=dict(
            type='mmdet.MSDeformAttnPixelDecoder', # Uses MMDetection's implementation
            num_outs=3,
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU'),
            encoder=dict(
                num_layers=6,
                layer_cfg=dict(
                    self_attn_cfg=dict(
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        im2col_step=64,
                        dropout=0.0,
                        batch_first=True,
                        norm_cfg=None,
                        init_cfg=None),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type='ReLU', inplace=True))),
                init_cfg=None),
            positional_encoding=dict(num_feats=128, normalize=True),
            init_cfg=None),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(
                self_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.0,
                    proj_drop=0.0,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.0,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None),
        # Loss functions
        loss_cls=dict(
            type='mmdet.CrossEntropyLoss', # Using MMDetection's CrossEntropyLoss
            use_sigmoid=False, # For multi-class classification of queries
            loss_weight=2.0,
            reduction='mean',
            class_weight=[0.8, 1.2, 0.1]
        ),
        loss_mask=dict(
            type='mmdet.CrossEntropyLoss', # BCE for masks
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type='mmdet.DiceLoss',
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        train_cfg=dict( # For Hungarian matcher and mask sampling
            num_points=12544, # (patch_size // 4)^2 * num_heads for ResNet backbone, might need adjustment for Swin
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            assigner=dict(
                type='mmdet.HungarianAssigner',
                match_costs=[
                    dict(type='mmdet.ClassificationCost', weight=2.0),
                    dict(type='mmdet.CrossEntropyLossCost', weight=5.0, use_sigmoid=True),
                    dict(type='mmdet.DiceCost', weight=5.0, pred_act=True, eps=1.0)
                ]),
            sampler=dict(type='mmdet.MaskPseudoSampler')
        )
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole') # 'whole' for semantic seg. 'slide' for sliding window
)

# --- Dataset Settings ---
dataset_type = 'BaseSegDataset' # MMSegmentation's simpler dataset type for file lists
data_root_competition_seg = 'data/preprocessed_data_seg_augmented/' # <<< MODIFIED: Root of preprocessed seg data

# Augmentations (adapted from ADE20K and considering potentially small images)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'), # Loads .png masks
    # dict(type='Resize', scale=crop_size, keep_ratio=False), # Ensure all images are at least crop_size if smaller
                                                            # Or use RandomResize from ADE20k
    dict(
        type='RandomResize',
        scale=crop_size, # Target scale for RandomCrop if images are larger
                         # For smaller images, this might upscale. Can set a scale range.
        ratio_range=(0.75, 1.5), # Resize image with a random ratio
        keep_ratio=True
    ),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75), # Crop if larger than crop_size
    dict(type='PackSegInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=crop_size, keep_ratio=True), # Resize to a fixed scale for testing
    dict(type='LoadAnnotations'), # Load GT for validation
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=1, # <<< MODIFIED: Swin-L is very memory heavy. Start with 1 for a single GPU.
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True), # For IterBasedTrainLoop
    dataset=dict(
        type=dataset_type,
        data_root=data_root_competition_seg,
        # data_prefix specifies subdirectories for images and masks relative to data_root
        data_prefix=dict(img_path='seg_images_png_aug/train/', seg_map_path='seg_masks_png_aug/train/'), # <<< MODIFIED
        # ann_file points to the list of training samples
        ann_file='augmented_train_seg_list.txt', # <<< MODIFIED
        pipeline=train_pipeline,
        img_suffix='.png', # Assuming images are in PNG format
        seg_map_suffix='.png', # Assuming masks are in PNG format
        # For BaseSegDataset, explicitly define classes and palette if not using a standard dataset
        metainfo=dict(classes=('background', 'vacant_lot'), palette=[[0,0,0], [220,20,60]]) # <<< MODIFIED
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_competition_seg,
        data_prefix=dict(img_path='seg_images_png_aug/val/', seg_map_path='seg_masks_png_aug/val/'), # <<< MODIFIED
        ann_file='augmented_val_seg_list.txt', # <<< MODIFIED
        pipeline=test_pipeline,
        img_suffix='.png', # Assuming images are in PNG format
        seg_map_suffix='.png', # Assuming masks are in PNG format
        metainfo=dict(classes=('background', 'vacant_lot'), palette=[[0,0,0], [220,20,60]]) # <<< MODIFIED
    )
)
test_dataloader = val_dataloader

# Validation evaluator
# For semantic segmentation, IoU and mIoU are standard. F1 (Dice) is also common.
# The competition uses F1-score. Dice is closely related to F1.
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice'], prefix='val') # <<< MODIFIED
test_evaluator = val_evaluator


# --- Optimizer & Schedule ---
# Adapted from mask2former_swin-b...ade20k.py (which uses AdamW and PolyLR)
# Max iterations. ADE20K uses 160k. For fine-tuning on a smaller dataset:
max_iters = 15000  # <<< MODIFIED: Start with fewer iterations for fine-tuning (e.g., 20k, 40k)

# Optimizer
# Mask2Former Swin-B config uses AdamW with lr=0.0001, wd=0.05.
# Swin-L might need smaller LR.
optimizer = dict(
    type='AdamW', lr=0.00006, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)) # <<< MODIFIED: Slightly smaller LR

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    # Paramwise cfg from the Swin-B Mask2Former config
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0), # Lower LR for backbone
            'backbone.patch_embed.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'backbone.norm': dict(lr_mult=0.1, decay_mult=0.0),
            'absolute_pos_embed': dict(lr_mult=0.1, decay_mult=0.0),
            'relative_position_bias_table': dict(lr_mult=0.1, decay_mult=0.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'query_feat': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0)
        },
        norm_decay_mult=0.0 # No weight decay for norm layers
    )
)

# Learning rate schedule (PolyLR is common for segmentation)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0, # Decays to 0
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False) # Iteration based
]

# Training loop
train_cfg = dict(
    type='IterBasedTrainLoop', max_iters=max_iters, val_interval=1000) # <<< MODIFIED: val_interval
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Default hooks from MMSeg's default_runtime.py, with modifications from Mask2Former
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=False, interval=1000, # <<< MODIFIED: checkpoint interval
        save_best='val/mDice', rule='greater', max_keep_ckpts=3), # Save best based on mDice
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False, interval=100) # Set draw=True to visualize
)

# Full Mask2Former Swin-L pre-trained model
load_from = 'https://download.openmmlab.com/mmsegmentation/v0.5/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth' # <<< VERIFIED (from your input)

# Work directory
work_dir = './work_dirs/mask2former_swinl_vacantlot' # <<< MODIFIED

# Reproducibility
randomness = dict(seed=42, diff_rank_seed=True)

# Auto scaling LR (keep disabled for manual LR tuning first)
auto_scale_lr = dict(enable=False, base_batch_size=16) # 8 GPUs * 2 samples/GPU
