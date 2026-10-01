# configs/dino_swinl_finetune_vacantlot.py

# This config is based on dino-5scale_swin-l_8xb2-12e_coco.py
# and its base dino-4scale_r50_8xb2-12e_coco.py

# --- Inherit default runtime settings ---
# We'll take the default_runtime.py content and allow overrides later
_base_ = ['../mmdetection/configs/_base_/default_runtime.py'] # Adjusted path

# --- Model Definition (Derived from dino-5scale_swin-l and dino-4scale_r50) ---
pretrained_swin_l = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa
num_levels = 5 # From dino-5scale_swin-l

model = dict(
    type='DINO',
    num_queries=900,
    with_box_refine=True,
    as_two_stage=True,
    num_feature_levels=num_levels, # From dino-5scale_swin-l
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=1
    ),
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        frozen_stages=2, # Swin-Large typically has frozen stages
        embed_dims=192,
        depths=[2, 2, 18, 2], # Swin-Large depths
        num_heads=[6, 12, 24, 48], # Swin-Large heads
        window_size=12, # Swin-Large window size
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2, # Standard for Swin-L
        patch_norm=True,
        out_indices=(0, 1, 2, 3), # For 5 levels, neck will take 4 features
        with_cp=True, # Use checkpointing to save memory
        convert_weights=True, # For Swin v1 to v2 compatibility if needed, usually fine
        init_cfg=dict(type='Pretrained', checkpoint=pretrained_swin_l)
    ),
    neck=dict(
        type='ChannelMapper',
        in_channels=[192, 384, 768, 1536], # Output channels of Swin-L for out_indices (0,1,2,3)
        kernel_size=1,
        out_channels=256, # Standard for DETR-like necks
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=num_levels # Should be 5
    ),
    encoder=dict(
        num_layers=6,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0), # num_levels = 5
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
        )
    ),
    decoder=dict(
        num_layers=6,
        return_intermediate=True,
        layer_cfg=dict(
            self_attn_cfg=dict(embed_dims=256, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=256, num_levels=num_levels, dropout=0.0), # num_levels = 5
            ffn_cfg=dict(embed_dims=256, feedforward_channels=2048, ffn_drop=0.0)
        ),
        post_norm_cfg=None
    ),
    positional_encoding=dict(
        num_feats=128,
        normalize=True,
        offset=0.0,
        temperature=20 # DINO uses 20, Deformable DETR uses 10000. Sticking to DINO's.
    ),
    bbox_head=dict(
        type='DINOHead',
        num_classes=1, # <<< MODIFIED: Only 'vacant_lot'
        sync_cls_avg_factor=True,
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)
    ),
    dn_cfg=dict(
        label_noise_scale=0.5,
        box_noise_scale=1.0,
        group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)
    ),
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            match_costs=[
                dict(type='FocalLossCost', weight=2.0),
                dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
                dict(type='IoUCost', iou_mode='giou', weight=2.0)
            ]
        )
    ),
    test_cfg=dict(max_per_img=300) # Max detections per image
)

# --- Dataset Settings ---
# Based on coco_detection.py and dino-4scale_r50_8xb2-12e_coco.py augmentations
dataset_type = 'CocoDataset'
data_root_competition = 'data/preprocessed_data/' # <<< MODIFIED: Your preprocessed data root

# Class name and palette
metainfo = {
    'classes': ('vacant_lot',),
    'palette': [(220, 20, 60)] # Example color, can be anything
}


backend_args = None
# --- Training Pipeline ---
# I have removed the redundant PhotoMetricDistortion. The Albu block is sufficient.
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs')
]

# --- Validation and Test Pipelines ---
# We need two separate pipelines: one for validation (with annotations)
# and one for testing (without annotations).

# Pipeline for VALIDATION (used by val_dataloader)
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # Standard DETR inference size
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'pad_shape'))
]

# Pipeline for PREDICTION/TESTING (used by test_dataloader for final submission)
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(1333, 800), keep_ratio=True), # Standard DETR inference size
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'img_norm_cfg', 'pad_shape'))
]

# --- Dataloaders ---

# Training dataloader
train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root='data/preprocessed_data_augmented/', # <<< MODIFIED: Your preprocessed data augmented root
        metainfo=metainfo,
        ann_file='augmented_coco.json',
        data_prefix=dict(img='augmented_images_png/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

# Validation dataloader (used during training loop)
val_dataloader = dict(
    batch_size=1, # TTA requires batch_size=1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root_competition,
        metainfo=metainfo,
        ann_file='coco_annotations/val_bbox_coco.json',
        data_prefix=dict(img='train_images_png/'),
        test_mode=True,
        pipeline=val_pipeline, # <<< USE THE DEDICATED VALIDATION PIPELINE
        backend_args=backend_args
    )
)

# Test dataloader (used by test loop or prediction scripts)
# This will be used by `DetInferencer` or `tools/test.py` for final submission generation.
test_dataloader = dict(
    batch_size=1, # TTA requires batch_size=1
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root='data/',
        metainfo=metainfo,
        # The annotation file is not used, but the pipeline needs to know not to load it.
        ann_file='preprocessed_data/coco_annotations/val_bbox_coco.json', # This is okay, as test_pipeline won't use it
        data_prefix=dict(img='evaluation_bbox_images_png/'), # <<< Point to eval images
        test_mode=True,
        pipeline=test_pipeline, # <<< USE THE DEDICATED TEST PIPELINE
        backend_args=backend_args
    )
)

# --- Evaluators ---
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root_competition + 'coco_annotations/val_bbox_coco.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args
)

test_evaluator = val_evaluator

# --- Optimizer & Schedule ---
max_epochs = 5 # Fine-tuning for 5 epochs

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=1) # Validate every epoch
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Learning rate
param_scheduler = [
    dict( # Linear warmup
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False, # Warmup per iteration
        begin=0,
        end=500 # Number of warmup iterations
    ),
    dict( # Main learning rate schedule
        type='CosineAnnealingLR',
        eta_min=1e-6,
        begin=0,
        end=max_epochs,
        T_max=max_epochs,
        by_epoch=True, # Update the LR every epoch
    )
]

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=3e-5, # Start with a smaller LR for Swin-L and fine-tuning
        weight_decay=0.0001
    ),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'level_embed': dict(lr_mult=1.0, decay_mult=0.0),
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }
    )
)

# --- Runtime Settings ---

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50), # Log every 50 iterations
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto', max_keep_ckpts=3), # Save every epoch
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook', draw=False, interval=100)
)

# Load DINO Swin-Large COCO pre-trained weights
load_from = 'https://download.openmmlab.com/mmdetection/v3.0/dino/dino-5scale_swin-l_8xb2-12e_coco/dino-5scale_swin-l_8xb2-12e_coco_20230228_072924-a654145f.pth'

# You might want to set a specific work_dir
work_dir = './work_dirs/dino_swinl_vacantlot' # <<< MODIFIED

# Set random seed for reproducibility
randomness = dict(seed=42, diff_rank_seed=True)

# Auto scaling LR
auto_scale_lr = dict(enable=False, base_batch_size=16)
