_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='TOOD',
    backbone=dict(
        type='MobileViTv2',
        out_indices=(2, 3, 4),
        norm_cfg=dict(type='BN', eps=0.001, momentum=0.03),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(type='TruncNormal', layer='Conv2d', std=0.03)),

    neck=dict(
        type='FeaturePyramidNetwork',
        in_channels=1024, # in_channels=(128, 160),
    ),

    bbox_head=dict(
        type='TOODHead',
        num_classes=5,
        in_channels=128,
        stacked_convs=6,
        feat_channels=128,
        anchor_type='anchor_based',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='CIoULoss', loss_weight=1.0)),
    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=9),
        assigner=dict(type='TaskAlignedAssigner', topk=13),
        alpha=1,
        beta=6,
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/cocolike_bur4_COCO'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 4)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.1, 0.3, 0.5, 0.7, 0.9),
        min_crop_size=0.3),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=320),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=320),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=0,
    train=dict(
        _delete_=True,
        type='RepeatDataset',  # use RepeatDataset to speed up training
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + '/annotations/instances_train2017.json',
            img_prefix=data_root + '/train2017/',
            pipeline=train_pipeline)),
    val=dict(ann_file=data_root + '/annotations/instances_val2017.json', img_prefix=data_root + '/val2017/', pipeline=test_pipeline),
    test=dict(ann_file=data_root + '/annotations/instances_val2017.json', img_prefix=data_root + '/val2017/', pipeline=test_pipeline))

runner = dict(type='EpochBasedRunner', max_epochs=110)# 120

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# # custom hooks
# custom_hooks = [dict(type='SetEpochInfoHook')]

# Avoid evaluation and saving weights too frequently
evaluation = dict(interval=1, metric='bbox', save_best='auto')
checkpoint_config = dict(interval=20)

custom_hooks = [
    dict(type='SetEpochInfoHook'),
    dict(type='NumClassCheckHook'),
    dict(type='CheckInvalidLossHook', interval=50, priority='VERY_LOW')
]