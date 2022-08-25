_base_ = '../../base.py'

# model settings s m l x
model = dict(
    type='YOLOX',
    test_conf=0.01,
    nms_thre=0.65,
    backbone='RepVGGYOLOX',
    model_type='s',  # s m l x tiny nano
    head=dict(
        type='YOLOXHead',
        model_type='s',
        obj_loss_type='BCE',
        reg_loss_type='giou',
        num_classes=80,
        decode_in_inference=
        True  # set to False when test speed to ignore decode and nms
    ))

# s m l x
img_scale = (640, 640)
random_size = (14, 26)
scale_ratio = (0.1, 2)

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# dataset settings
data_root = 'data/coco/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MMMosaic', img_scale=img_scale, pad_val=114.0),
    dict(
        type='MMRandomAffine',
        scaling_ratio_range=scale_ratio,
        border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(
        type='MMMixUp',  # s m x l; tiny nano will detele
        img_scale=img_scale,
        ratio_range=(0.8, 1.6),
        pad_val=114.0),
    dict(
        type='MMPhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMResize', keep_ratio=True),
    dict(type='MMPad', pad_to_square=True, pad_val=(114.0, 114.0, 114.0)),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MMPad', pad_to_square=True, pad_val=(114.0, 114.0, 114.0)),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

train_dataset = dict(
    type='DetImagesMixDataset',
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        filter_empty_gt=True,
        iscrowd=False),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)

val_dataset = dict(
    type='DetImagesMixDataset',
    imgs_per_gpu=2,
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        filter_empty_gt=False,
        test_mode=True,
        iscrowd=True),
    pipeline=test_pipeline,
    dynamic_scale=None,
    label_padding=False)

data = dict(
    imgs_per_gpu=16, workers_per_gpu=4, train=train_dataset, val=val_dataset)

# additional hooks
interval = 10
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        no_aug_epochs=15,
        skip_type_keys=('MMMosaic', 'MMRandomAffine', 'MMMixUp'),
        priority=48),
    dict(
        type='SyncRandomSizeHook',
        ratio_range=random_size,
        img_scale=img_scale,
        interval=interval,
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval=interval,
        priority=48)
]

# evaluation
eval_config = dict(
    interval=10,
    gpu_collect=False,
    visualization_config=dict(
        vis_num=10,
        score_thr=0.5,
    )  # show by TensorboardLoggerHookV2 and WandbLoggerHookV2
)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='CocoDetectionEvaluator', classes=CLASSES)],
    )
]

checkpoint_config = dict(interval=interval)

# optimizer
optimizer = dict(
    type='SGD', lr=0.02, momentum=0.9, weight_decay=5e-4, nesterov=True)
optimizer_config = {}

# learning policy
lr_config = dict(
    policy='YOLOX',
    warmup='exp',
    by_epoch=False,
    warmup_by_epoch=True,
    warmup_ratio=1,
    warmup_iters=5,  # 5 epoch
    num_last_epochs=15,
    min_lr_ratio=0.05)

# exponetial model average
ema = dict(decay=0.9998)

# runtime settings
total_epochs = 300

# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHookV2'),
        # dict(type='WandbLoggerHookV2'),
    ])

export = dict(export_type = 'ori', preprocess_jit = False, batch_size=1, blade_config=dict(enable_fp16=True, fp16_fallback_op_ratio=0.01), use_trt_efficientnms=False)
