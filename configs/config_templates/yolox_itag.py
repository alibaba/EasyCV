# oss io config
oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',
    buckets=['pai-vision-data-hz', 'your_bucket_2'])

# model settings
# model settings
# models s m l x

model = dict(
    type='YOLOX',
    num_classes=80,
    model_type='s',  # s m l x tiny nano
    test_conf=0.01,
    nms_thre=0.65)

# TODO: merge `img_size` and `img_scale`, support tuple `image_size` for `DetSourcePAI`
img_size = 640
img_scale = (640, 640)

random_size = (14, 26)
scale_ratio = (0.1, 2)

# tiny nano without mixup
# img_scale = (416, 416)
# random_size = (10, 20)
# scale_ratio = (0.5, 1.5)

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
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MMMosaic', img_scale='${img_scale}', pad_val=114.0),
    dict(
        type='MMRandomAffine',
        scaling_ratio_range='${scale_ratio}',
        border=['-${img_scale}[0] // 2', '-${img_scale}[1] // 2']),
    dict(
        type='MMMixUp',  # s m x l; tiny nano will detele
        img_scale='${img_scale}',
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
    dict(
        type='MMNormalize',
        mean='${img_norm_cfg.mean}',
        std='${img_norm_cfg.std}',
        to_rgb='${img_norm_cfg.to_rgb}'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='MMResize', img_scale='${img_scale}', keep_ratio=True),
    dict(type='MMPad', pad_to_square=True, pad_val=(114.0, 114.0, 114.0)),
    dict(
        type='MMNormalize',
        mean='${img_norm_cfg.mean}',
        std='${img_norm_cfg.std}',
        to_rgb='${img_norm_cfg.to_rgb}'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img'])
]

data = dict(
    imgs_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type='DetImagesMixDataset',
        data_source=dict(
            type='DetSourcePAI',
            path='data/coco/train2017.manifest',
            classes='${CLASSES}'),
        pipeline='${train_pipeline}',
        dynamic_scale='${img_scale}'),
    val=dict(
        type='DetImagesMixDataset',
        imgs_per_gpu=2,
        data_source=dict(
            type='DetSourcePAI',
            path='data/coco/val2017.manifest',
            classes='${CLASSES}'),
        pipeline='${test_pipeline}',
        dynamic_scale=None,
        label_padding=False))

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
        ratio_range='${random_size}',
        img_scale='${img_scale}',
        interval='${interval}',
        priority=48),
    dict(
        type='SyncNormHook',
        num_last_epochs=15,
        interval='${interval}',
        priority=48)
]

# evaluation
eval_config = dict(interval='${interval}', gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        data='${data.val}',
        evaluators=[dict(type='CocoDetectionEvaluator', classes=CLASSES)],
    )
]

checkpoint_config = dict(interval='${interval}')
# export model during training
checkpoint_sync_export = True
# optimizer
# basic_lr_per_img = 0.01 / 64.0
optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
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
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
dist_params = dict(backend='nccl')
cudnn_benchmark = True
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

export = dict(use_jit=False)
