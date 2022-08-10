_base_ = './yolox_s_8xb16_300e_coco.py'

# model settings
model = dict(model_type='tiny')

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

img_scale = (416, 416)
random_size = (10, 20)
scale_ratio = (0.5, 1.5)
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

data_root = 'data/coco/'

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
        filter_empty_gt=False,
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

eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='CocoDetectionEvaluator', classes=CLASSES)],
    )
]
