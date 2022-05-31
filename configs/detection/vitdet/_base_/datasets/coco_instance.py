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

image_size = (1024, 1024)
train_pipeline = [
    # large scale jittering
    dict(
        type='MMResize',
        img_scale=image_size,
        ratio_range=(0.1, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='MMRandomCrop',
        crop_type='absolute_range',
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type='MMFilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size=image_size),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg'))
]
test_pipeline = [
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=image_size,
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='MMPad', size_divisor=1024),
            dict(type='ImageToTensor', keys=['img']),
            dict(
                type='Collect',
                keys=['img'],
                meta_keys=('filename', 'ori_filename', 'ori_shape',
                           'ori_img_shape', 'img_shape', 'pad_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'))
        ])
]

train_dataset = dict(
    type='DetDataset',
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True)
        ],
        classes=CLASSES,
        test_mode=False,
        filter_empty_gt=True,
        iscrowd=False),
    pipeline=train_pipeline)

val_dataset = dict(
    type='DetDataset',
    imgs_per_gpu=1,
    data_source=dict(
        type='DetSourceCoco',
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=True,
        filter_empty_gt=False,
        iscrowd=True),
    pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=1, workers_per_gpu=2, train=train_dataset, val=val_dataset)
