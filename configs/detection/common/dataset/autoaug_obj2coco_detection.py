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
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(
        type='MMAutoAugment',
        policies=[
            [
                dict(
                    type='MMResize',
                    img_scale=[(720, 2000), (768, 2000), (816, 2000),
                               (864, 2000), (912, 2000), (960, 2000),
                               (1008, 2000), (1056, 2000), (1104, 2000),
                               (1152, 2000), (1200, 2000)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='MMResize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(600, 6300), (750, 6300), (900, 6300)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='MMRandomCrop',
                    crop_type='absolute_range',
                    crop_size=(576, 900),
                    allow_negative_crop=True),
                dict(
                    type='MMResize',
                    img_scale=[(720, 2000), (768, 2000), (816, 2000),
                               (864, 2000), (912, 2000), (960, 2000),
                               (1008, 2000), (1056, 2000), (1104, 2000),
                               (1152, 2000), (1200, 2000)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'ori_img_shape',
                   'img_shape', 'pad_shape', 'scale_factor', 'flip',
                   'flip_direction', 'img_norm_cfg'))
]
test_pipeline = [
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(2000, 1200),
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='MMPad', size_divisor=1),
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
            dict(type='LoadAnnotations', with_bbox=True)
        ],
        classes=CLASSES,
        test_mode=False,
        filter_empty_gt=False,
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
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
    val=val_dataset,
    drop_last=True)

# evaluation
eval_config = dict(initial=False, interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        dist_eval=False,
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]
