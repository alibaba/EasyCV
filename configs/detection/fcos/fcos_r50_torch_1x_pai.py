_base_ = ['./fcos.py', 'configs/base.py']

log_config = dict(
    interval=50, hooks=[
        dict(type='TextLoggerHook'),
    ])

checkpoint_config = dict(interval=10)
# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0001,
    paramwise_options=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    warmup_by_epoch=False,
    step=[8, 11])

total_epochs = 12

find_unused_parameters = False

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

img_scale = (1333, 800)

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='MMResize', img_scale=img_scale, keep_ratio=True),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size_divisor=32),
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
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='MMPad', size_divisor=32),
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

# dataset settings
data_type = 'DetSourcePAI'
train_path = 'data/coco/train2017.manifest'
val_path = 'data/coco/val2017.manifest'
test_batch_size = 1

train_dataset = dict(
    type='DetDataset',
    data_source=dict(type=data_type, path=train_path, classes=CLASSES),
    pipeline=train_pipeline)

val_dataset = dict(
    type='DetDataset',
    imgs_per_gpu=test_batch_size,
    data_source=dict(type=data_type, path=val_path, classes=CLASSES),
    pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset, val=val_dataset)

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]
