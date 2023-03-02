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
# data_root = 'data/coco/'
# train_ann_file = 'annotations/instances_train2017.json'
# train_img_prefix = 'train2017/'
# val_ann_file = 'annotations/instances_val2017.json'
# val_img_prefix = 'val2017/'
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

# train_dataset = dict(
#     type='DetDataset',
#     data_source=dict(
#         type='DetSourceCoco',
#         ann_file=train_ann_file,
#         img_prefix=train_img_prefix,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         classes=CLASSES,
#         test_mode=False,
#         filter_empty_gt=True,
#         iscrowd=False),
#     pipeline=train_pipeline)

# val_dataset = dict(
#     type='DetDataset',
#     imgs_per_gpu=1,
#     data_source=dict(
#         type='DetSourceCoco',
#         ann_file=val_ann_file,
#         img_prefix=val_img_prefix,
#         pipeline=[
#             dict(type='LoadImageFromFile'),
#             dict(type='LoadAnnotations', with_bbox=True)
#         ],
#         classes=CLASSES,
#         test_mode=True,
#         filter_empty_gt=False,
#         iscrowd=True),
#     pipeline=test_pipeline)

# dataset settings
data_type = 'DetSourceCoco'
test_batch_size = 2
train_path = ''
val_path = ''

train_ann_file = None
train_img_prefix = None
test_ann_file = None
test_img_prefix = None

if data_type == 'DetSourcePAI':

    train_dataset = dict(
        type='DetDataset',
        data_source=dict(type=data_type, path=train_path, classes=CLASSES),
        pipeline=train_pipeline)

    val_dataset = dict(
        type='DetDataset',
        imgs_per_gpu=test_batch_size,
        data_source=dict(type=data_type, path=val_path, classes=CLASSES),
        pipeline=test_pipeline)

elif data_type == 'DetSourceCoco':
    train_dataset = dict(
        type='DetDataset',
        data_source=dict(
            type=data_type,
            ann_file=train_ann_file,
            img_prefix=train_img_prefix,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=CLASSES,
            filter_empty_gt=True,
            iscrowd=False),
        pipeline=train_pipeline)

    val_dataset = dict(
        type='DetDataset',
        imgs_per_gpu=test_batch_size,
        data_source=dict(
            type=data_type,
            ann_file=test_ann_file,
            img_prefix=test_img_prefix,
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True)
            ],
            classes=CLASSES,
            test_mode=True,
            filter_empty_gt=False,
            iscrowd=True),
        pipeline=test_pipeline)

elif data_type == 'DetSourceVOC':
    train_dataset = dict(
        type='DetDataset',
        data_source=dict(
            type='DetSourceVOC',
            path=train_path,
            classes=CLASSES,
            cache_at_init=True),
        pipeline=train_pipeline)

    val_dataset = dict(
        type='DetDataset',
        imgs_per_gpu=test_batch_size,
        data_source=dict(
            type='DetSourceVOC',
            path=train_path,
            classes=CLASSES,
            cache_at_init=True),
        pipeline=test_pipeline)

data = dict(
    imgs_per_gpu=2, workers_per_gpu=2, train=train_dataset, val=val_dataset)

# evaluation
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        data=val_dataset,
        evaluators=[
            dict(type='CocoDetectionEvaluator', classes=CLASSES),
        ],
    )
]
