_base_ = './yolox_s_8xb16_300e_coco.py'

# oss io config
oss_io_config = dict(
    ak_id='your oss ak id',
    ak_secret='your oss ak secret',
    hosts='oss-cn-zhangjiakou.aliyuncs.com',  # your oss hosts
    buckets=['your_bucket'])  # your oss buckets

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

img_scale = (640, 640)
random_size = (14, 26)
scale_ratio = (0.1, 2)

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

# dataset settings
img_size = 640
train_dataset = dict(
    type='DetImagesMixDataset',
    data_source=dict(
        type='DetSourcePAI',
        path='data/coco/train2017.manifest',
        classes=CLASSES),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)

val_dataset = dict(
    type='DetImagesMixDataset',
    imgs_per_gpu=2,
    data_source=dict(
        type='DetSourcePAI',
        path='data/coco/val2017.manifest',
        classes=CLASSES),
    pipeline=test_pipeline,
    dynamic_scale=None,
    label_padding=False)

data = dict(
    imgs_per_gpu=16, workers_per_gpu=4, train=train_dataset, val=val_dataset)

eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='CocoDetectionEvaluator', classes=CLASSES)],
    )
]
