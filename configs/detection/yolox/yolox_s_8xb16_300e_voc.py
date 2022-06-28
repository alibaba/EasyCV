_base_ = './yolox_s_8xb16_300e_coco.py'

# s m l x
img_scale = (640, 640)
random_size = (14, 26)
scale_ratio = (0.1, 2)

# tiny nano without mixup
# img_scale = (416, 416)
# random_size = (10, 20)
# scale_ratio = (0.5, 1.5)

CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

model = dict(num_classes=20)

# dataset settings
data_root = 'data/voc/'
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
        type='DetSourceVOC',
        path=data_root + 'ImageSets/Main/train.txt',
        classes=CLASSES,
        cache_at_init=True),
    pipeline=train_pipeline,
    dynamic_scale=img_scale)

val_dataset = dict(
    type='DetImagesMixDataset',
    imgs_per_gpu=2,
    data_source=dict(
        type='DetSourceVOC',
        path=data_root + 'ImageSets/Main/val.txt',
        classes=CLASSES,
        cache_at_init=True),
    pipeline=test_pipeline,
    dynamic_scale=None,
    label_padding=False)

data = dict(
    imgs_per_gpu=16, workers_per_gpu=4, train=train_dataset, val=val_dataset)

# # evaluation
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        evaluators=[dict(type='CocoDetectionEvaluator', classes=CLASSES)],
    )
]
