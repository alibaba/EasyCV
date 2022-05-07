# dataset settings
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
    'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
dataset_type = 'SegDataset'
data_root = 'data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    # dict(type='LoadImageFromFile'),
    # dict(type='LoadAnnotations'),
    dict(type='MMResize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='SegRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMPhotoMetricDistortion'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', 
         keys=['img', 'gt_semantic_seg'], 
         meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg')),
]
test_pipeline = [
    # dict(type='LoadImageFromFile'),
    dict(
        type='MMMultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='MMResize', keep_ratio=True),
            dict(type='MMRandomFlip'),
            dict(type='MMNormalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', 
                 keys=['img'],
                 meta_keys=('filename', 'ori_filename', 'ori_shape',
                    'img_shape', 'pad_shape', 'scale_factor', 'flip',
                    'flip_direction', 'img_norm_cfg')),
        ])
]
data = dict(
    imgs_per_gpu=4,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_root=data_root + 'JPEGImages',
            label_root=data_root + 'SegmentationClass',
            split=data_root + 'ImageSets/Segmentation/train.txt',
            classes=CLASSES,
        ),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=1,
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_root=data_root + 'JPEGImages',
            label_root=data_root + 'SegmentationClass',
            split=data_root + 'ImageSets/Segmentation/val.txt',
            classes=CLASSES,
        ),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_root=data_root + 'JPEGImages',
            label_root=data_root + 'SegmentationClass',
            split=data_root + 'ImageSets/Segmentation/test.txt',
            classes=CLASSES,
        ),
        pipeline=test_pipeline))
