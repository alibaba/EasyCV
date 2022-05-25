_base_ = ['configs/base.py']

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# model settings
num_classes = 21
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(1, 2, 3, 4),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
    ),
    decode_head=dict(
        type='FCNHead',
        in_channels=2048,
        in_index=3,
        channels=512,
        num_convs=2,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# dataset settings
dataset_type = 'SegDataset'
data_root = 'data/VOCdevkit/VOC2012'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MMResize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
    dict(type='SegRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMPhotoMetricDistortion'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_semantic_seg'],
        meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'img_norm_cfg')),
]
test_pipeline = [
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
            dict(
                type='Collect',
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
        ignore_index=255,
        data_source=dict(
            type='SourceConcat',
            data_source_list=[
                dict(
                    type='SegSourceRaw',
                    img_root=data_root + 'JPEGImages',
                    label_root=data_root + 'SegmentationClass',
                    split=data_root + 'ImageSets/Segmentation/train.txt',
                    classes=CLASSES),
                dict(
                    type='SegSourceRaw',
                    img_root=data_root + 'JPEGImages',
                    label_root=data_root + 'SegmentationClassAug',
                    split=data_root + 'ImageSets/Segmentation/aug.txt',
                    classes=CLASSES),
            ]),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=1,
        ignore_index=255,
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

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)

# runtime settings
total_epochs = 60
checkpoint_config = dict(interval=1)
eval_config = dict(interval=1, gpu_collect=False)
eval_pipelines = [
    dict(
        mode='test',
        evaluators=[
            dict(
                type='SegmentationEvaluator',
                classes=CLASSES,
                metric_names=['mIoU'])
        ],
    )
]
