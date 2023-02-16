_base_ = ['configs/base.py']

# warning batch_size need >= 2
# model
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='STDCContextPathNet',
        backbone_cfg=dict(
            type='STDCNet',
            stdc_type='STDCNet1',
            in_channels=3,
            channels=(32, 64, 256, 512, 1024),
            bottleneck_type='cat',
            num_convs=4,
            norm_cfg=norm_cfg,
            act_cfg=dict(type='ReLU'),
            with_final_conv=False),
        last_in_channels=(1024, 512),
        out_channels=128,
        ffm_cfg=dict(in_channels=384, out_channels=256, scale_factor=4)),
    decode_head=dict(
        type='FCNHead',
        in_channels=256,
        channels=256,
        num_convs=1,
        num_classes=19,
        in_index=3,
        concat_input=False,
        dropout_ratio=0.1,
        norm_cfg=norm_cfg,
        align_corners=True,
        sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=[
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=2,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='FCNHead',
            in_channels=128,
            channels=64,
            num_convs=1,
            num_classes=19,
            in_index=1,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=False,
            sampler=dict(type='OHEMPixelSampler', thresh=0.7, min_kept=10000),
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='STDCHead',
            in_channels=256,
            channels=64,
            num_convs=1,
            num_classes=2,
            boundary_threshold=0.1,
            in_index=0,
            norm_cfg=norm_cfg,
            concat_input=False,
            align_corners=True,
            loss_decode=[
                dict(
                    type='CrossEntropyLoss',
                    loss_name='loss_ce',
                    use_sigmoid=True,
                    loss_weight=1.0),
                dict(type='DiceLoss', loss_name='loss_dice', loss_weight=1.0)
            ]),
    ],
    train_cfg=dict(),
    test_cfg=dict(mode='whole'),
    pretrained=
    'http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/EasyCV/modelzoo/segmentation/stdc/pretrain/stdc1_easycv.pth'
)

# dataset
CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
    'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
    'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)

train_pipeline = [
    dict(type='MMResize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='SegRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MMRandomFlip', flip_ratio=0.5),
    dict(type='MMPhotoMetricDistortion'),
    dict(type='MMNormalize', **img_norm_cfg),
    dict(type='MMPad', size=crop_size),
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
        img_scale=(2048, 1024),
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
dataset_type = 'SegDataset'
data_root = '../Cityscapes/'

train_img_root = data_root + 'leftImg8bit/train/'
train_label_root = data_root + 'gtFine/train/'

val_img_root = data_root + 'leftImg8bit/val/'
val_label_root = data_root + 'gtFine/val/'
data = dict(
    imgs_per_gpu=6,
    workers_per_gpu=4,
    persistent_workers=True,
    train=dict(
        type=dataset_type,
        ignore_index=255,
        data_source=dict(
            type='SegSourceCityscapes',
            img_root=train_img_root,
            label_root=train_label_root,
            classes=CLASSES),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=1,
        ignore_index=255,
        type=dataset_type,
        data_source=dict(
            type='SegSourceCityscapes',
            img_root=val_img_root,
            label_root=val_label_root,
            classes=CLASSES),
        pipeline=test_pipeline))

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=1e-4,
    warmup='linear',
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True,
    by_epoch=False)

# runtime settings
total_epochs = 1290
checkpoint_config = dict(interval=10)
eval_config = dict(interval=10, gpu_collect=False)
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

# export config
export = dict(export_neck=True)
checkpoint_sync_export = True
