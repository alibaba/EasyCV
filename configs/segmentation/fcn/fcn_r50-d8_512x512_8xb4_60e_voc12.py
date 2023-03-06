_base_ = ['./fcn_r50-d8_512x512_8xb4_60e_voc12aug.py']

CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# dataset settings
dataset_type = 'SegDataset'
data_root = 'data/VOCdevkit/VOC2012'

train_img_root = data_root + 'JPEGImages'
train_label_root = data_root + 'SegmentationClass'
train_list_file = data_root + 'ImageSets/Segmentation/train.txt'

val_img_root = data_root + 'JPEGImages'
val_label_root = data_root + 'SegmentationClass'
val_list_file = data_root + 'ImageSets/Segmentation/val.txt'

test_batch_size = 2

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='MMResize', img_scale=(2048, 512), ratio_range=(0.5, 2.0)),
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
            data_source=dict(
                type='SegSourceRaw',
                img_root=train_img_root,
                label_root=train_label_root,
                split=train_list_file,
                classes=CLASSES),
        ),
        pipeline=train_pipeline),
    val=dict(
        imgs_per_gpu=test_batch_size,
        ignore_index=255,
        type=dataset_type,
        data_source=dict(
            type='SegSourceRaw',
            img_root=val_img_root,
            label_root=val_label_root,
            split=val_list_file,
            classes=CLASSES,
        ),
        pipeline=test_pipeline),
)
