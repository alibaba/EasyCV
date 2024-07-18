_base_ = 'configs/classification/imagenet/inception/inceptionv3_b32x8_100e.py'

num_classes = 1000
# model settings
model = dict(
    type='Classification',
    backbone=dict(type='Inception4', num_classes=num_classes),
    head=[
        dict(
            type='ClsHead',
            with_fc=False,
            in_channels=1536,
            loss_config=dict(
                type='CrossEntropyLossWithLabelSmooth',
                label_smooth=0,
            ),
            num_classes=num_classes,
            input_feature_index=[1],
        ),
        dict(
            type='ClsHead',
            with_fc=False,
            in_channels=768,
            loss_config=dict(
                type='CrossEntropyLossWithLabelSmooth',
                label_smooth=0,
            ),
            num_classes=num_classes,
            input_feature_index=[0],
        )
    ])

img_norm_cfg = dict(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
