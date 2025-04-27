_base_ = '../common/dataset/imagenet_classification.py'

num_classes = 1000
# model settings
model = dict(
    type='Classification',
    backbone=dict(type='MobileNetV2'),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=1280,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth',
            label_smooth=0,
        ),
        num_classes=num_classes),
    pretrained=True)

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=5)

# runtime settings
total_epochs = 100
checkpoint_sync_export = True
export = dict(export_type='raw', export_neck=True)
