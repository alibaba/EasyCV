_base_ = '../common/dataset/imagenet_classification.py'

num_classes = 1000
# model settings
model = dict(
    type='Classification',
    backbone=dict(type='HRNet', arch='w18', multi_scale_output=True),
    neck=dict(type='HRFuseScales', in_channels=(18, 36, 72, 144)),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=2048,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth',
            label_smooth=0,
        ),
        num_classes=num_classes))

# optimizer
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0001)

# learning policy
lr_config = dict(policy='step', step=[30, 60, 90])
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 100
