_base_ = '../../base.py'

# model settings
model = dict(
    type='BYOL',
    pretrained=False,
    base_momentum=0.996,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    neck=dict(
        type='NonLinearNeckV2',
        in_channels=2048,
        hid_channels=4096,
        out_channels=256,
        with_avg_pool=True),
    head=dict(
        type='LatentPredictHead',
        size_average=True,
        predictor=dict(
            type='NonLinearNeckV2',
            in_channels=256,
            hid_channels=4096,
            out_channels=256,
            with_avg_pool=False)))

# dataset settings
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline1 = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(type='GaussianBlur', sigma=(0.1, 2.0), kernel_size=23)
        ],
        p=1.),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='Solarization')],
        p=0.),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
train_pipeline2 = [
    dict(type='RandomResizedCrop', size=224, interpolation=3),  # bicubic
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.4,
                contrast=0.4,
                saturation=0.2,
                hue=0.1)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(type='GaussianBlur', sigma=(0.1, 2.0), kernel_size=23)
        ],
        p=0.1),
    dict(
        type='RandomAppliedTrans',
        transforms=[dict(type='Solarization')],
        p=0.2),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]

data = dict(
    imgs_per_gpu=32,  # total 32*8=256
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='SSLSourceImageList',
            list_file='oss://path/to/data/imagenet-raw/meta/train.txt',
            root='oss://path/to/data/imagenet-raw/train/',
        ),
        num_views=[1, 1],
        pipelines=[train_pipeline1, train_pipeline2]))
# additional hooks
custom_hooks = [dict(type='BYOLHook', end_momentum=1.)]
# optimizer
optimizer = dict(
    type='LARS',
    lr=0.2,
    weight_decay=0.0000015,
    momentum=0.9,
    paramwise_options={
        '(bn|gn)(\d+)?.(weight|bias)':
        dict(weight_decay=0., lars_exclude=True),
        'bias': dict(weight_decay=0., lars_exclude=True)
    })
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0.,
    warmup='linear',
    warmup_iters=2,
    warmup_ratio=0.0001,  # cannot be 0
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
