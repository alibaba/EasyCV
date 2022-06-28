_base_ = '../../base.py'

# model settings
model = dict(
    type='SimCLR',
    pretrained=False,
    backbone=dict(
        type='ResNet',
        depth=50,
        in_channels=3,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='SyncBN')),
    neck=dict(
        type='NonLinearNeckV1',  # simple fc-relu-fc neck in MoCo v2
        in_channels=2048,
        hid_channels=2048,
        out_channels=128,
        with_avg_pool=True),
    head=dict(type='ContrastiveHead', temperature=0.1))

# dataset settings
data_train_list = 'data/imagenet/meta/train.txt'
data_train_root = 'data/imagenet/train'
dataset_type = 'MultiViewDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomHorizontalFlip'),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='ColorJitter',
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2)
        ],
        p=0.8),
    dict(type='RandomGrayscale', p=0.2),
    dict(
        type='RandomAppliedTrans',
        transforms=[
            dict(
                type='GaussianBlur',
                sigma_min=0.1,
                sigma_max=2.0,
                kernel_size=23)
        ],
        p=0.5),
    dict(type='ToTensor'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img'])
]
data = dict(
    imgs_per_gpu=32,  # total 32*8
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_source=dict(
            type='SSLSourceImageList',
            list_file=data_train_list,
            root=data_train_root),
        num_views=[1, 1],
        pipelines=[train_pipeline, train_pipeline]))
# optimizer
optimizer = dict(
    type='LARS',
    lr=0.3,
    weight_decay=0.000001,
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
    warmup_iters=10,
    warmup_ratio=0.0001,
    warmup_by_epoch=True)
checkpoint_config = dict(interval=10)
# runtime settings
total_epochs = 200
