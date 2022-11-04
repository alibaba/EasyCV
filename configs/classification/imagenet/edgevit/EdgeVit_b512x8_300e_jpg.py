_base_ = '../../../base.py'

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='EdgeVit',
        depth=[1, 1, 3, 2],
        embed_dim=[36, 72, 144, 288],
        head_dim=36,
        mlp_ratio=[4] * 4,
        qkv_bias=True,
        num_classes=1000,
        drop_path_rate=0.1,
        sr_ratios=[4, 2, 2, 1]),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=288,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth', label_smooth=0),
    ))

data_train_list = 'data/imagenet_raw/meta/train_labeled.txt'
data_train_root = 'data/imagenet_raw/train/'
data_test_list = 'data/imagenet_raw/meta/val_labeled.txt'
data_test_root = 'data/imagenet_raw/validation/'

dataset_type = 'ClsDataset'
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        color_jitter=0.4,
        auto_augment='rand-m9-mstd0.5-inc1',
        interpolation='bicubic',
        re_prob=0.25,
        re_mode='pixel',
        re_count=1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=True),
]
test_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        is_train=False,
    ),
]

data = dict(
    imgs_per_gpu=512,
    workers_per_gpu=10,
    use_repeated_augment_sampler=True,
    train=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_train_list,
            root=data_train_root,
            type='ClsSourceImageList'),
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_source=dict(
            list_file=data_test_list,
            root=data_test_root,
            type='ClsSourceImageList'),
        pipeline=test_pipeline))

eval_config = dict(initial=True, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# additional hooks
custom_hooks = []

# optimizer
optimizer = dict(type='AdamW', lr=2e-3, weight_decay=0.05)

# learning policy
lr_config = dict(
    policy='CosineAnnealingWarmupByEpoch',
    min_lr=1e-5,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=1e-6,
    # warmup_lr=1e-6,
    warmup_by_epoch=True,
    by_epoch=True)

checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 300

ema = dict(decay=0.99996)
