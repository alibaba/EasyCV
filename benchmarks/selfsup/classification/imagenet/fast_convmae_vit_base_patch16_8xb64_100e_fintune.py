_base_ = 'configs/base.py'

# open oss config when using oss
# sync local models and logs to oss
# oss_sync_config = dict(other_file_list=['**/events.out.tfevents*', '**/*log*'])
# oss_io_config = dict(
#     ak_id='your oss ak id',
#     ak_secret='your oss ak secret',
#     hosts='your oss hosts',
#     buckets=['your oss buckets'])

log_config = dict(
    interval=20,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# model settings
model = dict(
    type='Classification',
    train_preprocess=['mixUp'],
    mixup_cfg=dict(
        mixup_alpha=0.8,
        cutmix_alpha=1.0,
        cutmix_minmax=None,
        prob=1.0,
        switch_prob=0.5,
        mode='batch',
        label_smoothing=0.1,
        num_classes=1000),
    pretrained=None,
    backbone=dict(
        type='FastConvMAEViT',
        img_size=[224, 56, 28],
        patch_size=[4, 2, 2],
        in_channels=3,
        embed_dim=[256, 384, 768],
        depth=[2, 2, 11],
        num_heads=12,
        mlp_ratio=[4, 4, 4],
        drop_path_rate=0.1,
        init_pos_embed_by_sincos=False,
        with_fuse=False,
        global_pool=True),
    head=dict(
        type='ClsHead',
        with_fc=True,
        num_classes=1000,
        in_channels=768,
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        init_cfg=dict(type='TruncNormal', std=2e-5, layer='Linear', bias=0.)))

data_train_list = 'data/imagenet/meta/train_labeled.txt'
data_train_root = 'data/imagenet/train/'
data_test_list = 'data/imagenet/meta/val_labeled.txt'
data_test_root = 'data/imagenet/validation/'

dataset_type = 'ClsDataset'
train_pipeline = [
    dict(
        type='MAEFtAugment',
        input_size=224,
        color_jitter=None,
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
    imgs_per_gpu=128,  # 128*8
    workers_per_gpu=8,
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

eval_config = dict(initial=False, interval=1, gpu_collect=True)
eval_pipelines = [
    dict(
        mode='test',
        data=data['val'],
        dist_eval=True,
        evaluators=[dict(type='ClsEvaluator', topk=(1, 5))],
    )
]

# optimizer
update_interval = 1
optimizer_config = dict(update_interval=update_interval)
eff_batch_size = data['imgs_per_gpu'] * 8 * update_interval
lr_decay = 0.65
optimizer = dict(
    type='AdamW',
    lr=5e-4 * eff_batch_size / 256,
    weight_decay=0.05,
    paramwise_options={
        'norm': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0., lr_mult=lr_decay**12),
        'patch_embed': dict(lr_mult=lr_decay**12),
        'blocks1': dict(lr_mult=lr_decay**12),
        'blocks2': dict(lr_mult=lr_decay**12),
        'blocks3.0': dict(lr_mult=lr_decay**11),
        'blocks3.1': dict(lr_mult=lr_decay**10),
        'blocks3.2': dict(lr_mult=lr_decay**9),
        'blocks3.3': dict(lr_mult=lr_decay**8),
        'blocks3.4': dict(lr_mult=lr_decay**7),
        'blocks3.5': dict(lr_mult=lr_decay**6),
        'blocks3.6': dict(lr_mult=lr_decay**5),
        'blocks3.7': dict(lr_mult=lr_decay**4),
        'blocks3.8': dict(lr_mult=lr_decay**3),
        'blocks3.9': dict(lr_mult=lr_decay**2),
        'blocks3.10': dict(lr_mult=lr_decay**1),
    })
# learning policy
lr_config = dict(
    policy='StepFixCosineAnnealing',
    min_lr=1e-6,
    warmup='linear',
    warmup_iters=5,
    warmup_by_epoch=True,
    by_epoch=False)

checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 100
