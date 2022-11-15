_base_ = './EdgeVit_b512x8_300e_jpg.py'
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
    backbone=dict(
        type='EdgeVit',
        depth=[1, 2, 5, 3],
        embed_dim=[48, 96, 240, 384],
        head_dim=48,
        mlp_ratio=[4] * 4,
        qkv_bias=True,
        num_classes=1000,
        drop_path_rate=0.1,
        sr_ratios=[4, 2, 2, 1]),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=384,
        loss_config={
            'type': 'SoftTargetCrossEntropy',
        },
        with_fc=True))

data = dict(
    imgs_per_gpu=128,
    workers_per_gpu=10,
    use_repeated_augment_sampler=True,
)

# optimizer
update_interval = 8
optimizer_config = dict(update_interval=update_interval)
