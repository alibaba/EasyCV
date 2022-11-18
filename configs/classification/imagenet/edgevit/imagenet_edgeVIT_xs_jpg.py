_base_ = './EdgeVit_b512x8_300e_jpg.py'
model = dict(
    type='Classification',
    backbone=dict(
        type='EdgeVit',
        depth=[1, 1, 3, 1],
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
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth', label_smooth=0.1),
    ))

# input data settings
data = dict(
    imgs_per_gpu=256,
    workers_per_gpu=10,
    use_repeated_augment_sampler=True,
)

# optimizer
update_interval = 4
optimizer_config = dict(update_interval=update_interval)
