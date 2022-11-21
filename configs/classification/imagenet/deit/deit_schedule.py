_base_ = 'configs/base.py'

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

# additional hooks
custom_hooks = []

# optimizer
optimizer = dict(
    type='AdamW',
    lr=0.001,
    weight_decay=0.05,
    eps=1e-8,
    paramwise_options={
        'cls_token': dict(weight_decay=0.),
        'pos_embed': dict(weight_decay=0.),
        'bias': dict(weight_decay=0.),
        'norm': dict(weight_decay=0.),
    })
optimizer_config = dict(grad_clip=None, update_interval=1)

lr_config = dict(
    policy='CosineAnnealingWarmupByEpoch',
    by_epoch=True,
    min_lr_ratio=0.00001 / 0.001,
    warmup='linear',
    warmup_by_epoch=True,
    warmup_iters=5,
    warmup_ratio=0.000001 / 0.001,
)
checkpoint_config = dict(interval=10)

# runtime settings
total_epochs = 300

ema = dict(decay=0.99996)
