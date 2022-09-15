_base_ = 'configs/base.py'

checkpoint_config = dict(interval=10)
# optimizer
paramwise_options = {
    'norm': dict(weight_decay=0.),
    'bias': dict(weight_decay=0.),
    'pos_embed': dict(weight_decay=0.),
    'cls_token': dict(weight_decay=0.)
}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=None, loss_scale=512.)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.067,
    step=[88, 96])
total_epochs = 100

find_unused_parameters = False
