_base_ = 'configs/base.py'

checkpoint_config = dict(interval=10)
# optimizer
paramwise_options = {
    'backbone': dict(lr_mult=0.1),
}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
    paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[11])

total_epochs = 12

find_unused_parameters = False
