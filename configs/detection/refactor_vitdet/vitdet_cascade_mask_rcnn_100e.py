_base_ = [
    './_base_/models/vitdet_cascade_mask_rcnn.py',
    './_base_/datasets/coco_instance.py', 'configs/base.py'
]

log_config = dict(
    interval=200,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=10)
# optimizer
custom_keys = {
    'norm': dict(weight_decay=0.),
    'pos_embed': dict(weight_decay=0.),
}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    betas=(0.9, 0.999),
    weight_decay=0.1,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_options=dict(
        num_layers=12, layer_decay_rate=0.7, custom_keys=custom_keys))
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=250,
    warmup_ratio=0.001,
    step=[88, 96])
total_epochs = 100

find_unused_parameters = False
