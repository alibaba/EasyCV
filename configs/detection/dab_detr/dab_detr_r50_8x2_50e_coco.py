_base_ = [
    './dab_detr.py', '../_base_/dataset/autoaug_coco_detection.py',
    'configs/base.py'
]

checkpoint_config = dict(interval=10)
# optimizer
paramwise_options = {'backbone': dict(lr_mult=0.1, weight_decay_mult=1.0)}
optimizer = dict(
    type='AdamW',
    lr=1e-4,
    weight_decay=1e-4,
    paramwise_options=paramwise_options)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])

total_epochs = 50

find_unused_parameters = False
