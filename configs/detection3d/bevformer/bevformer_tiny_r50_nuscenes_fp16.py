_base_ = ['./bevformer_tiny_r50_nuscenes.py']

paramwise_cfg = {'img_backbone': dict(lr_mult=0.1)}
optimizer = dict(
    type='AdamW',
    lr=2.8e-4,
    paramwise_options=paramwise_cfg,
    weight_decay=0.01)

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2), loss_scale=512.)
