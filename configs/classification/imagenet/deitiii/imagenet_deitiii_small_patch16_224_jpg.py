_base_ = [
    './deitiii_base_patch16_192.py',
    './threeaug_imagenet_classification_224.py', './deitiii_schedule.py'
]
# model settings
model = dict(
    backbone=dict(
        type='VisionTransformer',
        img_size=[224],
        embed_dim=384,
        num_heads=6,
        drop_path_rate=0.05))

# optimizer
optimizer = dict(lr=0.004)

lr_config = dict(
    min_lr_ratio=0.00001 / 0.004,
    warmup_ratio=0.000001 / 0.004,
)
