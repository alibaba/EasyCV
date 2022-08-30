_base_ = './deitiii_small_patch16_224.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='DeiTIII',
        img_size=[224],
        num_classes=1000,
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.05))
