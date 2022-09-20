_base_ = './deitiii_base_patch16_192.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='VisionTransformer',
        img_size=[192],
        num_classes=1000,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        drop_path_rate=0.45,
        use_layer_scale=True))
