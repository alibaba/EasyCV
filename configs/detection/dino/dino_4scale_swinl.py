_base_ = './dino_4scale_r50.py'

# model settings
model = dict(
    pretrained=True,
    backbone=dict(
        type='DINOSwinTransformer',
        pretrain_img_size=384,
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        out_indices=(1, 2, 3),
        use_checkpoint=True))
