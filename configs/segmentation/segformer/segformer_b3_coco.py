_base_ = './segformer_b0_coco.py'

model = dict(
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b3_20220624-13b1141c.pth',
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 4, 18, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=768,
    ),
)
