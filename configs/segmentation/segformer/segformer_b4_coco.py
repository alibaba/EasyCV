_base_ = './segformer_b0_coco.py'

model = dict(
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b4_20220624-d588d980.pth',
    backbone=dict(
        embed_dims=64,
        num_layers=[3, 8, 27, 3],
    ),
    decode_head=dict(
        in_channels=[64, 128, 320, 512],
        channels=768,
    ),
)
