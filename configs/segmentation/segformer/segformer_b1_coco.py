_base_ = './segformer_b0_coco.py'

model = dict(
    pretrained=
    'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth',
    backbone=dict(embed_dims=64, ),
    decode_head=dict(in_channels=[64, 128, 320, 512], ),
)
