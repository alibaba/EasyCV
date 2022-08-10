_base_ = './resnet50_b32x8_100e_jpg.py'
# model settings
model = dict(
    type='Classification',
    backbone=dict(
        type='ResNet',
        depth=101,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))
