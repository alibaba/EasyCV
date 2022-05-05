_base_ = './resnext50-32x4d_b32x8_100e_jpg.py'
# model settings
model = dict(
    backbone=dict(
        type='ResNeXt',
        depth=101,
        groups=32,
        base_width=4,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))
