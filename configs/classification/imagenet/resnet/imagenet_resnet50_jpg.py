_base_ = './resnet50_b32x8_100e_jpg.py'
# model settings
model = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))

checkpoint_sync_export = True
export = dict(export_type='raw', export_neck=True)
