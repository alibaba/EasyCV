_base_ = './resnet50_b32x8_100e_jpg.py'
# model settings
model = dict(
    type='Classification',
    pretrained=None,
    backbone=dict(
        type='ResNet',
        depth=101,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')),
    head=dict(
        type='ClsHead',
        with_avg_pool=True,
        in_channels=2048,
        loss_config=dict(
            type='CrossEntropyLossWithLabelSmooth',
            label_smooth=0,
        ),
        num_classes=1000))
