_base_ = './resnet50_b32x8_100e_jpg.py'
# model settings
model = dict(
    # pretrained='http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/pretrained_models/easycv/resnet/torchvision/resnet50.pth',
    backbone=dict(
        type='ResNet',
        depth=50,
        out_indices=[4],  # 0: conv-1, x: stage-x
        norm_cfg=dict(type='BN')))
