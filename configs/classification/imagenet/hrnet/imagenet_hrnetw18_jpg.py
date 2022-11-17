_base_ = 'configs/classification/imagenet/hrnet/hrnetw18_b32x8_100e_jpg.py'
# model settings
model = dict(backbone=dict(type='HRNet', arch='w18', multi_scale_output=True))
