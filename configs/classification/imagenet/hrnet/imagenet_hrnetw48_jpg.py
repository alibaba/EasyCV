_base_ = './hrnetw18_b32x8_100e_jpg.py'

# model settings
model = dict(
    backbone=dict(type='HRNet', arch='w48', multi_scale_output=True),
    neck=dict(type='HRFuseScales', in_channels=(48, 96, 192, 384)))
