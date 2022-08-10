_base_ = './hrnetw18_b32x8_100e_jpg.py'
# model settings
model = dict(
    backbone=dict(type='HRNet', arch='w64', multi_scale_output=True),
    neck=dict(type='HRFuseScales', in_channels=(64, 128, 256, 512)))
