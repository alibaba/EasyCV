_base_ = './hrnetw18_b32x8_100e_jpg.py'
# model settings
model = dict(
    backbone=dict(type='HRNet', arch='w30', multi_scale_output=True),
    neck=dict(type='HRFuseScales', in_channels=(30, 60, 120, 240)))
