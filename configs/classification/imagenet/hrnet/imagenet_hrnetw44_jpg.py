_base_ = './hrnetw18_b32x8_100e_jpg.py'
# model settings
model = dict(
    backbone=dict(type='HRNet', arch='w44', multi_scale_output=True),
    neck=dict(type='HRFuseScales', in_channels=(44, 88, 176, 352)))
