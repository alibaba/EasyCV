_base_ = './hrnetw18_b32x8_100e_jpg.py'

# model settings
model = dict(backbone=dict(type='HRNet', arch='w48', multi_scale_output=True))
