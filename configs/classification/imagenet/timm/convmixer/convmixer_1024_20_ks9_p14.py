_base_ = '../timm_config.py'

# model settings
model = dict(backbone=dict(model_name='convmixer_1024_20_ks9_p14'))
