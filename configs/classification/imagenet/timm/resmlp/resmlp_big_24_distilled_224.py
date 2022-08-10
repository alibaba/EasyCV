_base_ = '../timm_config.py'

# model settings
model = dict(backbone=dict(model_name='resmlp_big_24_distilled_224'))
