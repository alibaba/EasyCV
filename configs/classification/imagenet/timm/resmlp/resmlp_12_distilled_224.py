_base_ = '../timm_config.py'

# model settings
model = dict(backbone=dict(model_name='resmlp_12_distilled_224'))
