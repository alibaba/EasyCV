_base_ = '../timm_config.py'

# model settings
model = dict(backbone=dict(model_name='resmlp_36_distilled_224'))
