_base_ = '../timm_config.py'

# model settings
model = dict(backbone=dict(model_name='deit_base_distilled_patch16_224'))
