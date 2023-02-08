_base_ = './vit_base_patch16_224_b64x64_300e_jpg.py'
# model settings
model = dict(backbone=dict(model_name='vit_large_patch32_224', ))
