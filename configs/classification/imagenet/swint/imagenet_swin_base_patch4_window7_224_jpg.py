_base_ = './swin_tiny_patch4_window7_224_b64x16_300e_jpg.py'
# model settings
model = dict(backbone=dict(model_name='swin_base_patch4_window7_224', ))
