_base_ = './imagenet_deit_base_patch16_224_jpg.py'

# model settings
model = dict(backbone=dict(hydra_attention=True, hydra_attention_layers=8))
