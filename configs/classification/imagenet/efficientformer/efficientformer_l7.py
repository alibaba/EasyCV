_base_ = './efficientformer_l1.py'
# model settings
model = dict(
    backbone=dict(
        layers=[6, 6, 18, 8], embed_dims=[96, 192, 384, 768], vit_num=8))
