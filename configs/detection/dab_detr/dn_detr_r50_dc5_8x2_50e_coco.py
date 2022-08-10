_base_ = './dn_detr_r50_8x2_50e_coco.py'

# model settings
model = dict(backbone=dict(strides=(1, 2, 2, 1), dilations=(1, 1, 1, 2)))
