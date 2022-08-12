_base_ = './dino_4scale_swinl.py'

# model settings
model = dict(
    backbone=dict(out_indices=(0, 1, 2, 3)),
    head=dict(num_feature_levels=5, transformer=dict(num_feature_levels=5)))
