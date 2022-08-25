_base_ = './dino_4sc_swinl.py'

# model settings
model = dict(
    backbone=dict(out_indices=(0, 1, 2, 3)),
    head=dict(
        in_channels=[192, 384, 768, 1536],
        num_feature_levels=5,
        transformer=dict(num_feature_levels=5)))
