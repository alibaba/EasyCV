_base_ = './dino_4sc_r50.py'

# model settings
model = dict(
    backbone=dict(out_indices=(1, 2, 3, 4)),
    head=dict(
        in_channels=[256, 512, 1024, 2048],
        num_feature_levels=5,
        transformer=dict(num_feature_levels=5)))
