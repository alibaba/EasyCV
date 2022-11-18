_base_ = './imagenet_deitiii_base_patch16_192_jpg.py'
# model settings
model = dict(
    backbone=dict(embed_dim=1024, depth=24, num_heads=16, drop_path_rate=0.45))
