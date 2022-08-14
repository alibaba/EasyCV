_base_ = './yolox_tiny_8xb16_300e_coco.py'

# model settings
model = dict(
    backbone=dict(
        model_type='nano',  # s m l x tiny nano
    ),
    head=dict(
        model_type='nano',
    )
)
