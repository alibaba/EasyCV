_base_ = './dino_4scale_r50_12e_coco.py'

# learning policy
lr_config = dict(policy='step', step=[33])

total_epochs = 36