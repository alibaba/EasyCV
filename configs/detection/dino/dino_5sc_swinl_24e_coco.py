_base_ = './dino_5sc_swinl_12e_coco.py'

# learning policy
lr_config = dict(policy='step', step=[20])

total_epochs = 24
