_base_ = './dino_4sc_swinl_12e_coco.py'

# learning policy
lr_config = dict(policy='step', step=[27, 33])

total_epochs = 36
