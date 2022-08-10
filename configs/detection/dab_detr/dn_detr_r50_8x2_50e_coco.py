_base_ = './dab_detr_r50_8x2_50e_coco.py'

# model settings
model = dict(
    head=dict(
        dn_components=dict(
            scalar=5, label_noise_scale=0.2, box_noise_scale=0.4)))
