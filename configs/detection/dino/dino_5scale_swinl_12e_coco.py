_base_ = [
    './dino_5scale_swinl.py', '../_base_/dataset/autoaug_coco_detection.py',
    './dino_schedule_1x.py', 'configs/base.py'
]
