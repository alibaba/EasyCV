_base_ = 'configs/base.py'

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

image_size2 = 224
image_size1 = int((256 / 224) * image_size2)
img_norm_cfg = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

predict = dict(
    type='ClassificationPredictor',
    pipelines=[
        dict(type='Resize', size=image_size1),
        dict(type='CenterCrop', size=image_size2),
        dict(type='ToTensor'),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='Collect', keys=['img'])
    ])
