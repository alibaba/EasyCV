_base_ = './deitiii_base_patch16_LS.py'
# model settings
model = dict(
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='dynamic_vitiii_base_p16',
        num_classes=1000,
    ))
