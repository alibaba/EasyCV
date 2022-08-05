_base_ = './deitiii_base_patch16_192.py'
# model settings
model = dict(
    backbone=dict(
        type='PytorchImageModelWrapper',
        model_name='deitiii_base_p16_192',
        num_classes=1000,
    ))
