_base_ = './yolox_s_8xb16_300e_coco.py'

# model settings
model = dict(model_type='x')

data = dict(imgs_per_gpu=8, workers_per_gpu=4)

optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
