_base_ = 'configs/detection/yolox/yolox_best_ori.py'
# _base_ = 'configs/detection/yolox/yolox_s_8xb16_300e_tal_asff_giou.py'

# model settings
model = dict(model_type='l')

data = dict(imgs_per_gpu=8, workers_per_gpu=4)

optimizer = dict(
    type='SGD', lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
